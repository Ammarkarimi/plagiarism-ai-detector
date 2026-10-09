"""FastAPI service: credit-metered AI detection and plagiarism checking.

Run it with::

    uvicorn plagai.api.app:app --host 0.0.0.0 --port 8000

Authentication: ``Authorization: Bearer plg_live_...`` (or ``X-API-Key``).
Admin routes take ``X-Admin-Token`` and should sit behind your internal
network or dashboard backend.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy import func, select

from plagai.api import billing
from plagai.api.db import Account, ApiKey, CreditTransaction, UsageRecord, make_session_factory
from plagai.api.ratelimit import RateLimiter
from plagai.api.schemas import (AccountCreate, AIDetectIn, AnalyzeIn, CreditGrant, DocumentIn, KeyCreate,
                                PlagiarismIn)
from plagai.config import Settings, get_settings
from plagai.text import count_words

log = logging.getLogger("plagai.api")


@dataclass
class Services:
    """Lazily-built model objects. Tests inject fakes through ``create_app(services=...)``."""
    settings: Settings
    _ai: Any = None
    _plag: Any = None
    _corpora: Any = None
    _web: Any = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def ai(self):
        with self._lock:
            if self._ai is None:
                from pathlib import Path

                from plagai.ai.detector import AIDetector
                if not (Path(self.settings.ai_model_dir) / "config.json").exists():
                    raise HTTPException(503, "AI detector model is not installed on this server")
                self._ai = AIDetector(self.settings.ai_model_dir, self.settings.device, self.settings.ai_max_length)
            return self._ai

    def _encoder_and_corpora(self):
        if self._corpora is None:
            from plagai.plagiarism.corpus import CorpusManager
            from plagai.plagiarism.embeddings import build_encoder
            self._corpora = CorpusManager(self.settings.corpus_dir,
                                          build_encoder(self.settings.embedding_model, self.settings.device))
        return self._corpora

    def corpora(self):
        with self._lock:
            return self._encoder_and_corpora()

    def plagiarism(self):
        with self._lock:
            if self._plag is None:
                from plagai.plagiarism.detector import PlagiarismConfig, PlagiarismDetector
                cfg = PlagiarismConfig(paraphrase_threshold=self.settings.plag_paraphrase_threshold,
                                       exact_overlap=self.settings.plag_exact_overlap)
                self._plag = PlagiarismDetector(self._encoder_and_corpora().encoder, cfg)
            return self._plag

    def web(self):
        with self._lock:
            if self._web is None:
                import os

                from plagai.plagiarism.web import BraveSearch, WebSourceFinder
                if not os.environ.get("PLAGAI_BRAVE_API_KEY"):
                    raise HTTPException(501, "Web search is not configured on this server")
                self._web = WebSourceFinder(BraveSearch(), self._encoder_and_corpora().encoder)
            return self._web


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or get_settings()
    services = services or Services(settings)
    SessionLocal = make_session_factory(settings.database_url)
    limiter = RateLimiter(settings.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if settings.admin_token == "change-me":
            log.warning("PLAGAI_ADMIN_TOKEN is the default value; set a secret before going live")
        yield

    app = FastAPI(title="PlagAI Detection API", version="0.1.0", lifespan=lifespan,
                  description="AI-generated text detection and plagiarism checking, billed in credits.")
    app.state.services = services
    app.state.session_factory = SessionLocal

    # ------------------------------------------------------------ deps
    def require_key(authorization: str | None = Header(None), x_api_key: str | None = Header(None)) -> ApiKey:
        plain = x_api_key or (authorization[7:] if authorization and authorization.lower().startswith("bearer ") else None)
        if not plain:
            raise HTTPException(401, "Missing API key", headers={"WWW-Authenticate": "Bearer"})
        with SessionLocal.begin() as db:
            key = billing.authenticate(db, plain.strip())
            if key is None:
                raise HTTPException(401, "Invalid or revoked API key", headers={"WWW-Authenticate": "Bearer"})
            db.flush()
            db.expunge(key)
        if not limiter.allow(key.id):
            raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": "60"})
        return key

    def require_admin(x_admin_token: str | None = Header(None)) -> None:
        if not x_admin_token or not secrets.compare_digest(x_admin_token, settings.admin_token):
            raise HTTPException(403, "Forbidden")

    def metered(key: ApiKey, endpoint: str, text: str, response: Response, fn: Callable[[], dict]) -> dict:
        words = count_words(text)
        if words > settings.max_words_per_request:
            raise HTTPException(413, f"Text exceeds {settings.max_words_per_request} words")
        cost = billing.cost_for(words, endpoint, settings.words_per_credit, settings.min_credits_per_call)
        request_id = uuid.uuid4().hex
        try:
            with SessionLocal.begin() as db:
                billing.charge(db, key.account_id, cost, request_id)
        except billing.InsufficientCredits as e:
            raise HTTPException(402, {"error": "insufficient_credits", "needed": e.needed,
                                      "available": e.available}) from None
        t0 = time.perf_counter()
        status = "ok"
        try:
            result = fn()
        except Exception:
            status = "refunded"
            with SessionLocal.begin() as db:
                billing.refund(db, key.account_id, cost, request_id)
            raise
        finally:
            with SessionLocal.begin() as db:
                db.add(UsageRecord(request_id=request_id, account_id=key.account_id, api_key_id=key.id,
                                   endpoint=endpoint, words=words, credits=cost, status=status,
                                   latency_ms=int(1000 * (time.perf_counter() - t0))))
                remaining = db.scalar(select(Account.credits).where(Account.id == key.account_id))
        response.headers["X-Credits-Remaining"] = str(remaining)
        response.headers["X-Request-Id"] = request_id
        result["usage"] = {"request_id": request_id, "credits_charged": cost, "credits_remaining": remaining}
        return result

    def _indexes(key: ApiKey, scope: str):
        mgr = services.corpora()
        out = []
        if scope in ("public", "all"):
            out.append(("public", mgr.get("public")))
        if scope in ("private", "all"):
            out.append(("private", mgr.get(f"acct-{key.account_id}")))
        return out

    def _plagiarism(key: ApiKey, body: PlagiarismIn) -> dict:
        indexes = _indexes(key, body.scope)
        if body.web:
            indexes.append(("web", services.web().build_index(body.text)))
        return services.plagiarism().check(body.text, indexes)

    # ------------------------------------------------------------ errors
    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        log.exception("unhandled error: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    # ------------------------------------------------------------ public
    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/ai-detect")
    def ai_detect(body: AIDetectIn, response: Response, key: ApiKey = Depends(require_key)):
        return metered(key, "ai-detect", body.text, response,
                       lambda: services.ai().detect(body.text, segments=body.include_segments))

    @app.post("/v1/plagiarism/check")
    def plagiarism_check(body: PlagiarismIn, response: Response, key: ApiKey = Depends(require_key)):
        endpoint = "plagiarism-web" if body.web else "plagiarism"
        return metered(key, endpoint, body.text, response, lambda: _plagiarism(key, body))

    @app.post("/v1/analyze")
    def analyze(body: AnalyzeIn, response: Response, key: ApiKey = Depends(require_key)):
        def run():
            return {"ai": services.ai().detect(body.text, segments=body.include_segments),
                    "plagiarism": _plagiarism(key, body)}
        return metered(key, "analyze", body.text, response, run)

    @app.post("/v1/documents", status_code=201)
    def add_document(body: DocumentIn, response: Response, key: ApiKey = Depends(require_key)):
        ns = f"acct-{key.account_id}"

        def run():
            mgr = services.corpora()
            doc_id = mgr.get(ns).add(body.text, body.title, body.url, body.id)
            mgr.save(ns)
            return {"id": doc_id}
        return metered(key, "index-document", body.text, response, run)

    @app.delete("/v1/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str, key: ApiKey = Depends(require_key)):
        ns = f"acct-{key.account_id}"
        mgr = services.corpora()
        if not mgr.get(ns).remove(doc_id):
            raise HTTPException(404, "Document not found")
        mgr.save(ns)

    @app.get("/v1/account")
    def account(key: ApiKey = Depends(require_key)):
        with SessionLocal() as db:
            acct = db.get(Account, key.account_id)
            used = db.execute(select(func.count(), func.coalesce(func.sum(UsageRecord.credits), 0))
                              .where(UsageRecord.account_id == acct.id, UsageRecord.status == "ok")).one()
            return {"id": acct.id, "email": acct.email, "credits": acct.credits,
                    "requests": used[0], "credits_used": int(used[1]),
                    "pricing": {"words_per_credit": settings.words_per_credit,
                                "minimum_per_call": settings.min_credits_per_call,
                                "multipliers": billing.ENDPOINT_MULTIPLIER}}

    @app.get("/v1/usage")
    def usage(limit: int = 50, key: ApiKey = Depends(require_key)):
        with SessionLocal() as db:
            rows = db.scalars(select(UsageRecord).where(UsageRecord.account_id == key.account_id)
                              .order_by(UsageRecord.id.desc()).limit(min(limit, 500))).all()
            return [{"request_id": r.request_id, "endpoint": r.endpoint, "words": r.words, "credits": r.credits,
                     "status": r.status, "latency_ms": r.latency_ms, "created_at": r.created_at.isoformat()}
                    for r in rows]

    # ------------------------------------------------------------ admin
    admin = [Depends(require_admin)]

    @app.post("/admin/accounts", dependencies=admin, status_code=201)
    def admin_create_account(body: AccountCreate):
        with SessionLocal.begin() as db:
            if db.scalar(select(Account).where(Account.email == body.email.lower())):
                raise HTTPException(409, "Account already exists")
            acct = billing.create_account(db, body.email, body.name, body.credits or settings.signup_free_credits)
            key, plain = billing.issue_key(db, acct.id)
            return {"account_id": acct.id, "api_key": plain, "key_id": key.id, "credits": acct.credits}

    @app.post("/admin/accounts/{account_id}/keys", dependencies=admin, status_code=201)
    def admin_issue_key(account_id: int, body: KeyCreate):
        with SessionLocal.begin() as db:
            if not db.get(Account, account_id):
                raise HTTPException(404, "Account not found")
            key, plain = billing.issue_key(db, account_id, body.name)
            return {"key_id": key.id, "api_key": plain, "prefix": key.prefix}

    @app.delete("/admin/keys/{key_id}", dependencies=admin, status_code=204)
    def admin_revoke_key(key_id: int):
        with SessionLocal.begin() as db:
            key = db.get(ApiKey, key_id)
            if not key:
                raise HTTPException(404, "Key not found")
            key.revoked = True

    @app.post("/admin/accounts/{account_id}/credits", dependencies=admin)
    def admin_grant(account_id: int, body: CreditGrant):
        """Add credits, for example from a payment-provider webhook (pass its payment id as ``reference``)."""
        with SessionLocal.begin() as db:
            if not db.get(Account, account_id):
                raise HTTPException(404, "Account not found")
            balance = billing.grant(db, account_id, body.amount, body.reason, body.reference)
            return {"account_id": account_id, "credits": balance}

    @app.get("/admin/accounts/{account_id}/ledger", dependencies=admin)
    def admin_ledger(account_id: int, limit: int = 100):
        with SessionLocal() as db:
            rows = db.scalars(select(CreditTransaction).where(CreditTransaction.account_id == account_id)
                              .order_by(CreditTransaction.id.desc()).limit(limit)).all()
            return [{"delta": r.delta, "balance_after": r.balance_after, "reason": r.reason,
                     "reference": r.reference, "created_at": r.created_at.isoformat()} for r in rows]

    return app


_app: FastAPI | None = None


def __getattr__(name: str):
    # ``uvicorn plagai.api.app:app`` builds the app on first access, so importing
    # this module in tests doesn't open the production database.
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)
