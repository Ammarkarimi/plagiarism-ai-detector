import pytest
from fastapi.testclient import TestClient

from plagai.api.app import Services, create_app
from plagai.config import Settings
from plagai.plagiarism.corpus import CorpusManager
from plagai.plagiarism.detector import PlagiarismDetector
from plagai.plagiarism.embeddings import HashingEncoder

ADMIN = {"X-Admin-Token": "secret"}
TEXT = ("Mitochondria have their own DNA, which is inherited exclusively from the mother in most species. " * 3)


class FakeAI:
    def __init__(self):
        self.fail = False

    def detect(self, text, segments=True):
        if self.fail:
            raise RuntimeError("model crashed")
        return {"label": "human", "ai_probability": 0.1}


@pytest.fixture
def ctx(tmp_path):
    settings = Settings(database_url=f"sqlite:///{tmp_path}/t.db", corpus_dir=str(tmp_path / "corpus"),
                        admin_token="secret", words_per_credit=10, rate_limit_per_minute=1000)
    services = Services(settings)
    services._ai = FakeAI()
    services._corpora = CorpusManager(settings.corpus_dir, HashingEncoder())
    services._plag = PlagiarismDetector(HashingEncoder())
    client = TestClient(create_app(settings, services), raise_server_exceptions=False)
    r = client.post("/admin/accounts", json={"email": "a@b.com", "credits": 20}, headers=ADMIN)
    assert r.status_code == 201
    return client, {"Authorization": f"Bearer {r.json()['api_key']}"}, r.json()["account_id"], services


def test_auth_required(ctx):
    client, *_ = ctx
    assert client.post("/v1/ai-detect", json={"text": "hi"}).status_code == 401
    assert client.post("/v1/ai-detect", json={"text": "hi"}, headers={"X-API-Key": "plg_live_nope"}).status_code == 401
    assert client.post("/admin/accounts", json={"email": "x@y.com"}).status_code == 403


def test_metering_charges_credits_and_blocks_when_empty(ctx):
    client, auth, *_ = ctx
    r = client.post("/v1/ai-detect", json={"text": "word " * 25}, headers=auth)  # 25 words / 10 per credit = 3
    assert r.status_code == 200, r.text
    assert r.json()["usage"]["credits_charged"] == 3
    assert r.json()["usage"]["credits_remaining"] == 17
    assert r.headers["X-Credits-Remaining"] == "17"
    r = client.post("/v1/ai-detect", json={"text": "word " * 500}, headers=auth)
    assert r.status_code == 402
    assert client.get("/v1/account", headers=auth).json()["credits"] == 17


def test_failed_request_is_refunded(ctx):
    client, auth, _, services = ctx
    services._ai.fail = True
    r = client.post("/v1/ai-detect", json={"text": "word " * 25}, headers=auth)
    assert r.status_code == 500
    assert client.get("/v1/account", headers=auth).json()["credits"] == 20
    assert client.get("/v1/usage", headers=auth).json()[0]["status"] == "refunded"


def test_private_corpus_flow(ctx):
    client, auth, account_id, _ = ctx
    client.post(f"/admin/accounts/{account_id}/credits", json={"amount": 100, "reference": "pay_1"}, headers=ADMIN)
    # Same payment reference twice: the second grant must be a no-op.
    r = client.post(f"/admin/accounts/{account_id}/credits", json={"amount": 100, "reference": "pay_1"}, headers=ADMIN)
    assert r.json()["credits"] == 120
    r = client.post("/v1/documents", json={"text": TEXT, "title": "Old essay", "id": "essay-1"}, headers=auth)
    assert r.status_code == 201 and r.json()["id"] == "essay-1"
    r = client.post("/v1/plagiarism/check", json={"text": TEXT, "scope": "private"}, headers=auth)
    body = r.json()
    assert body["label"] == "high" and body["sources"][0]["doc_id"] == "essay-1"
    assert client.delete("/v1/documents/essay-1", headers=auth).status_code == 204
    r = client.post("/v1/plagiarism/check", json={"text": TEXT, "scope": "private"}, headers=auth)
    assert r.json()["label"] == "original"


def test_revoked_key_rejected(ctx):
    client, auth, account_id, _ = ctx
    k = client.post(f"/admin/accounts/{account_id}/keys", json={"name": "ci"}, headers=ADMIN).json()
    hdr = {"Authorization": f"Bearer {k['api_key']}"}
    assert client.get("/v1/account", headers=hdr).status_code == 200
    assert client.delete(f"/admin/keys/{k['key_id']}", headers=ADMIN).status_code == 204
    assert client.get("/v1/account", headers=hdr).status_code == 401
