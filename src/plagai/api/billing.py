"""API key issuing and credit accounting.

Credits are deducted with a conditional ``UPDATE ... WHERE credits >= cost``,
so concurrent requests can never overdraw an account, even across API
replicas sharing one Postgres database. Every change is mirrored in the
append-only ``credit_transactions`` ledger.
"""

from __future__ import annotations

import hashlib
import math
import secrets

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from plagai.api.db import Account, ApiKey, CreditTransaction, utcnow

KEY_PREFIX = "plg_live_"

# Relative price of each endpoint (credits per ``words_per_credit`` words).
ENDPOINT_MULTIPLIER = {
    "ai-detect": 1.0,
    "plagiarism": 1.0,
    "plagiarism-web": 3.0,     # third-party search API calls cost money
    "analyze": 2.0,            # AI + plagiarism together
    "index-document": 0.2,
}


class InsufficientCredits(Exception):
    def __init__(self, needed: int, available: int):
        super().__init__(f"needed {needed} credits, have {available}")
        self.needed = needed
        self.available = available


def hash_key(key: str) -> str:
    # Keys carry 256 bits of entropy, so a fast hash is enough; no salt or KDF needed.
    return hashlib.sha256(key.encode()).hexdigest()


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def cost_for(words: int, endpoint: str, words_per_credit: int, minimum: int) -> int:
    mult = ENDPOINT_MULTIPLIER.get(endpoint, 1.0)
    return max(minimum, math.ceil(words * mult / words_per_credit))


def create_account(db: Session, email: str, name: str = "", credits: int = 0) -> Account:
    acct = Account(email=email.lower().strip(), name=name, credits=0)
    db.add(acct)
    db.flush()
    if credits:
        grant(db, acct.id, credits, "grant", "signup")
    return acct


def issue_key(db: Session, account_id: int, name: str = "default") -> tuple[ApiKey, str]:
    """Return the DB row and the plaintext key. The plaintext is shown once and never stored."""
    plain = generate_key()
    row = ApiKey(account_id=account_id, name=name, prefix=plain[: len(KEY_PREFIX) + 4], key_hash=hash_key(plain))
    db.add(row)
    db.flush()
    return row, plain


def authenticate(db: Session, plain: str) -> ApiKey | None:
    row = db.scalar(select(ApiKey).where(ApiKey.key_hash == hash_key(plain)))
    if row is None or row.revoked or not row.account.is_active:
        return None
    row.last_used_at = utcnow()
    return row


def _apply(db: Session, account_id: int, delta: int, reason: str, reference: str,
           require_balance: bool) -> int:
    stmt = update(Account).where(Account.id == account_id)
    if require_balance:
        stmt = stmt.where(Account.credits >= -delta)
    res = db.execute(stmt.values(credits=Account.credits + delta).returning(Account.credits))
    new_balance = res.scalar()
    if new_balance is None:
        available = db.scalar(select(Account.credits).where(Account.id == account_id)) or 0
        raise InsufficientCredits(-delta, available)
    db.add(CreditTransaction(account_id=account_id, delta=delta, balance_after=new_balance,
                             reason=reason, reference=reference))
    return new_balance


def charge(db: Session, account_id: int, amount: int, reference: str) -> int:
    return _apply(db, account_id, -amount, "usage", reference, require_balance=True)


def refund(db: Session, account_id: int, amount: int, reference: str) -> int:
    return _apply(db, account_id, amount, "refund", reference, require_balance=False)


def grant(db: Session, account_id: int, amount: int, reason: str = "purchase", reference: str = "") -> int:
    if reason == "purchase" and reference:
        # Make payment webhooks idempotent: one ledger entry per payment id.
        dup = db.scalar(select(CreditTransaction).where(CreditTransaction.reference == reference,
                                                        CreditTransaction.reason == "purchase"))
        if dup:
            return dup.balance_after
    return _apply(db, account_id, amount, reason, reference, require_balance=False)
