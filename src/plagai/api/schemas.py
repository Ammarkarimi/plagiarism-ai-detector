from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, EmailStr, Field


class TextIn(BaseModel):
    text: str = Field(..., min_length=1, description="Plain text to analyse")


class AIDetectIn(TextIn):
    include_segments: bool = True


class PlagiarismIn(TextIn):
    scope: Literal["private", "public", "all"] = "all"
    web: bool = Field(False, description="Also search the web (higher credit cost)")


class AnalyzeIn(PlagiarismIn):
    include_segments: bool = True


class DocumentIn(BaseModel):
    text: str = Field(..., min_length=1)
    title: str = ""
    url: str = ""
    id: str | None = Field(None, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")


class Usage(BaseModel):
    request_id: str
    credits_charged: int
    credits_remaining: int


class AccountCreate(BaseModel):
    email: EmailStr
    name: str = ""
    credits: int = Field(0, ge=0)


class CreditGrant(BaseModel):
    amount: int = Field(..., gt=0)
    reason: Literal["purchase", "grant", "adjustment"] = "purchase"
    reference: str = ""


class KeyCreate(BaseModel):
    name: str = "default"
