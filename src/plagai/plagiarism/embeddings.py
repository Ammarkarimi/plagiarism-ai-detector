"""Sentence encoders used for paraphrase-level matching."""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from plagai.text import words


class Encoder(Protocol):
    dim: int
    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised float32 embeddings of shape (n, dim)."""
        ...


class SentenceTransformerEncoder:
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        from plagai.config import resolve_device

        self.model = SentenceTransformer(model_name, device=resolve_device(device))
        self.name = model_name
        self.dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        # E5 models expect a "query: " prefix for symmetric similarity.
        self.prefix = "query: " if "e5" in model_name.lower() else ""

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        emb = self.model.encode([self.prefix + t for t in texts], batch_size=self.batch_size,
                                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return emb.astype(np.float32)


class HashingEncoder:
    """Dependency-free bag-of-words encoder. Use it for tests and offline development only.

    It catches verbatim and lightly edited copying, but not real paraphrase.
    """

    name = "hashing-bow"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = words(t)
            for tok in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
                out[i, h % self.dim] += 1.0 if h & (1 << 31) else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


def build_encoder(model_name: str, device: str = "auto") -> Encoder:
    if model_name in ("hashing", HashingEncoder.name):
        return HashingEncoder()
    return SentenceTransformerEncoder(model_name, device)
