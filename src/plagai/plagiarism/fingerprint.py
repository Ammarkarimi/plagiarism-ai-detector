"""Lexical fingerprints: word k-gram winnowing (MOSS) and MinHash with LSH banding."""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np

_MASK32 = (1 << 32) - 1
_PRIME = 4294967291  # largest prime below 2**32


def _h(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def kgram_hashes(tokens: list[str], k: int) -> list[int]:
    if len(tokens) < k:
        return [_h(" ".join(tokens))] if tokens else []
    return [_h(" ".join(tokens[i:i + k])) for i in range(len(tokens) - k + 1)]


def winnow(hashes: list[int], window: int) -> list[tuple[int, int]]:
    """Schleimer et al. (2003): keep the minimum hash in every window (rightmost on ties).

    Any shared run of at least ``window + k - 1`` tokens is guaranteed to share
    one fingerprint. Returns (hash, position) pairs.
    """
    if not hashes:
        return []
    if len(hashes) <= window:
        i = min(range(len(hashes)), key=lambda j: (hashes[j], -j))
        return [(hashes[i], i)]
    out: list[tuple[int, int]] = []
    last = -1
    for start in range(len(hashes) - window + 1):
        win = hashes[start:start + window]
        m = min(win)
        pos = start + max(j for j, v in enumerate(win) if v == m)
        if pos != last:
            out.append((m, pos))
            last = pos
    return out


def fingerprints(tokens: list[str], k: int = 5, window: int = 4) -> list[tuple[int, int]]:
    return winnow(kgram_hashes(tokens, k), window)


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, 1 << 31, num_perm, dtype=np.uint64)
        self.b = rng.integers(0, _PRIME, num_perm, dtype=np.uint64)

    def signature(self, shingle_hashes: list[int]) -> np.ndarray:
        if not shingle_hashes:
            return np.full(self.num_perm, _MASK32, dtype=np.uint64)
        h = np.array([x & _MASK32 for x in set(shingle_hashes)], dtype=np.uint64)
        # a < 2^31 and h < 2^32, so a*h + b < 2^63 + 2^32 and never overflows uint64.
        vals = (np.outer(h, self.a) + self.b) % np.uint64(_PRIME)
        return vals.min(axis=0)

    @staticmethod
    def jaccard(s1: np.ndarray, s2: np.ndarray) -> float:
        return float((s1 == s2).mean())


class LSHIndex:
    """Banded LSH over MinHash signatures, for near-duplicate document lookup."""

    def __init__(self, num_perm: int = 128, bands: int = 32):
        assert num_perm % bands == 0
        self.bands = bands
        self.rows = num_perm // bands
        self.buckets: dict[tuple[int, bytes], set[str]] = defaultdict(set)
        self.signatures: dict[str, np.ndarray] = {}

    def _keys(self, sig: np.ndarray):
        for b in range(self.bands):
            yield (b, sig[b * self.rows:(b + 1) * self.rows].tobytes())

    def add(self, key: str, sig: np.ndarray) -> None:
        self.signatures[key] = sig
        for k in self._keys(sig):
            self.buckets[k].add(key)

    def remove(self, key: str) -> None:
        sig = self.signatures.pop(key, None)
        if sig is None:
            return
        for k in self._keys(sig):
            self.buckets[k].discard(key)
            if not self.buckets[k]:
                del self.buckets[k]

    def query(self, sig: np.ndarray) -> set[str]:
        out: set[str] = set()
        for k in self._keys(sig):
            out |= self.buckets.get(k, set())
        return out
