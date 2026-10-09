"""Reference corpus index: winnowing fingerprints, MinHash-LSH and sentence embeddings.

This in-process index suits up to a few million sentences per namespace. At
larger scale, swap the dense part for FAISS, pgvector or Qdrant, and the
fingerprint postings for Redis or Postgres. The ``CorpusIndex`` interface
stays the same.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from plagai.plagiarism.embeddings import Encoder
from plagai.plagiarism.fingerprint import LSHIndex, MinHasher, fingerprints, kgram_hashes
from plagai.text import normalize, split_sentences, words

MIN_SENT_WORDS = 5


class CorpusIndex:
    def __init__(self, encoder: Encoder, k: int = 5, window: int = 4, num_perm: int = 128):
        self.encoder = encoder
        self.k = k
        self.window = window
        self.minhasher = MinHasher(num_perm)
        self.lsh = LSHIndex(num_perm)
        self.docs: dict[str, dict] = {}
        self.postings: dict[int, set[str]] = defaultdict(set)
        self.doc_fps: dict[str, set[int]] = {}
        self._emb_parts: list[np.ndarray] = []
        self._emb: np.ndarray = np.zeros((0, encoder.dim), dtype=np.float32)
        self.rows: list[tuple[str, int]] = []  # embedding row -> (doc_id, sentence index)
        self.alive: list[bool] = []
        self.doc_rows: dict[str, list[int]] = {}
        self.lock = threading.RLock()

    # ----------------------------------------------------------------- write
    def add(self, text: str, title: str = "", url: str = "", doc_id: str | None = None,
            meta: dict | None = None, _embeddings: np.ndarray | None = None) -> str:
        text = normalize(text)
        doc_id = doc_id or uuid.uuid4().hex
        sents = split_sentences(text)
        toks = words(text)
        fps = {h for h, _ in fingerprints(toks, self.k, self.window)}
        sig = self.minhasher.signature(kgram_hashes(toks, 3))
        idx = [i for i, s in enumerate(sents) if len(words(s.text)) >= MIN_SENT_WORDS]
        emb = _embeddings if _embeddings is not None else self.encoder.encode([sents[i].text for i in idx])
        with self.lock:
            if doc_id in self.docs:
                self.remove(doc_id)
            self.docs[doc_id] = {"title": title, "url": url, "text": text, "meta": meta or {},
                                 "sentences": [(s.start, s.end) for s in sents]}
            self.doc_fps[doc_id] = fps
            for h in fps:
                self.postings[h].add(doc_id)
            self.lsh.add(doc_id, sig)
            start = len(self.rows)
            self.rows.extend((doc_id, i) for i in idx)
            self.alive.extend([True] * len(idx))
            self.doc_rows[doc_id] = list(range(start, start + len(idx)))
            if len(idx):
                self._emb_parts.append(emb.astype(np.float32))
        return doc_id

    def remove(self, doc_id: str) -> bool:
        with self.lock:
            if doc_id not in self.docs:
                return False
            for h in self.doc_fps.pop(doc_id, set()):
                self.postings[h].discard(doc_id)
                if not self.postings[h]:
                    del self.postings[h]
            self.lsh.remove(doc_id)
            for r in self.doc_rows.pop(doc_id, []):
                self.alive[r] = False
            del self.docs[doc_id]
            return True

    def __len__(self) -> int:
        return len(self.docs)

    # ------------------------------------------------------------------ read
    @property
    def embeddings(self) -> np.ndarray:
        with self.lock:
            if self._emb_parts:
                self._emb = np.concatenate([self._emb, *self._emb_parts])
                self._emb_parts = []
            return self._emb

    def sentence_text(self, doc_id: str, i: int) -> tuple[str, int, int]:
        d = self.docs[doc_id]
        a, b = d["sentences"][i]
        return d["text"][a:b], a, b

    def lexical_candidates(self, tokens: list[str], top: int = 20, min_shared: int = 2) -> list[tuple[str, int]]:
        counts: Counter = Counter()
        for h, _ in fingerprints(tokens, self.k, self.window):
            for d in self.postings.get(h, ()):
                counts[d] += 1
        return [(d, c) for d, c in counts.most_common(top) if c >= min_shared]

    def near_duplicates(self, tokens: list[str], min_jaccard: float = 0.5) -> list[tuple[str, float]]:
        sig = self.minhasher.signature(kgram_hashes(tokens, 3))
        out = [(d, MinHasher.jaccard(sig, self.lsh.signatures[d])) for d in self.lsh.query(sig)]
        return sorted([x for x in out if x[1] >= min_jaccard], key=lambda x: -x[1])

    def dense_search(self, q: np.ndarray, top_k: int = 5, block: int = 200_000) -> list[list[tuple[int, float]]]:
        """Top-k (row, cosine) per query embedding over live rows."""
        E = self.embeddings
        if len(E) == 0 or len(q) == 0:
            return [[] for _ in range(len(q))]
        alive = np.array(self.alive, dtype=bool)
        best_s = np.full((len(q), 0), -1.0, dtype=np.float32)
        best_i = np.zeros((len(q), 0), dtype=np.int64)
        for s in range(0, len(E), block):
            sims = q @ E[s:s + block].T
            sims[:, ~alive[s:s + block]] = -1.0
            k = min(top_k, sims.shape[1])
            part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            best_s = np.concatenate([best_s, np.take_along_axis(sims, part, 1)], 1)
            best_i = np.concatenate([best_i, part + s], 1)
            if best_s.shape[1] > top_k:
                keep = np.argpartition(-best_s, top_k - 1, axis=1)[:, :top_k]
                best_s = np.take_along_axis(best_s, keep, 1)
                best_i = np.take_along_axis(best_i, keep, 1)
        order = np.argsort(-best_s, axis=1)
        return [[(int(best_i[r, c]), float(best_s[r, c])) for c in order[r] if best_s[r, c] > -1]
                for r in range(len(q))]

    # ----------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with self.lock:
            live = [i for i, a in enumerate(self.alive) if a]
            E = self.embeddings[live] if live else np.zeros((0, self.encoder.dim), dtype=np.float32)
            rows = [self.rows[i] for i in live]
            np.save(p / "embeddings.npy", E)
            meta = {"encoder": self.encoder.name, "dim": self.encoder.dim, "k": self.k, "window": self.window,
                    "rows": rows, "docs": self.docs}
            tmp = p / "index.json.tmp"
            tmp.write_text(json.dumps(meta))
            tmp.replace(p / "index.json")

    @classmethod
    def load(cls, path: str | Path, encoder: Encoder) -> CorpusIndex:
        p = Path(path)
        meta = json.loads((p / "index.json").read_text())
        if meta["encoder"] != encoder.name:
            raise ValueError(f"Index built with {meta['encoder']}, but encoder is {encoder.name}. Re-index.")
        idx = cls(encoder, meta["k"], meta["window"])
        E = np.load(p / "embeddings.npy")
        by_doc: dict[str, list[int]] = defaultdict(list)
        for r, (doc_id, _) in enumerate(meta["rows"]):
            by_doc[doc_id].append(r)
        for doc_id, d in meta["docs"].items():
            rows = by_doc.get(doc_id, [])
            idx.add(d["text"], d["title"], d["url"], doc_id, d["meta"],
                    _embeddings=E[rows] if rows else np.zeros((0, encoder.dim), dtype=np.float32))
        return idx

    @classmethod
    def load_or_create(cls, path: str | Path, encoder: Encoder) -> CorpusIndex:
        return cls.load(path, encoder) if (Path(path) / "index.json").exists() else cls(encoder)
