"""Plagiarism check: candidate retrieval, then sentence alignment, then passage merging and a report."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from plagai.plagiarism.embeddings import Encoder
from plagai.plagiarism.index import MIN_SENT_WORDS, CorpusIndex
from plagai.text import count_words, normalize, split_sentences, words


@dataclass
class PlagiarismConfig:
    paraphrase_threshold: float = 0.80   # cosine similarity for a paraphrase match
    exact_overlap: float = 0.80          # word-trigram containment for an "exact" match
    near_exact_overlap: float = 0.50     # containment for "near_exact" (light edits)
    dense_top_k: int = 5
    lexical_top_docs: int = 20
    max_candidate_rows: int = 4000


def _ngrams(toks: list[str], n: int = 3) -> set[tuple[str, ...]]:
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def containment(query: str, source: str) -> float:
    """Share of the query's word trigrams that also appear in the source."""
    q = _ngrams(words(query))
    return len(q & _ngrams(words(source))) / len(q) if q else 0.0


class PlagiarismDetector:
    def __init__(self, encoder: Encoder, config: PlagiarismConfig | None = None):
        self.encoder = encoder
        self.cfg = config or PlagiarismConfig()

    def check(self, text: str, indexes: list[tuple[str, CorpusIndex]]) -> dict:
        """Compare ``text`` against each ``(scope_name, index)`` pair."""
        text = normalize(text)
        total_words = count_words(text)
        sents = split_sentences(text)
        qi = [i for i, s in enumerate(sents) if len(words(s.text)) >= MIN_SENT_WORDS]
        q_emb = self.encoder.encode([sents[i].text for i in qi])
        toks = words(text)

        best: dict[int, dict] = {}
        near_dups = []
        for scope, index in indexes:
            if not len(index):
                continue
            with index.lock:
                for doc_id, jac in index.near_duplicates(toks):
                    near_dups.append({"scope": scope, "doc_id": doc_id, "title": index.docs[doc_id]["title"],
                                      "url": index.docs[doc_id]["url"], "jaccard": round(jac, 4)})
                matches = self._align(sents, qi, q_emb, index, toks)
                for i, m in matches.items():
                    m["scope"] = scope
                    if i not in best or (m["rank"], m["similarity"]) > (best[i]["rank"], best[i]["similarity"]):
                        best[i] = m
        return self._report(text, sents, best, total_words, near_dups)

    # ------------------------------------------------------------------
    def _align(self, sents, qi, q_emb, index: CorpusIndex, toks) -> dict[int, dict]:
        cfg = self.cfg
        E = index.embeddings
        # Candidate rows: all sentences of lexically similar docs, plus the dense nearest neighbours.
        lex_rows: list[int] = []
        for doc_id, _ in index.lexical_candidates(toks, cfg.lexical_top_docs):
            lex_rows.extend(index.doc_rows.get(doc_id, []))
        lex_rows = lex_rows[:cfg.max_candidate_rows]
        lex_sims = q_emb @ E[lex_rows].T if lex_rows and len(q_emb) else None
        dense = index.dense_search(q_emb, cfg.dense_top_k)

        out: dict[int, dict] = {}
        for qpos, si in enumerate(qi):
            cands: dict[int, float] = dict(dense[qpos])
            if lex_sims is not None:
                top = np.argsort(-lex_sims[qpos])[:cfg.dense_top_k]
                cands.update({lex_rows[j]: float(lex_sims[qpos, j]) for j in top})
            qtext = sents[si].text
            for row, cos in sorted(cands.items(), key=lambda x: -x[1])[:2 * cfg.dense_top_k]:
                doc_id, sidx = index.rows[row]
                src, a, b = index.sentence_text(doc_id, sidx)
                # Copying often crosses sentence boundaries, so also compare with the next source sentence.
                nxt = sidx + 1 < len(index.docs[doc_id]["sentences"])
                src2, _, b2 = index.sentence_text(doc_id, sidx + 1) if nxt else ("", a, b)
                cont = max(containment(qtext, src), containment(qtext, src + " " + src2) if nxt else 0.0)
                if cont >= cfg.exact_overlap:
                    kind, rank = "exact", 3
                elif cont >= cfg.near_exact_overlap:
                    kind, rank = "near_exact", 2
                elif cos >= cfg.paraphrase_threshold:
                    kind, rank = "paraphrase", 1
                else:
                    continue
                sim = max(cont, cos)
                if si not in out or (rank, sim) > (out[si]["rank"], out[si]["similarity"]):
                    d = index.docs[doc_id]
                    out[si] = {"rank": rank, "type": kind, "similarity": round(sim, 4),
                               "cosine": round(cos, 4), "overlap": round(cont, 4),
                               "doc_id": doc_id, "title": d["title"], "url": d["url"],
                               "source_start": a, "source_end": b, "source_text": src}
        return out

    def _report(self, text, sents, best: dict[int, dict], total_words: int, near_dups) -> dict:
        # Merge runs of consecutive matched sentences that point to the same source document.
        passages: list[dict] = []
        for si in sorted(best):
            m = best[si]
            s = sents[si]
            prev = passages[-1] if passages else None
            if prev and prev["doc_id"] == m["doc_id"] and prev["_last"] == si - 1:
                prev["end"] = s.end
                prev["_last"] = si
                prev["source_end"] = max(prev["source_end"], m["source_end"])
                prev["source_start"] = min(prev["source_start"], m["source_start"])
                prev["_sims"].append(m["similarity"])
                prev["_types"].add(m["type"])
                prev["words"] += count_words(s.text)
                continue
            passages.append({"start": s.start, "end": s.end, "doc_id": m["doc_id"], "scope": m["scope"],
                             "title": m["title"], "url": m["url"], "source_start": m["source_start"],
                             "source_end": m["source_end"], "_last": si, "_sims": [m["similarity"]],
                             "_types": {m["type"]}, "words": count_words(s.text)})
        for p in passages:
            p["text"] = text[p["start"]:p["end"]]
            p["similarity"] = round(float(np.mean(p.pop("_sims"))), 4)
            types = p.pop("_types")
            p["type"] = next(t for t in ("exact", "near_exact", "paraphrase") if t in types)
            del p["_last"]

        by_source: dict[str, dict] = defaultdict(lambda: {"words": 0})
        for p in passages:
            src = by_source[p["doc_id"]]
            src.update({"doc_id": p["doc_id"], "title": p["title"], "url": p["url"], "scope": p["scope"]})
            src["words"] += p["words"]
        matched_words = sum(p["words"] for p in passages)
        score = matched_words / total_words if total_words else 0.0
        sources = sorted(by_source.values(), key=lambda x: -x["words"])
        for src in sources:
            src["percent"] = round(100 * src["words"] / total_words, 2) if total_words else 0.0
        return {
            "similarity_percent": round(100 * score, 2),
            "label": "high" if score >= 0.4 else "moderate" if score >= 0.15 else "low" if score > 0 else "original",
            "words": total_words,
            "matched_words": matched_words,
            "passages": passages,
            "sources": sources,
            "near_duplicates": near_dups,
            "counts": {t: sum(1 for m in best.values() if m["type"] == t) for t in ("exact", "near_exact", "paraphrase")},
        }
