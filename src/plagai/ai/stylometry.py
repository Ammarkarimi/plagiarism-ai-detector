"""Cheap, model-free stylometric features.

They add a complementary signal to the ensemble (sentence-length
"burstiness", lexical diversity, punctuation habits and stock LLM
transition phrases). On their own they're weak. Never use them as the only
evidence.
"""

from __future__ import annotations

import math
import re

import numpy as np

from plagai.text import split_sentences, words

_STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being
it its this that these those i you he she we they me him her us them my your his our their
not no so than then there here have has had do does did will would can could should may might""".split())

_TRANSITIONS = ("additionally", "furthermore", "moreover", "in conclusion", "overall", "in summary",
                "it is important to note", "it's important to note", "ultimately", "in today's",
                "delve", "notably", "consequently", "firstly", "secondly", "lastly")

FEATURE_NAMES = [
    "log_words", "sent_len_mean", "sent_len_std", "sent_len_cv", "mattr", "word_len_mean",
    "comma_rate", "semicolon_rate", "colon_rate", "excl_rate", "question_rate", "dash_rate",
    "paren_rate", "quote_rate", "digit_rate", "upper_rate", "stopword_rate", "transition_rate",
    "newline_rate", "first_person_rate", "contraction_rate",
]


def _mattr(tokens: list[str], window: int = 50) -> float:
    """Moving-average type-token ratio, which is length-robust unlike the plain TTR."""
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    vals = [len(set(tokens[i:i + window])) / window for i in range(0, len(tokens) - window + 1, 10)]
    return float(np.mean(vals))


def features(text: str) -> dict[str, float]:
    toks = words(text)
    n = max(len(toks), 1)
    chars = max(len(text), 1)
    sents = split_sentences(text)
    lens = np.array([len(words(s.text)) for s in sents] or [0], dtype=float)
    mean = float(lens.mean())
    std = float(lens.std())
    low = text.lower()
    return {
        "log_words": math.log1p(len(toks)),
        "sent_len_mean": mean,
        "sent_len_std": std,
        "sent_len_cv": std / mean if mean else 0.0,
        "mattr": _mattr(toks),
        "word_len_mean": float(np.mean([len(t) for t in toks])) if toks else 0.0,
        "comma_rate": text.count(",") / n,
        "semicolon_rate": text.count(";") / n,
        "colon_rate": text.count(":") / n,
        "excl_rate": text.count("!") / n,
        "question_rate": text.count("?") / n,
        "dash_rate": len(re.findall(r"\s[-–—]\s|—", text)) / n,
        "paren_rate": text.count("(") / n,
        "quote_rate": len(re.findall(r"[\"“”]", text)) / n,
        "digit_rate": sum(c.isdigit() for c in text) / chars,
        "upper_rate": sum(c.isupper() for c in text) / chars,
        "stopword_rate": sum(t in _STOP for t in toks) / n,
        "transition_rate": sum(low.count(t) for t in _TRANSITIONS) / max(len(sents), 1),
        "newline_rate": text.count("\n") / n,
        "first_person_rate": sum(t in ("i", "me", "my", "mine", "i'm", "i've") for t in toks) / n,
        "contraction_rate": sum("'" in t for t in toks) / n,
    }


def feature_matrix(texts: list[str]) -> np.ndarray:
    rows = []
    for t in texts:
        f = features(t)
        rows.append([f[k] for k in FEATURE_NAMES])
    return np.array(rows, dtype=np.float32).reshape(len(texts), len(FEATURE_NAMES))
