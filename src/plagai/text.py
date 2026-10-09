"""Text normalisation, sentence splitting and chunking shared by both detectors."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Characters attackers insert to break detectors (RAID "zero-width" attack).
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­᠎"), None)

# Common Cyrillic/Greek homoglyphs of Latin letters (RAID "homoglyph" attack).
# NFKC does not fold these, so map them explicitly.
_HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "ɡ": "g", "һ": "h", "ӏ": "l",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S",
    "α": "a", "ο": "o", "ν": "v", "ρ": "p", "τ": "t", "Α": "A", "Β": "B",
    "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N",
    "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ",
})

_WS = re.compile(r"[ \t\r\f\v]+")
_MULTI_NL = re.compile(r"\n{3,}")
_WORD = re.compile(r"\w+(?:['’]\w+)?", re.UNICODE)
# Split after . ! ? (optionally followed by quotes/brackets) when followed by
# whitespace and an uppercase letter, digit or quote. Avoids common abbreviations.
_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
           "fig", "al", "no", "vol", "pp", "inc", "ltd", "co", "u.s"}
_SENT_END = re.compile(r"(?<=[.!?])[\"'”’)\]]*\s+(?=[\"'“‘(\[]?[A-Z0-9])")


_TOKEN = re.compile(r"\S+")
_LATIN = re.compile(r"[A-Za-z]")
_CONFUSABLE = re.compile(r"[Ͱ-ϿЀ-ӿɐ-ʯ]")
_PUNCT = str.maketrans({k: v for k, v in _HOMOGLYPHS.items() if chr(k) in "‘’“”–— "})


def _fold_mixed_script(m: re.Match) -> str:
    tok = m.group()
    # Only fold tokens that mix Latin with look-alike scripts, so genuine
    # Greek or Cyrillic words are left alone.
    if _LATIN.search(tok) and _CONFUSABLE.search(tok):
        return tok.translate(_HOMOGLYPHS)
    return tok


def normalize(text: str) -> str:
    """Undo common obfuscations and tidy whitespace without changing meaning."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ZERO_WIDTH).translate(_PUNCT)
    text = _TOKEN.sub(_fold_mixed_script, text)
    text = _WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def count_words(text: str) -> int:
    return len(_WORD.findall(text))


@dataclass(frozen=True)
class Span:
    text: str
    start: int  # character offset in the normalised document
    end: int


def split_sentences(text: str) -> list[Span]:
    """Regex sentence splitter that keeps character offsets.

    Paragraph breaks always end a sentence. Good enough for scoring; swap in
    spaCy or pysbd if you need better handling of edge cases.
    """
    spans: list[Span] = []
    for para in re.finditer(r"[^\n]+", text):
        p_start, chunk = para.start(), para.group()
        cursor = 0
        for m in _SENT_END.finditer(chunk):
            candidate = chunk[cursor:m.start()].rstrip()
            last = candidate.rsplit(" ", 1)[-1].rstrip(".!?\"'”’)]").lower()
            if last in _ABBREV or (len(last) == 1 and last.isalpha()):
                continue
            _push(spans, chunk, cursor, m.start(), p_start)
            cursor = m.end()
        _push(spans, chunk, cursor, len(chunk), p_start)
    return spans


def _push(spans: list[Span], chunk: str, a: int, b: int, offset: int) -> None:
    seg = chunk[a:b]
    stripped = seg.strip()
    if not stripped:
        return
    lead = len(seg) - len(seg.lstrip())
    start = offset + a + lead
    spans.append(Span(stripped, start, start + len(stripped)))


def sentence_windows(sentences: list[Span], size: int, stride: int) -> list[tuple[int, int]]:
    """Index ranges [i, j) of sliding windows over sentences."""
    if not sentences:
        return []
    if len(sentences) <= size:
        return [(0, len(sentences))]
    out = [(i, i + size) for i in range(0, len(sentences) - size + 1, stride)]
    if out[-1][1] < len(sentences):
        out.append((len(sentences) - size, len(sentences)))
    return out
