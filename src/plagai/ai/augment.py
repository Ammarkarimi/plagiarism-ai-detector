"""Training-time augmentations that mirror the RAID adversarial attacks.

They are applied to **both** classes with the same probability. Otherwise the
model would learn "noisy means human" instead of what we care about.
"""

from __future__ import annotations

import random
import re

_HOMO = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у", "i": "і"}
_ZW = ["​", "‌", "‍"]
_KEYBOARD = "qwertyuiopasdfghjklzxcvbnm"
_SYNONYMS = {
    "important": ["crucial", "key", "significant"], "however": ["but", "yet", "though"],
    "use": ["utilize", "employ"], "help": ["assist", "aid"], "show": ["demonstrate", "reveal"],
    "big": ["large", "huge"], "small": ["little", "tiny"], "many": ["numerous", "lots of"],
    "also": ["additionally", "too"], "good": ["great", "fine"], "very": ["really", "extremely"],
}


def homoglyph(text: str, rng: random.Random, rate: float = 0.05) -> str:
    return "".join(_HOMO[ch] if ch in _HOMO and rng.random() < rate else ch for ch in text)


def zero_width(text: str, rng: random.Random, rate: float = 0.03) -> str:
    return "".join(ch + rng.choice(_ZW) if ch == " " and rng.random() < rate else ch for ch in text)


def typos(text: str, rng: random.Random, rate: float = 0.02) -> str:
    out = []
    for w in text.split(" "):
        if len(w) > 3 and rng.random() < rate:
            i = rng.randrange(1, len(w) - 1)
            op = rng.random()
            if op < 0.33:
                w = w[:i] + w[i + 1] + w[i] + w[i + 2:]          # swap
            elif op < 0.66:
                w = w[:i] + w[i + 1:]                              # delete
            else:
                w = w[:i] + rng.choice(_KEYBOARD) + w[i:]          # insert
        out.append(w)
    return " ".join(out)


def whitespace(text: str, rng: random.Random, rate: float = 0.03) -> str:
    return re.sub(" ", lambda m: "  " if rng.random() < rate else " ", text)


def casing(text: str, rng: random.Random) -> str:
    return text.lower() if rng.random() < 0.5 else text


def synonyms(text: str, rng: random.Random, rate: float = 0.5) -> str:
    def sub(m: re.Match) -> str:
        w = m.group()
        alts = _SYNONYMS.get(w.lower())
        if alts and rng.random() < rate:
            r = rng.choice(alts)
            return r.capitalize() if w[0].isupper() else r
        return w
    return re.sub(r"\b\w+\b", sub, text)


def drop_sentences(text: str, rng: random.Random, rate: float = 0.1) -> str:
    sents = re.split(r"(?<=[.!?])\s+", text)
    if len(sents) < 4:
        return text
    kept = [s for s in sents if rng.random() >= rate]
    return " ".join(kept or sents)


def random_crop(text: str, rng: random.Random, max_words: int) -> str:
    """Keep a random contiguous window of words so the model sees every part of long texts."""
    toks = text.split(" ")
    if len(toks) <= max_words:
        return text
    start = rng.randrange(0, len(toks) - max_words + 1)
    return " ".join(toks[start:start + max_words])


ATTACKS = [homoglyph, zero_width, typos, whitespace, casing, synonyms, drop_sentences]


def augment(text: str, rng: random.Random, p: float = 0.3) -> str:
    """With probability ``p``, apply one or two random attacks."""
    if rng.random() >= p:
        return text
    for fn in rng.sample(ATTACKS, k=rng.choice([1, 1, 2])):
        text = fn(text, rng)
    return text
