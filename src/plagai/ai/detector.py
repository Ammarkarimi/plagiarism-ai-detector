"""Production entry point for AI-text detection."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from plagai.ai.ensemble import Ensemble, FeatureBuilder
from plagai.text import count_words, normalize, split_sentences

log = logging.getLogger(__name__)

MIN_RELIABLE_WORDS = 80
SEGMENT_SENTENCES = 3


class AIDetector:
    def __init__(self, model_dir: str, device: str = "auto", max_length: int = 512,
                 binoculars_models: tuple[str, str] | None = None):
        from plagai.ai.classifier import TransformerClassifier

        self.model_dir = Path(model_dir)
        self.classifier = TransformerClassifier(model_dir, device=device, max_length=max_length)
        self.ensemble = Ensemble.load(model_dir)
        bino = None
        if self.ensemble and self.ensemble.meta.get("binoculars"):
            from plagai.ai.zero_shot import Binoculars
            cfg = self.ensemble.meta["binoculars"]
            obs, perf = binoculars_models or (cfg["observer"], cfg["performer"])
            bino = Binoculars(obs, perf, device=device)
        self.features = FeatureBuilder(self.classifier, bino)
        if self.ensemble and self.ensemble.feature_names != self.features.names:
            raise ValueError(f"Ensemble expects {self.ensemble.feature_names}, got {self.features.names}")
        self.thresholds = self.ensemble.thresholds if self.ensemble else {"ai": 0.5, "human": 0.5}
        meta_path = self.model_dir / "train_meta.json"
        self.version = json.loads(meta_path.read_text()).get("base_model", "unknown") if meta_path.exists() else "unknown"

    def _doc_proba(self, text: str) -> tuple[float, dict]:
        if self.ensemble is None:
            p = float(self.classifier.predict_proba([text])[0])
            return p, {"classifier": p}
        X = self.features.build([text])
        p = float(self.ensemble.predict_proba(X)[0])
        signals = {"classifier": float(1 / (1 + np.exp(-X[0, 0])))}
        if "bino_score" in self.features.names:
            signals["binoculars_score"] = float(X[0, self.features.names.index("bino_score")])
        return p, signals

    def detect(self, text: str, segments: bool = True) -> dict:
        text = normalize(text)
        n_words = count_words(text)
        warnings = []
        if n_words < MIN_RELIABLE_WORDS:
            warnings.append(f"Text has {n_words} words; results under {MIN_RELIABLE_WORDS} words are unreliable.")

        prob, signals = self._doc_proba(text)
        segs = self._segments(text) if segments else []
        seg_words = sum(s["words"] for s in segs) or 1
        ai_fraction = sum(s["words"] for s in segs if s["ai_probability"] >= 0.5) / seg_words if segs else None

        t = self.thresholds
        if prob >= t["ai"]:
            label = "ai"
        elif prob < t["human"]:
            label = "human"
        elif ai_fraction is not None and 0.25 <= ai_fraction <= 0.75:
            label = "mixed"
        else:
            label = "uncertain"
        return {
            "label": label,
            "ai_probability": round(prob, 4),
            "ai_fraction": None if ai_fraction is None else round(ai_fraction, 4),
            "thresholds": t,
            "signals": {k: round(v, 4) for k, v in signals.items()},
            "words": n_words,
            "segments": segs,
            "warnings": warnings,
            "model_version": self.version,
        }

    def _segments(self, text: str) -> list[dict]:
        sents = split_sentences(text)
        if len(sents) <= 1:
            return []
        groups = [sents[i:i + SEGMENT_SENTENCES] for i in range(0, len(sents), SEGMENT_SENTENCES)]
        chunks = [text[g[0].start:g[-1].end] for g in groups]
        probs = self.classifier.predict_proba(chunks)
        return [{"start": g[0].start, "end": g[-1].end, "text": c, "words": count_words(c),
                 "ai_probability": round(float(p), 4)} for g, c, p in zip(groups, chunks, probs)]
