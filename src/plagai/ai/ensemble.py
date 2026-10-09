"""Stack classifier, zero-shot and stylometric signals, and calibrate decision thresholds.

The stacker is fitted on the **validation** split, which the classifier never
trained on. Thresholds come from out-of-fold predictions, so they aren't
fitted on the same rows that set them:

* ``ai``: at or above this, label "ai". Chosen so that only ``target_fpr``
  (default 1%) of human validation texts are flagged.
* ``human``: below this, label "human". Chosen so that only ``target_fnr``
  of AI texts fall below it.
* In between, the label is "mixed" or "uncertain".

Usage::

    plagai fit-ensemble --model-dir artifacts/ai-detector --data data/ai [--binoculars]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from plagai.ai import stylometry
from plagai.ai.data import read_jsonl
from plagai.ai.metrics import breakdown, summarize, threshold_at_fnr, threshold_at_fpr
from plagai.ai.zero_shot import ZS_FEATURES

log = logging.getLogger(__name__)

ENSEMBLE_FILE = "ensemble.joblib"
ENSEMBLE_META = "ensemble.json"


class FeatureBuilder:
    def __init__(self, classifier=None, binoculars=None, use_stylometry: bool = True):
        self.classifier = classifier
        self.binoculars = binoculars
        self.use_stylometry = use_stylometry

    @property
    def names(self) -> list[str]:
        names = []
        if self.classifier is not None:
            names.append("clf_logit")
        if self.binoculars is not None:
            names += ZS_FEATURES
        if self.use_stylometry:
            names += stylometry.FEATURE_NAMES
        return names

    def build(self, texts: list[str]) -> np.ndarray:
        cols = []
        if self.classifier is not None:
            cols.append(self.classifier.predict_logit(texts)[:, None])
        if self.binoculars is not None:
            cols.append(self.binoculars.features(texts))
        if self.use_stylometry:
            cols.append(stylometry.feature_matrix(texts))
        return np.concatenate(cols, axis=1).astype(np.float32)


class Ensemble:
    def __init__(self, pipeline, feature_names: list[str], thresholds: dict, meta: dict | None = None):
        self.pipeline = pipeline
        self.feature_names = feature_names
        self.thresholds = thresholds
        self.meta = meta or {}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def save(self, model_dir: str | Path) -> None:
        d = Path(model_dir)
        joblib.dump(self.pipeline, d / ENSEMBLE_FILE)
        meta = {"feature_names": self.feature_names, "thresholds": self.thresholds, **self.meta}
        (d / ENSEMBLE_META).write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, model_dir: str | Path) -> Ensemble | None:
        d = Path(model_dir)
        if not (d / ENSEMBLE_FILE).exists():
            return None
        meta = json.loads((d / ENSEMBLE_META).read_text())
        return cls(joblib.load(d / ENSEMBLE_FILE), meta.pop("feature_names"), meta.pop("thresholds"), meta)


def fit(X: np.ndarray, y: np.ndarray, names: list[str], target_fpr: float, target_fnr: float,
        seed: int = 0) -> Ensemble:
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    oof = cross_val_predict(pipe, X, y, cv=5, method="predict_proba")[:, 1]
    t_ai = threshold_at_fpr(oof[y == 0], target_fpr)
    t_human = min(threshold_at_fnr(oof[y == 1], target_fnr), t_ai)
    pipe.fit(X, y)
    m = summarize(y, oof, t_ai)
    log.info("out-of-fold validation: %s", json.dumps(m))
    return Ensemble(pipe, names, {"ai": t_ai, "human": t_human},
                    {"target_fpr": target_fpr, "target_fnr": target_fnr, "val_oof": m})


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default="artifacts/ai-detector")
    p.add_argument("--data", default="data/ai")
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--target-fnr", type=float, default=0.05)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--binoculars", action="store_true", help="include Binoculars zero-shot features")
    p.add_argument("--observer", default=None)
    p.add_argument("--performer", default=None)
    p.add_argument("--max-length", type=int, default=512)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from plagai.ai.classifier import TransformerClassifier
    from plagai.config import get_settings

    s = get_settings()
    clf = TransformerClassifier(a.model_dir, max_length=a.max_length)
    bino = None
    observer = performer = None
    if a.binoculars:
        from plagai.ai.zero_shot import Binoculars
        observer = a.observer or s.binoculars_observer
        performer = a.performer or s.binoculars_performer
        bino = Binoculars(observer, performer)
    fb = FeatureBuilder(clf, bino)

    data = Path(a.data)
    val = read_jsonl(data / "val.jsonl", a.max_rows)
    X = fb.build([r["text"] for r in val])
    y = np.array([r["label"] for r in val])
    ens = fit(X, y, fb.names, a.target_fpr, a.target_fnr)
    ens.meta.update({"binoculars": {"observer": observer, "performer": performer} if bino else None,
                     "max_length": a.max_length})

    report = {}
    for split in ("test", "test_ood"):
        path = data / f"{split}.jsonl"
        if not path.exists() or not path.stat().st_size:
            continue
        rows = read_jsonl(path, a.max_rows)
        probs = ens.predict_proba(fb.build([r["text"] for r in rows]))
        report[split] = {"overall": summarize([r["label"] for r in rows], probs, ens.thresholds["ai"]),
                         "by_source": breakdown(rows, probs, "source", ens.thresholds["ai"]),
                         "by_attack": breakdown(rows, probs, "attack", ens.thresholds["ai"])}
        log.info("%s: %s", split, json.dumps(report[split]["overall"]))
    ens.meta["test"] = {k: v["overall"] for k, v in report.items()}
    ens.save(a.model_dir)
    (Path(a.model_dir) / "ensemble_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"thresholds": ens.thresholds, **ens.meta["test"]}, indent=2))


if __name__ == "__main__":
    main()
