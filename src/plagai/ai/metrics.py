"""Detection metrics. Always report TPR at a fixed low FPR, not just AUROC."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve


def tpr_at_fpr(y: np.ndarray, scores: np.ndarray, fpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y, scores)
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def threshold_at_fpr(human_scores: np.ndarray, fpr_target: float) -> float:
    """Smallest threshold such that at most ``fpr_target`` of human texts score at or above it."""
    s = np.sort(np.asarray(human_scores, dtype=float))
    if len(s) == 0:
        return 0.5
    k = int(np.floor((1 - fpr_target) * len(s)))
    k = min(max(k, 0), len(s) - 1)
    return float(np.nextafter(s[k], np.inf))


def threshold_at_fnr(ai_scores: np.ndarray, fnr_target: float) -> float:
    """Threshold below which at most ``fnr_target`` of AI texts fall."""
    s = np.sort(np.asarray(ai_scores, dtype=float))
    if len(s) == 0:
        return 0.5
    k = min(max(int(np.floor(fnr_target * len(s))), 0), len(s) - 1)
    return float(s[k])


def summarize(y, scores, threshold: float = 0.5) -> dict:
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    out = {"n": int(len(y)), "pos_rate": float(y.mean()) if len(y) else 0.0}
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, scores))
        out["tpr@1%fpr"] = tpr_at_fpr(y, scores, 0.01)
        out["tpr@5%fpr"] = tpr_at_fpr(y, scores, 0.05)
    pred = (scores >= threshold).astype(int)
    out["accuracy"] = float(accuracy_score(y, pred))
    out["f1"] = float(f1_score(y, pred, zero_division=0))
    human = y == 0
    out["fpr"] = float(pred[human].mean()) if human.any() else 0.0
    out["fnr"] = float(1 - pred[~human].mean()) if (~human).any() else 0.0
    return out


def breakdown(rows: list[dict], scores, key: str, threshold: float = 0.5) -> dict:
    """Metrics per value of ``key`` (e.g. source, generator, attack)."""
    idx = defaultdict(list)
    for i, r in enumerate(rows):
        idx[r.get(key, "")].append(i)
    scores = np.asarray(scores)
    return {k: summarize([rows[i]["label"] for i in ii], scores[ii], threshold) for k, ii in sorted(idx.items())}
