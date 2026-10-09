import random

import numpy as np

from plagai.ai import augment, stylometry
from plagai.ai.data import collect, split_of
from plagai.ai.ensemble import Ensemble, fit
from plagai.ai.metrics import summarize, threshold_at_fpr, tpr_at_fpr


def test_augment_is_deterministic_and_changes_text():
    text = "However, this is a very important result that many people use every day. " * 5
    a = augment.augment(text, random.Random(1), p=1.0)
    b = augment.augment(text, random.Random(1), p=1.0)
    assert a == b and a != text


def test_stylometry_shapes():
    X = stylometry.feature_matrix(["One sentence. Another, longer sentence here!", "x"])
    assert X.shape == (2, len(stylometry.FEATURE_NAMES))
    assert np.isfinite(X).all()


def test_threshold_at_fpr_bounds_false_positives():
    rng = np.random.default_rng(0)
    human = rng.normal(0, 1, 10_000)
    t = threshold_at_fpr(human, 0.01)
    assert (human >= t).mean() <= 0.01


def test_metrics_perfect_separation():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    m = summarize(y, s)
    assert m["auroc"] == 1.0 and tpr_at_fpr(y, s, 0.01) == 1.0 and m["fpr"] == 0.0


def test_collect_balances_and_caps_generators():
    def loader(rng):
        for i in range(400):
            yield {"text": f"word{i} " * 60, "label": i % 2, "source": "s", "generator": "g1" if i % 4 == 1 else "g2",
                   "domain": "", "attack": "none", "group": str(i)}
    recs = collect(loader, 50, random.Random(0), max_per_generator=0.5)
    labels = [r["label"] for r in recs]
    assert labels.count(0) == labels.count(1) == 50
    assert sum(r["generator"] == "g1" for r in recs) == 25


def test_split_is_deterministic_per_group():
    assert split_of("hc3:42", 0.1, 0.1) == split_of("hc3:42", 0.1, 0.1)


def test_ensemble_fit_save_load(tmp_path):
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 1, (200, 3)), rng.normal(2, 1, (200, 3))]).astype(np.float32)
    y = np.array([0] * 200 + [1] * 200)
    ens = fit(X, y, ["a", "b", "c"], target_fpr=0.05, target_fnr=0.05)
    assert ens.thresholds["human"] <= ens.thresholds["ai"]
    ens.save(tmp_path)
    loaded = Ensemble.load(tmp_path)
    assert np.allclose(loaded.predict_proba(X), ens.predict_proba(X))
