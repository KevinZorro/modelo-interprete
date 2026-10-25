import numpy as np

from lsc70 import t06_no_sena as t6


def _toy(rng, n_per=60, classes=4):
    centers = rng.normal(0, 1, (classes, 63)).astype(np.float32)
    y = np.repeat(np.arange(classes), n_per)
    part = np.array([f"P{i % 6}" for i in range(len(y))])
    X = centers[y] + rng.normal(0, 0.05, (len(y), 63)).astype(np.float32)
    return X, y, part


def test_synthetic_negatives_stay_outside_letter_clouds():
    rng = np.random.default_rng(0)
    X, y, part = _toy(rng)
    r = t6.nn_radius(X, y, part)
    for kind in ("transition", "hybrid"):
        Z = t6.synthesize(kind, X, y, part, 50, rng, r)
        assert Z.shape[1] == 63 and len(Z) > 0
        from scipy.spatial import cKDTree
        assert cKDTree(X).query(Z)[0].min() > r


def test_thresholds_respect_budget_and_ignore_none_class():
    rng = np.random.default_rng(1)
    C, none = 4, 3
    y = rng.integers(0, 3, 600)
    P = rng.dirichlet(np.ones(C) * 0.3, 600).astype(np.float32)
    P[np.arange(600), y] += 1.0
    P /= P.sum(1, keepdims=True)
    for pol in ("global", "perclass"):
        thr = t6.fit_thresholds(pol, P, y, none, budget=0.08)
        assert thr[none] == 0.0
        assert 1 - t6.decide(P, thr, none)[1].mean() <= 0.08 + 1e-9
    assert not t6.fit_thresholds("argmax", P, y, none, 0.08).any()


def test_decide_rejects_none_and_low_confidence():
    P = np.array([[0.9, 0.05, 0.05], [0.4, 0.3, 0.3], [0.1, 0.1, 0.8]], np.float32)
    pred, acc = t6.decide(P, np.array([0.5, 0.5, 0.0]), none_idx=2)
    assert list(acc) == [True, False, False]
