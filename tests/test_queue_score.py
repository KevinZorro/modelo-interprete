import numpy as np

from lsc70 import queue_score as qs
from lsc70 import review_queue as rq


def test_event_clip_holds_exactly_one_event():
    # three held signs 0.5 s apart: the middle one must not loop over its neighbours
    segs = [(0.0, 1.0, 1.3), (1.5, 2.5, 1.3), (3.0, 4.0, 1.3)]
    clips = rq.event_clips(segs)
    lo, hi = clips[(1.5, 2.5)]
    assert lo >= 1.0 and hi <= 3.0          # cut at the neighbours' limits
    assert lo <= 1.5 and hi >= 2.5          # still contains its own segment
    for a, b, _ in segs:
        for other, (clo, chi) in clips.items():
            if other != (a, b):
                assert not (clo <= a <= chi and clo <= b <= chi)


def test_event_clip_context_and_max_length():
    clips = rq.event_clips([(10.0, 11.0, 1.2)])
    lo, hi = clips[(10.0, 11.0)]
    assert lo == 9.0 and hi == 12.0         # one second of context on each side
    long = rq.event_clips([(10.0, 20.0, 10.0)])
    lo, hi = long[(10.0, 20.0)]
    assert hi - lo <= rq.CLIP_MAX_S + 1e-9


def test_event_clip_ignores_unheld_transitions():
    # a one-frame transition between two signs is not an event: it does not cut the other clip
    clips = rq.event_clips([(0.0, 1.0, 1.2), (1.1, 1.1, 0.3), (5.0, 6.0, 1.2)])
    assert clips[(0.0, 1.0)][1] == 2.0


def _rows(n, step=0.33):
    rng = np.random.RandomState(0)
    base = rng.rand(21, 3)
    rows = [{"frame": i, "t_s": i * step, "landmarks": (base + 0.001 * i).tolist(),
             "features": rng.rand(63).tolist()} for i in range(n)]
    X = rq._unit(np.asarray([r["features"] for r in rows]))
    return rows, X


def test_pose_features_shape_and_repeat():
    rows, X = _rows(6)
    f = qs.pose_features(rows, X, 2, 0.1, 0.7)
    assert len(f) == len(qs.FEATURES)
    assert f[0] == 0.1 and 0 < f[4] <= 1.0   # repeat counts the sample itself
    X_same = np.tile(X[:1], (6, 1))
    assert qs.pose_features(rows, X_same, 2, 0.1, 0.7)[4] == 1.0


def test_pose_features_isolated_sample_counts_as_fast():
    rows, X = _rows(1)
    assert qs.pose_features(rows, X, 0, 0.1, 0.7)[3] == np.log(1.0 + 1e-3)


def test_fit_learns_known_direction():
    rng = np.random.RandomState(1)
    y = (rng.rand(400) < 0.3).astype(int)
    X = rng.randn(400, 5)
    X[:, 0] += -2.0 * y                       # closer (smaller dist) for real signs
    X[:, 4] += 2.0 * y                        # repeats more
    m, s, w, b = qs.fit(X, y)
    assert w[0] < 0 < w[4]
    assert qs._auc(y, ((X - m) / s) @ w + b) > 0.9


def test_cross_validation_keeps_videos_whole_and_beats_chance():
    rng = np.random.RandomState(2)
    groups = np.repeat([f"v{i}" for i in range(20)], 15)
    y = (rng.rand(300) < 0.3).astype(int)
    X = rng.randn(300, 5)
    X[:, 0] += -2.0 * y
    oof = qs.cross_validate(X, y, groups)
    assert oof.shape == (3, 300)
    assert qs._auc(y, oof.mean(0)) > 0.8
    assert qs.precision_at(oof.mean(0), y, 0.1) > y.mean()


def test_score_uses_config():
    cfg = {"mean": [0, 0], "std": [1, 1], "weights": [1.0, -2.0], "bias": 0.5}
    assert qs.score(cfg, [2.0, 1.0]) == 0.5
