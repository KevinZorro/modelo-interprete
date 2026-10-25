"""Contract tests for lsc70.preprocessing (input contract v1)."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lsc70.normalizacion import vector_features
from lsc70.preprocessing import CONTRACT_VERSION, REASONS, preprocess

VECTORS = json.loads((Path(__file__).parent / "vectors" / "preprocessing_v1.json").read_text())


def _hand(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 0.8, (21, 3)) * np.array([1, 1, 0.1])


def _reference(pts, is_left):
    """Independent re-statement of the contract, written from the spec, not the code."""
    p = np.asarray(pts, dtype=np.float64) - np.asarray(pts, dtype=np.float64)[0]
    p = p / np.linalg.norm(p, axis=1).max()
    if is_left:
        p[:, 0] *= -1
    return p.flatten()


def test_golden_vectors_match_contract():
    assert VECTORS["contract"] == CONTRACT_VERSION
    for c in VECTORS["cases"]:
        r = preprocess(np.array(c["landmarks"]), c["is_left"])
        assert r.ok, c["name"]
        np.testing.assert_allclose(r.features, c["features"], atol=VECTORS["atol"])
        np.testing.assert_allclose(r.extras, c["extras"], atol=VECTORS["atol"])


def test_matches_independent_reference_and_notebook_copy():
    for left in (False, True):
        pts = _hand(3)
        r = preprocess(pts, left)
        np.testing.assert_allclose(r.features, _reference(pts, left), atol=1e-5)
        lm = [SimpleNamespace(x=a, y=b, z=c) for a, b, c in pts]
        v, _ = vector_features(lm, left)
        np.testing.assert_array_equal(r.features, v.astype(np.float32))


def test_output_shape_dtype_and_invariants():
    r = preprocess(_hand(), False)
    assert r.features.shape == (63,) and r.features.dtype == np.float32
    assert r.extras.shape == (3,) and r.extras.dtype == np.float32
    assert np.allclose(r.features[:3], 0)                       # wrist at origin
    assert np.isclose(np.linalg.norm(r.features.reshape(21, 3), axis=1).max(), 1.0, atol=1e-5)
    moved = preprocess(_hand() * 0.5 + 0.1, False)              # translation/scale invariant
    np.testing.assert_allclose(moved.features, r.features, atol=1e-4)


def test_left_hand_is_mirrored_in_x_only():
    right, left = preprocess(_hand(), False), preprocess(_hand(), True)
    a, b = right.features.reshape(21, 3), left.features.reshape(21, 3)
    np.testing.assert_allclose(b[:, 0], -a[:, 0], atol=1e-6)
    np.testing.assert_allclose(b[:, 1:], a[:, 1:], atol=1e-6)


def test_accepts_landmark_objects():
    pts = _hand()
    lm = [SimpleNamespace(x=a, y=b, z=c) for a, b, c in pts]
    np.testing.assert_array_equal(preprocess(lm, False).features, preprocess(pts, False).features)


def _bad(mutate):
    p = _hand()
    mutate(p)
    return p


@pytest.mark.parametrize("pts,reason", [
    (np.zeros((20, 3)), "bad_shape"),
    (np.zeros((21, 2)), "bad_shape"),
    ([], "bad_shape"),
    (None, "bad_shape"),
    (_bad(lambda p: p.__setitem__((5, 0), np.nan)), "non_finite"),
    (_bad(lambda p: p.__setitem__((5, 1), np.inf)), "non_finite"),
    (_bad(lambda p: p.__setitem__((5, 0), 3.0)), "out_of_range"),
    (_bad(lambda p: p.__setitem__((5, 2), 99.0)), "out_of_range"),
    (np.full((21, 3), 0.5), "degenerate_scale"),
])
def test_rejects_with_reason_and_never_raises(pts, reason):
    r = preprocess(pts, False)
    assert not r.ok and r.reason == reason and r.reason in REASONS


def test_slightly_outside_image_is_kept():
    p = _hand()
    p[8, 0] = -0.1                       # fingertip cut by the frame edge
    assert preprocess(p, False).ok
