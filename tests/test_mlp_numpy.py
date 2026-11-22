from pathlib import Path

import numpy as np
import pytest

from lsc70 import mlp_numpy

CACHE = Path("landmarks_cache.npz")


def _has_interpreter():
    try:
        import ai_edge_litert.interpreter  # noqa: F401
        return True
    except ImportError:
        try:
            import tensorflow  # noqa: F401
            return True
        except ImportError:
            return False


def test_numpy_model_matches_config_and_returns_probabilities():
    m = mlp_numpy.NumpyMLP()
    assert len(m.labels) == 28 and m.labels[m.none_idx] == "none"
    P = m.predict_proba(np.random.default_rng(0).normal(0, 0.3, (5, 63)))
    assert P.shape == (5, 28) and np.allclose(P.sum(1), 1.0)


@pytest.mark.skipif(not _has_interpreter() or not CACHE.exists(), reason="needs LiteRT/TF and landmarks_cache.npz")
def test_numpy_model_reproduces_the_tflite():
    X = np.load(CACHE, allow_pickle=True)["Xtr"][::25][:300]   # train frames only, never the frozen test
    P, R = mlp_numpy.NumpyMLP().predict_proba(X), mlp_numpy.tflite_proba(X)
    assert np.abs(P - R).max() < 1e-5
    assert (P.argmax(1) == R.argmax(1)).all()
