import numpy as np

from lsc70.features_v2 import MIDDLE_MCP, palm_scale
from lsc70.preprocessing import preprocess


def _hand(seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.3, 0.7, (21, 3))
    pts[:, 2] *= 0.1
    return pts


def test_palm_is_unit_after_v2():
    f = preprocess(_hand(), False).features
    v2 = palm_scale(f[None])[0].reshape(21, 3)
    assert np.allclose(np.linalg.norm(v2[MIDDLE_MCP]), 1.0, atol=1e-5)
    assert np.allclose(v2[0], 0)


def test_v2_equals_raw_over_palm():
    """v2 from the v1 vector must equal scaling the raw wrist-centred hand by the palm."""
    raw = _hand(1)
    rel = raw - raw[0]
    expected = rel / np.linalg.norm(rel[MIDDLE_MCP])
    got = palm_scale(preprocess(raw, False).features[None])[0].reshape(21, 3)
    assert np.allclose(got, expected, atol=1e-5)


def test_v2_ignores_global_scale_and_degenerate_rows():
    f = preprocess(_hand(2), False).features
    assert np.allclose(palm_scale(f[None]), palm_scale((f * 0.5)[None]), atol=1e-5)
    z = np.zeros((1, 63), np.float32)
    assert (palm_scale(z) == 0).all()
