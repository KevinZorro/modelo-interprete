"""Regenerates tests/vectors/preprocessing_v1.json. Run only when the contract version
changes on purpose: the file IS the contract, so a diff in it is a breaking change."""
import json
from pathlib import Path

import numpy as np

from lsc70.preprocessing import CONTRACT_VERSION, preprocess


def hand(seed):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.2, 0.8, (21, 3)) * np.array([1, 1, 0.1])
    return np.round(pts, 6)


cases = []
for name, seed, left in [("right_a", 1, False), ("left_a", 1, True), ("right_b", 2, False)]:
    pts = hand(seed)
    r = preprocess(pts, left)
    cases.append({"name": name, "is_left": left, "landmarks": pts.tolist(),
                  "features": [round(float(v), 7) for v in r.features],
                  "extras": [round(float(v), 7) for v in r.extras]})
out = {"contract": CONTRACT_VERSION, "atol": 1e-6, "cases": cases}
Path(__file__).parent.joinpath("vectors", "preprocessing_v1.json").write_text(json.dumps(out))
