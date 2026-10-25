import json
from pathlib import Path

import numpy as np

from lsc70 import t08_export as t8
from lsc70.preprocessing import preprocess

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "signaco_mlp_t08.tflite"
META = ROOT / "models" / "signaco_mlp_t08.json"


def test_parity_helper_flags_decision_changes():
    thr = np.full(3, 0.5)
    a = np.array([[0.1, 0.8, 0.1], [0.6, 0.3, 0.1]])
    b = np.array([[0.1, 0.8, 0.1], [0.45, 0.4, 0.15]])  # same argmax, falls under the threshold
    r = t8.parity(a, b, thr, none_idx=2)
    assert r["argmax_equal_pct"] == 100 and r["decision_equal_pct"] == 50


def test_committed_model_matches_metadata_and_contract_vectors():
    meta = json.loads(META.read_text())
    assert len(meta["labels"]) == len(meta["thresholds"]) == 28
    assert meta["labels"][meta["none_idx"]] == "none"
    it = t8.make_interpreter(MODEL)
    assert list(it.get_input_details()[0]["shape"]) == [1, 63]
    vec = json.loads((ROOT / "tests/vectors/preprocessing_v1.json").read_text())
    X = np.stack([preprocess(np.array(c["landmarks"]), c["is_left"]).features for c in vec["cases"]])
    P = t8.run_tflite(it, X)
    assert P.shape == (len(X), 28) and np.allclose(P.sum(1), 1, atol=1e-4)
