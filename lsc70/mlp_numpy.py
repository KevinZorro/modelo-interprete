"""T-08 classifier as a NumPy forward pass, so the review tools run without TensorFlow or LiteRT.

The deployed model (`models/signaco_mlp_t08.tflite`) is three dense layers (the batch norm is
already folded into the first) with ReLU and a softmax. `extract` copies its weights, once, to
`models/signaco_mlp_t08_weights.npz`; `NumpyMLP` then reproduces its probabilities (max abs
difference ~5e-7 against LiteRT, checked by tests/test_mlp_numpy.py when an interpreter exists).

    python -m lsc70.mlp_numpy extract        # needs ai-edge-litert or tensorflow, only once

The input is the 63-float vector of `lsc70.preprocessing.preprocess` (contract v1).
"""
import argparse
import json
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parent.parent / "models"
TFLITE = MODELS / "signaco_mlp_t08.tflite"
WEIGHTS = MODELS / "signaco_mlp_t08_weights.npz"
CONFIG = MODELS / "signaco_mlp_t08.json"
LAYERS = ("1", "2", "3")


class NumpyMLP:
    """Same probabilities as the T-08 .tflite, labels and thresholds from its JSON."""

    def __init__(self, weights: Path = WEIGHTS, config: Path = CONFIG):
        w = np.load(weights)
        self.W = [w[f"W{k}"].astype(np.float64) for k in LAYERS]
        self.b = [w[f"b{k}"].astype(np.float64) for k in LAYERS]
        cfg = json.loads(Path(config).read_text(encoding="utf-8"))
        self.labels = list(cfg["labels"])
        self.thresholds = np.asarray(cfg["thresholds"], dtype=np.float64)
        self.none_idx = int(cfg["none_idx"])
        if self.W[-1].shape[0] != len(self.labels):
            raise ValueError("weights and config disagree on the number of classes")

    def predict_proba(self, X) -> np.ndarray:
        h = np.asarray(X, dtype=np.float64).reshape(-1, self.W[0].shape[1])
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = np.maximum(h @ W.T + b, 0.0)
        z = h @ self.W[-1].T + self.b[-1]
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


def load_default():
    """The T-08 model when its NumPy weights are present, else None (callers fall back)."""
    return NumpyMLP() if WEIGHTS.exists() and CONFIG.exists() else None


def _interpreter(path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
        it = Interpreter(model_path=str(path),
                         experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    except ImportError:
        import tensorflow as tf
        it = tf.lite.Interpreter(model_path=str(path))
    it.allocate_tensors()
    return it


def extract(tflite: Path = TFLITE, out: Path = WEIGHTS) -> dict:
    """Copy the dense weights of the .tflite to an .npz and check the graph is what we assume."""
    it = _interpreter(tflite)
    ops = it._get_ops_details()
    names = [o["op_name"] for o in ops if o["op_name"] != "DELEGATE"]
    if names != ["FULLY_CONNECTED"] * 3 + ["SOFTMAX"]:
        raise ValueError(f"unexpected graph {names}: update NumpyMLP before extracting")
    arrays = {}
    for k, op in zip(LAYERS, ops[:3]):
        _, w_idx, b_idx = op["inputs"]
        arrays[f"W{k}"], arrays[f"b{k}"] = it.get_tensor(w_idx), it.get_tensor(b_idx)
    np.savez(out, **arrays)
    return {k: list(v.shape) for k, v in arrays.items()}


def tflite_proba(X, tflite: Path = TFLITE) -> np.ndarray:
    """Reference probabilities from the interpreter, one frame at a time (tests only)."""
    it = _interpreter(tflite)
    ent, sal = it.get_input_details()[0], it.get_output_details()[0]
    out = []
    for x in np.asarray(X, dtype=np.float32):
        it.set_tensor(ent["index"], x[None])
        it.invoke()
        out.append(it.get_tensor(sal["index"])[0].copy())
    return np.asarray(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="NumPy copy of the T-08 classifier")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract")
    ap.parse_args(argv)
    print(extract())


if __name__ == "__main__":
    main()
