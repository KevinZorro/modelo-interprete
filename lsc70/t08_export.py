"""T-08: export the T-07 MLP (trial 19) to TFLite, unquantized, and measure parity and latency.

The deployed model is trained exactly like `t07_optuna.final` (55 dev people + resting-hand
negatives, seed 1) and paired with the cross-fitted per-class thresholds T-07 stored for that
seed. The frozen test is NOT touched. Parity is checked on (a) the contract vectors of
`lsc70.preprocessing` (landmarks -> preprocess -> model) and (b) real dev frames, comparing
probabilities AND the final accept/reject decision.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from lsc70.baseline import RAIZ
from lsc70.preprocessing import preprocess
from lsc70.t05_baselines import N_FOLDS
from lsc70.t06_no_sena import decide
from lsc70.t07_optuna import Data, fit_mlp

SEED_IDX = 0  # confirm seed 1; thresholds[0] belong to that seed
N_FRAMES = 2000
MAX_DIFF = 1e-5


def make_interpreter(path):
    try:  # LiteRT is what the app uses; fall back to TF's deprecated one
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=str(path), num_threads=1)
    it.allocate_tensors()
    return it


def run_tflite(it, X):
    ent, sal = it.get_input_details()[0], it.get_output_details()[0]
    out = np.zeros((len(X), sal["shape"][-1]), np.float32)
    for i, x in enumerate(X):
        it.set_tensor(ent["index"], x[None].astype(np.float32))
        it.invoke()
        out[i] = it.get_tensor(sal["index"])[0]
    return out


def convert(model, folder):
    import tensorflow as tf

    conv = tf.lite.TFLiteConverter.from_keras_model(model)  # no optimizations: float32
    path = Path(folder) / "signaco_mlp_t08.tflite"
    path.write_bytes(conv.convert())
    return path


def parity(keras_p, tfl_p, thr, none_idx):
    pk, ak = decide(keras_p, thr, none_idx)
    pt, at = decide(tfl_p, thr, none_idx)
    return {"n": len(keras_p), "max_abs_diff": float(np.abs(keras_p - tfl_p).max()),
            "argmax_equal_pct": 100 * float((pk == pt).mean()),
            "decision_equal_pct": 100 * float(((pk == pt) & (ak == at)).mean())}


def latency(it, X, reps=2000):
    ent, sal = it.get_input_details()[0], it.get_output_details()[0]
    x = X[:1].astype(np.float32)

    def once():
        it.set_tensor(ent["index"], x)
        it.invoke()
        it.get_tensor(sal["index"])

    for _ in range(100):
        once()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        once()
        t.append((time.perf_counter() - t0) * 1e3)
    return {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)),
            "threads": 1, "host": "cloud container CPU (NOT a phone)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", default=str(RAIZ / "docs/reports/T-07/dev_results.json"))
    ap.add_argument("--out", default=str(RAIZ / "models"))
    ap.add_argument("--report", default=str(RAIZ / "docs/reports/T-08/export.json"))
    a = ap.parse_args()
    dev = json.loads(Path(a.dev).read_text())
    best = dev["confirm"]["best"]
    p, thr = best["params"], np.array(best["thresholds"][SEED_IDX])
    seed = json.loads(Path(a.dev).read_text())["config"]["confirm_seeds"][SEED_IDX]

    data = Data(dev["config"])
    d, n = data.d, data.none_idx
    keep = data.nfold != N_FOLDS - 1  # same split as t07_optuna.final
    X, y, part = d["Xtr"], d["ytr"], d["ptr"].astype(str)
    Xt = np.vstack([X, data.Xn[keep]])
    yt = np.concatenate([y, np.full(keep.sum(), n)])
    gt = np.concatenate([part, [f"neg{i // 10}" for i in range(keep.sum())]])
    model = fit_mlp(p, Xt, yt, gt, n + 1, seed)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = convert(model, out)
    it = make_interpreter(path)

    vec = json.loads((Path(__file__).parent.parent / "tests/vectors/preprocessing_v1.json").read_text())
    Xv = np.stack([preprocess(np.array(c["landmarks"]), c["is_left"]).features for c in vec["cases"]])
    rng = np.random.default_rng(0)  # dev frames only: the frozen test is not used
    Xd = X[rng.choice(len(X), N_FRAMES, replace=False)]
    res = {"seed": seed, "params": p, "model_bytes": path.stat().st_size,
           "unquantized": True, "contract": vec["contract"]}
    for name, Xs in (("contract_vectors", Xv), ("dev_frames", Xd)):
        res[name] = parity(model.predict(Xs, verbose=0), run_tflite(it, Xs), thr, n)
    res["latency"] = latency(it, Xd)
    res["pass_max_diff"] = all(res[k]["max_abs_diff"] <= MAX_DIFF
                               for k in ("contract_vectors", "dev_frames"))

    labels = data.labels  # index == model output; `none_idx` is the no-sign class
    (out / "signaco_mlp_t08.json").write_text(json.dumps(
        {"labels": labels, "none_idx": n, "thresholds": thr.tolist(), "contract": vec["contract"],
         "input": "float32[1,63] from lsc70.preprocessing.preprocess",
         "decision": "accept iff argmax != none_idx and prob >= thresholds[argmax]"}, indent=2))
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    assert res["pass_max_diff"], "parity above tolerance"


if __name__ == "__main__":
    main()
