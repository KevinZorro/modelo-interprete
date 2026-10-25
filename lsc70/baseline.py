"""Baseline measurement of the current model on the frozen test (T-03).

Unit of evaluation is the toma (person + class, frames averaged). Rejection = argmax is the
last class ("none" / no_es_seña). Definitions:
  FR  : fraction of letter tomas rejected (predicted none).
  err : fraction of letter tomas assigned to another letter.
  FA  : fraction of negative frames predicted as a letter (negatives have no person id, so
        frames are not independent: the Wilson CI is optimistic).
Number tomas (class "none" in the cache) are reported apart: they are real signs, not noise.
Bootstrap resamples the 15 test people, not tomas.

    python -m lsc70.baseline --model "signaco_abecedario_rechazo (1).tflite" --out resultados/t03
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lsc70 import frozen_split
from lsc70.frozen_split import CONFIGS

RAIZ = Path(__file__).resolve().parent.parent
# HaGRID classes that are really letters R, L, L, S (CLAUDE.md): not valid negatives.
LETTER_LIKE = {"two_up", "palm", "three2", "one"}
THRESHOLDS = (0.5, 0.7, 0.9)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def predict_tflite(model_path, X):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=str(model_path))
    it.allocate_tensors()
    i = it.get_input_details()[0]["index"]
    o = it.get_output_details()[0]["index"]
    out = []
    for x in X:
        it.set_tensor(i, x[None].astype("float32"))
        it.invoke()
        out.append(it.get_tensor(o)[0].copy())
    return np.array(out)


def to_tomas(P, y, part):
    groups = {}
    for i, (p, c) in enumerate(zip(part, y)):
        groups.setdefault((str(p), int(c)), []).append(i)
    keys = sorted(groups)
    return (
        np.stack([P[groups[k]].mean(0) for k in keys]),
        np.array([k[1] for k in keys]),
        np.array([k[0] for k in keys]),
    )


def macro_f1(pred, true, n_letters):
    f1, per_class = [], []
    for c in range(n_letters):
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        f1.append(f)
        per_class.append({"precision": p, "recall": r, "f1": f, "n": tp + fn})
    return float(np.mean(f1)), per_class


def confusion(pred, true, n_classes):
    M = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true, pred):
        M[t, p] += 1
    return M


def summarize(P, y, part, none_idx):
    """Metrics on letter tomas. P are toma probabilities, y class ids (letters < none_idx)."""
    pred = P.argmax(1)
    f1, per_class = macro_f1(pred, y, none_idx)
    return {
        "f1_macro": f1,
        "top1": float((pred == y).mean()),
        "fr": float((pred == none_idx).mean()),
        "err": float(((pred != y) & (pred != none_idx)).mean()),
        "per_class": per_class,
        "confusion": confusion(pred, y, none_idx + 1),
    }


def bootstrap_people(P, y, part, none_idx, reps=1000, seed=0):
    rng = np.random.default_rng(seed)
    people = np.array(sorted(set(part)))
    idx_by = {p: np.where(part == p)[0] for p in people}
    out = {"f1_macro": [], "fr": [], "top1": []}
    for _ in range(reps):
        pick = rng.choice(people, len(people))
        idx = np.concatenate([idx_by[p] for p in pick])
        pred = P[idx].argmax(1)
        out["f1_macro"].append(macro_f1(pred, y[idx], none_idx)[0])
        out["fr"].append(float((pred == none_idx).mean()))
        out["top1"].append(float((pred == y[idx]).mean()))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


def fa_report(P, none_idx, classes=None):
    pred = P.argmax(1)
    acc = pred != none_idx
    rep = {"n": int(len(P)), "fa": float(acc.mean()), "ci95": wilson(int(acc.sum()), len(P))}
    rep["fa_at_threshold"] = {
        str(t): float(((P[:, :none_idx].max(1) >= t) & acc).mean()) for t in THRESHOLDS
    }
    if classes is not None:
        rep["by_class"] = {
            c: float(acc[classes == c].mean()) for c in sorted(set(classes.tolist()))
        }
    return rep


def run(model, cache, negatives, splits_path, out_dir):
    problems = frozen_split.verify(splits_path)
    if problems:
        raise SystemExit("Frozen test check failed: " + "; ".join(problems))
    d = np.load(cache, allow_pickle=True)
    labels = [str(e) for e in d["etiquetas"]]
    none_idx = labels.index("none")
    frozen = set(frozen_split.load_freeze()["test"])
    part_te = d["pte"].astype(str)
    assert set(part_te) == frozen, "cache test people are not the frozen test"

    P_frames = predict_tflite(model, d["Xte"])
    y = d["yte"]
    letter = y != none_idx
    Pt, yt, pt = to_tomas(P_frames[letter], y[letter], part_te[letter])
    res = summarize(Pt, yt, pt, none_idx)
    res["ci95"] = bootstrap_people(Pt, yt, pt, none_idx)
    res["n_tomas"] = int(len(yt))
    res["n_people"] = int(len(set(pt)))

    neg = np.load(negatives, allow_pickle=True)
    cls = neg["clase_por_muestra"].astype(str)
    Pn = predict_tflite(model, neg["X_neg"])
    clean = ~np.isin(cls, list(LETTER_LIKE))
    res["fa_clean"] = fa_report(Pn[clean], none_idx, cls[clean])
    res["fa_all_800"] = fa_report(Pn, none_idx)
    res["numbers_frames_as_letter"] = fa_report(P_frames[~letter], none_idx)
    # FR at the same thresholds, for the FA-vs-FR trade-off (feeds T-06)
    pl = Pt[:, :none_idx]
    res["fr_at_threshold"] = {
        str(t): float(((pl.max(1) < t) | (Pt.argmax(1) == none_idx)).mean()) for t in THRESHOLDS
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    conf = res.pop("confusion")
    np.savetxt(out / "confusion_tomas.csv", conf, fmt="%d", delimiter=",",
               header=",".join(labels), comments="")
    res["labels"] = labels
    (out / "baseline.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(RAIZ / "signaco_abecedario_rechazo (1).tflite"))
    ap.add_argument("--cache", default=str(RAIZ / "landmarks_cache.npz"))
    ap.add_argument("--negatives", default=str(RAIZ / "negativos_reposo.npz"))
    ap.add_argument("--splits", default=str(CONFIGS / "splits_participantes.json"))
    ap.add_argument("--out", default=str(RAIZ / "resultados" / "t03"))
    a = ap.parse_args()
    r = run(a.model, a.cache, a.negatives, a.splits, a.out)
    print(f"tomas={r['n_tomas']} F1 macro={r['f1_macro']:.3f} {r['ci95']['f1_macro']} "
          f"top1={r['top1']:.3f} FR={r['fr']:.3f} err={r['err']:.3f}")
    print(f"FA clean={r['fa_clean']['fa']:.3f} {r['fa_clean']['ci95']} "
          f"all800={r['fa_all_800']['fa']:.3f} numbers->letter={r['numbers_frames_as_letter']['fa']:.3f}")


if __name__ == "__main__":
    main()
