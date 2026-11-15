"""T-16: measure the Colab 36-class model (letters + numbers) next to the T-08 MLP.

The Colab model was trained on ALL participants (its config says so), and the T-08 MLP on all
55 dev people, so every number here on cached people is resubstitution (an upper bound), not
generalisation. Only the Colab model's own cross-validation figures are held-out. The frozen
test is NOT used. The cache has letters only: numbers cannot be measured with it.
Alias poses (configs/alias_poses.json, T-17): 1 = F, so F predicted as "1" counts as correct.

    python -m lsc70.t16_medir_36 --model PATH.tflite --out docs/reports/T-16
"""
import argparse
import json
from pathlib import Path

import numpy as np

from lsc70 import frozen_split
from lsc70.baseline import LETTER_LIKE, RAIZ

LABELS36 = ([chr(c) for c in range(65, 79)] + ["NN"] + [chr(c) for c in range(79, 91)]
            + ["1", "4", "5", "6", "7", "8", "9", "10", "no_es_seña"])
ALIAS = {"F": "1"}  # same pose, a letter prediction of the alias class is correct


def interpreter(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=str(path), num_threads=1)
    it.allocate_tensors()
    return it


def run(it, X):
    ent, sal = it.get_input_details()[0], it.get_output_details()[0]
    out = np.zeros((len(X), sal["shape"][-1]), np.float32)
    for i, x in enumerate(X):
        it.set_tensor(ent["index"], x[None].astype(np.float32))
        it.invoke()
        out[i] = it.get_tensor(sal["index"])[0]
    return out


def toma_top1(P, y, part, ok_cols):
    """Top-1 per toma (person + class, probabilities averaged); ok_cols[c] = accepted columns."""
    hit = tot = 0
    per = {}
    for p in np.unique(part):
        for c in np.unique(y[part == p]):
            m = (part == p) & (y == c)
            good = int(P[m].mean(0).argmax() in ok_cols[c])
            hit += good
            tot += 1
            per.setdefault(int(c), []).append(good)
    return hit / tot, tot, {k: float(np.mean(v)) for k, v in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = np.load(RAIZ / "landmarks_cache.npz", allow_pickle=True)
    lab = [str(e) for e in d["etiquetas"]][:27]
    dev = set(frozen_split.load_freeze()["dev_fold_de"])
    part = d["ptr"].astype(str)
    keep = np.isin(part, list(dev)) & (d["ytr"] < 27)
    X, y, part = d["Xtr"][keep], d["ytr"][keep], part[keep]
    neg = np.load(RAIZ / "negativos_reposo.npz", allow_pickle=True)
    Xn = neg["X_neg"][~np.isin(neg["clase_por_muestra"].astype(str), list(LETTER_LIKE))]
    res = {"people": len(np.unique(part)), "frames": int(len(X)), "neg_frames": int(len(Xn))}
    specs = {"colab36": (a.model, LABELS36), "t08": (RAIZ / "models/signaco_mlp_t08.tflite",
                                                    lab + ["none"])}
    for name, (path, labels) in specs.items():
        it = interpreter(path)
        P, Pn = run(it, X), run(it, Xn)
        none = labels.index("none" if name == "t08" else "no_es_seña")
        ok = {c: {labels.index(lab[c])} | ({labels.index(ALIAS[lab[c]])}
                                           if lab[c] in ALIAS and ALIAS[lab[c]] in labels else set())
              for c in range(27)}
        # same decision space for both: argmax over the model's own classes
        top, n, per = toma_top1(P, y, part, ok)
        pred = P.argmax(1)
        fr_frame = float((pred == none).mean())  # letter frames rejected at threshold 0
        fa_frame = float((Pn.argmax(1) != none).mean())  # resting-hand frames accepted
        to_num = float(np.isin(pred, [i for i, l in enumerate(labels)
                                      if l.isdigit()]).mean()) if name == "colab36" else 0.0
        res[name] = {"top1_toma_argmax": round(top, 4), "n_tomas": n,
                     "fr_frame_thr0": round(fr_frame, 4), "fa_rest_frame_thr0": round(fa_frame, 4),
                     "letters_read_as_number": round(to_num, 4),
                     "worst5": sorted(((lab[c], round(v, 2)) for c, v in per.items()),
                                      key=lambda t: t[1])[:5]}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "medicion.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
