"""T-17: fair comparison of the Colab 36-class MLP (letters + numbers) against the letters-only MLP.

T-16 could only measure the Colab model by resubstitution (it saw every person). Here the SAME
architecture (63-256-128-C, the T-07 control) is retrained inside the 5 person-folds of T-05/T-07
(55 dev people), so every number below is held-out by person. Two arms, same folds, seeds, negatives,
per-class threshold policy and frame-rejection budget as T-07:

  control  letters + "none" (28 outputs). Numbers present in the cache are labelled none, exactly
           as in T-07/T-08 (reproduces the T-07 control row as a sanity check).
  with36   letters + numbers 1,4,5,6,7,8,9,10 + none (36 outputs, the Colab label set), numbers
           taken from LSC70ANH landmarks extracted with lsc70.extraer (same normalisation).

Numbers are scored on the same held-out frames in both arms: with36 as recall/error, control as
"number accepted as a letter". 1 = F share one pose (configs/alias_poses.json), so the 36-class
probabilities of "1" are folded into F before scoring. MIL/MILLON are left out (not in the Colab
label set). The frozen test is NOT used.

    python -m lsc70.t17_numeros_letras --config configs/t17_numeros.yaml --numbers landmarks_anh.npz \\
        --out docs/reports/T-17
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from lsc70 import frozen_split
from lsc70.baseline import RAIZ, macro_f1, to_tomas, wilson
from lsc70.t05_baselines import N_FOLDS, load
from lsc70.t06_no_sena import commit, decide, fit_thresholds
from lsc70.t07_optuna import CONTROL, fit_mlp

NUMBERS = ["1", "4", "5", "6", "7", "8", "9", "10"]
DYNAMIC = {"6", "7", "8", "9", "10"}  # Kevin: signs with movement; a single pose cannot teach them
N_LETTERS = 27


def load_numbers(path, folds):
    """Frames of the Colab number classes for dev people: X, class index into NUMBERS, person."""
    z = np.load(path, allow_pickle=True)
    cls = z["y"].astype(str)
    part = z["participante"].astype(str)
    keep = np.isin(cls, NUMBERS) & np.isin(part, list(folds))
    return (z["X"][keep].astype(np.float32), np.array([NUMBERS.index(c) for c in cls[keep]]),
            part[keep])


def fold_scores(arm, data, seed, num):
    """Cross-fitted scores for letters (P), negatives (Pn) and number frames (Pnum)."""
    d, Xn, nfold, folds = data
    X, y, part = d["Xtr"], d["ytr"], d["ptr"].astype(str)
    fold_of = np.array([folds[q] for q in part])
    Xnum, ynum, pnum = num
    fnum = np.array([folds[q] for q in pnum])
    n_none = 27 if arm == "control" else N_LETTERS + len(NUMBERS)
    C = n_none + 1
    P, Pn, Pnum = (np.zeros((len(X), C), np.float32), np.zeros((len(Xn), C), np.float32),
                   np.zeros((len(Xnum), C), np.float32))
    for k in range(N_FOLDS):
        tr, ev, ntr, nev = fold_of != k, fold_of == k, nfold != k, nfold == k
        Xt, yt, gt = X[tr], y[tr], part[tr]
        if arm == "control":
            yt = np.where(yt == 27, n_none, yt)  # cache none (incl. numbers) = last class
        else:
            keep = yt < N_LETTERS  # drop the cache's number rows: replaced by labelled ones
            Xt, yt, gt = Xt[keep], yt[keep], gt[keep]
            m = fnum != k
            Xt = np.vstack([Xt, Xnum[m]])
            yt = np.concatenate([yt, N_LETTERS + ynum[m]])
            gt = np.concatenate([gt, pnum[m]])
        Xt = np.vstack([Xt, Xn[ntr]])
        yt = np.concatenate([yt, np.full(ntr.sum(), n_none)])
        gt = np.concatenate([gt, [f"neg{i // 10}" for i in np.where(ntr)[0]]])
        model = fit_mlp(CONTROL, Xt, yt, gt, C, seed)
        def pe(A):
            return (model.predict(A, verbose=0, batch_size=1024) if len(A)
                    else np.zeros((0, C), np.float32))
        P[ev], Pn[nev], Pnum[fnum == k] = pe(X[ev]), pe(Xn[nev]), pe(Xnum[fnum == k])
    return P, Pn, Pnum, fold_of, fnum


def fold_alias(P, F):
    """1 and F are one pose: move the probability of class '1' (index 27) onto F."""
    P = P.copy()
    P[:, F] += P[:, N_LETTERS]
    P[:, N_LETTERS] = 0.0
    return P


def evaluate(arm, P, Pn, Pnum, y, part, pnum, ynum, fold_of, fnum, nfold, neg_class, budget, F):
    n_none = P.shape[1] - 1
    if arm == "with36":
        P, Pn, Pnum = fold_alias(P, F), fold_alias(Pn, F), fold_alias(Pnum, F)
    letter = y < N_LETTERS
    thr = {}
    for k in range(N_FOLDS):
        m = letter & (fold_of != k)
        Pg, yg = P[m], y[m]
        if arm == "with36":  # numbers are genuine classes in this arm
            nm = fnum != k
            Pg = np.vstack([Pg, Pnum[nm]])
            yg = np.concatenate([yg, N_LETTERS + ynum[nm]])
            yg = np.where(yg == N_LETTERS, F, yg)  # '1' folded into F
        thr[k] = fit_thresholds("perclass", Pg, yg, n_none, budget)

    def accept(Pm, fo):
        acc, pred = np.zeros(len(Pm), bool), Pm.argmax(1)
        for k in range(N_FOLDS):
            r = fo == k
            acc[r] = decide(Pm[r], thr[k], n_none)[1]
        return acc, pred

    accL, predL = accept(P, fold_of)
    accN, predN = accept(Pn, nfold)
    accM, predM = accept(Pnum, fnum)
    Pl, yl, pl = P[letter], y[letter], part[letter]
    Pt, yt, _ = to_tomas(Pl, yl, pl)
    keys = {}
    for i, (p, c) in enumerate(zip(pl, yl)):
        keys.setdefault((str(p), int(c)), []).append(i)
    ks = sorted(keys)
    at = np.array([accL[letter][keys[k]].mean() >= 0.5 for k in ks]) & (Pt.argmax(1) != n_none)
    shown = np.where(at, Pt.argmax(1), n_none)
    f1, per = macro_f1(shown, yt, N_LETTERS)
    lo, hi = wilson(int(accN.sum()), len(accN))
    rest = neg_class == "no_gesture"
    res = {
        "f1_macro": f1, "top1": float((shown == yt).mean()), "fr_toma": float((~at).mean()),
        "err_toma": float((at & (Pt.argmax(1) != yt)).mean()),
        "fr_frame": float((~accL[letter]).mean()), "fa": float(accN.mean()), "fa_ci95": [lo, hi],
        "fa_letter": float((accN & (predN < N_LETTERS)).mean()),
        "fa_rest_hand": float(accN[rest].mean()), "n_tomas": int(len(yt)), "n_neg": int(len(accN)),
        "recall_worst3": sorted((p["recall"], i) for i, p in enumerate(per))[:3],
    }
    # numbers, toma level (person + class), alias 1 = F
    nk = {}
    for i, (p, c) in enumerate(zip(pnum, ynum)):
        nk.setdefault((p, int(c)), []).append(i)
    nums = {}
    for c, name in enumerate(NUMBERS):
        ts = [v for (p, cc), v in nk.items() if cc == c]
        if not ts:
            continue
        ok = lt = 0
        for v in ts:
            pm = Pnum[v].mean(0).argmax()
            acc = accM[v].mean() >= 0.5 and pm != n_none
            right = pm == (F if name == "1" else N_LETTERS + c)
            ok += int(acc and right)
            lt += int(acc and pm < N_LETTERS and not (name == "1" and pm == F))
        nums[name] = {"n_tomas": len(ts), "recall": ok / len(ts), "as_letter": lt / len(ts),
                      "dynamic": name in DYNAMIC}
    res["numbers"] = nums
    sel = np.array([NUMBERS[c] != "1" for c in ynum])
    res["numbers_as_letter_frame"] = float((accM & (predM < N_LETTERS))[sel].mean())
    return res


def run(cfg, numbers, out):
    d, labels, Xn, nfold = load(RAIZ / "landmarks_cache.npz", RAIZ / cfg["negatives"])
    folds = frozen_split.load_freeze()["dev_fold_de"]
    cls = np.load(RAIZ / cfg["negatives"], allow_pickle=True)["clase_por_muestra"].astype(str)
    from lsc70.baseline import LETTER_LIKE
    neg_class = cls[~np.isin(cls, list(LETTER_LIKE))]
    num = load_numbers(numbers, folds)
    F = labels.index("F")
    y, part = d["ytr"], d["ptr"].astype(str)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"numbers: {len(num[0])} frames, {len(set(num[2]))} dev people; letters {int((y < 27).sum())}")
    results = {"commit": commit(), "cfg": cfg, "n_number_frames": int(len(num[0])),
               "n_number_people": int(len(set(num[2]))), "arms": {}}
    for arm in ("control", "with36"):
        rs = []
        for seed in cfg["seeds"]:
            P, Pn, Pnum, fo, fnum = fold_scores(arm, (d, Xn, nfold, folds), seed, num)
            r = evaluate(arm, P, Pn, Pnum, y, part, num[2], num[1], fo, fnum, nfold, neg_class,
                         cfg["fr_budget"], F)
            r["seed"] = seed
            rs.append(r)
            print(arm, seed, {k: round(r[k], 3) for k in ("f1_macro", "fr_toma", "fa", "fa_letter")},
                  flush=True)
        results["arms"][arm] = rs
    (out / "dev_results.json").write_text(json.dumps(results, indent=1, default=float))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--numbers", required=True, help="landmarks npz from lsc70.extraer (ANH)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(yaml.safe_load(open(a.config)), a.numbers, a.out)


if __name__ == "__main__":
    main()
