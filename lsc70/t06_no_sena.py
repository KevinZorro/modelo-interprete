"""T-06: widen the "none" class with synthetic hard negatives and calibrate rejection thresholds.

One change at a time, on the same 5 development folds as T-05 (55 people). The control is the
T-05 setup (HaGRID negatives + numbers as none). Each variant adds ONE family of synthetic
negatives built only from the training-fold people, and is scored on REAL negatives only
(HaGRID frames held out by position, and number frames), so synthetic data never grades itself.

  transition : convex blend (alpha 0.3-0.7) of two different letters of the same person
  hybrid     : one or two fingers of a letter replaced by the same fingers of another letter

A synthetic vector is kept only if its nearest real letter frame is farther than `r`, the median
distance from a letter frame to its nearest same-class frame of ANOTHER person: it must fall
outside the usual spread of a letter, or it would teach the model to reject real signs.

Thresholds (policy "global" / "perclass") are fitted on the out-of-fold scores of the other four
folds and applied to the held-out fold (cross-fitted), so the dev numbers are not tuned on
themselves. The rule is the largest threshold whose genuine frame-rejection stays within a budget.

    python -m lsc70.t06_no_sena --config configs/t06_no_sena.yaml --out docs/reports/T-06
    python -m lsc70.t06_no_sena --final logreg:hybrid --out docs/reports/T-06
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

from lsc70 import frozen_split
from lsc70.baseline import LETTER_LIKE, RAIZ, macro_f1, to_tomas, wilson
from lsc70.t05_baselines import N_FOLDS, fit_scores, load

FINGERS = {  # landmark indices per finger (MediaPipe hand model)
    "thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20),
}
GRID = np.round(np.arange(0.0, 1.0, 0.005), 3)


def softmax(S):
    S = S - S.max(1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(1, keepdims=True)


def nn_radius(X, y, part):
    """Median distance from a letter frame to the nearest same-class frame of another person."""
    from scipy.spatial import cKDTree

    d = []
    for c in np.unique(y):
        Xc, pc = X[y == c], part[y == c]
        tree = cKDTree(Xc)
        # take the nearest few and keep the first one from a different person
        dist, idx = tree.query(Xc, k=min(30, len(Xc)))
        other = pc[idx] != pc[:, None]
        first = np.where(other.any(1), other.argmax(1), -1)
        ok = first >= 0
        d.append(dist[np.arange(len(Xc)), np.where(ok, first, 0)][ok])
    return float(np.median(np.concatenate(d)))


def _pairs(y, part, n, rng, same_person):
    """Index pairs (a, b) of different classes; b from the same person if `same_person`."""
    a = rng.integers(0, len(y), n)
    b = np.empty(n, dtype=int)
    by_person = {p: np.where(part == p)[0] for p in np.unique(part)} if same_person else None
    for i, ai in enumerate(a):
        pool = by_person[part[ai]] if same_person else np.arange(len(y))
        pool = pool[y[pool] != y[ai]]
        b[i] = rng.choice(pool) if len(pool) else ai
    return a, b


def synthesize(kind, X, y, part, n, rng, radius):
    """n synthetic vectors of `kind` from letter frames (X, y, part); filtered by `radius`."""
    from scipy.spatial import cKDTree

    tree = cKDTree(X)
    out, tries = [], 0
    while sum(len(o) for o in out) < n and tries < 20:
        m = 3 * n
        a, b = _pairs(y, part, m, rng, same_person=(kind == "transition"))
        keep = y[a] != y[b]
        a, b = a[keep], b[keep]
        if kind == "transition":
            alpha = rng.uniform(0.3, 0.7, (len(a), 1)).astype(np.float32)
            Z = (1 - alpha) * X[a] + alpha * X[b]
        elif kind == "hybrid":
            Z = X[a].copy().reshape(len(a), 21, 3)
            donor = X[b].reshape(len(b), 21, 3)
            names = list(FINGERS)
            for i in range(len(a)):
                for f in rng.choice(names, size=rng.integers(1, 3), replace=False):
                    Z[i, list(FINGERS[f])] = donor[i, list(FINGERS[f])]
            Z = Z.reshape(len(a), 63)
        else:
            raise ValueError(kind)
        dist, _ = tree.query(Z, k=1)
        out.append(Z[dist > radius])
        tries += 1
    Z = np.vstack(out)[:n]
    return Z.astype(np.float32)


def dev_probs(spec, kind, seed, d, Xn, nfold, none_idx, folds, n_syn):
    """Out-of-fold probabilities over the 55 dev people with an optional synthetic family."""
    X, y, part = d["Xtr"], d["ytr"], d["ptr"].astype(str)
    fold_of = np.array([folds[p] for p in part])
    C = none_idx + 1
    P, Pn = np.zeros((len(X), C), np.float32), np.zeros((len(Xn), C), np.float32)
    for k in range(N_FOLDS):
        tr, ev = fold_of != k, fold_of == k
        ntr, nev = nfold != k, nfold == k
        Xtr = [X[tr], Xn[ntr]]
        ytr = [y[tr], np.full(ntr.sum(), none_idx)]
        gtr = [part[tr], np.array([f"neg{i // 10}" for i in np.where(ntr)[0]])]
        if kind != "none":
            letter = tr & (y != none_idx)
            rng = np.random.default_rng([seed, k, 17])
            r = nn_radius(X[letter], y[letter], part[letter])
            Z = synthesize(kind, X[letter], y[letter], part[letter], n_syn, rng, r)
            Xtr.append(Z)
            ytr.append(np.full(len(Z), none_idx))
            gtr.append(np.array([f"syn{i // 10}" for i in range(len(Z))]))
        S = fit_scores(spec["kind"], np.vstack(Xtr), np.concatenate(ytr), np.concatenate(gtr),
                       np.vstack([X[ev], Xn[nev]]), C, seed)
        S = S if spec["kind"] == "mlp" else softmax(S)
        P[ev], Pn[nev] = S[:ev.sum()], S[ev.sum():]
    return P, Pn, fold_of


def decide(P, thr, none_idx):
    """Predicted class and acceptance as a letter, given per-class thresholds `thr`."""
    pred = P.argmax(1)
    conf = P[np.arange(len(P)), pred]
    return pred, (pred != none_idx) & (conf >= thr[pred])


def fit_thresholds(policy, Pl, yl, none_idx, budget):
    """Thresholds from genuine letter frames: the largest one with frame-rejection <= budget."""
    C = Pl.shape[1]
    if policy == "argmax":
        return np.zeros(C)
    pred = Pl.argmax(1)
    conf = Pl[np.arange(len(Pl)), pred]

    def vec(level):
        if policy == "global":
            t = np.full(C, level)
        else:  # perclass: quantile `level` of the confidence of correct frames of each class
            t = np.zeros(C)
            for c in range(none_idx):
                ok = (pred == c) & (yl == c)
                t[c] = np.quantile(conf[ok], level) if ok.sum() >= 20 else 0.0
        t[none_idx] = 0.0
        return t

    best = vec(0.0)
    for level in GRID:
        t = vec(level)
        fr = 1 - float(decide(Pl, t, none_idx)[1].mean())
        if fr <= budget:
            best = t
    return best


def score(Pl, yl, pl, Pnum, Pn, neg_class, thr, none_idx):
    """Metrics of one threshold vector. FR/err per frame and per toma, FA per negative frame."""
    pred, acc = decide(Pl, thr, none_idx)
    Pt, yt, _ = to_tomas(Pl, yl, pl)
    pt, at = decide(Pt, thr, none_idx)
    shown = np.where(at, pt, none_idx)
    f1, per = macro_f1(shown, yt, none_idx)
    fa = decide(Pn, thr, none_idx)[1]
    lo, hi = wilson(int(fa.sum()), len(fa))
    rest = neg_class == "no_gesture"
    return {
        "f1_macro": f1, "top1": float((shown == yt).mean()),
        "fr_toma": float((~at).mean()), "err_toma": float((at & (pt != yt)).mean()),
        "fr_frame": float((~acc).mean()), "err_frame": float((acc & (pred != yl)).mean()),
        "fa": float(fa.mean()), "fa_ci95": [lo, hi],
        "fa_rest_hand": float(fa[rest].mean()), "fa_other_neg": float(fa[~rest].mean()),
        "numbers_as_letter": float(decide(Pnum, thr, none_idx)[1].mean()),
        "recall_worst3": sorted((p["recall"], i) for i, p in enumerate(per))[:3],
        "n_tomas": int(len(yt)), "n_neg": int(len(fa)),
    }


def analyse(P, Pn, fold_of, nfold, d, neg_class, none_idx, budget, policies):
    """Cross-fitted dev metrics per policy and thresholds fitted on all dev folds (for --final)."""
    y, part = d["ytr"], d["ptr"].astype(str)
    letter = y != none_idx
    out, full = {}, {}
    for pol in policies:
        thr_k = {}
        for k in range(N_FOLDS):
            m = letter & (fold_of != k)
            thr_k[k] = fit_thresholds(pol, P[m], y[m], none_idx, budget)
        # apply each fold's own thresholds to its held-out rows, then pool
        accL, accN, predL, predN = (np.zeros(len(P), bool), np.zeros(len(Pn), bool),
                                    P.argmax(1), Pn.argmax(1))
        for k in range(N_FOLDS):
            r, rn = fold_of == k, nfold == k
            accL[r] = decide(P[r], thr_k[k], none_idx)[1]
            accN[rn] = decide(Pn[rn], thr_k[k], none_idx)[1]
        out[pol] = score_cross(P, y, part, Pn, neg_class, letter, predL, accL, predN, accN,
                               none_idx)
        full[pol] = fit_thresholds(pol, P[letter], y[letter], none_idx, budget)
    return out, full


def score_cross(P, y, part, Pn, neg_class, letter, predL, accL, predN, accN, none_idx):
    """Same metrics as `score`, from precomputed per-fold accept decisions (cross-fitted)."""
    Pl, yl, pl = P[letter], y[letter], part[letter]
    acc, pred = accL[letter], predL[letter]
    Pt, yt, pt_part = to_tomas(Pl, yl, pl)
    # toma decision: accept when the toma-mean prediction is a letter and at least half of its
    # frames were accepted at that frame's fold threshold (the toma's frames share one fold)
    keys = {}
    for i, (p, c) in enumerate(zip(pl, yl)):
        keys.setdefault((str(p), int(c)), []).append(i)
    ks = sorted(keys)
    at = np.array([acc[keys[k]].mean() >= 0.5 for k in ks])
    pt = Pt.argmax(1)
    at &= pt != none_idx
    shown = np.where(at, pt, none_idx)
    f1, per = macro_f1(shown, yt, none_idx)
    lo, hi = wilson(int(accN.sum()), len(accN))
    rest = neg_class == "no_gesture"
    num = ~letter
    return {
        "f1_macro": f1, "top1": float((shown == yt).mean()),
        "fr_toma": float((~at).mean()), "err_toma": float((at & (pt != yt)).mean()),
        "fr_frame": float((~acc).mean()), "err_frame": float((acc & (pred != yl)).mean()),
        "fa": float(accN.mean()), "fa_ci95": [lo, hi],
        "fa_rest_hand": float(accN[rest].mean()), "fa_other_neg": float(accN[~rest].mean()),
        "numbers_as_letter": float(accL[num].mean()),
        "recall_worst3": sorted((p["recall"], i) for i, p in enumerate(per))[:3],
        "n_tomas": int(len(yt)), "n_neg": int(len(accN)),
    }


def curve(P, Pn, y, none_idx, levels=(0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98)):
    """Pooled global-threshold curve: genuine frame rejection against negative false acceptance."""
    letter = y != none_idx
    rows = []
    for t in levels:
        thr = np.full(P.shape[1], t)
        thr[none_idx] = 0.0
        rows.append({"t": t, "fr_frame": float(1 - decide(P[letter], thr, none_idx)[1].mean()),
                     "fa": float(decide(Pn, thr, none_idx)[1].mean())})
    return rows


def commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True, cwd=RAIZ).stdout.strip()


def run(cfg, out, cache_dir):
    import mlflow

    d, labels, Xn, nfold = load(RAIZ / "landmarks_cache.npz", RAIZ / cfg["negatives"])
    none_idx = labels.index("none")
    neg = np.load(RAIZ / cfg["negatives"], allow_pickle=True)
    cls = neg["clase_por_muestra"].astype(str)
    neg_class = cls[~np.isin(cls, list(LETTER_LIKE))]
    folds = frozen_split.load_freeze()["dev_fold_de"]
    mlflow.set_tracking_uri(f"sqlite:///{RAIZ / 'mlflow.db'}")
    mlflow.set_experiment("T-06-no-sena")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for spec in cfg["models"]:
        seeds = cfg["seeds"] if spec["stochastic"] else cfg["seeds"][:1]
        for kind in cfg["variants"]:
            name = f"{spec['name']}:{kind}"
            runs, thr_full, curves = [], [], []
            for seed in seeds:
                f = cache_dir / f"{spec['name']}_{kind}_s{seed}.npz"
                if f.exists():
                    z = np.load(f)
                    P, Pn, fold_of = z["P"], z["Pn"], z["fold_of"]
                else:
                    P, Pn, fold_of = dev_probs(spec, kind, seed, d, Xn, nfold, none_idx, folds,
                                               cfg["n_synthetic"])
                    np.savez_compressed(f, P=P, Pn=Pn, fold_of=fold_of)
                res, full = analyse(P, Pn, fold_of, nfold, d, neg_class, none_idx,
                                    cfg["fr_budget"], cfg["policies"])
                runs.append(res)
                thr_full.append({p: v.tolist() for p, v in full.items()})
                curves.append(curve(P, Pn, d["ytr"], none_idx))
                with mlflow.start_run(run_name=f"{name}-s{seed}"):
                    mlflow.log_params({**spec, "variant": kind, "seed": seed, "commit": commit(),
                                       "split": "dev_folds_55_people", "n_syn": cfg["n_synthetic"]})
                    mlflow.log_metrics({f"{p}_{m}": v for p, r in res.items()
                                        for m, v in r.items() if isinstance(v, float)})
                print(name, seed, {p: (round(r["fr_toma"], 3), round(r["fa"], 3), round(r["f1_macro"], 3))
                                    for p, r in res.items()}, flush=True)
            keys = ["f1_macro", "top1", "fr_toma", "fr_frame", "err_toma", "fa", "fa_rest_hand",
                    "fa_other_neg", "numbers_as_letter"]
            results[name] = {
                "seeds": seeds, "thresholds_all_dev": thr_full, "curve_seed0": curves[0],
                "by_policy": {p: {**{k: [float(np.mean([r[p][k] for r in runs])),
                                         float(np.std([r[p][k] for r in runs]))] for k in keys},
                                   "fa_ci95_seed0": runs[0][p]["fa_ci95"],
                                   "recall_worst3_seed0": runs[0][p]["recall_worst3"]}
                              for p in cfg["policies"]},
            }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dev_results.json").write_text(json.dumps(
        {"labels": labels, "commit": commit(), "fr_budget": cfg["fr_budget"],
         "n_synthetic": cfg["n_synthetic"], "results": results}, indent=2, default=float))
    return results


def final(arg, cfg, out, policy):
    """Train on all 55 dev people (+ synthetic), apply dev thresholds, score the frozen test ONCE."""
    model, kind = arg.split(":")
    spec = next(s for s in cfg["models"] if s["name"] == model)
    d, labels, Xn, nfold = load(RAIZ / "landmarks_cache.npz", RAIZ / cfg["negatives"])
    none_idx = labels.index("none")
    neg = np.load(RAIZ / cfg["negatives"], allow_pickle=True)
    cls = neg["clase_por_muestra"].astype(str)
    neg_class = cls[~np.isin(cls, list(LETTER_LIKE))]
    assert set(d["pte"].astype(str)) == set(frozen_split.load_freeze()["test"])
    dev = json.loads((Path(out) / "dev_results.json").read_text())["results"][arg]
    keep = nfold != N_FOLDS - 1
    X, y, part = d["Xtr"], d["ytr"], d["ptr"].astype(str)
    Xtr, ytr = [X, Xn[keep]], [y, np.full(keep.sum(), none_idx)]
    gtr = [part, np.array([f"neg{i // 10}" for i in range(keep.sum())])]
    Xev = np.vstack([d["Xte"], Xn[~keep]])
    seeds = cfg["seeds"] if spec["stochastic"] else cfg["seeds"][:1]
    runs = []
    for seed in seeds:
        if kind != "none":
            letter = y != none_idx
            rng = np.random.default_rng([seed, 99, 17])
            Z = synthesize(kind, X[letter], y[letter], part[letter], cfg["n_synthetic"], rng,
                           nn_radius(X[letter], y[letter], part[letter]))
            Xs, ys, gs = Xtr + [Z], ytr + [np.full(len(Z), none_idx)], gtr + [
                np.array([f"syn{i // 10}" for i in range(len(Z))])]
        else:
            Xs, ys, gs = Xtr, ytr, gtr
        S = fit_scores(spec["kind"], np.vstack(Xs), np.concatenate(ys), np.concatenate(gs), Xev,
                       none_idx + 1, seed)
        S = S if spec["kind"] == "mlp" else softmax(S)
        P, Pn = S[:len(d["Xte"])], S[len(d["Xte"]):]
        thr = np.array(dev["thresholds_all_dev"][seeds.index(seed)][policy])
        yte, pte = d["yte"], d["pte"].astype(str)
        L = yte != none_idx
        runs.append({pol: score(P[L], yte[L], pte[L], P[~L], Pn, neg_class[~keep], t, none_idx)
                     for pol, t in (("argmax", np.zeros(none_idx + 1)), (policy, thr))})
    keys = ["f1_macro", "top1", "fr_toma", "fr_frame", "err_toma", "fa", "numbers_as_letter"]
    res = {"model": arg, "policy": policy, "commit": commit(), "n_neg_heldout": runs[0][policy]["n_neg"],
           **{pol: {k: [float(np.mean([r[pol][k] for r in runs])),
                        float(np.std([r[pol][k] for r in runs]))] for k in keys}
              for pol in ("argmax", policy)}}
    (Path(out) / f"test_{model}_{kind}.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(RAIZ / "configs" / "t06_no_sena.yaml"))
    ap.add_argument("--out", default=str(RAIZ / "docs" / "reports" / "T-06"))
    ap.add_argument("--cache", default=str(RAIZ / "resultados" / "t06"))
    ap.add_argument("--final", help="model:variant, e.g. logreg:hybrid; uses the test ONCE")
    ap.add_argument("--policy", default="perclass")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.final:
        final(a.final, cfg, a.out, a.policy)
    else:
        run(cfg, a.out, a.cache)


if __name__ == "__main__":
    main()
