"""T-05: baselines compared on the development folds, three seeds, logged in MLflow.

Unit of evaluation is the toma (person + class, frames averaged), as in T-03. Models are
trained on 4 dev folds (people) and scored on the held-out fold; metrics pool the 5 held-out
folds. Rejection = argmax is the "none" class. Negatives (HaGRID, no person id) are split by
sample, so FA is per frame and optimistic. The frozen test is NOT touched here except in
`--final`, which trains the chosen config on all 55 people and uses the test once.

    python -m lsc70.t05_baselines --config configs/t05_baselines.yaml --out docs/reports/T-05
    python -m lsc70.t05_baselines --final mlp_v2 --out docs/reports/T-05
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

from lsc70 import frozen_split
from lsc70.baseline import LETTER_LIKE, RAIZ, macro_f1, to_tomas, wilson
from lsc70.features_v2 import palm_scale

N_FOLDS = frozen_split.N_DEV_FOLDS
FEATURES = {"v1": lambda X: X, "v2": palm_scale}


def load(cache, negatives):
    d = np.load(cache, allow_pickle=True)
    labels = [str(e) for e in d["etiquetas"]]
    neg = np.load(negatives, allow_pickle=True)
    keep = ~np.isin(neg["clase_por_muestra"].astype(str), list(LETTER_LIKE))
    Xn = neg["X_neg"][keep]
    # negatives get a fold by position after a fixed shuffle (no person id exists)
    nfold = np.random.default_rng(frozen_split.DEV_SEED).permutation(len(Xn)) % N_FOLDS
    return d, labels, Xn, nfold


def make_model(kind, n_classes, seed):
    if kind == "mlp":
        from lsc70.entrenar_evaluar import construir_red

        return construir_red(63, n_classes)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    clf = LogisticRegression(max_iter=1000) if kind == "logreg" else SVC(C=10, kernel="rbf")
    return make_pipeline(StandardScaler(), clf)  # scaler is fit on the training fold only


def fit_scores(kind, Xtr, ytr, gtr, Xev, n_classes, seed):
    """Train and return one score row per eval frame (probabilities or margins)."""
    m = make_model(kind, n_classes, seed)
    if kind != "mlp":
        m.fit(Xtr, ytr)
        s = m.decision_function(Xev)
        out = np.full((len(Xev), n_classes), -1e9, dtype=np.float32)
        out[:, m.classes_] = s
        return out
    from sklearn.model_selection import GroupShuffleSplit
    from tensorflow import keras

    from lsc70.entrenar_evaluar import BATCH_SIZE, EPOCHS_MAX, FRAC_VAL, PACIENCIA

    keras.utils.set_random_seed(seed)
    i_tr, i_val = next(GroupShuffleSplit(1, test_size=FRAC_VAL, random_state=seed)
                       .split(Xtr, ytr, groups=gtr))
    assert not set(gtr[i_tr]) & set(gtr[i_val]), "train/val leak"
    m.fit(Xtr[i_tr], ytr[i_tr], validation_data=(Xtr[i_val], ytr[i_val]), epochs=EPOCHS_MAX,
          batch_size=BATCH_SIZE, verbose=0, shuffle=True,
          callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=PACIENCIA,
                                                   restore_best_weights=True)])
    return m.predict(Xev, verbose=0, batch_size=1024)


def metrics(P_letters, y, part, P_neg, P_num, none_idx):
    Pt, yt, pt = to_tomas(P_letters, y, part)
    pred = Pt.argmax(1)
    f1, per = macro_f1(pred, yt, none_idx)
    fa_k = int((P_neg.argmax(1) != none_idx).sum())
    lo, hi = wilson(fa_k, len(P_neg))
    return {
        "f1_macro": f1,
        "top1": float((pred == yt).mean()),
        "fr": float((pred == none_idx).mean()),
        "err": float(((pred != yt) & (pred != none_idx)).mean()),
        "fa": fa_k / len(P_neg), "fa_ci95": [lo, hi],
        "numbers_as_letter": float((P_num.argmax(1) != none_idx).mean()),
        "recall_worst3": sorted((p["recall"], i) for i, p in enumerate(per))[:3],
        "n_tomas": int(len(yt)), "n_neg": int(len(P_neg)),
    }


def split_eval(P, y, part, none_idx):
    """Split frame scores into letters (for tomas) and number frames (label none)."""
    letter = y != none_idx
    return P[letter], y[letter], part[letter], P[~letter]


def dev_scores(spec, seed, d, Xn, nfold, none_idx, folds):
    """Out-of-fold scores over the 55 dev people; returns pooled score arrays."""
    feat = FEATURES[spec["features"]]
    X, y, part = feat(d["Xtr"]), d["ytr"], d["ptr"].astype(str)
    Xneg = feat(Xn)
    fold_of = np.array([folds[p] for p in part])
    n_classes = none_idx + 1
    P, Pn = np.zeros((len(X), n_classes), np.float32), np.zeros((len(Xneg), n_classes), np.float32)
    for k in range(N_FOLDS):
        tr, ev = fold_of != k, fold_of == k
        ntr, nev = nfold != k, nfold == k
        Xtr = np.vstack([X[tr], Xneg[ntr]])
        ytr = np.concatenate([y[tr], np.full(ntr.sum(), none_idx)])
        gtr = np.concatenate([part[tr], np.array([f"neg{i // 10}" for i in np.where(ntr)[0]])])
        S = fit_scores(spec["kind"], Xtr, ytr, gtr, np.vstack([X[ev], Xneg[nev]]), n_classes, seed)
        P[ev], Pn[nev] = S[:ev.sum()], S[ev.sum():]
    return P, Pn, y, part


def commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True, cwd=RAIZ).stdout.strip()


def run(cfg, out):
    import mlflow

    d, labels, Xn, nfold = load(RAIZ / "landmarks_cache.npz", RAIZ / cfg["negatives"])
    none_idx = labels.index("none")
    folds = frozen_split.load_freeze()["dev_fold_de"]
    mlflow.set_tracking_uri(f"sqlite:///{RAIZ / 'mlflow.db'}")
    mlflow.set_experiment("T-05-baselines")
    results, oof = {}, {}
    for spec in cfg["models"]:
        seeds = cfg["seeds"] if spec["stochastic"] else cfg["seeds"][:1]
        runs = []
        for seed in seeds:
            with mlflow.start_run(run_name=f"{spec['name']}-s{seed}"):
                mlflow.log_params({**spec, "seed": seed, "commit": commit(),
                                   "split": "dev_folds_55_people"})
                P, Pn, y, part = dev_scores(spec, seed, d, Xn, nfold, none_idx, folds)
                Pl, yl, pl, Pnum = split_eval(P, y, part, none_idx)
                m = metrics(Pl, yl, pl, Pn, Pnum, none_idx)
                mlflow.log_metrics({k: v for k, v in m.items() if isinstance(v, float)})
                runs.append(m)
                if seed == seeds[0]:
                    Pt, yt, pt = to_tomas(Pl, yl, pl)
                    oof[spec["name"]] = (Pt.argmax(1) == yt, pt)
        keys = ["f1_macro", "top1", "fr", "err", "fa", "numbers_as_letter"]
        results[spec["name"]] = {
            "seeds": seeds,
            **{k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))]
               for k in keys},
            "fa_ci95_seed0": runs[0]["fa_ci95"], "recall_worst3_seed0": runs[0]["recall_worst3"],
            "n_tomas": runs[0]["n_tomas"], "n_neg": runs[0]["n_neg"],
        }
        print(spec["name"], {k: round(results[spec["name"]][k][0], 4) for k in keys}, flush=True)
    results["_paired_top1"] = paired_bootstrap(oof, "mlp_v2", "mlp_v1")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dev_results.json").write_text(json.dumps(
        {"labels": labels, "commit": commit(), "results": results}, indent=2, default=float))
    return results


def paired_bootstrap(oof, a, b, n=2000, seed=0):
    """Top-1 difference a-b on seed-0 tomas, resampling PEOPLE (paired)."""
    if a not in oof or b not in oof:
        return None
    (ca, pa), (cb, _) = oof[a], oof[b]
    people = np.unique(pa)
    idx = {p: np.where(pa == p)[0] for p in people}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        s = np.concatenate([idx[p] for p in rng.choice(people, len(people))])
        diffs.append(ca[s].mean() - cb[s].mean())
    return {"diff": float(ca.mean() - cb.mean()),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}


def final(name, cfg, out):
    """Train on all 55 dev people (negatives of folds 0-3), score the frozen test ONCE."""
    d, labels, Xn, nfold = load(RAIZ / "landmarks_cache.npz", RAIZ / cfg["negatives"])
    none_idx = labels.index("none")
    spec = next(s for s in cfg["models"] if s["name"] == name)
    feat = FEATURES[spec["features"]]
    keep = nfold != N_FOLDS - 1
    Xtr = np.vstack([feat(d["Xtr"]), feat(Xn[keep])])
    ytr = np.concatenate([d["ytr"], np.full(keep.sum(), none_idx)])
    gtr = np.concatenate([d["ptr"].astype(str), np.array([f"neg{i // 10}" for i in range(keep.sum())])])
    Xev = np.vstack([feat(d["Xte"]), feat(Xn[~keep])])
    part_te = d["pte"].astype(str)
    assert set(part_te) == set(frozen_split.load_freeze()["test"]), "cache test != frozen test"
    runs = []
    for seed in (cfg["seeds"] if spec["stochastic"] else cfg["seeds"][:1]):
        S = fit_scores(spec["kind"], Xtr, ytr, gtr, Xev, none_idx + 1, seed)
        P, Pn = S[:len(d["Xte"])], S[len(d["Xte"]):]
        Pl, yl, pl, Pnum = split_eval(P, d["yte"], part_te, none_idx)
        runs.append(metrics(Pl, yl, pl, Pn, Pnum, none_idx))
    keys = ["f1_macro", "top1", "fr", "err", "fa", "numbers_as_letter"]
    res = {"model": name, "commit": commit(), "n_neg_heldout": runs[0]["n_neg"],
           **{k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))]
              for k in keys}}
    (Path(out) / f"test_{name}.json").write_text(json.dumps(res, indent=2, default=float))
    print(res)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(RAIZ / "configs" / "t05_baselines.yaml"))
    ap.add_argument("--out", default=str(RAIZ / "docs" / "reports" / "T-05"))
    ap.add_argument("--final", help="model name: train on all dev people, use the test once")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.final:
        final(a.final, cfg, a.out)
    else:
        run(cfg, a.out)


if __name__ == "__main__":
    main()
