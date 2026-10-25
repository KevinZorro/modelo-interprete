"""T-07: Optuna search over the MLP of T-06 (per-class threshold policy), validation only.

Objective (minimise): dev false acceptance with thresholds cross-fitted exactly as in T-06
(`analyse`, policy "perclass", frame-rejection budget 8 %), plus a penalty when dev F1 macro or
toma-rejection leave their guard rails, so the search cannot buy FA by wrecking the letters.
Search runs on seed 0; the best parameters and the T-06 control are re-scored on fresh seeds, and
`--final` trains on all 55 dev people and uses the frozen test ONCE.

    python -m lsc70.t07_optuna --config configs/t07_optuna.yaml --out docs/reports/T-07
    python -m lsc70.t07_optuna --final --out docs/reports/T-07
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from lsc70 import frozen_split
from lsc70.baseline import LETTER_LIKE, RAIZ
from lsc70.entrenar_evaluar import EPOCHS_MAX, FRAC_VAL, PACIENCIA
from lsc70.t05_baselines import N_FOLDS, load
from lsc70.t06_no_sena import analyse, commit, score

CONTROL = {"units1": 256, "units2": 128, "dropout1": 0.4, "dropout2": 0.3, "lr": 0.001,
           "batch": 16, "l2": 0.0}  # the T-06 / notebook v8 network


def train_scores(p, Xtr, ytr, gtr, Xev, n_classes, seed):
    """Train the MLP with parameters `p` (group-split validation for early stopping); score Xev."""
    from sklearn.model_selection import GroupShuffleSplit
    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    reg = keras.regularizers.l2(p["l2"]) if p["l2"] > 0 else None
    m = keras.Sequential([
        keras.layers.Input(shape=(Xtr.shape[1],)), keras.layers.BatchNormalization(),
        keras.layers.Dense(p["units1"], activation="relu", kernel_regularizer=reg),
        keras.layers.Dropout(p["dropout1"]),
        keras.layers.Dense(p["units2"], activation="relu", kernel_regularizer=reg),
        keras.layers.Dropout(p["dropout2"]),
        keras.layers.Dense(n_classes, activation="softmax")])
    m.compile(optimizer=keras.optimizers.Adam(p["lr"]), loss="sparse_categorical_crossentropy")
    i_tr, i_val = next(GroupShuffleSplit(1, test_size=FRAC_VAL, random_state=seed)
                       .split(Xtr, ytr, groups=gtr))
    assert not set(gtr[i_tr]) & set(gtr[i_val]), "train/val leak"
    m.fit(Xtr[i_tr], ytr[i_tr], validation_data=(Xtr[i_val], ytr[i_val]), epochs=EPOCHS_MAX,
          batch_size=int(p["batch"]), verbose=0, shuffle=True,
          callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=PACIENCIA,
                                                   restore_best_weights=True)])
    return m.predict(Xev, verbose=0, batch_size=1024)


class Data:
    def __init__(self, cfg):
        self.cfg = cfg
        self.d, self.labels, self.Xn, self.nfold = load(RAIZ / "landmarks_cache.npz",
                                                        RAIZ / cfg["negatives"])
        self.none_idx = self.labels.index("none")
        cls = np.load(RAIZ / cfg["negatives"], allow_pickle=True)["clase_por_muestra"].astype(str)
        self.neg_class = cls[~np.isin(cls, list(LETTER_LIKE))]
        self.folds = frozen_split.load_freeze()["dev_fold_de"]

    def dev_eval(self, p, seed):
        """Cross-fitted dev metrics (policy perclass) of parameters `p` on one seed."""
        d, n = self.d, self.none_idx
        X, y, part = d["Xtr"], d["ytr"], d["ptr"].astype(str)
        fold_of = np.array([self.folds[q] for q in part])
        C = n + 1
        P, Pn = np.zeros((len(X), C), np.float32), np.zeros((len(self.Xn), C), np.float32)
        for k in range(N_FOLDS):
            tr, ev = fold_of != k, fold_of == k
            ntr, nev = self.nfold != k, self.nfold == k
            Xt = np.vstack([X[tr], self.Xn[ntr]])
            yt = np.concatenate([y[tr], np.full(ntr.sum(), n)])
            gt = np.concatenate([part[tr], [f"neg{i // 10}" for i in np.where(ntr)[0]]])
            S = train_scores(p, Xt, yt, gt, np.vstack([X[ev], self.Xn[nev]]), C, seed)
            P[ev], Pn[nev] = S[:ev.sum()], S[ev.sum():]
        res, full = analyse(P, Pn, fold_of, self.nfold, d, self.neg_class, n,
                            self.cfg["fr_budget"], ["perclass"])
        return res["perclass"], full["perclass"]


def objective_value(r, cfg):
    pen = 10 * max(0.0, cfg["f1_floor"] - r["f1_macro"])
    pen += 10 * max(0.0, r["fr_toma"] - cfg["fr_toma_ceiling"])
    return r["fa"] + pen


def sample(trial, space):
    return {
        "units1": trial.suggest_int("units1", *space["units1"], step=64),
        "units2": trial.suggest_int("units2", *space["units2"], step=32),
        "dropout1": trial.suggest_float("dropout1", *space["dropout1"]),
        "dropout2": trial.suggest_float("dropout2", *space["dropout2"]),
        "lr": trial.suggest_float("lr", *space["lr"], log=True),
        "batch": trial.suggest_categorical("batch", space["batch"]),
        "l2": trial.suggest_float("l2", *space["l2"], log=True),
    }


KEYS = ["f1_macro", "top1", "fr_toma", "fr_frame", "fa", "fa_rest_hand", "numbers_as_letter"]


def summarise(rs):
    return {k: [float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))] for k in KEYS}


def run(cfg, out):
    import mlflow
    import optuna

    data = Data(cfg)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{RAIZ / 'mlflow.db'}")
    mlflow.set_experiment(cfg["study_name"])
    cm = commit()
    trials = []

    def objective(trial):
        p = CONTROL if trial.number == 0 else sample(trial, cfg["space"])
        r, _ = data.dev_eval(p, cfg["search_seed"])
        v = objective_value(r, cfg)
        with mlflow.start_run(run_name=f"trial-{trial.number}"):
            mlflow.log_params({**p, "seed": cfg["search_seed"], "commit": cm,
                               "split": "dev_folds_55_people"})
            mlflow.log_metrics({**{k: r[k] for k in KEYS}, "objective": v})
        trials.append({"number": trial.number, "params": p, "objective": v,
                       **{k: r[k] for k in KEYS}})
        print(trial.number, round(v, 4), {k: round(r[k], 3) for k in ("fa", "f1_macro", "fr_toma")},
              flush=True)
        return v

    sampler = optuna.samplers.TPESampler(seed=cfg["sampler_seed"])
    study = optuna.create_study(direction="minimize", sampler=sampler)
    # trial 0 is the T-06 control: the search is judged against it on the same seed
    study.enqueue_trial({"units1": 256, "units2": 128, "dropout1": 0.4, "dropout2": 0.3,
                         "lr": 0.001, "batch": 16, "l2": 1e-5})
    study.optimize(objective, n_trials=cfg["n_trials"])
    best_no = study.best_trial.number
    best = trials[best_no]["params"]

    confirm = {}
    for name, p in (("control", CONTROL), ("best", best)):
        rs, thr = [], []
        for s in cfg["confirm_seeds"]:
            r, full = data.dev_eval(p, s)
            rs.append(r)
            thr.append(full.tolist())
            print("confirm", name, s, {k: round(r[k], 3) for k in ("fa", "f1_macro", "fr_toma")},
                  flush=True)
        confirm[name] = {"params": p, "per_seed": rs, "summary": summarise(rs), "thresholds": thr}
    diff = [a["fa"] - b["fa"] for a, b in zip(confirm["best"]["per_seed"],
                                               confirm["control"]["per_seed"])]
    (out / "dev_results.json").write_text(json.dumps({
        "commit": cm, "config": cfg, "trials": trials, "best_trial": best_no,
        "confirm": confirm, "fa_diff_best_minus_control_per_seed": diff,
    }, indent=2, default=float))
    return confirm


def final(cfg, out):
    """Train best and control on all 55 dev people, apply dev thresholds, score the test ONCE."""
    data = Data(cfg)
    dev = json.loads((Path(out) / "dev_results.json").read_text())
    d, n = data.d, data.none_idx
    assert set(d["pte"].astype(str)) == set(frozen_split.load_freeze()["test"])
    keep = data.nfold != N_FOLDS - 1  # same negatives split as T-06 --final
    X, y, part = d["Xtr"], d["ytr"], d["ptr"].astype(str)
    Xt = np.vstack([X, data.Xn[keep]])
    yt = np.concatenate([y, np.full(keep.sum(), n)])
    gt = np.concatenate([part, [f"neg{i // 10}" for i in range(keep.sum())]])
    Xev = np.vstack([d["Xte"], data.Xn[~keep]])
    res = {"commit": commit(), "n_neg_heldout": int((~keep).sum())}
    for name in ("control", "best"):
        p = dev["confirm"][name]["params"]
        runs = []
        for i, s in enumerate(cfg["confirm_seeds"]):
            S = train_scores(p, Xt, yt, gt, Xev, n + 1, s)
            P, Pn = S[:len(d["Xte"])], S[len(d["Xte"]):]
            thr = np.array(dev["confirm"][name]["thresholds"][i])
            yte, pte = d["yte"], d["pte"].astype(str)
            L = yte != n
            runs.append(score(P[L], yte[L], pte[L], P[~L], Pn, data.neg_class[~keep], thr, n))
        res[name] = {"params": p, "summary": summarise(runs)}
    (Path(out) / "test_final.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(RAIZ / "configs" / "t07_optuna.yaml"))
    ap.add_argument("--out", default=str(RAIZ / "docs" / "reports" / "T-07"))
    ap.add_argument("--final", action="store_true", help="uses the frozen test ONCE")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    final(cfg, a.out) if a.final else run(cfg, a.out)


if __name__ == "__main__":
    main()
