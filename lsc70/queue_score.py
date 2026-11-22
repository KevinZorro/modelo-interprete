"""Acceptance score of a review candidate: how likely a human says "this is a real sign".

    python -m lsc70.queue_score train --out data/scraper

Why: in the strict queue, how long a pose is held and the margin to the second class carry no
signal about being a sign (AUC ~0.5 against the reviewer's decisions; a hand resting or talking
stays still too). What does: how close the pose is to a reference, hand size in the frame, wrist
height, local speed, and how often the same pose repeats in the video (a lesson shows signs
that recur; a hand that wanders does not). A small logistic regression over those five numbers,
trained on the reviewer's own decisions and checked with cross-validation GROUPED BY VIDEO,
orders the queue and cuts the tail. It only ranks; it never sets a label.
"""
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "queue_scorer.json"
FEATURES = ("dist", "hand_size", "wrist_y", "log_speed", "repeat")
REPEAT_DIST = 0.05      # two samples of a video are "the same pose" under this cosine distance
PRECISION_TARGET = 0.40  # the cut keeps the largest share whose held-out precision is >= this


def _hand_size(lm: np.ndarray) -> float:
    return float(np.linalg.norm(lm[:, :2] - lm[0, :2], axis=1).max()) or 1.0


def pose_features(rows: list, unit_features: np.ndarray, i: int, distance: float, max_gap_s: float) -> list:
    """The FEATURES of sample `i` of one video. `rows` sorted by time, `unit_features` their
    L2-normalised feature vectors (same order). Samples with no time neighbour count as speed 1."""
    lm = np.asarray(rows[i]["landmarks"], dtype=np.float64)

    def t(r):
        return float(r.get("t_s", r["frame"] / 30.0))
    speeds = []
    for j in (i - 1, i + 1):
        if 0 <= j < len(rows) and abs(t(rows[j]) - t(rows[i])) <= max_gap_s:
            o = np.asarray(rows[j]["landmarks"], dtype=np.float64)
            speeds.append(float(np.linalg.norm(lm[:, :2] - o[:, :2], axis=1).mean()) / _hand_size(lm))
    speed = min(speeds) if speeds else 1.0
    repeat = float(np.mean(1.0 - unit_features @ unit_features[i] < REPEAT_DIST))
    return [float(distance), _hand_size(lm), float(lm[0, 1]), float(np.log(speed + 1e-3)), repeat]


def load(path: Path = CONFIG) -> Optional[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def score(cfg: dict, feats: list) -> float:
    z = (np.asarray(feats, dtype=np.float64) - np.asarray(cfg["mean"])) / np.asarray(cfg["std"])
    return float(z @ np.asarray(cfg["weights"]) + cfg["bias"])


def fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> tuple:
    """Class-balanced L2 logistic regression by Newton steps on standardised X.
    Returns (mean, std, weights, bias)."""
    mean, std = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([(X - mean) / std, np.ones((len(X), 1))])
    pos = max(float(y.sum()), 1.0)
    w_s = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * max(len(y) - pos, 1.0)))
    beta = np.zeros(Z.shape[1])
    reg = l2 * np.eye(Z.shape[1])
    reg[-1, -1] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ beta, -30, 30)))
        g = Z.T @ (w_s * (p - y)) + reg @ beta
        H = (Z * (w_s * p * (1 - p))[:, None]).T @ Z + reg + 1e-9 * np.eye(Z.shape[1])
        step = np.linalg.solve(H, g)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return mean, std, beta[:-1], float(beta[-1])


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def cross_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seeds=(0, 1, 2), k: int = 6) -> np.ndarray:
    """Out-of-fold score per sample, one row per seed; folds are made of whole videos."""
    vids = sorted(set(groups.tolist()))
    oof = np.zeros((len(seeds), len(y)))
    for si, seed in enumerate(seeds):
        perm = np.random.RandomState(seed).permutation(vids)
        fold = {v: n % k for n, v in enumerate(perm)}
        f = np.array([fold[g] for g in groups.tolist()])
        for n in range(k):
            tr, te = f != n, f == n
            m, s, w, b = fit(X[tr], y[tr])
            oof[si, te] = ((X[te] - m) / s) @ w + b
    return oof


def precision_at(s: np.ndarray, y: np.ndarray, q: float) -> float:
    n = max(1, int(round(len(s) * q)))
    return float(y[np.argsort(-s)[:n]].mean())


def train(out: Path, config: Path = CONFIG) -> dict:
    """Fit on the reviewer's decisions, report grouped cross-validation, write the config."""
    from lsc70 import review_queue as rq
    items = [i for i in rq.review_items(out, strict=False) if i["decision"] is not None and i.get("media") != "image"]
    y = np.array([int(i["decision"]["decision"] == "accept"
                      or (i["decision"]["decision"] == "relabel" and i["decision"]["label"] != "none"))
                  for i in items])
    X = np.array([i["pose"] for i in items])
    groups = np.array([i["video_id"] for i in items])
    oof = cross_validate(X, y, groups)
    aucs = [_auc(y, o) for o in oof]
    p_at = {f"{int(q * 100)}%": round(float(np.mean([precision_at(o, y, q) for o in oof])), 3)
            for q in (0.1, 0.2, 0.3, 0.5)}
    # cut: the lowest out-of-fold score (mean over seeds) at which the kept share keeps the target precision
    mo = oof.mean(0)
    order = np.argsort(-mo)
    cum = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    ok = np.where((cum >= PRECISION_TARGET) & (np.arange(1, len(y) + 1) >= 10))[0]
    cut = float(mo[order][ok.max()]) if len(ok) else None
    mean, std, w, b = fit(X, y)
    cfg = {"features": list(FEATURES), "mean": mean.tolist(), "std": std.tolist(), "weights": w.tolist(),
           "bias": b, "min_score": cut,
           "meta": {"n": int(len(y)), "positives": int(y.sum()), "videos": int(len(set(groups.tolist()))),
                    "cv": "grouped by video, 6 folds, seeds 0-2", "auc_mean": round(float(np.nanmean(aucs)), 3),
                    "auc_std": round(float(np.nanstd(aucs)), 3), "precision_at_top": p_at,
                    "base_rate": round(float(y.mean()), 3), "precision_target": PRECISION_TARGET,
                    "target": "reviewer says it is a real sign (accept, or relabel to a non-none label)"}}
    config.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return cfg["meta"] | {"min_score": cut, "weights": dict(zip(FEATURES, np.round(w, 3).tolist()))}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Acceptance score of the review queue")
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    print(json.dumps(train(a.out), indent=2))


if __name__ == "__main__":
    main()
