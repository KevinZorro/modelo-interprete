"""How often would the reviewer only press "accept"? Measured on the reviewer's own decisions.

    python -m lsc70.review_queue build --out data/scraper --refs landmarks_cache.npz
    python -m lsc70.queue_eval --out data/scraper

Ground truth is reviews.jsonl: an accepted sample's label is the one shown, a relabelled
sample's label is the reviewer's, a rejected or "none" sample has no sign. For each way of
choosing what to show (centroid distance, one-frame classifier, spotted sign, ...) it reports,
on the decided samples it would show: precision (= share the reviewer would only accept, the
goal), coverage, and a 95 % Wilson interval. Thresholds are chosen by cross-validation over
videos (never on the samples they are scored on), so the precision is an honest estimate.
It also counts how many different signs one loop of the player shows, before and after.
Nothing here writes labels.
"""
import argparse
import json
import zlib
from pathlib import Path

import numpy as np

from lsc70 import alias_poses
from lsc70 import review_queue as rq
from lsc70 import sign_spotter as ss

REJECT = "__reject__"
TARGET = 0.85          # precision the threshold search aims for on the training folds
N_FOLDS = 5
MIN_SHOWN = 20         # a threshold must show at least this many training samples
GRID = [round(x, 2) for x in np.arange(0.30, 0.995, 0.05)]
DIST_GRID = [0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.17, 0.25, 0.40]


def norm(label):
    """Pose-level class with Ñ spelled as the model spells it (NN)."""
    if label is None:
        return None
    lab = str(label).strip()
    lab = "NN" if lab.upper() in ("Ñ", "NN") else (lab.upper() if len(lab) == 1 else lab)
    return alias_poses.canonical(lab)


def truth_of(rev: dict) -> str:
    if rev["decision"] == "reject":
        return REJECT
    return norm(rev["label"])


def is_correct(suggestion, truth) -> bool:
    return truth not in (REJECT, "none", None) and norm(suggestion) == truth


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def fold_of(video_id: str) -> int:
    return zlib.crc32(video_id.encode("utf-8")) % N_FOLDS


# ----- methods: each maps a decided sample's queue row to (suggestion, score) -----
# score is None when the method would never show the sample; higher score = more confident.

def m_centroid(q):
    return q["suggested"], -q["distance"]


def m_frame(q):
    return q.get("mlp"), (q.get("mlp_p") if q.get("mlp") not in (None, "none") else None)


def m_sign(q):
    if q.get("seg_label") is None:
        return None, None
    if q.get("media") != "image" and q.get("seg_held", 0) < rq.REVIEW_HOLD_S:
        return q["seg_label"], None
    return q["seg_label"], q["seg_p"]


def m_sign_agree(q):
    lab, sc = m_sign(q)
    return (lab, sc) if lab is not None and norm(lab) == norm(q["suggested"]) else (lab, None)


def m_sign_static(q, dynamic=frozenset(rq.load_dynamic_signs())):
    lab, sc = m_sign(q)
    return (lab, sc) if lab is not None and lab not in dynamic else (lab, None)


METHODS = {"centroid": m_centroid, "frame": m_frame, "sign": m_sign,
           "sign_agree_centroid": m_sign_agree, "sign_static_only": m_sign_static}


def score_table(samples: list, method) -> list:
    """[(video, correct, score, is_reject_or_none)] for every decided sample."""
    out = []
    for q, truth in samples:
        lab, sc = method(q)
        out.append((q["sample_id"].rsplit(":", 1)[0], is_correct(lab, truth), sc,
                    truth in (REJECT, "none")))
    return out


def at_threshold(table, thr) -> dict:
    shown = [t for t in table if t[2] is not None and t[2] >= thr]
    k = sum(t[1] for t in shown)
    return {"thr": thr, "shown": len(shown), "correct": k,
            "precision": round(k / len(shown), 3) if shown else None, "ci95": wilson(k, len(shown)),
            "no_sign_share": round(sum(t[3] for t in shown) / len(shown), 3) if shown else None,
            "coverage": round(len(shown) / len(table), 3) if table else None}


def pick_threshold(table, grid, target=TARGET):
    """Lowest threshold (most items shown) reaching `target` precision with >= MIN_SHOWN items."""
    for thr in sorted(grid):
        r = at_threshold(table, thr)
        if r["shown"] >= MIN_SHOWN and r["precision"] is not None and r["precision"] >= target:
            return thr
    return None


def cross_validated(table, grid, target=TARGET) -> dict:
    """Pick the threshold on four folds of videos, score the fifth; sum over the five."""
    k = n = 0
    thrs = []
    for f in range(N_FOLDS):
        train = [t for t in table if fold_of(t[0]) != f]
        test = [t for t in table if fold_of(t[0]) == f]
        thr = pick_threshold(train, grid, target)
        thrs.append(thr)
        if thr is None:
            continue
        r = at_threshold(test, thr)
        k += r["correct"]
        n += r["shown"]
    return {"target": target, "thresholds_per_fold": thrs, "shown": n, "correct": k,
            "precision": round(k / n, 3) if n else None, "ci95": wilson(k, n),
            "coverage": round(n / len(table), 3) if table else None}


def _right(samples, method) -> dict:
    """How often the method's suggestion (any confidence) is the reviewer's label."""
    k = sum(is_correct(method(q)[0], t) for q, t in samples)
    return {"n": len(samples), "right": k, "share": round(k / len(samples), 3) if samples else None,
            "ci95": wilson(k, len(samples))}


def per_class(samples, method, thr, min_n=8) -> dict:
    by = {}
    for q, truth in samples:
        lab, sc = method(q)
        if sc is None or sc < thr:
            continue
        by.setdefault(lab, []).append(is_correct(lab, truth))
    return {lab: {"n": len(v), "precision": round(sum(v) / len(v), 3)}
            for lab, v in sorted(by.items(), key=lambda x: -len(x[1])) if len(v) >= min_n}


def loop_counts(out: Path, items: list) -> dict:
    """Different confident signs one loop shows: fixed 3.5 s window vs the spotted window."""
    queue = {q["sample_id"]: q for q in rq.read_jsonl(out / "review_queue.jsonl")}
    labels = sorted({q.get("mlp") for q in queue.values() if q.get("mlp")})
    idx = {lab: i for i, lab in enumerate(labels)}
    by_video = {}
    for q in queue.values():
        if q.get("mlp") is not None and q.get("media") != "image":
            by_video.setdefault(q["sample_id"].rsplit(":", 1)[0], []).append(q)
    series = {}
    for vid, qs in by_video.items():
        qs.sort(key=lambda q: q["t_s"])
        P = np.zeros((len(qs), len(labels)))
        for i, q in enumerate(qs):
            P[i, idx[q["mlp"]]] = q["mlp_p"]
        series[vid] = ([q["t_s"] for q in qs], [idx[q["mlp"]] for q in qs], P)
    old, new = [], []
    for it in items:
        if it.get("media") == "image" or it.get("loop_start") is None or it["video_id"] not in series:
            continue
        times, lab, P = series[it["video_id"]]
        old.append(ss.signs_in_window(times, lab, P, it["t_s"] - rq.TRACK_PRE_S, it["t_s"] + rq.TRACK_POST_S))
        new.append(ss.signs_in_window(times, lab, P, it["loop_start"], it["loop_end"]))

    def dist(v):
        v = np.asarray(v)
        return {"n": int(len(v)), "mean": round(float(v.mean()), 2) if len(v) else None,
                "share_no_confident_sign": round(float((v == 0).mean()), 3) if len(v) else None,
                "share_one_sign": round(float((v == 1).mean()), 3) if len(v) else None,
                "share_3_or_more": round(float((v >= 3).mean()), 3) if len(v) else None}
    return {"fixed_3.5s_window": dist(old), "spotted_window": dist(new)}


# ----- a learned check: does combining signals beat the plain probability? -----

def _features(q, dynamic):
    p, p2 = q.get("seg_p", 0.0), q.get("seg_p_second", 0.0)
    return [p, q.get("seg_p_min", 0.0), p - p2, np.log1p(q.get("seg_n", 0)),
            float(norm(q.get("seg_label")) == norm(q["suggested"])), q["distance"],
            float(q.get("seg_label") in dynamic)]


def _logreg(X, y, l2=1.0, iters=200):
    X = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(X.shape[1])
    for _ in range(iters):  # Newton steps on the L2-penalised log-likelihood
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (p - y) + l2 * np.r_[w[:-1], 0]
        H = (X.T * (p * (1 - p))) @ X + l2 * np.diag(np.r_[np.ones(len(w) - 1), 0])
        dw = np.linalg.solve(H, g)
        w -= dw
        if np.abs(dw).max() < 1e-8:
            break
    return w


def learned_check(samples, target=TARGET) -> dict:
    dynamic = rq.load_dynamic_signs()
    rows = [(q, t) for q, t in samples if m_sign(q)[1] is not None]
    if len(rows) < 60:
        return {"skipped": f"only {len(rows)} decided samples in spotted signs"}
    X = np.asarray([_features(q, dynamic) for q, _ in rows], dtype=np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    X = (X - mu) / sd
    y = np.asarray([is_correct(q["seg_label"], t) for q, t in rows], dtype=np.float64)
    vids = [q["sample_id"].rsplit(":", 1)[0] for q, _ in rows]
    table = []
    for f in range(N_FOLDS):
        tr = np.array([fold_of(v) != f for v in vids])
        if tr.all() or not tr.any():
            continue
        w = _logreg(X[tr], y[tr])
        s = 1 / (1 + np.exp(-(np.hstack([X, np.ones((len(X), 1))]) @ w)))
        table += [(vids[i], bool(y[i]), float(s[i]), False) for i in range(len(rows)) if not tr[i]]
    # the learned scores are already out-of-fold; pick a cut-off by inner CV like the others
    return {"note": "logistic regression on 7 signals, scores out-of-fold by video",
            "cv": cross_validated(table, GRID, target)}


def evaluate(out: Path, target: float = TARGET) -> dict:
    queue = {q["sample_id"]: q for q in rq.read_jsonl(out / "review_queue.jsonl")}
    if not any("seg_label" in q for q in queue.values()):
        raise SystemExit("build the queue with the classifier first: python -m lsc70.review_queue build ...")
    reviews = rq.latest_reviews(out)
    samples = [(queue[s], truth_of(r)) for s, r in reviews.items() if s in queue]
    truths = [t for _, t in samples]
    report = {"decided": {"n": len(samples),
                          "accept_or_relabel_to_a_sign": sum(t not in (REJECT, "none") for t in truths),
                          "none": truths.count("none"), "reject": truths.count(REJECT),
                          "videos": len({q["sample_id"].rsplit(":", 1)[0] for q, _ in samples})},
              "methods": {}}
    real = [(q, t) for q, t in samples if t not in (REJECT, "none")]
    letters = [(q, t) for q, t in real if not str(t).isdigit() and t not in ("MIL", "MILLON")]
    report["decided"]["real_signs"] = len(real)
    report["decided"]["real_letters"] = len(letters)
    for name, m in METHODS.items():
        table = score_table(samples, m)
        grid = [-d for d in DIST_GRID] if name == "centroid" else GRID
        report["methods"][name] = {
            "suggestion_right_on_real_signs": _right(real, m),
            "suggestion_right_on_real_letters": _right(letters, m),
            "all_decided_precision": at_threshold(table, -1e9)["precision"],
            "curve": [at_threshold(table, t) for t in sorted(grid)],
            "cv": cross_validated(table, grid, target)}
    report["t08_rule_one_frame"] = at_threshold(
        [(q["sample_id"], is_correct(q.get("mlp"), t),
          1.0 if q.get("mlp") not in (None, "none") and q["mlp_p"] >= _t08(q["mlp"]) else None,
          t in (REJECT, "none")) for q, t in samples], 0.5)
    thr = rq.READY_P
    picked = [x for x in report["methods"]["sign"]["cv"]["thresholds_per_fold"] if x is not None]
    report["per_class_sign"] = {"thr": float(np.median(picked)) if picked else thr}
    report["per_class_sign"]["classes"] = per_class(samples, m_sign, report["per_class_sign"]["thr"])
    report["learned"] = learned_check(samples, target)
    vol = {}
    for p in (0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        rq.READY_P = p
        items = [i for i in rq.review_items(out, strict=True) if i["decision"] is None]
        vol[str(p)] = {"pending_shown": len(items), "videos": len({i["video_id"] for i in items})}
    rq.READY_P = thr
    report["pending_volume_by_ready_p"] = vol
    report["loops"] = loop_counts(out, [i for i in rq.review_items(out, strict=True) if i["decision"] is None])
    return report


_T08 = None


def _t08(label):
    global _T08
    if _T08 is None:
        from lsc70 import mlp_numpy
        m = mlp_numpy.NumpyMLP()
        _T08 = dict(zip(m.labels, m.thresholds))
    return _T08.get(label, 1.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Share of shown items the reviewer would only accept")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", type=float, default=TARGET)
    ap.add_argument("--json", type=Path, help="also write the report here")
    a = ap.parse_args(argv)
    rep = evaluate(a.out, a.target)
    txt = json.dumps(rep, indent=1, ensure_ascii=False)
    if a.json:
        a.json.write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
