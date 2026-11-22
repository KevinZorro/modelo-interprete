"""Which letter should the review page suggest? Measured on the reviewer's own decisions.

Every sample the reviewer accepted, or relabelled to a sign, is a ground truth (none and
reject are left out: the question here is the letter, not whether it is a sign).

  centroid   nearest ANH class centroid: what the page suggests today
  domain     centroid plus the reviewer's own labelled signs from OTHER videos as references
             (leave one video out), so digits, which ANH lacks, can be named
  +kind      the title says alphabet or numbers: only those classes are named
  order      a lesson shows the alphabet (or 1..10) in order: a Viterbi pass over the video's
             held poses sorted by time; staying is free, each letter skipped forward costs
             `step` and going back costs `back`. (step, back) is picked on four folds of
             videos and scored on the fifth; (0, 0) means no order, so it can be picked.

Top-1 is what "only accept" needs; top-3 is what three one-key buttons on the page would
cover. Nothing here labels data.

    python -m lsc70.label_eval --out data/scraper --refs landmarks_cache.npz --json data/scraper/label_eval.json
"""
import argparse
import json
import re
import zlib
from pathlib import Path

import numpy as np

from lsc70 import alias_poses
from lsc70 import review_queue as rq

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "NN",
           "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
DIGITS = ["F", "V", "W", "4", "5", "6", "7", "8", "9", "10"]  # 1, 2, 3 share the pose of F, V, W
NO_REF = 0.35   # order pass: cost of a class with no reference (worse than any real fit)
TAU = 0.05      # cosine distance per unit of cost: about one top-1/top-2 margin
GRID = [(0.0, 0.0)] + [(s, b) for s in (0.1, 0.5) for b in (1.0, 3.0, 10.0)]
N_FOLDS = 5
NUMBERS_TITLE = re.compile(r"n[uú]mer|numbers?\b|\bcontar\b|counting|\b1 al 10\b", re.I)
LETTERS_TITLE = re.compile(r"abecedario|alfabeto|alphabet|spelling|deletre|dactilolog|\bletras?\b", re.I)


def norm(label):
    """Pose-level class (F/1, V/2, W/3 are one pose), Ñ spelled NN as the model spells it."""
    if label is None:
        return None
    lab = str(label).strip()
    if lab.lower() == "none":
        return "none"
    lab = "NN" if lab.upper() in ("Ñ", "NN") else lab.upper()
    return alias_poses.canonical(lab)


def video_kind(title, query_kind=None) -> str:
    """letters, numbers or mixed, from the title; the search that found it when the title says neither."""
    t = title or ""
    num, let = bool(NUMBERS_TITLE.search(t)), bool(LETTERS_TITLE.search(t))
    if num and let:
        return "mixed"
    if num or let:
        return "numbers" if num else "letters"
    return {"numeros": "numbers", "lsc": "letters"}.get(query_kind, "mixed")


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


def fold_of(video_id: str) -> int:
    return zlib.crc32(video_id.encode("utf-8")) % N_FOLDS


def viterbi_order(D: np.ndarray, step: float, back: float, tau: float = TAU) -> list:
    """Column index per row of the cost matrix D (rows in time, columns in teaching order)."""
    T, C = D.shape
    jump = np.arange(C)[None, :] - np.arange(C)[:, None]          # jump[i, j] = j - i
    trans = np.where(jump >= 0, step * jump, back)
    cost, ptr = D[0] / tau, np.zeros((T, C), dtype=int)
    for t in range(1, T):
        tot = cost[:, None] + trans
        ptr[t] = tot.argmin(axis=0)
        cost = tot.min(axis=0) + D[t] / tau
    path = [int(cost.argmin())]
    for t in range(T - 1, 0, -1):
        path.append(int(ptr[t, path[-1]]))
    return path[::-1]


def order_moves(labels: list, order: list) -> dict:
    """How consecutive labels (time order) move in the teaching order."""
    pos = [order.index(x) for x in labels if x in order]
    moves = [b - a for a, b in zip(pos, pos[1:])]
    return {"forward": sum(m > 0 for m in moves), "same": sum(m == 0 for m in moves),
            "back": sum(m < 0 for m in moves)}


def _spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _classes(truths) -> list:
    extra = sorted({t for t in truths if t not in LETTERS and t not in DIGITS})
    return LETTERS + [d for d in DIGITS if d not in LETTERS] + extra


def _allowed(kind: str, classes: list) -> np.ndarray:
    ok = (set(LETTERS) if kind == "letters"
          else set(DIGITS) | {c for c in classes if c not in LETTERS} if kind == "numbers"
          else set(classes))
    return np.array([c in ok for c in classes])


def _ranked(row: np.ndarray, classes: list, mask=None, k: int = 4) -> list:
    d = row if mask is None else np.where(mask, row, np.inf)
    return [classes[j] for j in np.argsort(d, kind="stable")[:k]]


def _holds(rows: list) -> list:
    return [r for r in rq.dedupe_segments(rows) if r["held_s"] >= rq.REVIEW_HOLD_S]


def collect(out: Path, refs: Path) -> tuple:
    """Per ground-truth sample: its truth, video kind and each suggester's ranked labels."""
    by_video = {}
    for r in rq.read_jsonl(out / "landmarks.jsonl"):
        by_video.setdefault(r["video_id"], []).append(r)
    truth = {}
    for sid, rev in rq.latest_reviews(out).items():
        lab = norm(rev.get("label")) if rev["decision"] != "reject" else None
        if lab and lab != "none":
            truth[sid] = lab
    titles = rq.load_titles(out)
    cent, names = rq.load_refs(refs)
    col = {norm(n): i for i, n in enumerate(names)}
    classes = _classes(truth.values())
    rows_of = {rq.sample_id(r): r for rows in by_video.values() for r in rows}
    ex = [s for s in truth if s in rows_of]
    E = rq._unit(np.asarray([rows_of[s]["features"] for s in ex], dtype=np.float64).reshape(-1, 63))
    ex_lab, ex_vid = [truth[s] for s in ex], [rows_of[s]["video_id"] for s in ex]
    recs, unlabelled_rho = [], []
    for vid, rows in by_video.items():
        rows.sort(key=lambda r: r["frame"])
        image = rows[0].get("media") == "image"
        kind = "mixed" if image else video_kind(titles.get(vid) or rows[0].get("title"), rows[0].get("query_kind"))
        seq = {} if image else {rq.sample_id(r): r for r in _holds(rows)}
        seq.update({rq.sample_id(r): r for r in rows if rq.sample_id(r) in truth})
        seq = sorted(seq.values(), key=lambda r: r["frame"])
        if not seq:
            continue
        F = rq._unit(np.asarray([r["features"] for r in seq], dtype=np.float64))
        d = 1.0 - F @ cent.T
        Dc = np.full((len(seq), len(classes)), np.inf)   # no reference: never ranked first
        for j, c in enumerate(classes):
            if c in col:
                Dc[:, j] = d[:, col[c]]
        De = np.full_like(Dc, np.inf)
        keep = [i for i, v in enumerate(ex_vid) if v != vid]
        if keep:
            de = 1.0 - F @ E[keep].T
            for j, c in enumerate(classes):
                m = [k for k, i in enumerate(keep) if ex_lab[i] == c]
                if m:
                    De[:, j] = de[:, m].min(axis=1)
        Dd = np.minimum(Dc, De)
        mask = _allowed(kind, classes)
        if kind == "letters":   # label-free order check: confident centroid letters against time
            li = [LETTERS.index(classes[j]) for j in Dc[:, :len(LETTERS)].argmin(axis=1)]
            ok = [i for i in range(len(seq)) if Dc[i, :len(LETTERS)].min() <= 0.10]
            if len(ok) >= 6:
                unlabelled_rho.append(_spearman(ok, [li[i] for i in ok]))
        order = {"letters": LETTERS, "numbers": DIGITS}.get(kind)
        paths = {}
        for name, D in (("order", Dc), ("order+domain", Dd)):
            for g in GRID:
                if order is None:
                    continue
                cols = [classes.index(c) for c in order]
                C = np.where(np.isfinite(D[:, cols]), D[:, cols], NO_REF)
                paths[(name, g)] = [order[p] for p in viterbi_order(C, *g)]
        for i, r in enumerate(seq):
            sid = rq.sample_id(r)
            if sid not in truth:
                continue
            pred = {"centroid": _ranked(Dc[i], classes), "centroid+kind": _ranked(Dc[i], classes, mask),
                    "domain": _ranked(Dd[i], classes), "domain+kind": _ranked(Dd[i], classes, mask)}
            grid = {}
            for name, base in (("order", "centroid+kind"), ("order+domain", "domain+kind")):
                for g in GRID:
                    lab = paths[(name, g)][i] if (name, g) in paths else pred[base][0]
                    grid[(name, g)] = [lab] + [c for c in pred[base] if c != lab][:2]
            recs.append({"sid": sid, "vid": vid, "truth": truth[sid], "kind": kind, "frame": r["frame"],
                         "pred": pred, "grid": grid})
    return recs, unlabelled_rho


def _score(recs: list, get) -> dict:
    def part(rs):
        n = len(rs)
        k1 = sum(get(r)[0] == r["truth"] for r in rs)
        k3 = sum(r["truth"] in get(r)[:3] for r in rs)
        return {"n": n, "top1": round(k1 / n, 3) if n else None, "top1_ci95": wilson(k1, n),
                "top3": round(k3 / n, 3) if n else None, "top3_ci95": wilson(k3, n)}
    return {"all": part(recs), "letters": part([r for r in recs if r["truth"] in LETTERS]),
            "other": part([r for r in recs if r["truth"] not in LETTERS])}


def _cv_order(recs: list, name: str) -> dict:
    """Pick (step, back) on four folds of videos by top-1, score the fifth."""
    chosen, picked = {}, []
    for f in range(N_FOLDS):
        train = [r for r in recs if fold_of(r["vid"]) != f]
        best = max(GRID, key=lambda g: sum(r["grid"][(name, g)][0] == r["truth"] for r in train))
        picked.append(list(best))
        chosen.update({r["sid"]: best for r in recs if fold_of(r["vid"]) == f})
    rep = _score(recs, lambda r: r["grid"][(name, chosen[r["sid"]])])
    rep["params_per_fold"] = picked
    rep["in_sample_top1"] = {f"{g[0]}/{g[1]}": round(np.mean([r["grid"][(name, g)][0] == r["truth"]
                                                             for r in recs]), 3) for g in GRID}
    return rep


def evaluate(out: Path, refs: Path) -> dict:
    recs, rho = collect(out, refs)
    by_label, by_kind = {}, {}
    for r in recs:
        by_label[r["truth"]] = by_label.get(r["truth"], 0) + 1
        k = by_kind.setdefault(r["kind"], {"letter": 0, "other": 0})
        k["letter" if r["truth"] in LETTERS else "other"] += 1
    moves = {"forward": 0, "same": 0, "back": 0}
    for vid in {r["vid"] for r in recs}:
        rs = sorted((r for r in recs if r["vid"] == vid), key=lambda r: r["frame"])
        order = {"letters": LETTERS, "numbers": DIGITS}.get(rs[0]["kind"])
        if order:
            for k, v in order_moves([r["truth"] for r in rs], order).items():
                moves[k] += v
    rep = {"truth": {"n": len(recs), "videos": len({r["vid"] for r in recs}),
                     "by_label": dict(sorted(by_label.items(), key=lambda x: -x[1])), "by_kind": by_kind},
           "order_check": {"labelled_consecutive_pairs": moves,
                           "unlabelled_letter_videos": {"n": len(rho),
                                                        "median_rho": round(float(np.median(rho)), 3) if rho else None,
                                                        "share_rho_ge_0.5": round(float(np.mean(np.asarray(rho) >= 0.5)), 3) if rho else None}},
           "methods": {m: _score(recs, lambda r, m=m: r["pred"][m])
                       for m in ("centroid", "centroid+kind", "domain", "domain+kind")}}
    for name in ("order", "order+domain"):
        rep["methods"][name] = _cv_order(recs, name)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description="Which suggested letter the reviewer would only accept")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--refs", type=Path, required=True)
    ap.add_argument("--json", type=Path, help="also write the report here")
    a = ap.parse_args(argv)
    txt = json.dumps(evaluate(a.out, a.refs), indent=1, ensure_ascii=False)
    if a.json:
        a.json.write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
