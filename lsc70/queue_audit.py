"""Reproducible junk rate of the review queue, with no human in the loop.

    python -m lsc70.queue_audit --out data/scraper --refs landmarks_cache.npz

A queued sample is junk when ANY of these holds (all computed from the landmarks):
  isolated  no other detection of the same video within MAX_GAP_S (a passing frame);
  moving    the raw 2D landmarks moved more than MOVING hand-sizes versus the nearest
            neighbour in time (the hand is travelling, whatever its shape);
  no_fit    cosine distance to every reference pose above FIT_MAX_DIST.
`moving` uses raw positions while the filter uses normalized shape, so it is not the same
test the filter applies. Prints the rate for the unfiltered and the filtered queue.
`page_after` also holds samples that already have a decision (the page always keeps them, so
progress is not lost); `page_after_pending` is only what the filter newly selects, the number
that says whether the filter works.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from lsc70 import mlp_numpy
from lsc70 import review_queue as rq

MOVING = 0.15   # mean landmark displacement, in hand sizes, between neighbouring samples


def _hand_size(lm: np.ndarray) -> float:
    return float(np.linalg.norm(lm[:, :2] - lm[0, :2], axis=1).max()) or 1.0


def junk_flags(rows: list, distances: dict) -> dict:
    """{sample_id: set of reasons} for rows of one video sorted by time."""
    out = {}
    lms = [np.asarray(r["landmarks"], dtype=np.float64) for r in rows]
    for i, r in enumerate(rows):
        reasons = set()
        near = [j for j in (i - 1, i + 1) if 0 <= j < len(rows)
                and abs(rq._t(rows[j]) - rq._t(r)) <= rq.MAX_GAP_S]
        if not near:
            reasons.add("isolated")
        else:
            move = min(float(np.linalg.norm(lms[j][:, :2] - lms[i][:, :2], axis=1).mean()) / _hand_size(lms[i])
                       for j in near)
            if move > MOVING:
                reasons.add("moving")
        if distances.get(rq.sample_id(r), 1.0) > rq.FIT_MAX_DIST:
            reasons.add("no_fit")
        out[rq.sample_id(r)] = reasons
    return out


def audit(out: Path) -> dict:
    queue = {q["sample_id"]: q for q in rq.read_jsonl(out / "review_queue.jsonl")}
    by_video = {}
    for r in rq.read_jsonl(out / "landmarks.jsonl"):
        if rq.sample_id(r) in queue:
            by_video.setdefault(r["video_id"], []).append(r)
    dist = {k: q["distance"] for k, q in queue.items()}
    flags = {}
    for rows in by_video.values():
        rows.sort(key=rq._t)
        flags.update(junk_flags(rows, dist))

    def rate(ids):
        ids = list(ids)
        n = len(ids) or 1
        c = {k: sum(k in flags[i] for i in ids) / n for k in ("isolated", "moving", "no_fit")}
        return {"n": len(ids), "junk": round(sum(bool(flags[i]) for i in ids) / n, 3),
                **{k: round(v, 3) for k, v in c.items()}}
    before = {it["sample_id"] for it in rq.review_items(out, strict=False)} & set(flags)
    items = rq.review_items(out, strict=True)
    after = {it["sample_id"] for it in items} & set(flags)
    pending = {it["sample_id"] for it in items if it["decision"] is None} & set(flags)
    return {"all_samples": rate(flags), "page_before": rate(before), "page_after": rate(after),
            "page_after_pending": rate(pending), "videos": len(by_video)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Junk rate of the review queue")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--refs", type=Path, help="rebuild the queue first")
    a = ap.parse_args(argv)
    if a.refs:
        rq.build(a.out, a.refs, mlp_numpy.load_default())
    print(json.dumps(audit(a.out), indent=2))


if __name__ == "__main__":
    main()
