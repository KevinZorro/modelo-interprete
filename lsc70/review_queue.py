"""T-14 — human review queue for scraped landmarks.

Similarity to native references only *orders* the queue and shows a suggestion; a label
exists only after a reviewer who knows LSC writes it. Nothing here sets `label` on its own.

    python -m lsc70.review_queue build  --out data/scraper --refs landmarks_cache.npz
    python -m lsc70.review_queue sheet  --out data/scraper          # CSV for the reviewer
    python -m lsc70.review_queue import --out data/scraper --csv review.csv --reviewer NAME
    python -m lsc70.review_queue export --out data/scraper          # validated.jsonl
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

DECISIONS = ("accept", "relabel", "reject")
SHEET_COLS = ["sample_id", "url_at_time", "query_kind", "suggested", "distance",
              "second", "decision", "label", "notes"]


def sample_id(row: dict) -> str:
    return f"{row['video_id']}:{row['frame']}"


def _unit(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def class_centroids(X: np.ndarray, y: np.ndarray, names) -> tuple:
    """Per-class mean of unit-normalised feature vectors. Returns (centroids, class_names)."""
    Xn = _unit(np.asarray(X, dtype=np.float64))
    ids = np.unique(y)
    cent = _unit(np.stack([Xn[y == i].mean(axis=0) for i in ids]))
    return cent, [str(names[i]) for i in ids]


def rank_samples(features: np.ndarray, cent: np.ndarray, names: list) -> list:
    """For each feature vector: nearest class, cosine distance, runner-up and margin."""
    dist = 1.0 - _unit(np.asarray(features, dtype=np.float64)) @ cent.T
    order = np.argsort(dist, axis=1)
    out = []
    for i in range(len(dist)):
        a, b = order[i, 0], order[i, 1]
        out.append({"suggested": names[a], "distance": round(float(dist[i, a]), 4),
                    "second": names[b], "margin": round(float(dist[i, b] - dist[i, a]), 4)})
    return out


def load_refs(npz_path: Path) -> tuple:
    """Train-split references only: the frozen test split never feeds the queue."""
    d = np.load(npz_path, allow_pickle=True)
    return class_centroids(d["Xtr"], d["ytr"], d["etiquetas"])


def read_jsonl(p: Path) -> list:
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x]


def latest_reviews(out: Path) -> dict:
    """Last decision per sample from the append-only log; an `undo` record cancels it."""
    last = {}
    for r in read_jsonl(out / "reviews.jsonl"):
        last[r["sample_id"]] = r
    return {k: v for k, v in last.items() if v["decision"] in DECISIONS}


KIND_PRIORITY = {"numeros": 0, "reposo": 1, "lsc": 2}


def dedupe_segments(rows: list, threshold: float = 0.06, long_segment: int = 8) -> list:
    """Keep one or two samples per stretch of near-identical frames of one video.

    `rows` are landmark rows of a single video sorted by frame. A segment starts at the
    first row and ends when cosine distance to its first row exceeds `threshold`; the
    middle row is kept, plus the first one when the segment has >= `long_segment` rows.
    """
    if not rows:
        return []
    F = _unit(np.asarray([r["features"] for r in rows], dtype=np.float64))
    keep, start = [], 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or 1.0 - F[i] @ F[start] > threshold:
            seg = list(range(start, i))
            picks = {seg[len(seg) // 2]} | ({seg[0]} if len(seg) >= long_segment else set())
            keep += [rows[j] for j in sorted(picks)]
            start = i
    return keep


def review_items(out: Path) -> list:
    """Queue joined with landmarks, deduplicated, numbers first then rest, with any decision."""
    queue = {q["sample_id"]: q for q in read_jsonl(out / "review_queue.jsonl")}
    by_video = {}
    for r in read_jsonl(out / "landmarks.jsonl"):
        if sample_id(r) in queue:
            by_video.setdefault(r["video_id"], []).append(r)
    decided = latest_reviews(out)
    items = []
    for rows in by_video.values():
        rows.sort(key=lambda r: r["frame"])
        for r in dedupe_segments(rows):
            q = queue[sample_id(r)]
            items.append({**q, "video_id": r["video_id"], "frame": r["frame"],
                          "is_left": r["is_left"], "landmarks": r["landmarks"],
                          "decision": decided.get(q["sample_id"])})
    items.sort(key=lambda x: (KIND_PRIORITY.get(x["query_kind"], 9), x["video_id"], x["frame"]))
    return items


def build(out: Path, refs: Path) -> int:
    """Write review_queue.jsonl: pending, unreviewed samples, closest to a reference first."""
    reviewed = set(latest_reviews(out))
    rows = [r for r in read_jsonl(out / "landmarks.jsonl")
            if r.get("review_status") == "pending" and sample_id(r) not in reviewed]
    queue = []
    if rows:
        cent, names = load_refs(refs)
        ranked = rank_samples(np.asarray([r["features"] for r in rows]), cent, names)
        queue = [{"sample_id": sample_id(r), "url": r["url"], "t_s": r["t_s"],
                  "query_kind": r["query_kind"], **k} for r, k in zip(rows, ranked)]
        queue.sort(key=lambda q: (q["distance"], -q["margin"]))
    with (out / "review_queue.jsonl").open("w", encoding="utf-8") as f:
        for q in queue:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    return len(queue)


def url_at(url: str, t_s: float) -> str:
    return f"{url}{'&' if '?' in url else '?'}t={int(t_s)}s"


def write_sheet(out: Path, csv_path: Optional[Path] = None) -> Path:
    csv_path = csv_path or out / "review_sheet.csv"
    done = latest_reviews(out)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLS)
        w.writeheader()
        for q in read_jsonl(out / "review_queue.jsonl"):
            w.writerow({"sample_id": q["sample_id"], "url_at_time": url_at(q["url"], q["t_s"]),
                        "query_kind": q["query_kind"], "suggested": q["suggested"],
                        "distance": q["distance"], "second": q["second"],
                        "decision": done.get(q["sample_id"], {}).get("decision", ""),
                        "label": done.get(q["sample_id"], {}).get("label") or "",
                        "notes": done.get(q["sample_id"], {}).get("notes", "")})
    return csv_path


def parse_decision(row: dict) -> Optional[dict]:
    """A filled sheet row -> decision dict, or None when the reviewer left it blank."""
    dec = (row.get("decision") or "").strip().lower()
    if not dec:
        return None
    if dec not in DECISIONS:
        raise ValueError(f"{row['sample_id']}: decision must be one of {DECISIONS}, got {dec!r}")
    label = (row.get("label") or "").strip()
    if dec == "accept":
        label = label or row["suggested"]
    if dec == "relabel" and not label:
        raise ValueError(f"{row['sample_id']}: relabel needs a label")
    return {"sample_id": row["sample_id"], "decision": dec,
            "label": None if dec == "reject" else label, "notes": (row.get("notes") or "").strip()}


def record_review(out: Path, sample: str, decision: str, label: Optional[str], reviewer: str,
                  notes: str = "") -> dict:
    """Append one decision (or `undo`) from the review page to reviews.jsonl."""
    if not reviewer.strip():
        raise ValueError("reviewer name is required")
    if decision != "undo":
        d = parse_decision({"sample_id": sample, "decision": decision, "label": label or "",
                            "suggested": "", "notes": notes})
        if d["decision"] == "accept" and not d["label"]:
            raise ValueError("accept needs the suggested label")
    else:
        d = {"sample_id": sample, "decision": "undo", "label": None, "notes": ""}
    rec = {**d, "reviewer": reviewer.strip(),
           "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with (out / "reviews.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    write_sheet(out)
    return rec


def import_sheet(out: Path, csv_path: Path, reviewer: str) -> int:
    """Append reviewer decisions to reviews.jsonl (append-only audit log)."""
    if not reviewer.strip():
        raise ValueError("reviewer name is required")
    seen = {r["sample_id"] for r in read_jsonl(out / "reviews.jsonl")}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            d = parse_decision(row)
            if d and d["sample_id"] not in seen:
                new.append({**d, "reviewer": reviewer.strip(), "reviewed_at": now})
    with (out / "reviews.jsonl").open("a", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(new)


def export_validated(out: Path) -> int:
    """validated.jsonl: samples with a human label; rejected ones are left out."""
    reviews = latest_reviews(out)
    n = 0
    with (out / "validated.jsonl").open("w", encoding="utf-8") as f:
        for r in read_jsonl(out / "landmarks.jsonl"):
            rev = reviews.get(sample_id(r))
            if not rev or rev["decision"] == "reject":
                continue
            f.write(json.dumps({**r, "label": rev["label"], "review_status": "validated",
                                "reviewer": rev["reviewer"], "reviewed_at": rev["reviewed_at"]},
                               ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description="Human review queue for scraped landmarks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "sheet", "import", "export"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        if name == "build":
            p.add_argument("--refs", type=Path, required=True, help="npz with Xtr, ytr, etiquetas")
        if name == "import":
            p.add_argument("--csv", type=Path, required=True)
            p.add_argument("--reviewer", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        print({"queued": build(a.out, a.refs)})
    elif a.cmd == "sheet":
        print(write_sheet(a.out))
    elif a.cmd == "import":
        print({"imported": import_sheet(a.out, a.csv, a.reviewer)})
    else:
        print({"validated": export_validated(a.out)})


if __name__ == "__main__":
    main()
