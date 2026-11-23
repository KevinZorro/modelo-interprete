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
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from lsc70 import alias_poses, queue_score

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


def load_refs(npz_path: Path, numbers: bool = False) -> tuple:
    """Train-split references only: the frozen test split never feeds the queue.

    The cache's `none` class is the ANH numbers (1, 4-10, MIL, MILLON). With `numbers` each
    becomes its own class, named from the sample's file (Per01__10__Per01_10_0.jpg is a 10),
    and 1 joins F because they are the same pose."""
    d = np.load(npz_path, allow_pickle=True)
    if not numbers or "ntr" not in d:
        return class_centroids(d["Xtr"], d["ytr"], d["etiquetas"])
    names = [str(e) for e in d["etiquetas"]]
    y = []
    for i, f in zip(d["ytr"], d["ntr"]):
        parts = str(f).split("__")
        y.append(alias_poses.canonical(parts[1]) if names[i] == "none" and len(parts) >= 3 else names[i])
    labels = sorted(set(y))
    return class_centroids(d["Xtr"], np.array([labels.index(v) for v in y]), labels)


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


HOLD_S = 1.0        # a sign being taught is shown and held at least this long (Kevin, 2026-10-25)
MAX_GAP_S = 0.7     # a longer hole in the detections (hand lost) ends a segment


def _t(r: dict) -> float:
    return float(r.get("t_s", r["frame"] / 30.0))


STILL_MOVE = 0.10   # a held sign: raw landmarks drift less than this many hand sizes in a segment


def _drift(a: dict, b: dict) -> float:
    """Mean raw 2D landmark displacement between two rows, in hand sizes (0 if no landmarks)."""
    if "landmarks" not in a or "landmarks" not in b:
        return 0.0
    A, B = np.asarray(a["landmarks"])[:, :2], np.asarray(b["landmarks"])[:, :2]
    size = float(np.linalg.norm(A - A[0], axis=1).max()) or 1.0
    return float(np.linalg.norm(A - B, axis=1).mean()) / size


def dedupe_segments(rows: list, threshold: float = 0.06, long_segment: int = 8) -> list:
    """Keep one or two samples per stretch of near-identical frames of one video.

    `rows` are landmark rows of a single video sorted by frame. A segment starts at the
    first row and ends when cosine distance to its first row exceeds `threshold`, the hand drifts more than
    STILL_MOVE hand sizes from where it started, or the
    detections have a gap above MAX_GAP_S; the middle row is kept, plus the first one when
    the segment has >= `long_segment` rows. Kept rows are copies with `held_s`: how long
    the pose stayed put (segment span plus one sampling step).
    """
    if not rows:
        return []
    F = _unit(np.asarray([r["features"] for r in rows], dtype=np.float64))
    gaps = [_t(b) - _t(a) for a, b in zip(rows, rows[1:]) if 0 < _t(b) - _t(a) <= MAX_GAP_S]
    step = float(np.median(gaps)) if gaps else 0.0
    keep, start = [], 0
    for i in range(1, len(rows) + 1):
        if (i == len(rows) or 1.0 - F[i] @ F[start] > threshold
                or _drift(rows[start], rows[i]) > STILL_MOVE
                or _t(rows[i]) - _t(rows[i - 1]) > MAX_GAP_S):
            seg = list(range(start, i))
            held = round(_t(rows[seg[-1]]) - _t(rows[seg[0]]) + step, 3)
            picks = {seg[len(seg) // 2]} | ({seg[0]} if len(seg) >= long_segment else set())
            keep += [{**rows[j], "held_s": held, "seg_start": _t(rows[seg[0]]), "seg_end": _t(rows[seg[-1]])}
                     for j in sorted(picks)]
            start = i
    return keep


DYNAMIC_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "dynamic_signs.json"


def load_dynamic_signs(path: Path = DYNAMIC_CONFIG) -> set:
    """Names of the signs that need movement; the review player loops these, pauses the rest."""
    return set(json.loads(path.read_text(encoding="utf-8"))["dynamic"])


FIT_MAX_DIST = 0.17    # p95 of cosine distance to the own-class centroid on real held-out poses
REVIEW_HOLD_S = 0.6    # two consecutive samples of the 3 fps scrape (~0.67 s): quick signs in a fast clip count
MAX_PER_VIDEO = 30     # an alphabet video has 27 signs; the cap only stops a runaway clip


def is_reviewable(item: dict) -> bool:
    """A sample worth the reviewer's time: the pose was held and it resembles some sign.

    Moving hands and transitions are single frames (never held) or sit far from every
    reference pose. Dynamic signs are not exempt: a sequence needs another review, not a frame.
    """
    held = item.get("media") == "image" or item["held_s"] >= REVIEW_HOLD_S   # a picture has no hold
    return held and item["distance"] <= FIT_MAX_DIST


MOVING_DISP = 0.15       # hand sizes of raw landmark travel between neighbouring samples
VIDEO_MOTION_MAX = 0.80  # a video where more samples than this are in motion shows a travelling hand


def video_motion_share(rows: list) -> float:
    """Share of a video's samples (sorted by time) whose raw 2D landmarks moved more than
    MOVING_DISP hand sizes versus the nearest neighbour within MAX_GAP_S. Samples with no
    neighbour do not count as moving. A video that is mostly motion keeps leaking poses even
    after the per-sample hold test, so it is left out of the strict queue as a whole."""
    n = moving = 0
    for i, r in enumerate(rows):
        near = [rows[j] for j in (i - 1, i + 1) if 0 <= j < len(rows)
                and abs(_t(rows[j]) - _t(r)) <= MAX_GAP_S and "landmarks" in rows[j]]
        if not near or "landmarks" not in r:
            continue
        n += 1
        if min(_drift(r, o) for o in near) > MOVING_DISP:
            moving += 1
    return moving / n if n else 0.0


DUP_DIST = 0.02   # cosine distance under which two poses of one video are the same pose


def best_per_video(items: list, per_video: Optional[int] = None, dup_dist: float = DUP_DIST) -> list:
    """Keep up to `per_video` distinct poses of each video, the most confident first (longest
    held, then nearest). A pose within `dup_dist` of one already kept in that video is the same
    sign shown again, so it is dropped: one review covers it. Items without `features` are
    never collapsed."""
    per_video = MAX_PER_VIDEO if per_video is None else per_video
    keep, kept_units = [], {}
    for it in sorted(items, key=lambda x: (-x["held_s"], x["distance"])):
        units = kept_units.setdefault(it["video_id"], [])
        if len(units) >= per_video:
            continue
        u = _unit(np.asarray([it["features"]], dtype=np.float64))[0] if it.get("features") else None
        if u is not None and any(1.0 - float(u @ v) < dup_dist for v in units if v is not None):
            continue
        units.append(u)
        keep.append(it)
    return keep


TRACK_PRE_S, TRACK_POST_S = 1.0, 2.5   # old fixed loop window (kept for samples with no event bounds)
CLIP_PAD_PRE, CLIP_PAD_POST = 1.0, 1.0  # context around one sign event, in seconds
CLIP_MAX_S = 4.0


def event_clips(segments: list, held_min: float = None) -> dict:
    """{(seg_start, seg_end): (clip_start, clip_end)} for the held segments of one video.

    A loop of a fixed 3.5 s window showed ~3 consecutive signs of a fast lesson. The clip of a
    sign event is its own segment plus CLIP_PAD_* seconds of context, cut at the end of the
    previous event and the start of the next one, and never longer than CLIP_MAX_S, so a loop
    contains exactly one held sign. `segments` are (start, end, held_s) tuples.
    """
    held_min = REVIEW_HOLD_S if held_min is None else held_min
    segs = sorted(set(segments))
    events = [(a, b) for a, b, h in segs if h >= held_min]
    out = {}
    for a, b, _ in segs:
        prev_end = max([e for s0, e in events if e < a], default=None)
        next_start = min([s0 for s0, e in events if s0 > b], default=None)
        lo = a - CLIP_PAD_PRE if prev_end is None else max(a - CLIP_PAD_PRE, (prev_end + a) / 2)
        hi = b + CLIP_PAD_POST if next_start is None else min(b + CLIP_PAD_POST, (b + next_start) / 2)
        if hi - lo > CLIP_MAX_S:
            mid = (a + b) / 2
            lo, hi = max(lo, mid - CLIP_MAX_S / 2), min(hi, mid + CLIP_MAX_S / 2)
        out[(a, b)] = (round(max(0.0, lo), 3), round(hi, 3))
    return out


def track_around(rows: list, t: float, lo: float = None, hi: float = None) -> list:
    """[(t_s, landmarks)] of the video's rows inside the player's loop window.

    The window is the item's own event clip [lo, hi]; without bounds, the old fixed one around `t`."""
    lo = t - TRACK_PRE_S if lo is None else lo
    hi = t + TRACK_POST_S if hi is None else hi
    return [[round(_t(r), 3), np.round(np.asarray(r["landmarks"])[:, :2], 3).tolist()]
            for r in rows if lo <= _t(r) <= hi]


RELEVANT_TITLE = re.compile(
    r"abecedario|alfabeto|alphabet|spelling|deletre|dactilolog|\bletras?\b|n[uú]meros?\b|numbers?\b"
    r"|\bcontar\b|counting", re.I)
OTHER_LANGUAGE = re.compile(
    r"\basl\b|american|mexican|\blsm\b|ecuador|argentin|\blse\b|espa[ñn]ola|universal", re.I)


def is_relevant_title(title: Optional[str]) -> bool:
    """A video is worth reviewing when its title is about the alphabet or numbers and does
    not name another sign language. Titles only: it filters the queue, it never labels."""
    t = title or ""
    return bool(RELEVANT_TITLE.search(t)) and not OTHER_LANGUAGE.search(t)


LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "NN",
           "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
NUMBER_POSES = ["F", "V", "W", "4", "5", "6", "7", "8", "9", "10", "MIL", "MILLON"]  # 1, 2, 3 = F, V, W
NUMBERS_TITLE = re.compile(r"n[uú]mer|numbers?\b|\bcontar\b|counting|\b1 al 10\b", re.I)
LETTERS_TITLE = re.compile(r"abecedario|alfabeto|alphabet|spelling|deletre|dactilolog|\bletras?\b", re.I)


def video_kind(title: Optional[str], query_kind: Optional[str] = None) -> str:
    """letters, numbers or mixed, from the title; the search that found it when the title says neither."""
    t = title or ""
    num, let = bool(NUMBERS_TITLE.search(t)), bool(LETTERS_TITLE.search(t))
    if num and let:
        return "mixed"
    if num or let:
        return "numbers" if num else "letters"
    return {"numeros": "numbers", "lsc": "letters"}.get(query_kind, "mixed")


def allowed_classes(kind: str, names: list) -> np.ndarray:
    """Reference classes a video of this kind may be suggested; never none (the N key says that)."""
    ok = set(LETTERS) if kind == "letters" else set(NUMBER_POSES) if kind == "numbers" else set(names)
    mask = np.array([n in ok and n != "none" for n in names])
    return mask if mask.any() else np.array([n != "none" for n in names])


def suggestions(features: np.ndarray, cent: np.ndarray, names: list, kinds: list, k: int = 3) -> list:
    """The k nearest reference classes each row's video kind allows. An alphabet lesson is only
    offered letters and a number lesson the number poses: on the reviewer's own decisions this
    raised the right suggestion from 46 % to 56 % of real signs (docs/reports/T-22.md)."""
    dist = 1.0 - _unit(np.asarray(features, dtype=np.float64)) @ cent.T
    out = []
    for d, kind in zip(dist, kinds):
        d = np.where(allowed_classes(kind, names), d, np.inf)
        out.append([names[j] for j in np.argsort(d, kind="stable")[:k]])
    return out


def load_titles(out: Path) -> dict:
    return {c["video_id"]: c.get("title") for c in read_jsonl(out / "candidates.jsonl")}


def load_flags(out: Path) -> dict:
    """{video_id: embeddable bool}; the last record per video wins."""
    return {f["video_id"]: f["embeddable"] for f in read_jsonl(out / "video_flags.jsonl")}


def set_flag(out: Path, video_id: str, embeddable: bool, how: str = "page") -> None:
    rec = {"video_id": video_id, "embeddable": bool(embeddable), "how": how,
           "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with (out / "video_flags.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def review_items(out: Path, strict: bool = True) -> list:
    """Queue joined with landmarks, deduplicated, numbers first then rest, with any decision.

    With `strict` (default) only held poses that resemble a sign are returned, at most
    MAX_PER_VIDEO per video, from videos the page can play (embedding not known to be blocked); samples already decided always stay. `strict=False` returns all.
    """
    queue = {q["sample_id"]: q for q in read_jsonl(out / "review_queue.jsonl")}
    by_video = {}
    for r in read_jsonl(out / "landmarks.jsonl"):
        if sample_id(r) in queue:
            by_video.setdefault(r["video_id"], []).append(r)
    decided = latest_reviews(out)
    dynamic = load_dynamic_signs()
    titles, flags = load_titles(out), load_flags(out)
    scorer = queue_score.load()
    items = []
    travelling = set()
    for vid, rows in by_video.items():
        rows.sort(key=lambda r: r["frame"])
        if strict and video_motion_share(rows) > VIDEO_MOTION_MAX:
            travelling.add(vid)
        shown = dedupe_segments(rows)
        clips = event_clips([(r["seg_start"], r["seg_end"], r["held_s"]) for r in shown])
        have = {sample_id(r) for r in shown}
        unit = _unit(np.asarray([r["features"] for r in rows], dtype=np.float64))
        at = {r["frame"]: n for n, r in enumerate(rows)}
        # a decided sample always stays visible, even if a newer dedupe would pick another one
        shown += [{**r, "held_s": 0.0} for r in rows if sample_id(r) in decided and sample_id(r) not in have]
        for r in shown:
            q = queue[sample_id(r)]
            pose = queue_score.pose_features(rows, unit, at[r["frame"]], q["distance"], MAX_GAP_S)
            items.append({**q, "video_id": r["video_id"], "frame": r["frame"], "pose": pose,
                          "score": queue_score.score(scorer, pose) if scorer else None,
                          "dynamic": q.get("media") != "image" and q["suggested"] in dynamic,
                          "held_s": r["held_s"],
                          "title": q.get("title") or titles.get(r["video_id"]),
                          "relevant": q.get("media") == "image" or is_relevant_title(titles.get(r["video_id"])),
                          "embeddable": flags.get(r["video_id"]),
                          "teaching": q.get("media") == "image" or r["held_s"] >= REVIEW_HOLD_S
                          or q["suggested"] in dynamic,
                          "is_left": r["is_left"], "landmarks": r["landmarks"], "features": r["features"],
                          "clip": list(clips.get((r.get("seg_start"), r.get("seg_end")),
                                                 (_t(r) - TRACK_PRE_S, _t(r) + TRACK_POST_S))),
                          "track": track_around(rows, _t(r), *clips[(r["seg_start"], r["seg_end"])])
                          if "seg_start" in r else track_around(rows, _t(r)),
                          "decision": decided.get(q["sample_id"])})
    if strict:
        cut = scorer.get("min_score") if scorer else None
        kept = best_per_video([x for x in items if x["decision"] is None and is_reviewable(x)
                               and x["video_id"] not in travelling and x["embeddable"] is not False
                               and (cut is None or x.get("media") == "image" or x["score"] >= cut)])
        ids = {x["sample_id"] for x in kept}
        items = [x for x in items if x["decision"] is not None or x["sample_id"] in ids]
    # most likely real signs first (score), so the quick accepts come early; no scorer: by video/time
    items.sort(key=lambda x: (not x["teaching"], KIND_PRIORITY.get(x["query_kind"], 9),
                              -(x["score"] if x["score"] is not None else 0.0), x["video_id"], x["frame"]))
    return items


def build(out: Path, refs: Path) -> int:
    """Write review_queue.jsonl: every scraped sample, closest to a reference first.

    `distance` and `margin` say how much the pose looks like any reference sign (the filters
    and the scorer use them); `suggested` is the nearest class the video's kind allows.

    Reviewed samples stay in the queue: the page joins their decision from reviews.jsonl, so
    progress survives a rebuild (dropping them hid the work already done).
    """
    rows = [r for r in read_jsonl(out / "landmarks.jsonl") if r.get("review_status") == "pending"]
    queue = []
    if rows:
        cent, names = load_refs(refs)
        F = np.asarray([r["features"] for r in rows])
        ranked = rank_samples(F, cent, names)
        titles = load_titles(out)
        kinds = ["mixed" if r.get("media") == "image" else
                 video_kind(titles.get(r["video_id"]) or r.get("title"), r.get("query_kind")) for r in rows]
        sug = suggestions(F, cent, names, kinds, k=2)
        queue = [{"sample_id": sample_id(r), "url": r["url"], "t_s": r["t_s"],
                  "query_kind": r["query_kind"], "media": r.get("media", "video"),
                  "image_url": r.get("image_url"), "hint": r.get("hint"), "title": r.get("title"),
                  "license": r.get("license"), **k, "kind": kind, "suggested": s2[0], "second": s2[1],
                  "suggested_names": alias_poses.display_names(s2[0])}
                 for r, k, kind, s2 in zip(rows, ranked, kinds, sug)]
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
            w.writerow({"sample_id": q["sample_id"],
                        "url_at_time": q["url"] if q.get("media") == "image" else url_at(q["url"], q["t_s"]),
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
            f.write(json.dumps({**r, "label": rev["label"],
                                "pose": alias_poses.canonical(rev["label"]),
                                "review_status": "validated",
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
