"""Landmark scraper (T-13): public videos in, hand landmarks + provenance out.

Each video is downloaded into a TemporaryDirectory, decoded frame by frame in memory,
passed through MediaPipe and `preprocess`, and the directory is deleted before the
function returns (even on error). Frames are never written anywhere: the only thing that
reaches disk is `landmarks.jsonl`, one record per sample, with its provenance.

Labels are NEVER assigned here. `label` stays None and `review_status` is "pending"; the
`query_kind` field only records which search found the video (a hint for the review
queue, T-14), not what the hand is doing.

    python -m lsc70.scraper discover "abecedario LSC" --kind lsc --out data/scraper
    python -m lsc70.scraper run --out data/scraper --model hand_landmarker.task
"""
import argparse
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import numpy as np

from lsc70.preprocessing import CONTRACT_VERSION, preprocess

KINDS = ("lsc", "reposo", "numeros")
MIN_SECONDS_BETWEEN_REQUESTS = 5.0
DEFAULT_STRIDE = 10          # one frame in ten: ~3 fps at 30 fps, enough poses, little redundancy
DEFAULT_MAX_PER_VIDEO = 60
MAX_HEIGHT = 480             # lowest resolution that still lets MediaPipe find a hand


class RateLimiter:
    """Blocks so that consecutive `wait()` calls are at least `min_gap` seconds apart."""

    def __init__(self, min_gap: float = MIN_SECONDS_BETWEEN_REQUESTS, clock=time.monotonic,
                 sleep=time.sleep):
        self.min_gap, self._clock, self._sleep, self._last = min_gap, clock, sleep, None

    def wait(self):
        now = self._clock()
        if self._last is not None and now - self._last < self.min_gap:
            self._sleep(self.min_gap - (now - self._last))
        self._last = self._clock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----- discovery (metadata only, nothing downloaded) -----

def discover(query: str, kind: str, limit: int, limiter: Optional[RateLimiter] = None) -> list:
    """Candidate videos for `query` as dicts of metadata. Needs `pip install yt-dlp`."""
    import yt_dlp

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    (limiter or RateLimiter()).wait()
    opts = {"quiet": True, "skip_download": True, "extract_flat": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
    return [
        {
            "video_id": e.get("id"),
            "url": e.get("url") or f"https://www.youtube.com/watch?v={e.get('id')}",
            "channel": e.get("channel") or e.get("uploader"),
            "channel_url": e.get("channel_url"),
            "title": e.get("title"),
            "duration_s": e.get("duration"),
            "query": query,
            "query_kind": kind,
            "discovered_at": _now(),
        }
        for e in (info.get("entries") or []) if e.get("id")
    ]


# ----- per-video processing -----

def samples_from_frames(frames: Iterable, detect: Callable, stride: int = DEFAULT_STRIDE,
                        max_samples: int = DEFAULT_MAX_PER_VIDEO) -> Iterator[dict]:
    """Yield sample dicts from `frames` (iterable of (index, t_seconds, image)).

    `detect(image)` returns `(landmarks, is_left)` or None. Frames with no hand or a
    rejected landmark set are skipped; the image is dropped as soon as the loop moves on.
    """
    n = 0
    for idx, t, img in frames:
        if idx % stride:
            continue
        found = detect(img)
        if found is None:
            continue
        landmarks, is_left = found
        res = preprocess(landmarks, is_left)
        if not res.ok:
            continue
        raw = np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=np.float64) \
            if hasattr(landmarks[0], "x") else np.asarray(landmarks, dtype=np.float64)
        yield {
            "frame": int(idx),
            "t_s": round(float(t), 3),
            "is_left": bool(is_left),
            "landmarks": np.round(raw, 5).tolist(),
            "features": np.round(res.features, 6).tolist(),
        }
        n += 1
        if n >= max_samples:
            return


def _video_frames(path: str):
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                return
            yield i, i / fps, img
            i += 1
    finally:
        cap.release()


def mediapipe_detect(model_path: str) -> Callable:
    from lsc70.normalizacion import crear_detector, detectar, es_mano_izquierda

    det = crear_detector(model_path)

    def detect(img):
        res = detectar(det, img)
        if not res.hand_landmarks:
            return None
        return res.hand_landmarks[0], es_mano_izquierda(res)

    return detect


def process_video(meta: dict, detect: Callable, stride: int = DEFAULT_STRIDE,
                  max_samples: int = DEFAULT_MAX_PER_VIDEO, downloader: Optional[Callable] = None,
                  frames_fn: Callable = _video_frames) -> list:
    """Download `meta["url"]` to a temp dir, extract samples, delete the video.

    `downloader(url, dir) -> path` and `frames_fn(path)` are injectable for tests.
    """
    downloader = downloader or _ytdlp_download
    with tempfile.TemporaryDirectory(prefix="lsc_scraper_") as tmp:
        path = downloader(meta["url"], tmp)
        samples = list(samples_from_frames(frames_fn(path), detect, stride, max_samples))
    prov = {
        "url": meta["url"], "video_id": meta["video_id"], "channel": meta.get("channel"),
        "channel_url": meta.get("channel_url"), "query": meta.get("query"),
        "query_kind": meta.get("query_kind"), "processed_at": _now(),
        "contract": CONTRACT_VERSION,
    }
    return [{**prov, **s, "label": None, "review_status": "pending"} for s in samples]


def _ytdlp_download(url: str, directory: str) -> str:
    import yt_dlp

    opts = {
        "quiet": True, "noplaylist": True, "outtmpl": f"{directory}/v.%(ext)s",
        "format": f"worst[height>={MAX_HEIGHT}]/best[height<={MAX_HEIGHT}]/worst",
        "retries": 2,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    files = list(Path(directory).glob("v.*"))
    if not files:
        raise RuntimeError("download produced no file")
    return str(files[0])


# ----- state on disk -----

def load_candidates(out: Path) -> list:
    p = out / "candidates.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x] if p.exists() else []


def done_ids(out: Path) -> set:
    p = out / "done.txt"
    return set(p.read_text().split()) if p.exists() else set()


def append_jsonl(path: Path, rows: Iterable[dict]):
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(out: Path, detect: Callable, limit_videos: int, limiter: Optional[RateLimiter] = None,
        downloader: Optional[Callable] = None, frames_fn: Callable = _video_frames,
        **kw) -> dict:
    """Process pending candidates; append samples to landmarks.jsonl. Failures are logged
    in failed.jsonl and do not stop the run."""
    limiter = limiter or RateLimiter()
    done = done_ids(out)
    todo = [c for c in load_candidates(out) if c["video_id"] not in done][:limit_videos]
    stats = {"videos": 0, "samples": 0, "failed": 0}
    for meta in todo:
        limiter.wait()
        try:
            rows = process_video(meta, detect, downloader=downloader, frames_fn=frames_fn, **kw)
        except Exception as e:  # network, removed video, codec: log and go on
            append_jsonl(out / "failed.jsonl", [{"video_id": meta["video_id"], "error": repr(e)[:300]}])
            stats["failed"] += 1
            continue
        append_jsonl(out / "landmarks.jsonl", rows)
        with (out / "done.txt").open("a") as f:
            f.write(meta["video_id"] + "\n")
        stats["videos"] += 1
        stats["samples"] += len(rows)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover")
    d.add_argument("query")
    d.add_argument("--kind", choices=KINDS, required=True)
    d.add_argument("--limit", type=int, default=20)
    d.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--model", required=True, help="hand_landmarker.task")
    r.add_argument("--limit-videos", type=int, default=10)
    r.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    r.add_argument("--max-per-video", type=int, default=DEFAULT_MAX_PER_VIDEO)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "discover":
        known = {c["video_id"] for c in load_candidates(a.out)}
        new = [c for c in discover(a.query, a.kind, a.limit) if c["video_id"] not in known]
        append_jsonl(a.out / "candidates.jsonl", new)
        print(f"{len(new)} new candidates")
    else:
        print(run(a.out, mediapipe_detect(a.model), a.limit_videos, stride=a.stride,
                  max_samples=a.max_per_video))


if __name__ == "__main__":
    main()
