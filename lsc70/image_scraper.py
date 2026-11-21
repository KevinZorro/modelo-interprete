"""Landmark scraper for public web images (static poses): search -> in-memory -> landmarks.

Same rules as `lsc70.scraper`: the image is decoded in memory and dropped, only landmarks
are stored, with per-sample provenance (page URL, image URL, source, licence when the source
gives one, date). `label` stays null and `review_status` pending: the query `hint` is only a
suggestion for the human reviewer, who needs to check it (web labels are often wrong, and ASL
shows up as LSC). Rate limit between requests; no cookies; a block (HTTP 403/429 or the
search library's rate-limit error) stops the run and is reported, never worked around.

    python -m lsc70.image_scraper discover "letra B lengua de señas colombiana" --kind lsc \
        --hint B --source ddg|openverse|commons --out data/scraper
    python -m lsc70.image_scraper run --model hand_landmarker.task --limit-images 100 --out data/scraper
"""
import argparse
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from lsc70.preprocessing import CONTRACT_VERSION, preprocess
from lsc70.scraper import KINDS, RateLimiter, _now, append_jsonl

SOURCES = ("ddg", "openverse", "commons")
USER_AGENT = "SignaCo-LSC-research/0.1 (+https://github.com/KevinZorro/modelo-interprete)"
MAX_BYTES = 8_000_000
MAX_SIDE = 1280          # larger images are shrunk before detection (speed, not accuracy)
TIMEOUT_S = 20
MAX_CONSECUTIVE_BLOCKS = 5


class Blocked(Exception):
    """The source refused us (403/429 or its own rate limit). Stop; do not retry around it."""


def image_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def _candidate(image_url, page_url, title, source, license_, meta) -> dict:
    return {"image_id": image_id(image_url), "image_url": image_url, "page_url": page_url or image_url,
            "title": title, "source": source, "license": license_, "query": meta["query"],
            "query_kind": meta["kind"], "hint": meta.get("hint"), "discovered_at": _now()}


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (403, 429):
            raise Blocked(f"HTTP {e.code} from {urllib.parse.urlparse(url).netloc}") from e
        raise


def search_ddg(query: str, limit: int, meta: dict) -> list:
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException, RatelimitException

    try:
        hits = DDGS().images(query, max_results=limit)
    except RatelimitException as e:
        raise Blocked(f"search engine rate limit: {e}") from e
    except DDGSException as e:
        raise Blocked(f"search engine error: {e}") from e
    return [_candidate(h["image"], h.get("url"), h.get("title"), "ddg:" + (h.get("source") or ""),
                       None, meta) for h in hits if h.get("image")]


def search_openverse(query: str, limit: int, meta: dict) -> list:
    q = urllib.parse.urlencode({"q": query, "page_size": min(limit, 20)})
    data = _get_json(f"https://api.openverse.org/v1/images/?{q}")
    return [_candidate(h["url"], h.get("foreign_landing_url"), h.get("title"), "openverse:" + str(h.get("source")),
                       f"{h.get('license')} {h.get('license_version') or ''}".strip(), meta)
            for h in data.get("results", []) if h.get("url")]


def search_commons(query: str, limit: int, meta: dict) -> list:
    q = urllib.parse.urlencode({
        "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
        "gsrsearch": f"{query} filetype:bitmap", "gsrlimit": min(limit, 50), "prop": "imageinfo",
        "iiprop": "url|extmetadata", "iiurlwidth": 800})
    pages = _get_json(f"https://commons.wikimedia.org/w/api.php?{q}").get("query", {}).get("pages", {})
    out = []
    for p in pages.values():
        info = (p.get("imageinfo") or [{}])[0]
        lic = (info.get("extmetadata") or {}).get("LicenseShortName", {}).get("value")
        url = info.get("thumburl") or info.get("url")
        if url:
            out.append(_candidate(url, info.get("descriptionurl"), p.get("title"), "commons", lic, meta))
    return out


SEARCHERS = {"ddg": search_ddg, "openverse": search_openverse, "commons": search_commons}


def discover(query: str, kind: str, source: str, limit: int, hint: Optional[str] = None,
             limiter: Optional[RateLimiter] = None) -> list:
    if kind not in KINDS or source not in SOURCES:
        raise ValueError(f"kind must be one of {KINDS} and source one of {SOURCES}")
    (limiter or RateLimiter()).wait()
    return SEARCHERS[source](query, limit, {"query": query, "kind": kind, "hint": hint})


def fetch_bytes(url: str) -> bytes:
    """Download one image into memory (capped); never touches disk."""
    url = urllib.parse.quote(url, safe=":/?&=%#@+,;~!*()$'[]")   # non-ASCII paths break urllib
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            if not (r.headers.get("Content-Type") or "").startswith("image/"):
                raise ValueError("not an image")
            data = r.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as e:
        if e.code in (403, 429):
            raise Blocked(f"HTTP {e.code} from {urllib.parse.urlparse(url).netloc}") from e
        raise
    if len(data) > MAX_BYTES:
        raise ValueError("image too large")
    return data


def decode(data: bytes):
    import cv2

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cannot decode image")
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        k = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    return img


def sample_from_image(meta: dict, detect: Callable, fetcher: Callable = fetch_bytes,
                      decoder: Callable = decode) -> tuple:
    """(row or None, stats). Row has the same fields as video samples plus image provenance."""
    img = decoder(fetcher(meta["image_url"]))
    found = detect(img)
    del img
    stats = {"image_id": meta["image_id"], "hand": found is not None, "ok": False}
    if found is None:
        return None, stats
    landmarks, is_left = found
    res = preprocess(landmarks, is_left)
    if not res.ok:
        return None, stats
    stats["ok"] = True
    raw = np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=np.float64) \
        if hasattr(landmarks[0], "x") else np.asarray(landmarks, dtype=np.float64)
    host = urllib.parse.urlparse(meta["page_url"]).netloc
    return {
        "media": "image", "video_id": "img_" + meta["image_id"], "frame": 0, "t_s": 0.0,
        "url": meta["page_url"], "page_url": meta["page_url"], "image_url": meta["image_url"],
        "title": meta.get("title"), "license": meta.get("license"), "source": meta["source"],
        "channel": host, "channel_url": None, "query": meta["query"], "query_kind": meta["query_kind"],
        "hint": meta.get("hint"), "processed_at": _now(), "contract": CONTRACT_VERSION,
        "is_left": bool(is_left), "landmarks": np.round(raw, 5).tolist(),
        "features": np.round(res.features, 6).tolist(), "label": None, "review_status": "pending",
    }, stats


def _read(p: Path) -> list:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x] if p.exists() else []


def add_candidates(out: Path, cands: list) -> int:
    seen = {c["image_id"] for c in _read(out / "images_candidates.jsonl")}
    new = [c for c in cands if c["image_id"] not in seen]
    append_jsonl(out / "images_candidates.jsonl", new)
    return len(new)


def run(out: Path, detect: Callable, limit_images: int, limiter: Optional[RateLimiter] = None,
        fetcher: Callable = fetch_bytes, decoder: Callable = decode) -> dict:
    """Process pending image candidates.

    A host that answers 403/429 is skipped for the rest of the run (we do not retry or work
    around it); MAX_CONSECUTIVE_BLOCKS refusals in a row, from any hosts, stop the run.
    Other failures (dead link, not an image, decode error) are logged and the run goes on.
    """
    limiter = limiter or RateLimiter()
    done_p = out / "images_done.txt"
    done = set(done_p.read_text(encoding="utf-8").split()) if done_p.exists() else set()
    todo = [c for c in _read(out / "images_candidates.jsonl") if c["image_id"] not in done][:limit_images]
    stats = {"images": 0, "samples": 0, "no_hand": 0, "failed": 0, "blocked": None, "hosts_skipped": 0}
    refused, streak = set(), 0
    for meta in todo:
        host = urllib.parse.urlparse(meta["image_url"]).netloc
        if host in refused:
            continue
        limiter.wait()
        try:
            row, st = sample_from_image(meta, detect, fetcher, decoder)
            streak = 0
        except Blocked as e:
            refused.add(host)
            stats["hosts_skipped"] = len(refused)
            streak += 1
            append_jsonl(out / "images_failed.jsonl", [{"image_id": meta["image_id"], "error": str(e)}])
            if streak >= MAX_CONSECUTIVE_BLOCKS:
                stats["blocked"] = f"{streak} refusals in a row, last: {e}"
                break
            continue
        except Exception as e:  # dead link, not an image, decode error: log and go on
            append_jsonl(out / "images_failed.jsonl", [{"image_id": meta["image_id"], "error": repr(e)[:300]}])
            stats["failed"] += 1
            with done_p.open("a", encoding="utf-8") as f:   # a dead link stays dead: do not refetch
                f.write(meta["image_id"] + "\n")
            continue
        append_jsonl(out / "images_stats.jsonl", [{**st, "query_kind": meta["query_kind"], "source": meta["source"]}])
        if row:
            append_jsonl(out / "landmarks.jsonl", [row])
            stats["samples"] += 1
        else:
            stats["no_hand"] += 1
        with done_p.open("a", encoding="utf-8") as f:
            f.write(meta["image_id"] + "\n")
        stats["images"] += 1
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description="Landmark scraper for web images")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover")
    d.add_argument("query")
    d.add_argument("--kind", choices=KINDS, required=True)
    d.add_argument("--source", choices=SOURCES, required=True)
    d.add_argument("--hint", help="label the query is about (a suggestion for the reviewer)")
    d.add_argument("--limit", type=int, default=30)
    d.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--model", required=True, help="hand_landmarker.task")
    r.add_argument("--limit-images", type=int, default=100)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "discover":
        try:
            cands = discover(a.query, a.kind, a.source, a.limit, a.hint)
        except Blocked as e:
            print({"blocked": str(e)})
            return
        print(f"{add_candidates(a.out, cands)} new candidates")
    else:
        from lsc70.scraper import mediapipe_detect
        print(run(a.out, mediapipe_detect(a.model), a.limit_images))


if __name__ == "__main__":
    main()
