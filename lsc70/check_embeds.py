"""Mark which scraped videos the review page can play (embedding allowed) via YouTube oEmbed.

    python -m lsc70.check_embeds --out data/scraper

oEmbed answers 200 when the video can be embedded and 401 when its owner disabled embedding
(404 = removed). One request per video, 5 s apart; the result goes to video_flags.jsonl.
"""
import argparse
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from lsc70 import review_queue as rq
from lsc70.scraper import RateLimiter

OEMBED = "https://www.youtube.com/oembed?format=json&url="


def check(video_id: str, opener=urllib.request.urlopen) -> bool:
    url = OEMBED + urllib.parse.quote(f"https://www.youtube.com/watch?v={video_id}", safe="")
    try:
        opener(urllib.request.Request(url, headers={"User-Agent": "SignaCo-LSC-research/0.1"}), timeout=15)
        return True
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            return False
        raise


def run(out: Path, limiter=None, opener=urllib.request.urlopen, recheck: bool = False) -> dict:
    limiter = limiter or RateLimiter()
    known = rq.load_flags(out)
    ids = sorted({r["video_id"] for r in rq.read_jsonl(out / "landmarks.jsonl") if r.get("media") != "image"})
    stats = {"checked": 0, "embeddable": 0, "blocked": 0, "error": 0}
    for vid in ids:
        if vid in known and not recheck:
            continue
        limiter.wait()
        try:
            ok = check(vid, opener)
        except Exception:
            stats["error"] += 1       # unknown stays unknown (page treats it as playable)
            continue
        rq.set_flag(out, vid, ok, "oembed")
        stats["checked"] += 1
        stats["embeddable" if ok else "blocked"] += 1
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check which videos allow embedding")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--recheck", action="store_true")
    a = ap.parse_args(argv)
    print(run(a.out, recheck=a.recheck))


if __name__ == "__main__":
    main()
