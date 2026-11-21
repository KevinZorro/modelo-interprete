import urllib.error

from lsc70 import check_embeds as ce
from lsc70 import review_queue as rq
from lsc70.scraper import RateLimiter


def _opener(blocked, boom=()):
    def op(req, timeout):
        vid = req.full_url.split("%3Dv%3D")[-1] if False else req.full_url
        if any(b in vid for b in blocked):
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)
        if any(b in vid for b in boom):
            raise urllib.error.URLError("net down")
        return object()
    return op


def test_run_flags_blocked_videos_and_skips_known(tmp_path):
    import json
    rows = [{"video_id": v, "frame": 0, "features": [1.0] * 63} for v in ("aaa", "bbb", "ccc")]
    (tmp_path / "landmarks.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    lim = RateLimiter(0, sleep=lambda s: None)
    st = ce.run(tmp_path, lim, _opener(["bbb"], ["ccc"]))
    assert st == {"checked": 2, "embeddable": 1, "blocked": 1, "error": 1}
    assert rq.load_flags(tmp_path) == {"aaa": True, "bbb": False}     # ccc stays unknown
    assert ce.run(tmp_path, lim, _opener([]))["checked"] == 1           # only ccc left
