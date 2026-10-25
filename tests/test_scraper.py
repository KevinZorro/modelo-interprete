import json
import os

import numpy as np

from lsc70 import scraper

META = {"video_id": "abc", "url": "https://example.test/v/abc", "channel": "ch",
        "channel_url": None, "query": "q", "query_kind": "lsc"}


def _hand():
    rng = np.random.default_rng(0)
    return rng.uniform(0.2, 0.8, (21, 3)) * [1, 1, 0.1], False


def _fake_download(url, directory):
    p = os.path.join(directory, "v.mp4")
    open(p, "wb").write(b"x")
    return p


def _frames(path):
    return ((i, i / 30, object()) for i in range(100))


def test_rate_limiter_enforces_gap():
    t = [0.0]
    slept = []
    lim = scraper.RateLimiter(5, clock=lambda: t[0], sleep=lambda s: (slept.append(s), t.__setitem__(0, t[0] + s)))
    lim.wait()
    t[0] = 2.0
    lim.wait()
    assert slept == [3.0]


def test_samples_use_stride_and_cap():
    out = list(scraper.samples_from_frames(_frames(None), lambda img: _hand(), stride=10, max_samples=4))
    assert [s["frame"] for s in out] == [0, 10, 20, 30]
    assert len(out[0]["features"]) == 63 and len(out[0]["landmarks"]) == 21


def test_no_hand_and_degenerate_are_skipped():
    flat = (np.zeros((21, 3)), False)
    assert list(scraper.samples_from_frames(_frames(None), lambda img: None)) == []
    assert list(scraper.samples_from_frames(_frames(None), lambda img: flat)) == []


def test_process_video_deletes_video_and_stores_no_pixels(tmp_path):
    seen = {}

    def dl(url, d):
        seen["dir"] = d
        return _fake_download(url, d)

    rows = scraper.process_video(META, lambda img: _hand(), downloader=dl, frames_fn=_frames)
    assert rows and not os.path.exists(seen["dir"])
    assert all(r["label"] is None and r["review_status"] == "pending" for r in rows)
    assert all(r["url"] == META["url"] and r["channel"] == "ch" and r["processed_at"] for r in rows)
    allowed = {"frame", "t_s", "is_left", "landmarks", "features", "label", "review_status", "url",
               "video_id", "channel", "channel_url", "query", "query_kind", "processed_at", "contract"}
    assert set(rows[0]) == allowed


def test_run_is_resumable_and_logs_failures(tmp_path):
    scraper.append_jsonl(tmp_path / "candidates.jsonl",
                         [META, {**META, "video_id": "bad", "url": "u2"}])

    def dl(url, d):
        if url == "u2":
            raise RuntimeError("gone")
        return _fake_download(url, d)

    lim = scraper.RateLimiter(0)
    s1 = scraper.run(tmp_path, lambda img: _hand(), 10, lim, dl, _frames)
    assert s1["videos"] == 1 and s1["failed"] == 1
    n = len((tmp_path / "landmarks.jsonl").read_text().splitlines())
    scraper.run(tmp_path, lambda img: _hand(), 10, lim, dl, _frames)
    assert len((tmp_path / "landmarks.jsonl").read_text().splitlines()) == n  # "abc" not redone
    json.loads((tmp_path / "failed.jsonl").read_text().splitlines()[0])
