import json

import numpy as np
import pytest

from lsc70 import image_scraper as isc
from lsc70 import review_queue as rq
from lsc70.scraper import RateLimiter

META = {"image_id": "abc123", "image_url": "https://img.test/a.jpg", "page_url": "https://site.test/p",
        "title": "Letra B", "source": "ddg:site", "license": None, "query": "q", "query_kind": "lsc",
        "hint": "B"}


def _hand():
    rng = np.random.default_rng(0)
    return rng.uniform(0.2, 0.8, (21, 3)) * [1, 1, 0.1], False


def _nolimit():
    return RateLimiter(0, sleep=lambda s: None)


def test_sample_has_provenance_and_no_label():
    row, st = isc.sample_from_image(META, lambda img: _hand(), lambda u: b"x", lambda b: object())
    assert st == {"image_id": "abc123", "hand": True, "ok": True}
    assert row["label"] is None and row["review_status"] == "pending" and row["media"] == "image"
    assert row["page_url"] == "https://site.test/p" and row["image_url"] == META["image_url"]
    assert row["channel"] == "site.test" and row["hint"] == "B"
    assert row["video_id"] == "img_abc123" and row["frame"] == 0
    assert len(row["features"]) == 63 and len(row["landmarks"]) == 21
    assert "pixels" not in row and not any(k in row for k in ("image", "bytes", "data"))


def test_no_hand_and_degenerate_give_no_row():
    row, st = isc.sample_from_image(META, lambda img: None, lambda u: b"x", lambda b: object())
    assert row is None and st["hand"] is False
    row, st = isc.sample_from_image(META, lambda img: (np.zeros((21, 3)), False), lambda u: b"x", lambda b: object())
    assert row is None and st == {"image_id": "abc123", "hand": True, "ok": False}


def test_run_skips_refusing_host_and_stops_after_repeated_blocks(tmp_path):
    cands = [{**META, "image_id": f"id{i}", "image_url": f"https://h{i % 2}.test/{i}.jpg"} for i in range(6)]
    isc.add_candidates(tmp_path, cands)

    def fetch(url):
        if "h1.test" in url:
            raise isc.Blocked("HTTP 403")
        if url.endswith("2.jpg"):
            raise ValueError("not an image")
        return b"x"

    st = isc.run(tmp_path, lambda img: _hand(), 10, _nolimit(), fetch, lambda b: object())
    # h0: id0 ok, id2 dead link, id4 ok; h1: id1 refused once, id3 and id5 skipped without a request
    assert st == {"images": 2, "samples": 2, "no_hand": 0, "failed": 1, "blocked": None, "hosts_skipped": 1}
    rows = [json.loads(x) for x in (tmp_path / "landmarks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["video_id"] for r in rows] == ["img_id0", "img_id4"]
    assert isc.run(tmp_path, lambda img: _hand(), 10, _nolimit(), fetch, lambda b: object())["images"] == 0


def test_run_stops_when_every_host_refuses(tmp_path):
    cands = [{**META, "image_id": f"id{i}", "image_url": f"https://h{i}.test/{i}.jpg"} for i in range(8)]
    isc.add_candidates(tmp_path, cands)

    def fetch(url):
        raise isc.Blocked("HTTP 429")

    st = isc.run(tmp_path, lambda img: _hand(), 10, _nolimit(), fetch, lambda b: object())
    assert st["images"] == 0 and st["hosts_skipped"] == isc.MAX_CONSECUTIVE_BLOCKS
    assert "refusals in a row" in st["blocked"]


def test_search_parsers(monkeypatch):
    ov = {"results": [{"url": "https://o.test/1.jpg", "foreign_landing_url": "https://o.test/p1", "title": "t",
                       "source": "flickr", "license": "by", "license_version": "2.0"}]}
    monkeypatch.setattr(isc, "_get_json", lambda url: ov)
    c = isc.search_openverse("q", 5, {"query": "q", "kind": "lsc", "hint": "A"})
    assert c[0]["license"] == "by 2.0" and c[0]["source"] == "openverse:flickr" and c[0]["hint"] == "A"
    cm = {"query": {"pages": {"1": {"title": "File:A.jpg", "imageinfo": [{
        "thumburl": "https://c.test/t.jpg", "descriptionurl": "https://c.test/File:A.jpg",
        "extmetadata": {"LicenseShortName": {"value": "CC BY-SA 4.0"}}}]}}}}
    monkeypatch.setattr(isc, "_get_json", lambda url: cm)
    c = isc.search_commons("q", 5, {"query": "q", "kind": "lsc"})
    assert c[0]["license"] == "CC BY-SA 4.0" and c[0]["image_url"] == "https://c.test/t.jpg"


def test_discover_validates_arguments():
    with pytest.raises(ValueError):
        isc.discover("q", "bad", "ddg", 5)
    with pytest.raises(ValueError):
        isc.discover("q", "lsc", "nowhere", 5)


def test_image_rows_flow_into_review_queue(tmp_path):
    row, _ = isc.sample_from_image(META, lambda img: _hand(), lambda u: b"x", lambda b: object())
    (tmp_path / "landmarks.jsonl").write_text(json.dumps(row), encoding="utf-8")
    cent, names = rq.class_centroids(np.vstack([row["features"], np.ones(63)]), np.array([0, 1]), ["A", "B"])
    np.savez(tmp_path / "refs.npz", Xtr=np.vstack([row["features"], np.ones(63)]), ytr=np.array([0, 1]),
             etiquetas=np.array(names))
    assert rq.build(tmp_path, tmp_path / "refs.npz") == 1
    item = rq.review_items(tmp_path)[0]
    assert item["media"] == "image" and item["teaching"] is True and item["dynamic"] is False
    assert item["image_url"] == META["image_url"] and item["hint"] == "B"
    sheet = rq.write_sheet(tmp_path).read_text(encoding="utf-8-sig")
    assert "https://site.test/p" in sheet and "&t=" not in sheet


def test_fetch_quotes_non_ascii_urls(monkeypatch):
    seen = {}

    class R:
        headers = {"Content-Type": "image/png"}
        def read(self, n): return b"x"
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake(req, timeout):
        seen["url"] = req.full_url
        return R()
    monkeypatch.setattr(isc.urllib.request, "urlopen", fake)
    assert isc.fetch_bytes("https://x.test/media/seña ñ.png") == b"x"
    assert seen["url"] == "https://x.test/media/se%C3%B1a%20%C3%B1.png"
