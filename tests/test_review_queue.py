import csv
import json

import numpy as np
import pytest

from lsc70 import review_queue as rq


def _refs():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal([1, 0, 0] + [0] * 60, 0.05, (20, 63)),
                   rng.normal([0, 1, 0] + [0] * 60, 0.05, (20, 63))])
    return rq.class_centroids(X, np.repeat([0, 1], 20), ["A", "B"])


def test_rank_suggests_nearest_class_and_orders_by_distance():
    cent, names = _refs()
    f = np.zeros((2, 63))
    f[0, 1] = 1.0
    f[1, 0], f[1, 2] = 1.0, 0.8
    r = rq.rank_samples(f, cent, names)
    assert r[0]["suggested"] == "B" and r[1]["suggested"] == "A"
    assert r[0]["distance"] < r[1]["distance"]


def _write_run(tmp_path, n=3):
    rows = [{"video_id": "v", "frame": i * 10, "t_s": i, "url": "https://y.test/watch?v=v",
             "query_kind": "lsc", "features": ([1.0] + [0.0] * 62) if i else ([0.0, 1.0] + [0.0] * 61),
             "label": None, "review_status": "pending"} for i in range(n)]
    (tmp_path / "landmarks.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    cent, names = _refs()
    np.savez(tmp_path / "refs.npz", Xtr=np.vstack([cent[0]] * 3 + [cent[1]] * 3),
             ytr=np.array([0, 0, 0, 1, 1, 1]), etiquetas=np.array(names))


def test_build_never_sets_label_and_sheet_roundtrip(tmp_path):
    _write_run(tmp_path)
    assert rq.build(tmp_path, tmp_path / "refs.npz") == 3
    q = rq.read_jsonl(tmp_path / "review_queue.jsonl")
    assert all("label" not in x for x in q)
    sheet = rq.write_sheet(tmp_path)
    rows = list(csv.DictReader(sheet.open(encoding="utf-8-sig")))
    assert "&t=" in rows[0]["url_at_time"]
    rows[0]["decision"] = "accept"
    rows[1]["decision"], rows[1]["label"] = "relabel", "C"
    rows[2]["decision"] = "reject"
    with sheet.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rq.SHEET_COLS)
        w.writeheader()
        w.writerows(rows)
    assert rq.import_sheet(tmp_path, sheet, "Ana") == 3
    assert rq.import_sheet(tmp_path, sheet, "Ana") == 0          # idempotent
    assert rq.export_validated(tmp_path) == 2                     # reject is dropped
    val = rq.read_jsonl(tmp_path / "validated.jsonl")
    assert {v["label"] for v in val} == {rows[0]["suggested"], "C"}
    assert all(v["reviewer"] == "Ana" and v["review_status"] == "validated" for v in val)
    assert rq.build(tmp_path, tmp_path / "refs.npz") == 0         # reviewed samples leave the queue


def test_bad_decisions_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        rq.parse_decision({"sample_id": "x", "decision": "maybe"})
    with pytest.raises(ValueError):
        rq.parse_decision({"sample_id": "x", "decision": "relabel", "label": ""})
    assert rq.parse_decision({"sample_id": "x", "decision": ""}) is None
    with pytest.raises(ValueError):
        rq.import_sheet(tmp_path, tmp_path / "none.csv", " ")
