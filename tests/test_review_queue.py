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
             "query_kind": "lsc", "is_left": False, "landmarks": [[0.0, 0.0, 0.0]] * 21, "features": ([1.0] + [0.0] * 62) if i else ([0.0, 1.0] + [0.0] * 61),
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
    assert rq.build(tmp_path, tmp_path / "refs.npz") == 3         # a rebuild keeps reviewed samples...
    assert all(i["decision"] is None or i["decision"]["decision"] != "undo" for i in rq.review_items(tmp_path))


def test_bad_decisions_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        rq.parse_decision({"sample_id": "x", "decision": "maybe"})
    with pytest.raises(ValueError):
        rq.parse_decision({"sample_id": "x", "decision": "relabel", "label": ""})
    assert rq.parse_decision({"sample_id": "x", "decision": ""}) is None
    with pytest.raises(ValueError):
        rq.import_sheet(tmp_path, tmp_path / "none.csv", " ")


def test_dedupe_keeps_one_or_two_per_segment():
    def row(i, vec):
        return {"frame": i, "features": vec}
    a, b = [1.0] + [0.0] * 62, [0.0, 1.0] + [0.0] * 61
    rows = [row(i, a) for i in range(3)] + [row(3 + i, b) for i in range(10)]
    kept = rq.dedupe_segments(rows)
    assert [r["frame"] for r in kept] == [1, 3, 8]      # middle of A; first+middle of long B
    assert rq.dedupe_segments([]) == []


def test_undo_cancels_and_review_items_order(tmp_path):
    _write_run(tmp_path, n=3)
    rq.build(tmp_path, tmp_path / "refs.npz")
    sid = "v:20"
    rq.record_review(tmp_path, sid, "relabel", "7", "Kevin")
    assert rq.latest_reviews(tmp_path)[sid]["label"] == "7"
    assert [i["decision"] is not None for i in rq.review_items(tmp_path) if i["sample_id"] == sid] == [True]
    rq.record_review(tmp_path, sid, "undo", None, "Kevin")
    assert sid not in rq.latest_reviews(tmp_path)
    assert rq.build(tmp_path, tmp_path / "refs.npz") == 3
    with pytest.raises(ValueError):
        rq.record_review(tmp_path, sid, "relabel", "", "Kevin")


def test_rebuild_keeps_review_progress(tmp_path):
    _write_run(tmp_path, n=3)
    rq.build(tmp_path, tmp_path / "refs.npz")
    rq.record_review(tmp_path, "v:20", "accept", "A", "Kevin")
    rq.build(tmp_path, tmp_path / "refs.npz")
    done = [i for i in rq.review_items(tmp_path) if i["decision"]]
    assert [d["sample_id"] for d in done] == ["v:20"] and done[0]["decision"]["label"] == "A"


def test_export_adds_pose_class_for_aliases(tmp_path):
    _write_run(tmp_path, n=3)
    rq.build(tmp_path, tmp_path / "refs.npz")
    assert all("suggested_names" in q for q in rq.read_jsonl(tmp_path / "review_queue.jsonl"))
    rq.record_review(tmp_path, "v:20", "relabel", "1", "Kevin")
    rq.export_validated(tmp_path)
    val = rq.read_jsonl(tmp_path / "validated.jsonl")
    assert val[0]["label"] == "1" and val[0]["pose"] == "F"


def test_dynamic_flag_follows_confirmed_sign_list(tmp_path):
    assert rq.load_dynamic_signs() == {"6", "7", "8", "9", "10", "G", "H", "J", "Ñ", "S", "Z"}
    _write_run(tmp_path, n=3)
    rq.build(tmp_path, tmp_path / "refs.npz")
    items = rq.review_items(tmp_path)
    assert items and all(i["dynamic"] == (i["suggested"] in rq.load_dynamic_signs()) for i in items)


def test_held_seconds_and_teaching_flag(tmp_path):
    a, b = [1.0] + [0.0] * 62, [0.0, 1.0] + [0.0] * 61
    # pose A held 4 samples 0.33 s apart (+ one step = 1.33 s), then a lone B 1 s later
    rows = [{"frame": i * 10, "t_s": i / 3, "features": a} for i in range(4)]
    rows += [{"frame": 40, "t_s": 1.67, "features": b}]
    kept = rq.dedupe_segments(rows)
    assert [round(k["held_s"], 2) for k in kept] == [1.33, 0.33]
    assert kept[0]["held_s"] >= rq.HOLD_S > kept[1]["held_s"]
    far = rows[:2] + [{"frame": 300, "t_s": 10.0, "features": a}]
    assert [k["held_s"] for k in rq.dedupe_segments(far)][-1] < rq.HOLD_S   # gap ends the segment


def test_track_around_keeps_only_the_loop_window():
    lm = [[0.1 * i, 0.2, 0.0] for i in range(21)]
    rows = [{"frame": i * 10, "t_s": float(t), "landmarks": lm} for i, t in enumerate([0, 3, 4, 5, 6, 9])]
    tr = rq.track_around(rows, 4.0)
    assert [x[0] for x in tr] == [3.0, 4.0, 5.0, 6.0]            # 4-1 .. 4+2.5
    assert len(tr[0][1]) == 21 and len(tr[0][1][0]) == 2          # x, y only


def test_decided_sample_stays_visible_even_if_dedupe_skips_it(tmp_path):
    _write_run(tmp_path, n=3)
    rows = rq.read_jsonl(tmp_path / "landmarks.jsonl")
    for r in rows:
        r["t_s"] = r["frame"] / 30      # close enough in time to share a segment
    (tmp_path / "landmarks.jsonl").write_text("
".join(json.dumps(r) for r in rows), encoding="utf-8")
    rq.build(tmp_path, tmp_path / "refs.npz")
    shown = {i["sample_id"] for i in rq.review_items(tmp_path)}
    skipped = ({"v:0", "v:10", "v:20"} - shown).pop()
    rq.record_review(tmp_path, skipped, "accept", "A", "Kevin")
    assert skipped in {i["sample_id"] for i in rq.review_items(tmp_path)}
