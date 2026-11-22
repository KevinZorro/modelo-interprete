import json

import numpy as np
import pytest

from lsc70 import label_eval as le


def test_order_pass_fixes_a_letter_out_of_order():
    order = le.LETTERS
    D = np.full((4, len(order)), 0.3)
    for t, c in enumerate("ABCD"):
        D[t, order.index(c)] = 0.05
    D[2, order.index("S")] = 0.04                       # the nearest centroid says S, not C
    assert [order[p] for p in le.viterbi_order(D, 0.0, 0.0)] == ["A", "B", "S", "D"]
    assert [order[p] for p in le.viterbi_order(D, 0.1, 3.0)] == ["A", "B", "C", "D"]


@pytest.mark.parametrize("title,query_kind,kind", [
    ("Abecedario en LSC", None, "letters"), ("Números del 1 al 10 en lengua de señas", None, "numbers"),
    ("Abecedario y números LSC", None, "mixed"), (None, "numeros", "numbers"),
    ("Clase 1 de señas", "lsc", "letters"), (None, None, "mixed")])
def test_video_kind(title, query_kind, kind):
    assert le.video_kind(title, query_kind) == kind


def test_norm_and_order_moves():
    assert [le.norm(x) for x in ("ñ", "1", "b", "none", "10")] == ["NN", "F", "B", "none", "10"]
    assert le.order_moves(["A", "B", "B", "D", "C"], le.LETTERS) == {"forward": 2, "same": 1, "back": 1}


def _unit(v):
    return v / np.linalg.norm(v)


def _write(tmp_path):
    """Two alphabet videos (A..E) and two number videos (4, 5); in the first alphabet video
    the C is closer to the S centroid. Every hold's middle sample is labelled."""
    rng = np.random.default_rng(0)
    proto = {c: _unit(rng.normal(size=63)) for c in le.LETTERS + ["4", "5", "none"]}
    names = le.LETTERS + ["none"]
    Xtr = np.concatenate([[proto[c] + 0.01 * rng.normal(size=63) for _ in range(5)] for c in names])
    ytr = np.repeat(np.arange(len(names)), 5)
    np.savez(tmp_path / "refs.npz", Xtr=Xtr, ytr=ytr, etiquetas=np.array(names))
    videos = {"L1": ("Abecedario LSC", "ABCDE"), "L2": ("Abecedario LSC", "ABCDE"),
              "N1": ("Números en LSC", ["4", "5"]), "N2": ("Números en LSC", ["4", "5"])}
    rows, reviews, cands = [], [], []
    for vid, (title, signs) in videos.items():
        cands.append({"video_id": vid, "title": title})
        i = 0
        for h, c in enumerate(signs):
            f = _unit(0.48 * proto["C"] + 0.52 * proto["S"]) if (vid, c) == ("L1", "C") else proto[c]
            for k in range(3):
                rows.append({"video_id": vid, "frame": i * 10, "t_s": round(i / 3, 3),
                             "query_kind": "lsc", "features": list(f + 0.002 * rng.normal(size=63)),
                             "landmarks": [[0.1 + 0.4 * h + 0.01 * j, 0.5 + 0.01 * j, 0.0] for j in range(21)]})
                if k == 1 and (vid, c) != ("N2", "5"):          # the second 5 is never labelled
                    reviews.append({"sample_id": f"{vid}:{i * 10}", "decision": "relabel", "label": c})
                i += 1
    for name, data in (("landmarks.jsonl", rows), ("reviews.jsonl", reviews), ("candidates.jsonl", cands)):
        (tmp_path / name).write_text("".join(json.dumps(x) + "\n" for x in data), encoding="utf-8")


def test_evaluate_on_synthetic_lessons(tmp_path):
    _write(tmp_path)
    rep = le.evaluate(tmp_path, tmp_path / "refs.npz")
    assert rep["truth"]["n"] == 13
    assert rep["truth"]["by_kind"] == {"letters": {"letter": 10, "other": 0}, "numbers": {"letter": 0, "other": 3}}
    assert rep["order_check"]["labelled_consecutive_pairs"] == {"forward": 9, "same": 0, "back": 0}
    cen = rep["methods"]["centroid"]
    assert cen["letters"]["top1"] == 0.9 and cen["letters"]["top3"] == 1.0 and cen["other"]["top1"] == 0.0
    # the reviewer's 4 from the other number video names the 4; the unpaired 5 has no reference
    assert rep["methods"]["domain"]["other"]["top1"] == pytest.approx(2 / 3, abs=1e-3)
    grid = rep["methods"]["order"]["in_sample_top1"]
    assert grid["0.1/3.0"] > grid["0.0/0.0"]
    # with the 4 referenced, the order pass names the next hold 5
    assert rep["methods"]["order+domain"]["in_sample_top1"]["0.1/3.0"] == 1.0
