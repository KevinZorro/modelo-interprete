import json

import numpy as np
import pytest

from lsc70 import label_eval as le
from lsc70 import review_queue as rq


@pytest.mark.parametrize("title,query_kind,kind", [
    ("Abecedario en LSC", None, "letters"), ("Números del 1 al 10 en lengua de señas", None, "numbers"),
    ("Abecedario y números LSC", None, "mixed"), (None, "numeros", "numbers"),
    ("Clase 1 de señas", "lsc", "letters"), (None, None, "mixed")])
def test_video_kind(title, query_kind, kind):
    assert le.video_kind(title, query_kind) == kind


def test_norm():
    assert [le.norm(x) for x in ("ñ", "1", "b", "none", "10")] == ["NN", "F", "B", "none", "10"]


def _unit(v):
    return v / np.linalg.norm(v)


def _refs(tmp_path, proto, rng):
    """ANH-like cache: letters, plus a `none` class that holds the numbers 1, 4 and 5."""
    names = le.LETTERS + ["none"]
    X, y, files = [], [], []
    for c in le.LETTERS + ["1", "4", "5"]:
        for k in range(5):
            X.append((proto["F"] if c == "1" else proto[c]) + 0.01 * rng.normal(size=63))
            y.append(names.index(c) if c in le.LETTERS else names.index("none"))
            files.append(f"Per0{k}__{c}__Per0{k}_{c}_0.jpg")
    np.savez(tmp_path / "refs.npz", Xtr=np.asarray(X), ytr=np.asarray(y), ntr=np.asarray(files),
             etiquetas=np.array(names))


def test_load_refs_splits_the_numbers_out_of_none(tmp_path):
    rng = np.random.default_rng(0)
    proto = {c: _unit(rng.normal(size=63)) for c in le.LETTERS + ["4", "5"]}
    _refs(tmp_path, proto, rng)
    assert "none" in rq.load_refs(tmp_path / "refs.npz")[1]
    cent, names = rq.load_refs(tmp_path / "refs.npz", numbers=True)
    assert "none" not in names and {"4", "5"} <= set(names) and "1" not in names   # 1 is the F pose
    assert len(names) == len(le.LETTERS) + 2


def test_evaluate_numbers_and_kind(tmp_path):
    rng = np.random.default_rng(1)
    proto = {c: _unit(rng.normal(size=63)) for c in le.LETTERS + ["4", "5"]}
    _refs(tmp_path, proto, rng)
    between = _unit(0.52 * proto["B"] + 0.48 * proto["4"])      # a 4 that looks a bit more like a B
    videos = {"L1": ("Abecedario LSC", [("A", proto["A"]), ("B", proto["B"])]),
              "N1": ("Números en LSC", [("4", proto["4"]), ("5", proto["5"]), ("1", proto["F"]), ("4", between)])}
    rows, reviews, cands = [], [], []
    for vid, (title, signs) in videos.items():
        cands.append({"video_id": vid, "title": title})
        for i, (lab, f) in enumerate(signs):
            rows.append({"video_id": vid, "frame": i * 10, "t_s": i, "query_kind": "lsc",
                         "features": list(f + 0.002 * rng.normal(size=63))})
            reviews.append({"sample_id": f"{vid}:{i * 10}", "decision": "relabel", "label": lab})
    for name, data in (("landmarks.jsonl", rows), ("reviews.jsonl", reviews), ("candidates.jsonl", cands)):
        (tmp_path / name).write_text("".join(json.dumps(x) + "\n" for x in data), encoding="utf-8")
    rep = le.evaluate(tmp_path, tmp_path / "refs.npz")
    assert rep["truth"]["n"] == 6
    m = rep["methods"]
    assert m["centroid"]["all"]["top1"] == 0.5            # A, B and the 1 (= F); no 4 or 5 class
    assert m["centroid+kind"]["all"]["top1"] == 0.5
    assert m["numbers"]["all"]["top1"] == pytest.approx(5 / 6, abs=1e-3)   # the look-alike 4 is a B
    assert m["numbers+kind"]["all"]["top1"] == 1.0         # a number video is not offered a B
    assert m["numbers+kind"]["by_kind"]["numbers"]["n"] == 4
