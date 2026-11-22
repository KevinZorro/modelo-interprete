import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from synthetic_queue import make_videos  # noqa: E402

from lsc70 import queue_eval as qe  # noqa: E402
from lsc70 import review_queue as rq  # noqa: E402

CACHE = Path("landmarks_cache.npz")


def test_truth_and_correctness_follow_aliases_and_n_tilde():
    assert qe.truth_of({"decision": "reject", "label": None}) == qe.REJECT
    assert qe.truth_of({"decision": "relabel", "label": "ñ"}) == "NN"
    assert qe.truth_of({"decision": "relabel", "label": "1"}) == "F"          # F / 1 share a pose
    assert qe.is_correct("F", qe.truth_of({"decision": "accept", "label": "1"}))
    assert not qe.is_correct("A", "none") and not qe.is_correct("A", qe.REJECT)


def test_wilson_interval_brackets_the_rate():
    lo, hi = qe.wilson(45, 50)
    assert lo < 0.9 < hi and 0 <= lo and hi <= 1
    assert qe.wilson(0, 0) == [None, None]


def test_threshold_is_chosen_on_other_videos():
    table = [(f"v{i % 10}", i % 4 != 0, 0.5 + 0.05 * (i % 10), False) for i in range(400)]
    cv = qe.cross_validated(table, qe.GRID, target=0.7)
    assert cv["shown"] > 0 and len(cv["thresholds_per_fold"]) == qe.N_FOLDS


@pytest.mark.skipif(not CACHE.exists(), reason="needs landmarks_cache.npz")
def test_evaluate_on_a_simulated_reviewer(tmp_path):
    from lsc70 import mlp_numpy
    cls = {}
    rows, truth = make_videos(CACHE, n_videos=40, seed=7, fast_share=0.5, classes=cls)
    (tmp_path / "landmarks.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    rq.build(tmp_path, CACHE, mlp_numpy.NumpyMLP())
    queue = {q["sample_id"]: q for q in rq.read_jsonl(tmp_path / "review_queue.jsonl")}
    rng = np.random.default_rng(0)
    for sid in rng.choice(sorted(queue), size=300, replace=False):
        q, c = queue[sid], cls.get(sid)
        if c is None:
            dec, lab = ("reject", None) if rng.random() < 0.5 else ("relabel", "none")
        elif qe.norm(q["suggested"]) == qe.norm(c):
            dec, lab = "accept", q["suggested"]
        else:
            dec, lab = "relabel", c
        rq.record_review(tmp_path, sid, dec, lab, "sim")
    rep = qe.evaluate(tmp_path)
    assert rep["decided"]["n"] == 300
    sign = rep["methods"]["sign"]
    assert sign["cv"]["precision"] is not None and sign["cv"]["precision"] >= 0.8
    assert sign["curve"][-1]["shown"] <= sign["curve"][0]["shown"]
    assert set(rep["loops"]) == {"fixed_3.5s_window", "spotted_window"}
    assert "0.8" in rep["pending_volume_by_ready_p"]
