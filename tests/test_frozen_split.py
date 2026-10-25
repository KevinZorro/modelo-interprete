"""T-03: the test set is frozen, dev folds never touch it, baseline metrics are sane."""

import json
from pathlib import Path

import numpy as np

from lsc70 import baseline, frozen_split

RAIZ = Path(__file__).resolve().parent.parent


def test_frozen_test_matches_splits():
    assert frozen_split.verify() == []


def test_cache_test_people_are_the_frozen_test():
    d = np.load(RAIZ / "landmarks_cache.npz", allow_pickle=True)
    assert set(d["pte"].astype(str)) == set(frozen_split.load_freeze()["test"])
    assert not set(d["ptr"].astype(str)) & set(frozen_split.load_freeze()["test"])


def test_verify_detects_a_changed_test(tmp_path):
    s = json.loads(frozen_split.SPLITS.read_text(encoding="utf-8"))
    moved = s["train"][0]
    s["train"] = s["train"][1:]
    s["test"] = sorted(s["test"][1:] + [moved])
    p = tmp_path / "splits.json"
    p.write_text(json.dumps(s), encoding="utf-8")
    assert frozen_split.verify(splits_path=p)


def test_dev_folds_cover_train_only_and_are_balanced():
    f = frozen_split.load_freeze()
    s = json.loads(frozen_split.SPLITS.read_text(encoding="utf-8"))
    assert set(f["dev_fold_de"]) == set(s["train"])
    sizes = np.bincount(list(f["dev_fold_de"].values()))
    assert len(sizes) == frozen_split.N_DEV_FOLDS and sizes.max() - sizes.min() <= 1
    assert frozen_split.dev_folds(s["train"]) == f["dev_fold_de"]


def test_macro_f1_and_confusion():
    true = np.array([0, 0, 1, 1, 2])
    pred = np.array([0, 1, 1, 1, 3])  # class 2 rejected (3 = none)
    f1, per = baseline.macro_f1(pred, true, 3)
    assert per[0]["recall"] == 0.5 and per[1]["precision"] == 2 / 3 and per[2]["f1"] == 0.0
    assert abs(f1 - np.mean([per[0]["f1"], per[1]["f1"], 0.0])) < 1e-9
    M = baseline.confusion(pred, true, 4)
    assert M.sum() == 5 and M[2, 3] == 1


def test_wilson_contains_point_estimate():
    lo, hi = baseline.wilson(5, 100)
    assert lo < 0.05 < hi
