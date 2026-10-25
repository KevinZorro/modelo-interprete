"""Frozen test set and development folds (T-03).

The 15-person test of splits_participantes.json is kept (same rule as the notebook, so history
stays comparable) but it is declared CONTAMINATED: it was used many times for comparisons.
It is frozen by hash so nobody changes it silently, and every use is logged. Model selection
must use `dev_fold_de` (5 folds over the 55 train people only), never the 70-person folds,
which include the test people.

    python -m lsc70.frozen_split            # create configs/test_freeze.json if missing
    python -m lsc70.frozen_split --verify   # exit 1 if the test list no longer matches
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
SPLITS = CONFIGS / "splits_participantes.json"
FREEZE = CONFIGS / "test_freeze.json"
DEV_SEED = 43
N_DEV_FOLDS = 5


def digest(participants):
    return hashlib.sha256("\n".join(sorted(participants)).encode()).hexdigest()


def dev_folds(train, seed=DEV_SEED, n_folds=N_DEV_FOLDS):
    people = sorted(train)
    order = np.random.default_rng(seed).permutation(len(people))
    return {people[i]: int(r % n_folds) for r, i in enumerate(order)}


def build(splits):
    return {
        "status": "contaminated",
        "note": (
            "Test of 15 people, already used repeatedly before T-03. Frozen by hash; "
            "use it once per model version and append the use to usage_log. "
            "A virgin test needs new people (T-15)."
        ),
        "test": sorted(splits["test"]),
        "test_sha256": digest(splits["test"]),
        "dev_seed": DEV_SEED,
        "dev_fold_de": dev_folds(splits["train"]),
        "usage_log": [],
    }


def load_freeze(path=FREEZE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(splits_path=SPLITS, freeze_path=FREEZE):
    """Returns a list of problems; empty means the test is still the frozen one."""
    s = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    f = load_freeze(freeze_path)
    problems = []
    if digest(s["test"]) != f["test_sha256"]:
        problems.append("test list in splits_participantes.json differs from the frozen hash")
    if sorted(f["test"]) != sorted(s["test"]):
        problems.append("test_freeze.json test list differs from splits_participantes.json")
    if set(f["dev_fold_de"]) != set(s["train"]):
        problems.append("dev folds do not cover exactly the train people")
    if set(f["dev_fold_de"]) & set(s["test"]):
        problems.append("a test person appears in the dev folds")
    return problems


def log_use(version, commit, purpose, date, freeze_path=FREEZE):
    f = load_freeze(freeze_path)
    f["usage_log"].append(
        {"version": version, "commit": commit, "purpose": purpose, "date": date}
    )
    Path(freeze_path).write_text(
        json.dumps(f, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        problems = verify()
        print("\n".join(problems) or "test frozen and consistent")
        sys.exit(1 if problems else 0)
    if FREEZE.exists():
        sys.exit(f"{FREEZE} already exists; the freeze is not rebuilt.")
    s = json.loads(SPLITS.read_text(encoding="utf-8"))
    FREEZE.write_text(
        json.dumps(build(s), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {FREEZE}")


if __name__ == "__main__":
    main()
