import pytest

from lsc70 import alias_poses as ap


def test_config_has_confirmed_groups():
    assert ap.load_groups() == [["F", "1"], ["V", "2"], ["W", "3"]]


def test_canonical_and_same_pose():
    assert ap.canonical("1") == "F" and ap.canonical("F") == "F" and ap.canonical("A") == "A"
    assert ap.canonical(None) is None
    assert ap.same_pose("2", "V") and ap.same_pose("W", "3") and not ap.same_pose("F", "V")
    assert ap.display_names("V") == "V / 2" and ap.display_names("B") == "B"


def test_pose_accuracy_counts_alias_as_correct():
    assert ap.pose_accuracy(["1", "2", "A"], ["F", "B", "A"]) == pytest.approx(2 / 3)


def test_overlapping_groups_are_rejected(tmp_path):
    p = tmp_path / "a.json"
    p.write_text('{"groups": [["F", "1"], ["1", "V"]]}', encoding="utf-8")
    with pytest.raises(ValueError):
        ap.load_groups(p)
