"""Names that share one hand pose (F/1, V/2, W/3): the model predicts the pose, not the name.

The pose-level class is the first name of each group. A prediction counts as correct when
it falls in the same group as the truth; the app picks the displayed name from its mode
(letters or numbers).
"""
import json
from pathlib import Path
from typing import Optional

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "alias_poses.json"


def load_groups(path: Path = CONFIG) -> list:
    groups = json.loads(path.read_text(encoding="utf-8"))["groups"]
    seen = set()
    for g in groups:
        if len(g) < 2 or seen & set(g):
            raise ValueError(f"alias groups must have >= 2 names and no overlaps: {g}")
        seen |= set(g)
    return groups


def canonical(label: Optional[str], groups: Optional[list] = None) -> Optional[str]:
    """Pose-level class of `label`; labels outside any group are their own class."""
    if label is None:
        return None
    for g in (groups if groups is not None else load_groups()):
        if label in g:
            return g[0]
    return label


def same_pose(a: str, b: str, groups: Optional[list] = None) -> bool:
    groups = groups if groups is not None else load_groups()
    return canonical(a, groups) == canonical(b, groups)


def display_names(label: str, groups: Optional[list] = None) -> str:
    """"F" -> "F / 1" for the review page; unaliased labels come back unchanged."""
    for g in (groups if groups is not None else load_groups()):
        if label in g:
            return " / ".join(g)
    return label


def pose_accuracy(y_true, y_pred, groups: Optional[list] = None) -> float:
    groups = groups if groups is not None else load_groups()
    hits = sum(same_pose(t, p, groups) for t, p in zip(y_true, y_pred))
    return hits / max(len(y_true), 1)
