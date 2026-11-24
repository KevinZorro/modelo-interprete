"""Delete / space gestures from the normalized landmark vector (T-22).

The classifier has no classes for them (no data, and retraining is out of scope), so they come
from hand geometry on the SAME 63-float vector the model sees (contract v1: wrist at origin,
scale = max norm). They are checked before the classifier: the shapes were chosen so that no
letter has them, and ``python -m lsc70.gestures`` measures how often letters trigger them.

- DELETE = open hand, five fingers extended and spread (the "5").
- SPACE  = thumb, index and pinky extended with middle and ring curled ("te quiero").

Thresholds live in configs/spelling.json ("gestures") so they can be tuned without recompiling.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lsc70.spelling import CONFIG_PATH, GESTURE_DELETE, GESTURE_SPACE

TIP_PIP = {"index": (8, 6), "middle": (12, 10), "ring": (16, 14), "pinky": (20, 18)}


@dataclass
class GestureConfig:
    extended_min: float = 1.2    # tip-to-wrist / pip-to-wrist at or above this = extended
    curled_max: float = 1.0      # at or below this = curled
    spread_min: float = 0.25     # mean distance between adjacent fingertips (open hand only)
    thumb_out_min: float = 1.0   # thumb tip from index base / palm width

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "GestureConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8")).get("gestures", {})
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


def _d(a, b) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1)


def features(x: np.ndarray) -> dict:
    """x: (..., 21, 3) normalized landmarks."""
    w = x[..., 0, :]
    f = {n: _d(x[..., t, :], w) / np.maximum(_d(x[..., p, :], w), 1e-6) for n, (t, p) in TIP_PIP.items()}
    f["spread"] = (_d(x[..., 8, :], x[..., 12, :]) + _d(x[..., 12, :], x[..., 16, :])
                   + _d(x[..., 16, :], x[..., 20, :])) / 3
    f["thumb_out"] = _d(x[..., 4, :], x[..., 5, :]) / np.maximum(_d(x[..., 5, :], x[..., 17, :]), 1e-6)
    return f


def masks(x: np.ndarray, cfg: GestureConfig) -> tuple[np.ndarray, np.ndarray]:
    f = features(x)
    thumb = f["thumb_out"] >= cfg.thumb_out_min
    ext = {n: f[n] >= cfg.extended_min for n in TIP_PIP}
    opened = ext["index"] & ext["middle"] & ext["ring"] & ext["pinky"] & (f["spread"] >= cfg.spread_min) & thumb
    ily = (ext["index"] & ext["pinky"] & (f["middle"] <= cfg.curled_max)
           & (f["ring"] <= cfg.curled_max) & thumb)
    return opened, ily


def classify_gesture(vector63, cfg: GestureConfig | None = None) -> str | None:
    """Return GESTURE_DELETE, GESTURE_SPACE or None for one normalized 63-float vector."""
    cfg = cfg or GestureConfig.load()
    opened, ily = masks(np.asarray(vector63, dtype=float).reshape(21, 3), cfg)
    if bool(ily):
        return GESTURE_SPACE
    if bool(opened):
        return GESTURE_DELETE
    return None


def main() -> None:
    """Share of landmark frames of each class that would trigger a gesture (collision check)."""
    d = np.load(Path(__file__).resolve().parent.parent / "landmarks_cache.npz", allow_pickle=True)
    x = np.concatenate([d["Xtr"], d["Xte"]]).reshape(-1, 21, 3)
    y = np.concatenate([d["ytr"], d["yte"]])
    labels = [str(v) for v in d["etiquetas"]]
    opened, ily = masks(x, GestureConfig.load())
    rows = {}
    for name, m in (("DELETE", opened), ("SPACE", ily)):
        rows[name] = {"all_frames": round(float(m.mean()), 4),
                      "worst_classes": {labels[i]: round(float(m[y == i].mean()), 3)
                                        for i in range(len(labels)) if (y == i).any() and m[y == i].mean() > 0}}
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
