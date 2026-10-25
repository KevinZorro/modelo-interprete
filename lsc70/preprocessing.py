"""Single entry point for landmark cleaning + normalization (contract v1).

Everything that turns MediaPipe output into the 63 floats the classifier eats goes
through `preprocess`. The math is NOT reimplemented: it delegates to
`normalizacion.vector_features`, the verbatim notebook copy, so the existing
notebook-equality test keeps guarding it. This module only adds the cleaning
(validation) in front and a typed result with an explicit reject reason.

Contract (see docs/contrato-entrada.md):
  input : 21 landmarks (objects with .x/.y/.z, or an array-like of shape (21, 3)) in
          MediaPipe image-normalized coordinates, plus the handedness flag.
  output: `Result(features=float32[63], extras=float32[3])` or `Result(reason=...)`.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from lsc70.normalizacion import vector_features

CONTRACT_VERSION = "v1"
NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3
# Slack around [0, 1] for x/y: MediaPipe can return points slightly outside the image
# when a finger is cut off by the frame edge. Anything beyond this is corrupt input.
COORD_MARGIN = 0.5
Z_ABS_MAX = 5.0

REASONS = ("bad_shape", "non_finite", "out_of_range", "degenerate_scale")


@dataclass(frozen=True)
class Result:
    features: Optional[np.ndarray] = None   # float32 (63,), None if rejected
    extras: Optional[np.ndarray] = None     # float32 (3,): wrist_x, wrist_y, scale
    reason: Optional[str] = None            # one of REASONS if rejected

    @property
    def ok(self) -> bool:
        return self.features is not None


class _Point:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)


def _as_array(landmarks) -> Optional[np.ndarray]:
    try:
        if len(landmarks) and hasattr(landmarks[0], "x"):
            arr = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float64)
        else:
            arr = np.asarray(landmarks, dtype=np.float64)
    except (TypeError, ValueError, AttributeError):
        return None
    return arr if arr.shape == (NUM_LANDMARKS, 3) else None


def preprocess(landmarks, is_left: bool) -> Result:
    """Clean and normalize one hand. Never raises on bad input; returns a reason."""
    arr = _as_array(landmarks)
    if arr is None:
        return Result(reason="bad_shape")
    if not np.isfinite(arr).all():
        return Result(reason="non_finite")
    xy = arr[:, :2]
    if (xy < -COORD_MARGIN).any() or (xy > 1 + COORD_MARGIN).any() or (
        np.abs(arr[:, 2]) > Z_ABS_MAX
    ).any():
        return Result(reason="out_of_range")
    features, extras = vector_features([_Point(*p) for p in arr], bool(is_left))
    if features is None:
        return Result(reason="degenerate_scale")
    return Result(features=features.astype(np.float32), extras=extras.astype(np.float32))
