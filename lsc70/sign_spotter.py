"""Find the signs in a scraped video before a person reviews it.

A letter shown in a lesson is a stretch of samples where the hand stays put and the T-08
classifier keeps naming the same letter. `spot` turns one video's samples (sorted by time)
into those segments: the letter, how sure the classifier is over the whole segment (not one
frame), the runner-up, and where the sign starts and ends. The review page loops exactly that
stretch, so one loop shows one sign instead of the three a fixed 3.5 s window used to show.

Nothing here labels data: the letter is a suggestion that a person accepts or corrects.
"""
from typing import Optional

import numpy as np

from lsc70.review_queue import MAX_GAP_S, STILL_MOVE, _drift, _t

NONE = "none"
BRIDGE_MIN_P = 0.15  # a one-sample flicker (A X A) is bridged if the run's letter keeps this probability
MIN_LOOP_S = 0.8     # shortest loop: under this a person cannot see the hand at normal speed
DYN_PRE_S, DYN_POST_S = 0.4, 0.6   # moving signs: show how the hand arrives and leaves


def _sampling_step(times: list) -> float:
    gaps = [b - a for a, b in zip(times, times[1:]) if 0 < b - a <= MAX_GAP_S]
    return float(np.median(gaps)) if gaps else 0.0


def bridge_flickers(lab: np.ndarray, P: np.ndarray, times: list) -> np.ndarray:
    """A X A -> A A A when X is one sample and A still has BRIDGE_MIN_P there: at 3 samples/s a
    one-sample change inside a hold is classifier noise, not a 0.3 s letter."""
    lab = lab.copy()
    for i in range(1, len(lab) - 1):
        a = lab[i - 1]
        if (a == lab[i + 1] != lab[i] and times[i + 1] - times[i - 1] <= 2 * MAX_GAP_S
                and P[i, a] >= BRIDGE_MIN_P):
            lab[i] = a
    return lab


def spot(rows: list, P: np.ndarray, labels: list, still_move: float = STILL_MOVE) -> list:
    """Segments of one video: runs of samples with the same predicted letter, no detection gap
    above MAX_GAP_S and the hand within `still_move` hand sizes of where the run started.

    `rows` are sorted by time, `P` holds the classifier probabilities for each row. Runs whose
    letter is `none` are dropped. Each segment is a dict with the row indices, the letter, the
    mean and minimum probability of that letter over the run, the runner-up of the mean
    probabilities, the representative row (most confident), first/last time and `held_s`.
    """
    if not rows:
        return []
    times = [_t(r) for r in rows]
    step = _sampling_step(times)
    lab = bridge_flickers(np.asarray(P).argmax(axis=1), P, times)
    segs, start = [], 0
    for i in range(1, len(rows) + 1):
        if (i < len(rows) and lab[i] == lab[start] and times[i] - times[i - 1] <= MAX_GAP_S
                and _drift(rows[start], rows[i]) <= still_move):
            continue
        k = int(lab[start])
        if labels[k] != NONE:
            segs.append(_summary(list(range(start, i)), k, P, labels, times, step))
        start = i
    return segs


def _summary(idx: list, k: int, P: np.ndarray, labels: list, times: list, step: float) -> dict:
    pk = P[idx, k]
    mean = P[idx].mean(axis=0)
    rival = mean.copy()
    rival[k] = -1.0
    s = int(rival.argmax())
    return {"rows": idx, "label": labels[k], "rep": idx[int(pk.argmax())], "n": len(idx),
            "p_mean": round(float(pk.mean()), 4), "p_min": round(float(pk.min()), 4),
            "second": labels[s], "p_second": round(float(mean[s]), 4),
            "t_first": round(times[idx[0]], 3), "t_last": round(times[idx[-1]], 3),
            "held_s": round(times[idx[-1]] - times[idx[0]] + step, 3), "step": round(step, 3)}


def loop_window(seg: dict, dynamic: bool = False) -> tuple:
    """(start, end) seconds the player should loop: the sign itself, padded by half a sampling
    step on each side (the hold began and ended somewhere between samples), at least
    MIN_LOOP_S long; moving signs also show the hand arriving and leaving."""
    half = seg["step"] / 2 if seg["step"] else 0.15
    start, end = seg["t_first"] - half, seg["t_last"] + half
    if dynamic:
        start, end = start - DYN_PRE_S, end + DYN_POST_S
    if end - start < MIN_LOOP_S:
        grow = (MIN_LOOP_S - (end - start)) / 2
        start, end = start - grow, end + grow
    return round(max(0.0, start), 3), round(end, 3)


def signs_in_window(times: list, lab: list, P: np.ndarray, start: float, end: float,
                    min_p: float = 0.5) -> int:
    """How many different confident letters (two or more samples, p >= min_p) a loop shows.
    This is what the reviewer sees: 1 means the loop shows one sign."""
    seen, run_lab, run_n = set(), None, 0
    for t, k, p in zip(times, lab, P.max(axis=1) if len(P) else []):
        if not start <= t <= end:
            continue
        if p < min_p:
            run_lab, run_n = None, 0
            continue
        run_n = run_n + 1 if k == run_lab else 1
        run_lab = k
        if run_n >= 2:
            seen.add(k)
    return len(seen)


def predict_rows(rows: list, model) -> Optional[np.ndarray]:
    """Classifier probabilities for each row's 63 features, or None without a model."""
    if model is None or not rows:
        return None
    return model.predict_proba(np.asarray([r["features"] for r in rows], dtype=np.float64))
