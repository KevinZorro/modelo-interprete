import numpy as np
import pytest

from lsc70 import sign_spotter as ss

LABELS = ["A", "B", "none"]


def _rows(n, pos=None):
    pos = pos or [0.5] * n
    return [{"frame": i * 10, "t_s": round(i / 3, 3), "features": [1.0] + [0.0] * 62,
             "landmarks": [[pos[i] + 0.01 * k, 0.5 + 0.01 * k, 0.0] for k in range(21)]}
            for i in range(n)]


def _probs(seq, p=0.9, flicker_keep=0.3):
    """One row of probabilities per letter in `seq`; 'x' is a one-sample flicker to B that
    keeps `flicker_keep` on the letter before it."""
    P = []
    for i, c in enumerate(seq):
        row = np.full(len(LABELS), (1 - p) / (len(LABELS) - 1))
        if c == "x":
            prev = LABELS.index(seq[i - 1])
            row[:] = 0.0
            row[prev], row[1] = flicker_keep, 1 - flicker_keep
        else:
            row[LABELS.index(c)] = p
        P.append(row)
    return np.asarray(P)


def test_spot_bridges_one_sample_flicker_and_drops_none():
    seq = ["A", "A", "A", "x", "A", "B", "B", "none"]
    segs = ss.spot(_rows(len(seq)), _probs(seq), LABELS)
    assert [(s["label"], s["n"]) for s in segs] == [("A", 5), ("B", 2)]
    a = segs[0]
    assert a["p_min"] == pytest.approx(0.3) and a["p_mean"] < 0.9      # the flicker lowers confidence
    assert a["held_s"] == pytest.approx(4 / 3 + 1 / 3, abs=1e-3)
    assert a["rep"] in a["rows"] and a["second"] == "B"


def test_flicker_with_no_trace_of_the_letter_is_not_bridged():
    seq = ["A", "A", "x", "A"]
    segs = ss.spot(_rows(4), _probs(seq, flicker_keep=0.05), LABELS)
    assert [(s["label"], s["n"]) for s in segs] == [("A", 2), ("B", 1), ("A", 1)]


def test_travelling_hand_splits_the_sign():
    seq = ["A"] * 6
    still = ss.spot(_rows(6), _probs(seq), LABELS)
    moving = ss.spot(_rows(6, pos=[0.1 + 0.15 * i for i in range(6)]), _probs(seq), LABELS)
    assert len(still) == 1 and len(moving) == 6


def test_detection_gap_ends_the_sign():
    rows = _rows(4)
    rows[2]["t_s"], rows[3]["t_s"] = 5.0, 5.333
    segs = ss.spot(rows, _probs(["A"] * 4), LABELS)
    assert [s["n"] for s in segs] == [2, 2]


def test_loop_window_covers_the_sign_and_only_it():
    seg = {"t_first": 2.0, "t_last": 3.0, "step": 1 / 3}
    assert ss.loop_window(seg) == pytest.approx((2.0 - 1 / 6, 3.0 + 1 / 6), abs=1e-3)
    short = ss.loop_window({"t_first": 2.0, "t_last": 2.0, "step": 1 / 3})
    assert short[1] - short[0] == pytest.approx(ss.MIN_LOOP_S, abs=1e-3)
    dyn = ss.loop_window(seg, dynamic=True)
    assert dyn[0] < 2.0 - ss.DYN_PRE_S + 0.01 and dyn[1] > 3.0 + ss.DYN_POST_S - 0.01
    assert ss.loop_window({"t_first": 0.1, "t_last": 0.5, "step": 1 / 3})[0] == 0.0


def test_signs_in_window_counts_confident_runs():
    seq = ["A", "A", "A", "B", "B", "A", "A"]
    rows = _rows(len(seq))
    P = _probs(seq)
    times = [r["t_s"] for r in rows]
    lab = list(P.argmax(1))
    assert ss.signs_in_window(times, lab, P, 0.0, 10.0) == 2
    assert ss.signs_in_window(times, lab, P, 0.0, 0.7) == 1
    assert ss.signs_in_window(times, lab, _probs(seq, p=0.4), 0.0, 10.0) == 0
