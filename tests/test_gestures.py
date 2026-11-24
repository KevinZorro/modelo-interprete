import numpy as np

from lsc70.gestures import classify_gesture
from lsc70.spelling import GESTURE_DELETE, GESTURE_SPACE

# synthetic hands: finger base on the palm, then 3 more points along a direction (extended) or
# folded back toward the palm (curled)
BASE = {"thumb": (-0.22, 0.18), "index": (-0.16, 0.50), "middle": (0.0, 0.55),
        "ring": (0.16, 0.50), "pinky": (0.30, 0.42)}
FIRST = {"thumb": 1, "index": 5, "middle": 9, "ring": 13, "pinky": 17}


def hand(extended, fan=0.35):
    pts = np.zeros((21, 3))
    for k, (name, (bx, by)) in enumerate(BASE.items()):
        i0 = FIRST[name]
        pts[i0] = (bx, by, 0)
        ang = (k - 2) * fan * (1.6 if name == "thumb" else 1.0)
        if name == "thumb":
            ang = -1.2 if extended[name] else -0.2
        dx, dy = np.sin(ang), np.cos(ang)
        if name == "thumb":
            dx, dy = -abs(np.sin(ang)) , np.cos(ang) * 0.4 + 0.2
        for s in range(1, 4):
            if extended[name]:
                pts[i0 + s] = (bx + dx * 0.18 * s, by + dy * 0.18 * s, 0)
            else:  # fold toward the palm: tip ends closer to the wrist than the pip
                back = 0.12 if s == 1 else (0.05 if s == 2 else -0.02)
                pts[i0 + s] = (bx * 0.8 + dx * back, by - 0.05 * s + dy * back, 0.02 * s)
    pts /= np.linalg.norm(pts, axis=1).max()
    return pts.reshape(-1)


ALL = dict.fromkeys(BASE, True)


def test_open_hand_is_delete():
    assert classify_gesture(hand(ALL)) == GESTURE_DELETE


def test_ily_is_space():
    g = dict(ALL, middle=False, ring=False)
    assert classify_gesture(hand(g)) == GESTURE_SPACE


def test_fist_and_pointing_are_not_gestures():
    fist = dict.fromkeys(BASE, False)
    assert classify_gesture(hand(fist)) is None
    assert classify_gesture(hand(dict(fist, index=True))) is None


def test_y_and_l_are_not_space():
    fist = dict.fromkeys(BASE, False)
    assert classify_gesture(hand(dict(fist, thumb=True, pinky=True))) is None   # Y
    assert classify_gesture(hand(dict(fist, thumb=True, index=True))) is None   # L


def test_closed_fingers_together_is_not_delete():
    # B: four fingers extended but together, thumb folded
    b = dict(ALL, thumb=False)
    assert classify_gesture(hand(b, fan=0.0)) is None
