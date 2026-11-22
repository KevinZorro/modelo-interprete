"""Synthetic scraped videos with known ground truth, built from real dataset poses.

Holds: one pose (landmarks from landmarks_cache Xtr) kept 4-7 samples with small jitter.
Transitions: 1-3 samples morphing between two poses while the hand travels across the frame.
Samples are 1/3 s apart, like the scraper's stride. Returns rows for landmarks.jsonl plus
`is_hold` per sample_id. A stand-in for real videos, not a substitute for them.
"""
import numpy as np

from lsc70 import review_queue as rq

STEP_S = 1 / 3


def make_videos(npz_path, n_videos=30, seed=0, hold_share=0.25, fast_share=0.0, classes=None):
    d = np.load(npz_path, allow_pickle=True)
    X, y = d["Xtr"], d["ytr"]
    names = [str(e) for e in d["etiquetas"]]
    keep = y != names.index("none")
    X, y = X[keep], y[keep]
    rng = np.random.default_rng(seed)
    rows, truth = [], {}
    for v in range(n_videos):
        t, frame, pos = 0.0, 0, rng.uniform(0.3, 0.7, 2)
        ix = rng.integers(len(X))
        cur = X[ix].reshape(21, 3)
        for _ in range(rng.integers(3, 7)):
            nx = rng.integers(len(X))
            nxt = X[nx].reshape(21, 3)
            moves = []
            # transitions dominate real conversational video; holds are the minority
            n_hold = int(rng.integers(4, 8)) if rng.random() < hold_share * 2 else 0
            if n_hold and rng.random() < fast_share:
                n_hold = int(rng.integers(2, 4))   # a quick sign: 0.7-1.0 s at 3 samples/s
            for k in range(n_hold):
                moves.append((cur + rng.normal(0, 0.01, cur.shape), pos + rng.normal(0, 0.002, 2), names[y[ix]]))
            n_tr = int(rng.integers(2, 6))
            target = rng.uniform(0.2, 0.8, 2)
            for k in range(1, n_tr + 1):
                a = k / (n_tr + 1)
                moves.append(((1 - a) * cur + a * nxt, pos + a * (target - pos), None))
            for shape, p, hold in moves:
                lm = np.round(shape[:, :2] * 0.15 + p, 5)
                lm3 = np.concatenate([lm, shape[:, 2:] * 0.15], axis=1)
                r = {"video_id": f"v{v}", "url": f"https://x/{v}", "frame": frame, "t_s": round(t, 3),
                     "is_left": False, "landmarks": lm3.tolist(), "features": shape.reshape(-1).tolist(),
                     "query_kind": "lsc", "review_status": "pending"}
                rows.append(r)
                truth[rq.sample_id(r)] = hold is not None
                if classes is not None and hold is not None:
                    classes[rq.sample_id(r)] = hold
                t += STEP_S
                frame += 10
            cur, pos, ix = nxt, target, nx
    return rows, truth
