"""Which letter should the review page suggest? Measured on the reviewer's own decisions.

Every sample the reviewer accepted, or relabelled to a sign, is a ground truth (none and
reject are left out: the question here is the letter, not whether it is a sign).

  centroid   nearest ANH class centroid: what the page suggests today (no numbers but 1/2/3)
  numbers    the same, plus the ANH numbers (4-10, MIL, MILLON) as their own classes
  +kind      the title says alphabet or numbers: only those classes are suggested

Top-1 is what "only accept" needs; top-3 is what three one-key buttons would cover. Also
counts the pending items by video kind, to see how much each kind weighs in what is left.
Tried and dropped on real decisions (T-22): the T-08 classifier as suggester, the
reviewer's own labels as references and an alphabet-order pass. Nothing here labels data.

    python -m lsc70.label_eval --out data/scraper --refs landmarks_cache.npz --json data/scraper/label_eval.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from lsc70 import alias_poses
from lsc70 import review_queue as rq

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "NN",
           "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
NUMBERS = ["F", "V", "W", "4", "5", "6", "7", "8", "9", "10", "MIL", "MILLON"]  # 1, 2, 3 = F, V, W
NUMBERS_TITLE = re.compile(r"n[uú]mer|numbers?\b|\bcontar\b|counting|\b1 al 10\b", re.I)
LETTERS_TITLE = re.compile(r"abecedario|alfabeto|alphabet|spelling|deletre|dactilolog|\bletras?\b", re.I)


def norm(label):
    """Pose-level class (F/1, V/2, W/3 are one pose), Ñ spelled NN as the model spells it."""
    if label is None:
        return None
    lab = str(label).strip()
    if lab.lower() == "none":
        return "none"
    lab = "NN" if lab.upper() in ("Ñ", "NN") else lab.upper()
    return alias_poses.canonical(lab)


def video_kind(title, query_kind=None) -> str:
    """letters, numbers or mixed, from the title; the search that found it when the title says neither."""
    t = title or ""
    num, let = bool(NUMBERS_TITLE.search(t)), bool(LETTERS_TITLE.search(t))
    if num and let:
        return "mixed"
    if num or let:
        return "numbers" if num else "letters"
    return {"numeros": "numbers", "lsc": "letters"}.get(query_kind, "mixed")


def allowed(kind: str, names: list) -> np.ndarray:
    """Which reference classes a video of this kind may be suggested."""
    ok = set(LETTERS) if kind == "letters" else set(NUMBERS) if kind == "numbers" else set(names) - {"none"}
    return np.array([n in ok for n in names])


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


def _ranked(F: np.ndarray, cent: np.ndarray, names: list, mask=None, k: int = 3) -> list:
    d = 1.0 - rq._unit(F) @ cent.T
    if mask is not None:
        d = np.where(mask[None, :], d, np.inf)
    return [[names[j] for j in row[:k]] for row in np.argsort(d, axis=1, kind="stable")]


def collect(out: Path, refs: Path) -> list:
    """Per ground-truth sample: its truth, video kind and each suggester's three best labels."""
    truth = {}
    for sid, rev in rq.latest_reviews(out).items():
        lab = norm(rev.get("label")) if rev["decision"] != "reject" else None
        if lab and lab != "none":
            truth[sid] = lab
    titles = rq.load_titles(out)
    base = rq.load_refs(refs)
    base = (base[0], [norm(n) for n in base[1]])
    nums = rq.load_refs(refs, numbers=True)
    rows = [r for r in rq.read_jsonl(out / "landmarks.jsonl") if rq.sample_id(r) in truth]
    recs = []
    for r in rows:
        image = r.get("media") == "image"
        kind = "mixed" if image else video_kind(titles.get(r["video_id"]) or r.get("title"), r.get("query_kind"))
        F = np.asarray([r["features"]], dtype=np.float64)
        pred = {}
        for name, (cent, names) in (("centroid", base), ("numbers", nums)):
            pred[name] = _ranked(F, cent, names, allowed("mixed", names))[0]
            pred[name + "+kind"] = _ranked(F, cent, names, allowed(kind, names))[0]
        recs.append({"sid": rq.sample_id(r), "vid": r["video_id"], "truth": truth[rq.sample_id(r)],
                     "kind": kind, "pred": pred})
    return recs


def _score(recs: list, method: str) -> dict:
    def part(rs):
        n = len(rs)
        k1 = sum(r["pred"][method][0] == r["truth"] for r in rs)
        k3 = sum(r["truth"] in r["pred"][method][:3] for r in rs)
        return {"n": n, "top1": round(k1 / n, 3) if n else None, "top1_ci95": wilson(k1, n),
                "top3": round(k3 / n, 3) if n else None, "top3_ci95": wilson(k3, n)}
    rep = {"all": part(recs), "letters": part([r for r in recs if r["truth"] in LETTERS]),
           "other": part([r for r in recs if r["truth"] not in LETTERS])}
    rep["by_kind"] = {k: part([r for r in recs if r["kind"] == k]) for k in sorted({r["kind"] for r in recs})}
    return rep


def pending_by_kind(out: Path) -> dict:
    """Items the strict page still shows, by video kind."""
    titles = rq.load_titles(out)
    counts = {}
    for it in rq.review_items(out, strict=True):
        if it["decision"] is None:
            k = "mixed" if it.get("media") == "image" else video_kind(titles.get(it["video_id"]) or it.get("title"),
                                                                     it.get("query_kind"))
            counts[k] = counts.get(k, 0) + 1
    return counts


def evaluate(out: Path, refs: Path) -> dict:
    recs = collect(out, refs)
    by_label = {}
    for r in recs:
        by_label[r["truth"]] = by_label.get(r["truth"], 0) + 1
    rep = {"truth": {"n": len(recs), "videos": len({r["vid"] for r in recs}),
                     "by_label": dict(sorted(by_label.items(), key=lambda x: -x[1]))},
           "methods": {m: _score(recs, m) for m in ("centroid", "centroid+kind", "numbers", "numbers+kind")}}
    try:
        rep["pending_by_kind"] = pending_by_kind(out)
    except (FileNotFoundError, KeyError) as e:   # no queue built yet
        rep["pending_by_kind"] = {"skipped": str(e)}
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description="Which suggested letter the reviewer would only accept")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--refs", type=Path, required=True)
    ap.add_argument("--json", type=Path, help="also write the report here")
    a = ap.parse_args(argv)
    txt = json.dumps(evaluate(a.out, a.refs), indent=1, ensure_ascii=False)
    if a.json:
        a.json.write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
