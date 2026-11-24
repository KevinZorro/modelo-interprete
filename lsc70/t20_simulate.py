"""T-20: how many words does the decoder recover when letters fail? (simulation, not camera)

Each letter of each lexicon word is confirmed correctly with probability ``p`` and otherwise
replaced by a random different letter with a LOW confidence (misses are usually doubtful),
or - with probability ``p_conf_wrong`` of the misses - by a confidently wrong letter.
``python -m lsc70.t20_simulate``  (seeded, 3 seeds)
"""
from __future__ import annotations

import json
import random
import statistics
import sys

from lsc70.spelling import Config, decode, load_lexicon

LETTERS = list("ABCDEFIKLMNOPQRTUVWXY")  # letters the static lexicon can contain


def simulate(p: float, p_conf_wrong: float, seed: int, reps: int = 5, p_edge: float = 0.0):
    cfg = Config.load()
    lex = load_lexicon(excluded=cfg.excluded_letters)
    rng = random.Random(seed)
    raw_ok = dec_ok = wrong = total = 0
    for _ in range(reps):
        for word, units in lex:
            obs = []
            for u in units:
                if rng.random() < p:
                    obs.append((u, rng.uniform(0.6, 1.0)))
                else:
                    other = rng.choice([x for x in LETTERS if x != u])
                    conf = rng.uniform(0.6, 1.0) if rng.random() < p_conf_wrong else rng.uniform(0.35, 0.6)
                    obs.append((other, conf))
            if rng.random() < p_edge:  # hand raised through a letter-like pose
                obs.insert(0, (rng.choice(LETTERS), rng.uniform(0.5, 0.8)))
            if rng.random() < p_edge:  # ...and lowered through another
                obs.append((rng.choice(LETTERS), rng.uniform(0.5, 0.8)))
            total += 1
            raw_ok += [o[0] for o in obs] == list(units)
            best, _, _ = decode(obs, lex, cfg)
            dec_ok += best == word
            wrong += best is not None and best != word
    return raw_ok / total, dec_ok / total, wrong / total, len(lex)


def main() -> None:
    rows = []
    for p, pcw, pe in ((0.888, 0.0, 0.0), (0.888, 0.5, 0.0), (0.888, 1.0, 0.0), (0.95, 1.0, 0.0),
                       (0.95, 0.5, 0.5), (0.95, 0.5, 1.0)):
        if True:
            res = [simulate(p, pcw, s, p_edge=pe) for s in (0, 1, 2)]
            def m(i):
                return round(statistics.mean(r[i] for r in res), 3)

            rows.append({"letter_acc": p, "p_conf_wrong": pcw, "p_edge": pe, "raw": m(0), "decoded": m(1),
                         "wrong_word": m(2), "lexicon": res[0][3]})
    json.dump(rows, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
