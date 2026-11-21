"""Spelling engine over static letters (T-20).

Input: one classifier result per camera frame (``Frame``). Output: events
(``LetterConfirmed``, ``WordFinished``). Pure Python, no TensorFlow, no clock: time comes
from the frames, so tests are deterministic and the Kotlin port can mirror it line by line.

Why this shape (see docs/reports/T-20.md):
- A letter is confirmed after being held ``hold_ms`` (no one-shot latch tied to hand removal).
- The SAME letter twice (LL, RR, "casa" -> A, A is not needed but "calle" needs L, L) is
  written by releasing: a non-letter gap of ``release_ms``. A DIFFERENT letter needs no release.
- A word ends after ``word_pause_ms`` without a letter; the letters are then decoded against a
  closed lexicon with a confidence-weighted edit distance, so a misread letter can be fixed.
- LL and RR are two repeated base letters (L L, R R); N-tilde is the class ``NN``. The model has
  no class for the digraphs, so they live in the lexicon, not in the classifier.
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "spelling.json"
LEXICON_PATH = ROOT / "configs" / "palabras_deletreo.txt"

LETTER, REJECTED, NO_HAND = "letter", "rejected", "no_hand"


@dataclass(frozen=True)
class Frame:
    """What the classifier said for one camera frame (after threshold, contract v1)."""
    t_ms: int
    kind: str  # LETTER | REJECTED | NO_HAND
    label: str | None = None
    conf: float = 0.0


@dataclass(frozen=True)
class LetterConfirmed:
    t_ms: int
    letter: str
    conf: float


@dataclass(frozen=True)
class Candidate:
    word: str
    cost: float


@dataclass(frozen=True)
class WordFinished:
    t_ms: int
    letters: tuple[str, ...]
    best: str | None            # None: nothing in the lexicon is close enough
    candidates: tuple[Candidate, ...]
    ambiguous: bool             # runner-up within ``ambiguity_margin`` of the best


@dataclass
class Config:
    hold_ms: int = 400
    min_frames: int = 3
    release_ms: int = 200
    word_pause_ms: int = 1500
    max_word_letters: int = 20
    excluded_letters: tuple[str, ...] = ("G", "H", "J", "S", "Z", "NN")
    min_sub_cost: float = 0.35
    max_cost_by_length: dict = field(default_factory=lambda: {
        "1": 0.0, "2": 0.0, "3": 1.0, "4": 1.0, "5": 1.0, "6": 1.5, "7": 1.5, "default": 2.0})
    ambiguity_margin: float = 0.25
    max_candidates: int = 3

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        keys = cls.__dataclass_fields__.keys()
        kw = {k: v for k, v in raw.items() if k in keys}
        if "excluded_letters" in kw:
            kw["excluded_letters"] = tuple(kw["excluded_letters"])
        return cls(**kw)

    def max_cost(self, n_letters: int) -> float:
        return float(self.max_cost_by_length.get(str(n_letters), self.max_cost_by_length["default"]))


# ----------------------------------------------------------------------------- lexicon
def word_to_units(word: str) -> tuple[str, ...]:
    """'calle' -> C A L L E ; 'pino' stays; N-tilde -> NN; accents dropped; RR -> R R."""
    out = []
    for ch in word.strip().upper():
        if ch == "Ñ":
            out.append("NN")
            continue
        base = unicodedata.normalize("NFD", ch)[0]
        if not ("A" <= base <= "Z"):
            raise ValueError(f"unsupported character {ch!r} in {word!r}")
        out.append(base)
    return tuple(out)


def load_lexicon(path: Path = LEXICON_PATH, excluded: tuple[str, ...] = ()) -> list[tuple[str, tuple[str, ...]]]:
    """Return [(word, units)] in file order (= priority), dropping duplicates, comments,
    multi-word lines and words that use an excluded (dynamic) letter."""
    seen, out = set(), []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        w = line.strip()
        if not w or w.startswith("#") or " " in w:
            continue
        w = w.lower()
        if w in seen:
            continue
        seen.add(w)
        units = word_to_units(w)
        if any(u in excluded for u in units):
            continue
        out.append((w.upper(), units))
    return out


# ----------------------------------------------------------------------------- decoder
def weighted_distance(observed: list[tuple[str, float]], target: tuple[str, ...], min_sub: float) -> float:
    """Levenshtein where insert/delete cost 1 and substituting an observed letter costs its
    confidence (floored at ``min_sub``): a doubtful letter is cheap to overrule."""
    n, m = len(observed), len(target)
    prev = [float(j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [float(i)] + [0.0] * m
        letter, conf = observed[i - 1]
        sub = max(min_sub, min(1.0, conf))
        for j in range(1, m + 1):
            same = letter == target[j - 1]
            cur[j] = min(prev[j] + 1.0, cur[j - 1] + 1.0, prev[j - 1] + (0.0 if same else sub))
        prev = cur
    return prev[m]


def decode(observed: list[tuple[str, float]], lexicon, cfg: Config):
    """Return (best, candidates, ambiguous) for the confirmed letters of one word."""
    if not observed:
        return None, (), False
    scored = []
    for rank, (word, units) in enumerate(lexicon):
        if abs(len(units) - len(observed)) > 2:  # cannot be within budget anyway
            continue
        scored.append((weighted_distance(observed, units, cfg.min_sub_cost), rank, word))
    scored.sort()
    limit = cfg.max_cost(len(observed))
    good = [Candidate(w, round(c, 4)) for c, _, w in scored if c <= limit + 1e-9][: cfg.max_candidates]
    if not good:
        return None, (), False
    amb = len(good) > 1 and good[1].cost - good[0].cost < cfg.ambiguity_margin
    return good[0].word, tuple(good), amb


# ----------------------------------------------------------------------------- engine
class SpellingEngine:
    def __init__(self, cfg: Config | None = None, lexicon=None):
        self.cfg = cfg or Config.load()
        self.lexicon = lexicon if lexicon is not None else load_lexicon(excluded=self.cfg.excluded_letters)
        self.reset()

    def reset(self) -> None:
        self._letters: list[tuple[str, float]] = []   # confirmed letters of the current word
        self._cand: str | None = None                  # letter being held
        self._cand_start = 0
        self._cand_frames = 0
        self._cand_conf = 0.0
        self._blocked: str | None = None               # last confirmed letter, until released
        self._idle_start: int | None = None            # start of the current non-letter stretch
        self._last_t = 0

    # public -----------------------------------------------------------------------
    def push(self, f: Frame) -> list:
        events: list = []
        self._last_t = f.t_ms
        if f.kind == LETTER and f.label:
            self._on_letter(f, events)
        else:
            self._on_idle(f, events)
        return events

    def tick(self, t_ms: int) -> list:
        """Advance time with no information (e.g. camera stopped): counts as no hand."""
        return self.push(Frame(t_ms, NO_HAND))

    def finish(self, t_ms: int) -> list:
        """Force the end of the word (user pressed 'done')."""
        events: list = []
        self._end_word(t_ms, events)
        return events

    # internals ---------------------------------------------------------------------
    def _on_letter(self, f: Frame, events: list) -> None:
        idle = 0 if self._idle_start is None else f.t_ms - self._idle_start
        self._idle_start = None
        if idle >= self.cfg.release_ms:
            self._blocked = None
        if f.label != self._cand or idle >= self.cfg.release_ms:
            self._cand, self._cand_start = f.label, f.t_ms
            self._cand_frames, self._cand_conf = 0, 0.0
        self._cand_frames += 1
        self._cand_conf += f.conf
        held = f.t_ms - self._cand_start
        if (held >= self.cfg.hold_ms and self._cand_frames >= self.cfg.min_frames
                and f.label != self._blocked):
            conf = self._cand_conf / self._cand_frames
            self._letters.append((f.label, conf))
            self._blocked = f.label
            events.append(LetterConfirmed(f.t_ms, f.label, round(conf, 4)))
            if len(self._letters) >= self.cfg.max_word_letters:
                self._end_word(f.t_ms, events)

    def _on_idle(self, f: Frame, events: list) -> None:
        if self._idle_start is None:
            self._idle_start = f.t_ms
        if self._letters and f.t_ms - self._idle_start >= self.cfg.word_pause_ms:
            self._end_word(f.t_ms, events)

    def _end_word(self, t_ms: int, events: list) -> None:
        if self._letters:
            best, cands, amb = decode(self._letters, self.lexicon, self.cfg)
            events.append(WordFinished(t_ms, tuple(l for l, _ in self._letters), best, cands, amb))
        self.reset()
        self._idle_start = t_ms
