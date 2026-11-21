from lsc70.spelling import (LETTER, NO_HAND, REJECTED, Config, Frame, LetterConfirmed,
                            SpellingEngine, WordFinished, decode, load_lexicon, word_to_units)

STEP = 33  # ~30 fps


def hold(letter, ms, t, conf=0.9):
    return [Frame(t + i * STEP, LETTER, letter, conf) for i in range(ms // STEP + 1)]


def gap(ms, t, kind=NO_HAND):
    return [Frame(t + i * STEP, kind) for i in range(1, ms // STEP + 1)]


def run(segments):
    """segments: list of ('L', ms[, conf]) or ('_', ms) for no hand, ('r', ms) rejected."""
    eng, t, events = SpellingEngine(), 0, []
    for seg in segments:
        kind, ms = seg[0], seg[1]
        if kind == "_":
            frames = gap(ms, t)
        elif kind == "r":
            frames = gap(ms, t, REJECTED)
        else:
            frames = hold(kind, ms, t, seg[2] if len(seg) > 2 else 0.9)
        for f in frames:
            events += eng.push(f)
        t = frames[-1].t_ms
    return events


def letters(events):
    return [e.letter for e in events if isinstance(e, LetterConfirmed)]


def words(events):
    return [e for e in events if isinstance(e, WordFinished)]


def test_long_hold_confirms_once():
    assert letters(run([("A", 3000)])) == ["A"]


def test_short_touch_is_not_a_letter():
    assert letters(run([("A", 150), ("_", 400)])) == []


def test_same_letter_needs_release():
    assert letters(run([("L", 1500)])) == ["L"]
    assert letters(run([("L", 600), ("_", 300), ("L", 600)])) == ["L", "L"]


def test_short_gap_does_not_repeat():
    # a 100 ms dropout inside a hold is tracking noise, not a second L
    assert letters(run([("L", 600), ("r", 100), ("L", 600)])) == ["L"]


def test_different_letters_need_no_release():
    assert letters(run([("C", 600), ("A", 600), ("R", 600)])) == ["C", "A", "R"]


def test_one_frame_flicker_does_not_double_a_letter():
    ev = run([("L", 600), ("K", 33), ("L", 600)])
    assert letters(ev) == ["L"]


def test_word_calle_with_repeated_l():
    ev = run([("C", 600), ("A", 600), ("L", 600), ("_", 300), ("L", 600), ("E", 600), ("_", 2000)])
    w = words(ev)
    assert len(w) == 1 and w[0].letters == ("C", "A", "L", "L", "E") and w[0].best == "CALLE"


def test_word_perro_rr_is_two_r():
    ev = run([("P", 500), ("E", 500), ("R", 500), ("_", 300), ("R", 500), ("O", 500), ("_", 2000)])
    assert words(ev)[0].best == "PERRO"


def test_misread_low_confidence_letter_is_corrected():
    ev = run([("C", 600), ("A", 600), ("L", 600), ("_", 300), ("L", 600), ("O", 600, 0.4), ("_", 2000)])
    w = words(ev)[0]
    assert w.letters[-1] == "O" and w.best == "CALLE"


def test_confident_wrong_letter_is_not_hidden():
    # distance > budget for 5 letters with two confident substitutions -> no match, raw letters kept
    ev = run([("X", 500), ("X", 500), ("_", 300), ("X", 500), ("U", 500), ("_", 2000)])
    w = words(ev)[0]
    assert w.letters and w.best is None


def test_short_words_must_match_exactly():
    eng = SpellingEngine()
    best, _, _ = decode([("P", 0.5), ("A", 0.5)], eng.lexicon, eng.cfg)
    assert best is None  # no exact 2-letter word 'PA' in the list


def test_word_ends_only_after_pause():
    ev = run([("A", 600), ("_", 900)])
    assert not words(ev)
    ev = run([("A", 600), ("_", 1600)])
    assert len(words(ev)) == 1


def test_rejected_frames_between_letters_do_not_end_word():
    ev = run([("C", 600), ("r", 500), ("A", 600), ("r", 500), ("L", 600), ("_", 2000)])
    assert words(ev)[0].letters == ("C", "A", "L")


def test_finish_flushes_word():
    eng = SpellingEngine()
    for f in hold("M", 600, 0) + gap(300, 600):
        eng.push(f)
    ev = eng.finish(1000)
    assert isinstance(ev[0], WordFinished) and ev[0].letters == ("M",)


def test_two_words_back_to_back():
    ev = run([("M", 600), ("A", 600), ("R", 600), ("_", 2000), ("S", 10), ("L", 600), ("U", 600),
              ("N", 600), ("A", 600), ("_", 2000)])
    assert [w.best for w in words(ev)] == ["MAR", "LUNA"]


def test_units():
    assert word_to_units("calle") == ("C", "A", "L", "L", "E")
    assert word_to_units("año") == ("A", "NN", "O")
    assert word_to_units("café") == ("C", "A", "F", "E")


def test_lexicon_drops_dynamic_letters():
    cfg = Config.load()
    lex = load_lexicon(excluded=cfg.excluded_letters)
    assert len(lex) > 200
    bad = set(cfg.excluded_letters)
    assert all(not (set(u) & bad) for _, u in lex)
    assert "CALLE" in [w for w, _ in lex] and "CASA" not in [w for w, _ in lex]
