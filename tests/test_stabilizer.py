"""Prediction stabilisation -- the rules that turn frames into letters.

The scenarios here are written from real failure modes: a sign the model is
only 60% sure about, two similar letters flickering, a dropped frame in the
middle of a hold. All of those used to leave the progress bar at zero.
"""

from __future__ import annotations

import pytest

from backend.app.services.stabilizer import (
    PredictionStabilizer,
    RecognitionStatus,
    StabilizerConfig,
)

CONFIG = StabilizerConfig(
    min_confidence=0.55,
    partial_confidence=0.30,
    stable_frames=6,
    decay=0.8,
    commit_cooldown_ms=450,
    duplicate_suppression_ms=900,
    release_frames=4,
)


@pytest.fixture
def stabilizer() -> PredictionStabilizer:
    return PredictionStabilizer(CONFIG)


def feed(stabilizer, letter, confidence=0.9, frames=1, now=0.0):
    """Feed N identical frames, returning every update."""
    return [
        stabilizer.update(letter, confidence, now=now) for _ in range(frames)
    ]


def committed(updates):
    return [u.committed_letter for u in updates if u.committed_letter]


# ----------------------------------------------------------------------
# The reported bug
# ----------------------------------------------------------------------


def test_a_moderately_confident_sign_still_fills_the_bar(stabilizer):
    """The reported bug: letter detected, bar frozen at zero, nothing typed.

    0.62 is a perfectly ordinary live confidence for this model.
    """
    updates = feed(stabilizer, "H", confidence=0.62, frames=6)
    assert updates[0].progress > 0
    assert updates[2].progress > updates[0].progress   # it moves
    assert committed(updates) == ["H"]


def test_progress_is_monotonic_while_holding_one_sign(stabilizer):
    updates = feed(stabilizer, "B", confidence=0.7, frames=5)
    progresses = [u.progress for u in updates]
    assert progresses == sorted(progresses)
    assert progresses[-1] > 0.5


def test_partial_confidence_still_makes_progress_just_slower(stabilizer):
    """A 0.42 sign is halfway between the two thresholds, so worth half."""
    weak = PredictionStabilizer(CONFIG)
    strong = PredictionStabilizer(CONFIG)
    for _ in range(4):
        weak_update = weak.update("K", 0.425, now=0.0)
        strong_update = strong.update("K", 0.95, now=0.0)
    assert 0 < weak_update.progress < strong_update.progress


def test_flicker_between_similar_letters_still_converges(stabilizer):
    """A/E is this model's most common confusion. A majority must still win."""
    pattern = ["A", "A", "E", "A", "A", "E", "A", "A", "A", "A"]
    updates = [stabilizer.update(letter, 0.7, now=0.0) for letter in pattern]
    assert committed(updates) == ["A"]


def test_one_dropped_frame_does_not_reset_progress(stabilizer):
    feed(stabilizer, "C", confidence=0.9, frames=3)
    before = stabilizer.scores["C"]
    stabilizer.update(None, 0.0, now=0.0)      # a single lost frame
    after = stabilizer.scores["C"]
    assert after > before * 0.5                 # decayed, not erased


# ----------------------------------------------------------------------
# Core behaviour
# ----------------------------------------------------------------------


def test_no_hand_is_idle(stabilizer):
    update = stabilizer.update(None, 0.0, now=0.0)
    assert update.status is RecognitionStatus.IDLE
    assert update.committed_letter is None


def test_a_genuinely_unclear_sign_never_commits(stabilizer):
    updates = feed(stabilizer, "A", confidence=0.2, frames=30)
    assert committed(updates) == []
    assert updates[-1].status is RecognitionStatus.LOW_CONFIDENCE


def test_a_confident_hold_commits_once(stabilizer):
    updates = feed(stabilizer, "A", frames=6)
    assert committed(updates) == ["A"]
    assert updates[-1].status is RecognitionStatus.COMMITTED


def test_a_long_hold_types_one_letter(stabilizer):
    """The headline requirement: AAAAAA... must not become 'AAAAAA'."""
    updates = feed(stabilizer, "A", frames=60)
    assert committed(updates) == ["A"]
    assert updates[-1].status is RecognitionStatus.HOLD_RELEASE


def test_changing_pose_allows_the_next_letter(stabilizer):
    feed(stabilizer, "A", frames=6, now=0.0)
    updates = feed(stabilizer, "B", frames=6, now=1.0)
    assert committed(updates) == ["B"]


def test_releasing_and_re_signing_types_a_double_letter(stabilizer):
    """LL in HELLO: possible, but only after releasing and waiting."""
    feed(stabilizer, "L", frames=6, now=0.0)
    feed(stabilizer, None, confidence=0.0, frames=5, now=1.0)   # release
    updates = feed(stabilizer, "L", frames=6, now=2.0)
    assert committed(updates) == ["L"]


def test_duplicate_letter_is_suppressed_within_the_window(stabilizer):
    feed(stabilizer, "L", frames=6, now=0.0)
    feed(stabilizer, None, confidence=0.0, frames=5, now=0.2)
    updates = feed(stabilizer, "L", frames=6, now=0.5)          # only 500 ms
    assert committed(updates) == []
    assert updates[-1].status is RecognitionStatus.COOLDOWN


def test_cooldown_gates_two_different_letters(stabilizer):
    feed(stabilizer, "A", frames=6, now=0.0)
    updates = feed(stabilizer, "B", frames=6, now=0.1)          # 100 ms
    assert committed(updates) == []
    assert updates[-1].status is RecognitionStatus.COOLDOWN


def test_the_bar_stays_full_during_cooldown(stabilizer):
    """It must not creep past 100% or drop back while waiting."""
    feed(stabilizer, "A", frames=6, now=0.0)
    updates = feed(stabilizer, "B", frames=10, now=0.1)
    assert updates[-1].progress == pytest.approx(1.0)


def test_reset_forgets_everything(stabilizer):
    feed(stabilizer, "A", frames=4)
    stabilizer.reset()
    assert stabilizer.scores == {}
    assert stabilizer.update("A", 0.9, now=0.0).progress < 0.5


def test_status_progresses_from_detecting_to_locking(stabilizer):
    updates = feed(stabilizer, "C", frames=4)
    assert updates[0].status is RecognitionStatus.DETECTING
    assert updates[-1].status is RecognitionStatus.LOCKING


def test_a_long_hold_after_release_does_not_double_type(stabilizer):
    """Holding through the release window must still type only once."""
    updates = feed(stabilizer, "W", frames=15, now=0.0)
    assert committed(updates) == ["W"]


# ----------------------------------------------------------------------
# Realistic stream
# ----------------------------------------------------------------------


def test_spelling_a_word_from_a_noisy_stream():
    """End to end: HI, with the kind of noise a real camera produces."""
    stabilizer = PredictionStabilizer(CONFIG)
    typed: list[str] = []
    clock = 0.0

    def stream(frames):
        nonlocal clock
        for letter, confidence in frames:
            clock += 1 / 15  # 15 fps
            update = stabilizer.update(letter, confidence, now=clock)
            if update.committed_letter:
                typed.append(update.committed_letter)

    # reaching for the sign, then holding H with normal wobble
    stream([(None, 0.0)] * 3)
    stream([("H", 0.4), ("H", 0.7), ("N", 0.5), ("H", 0.8), ("H", 0.75),
            ("H", 0.9), ("H", 0.85), ("H", 0.9)])
    # moving the hand between letters
    stream([(None, 0.0)] * 5)
    # holding I
    stream([("I", 0.6), ("I", 0.8), ("J", 0.5), ("I", 0.85), ("I", 0.9),
            ("I", 0.88), ("I", 0.9), ("I", 0.9)])

    assert typed == ["H", "I"]


def test_config_rejects_nonsense():
    with pytest.raises(ValueError):
        StabilizerConfig(stable_frames=0)
    with pytest.raises(ValueError):
        StabilizerConfig(min_confidence=1.5)
    with pytest.raises(ValueError):
        StabilizerConfig(min_confidence=0.4, partial_confidence=0.9)
    with pytest.raises(ValueError):
        StabilizerConfig(decay=1.0)
