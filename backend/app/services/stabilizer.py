"""Turns a stream of noisy per-frame predictions into deliberate letters.

The classifier fires 15-30 times a second and is not clean: confidence on a
live webcam sits well below what it scores on recorded data, and similar signs
(A/E, U/V, O/C) flicker back and forth between frames.

An earlier version of this file required N *consecutive, high-confidence,
identical* predictions. On real input that almost never happens, so the
progress bar sat at zero and nothing could be typed. It also threw away all
progress on a single disagreeing frame.

This version accumulates **evidence** instead:

* every frame adds to the score of the letter it saw, weighted by how
  confident it was -- a 0.9 frame counts fully, a 0.45 frame counts partially,
  and only genuinely poor frames count for nothing;
* rival letters *decay* rather than reset, so A/E flicker still converges on
  whichever one is winning instead of cancelling out;
* a letter commits once its score reaches ``stable_frames`` worth of evidence.

The visible effect is that the bar always moves while a hand is in frame, and
fills faster the more confident the sign is. Holding a sign steadily still
types it in about the same time as before.

The safeguards are unchanged: one hold types one letter, a cooldown separates
letters, and repeating the same letter needs a longer, deliberate gap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class RecognitionStatus(str, Enum):
    """What the recogniser is doing right now, for the UI status line."""

    IDLE = "idle"                      # no hand in frame
    LOW_CONFIDENCE = "low_confidence"  # hand seen, but the sign is unclear
    DETECTING = "detecting"            # evidence is building
    LOCKING = "locking"                # nearly there, keep holding
    COMMITTED = "committed"            # a letter was just added
    HOLD_RELEASE = "hold_release"      # committed; change pose to continue
    COOLDOWN = "cooldown"              # too soon after the previous letter


@dataclass(frozen=True, slots=True)
class StabilizerConfig:
    #: At or above this confidence a frame counts as a full frame of evidence.
    min_confidence: float = 0.55
    #: Below this it counts for nothing. Between the two it counts partially,
    #: which is what keeps the bar moving on a merely-decent sign.
    partial_confidence: float = 0.30
    #: Evidence needed to commit, measured in full-confidence frames.
    stable_frames: int = 6
    #: How fast a letter's score fades while a different letter is seen.
    #: This is the flicker-tolerance dial. Too low and a rival frame guts
    #: the leader's progress (0.6 leaves A/E flicker stuck at 5.9/6 and
    #: nothing is ever typed); too high and switching letters feels slow.
    decay: float = 0.8
    #: No letter may be committed within this long of the previous one.
    commit_cooldown_ms: int = 450
    #: Re-committing the *same* letter additionally requires this gap.
    duplicate_suppression_ms: int = 900
    #: Frames without a hand needed to release the pose and allow a repeat.
    release_frames: int = 4

    def __post_init__(self) -> None:
        if self.stable_frames < 1:
            raise ValueError("stable_frames must be >= 1")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if not 0.0 <= self.partial_confidence <= self.min_confidence:
            raise ValueError(
                "partial_confidence must be between 0 and min_confidence"
            )
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("decay must be between 0 and 1")

    def evidence_for(self, confidence: float) -> float:
        """How much a frame at this confidence is worth, from 0.0 to 1.0."""
        if confidence >= self.min_confidence:
            return 1.0
        if confidence <= self.partial_confidence:
            return 0.0
        span = self.min_confidence - self.partial_confidence
        if span <= 0:
            return 0.0
        return (confidence - self.partial_confidence) / span


@dataclass(frozen=True, slots=True)
class StabilizerUpdate:
    """Result of feeding one frame to the stabilizer."""

    status: RecognitionStatus
    #: The letter currently winning, if any.
    candidate: Optional[str] = None
    confidence: float = 0.0
    #: Accumulated evidence for the candidate, and how much is needed.
    #: Kept as ``stable_count`` for the API's sake; it is no longer an integer
    #: count of frames but the same thing in spirit.
    stable_count: float = 0.0
    required_frames: int = 1
    #: The letter to append to the text buffer -- ``None`` on most frames.
    committed_letter: Optional[str] = None

    @property
    def progress(self) -> float:
        """0.0 - 1.0, drives the "hold to confirm" bar in the UI."""
        if self.required_frames <= 0:
            return 0.0
        return min(1.0, max(0.0, self.stable_count / self.required_frames))


class PredictionStabilizer:
    """Debounces per-frame predictions into committed letters.

    One instance per session. Not thread-safe by itself; the session that owns
    it serialises access.
    """

    def __init__(self, config: StabilizerConfig | None = None) -> None:
        self.config = config or StabilizerConfig()
        self.reset()

    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._scores: Dict[str, float] = {}
        self._idle_frames: int = 0
        #: The letter typed by the pose currently being held, if any. Blocks
        #: that same pose from typing twice without being released.
        self._consumed: Optional[str] = None
        self._last_commit_letter: Optional[str] = None
        self._last_commit_at: float = 0.0

    # ------------------------------------------------------------------

    @property
    def scores(self) -> Dict[str, float]:
        """Current evidence per letter -- exposed for tests and debugging."""
        return dict(self._scores)

    def _leader(self) -> tuple[Optional[str], float]:
        if not self._scores:
            return None, 0.0
        letter = max(self._scores, key=self._scores.__getitem__)
        return letter, self._scores[letter]

    def _decay_all(self) -> None:
        decayed = {
            letter: score * self.config.decay
            for letter, score in self._scores.items()
        }
        # Drop anything that has faded to irrelevance.
        self._scores = {l: s for l, s in decayed.items() if s > 0.05}

    # ------------------------------------------------------------------

    def update(
        self,
        letter: Optional[str],
        confidence: float = 0.0,
        *,
        now: Optional[float] = None,
    ) -> StabilizerUpdate:
        """Feed one frame.

        ``letter`` is ``None`` when no hand was detected. ``now`` is a
        monotonic timestamp in seconds; it is injectable so tests need not
        sleep.
        """
        cfg = self.config
        now = time.monotonic() if now is None else now
        evidence = cfg.evidence_for(confidence) if letter is not None else 0.0

        # -- no hand, or a sign too unclear to count -----------------------
        if letter is None or evidence <= 0.0:
            self._idle_frames += 1
            self._decay_all()
            if self._idle_frames >= cfg.release_frames:
                # The pose has been released: the same letter may be typed
                # again, and stale evidence is dropped.
                self._scores.clear()
                self._consumed = None

            leader, score = self._leader()
            return StabilizerUpdate(
                status=(
                    RecognitionStatus.IDLE
                    if letter is None
                    else RecognitionStatus.LOW_CONFIDENCE
                ),
                candidate=letter,
                confidence=confidence,
                stable_count=score,
                required_frames=cfg.stable_frames,
            )

        self._idle_frames = 0

        # -- still holding the pose we just typed? -------------------------
        if self._consumed is not None:
            if letter == self._consumed:
                return StabilizerUpdate(
                    status=RecognitionStatus.HOLD_RELEASE,
                    candidate=letter,
                    confidence=confidence,
                    stable_count=cfg.stable_frames,
                    required_frames=cfg.stable_frames,
                )
            # A different sign: the user has moved on.
            self._consumed = None

        # -- accumulate evidence for this letter, fade the rest ------------
        for other in list(self._scores):
            if other != letter:
                self._scores[other] *= cfg.decay
                if self._scores[other] <= 0.05:
                    del self._scores[other]
        self._scores[letter] = self._scores.get(letter, 0.0) + evidence

        leader, score = self._leader()

        if score < cfg.stable_frames:
            status = (
                RecognitionStatus.LOCKING
                if score >= cfg.stable_frames * 0.5
                else RecognitionStatus.DETECTING
            )
            return StabilizerUpdate(
                status=status,
                candidate=leader,
                confidence=confidence,
                stable_count=score,
                required_frames=cfg.stable_frames,
            )

        # -- enough evidence: may we commit? -------------------------------
        elapsed_ms = (now - self._last_commit_at) * 1000.0
        required_gap = cfg.commit_cooldown_ms
        if leader == self._last_commit_letter:
            required_gap = max(required_gap, cfg.duplicate_suppression_ms)

        if self._last_commit_letter is not None and elapsed_ms < required_gap:
            # Hold the evidence at the threshold so the bar stays full rather
            # than growing without bound while we wait.
            self._scores[leader] = float(cfg.stable_frames)
            return StabilizerUpdate(
                status=RecognitionStatus.COOLDOWN,
                candidate=leader,
                confidence=confidence,
                stable_count=score,
                required_frames=cfg.stable_frames,
            )

        self._scores.clear()
        self._consumed = leader
        self._last_commit_letter = leader
        self._last_commit_at = now
        return StabilizerUpdate(
            status=RecognitionStatus.COMMITTED,
            candidate=leader,
            confidence=confidence,
            stable_count=float(cfg.stable_frames),
            required_frames=cfg.stable_frames,
            committed_letter=leader,
        )


__all__ = [
    "PredictionStabilizer",
    "StabilizerConfig",
    "StabilizerUpdate",
    "RecognitionStatus",
]
