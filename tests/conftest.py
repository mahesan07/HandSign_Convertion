"""Shared fixtures.

Most tests run against a scripted recogniser so they are fast and do not need
a camera. The tests that genuinely exercise the trained model are marked
``slow`` and load it for real.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import pytest

from backend.app.core.config import Settings
from ml.landmarks import Landmark
from ml.recognizer import RecognitionResult


# ----------------------------------------------------------------------
# Doubles
# ----------------------------------------------------------------------


@dataclass
class ScriptedRecognizer:
    """Stands in for :class:`ml.SignRecognizer`.

    ``script`` is replayed one entry per frame and then repeats its last
    entry, so a test can say "six frames of A" without building images.
    """

    script: List[Tuple[Optional[str], float]] = field(
        default_factory=lambda: [(None, 0.0)]
    )
    calls: int = 0

    @property
    def classes(self) -> List[str]:
        return [chr(c) for c in range(ord("A"), ord("Z") + 1)]

    def recognize(self, rgb_image) -> RecognitionResult:  # noqa: ANN001
        index = min(self.calls, len(self.script) - 1)
        self.calls += 1
        letter, confidence = self.script[index]
        if letter is None:
            return RecognitionResult(hand_detected=False, elapsed_ms=1.0)
        return RecognitionResult(
            hand_detected=True,
            letter=letter,
            confidence=confidence,
            alternatives=((letter, confidence),),
            landmarks=[Landmark(0.5, 0.5, 0.0)] * 21,
            elapsed_ms=1.0,
        )

    def recognize_landmarks(self, landmarks: Sequence[Landmark]):  # noqa: ANN001
        raise NotImplementedError

    def close(self) -> None:
        pass


class StubGemini:
    """A Gemini service that returns, fails or hangs exactly on demand."""

    def __init__(self, payload=None, error: Exception | None = None, available=True):
        self.payload = payload
        self.error = error
        self._available = available
        self.calls = 0

    @property
    def available(self) -> bool:
        return self._available

    @property
    def model_name(self) -> str:
        return "stub"

    async def suggest(self, ctx):  # noqa: ANN001
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.payload


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def settings() -> Settings:
    """Settings with no API key and fast timings, ignoring any local .env."""
    return Settings(
        _env_file=None,
        gemini_api_key=None,
        stable_frames=3,
        min_confidence=0.8,
        commit_cooldown_ms=0,
        duplicate_suppression_ms=0,
        release_frames=2,
        suggestion_debounce_ms=10,
    )


@pytest.fixture
def blank_frame() -> str:
    """A tiny valid JPEG, base64 encoded."""
    import cv2
    import numpy as np

    ok, buffer = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))
    assert ok
    return base64.b64encode(buffer.tobytes()).decode()


@pytest.fixture
def client(monkeypatch, settings):
    """A TestClient whose recogniser is scripted, not real.

    The fixture yields ``(client, recognizer)`` so a test can rewrite the
    script before sending frames.
    """
    from fastapi.testclient import TestClient

    import backend.app.main as main
    from backend.app.services.recognition_service import RecognitionService

    recognizer = ScriptedRecognizer()

    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(
        main,
        "RecognitionService",
        lambda cfg: RecognitionService(cfg, recognizer=recognizer),
    )

    app = main.create_app()
    with TestClient(app) as test_client:
        yield test_client, recognizer
