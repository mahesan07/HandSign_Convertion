"""The complete recognition pipeline, in one object.

    RGB frame -> hand detection -> 63 features -> classifier -> letter

This is the exact pipeline the original ``live_prediction.py`` ran inline; it
is now reusable by the FastAPI backend, the CLI demo and the tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ml.classifier import Prediction, SignClassifier
from ml.detector import HandDetector
from ml.features import extract_features
from ml.landmarks import Landmark
from ml.paths import HAND_LANDMARKER_PATH, SIGN_MODEL_PATH


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    """Outcome of running one frame through the pipeline."""

    hand_detected: bool
    letter: Optional[str] = None
    confidence: float = 0.0
    #: ``(letter, probability)`` pairs, best first.
    alternatives: Tuple[Tuple[str, float], ...] = ()
    #: Landmarks in normalized image space, for drawing the overlay.
    landmarks: List[Landmark] = field(default_factory=list)
    #: Wall-clock time spent in this call, in milliseconds.
    elapsed_ms: float = 0.0

    @property
    def landmarks_xy(self) -> List[Tuple[float, float]]:
        """Landmarks as ``(x, y)`` pairs -- what the UI overlay needs."""
        return [(lm.x, lm.y) for lm in self.landmarks]


class SignRecognizer:
    """Loads the detector and classifier once, then recognises frames."""

    def __init__(
        self,
        landmarker_path: Path | str = HAND_LANDMARKER_PATH,
        classifier_path: Path | str = SIGN_MODEL_PATH,
        *,
        detector: HandDetector | None = None,
        classifier: SignClassifier | None = None,
    ) -> None:
        self.detector = detector or HandDetector(landmarker_path)
        self.classifier = classifier or SignClassifier(classifier_path)

    @property
    def classes(self) -> List[str]:
        return self.classifier.classes

    # ------------------------------------------------------------------

    def recognize(self, rgb_image: np.ndarray) -> RecognitionResult:
        """Run the full pipeline on one RGB frame."""
        started = time.perf_counter()
        landmarks = self.detector.detect(rgb_image)

        if landmarks is None:
            return RecognitionResult(
                hand_detected=False,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )

        prediction = self.recognize_landmarks(landmarks)
        return RecognitionResult(
            hand_detected=True,
            letter=prediction.letter,
            confidence=prediction.confidence,
            alternatives=prediction.alternatives,
            landmarks=landmarks,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def recognize_landmarks(self, landmarks: List[Landmark]) -> Prediction:
        """Classify landmarks that were detected elsewhere."""
        return self.classifier.predict(extract_features(landmarks))

    def close(self) -> None:
        self.detector.close()

    def __enter__(self) -> "SignRecognizer":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = ["SignRecognizer", "RecognitionResult"]
