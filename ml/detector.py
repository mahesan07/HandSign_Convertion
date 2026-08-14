"""MediaPipe hand-landmark detection.

Wraps ``mediapipe.tasks.vision.HandLandmarker`` with exactly the options the
original scripts used, so detection behaviour is unchanged -- the only
difference is that the detector is created once and reused instead of being
rebuilt by every script.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional

import numpy as np

from ml.landmarks import Landmark
from ml.paths import HAND_LANDMARKER_PATH


class HandDetector:
    """Detects a single hand and returns its 21 landmarks.

    MediaPipe's task objects are not documented as thread-safe, so calls are
    serialised with a lock.  Detection is ~8-12 ms, well inside a frame budget.
    """

    def __init__(
        self,
        model_path: Path | str = HAND_LANDMARKER_PATH,
        *,
        num_hands: int = 1,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe hand landmarker not found at {self.model_path}."
            )

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(self.model_path)
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._lock = threading.Lock()
        self._closed = False

    def detect(self, rgb_image: np.ndarray) -> Optional[List[Landmark]]:
        """Return the landmarks of the most prominent hand, or ``None``.

        ``rgb_image`` must be an ``HxWx3`` uint8 array in **RGB** order.
        """
        import mediapipe as mp

        if self._closed:
            raise RuntimeError("HandDetector has been closed")

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        with self._lock:
            result = self._detector.detect(mp_image)

        if not result.hand_landmarks:
            return None
        return [
            Landmark(float(lm.x), float(lm.y), float(lm.z))
            for lm in result.hand_landmarks[0]
        ]

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._detector.close()

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = ["HandDetector"]
