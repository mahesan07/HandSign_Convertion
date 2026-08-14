"""Frame in, letter out -- the async face of :class:`ml.SignRecognizer`.

The model is loaded exactly once, at application startup, and every frame is
handed to a small thread pool so that MediaPipe and the random forest (both
CPU-bound, both releasing the GIL) never stall the event loop that is serving
the websocket.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from backend.app.core.config import Settings
from backend.app.core.logging import get_logger
from ml.recognizer import RecognitionResult, SignRecognizer

logger = get_logger(__name__)

_DATA_URL_PREFIX = "base64,"


class FrameDecodeError(ValueError):
    """The client sent something that is not a decodable image."""


@dataclass(frozen=True, slots=True)
class RecognitionStats:
    frames: int
    total_ms: float

    @property
    def average_ms(self) -> float:
        return self.total_ms / self.frames if self.frames else 0.0


class RecognitionService:
    def __init__(
        self,
        settings: Settings,
        recognizer: SignRecognizer | None = None,
    ) -> None:
        self._settings = settings
        self._recognizer = recognizer or SignRecognizer(
            classifier_path=settings.resolved_model_path()
        )
        # One worker: MediaPipe detection is serialised by a lock anyway, so
        # more threads would only add contention and reorder frames.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="recognizer"
        )
        self._frames = 0
        self._total_ms = 0.0

    @property
    def classes(self) -> List[str]:
        return self._recognizer.classes

    @property
    def stats(self) -> RecognitionStats:
        return RecognitionStats(frames=self._frames, total_ms=self._total_ms)

    # ------------------------------------------------------------------

    def decode_frame(self, payload: str, *, mirrored: bool = True) -> np.ndarray:
        """Decode a base64 (optionally data-URL) JPEG/PNG into an RGB array.

        ``mirrored`` says whether the client already flipped the image.  The
        classifier was trained on mirrored frames -- the original capture
        scripts ran ``cv2.flip(frame, 1)`` -- so if the client did not flip,
        we do it here to keep the input distribution identical.
        """
        import cv2

        if not payload:
            raise FrameDecodeError("empty frame payload")

        index = payload.find(_DATA_URL_PREFIX)
        if index != -1:
            payload = payload[index + len(_DATA_URL_PREFIX) :]

        try:
            raw = base64.b64decode(payload, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise FrameDecodeError("frame is not valid base64") from exc

        if not raw:
            raise FrameDecodeError("empty frame payload")
        if len(raw) > self._settings.max_frame_bytes:
            raise FrameDecodeError(
                f"frame is too large ({len(raw)} bytes)"
            )

        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise FrameDecodeError("frame could not be decoded as an image")

        if not mirrored:
            image = cv2.flip(image, 1)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------

    def recognize_sync(self, rgb_image: np.ndarray) -> RecognitionResult:
        result = self._recognizer.recognize(rgb_image)
        self._frames += 1
        self._total_ms += result.elapsed_ms
        return result

    async def recognize(self, rgb_image: np.ndarray) -> RecognitionResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self.recognize_sync, rgb_image
        )

    def recognize_encoded_sync(
        self, payload: str, mirrored: bool = True
    ) -> RecognitionResult:
        return self.recognize_sync(self.decode_frame(payload, mirrored=mirrored))

    async def recognize_encoded(
        self, payload: str, *, mirrored: bool = True
    ) -> RecognitionResult:
        """Decode and recognise in a single hop to the worker thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self.recognize_encoded_sync, payload, mirrored
        )

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._recognizer.close()


def landmarks_payload(result: RecognitionResult) -> List[Tuple[float, float]]:
    """Landmarks rounded to 4 decimals -- plenty for a canvas overlay, and it
    roughly halves the size of every websocket message."""
    return [(round(x, 4), round(y, 4)) for x, y in result.landmarks_xy]


__all__ = [
    "RecognitionService",
    "RecognitionStats",
    "FrameDecodeError",
    "landmarks_payload",
]
