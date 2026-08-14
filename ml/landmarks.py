"""Landmark value type and the hand skeleton topology.

MediaPipe returns its own ``NormalizedLandmark`` objects.  Everything in this
project only ever reads ``.x`` / ``.y`` / ``.z``, so :class:`Landmark` is a
drop-in stand-in that lets us rebuild a hand from a CSV row or a JSON payload
without depending on MediaPipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

#: Number of landmarks MediaPipe emits for one hand.
NUM_LANDMARKS = 21

#: Bones of the hand, as (start, end) landmark indices.  Used for drawing the
#: skeleton overlay in the UI and for the generated sign illustrations.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # little
    (0, 17),                                   # palm base
)

#: Landmark indices of the five finger tips, in thumb -> little order.
FINGERTIPS: tuple[int, ...] = (4, 8, 12, 16, 20)


@dataclass(frozen=True, slots=True)
class Landmark:
    """A single 3D hand landmark in MediaPipe's normalized image space."""

    x: float
    y: float
    z: float


def landmarks_from_flat(values: Sequence[float]) -> List[Landmark]:
    """Rebuild 21 landmarks from a flat ``[x0, y0, z0, x1, ...]`` sequence."""
    if len(values) != NUM_LANDMARKS * 3:
        raise ValueError(
            f"expected {NUM_LANDMARKS * 3} values, got {len(values)}"
        )
    return [
        Landmark(float(values[i]), float(values[i + 1]), float(values[i + 2]))
        for i in range(0, len(values), 3)
    ]


def landmarks_to_flat(landmarks: Iterable[Landmark]) -> List[float]:
    """Flatten landmarks back into ``[x0, y0, z0, x1, ...]``."""
    flat: List[float] = []
    for lm in landmarks:
        flat.extend((lm.x, lm.y, lm.z))
    return flat


__all__ = [
    "NUM_LANDMARKS",
    "HAND_CONNECTIONS",
    "FINGERTIPS",
    "Landmark",
    "landmarks_from_flat",
    "landmarks_to_flat",
]
