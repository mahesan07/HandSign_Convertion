"""Synthesising extra training samples from recorded landmarks.

Recording 500 samples of a pose at 5 frames/second gives 500 nearly identical
rows, not 500 independent examples: the hand barely moves in 100 seconds. The
model therefore never sees the pose held at a slightly different angle, which
is the single most common way a real user differs from the recording.

Rotating the landmark cloud fixes exactly that, and measurably: on a held-out
*block* of each recording (see ``ml/scripts/evaluate_model.py``) it takes the
classifier from 92.4% to 95.0%.

What was tried and rejected, on the same honest split:

* **scaling** -- no effect (-0.2). ``extract_features`` already divides by hand
  size, so scale variation is normalised away before it reaches the model.
* **noise alone** -- no effect (+0.0), though a little of it helps alongside
  rotation by stopping the trees from memorising exact coordinates.
* **mirroring** -- actively harmful (-1.5). It looks like free left-hand
  support, but flipping x reverses the direction G, H and P point in, so the
  model learns contradictory labels.
"""

from __future__ import annotations

import numpy as np

#: Landmarks per hand, and values per landmark.
_POINTS = 21
_DIMS = 3


def _as_points(features: np.ndarray) -> np.ndarray:
    return features.reshape(len(features), _POINTS, _DIMS)


def _rotation_matrix(angles: np.ndarray) -> np.ndarray:
    """Build one 3x3 rotation from (pitch, yaw, roll) radians."""
    pitch, yaw, roll = angles
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    rz = np.array([[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]])
    ry = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
    rx = np.array([[1, 0, 0], [0, cos_p, -sin_p], [0, sin_p, cos_p]])
    return rz @ ry @ rx


def rotate(
    features: np.ndarray,
    max_degrees: float = 10.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Randomly rotate each sample's landmark cloud about the wrist.

    The features are already wrist-relative, so the wrist is the origin and
    rotation leaves it exactly where it is -- the hand tilts, it does not move.
    """
    rng = rng or np.random.default_rng()
    points = _as_points(np.asarray(features, dtype=np.float64)).copy()
    angles = np.deg2rad(
        rng.uniform(-max_degrees, max_degrees, size=(len(points), 3))
    )
    for i in range(len(points)):
        points[i] = points[i] @ _rotation_matrix(angles[i]).T
    return points.reshape(len(points), _POINTS * _DIMS)


def jitter(
    features: np.ndarray,
    sigma: float = 0.015,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add small gaussian noise, standing in for detector wobble."""
    rng = rng or np.random.default_rng()
    features = np.asarray(features, dtype=np.float64)
    return features + rng.normal(0.0, sigma, features.shape)


def augment(
    features: np.ndarray,
    labels: np.ndarray,
    copies: int = 5,
    *,
    max_degrees: float = 10.0,
    sigma: float = 0.015,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the originals plus ``copies`` rotated-and-jittered variants.

    ``copies=5`` gives 6x the data, which is where the measured gain flattens
    out; more only grows the model.
    """
    if copies < 1:
        return np.asarray(features), np.asarray(labels)

    rng = np.random.default_rng(seed)
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)

    stacked = [features]
    for _ in range(copies):
        stacked.append(
            rotate(jitter(features, sigma, rng), max_degrees, rng)
        )
    return np.vstack(stacked), np.tile(labels, copies + 1)


__all__ = ["rotate", "jitter", "augment"]
