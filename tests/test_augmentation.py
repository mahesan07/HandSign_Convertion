"""Landmark augmentation.

The properties that matter: the wrist must stay at the origin (the features are
defined relative to it), the label must stay valid, and augmented samples must
be *different* from the originals but still recognisably the same sign.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.augmentation import augment, jitter, rotate
from ml.features import extract_features
from ml.landmarks import Landmark


@pytest.fixture
def samples() -> np.ndarray:
    """Twelve plausible feature vectors."""
    rows = []
    for k in range(12):
        hand = [
            Landmark(0.1 + i * 0.011 + k * 0.002, 0.2 + i * 0.013, 0.05 + i * 0.007)
            for i in range(21)
        ]
        rows.append(extract_features(hand))
    return np.asarray(rows)


def test_rotate_preserves_shape(samples):
    assert rotate(samples).shape == samples.shape


def test_rotate_keeps_the_wrist_at_the_origin(samples):
    """The wrist is landmark 0 and defines the coordinate system."""
    rotated = rotate(samples, rng=np.random.default_rng(0))
    assert rotated[:, :3] == pytest.approx(0.0, abs=1e-12)


def test_rotate_preserves_distances_from_the_wrist(samples):
    """A rotation is rigid: it must not stretch the hand."""
    rotated = rotate(samples, rng=np.random.default_rng(0))
    before = np.linalg.norm(samples.reshape(-1, 21, 3), axis=2)
    after = np.linalg.norm(rotated.reshape(-1, 21, 3), axis=2)
    assert after == pytest.approx(before, abs=1e-10)


def test_rotate_actually_changes_the_sample(samples):
    rotated = rotate(samples, max_degrees=10.0, rng=np.random.default_rng(0))
    assert not np.allclose(rotated, samples)


def test_rotate_by_zero_degrees_is_the_identity(samples):
    assert rotate(samples, max_degrees=0.0) == pytest.approx(samples, abs=1e-12)


def test_rotation_is_bounded(samples):
    """Small tilts only -- a 90 degree rotation would be a different sign."""
    rotated = rotate(samples, max_degrees=10.0, rng=np.random.default_rng(1))
    displacement = np.linalg.norm(
        rotated.reshape(-1, 21, 3) - samples.reshape(-1, 21, 3), axis=2
    )
    radius = np.linalg.norm(samples.reshape(-1, 21, 3), axis=2)
    moved = displacement[radius > 0.1] / radius[radius > 0.1]
    assert moved.max() < 0.35


def test_jitter_is_small_and_random(samples):
    noisy = jitter(samples, sigma=0.015, rng=np.random.default_rng(0))
    assert noisy.shape == samples.shape
    assert not np.allclose(noisy, samples)
    assert np.abs(noisy - samples).mean() < 0.05


def test_augment_multiplies_the_dataset(samples):
    labels = np.array(list("ABCDEFGHIJKL"))
    X, y = augment(samples, labels, copies=5)
    assert len(X) == len(samples) * 6
    assert len(y) == len(X)


def test_augment_keeps_labels_aligned(samples):
    labels = np.array(list("ABCDEFGHIJKL"))
    X, y = augment(samples, labels, copies=3)
    # The originals come first, then each copy in the same order.
    assert list(y[: len(labels)]) == list(labels)
    for copy in range(4):
        block = y[copy * len(labels) : (copy + 1) * len(labels)]
        assert list(block) == list(labels)
    assert X[: len(samples)] == pytest.approx(samples)


def test_augment_with_zero_copies_is_a_no_op(samples):
    labels = np.array(list("ABCDEFGHIJKL"))
    X, y = augment(samples, labels, copies=0)
    assert X.shape == samples.shape
    assert len(y) == len(labels)


def test_augment_is_deterministic(samples):
    labels = np.array(list("ABCDEFGHIJKL"))
    first, _ = augment(samples, labels, copies=2, seed=7)
    second, _ = augment(samples, labels, copies=2, seed=7)
    assert first == pytest.approx(second)


def test_augmented_features_stay_in_a_sane_range(samples):
    """Nothing should explode -- these feed straight into the classifier."""
    labels = np.array(list("ABCDEFGHIJKL"))
    X, _ = augment(samples, labels, copies=5)
    assert np.isfinite(X).all()
    assert np.abs(X).max() < np.abs(samples).max() * 2 + 1
