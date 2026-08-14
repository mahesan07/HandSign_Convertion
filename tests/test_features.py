"""The feature contract.

``extract_features`` defines the input space the shipped model was trained on.
If these tests change, the model must be retrained -- that is the whole point
of pinning them down.
"""

from __future__ import annotations

import math

import pytest

from ml.features import extract_features
from ml.landmarks import Landmark


def make_hand(offset=(0.0, 0.0, 0.0), scale=1.0):
    """21 landmarks in a fixed arrangement, optionally moved and resized."""
    ox, oy, oz = offset
    return [
        Landmark(
            ox + scale * (0.10 + i * 0.011),
            oy + scale * (0.20 + i * 0.013),
            oz + scale * (0.05 + i * 0.007),
        )
        for i in range(21)
    ]


def test_produces_63_features():
    assert len(extract_features(make_hand())) == 63


def test_wrist_is_the_origin():
    features = extract_features(make_hand())
    assert features[:3] == [0.0, 0.0, 0.0]


def test_is_translation_invariant():
    """The same sign in a different part of the frame is the same features."""
    a = extract_features(make_hand())
    b = extract_features(make_hand(offset=(0.4, -0.2, 0.05)))
    assert a == pytest.approx(b, abs=1e-12)


def test_is_scale_invariant():
    """A hand nearer the camera produces the same features."""
    a = extract_features(make_hand())
    b = extract_features(make_hand(scale=2.5))
    assert a == pytest.approx(b, rel=1e-9)


def test_normalises_by_the_wrist_to_middle_mcp_distance():
    hand = make_hand()
    features = extract_features(hand)
    # Landmark 9 is the middle-finger MCP, and it defines the unit of scale,
    # so its distance from the origin must be exactly 1.
    x, y, z = features[27], features[28], features[29]
    assert math.sqrt(x * x + y * y + z * z) == pytest.approx(1.0)


def test_degenerate_hand_does_not_divide_by_zero():
    """All landmarks in one spot must not raise -- a real detector can do this."""
    flat = [Landmark(0.5, 0.5, 0.5)] * 21
    features = extract_features(flat)
    assert len(features) == 63
    assert all(value == 0.0 for value in features)
