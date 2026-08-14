"""Regression guard on the shipped model.

The refactor changed *how* the classifier is called (``n_jobs=1``, one
``predict_proba`` instead of ``predict`` + ``predict_proba``). These tests
assert that the change is invisible: same labels, same probabilities.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ml.classifier import SignClassifier
from ml.paths import DATASET_DIR, SIGN_MODEL_PATH

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def classifier() -> SignClassifier:
    if not SIGN_MODEL_PATH.exists():
        pytest.skip("models/sign_model.pkl is not present")
    return SignClassifier()


def sample_rows(letter: str, limit: int = 20) -> list[list[float]]:
    path: Path = DATASET_DIR / f"{letter}.csv"
    if not path.exists():
        pytest.skip(f"dataset/{letter}.csv is not present")
    rows: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            rows.append([float(value) for value in row[:63]])
            if len(rows) >= limit:
                break
    return rows


def test_knows_the_full_alphabet(classifier: SignClassifier):
    assert classifier.classes == [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    assert classifier.n_features == 63


def test_recognises_its_own_training_data(classifier: SignClassifier):
    """A sanity floor: the model must still classify the data it learned from."""
    for letter in ["A", "E", "L", "S", "W"]:
        rows = sample_rows(letter)
        predictions = [classifier.predict(row).letter for row in rows]
        correct = sum(1 for p in predictions if p == letter)
        assert correct / len(predictions) >= 0.9, (
            f"{letter}: only {correct}/{len(predictions)} correct"
        )


def test_confidence_is_a_probability(classifier: SignClassifier):
    prediction = classifier.predict(sample_rows("B", limit=1)[0])
    assert 0.0 <= prediction.confidence <= 1.0
    assert prediction.alternatives[0] == (
        prediction.letter,
        prediction.confidence,
    )
    # Alternatives come back best-first.
    probabilities = [probability for _, probability in prediction.alternatives]
    assert probabilities == sorted(probabilities, reverse=True)


def test_matches_the_original_call_path(classifier: SignClassifier):
    """``argmax(predict_proba)`` must equal what plain ``predict`` returned.

    This is the exact equivalence the latency optimisation relies on.
    """
    import joblib
    import numpy as np

    original = joblib.load(SIGN_MODEL_PATH)
    rows = [row for letter in "ACGMTZ" for row in sample_rows(letter, limit=5)]
    batch = np.asarray(rows)

    expected_labels = original.predict(batch)
    expected_confidence = original.predict_proba(batch).max(axis=1)

    for row, label, confidence in zip(rows, expected_labels, expected_confidence):
        prediction = classifier.predict(row)
        assert prediction.letter == label
        assert prediction.confidence == pytest.approx(confidence, abs=1e-12)


def test_rejects_the_wrong_number_of_features(classifier: SignClassifier):
    with pytest.raises(ValueError, match="expected 63 features"):
        classifier.predict([0.0] * 10)


def test_missing_model_file_explains_itself(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_model"):
        SignClassifier(tmp_path / "nope.pkl")
