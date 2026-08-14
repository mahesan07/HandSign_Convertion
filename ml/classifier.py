"""Thin wrapper around the trained scikit-learn sign classifier.

The model itself (``models/sign_model.pkl``) is **not** modified or retrained
here -- this module only changes *how* it is called, for two measured reasons:

1. ``RandomForestClassifier`` was pickled with ``n_jobs=-1``.  For a single
   63-feature sample the joblib worker dispatch dominates: 28 ms per call
   versus 5.6 ms with ``n_jobs=1``.  Parallelism only pays off for batches.
2. The original live loop called ``predict()`` *and* ``predict_proba()``,
   doing the same forest traversal twice (52 ms/frame).  For a random forest,
   ``predict`` is by definition ``classes_[argmax(predict_proba)]``, so one
   call gives both the label and the confidence.

Together those take the classifier from ~19 fps to ~180 fps with bit-identical
outputs (asserted in ``tests/test_classifier.py``).
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from ml.paths import SIGN_MODEL_PATH

#: The model was fitted from a pandas DataFrame, so it remembers column names
#: and warns when handed a bare ndarray.  Building a 1x63 DataFrame per frame
#: costs ~1.2 ms purely to silence that, so we mute the warning instead.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)


@dataclass(frozen=True, slots=True)
class Prediction:
    """One classifier verdict."""

    letter: str
    confidence: float
    #: ``(letter, probability)`` pairs, highest first.  Useful for the UI's
    #: "did you mean" hints and for debugging confusable signs (M/N/S/T).
    alternatives: Tuple[Tuple[str, float], ...] = ()


class SignClassifier:
    """Loads ``sign_model.pkl`` once and predicts letters from 63 features.

    The instance is cheap to share: :meth:`predict` is stateless apart from a
    lock that guards the (not documented as thread-safe) estimator.
    """

    def __init__(
        self,
        model_path: Path | str = SIGN_MODEL_PATH,
        *,
        top_k: int = 3,
    ) -> None:
        import joblib  # imported lazily: keeps `import ml` light for tests

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained classifier not found at {self.model_path}. "
                "Run `python -m ml.scripts.train_model` to create it."
            )

        self._model = joblib.load(self.model_path)
        # See the module docstring: single-sample inference is ~5x faster
        # without the joblib worker pool.  This does not touch the trees.
        if hasattr(self._model, "n_jobs"):
            self._model.n_jobs = 1

        # The model was fitted from a DataFrame, so scikit-learn compares
        # column names on every call and warns when handed a plain array --
        # 15 times a second, forever.  Dropping the remembered names is the
        # cheap fix: it is metadata only, the trees and ``n_features_in_``
        # are untouched, and we validate the vector length ourselves below.
        if hasattr(self._model, "feature_names_in_"):
            self.feature_names = [str(n) for n in self._model.feature_names_in_]
            del self._model.feature_names_in_
        else:
            self.feature_names = []

        self._classes: np.ndarray = np.asarray(self._model.classes_)
        self._n_features: int = int(getattr(self._model, "n_features_in_", 63))
        self._top_k = max(1, min(top_k, len(self._classes)))
        self._lock = threading.Lock()

        # Warm the estimator so the first real frame is not the slow one.
        self._model.predict_proba(np.zeros((1, self._n_features)))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def classes(self) -> List[str]:
        """The letters this model can recognise, e.g. ``['A', ..., 'Z']``."""
        return [str(c) for c in self._classes]

    @property
    def n_features(self) -> int:
        return self._n_features

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, features: Sequence[float]) -> Prediction:
        """Classify one feature vector produced by :func:`ml.extract_features`."""
        if len(features) != self._n_features:
            raise ValueError(
                f"expected {self._n_features} features, got {len(features)}"
            )

        sample = np.asarray(features, dtype=np.float64).reshape(1, -1)
        with self._lock:
            probabilities = self._model.predict_proba(sample)[0]

        order = np.argsort(probabilities)[::-1]
        best = order[0]
        alternatives = tuple(
            (str(self._classes[i]), float(probabilities[i]))
            for i in order[: self._top_k]
        )
        return Prediction(
            letter=str(self._classes[best]),
            confidence=float(probabilities[best]),
            alternatives=alternatives,
        )


__all__ = ["SignClassifier", "Prediction"]
