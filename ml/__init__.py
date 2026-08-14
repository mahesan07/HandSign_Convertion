"""Machine-learning layer for HandSign Conversion.

This package owns everything that touches the camera, MediaPipe or the trained
sign classifier.  It has **no** web-framework dependencies so it can be used
from the FastAPI backend, from the offline CLI demo and from the tests alike.

The public surface is small on purpose:

    from ml import SignRecognizer, RecognitionResult, extract_features
"""

from ml.features import extract_features
from ml.landmarks import Landmark, HAND_CONNECTIONS
from ml.classifier import SignClassifier, Prediction
from ml.detector import HandDetector
from ml.recognizer import SignRecognizer, RecognitionResult

__all__ = [
    "extract_features",
    "Landmark",
    "HAND_CONNECTIONS",
    "SignClassifier",
    "Prediction",
    "HandDetector",
    "SignRecognizer",
    "RecognitionResult",
]
