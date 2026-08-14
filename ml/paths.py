"""Filesystem locations, resolved from this file so that scripts work no
matter which directory they are launched from."""

from pathlib import Path

# ml/paths.py -> ml/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"

#: MediaPipe hand-landmark detector (21 landmarks per hand).
HAND_LANDMARKER_PATH = MODELS_DIR / "hand_landmarker.task"

#: The trained scikit-learn classifier that maps 63 features -> a letter.
SIGN_MODEL_PATH = MODELS_DIR / "sign_model.pkl"

#: Where generated sign illustrations are written.
SIGN_ASSETS_DIR = PROJECT_ROOT / "frontend" / "public" / "signs"

__all__ = [
    "PROJECT_ROOT",
    "DATASET_DIR",
    "MODELS_DIR",
    "HAND_LANDMARKER_PATH",
    "SIGN_MODEL_PATH",
    "SIGN_ASSETS_DIR",
]
