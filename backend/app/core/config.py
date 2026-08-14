"""Application settings.

Every tunable lives here and can be overridden from ``.env`` or the process
environment -- nothing in the recognition or suggestion pipeline hard-codes a
threshold.  See ``.env.example`` for the full list.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, List

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from ml.paths import PROJECT_ROOT, SIGN_MODEL_PATH


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    app_name: str = "HandSign Conversion"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    #: Origins allowed to call the API.  The Vite dev server runs on 5173.
    #: ``NoDecode`` is required: without it pydantic-settings tries to JSON
    #: parse any list-typed environment variable before validators run, so a
    #: plain ``CORS_ORIGINS=http://a,http://b`` in .env would blow up.
    cors_origins: Annotated[List[str], NoDecode] = Field(
        default=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ]
    )

    # ------------------------------------------------------------------
    # Recognition / stabilisation
    # ------------------------------------------------------------------
    #: Which classifier to load.  Left unset it uses models/sign_model.pkl.
    #: Point it at another .pkl to A/B a retrained model without moving files:
    #:     SIGN_MODEL_PATH=models/sign_model_augmented.pkl
    sign_model_path: Path | None = None
    #: A frame at or above this confidence counts as one full frame of
    #: evidence. Live webcam confidence is far below what the model scores on
    #: recorded data, so this is deliberately not near 1.0.
    min_confidence: float = 0.55
    #: Below this a frame counts for nothing; between the two it counts
    #: proportionally, which keeps the progress bar moving on a decent sign.
    partial_confidence: float = 0.30
    #: Evidence needed to commit a letter, in full-confidence frames.
    stable_frames: int = 6
    #: How fast a rival letter's evidence fades. Lower = less flicker
    #: tolerance; below ~0.7 similar letters cancel out and nothing types.
    prediction_decay: float = 0.8
    #: No letter may be committed within this long of the previous one.
    commit_cooldown_ms: int = 450
    #: Re-committing the *same* letter additionally requires this gap.
    duplicate_suppression_ms: int = 900
    #: Frames without a confident hand needed to release the current pose.
    release_frames: int = 4
    #: Reject oversized frames on the websocket (bytes, after base64 decode).
    max_frame_bytes: int = 2 * 1024 * 1024

    # ------------------------------------------------------------------
    # Suggestions
    # ------------------------------------------------------------------
    max_word_suggestions: int = 4
    max_sentence_suggestions: int = 3
    #: Quiet period before a Gemini call is actually issued.
    suggestion_debounce_ms: int = 350
    suggestion_cache_size: int = 512
    suggestion_cache_ttl_seconds: int = 900

    # ------------------------------------------------------------------
    # Gemini
    # ------------------------------------------------------------------
    gemini_api_key: SecretStr | None = None
    #: A rolling alias, on purpose.  Pinned Gemini versions get retired and
    #: start returning 404, which is a confusing failure for something that is
    #: meant to be optional; "-latest" keeps working.  The lite tier is the
    #: right trade-off for a predictive keyboard: ~1.3 s versus ~3 s for the
    #: full flash model, for suggestions of the same quality.
    gemini_model: str = "gemini-flash-lite-latest"
    #: Generous, because a late suggestion costs nothing -- it arrives on the
    #: websocket and is discarded if the user has already typed on.
    gemini_timeout_seconds: float = 6.0
    gemini_max_output_tokens: int = 256
    gemini_temperature: float = 0.2

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------
    session_ttl_seconds: int = 3600
    max_sessions: int = 64

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Allow ``CORS_ORIGINS=http://a,http://b`` in the .env file."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("sign_model_path", "gemini_api_key", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        """Treat ``KEY=`` in a .env file as "not set".

        Without this, an empty ``SIGN_MODEL_PATH=`` becomes ``Path(".")`` and
        the app tries to load the project directory as a model file.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def resolved_model_path(self) -> Path:
        """The classifier to load, absolute, defaulting to the shipped one."""
        if self.sign_model_path is None:
            return SIGN_MODEL_PATH
        path = Path(self.sign_model_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def gemini_enabled(self) -> bool:
        key = self.gemini_api_key.get_secret_value() if self.gemini_api_key else ""
        return bool(key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance -- the .env file is read once per process."""
    return Settings()


__all__ = ["Settings", "get_settings"]
