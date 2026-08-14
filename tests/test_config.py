"""Settings loading -- especially from a real .env file.

These exist because a hand-written .env behaves differently from constructor
keyword arguments, and the difference is easy to miss until someone actually
writes one.
"""

from __future__ import annotations

import pytest

from backend.app.core.config import Settings

ENV_TEMPLATE = """\
GEMINI_API_KEY={key}
GEMINI_MODEL=gemini-2.0-flash
MIN_CONFIDENCE=0.75
STABLE_FRAMES=8
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173,https://example.com
"""


def write_env(tmp_path, key: str = "test-key"):
    path = tmp_path / ".env"
    path.write_text(ENV_TEMPLATE.format(key=key), encoding="utf-8")
    return Settings(_env_file=str(path))


def test_reads_values_from_a_dotenv_file(tmp_path):
    settings = write_env(tmp_path)
    assert settings.min_confidence == 0.75
    assert settings.stable_frames == 8
    assert settings.gemini_model == "gemini-2.0-flash"


def test_comma_separated_cors_origins_parse(tmp_path):
    """A plain comma-separated list must work; nobody writes JSON in a .env."""
    settings = write_env(tmp_path)
    assert settings.cors_origins == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://example.com",
    ]


def test_cors_origins_still_accept_a_real_list():
    assert Settings(_env_file=None, cors_origins=["http://a"]).cors_origins == [
        "http://a"
    ]


def test_defaults_apply_with_no_env_file():
    settings = Settings(_env_file=None)
    assert settings.cors_origins
    assert settings.gemini_enabled is False


def test_a_key_enables_gemini(tmp_path):
    assert write_env(tmp_path, key="some-key").gemini_enabled is True


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_key_does_not_enable_gemini(tmp_path, blank):
    assert write_env(tmp_path, key=blank).gemini_enabled is False


def test_whitespace_around_the_equals_sign_is_tolerated(tmp_path):
    """`KEY = value` is what people actually type."""
    path = tmp_path / ".env"
    path.write_text("GEMINI_API_KEY = spaced-key\n", encoding="utf-8")
    settings = Settings(_env_file=str(path))
    assert settings.gemini_enabled is True
    assert settings.gemini_api_key.get_secret_value() == "spaced-key"


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_model_path_means_unset(blank):
    """`SIGN_MODEL_PATH=` in a .env must not resolve to the project folder."""
    from ml.paths import SIGN_MODEL_PATH

    settings = Settings(_env_file=None, sign_model_path=blank)
    assert settings.sign_model_path is None
    assert settings.resolved_model_path() == SIGN_MODEL_PATH


def test_a_blank_model_path_in_a_real_env_file(tmp_path):
    from ml.paths import SIGN_MODEL_PATH

    path = tmp_path / ".env"
    path.write_text("SIGN_MODEL_PATH=\nGEMINI_API_KEY=\n", encoding="utf-8")
    settings = Settings(_env_file=str(path))
    assert settings.resolved_model_path() == SIGN_MODEL_PATH
    assert settings.gemini_enabled is False


def test_model_path_defaults_to_the_shipped_model():
    from ml.paths import SIGN_MODEL_PATH

    assert Settings(_env_file=None).resolved_model_path() == SIGN_MODEL_PATH


def test_a_relative_model_path_resolves_against_the_project_root():
    from ml.paths import PROJECT_ROOT

    settings = Settings(
        _env_file=None, sign_model_path="models/sign_model_augmented.pkl"
    )
    resolved = settings.resolved_model_path()
    assert resolved.is_absolute()
    assert resolved == PROJECT_ROOT / "models" / "sign_model_augmented.pkl"


def test_an_absolute_model_path_is_left_alone(tmp_path):
    target = tmp_path / "other.pkl"
    settings = Settings(_env_file=None, sign_model_path=str(target))
    assert settings.resolved_model_path() == target


def test_the_key_is_not_exposed_by_repr(tmp_path):
    """A SecretStr must not leak into logs or error messages."""
    settings = write_env(tmp_path, key="super-secret-value")
    assert "super-secret-value" not in repr(settings)
    assert "super-secret-value" not in str(settings.gemini_api_key)
