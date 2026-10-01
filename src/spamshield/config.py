"""Runtime configuration, read once from environment variables.

Every setting has a safe default, so the app runs with no ``.env`` file at all.
See ``.env.example`` for the full list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "spamshield.joblib"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "sms_spam.csv"


class ConfigError(ValueError):
    """Raised when an environment variable holds an invalid value."""


def _env_float(name: str, default: float, *, low: float, high: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if not low <= value <= high:
        raise ConfigError(f"{name} must be between {low} and {high}, got {value}")
    return value


def _env_int(name: str, default: int, *, low: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value < low:
        raise ConfigError(f"{name} must be at least {low}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    """Application settings. Build with :meth:`from_env`."""

    model_path: Path = DEFAULT_MODEL_PATH
    #: Spam probability at or above which a message is labelled spam.
    threshold: float = 0.5
    #: Upper bound on messages read from one mailbox, to keep scans bounded.
    max_messages: int = 5_000
    #: Characters of a single message passed to the model; the rest is ignored.
    max_chars: int = 20_000
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        model_path = os.environ.get("SPAMSHIELD_MODEL_PATH")
        return cls(
            model_path=Path(model_path).expanduser() if model_path else DEFAULT_MODEL_PATH,
            threshold=_env_float("SPAMSHIELD_THRESHOLD", cls.threshold, low=0.01, high=0.99),
            max_messages=_env_int("SPAMSHIELD_MAX_MESSAGES", cls.max_messages, low=1),
            max_chars=_env_int("SPAMSHIELD_MAX_CHARS", cls.max_chars, low=100),
            log_level=os.environ.get("SPAMSHIELD_LOG_LEVEL", cls.log_level).upper(),
        )
