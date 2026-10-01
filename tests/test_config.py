from pathlib import Path

import pytest

from spamshield.config import DEFAULT_MODEL_PATH, ConfigError, Settings


def test_defaults_without_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SPAMSHIELD_MODEL_PATH", "SPAMSHIELD_THRESHOLD", "SPAMSHIELD_MAX_MESSAGES"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env()
    assert settings.model_path == DEFAULT_MODEL_PATH
    assert settings.threshold == 0.5


def test_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPAMSHIELD_MODEL_PATH", "/tmp/model.joblib")
    monkeypatch.setenv("SPAMSHIELD_THRESHOLD", "0.7")
    monkeypatch.setenv("SPAMSHIELD_MAX_MESSAGES", "10")
    monkeypatch.setenv("SPAMSHIELD_LOG_LEVEL", "debug")
    settings = Settings.from_env()
    assert settings.model_path == Path("/tmp/model.joblib")
    assert settings.threshold == 0.7
    assert settings.max_messages == 10
    assert settings.log_level == "DEBUG"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SPAMSHIELD_THRESHOLD", "high"),
        ("SPAMSHIELD_THRESHOLD", "1.5"),
        ("SPAMSHIELD_MAX_MESSAGES", "0"),
        ("SPAMSHIELD_MAX_MESSAGES", "ten"),
    ],
)
def test_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError, match=name):
        Settings.from_env()
