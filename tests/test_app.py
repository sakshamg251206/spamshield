"""UI tests that drive the real Streamlit script headlessly."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "app.py")


@pytest.fixture
def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def markdown_text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_first_visit_explains_the_app(app: AppTest) -> None:
    assert "Not sure whether a message is a scam?" in markdown_text(app)
    assert [t.label for t in app.tabs] == [
        "✉️  Check a message",
        "📥  Scan a mailbox",
        "📊  Accuracy",
    ]
    check = next(b for b in app.button if b.label == "Check message")
    assert check.disabled  # nothing to check yet
    assert any("Your result will appear here" in i.value for i in app.info)


def test_example_button_fills_and_checks_spam(app: AppTest) -> None:
    next(b for b in app.button if b.label == "Prize scam").click().run()
    assert "CONGRATULATIONS" in app.text_area[0].value
    next(b for b in app.button if b.label == "Check message").click().run()
    assert not app.exception
    text = markdown_text(app)
    assert "spam</h3>" in text and "not spam" not in text
    assert "chip-spam" in text


def test_normal_message_is_not_spam(app: AppTest) -> None:
    app.text_area[0].input("Hi Mum, I'll be home for dinner around 7. Love you").run()
    next(b for b in app.button if b.label == "Check message").click().run()
    assert "not spam</h3>" in markdown_text(app)


def test_accuracy_tab_shows_model_card(app: AppTest) -> None:
    labels = [m.label for m in app.metric]
    assert {"Spam caught", "Flags that were right", "False alarms", "F1 score"} <= set(labels)


def test_missing_model_shows_friendly_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SPAMSHIELD_MODEL_PATH", str(tmp_path / "missing.joblib"))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert any("could not start" in e.value for e in at.error)
    assert any("spamshield train" in m.value for m in at.markdown)
