import json
import shutil
from pathlib import Path

import pytest

from spamshield.classifier import ModelLoadError, Prediction, SpamClassifier
from spamshield.model_card import card_path_for
from spamshield.presentation import EXAMPLES


def copy_model(src: Path, dest_dir: Path) -> Path:
    dest = dest_dir / src.name
    shutil.copy(src, dest)
    shutil.copy(card_path_for(src), card_path_for(dest))
    return dest


def test_training_writes_a_complete_model_card(trained_model: Path) -> None:
    card = json.loads(card_path_for(trained_model).read_text())
    assert card["model_name"] in {"logistic_regression", "naive_bayes"}
    assert len(card["candidates"]) == 2
    assert len(card["model_sha256"]) == 64
    assert 0.5 < card["test_metrics"]["f1"] <= 1
    assert card["n_train"] + card["n_test"] == card["n_spam"] + card["n_ham"]


def test_predicts_obvious_cases(classifier: SpamClassifier) -> None:
    spam = classifier.predict("WINNER! Claim your FREE prize, call 09061701461 now")
    ham = classifier.predict("Are we still meeting for lunch tomorrow?")
    assert spam.is_spam and spam.spam_probability > 0.5
    assert not ham.is_spam and ham.spam_probability < 0.5


def test_predict_rejects_empty_text(classifier: SpamClassifier) -> None:
    with pytest.raises(ValueError, match="empty"):
        classifier.predict("   ")


def test_predict_many_matches_predict_and_handles_blanks(classifier: SpamClassifier) -> None:
    texts = ["Free entry! Text WIN to 87121", "", "see you soon"]
    batch = classifier.predict_many(texts)
    assert batch[1] == Prediction("ham", 0.0)
    assert batch[0] == classifier.predict(texts[0])
    assert classifier.predict_many([]) == []


def test_long_input_is_clipped(classifier: SpamClassifier) -> None:
    assert classifier.predict("hello " * 50_000).label in {"spam", "ham"}


def test_threshold_controls_label(trained_model: Path) -> None:
    text = "Call now to claim"
    strict = SpamClassifier.load(trained_model, threshold=0.99)
    lenient = SpamClassifier.load(trained_model, threshold=0.01)
    assert strict.predict(text).spam_probability == lenient.predict(text).spam_probability
    assert lenient.predict(text).is_spam


@pytest.mark.parametrize("threshold", [0, 1, 1.5])
def test_invalid_threshold(classifier: SpamClassifier, threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        SpamClassifier(classifier.pipeline, threshold=threshold)


def test_confidence_is_probability_of_predicted_label() -> None:
    assert Prediction("spam", 0.8).confidence == 0.8
    assert Prediction("ham", 0.2).confidence == pytest.approx(0.8)


def test_explain_surfaces_spam_signals(classifier: SpamClassifier) -> None:
    signals = classifier.explain("WINNER! Claim your FREE prize, call 09061701461 now")
    assert signals, "expected at least one influential word"
    assert signals == sorted(signals, key=lambda s: abs(s.impact), reverse=True)
    assert any(s.impact > 0 for s in signals)
    assert "a phone number" in {s.display for s in signals}


def test_explain_handles_unknown_or_empty_text(classifier: SpamClassifier) -> None:
    assert classifier.explain("") == []
    assert classifier.explain("zzqxv qqwxz") == []


def test_load_refuses_tampered_model(trained_model: Path, tmp_path: Path) -> None:
    model = copy_model(trained_model, tmp_path)
    with model.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ModelLoadError, match="Checksum mismatch"):
        SpamClassifier.load(model)


def test_load_requires_model_and_card(trained_model: Path, tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="No model found"):
        SpamClassifier.load(tmp_path / "missing.joblib")
    model = copy_model(trained_model, tmp_path)
    card_path_for(model).unlink()
    with pytest.raises(ModelLoadError, match="card missing"):
        SpamClassifier.load(model)


def test_load_rejects_corrupt_card(trained_model: Path, tmp_path: Path) -> None:
    model = copy_model(trained_model, tmp_path)
    card_path_for(model).write_text("{not json")
    with pytest.raises(ModelLoadError, match="unreadable"):
        SpamClassifier.load(model)


class TestShippedModel:
    """Guards the model committed to the repository, which the app serves."""

    @pytest.mark.parametrize("name", ["Prize scam", "Fake bank alert"])
    def test_flags_spam_examples(self, shipped_classifier: SpamClassifier, name: str) -> None:
        assert shipped_classifier.predict(EXAMPLES[name]).is_spam

    @pytest.mark.parametrize("name", ["Friend", "Work update"])
    def test_passes_normal_examples(self, shipped_classifier: SpamClassifier, name: str) -> None:
        assert not shipped_classifier.predict(EXAMPLES[name]).is_spam

    def test_card_reports_held_out_metrics(self, shipped_classifier: SpamClassifier) -> None:
        card = shipped_classifier.card
        assert card is not None
        assert card.test_metrics.f1 > 0.9
        assert card.candidates[0].name == card.model_name
