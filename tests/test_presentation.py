import pytest

from spamshield.classifier import Prediction
from spamshield.presentation import describe, format_probability


@pytest.mark.parametrize(
    ("probability", "text"),
    [(0.9999, ">99%"), (0.001, "<1%"), (0.42, "42%"), (0.995, ">99%")],
)
def test_format_probability(probability: float, text: str) -> None:
    assert format_probability(probability) == text


def test_describe_spam_ham_and_uncertain() -> None:
    assert describe(Prediction("spam", 0.97)).tone == "spam"
    assert describe(Prediction("spam", 0.97)).title == "Very likely spam"
    assert describe(Prediction("spam", 0.8)).title == "Likely spam"
    assert describe(Prediction("ham", 0.02)).title == "Very likely not spam"
    assert describe(Prediction("ham", 0.3)).title == "Probably not spam"
    assert describe(Prediction("spam", 0.55)).tone == "uncertain"
    assert describe(Prediction("ham", 0.45)).tone == "uncertain"


def test_describe_never_claims_certainty() -> None:
    assert "100%" not in describe(Prediction("spam", 0.99999)).summary
