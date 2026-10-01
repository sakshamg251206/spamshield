"""Loading a trained model and turning text into explained predictions."""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline

from spamshield.data import file_sha256
from spamshield.model_card import ModelCard, card_path_for

logger = logging.getLogger(__name__)

Label = Literal["spam", "ham"]


class ModelLoadError(RuntimeError):
    """Raised when the model artifact is missing, tampered with or unreadable."""


@dataclass(frozen=True)
class Prediction:
    label: Label
    #: Model's probability that the message is spam, from 0 to 1.
    spam_probability: float

    @property
    def is_spam(self) -> bool:
        return self.label == "spam"

    @property
    def confidence(self) -> float:
        """Probability of the predicted label, from 0.5 to 1."""
        return self.spam_probability if self.is_spam else 1 - self.spam_probability


class SpamClassifier:
    """A thin, typed wrapper around the saved scikit-learn pipeline."""

    def __init__(
        self,
        pipeline: Pipeline,
        card: ModelCard | None = None,
        *,
        threshold: float = 0.5,
        max_chars: int = 20_000,
    ) -> None:
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        self.pipeline = pipeline
        self.card = card
        self.threshold = threshold
        self.max_chars = max_chars

    @classmethod
    def load(
        cls,
        model_path: Path,
        *,
        threshold: float = 0.5,
        max_chars: int = 20_000,
    ) -> SpamClassifier:
        """Load a model saved by :func:`spamshield.training.train`.

        joblib files can execute code when loaded, so the file's SHA-256 is
        checked against the model card first. A model whose checksum does not
        match the card it was saved with is refused.
        """
        if not model_path.is_file():
            raise ModelLoadError(
                f"No model found at {model_path}. Train one with `spamshield train`."
            )
        card_path = card_path_for(model_path)
        if not card_path.is_file():
            raise ModelLoadError(f"Model card missing: {card_path}")
        try:
            card = ModelCard.from_dict(json.loads(card_path.read_text(encoding="utf-8")))
        except (ValueError, TypeError, KeyError) as exc:
            raise ModelLoadError(f"Model card {card_path} is unreadable: {exc}") from exc

        actual = file_sha256(model_path)
        if actual != card.model_sha256:
            raise ModelLoadError(
                f"Checksum mismatch for {model_path.name}: the file does not match its model "
                "card. Retrain the model or restore the original file."
            )
        if card.sklearn_version != sklearn.__version__:
            logger.warning(
                "Model was trained with scikit-learn %s but %s is installed; "
                "predictions may differ slightly. Retrain to silence this warning.",
                card.sklearn_version,
                sklearn.__version__,
            )

        with warnings.catch_warnings():
            # The version mismatch is already reported above, once and clearly.
            warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")
            try:
                pipeline = joblib.load(model_path)
            except Exception as exc:
                raise ModelLoadError(f"Could not load {model_path.name}: {exc}") from exc
        if not hasattr(pipeline, "predict_proba"):
            raise ModelLoadError("Loaded object is not a probabilistic classifier")
        return cls(pipeline, card, threshold=threshold, max_chars=max_chars)

    def _spam_probabilities(self, texts: Sequence[str]) -> np.ndarray:
        clipped = [text[: self.max_chars] for text in texts]
        probabilities: np.ndarray = self.pipeline.predict_proba(clipped)[:, 1]
        return probabilities

    def _label(self, spam_probability: float) -> Label:
        return "spam" if spam_probability >= self.threshold else "ham"

    def predict(self, text: str) -> Prediction:
        """Classify one message. Raises ``ValueError`` for empty input."""
        if not text or not text.strip():
            raise ValueError("Cannot classify an empty message")
        probability = float(self._spam_probabilities([text])[0])
        return Prediction(self._label(probability), probability)

    def predict_many(self, texts: Sequence[str]) -> list[Prediction]:
        """Classify many messages in one vectorised call. Empty texts are ham."""
        if not texts:
            return []
        probabilities = self._spam_probabilities([t or "" for t in texts])
        return [
            Prediction(self._label(float(p)), float(p))
            if text and text.strip()
            else Prediction("ham", 0.0)
            for text, p in zip(texts, probabilities, strict=True)
        ]
