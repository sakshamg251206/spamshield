"""The model card: metadata saved next to every trained model.

It records how the model was chosen, how well it scored on held-out data and
the checksum the loader verifies before unpickling the model file.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

CARD_SUFFIX = ".card.json"


@dataclass
class CandidateResult:
    name: str
    display_name: str
    cv_f1: float
    cv_f1_std: float
    cv_precision: float
    cv_recall: float
    best_params: dict[str, Any]
    fit_seconds: float


@dataclass
class TestMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    average_precision: float
    #: ``[[true ham, false spam], [missed spam, caught spam]]``
    confusion_matrix: list[list[int]]
    false_positive_rate: float


@dataclass
class ModelCard:
    """Everything needed to understand, reproduce and trust a saved model."""

    model_name: str
    model_display_name: str
    best_params: dict[str, Any]
    test_metrics: TestMetrics
    candidates: list[CandidateResult]
    threshold: float
    trained_at: str
    dataset_path: str
    dataset_sha256: str
    n_train: int
    n_test: int
    n_spam: int
    n_ham: int
    cv_folds: int
    model_sha256: str = ""
    sklearn_version: str = field(default_factory=lambda: sklearn.__version__)
    python_version: str = field(default_factory=platform.python_version)
    vocabulary_size: int = 0
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=_json_default)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCard:
        data = dict(data)
        data["test_metrics"] = TestMetrics(**data["test_metrics"])
        data["candidates"] = [CandidateResult(**c) for c in data["candidates"]]
        return cls(**data)


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def card_path_for(model_path: Path) -> Path:
    return model_path.with_name(model_path.stem + CARD_SUFFIX)
