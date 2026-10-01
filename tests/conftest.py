from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from spamshield.classifier import SpamClassifier
from spamshield.config import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH
from spamshield.training import train

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_mbox() -> Path:
    return FIXTURES / "sample.mbox"


@pytest.fixture(scope="session")
def small_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A stratified 800-message slice of the real dataset, small enough to train in seconds."""
    frame = pd.read_csv(DEFAULT_DATASET_PATH, encoding="utf-8-sig")
    sample = frame.groupby("Category", group_keys=False).sample(frac=0.15, random_state=0)
    path = tmp_path_factory.mktemp("data") / "small.csv"
    sample.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory: pytest.TempPathFactory, small_dataset: Path) -> Path:
    path = tmp_path_factory.mktemp("models") / "test.joblib"
    train(small_dataset, path, cv_folds=3, only=["logistic_regression", "naive_bayes"], n_jobs=1)
    return path


@pytest.fixture(scope="session")
def classifier(trained_model: Path) -> SpamClassifier:
    return SpamClassifier.load(trained_model)


@pytest.fixture(scope="session")
def shipped_classifier() -> SpamClassifier:
    return SpamClassifier.load(DEFAULT_MODEL_PATH)
