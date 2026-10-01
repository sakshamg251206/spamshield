"""Model selection, evaluation and artifact export.

Each candidate is a full ``TfidfVectorizer -> classifier`` pipeline, tuned with
cross-validated grid search on the training split only. The candidate with the
best cross-validated spam F1 wins; the held-out test split is touched exactly
once, to report how the chosen model performs on unseen messages.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from spamshield.data import LABEL_COLUMN, TEXT_COLUMN, file_sha256, load_dataset
from spamshield.model_card import CandidateResult, ModelCard, TestMetrics, card_path_for
from spamshield.preprocessing import normalize_text

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


@dataclass(frozen=True)
class Candidate:
    name: str
    display_name: str
    estimator: BaseEstimator
    param_grid: dict[str, list[Any]]


def candidates() -> list[Candidate]:
    """The model families compared during training.

    These are the same families explored in the original notebook (plus Naive
    Bayes, the classic spam-filter baseline). Grids are kept small on purpose:
    the dataset has ~5k messages and wider grids bought no measurable gain.
    Every estimator exposes ``predict_proba`` so the app can show a calibrated
    spam probability rather than a bare label.
    """
    return [
        Candidate(
            "logistic_regression",
            "Logistic Regression",
            LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE),
            {"clf__C": [1, 10, 100], "clf__class_weight": [None, "balanced"]},
        ),
        Candidate(
            "linear_svm",
            "Linear SVM (calibrated)",
            CalibratedClassifierCV(LinearSVC(random_state=RANDOM_STATE), cv=3),
            {"clf__estimator__C": [0.1, 1, 10], "clf__estimator__class_weight": [None, "balanced"]},
        ),
        Candidate(
            "naive_bayes",
            "Complement Naive Bayes",
            ComplementNB(),
            {"clf__alpha": [0.01, 0.05, 0.1, 0.5]},
        ),
        Candidate(
            "random_forest",
            "Random Forest",
            RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
            {"clf__max_depth": [None, 60], "clf__class_weight": [None, "balanced_subsample"]},
        ),
        Candidate(
            "decision_tree",
            "Decision Tree",
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"clf__max_depth": [None, 40], "clf__min_samples_leaf": [1, 3]},
        ),
        Candidate(
            "knn",
            "k-Nearest Neighbours",
            KNeighborsClassifier(metric="cosine"),
            {"clf__n_neighbors": [3, 7], "clf__weights": ["uniform", "distance"]},
        ),
    ]


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Wrap a classifier with the shared text vectoriser."""
    vectorizer = TfidfVectorizer(
        preprocessor=normalize_text,
        # No stop-word list: scikit-learn's English list removes words such as
        # "call", "now" and "get", which are some of the strongest spam signals.
        # Unigrams: bigrams scored within noise in cross-validation (+0.001 F1)
        # and single-word features keep the per-word explanations exact.
        sublinear_tf=True,
        strip_accents="unicode",
    )
    return Pipeline([("tfidf", vectorizer), ("clf", estimator)])


def evaluate(pipeline: Pipeline, texts: pd.Series, labels: np.ndarray) -> TestMetrics:
    spam_probability = pipeline.predict_proba(texts)[:, 1]
    predicted = (spam_probability >= 0.5).astype(int)
    (tn, fp), (fn, tp) = confusion_matrix(labels, predicted, labels=[0, 1]).tolist()
    return TestMetrics(
        accuracy=float(accuracy_score(labels, predicted)),
        precision=float(precision_score(labels, predicted, zero_division=0)),
        recall=float(recall_score(labels, predicted, zero_division=0)),
        f1=float(f1_score(labels, predicted, zero_division=0)),
        roc_auc=float(roc_auc_score(labels, spam_probability)),
        average_precision=float(average_precision_score(labels, spam_probability)),
        confusion_matrix=[[tn, fp], [fn, tp]],
        false_positive_rate=fp / (fp + tn) if (fp + tn) else 0.0,
    )


def train(
    dataset_path: Path,
    model_path: Path,
    *,
    cv_folds: int = 5,
    test_size: float = 0.2,
    only: list[str] | None = None,
    n_jobs: int = -1,
) -> ModelCard:
    """Select, evaluate and save the best model. Returns its model card."""
    frame = load_dataset(dataset_path)
    texts, labels = frame[TEXT_COLUMN], frame[LABEL_COLUMN].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, stratify=labels, random_state=RANDOM_STATE
    )
    logger.info("Split: %d train / %d test messages", len(x_train), len(x_test))

    pool = candidates()
    if only:
        unknown = set(only) - {c.name for c in pool}
        if unknown:
            raise ValueError(f"Unknown model(s): {', '.join(sorted(unknown))}")
        pool = [c for c in pool if c.name in only]

    folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    results: list[CandidateResult] = []
    searches: dict[str, GridSearchCV] = {}
    for candidate in pool:
        logger.info("Tuning %s over %s", candidate.display_name, candidate.param_grid)
        search = GridSearchCV(
            build_pipeline(candidate.estimator),
            candidate.param_grid,
            scoring={"f1": "f1", "precision": "precision", "recall": "recall"},
            refit="f1",
            cv=folds,
            n_jobs=n_jobs,
        )
        started = time.perf_counter()
        search.fit(x_train, y_train)
        elapsed = time.perf_counter() - started
        best = search.best_index_
        cv = search.cv_results_
        result = CandidateResult(
            name=candidate.name,
            display_name=candidate.display_name,
            cv_f1=float(cv["mean_test_f1"][best]),
            cv_f1_std=float(cv["std_test_f1"][best]),
            cv_precision=float(cv["mean_test_precision"][best]),
            cv_recall=float(cv["mean_test_recall"][best]),
            best_params={k.rsplit("__", 1)[-1]: v for k, v in search.best_params_.items()},
            fit_seconds=round(elapsed, 2),
        )
        logger.info(
            "%s: CV F1 %.4f ± %.4f (%.1fs)",
            candidate.display_name,
            result.cv_f1,
            result.cv_f1_std,
            elapsed,
        )
        results.append(result)
        searches[candidate.name] = search

    results.sort(key=lambda r: r.cv_f1, reverse=True)
    winner = results[0]
    pipeline: Pipeline = searches[winner.name].best_estimator_
    metrics = evaluate(pipeline, x_test, y_test)
    logger.info(
        "Selected %s. Test F1 %.4f, precision %.4f, recall %.4f",
        winner.display_name,
        metrics.f1,
        metrics.precision,
        metrics.recall,
    )

    card = ModelCard(
        model_name=winner.name,
        model_display_name=winner.display_name,
        best_params=winner.best_params,
        test_metrics=metrics,
        candidates=results,
        threshold=0.5,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        dataset_path=_relative(dataset_path),
        dataset_sha256=file_sha256(dataset_path),
        n_train=len(x_train),
        n_test=len(x_test),
        n_spam=int(labels.sum()),
        n_ham=int(len(labels) - labels.sum()),
        cv_folds=cv_folds,
        vocabulary_size=len(pipeline.named_steps["tfidf"].vocabulary_),
        notes=[
            "Trained on the SMS Spam Collection; email wording differs, so treat "
            "scores on long emails as indicative rather than definitive.",
        ],
    )
    save_model(pipeline, card, model_path)
    return card


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def save_model(pipeline: Pipeline, card: ModelCard, model_path: Path) -> None:
    """Write the pipeline and its card atomically, recording the model checksum."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=model_path.parent, suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        joblib.dump(pipeline, tmp, compress=3)
        card.model_sha256 = file_sha256(tmp)
        tmp.chmod(0o644)  # mkstemp creates owner-only files
        tmp.replace(model_path)
    finally:
        tmp.unlink(missing_ok=True)
    card_path_for(model_path).write_text(card.to_json() + "\n", encoding="utf-8")
    logger.info("Saved model to %s", model_path)
