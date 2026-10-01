"""Loading and validating the labelled training dataset."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
#: Integer encoding used everywhere: spam is the positive class.
SPAM, HAM = 1, 0

_SOURCE_COLUMNS = {"Message": TEXT_COLUMN, "Category": LABEL_COLUMN}
_LABEL_MAP = {"spam": SPAM, "ham": HAM}


class DatasetError(ValueError):
    """Raised when the dataset file is missing or malformed."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path) -> pd.DataFrame:
    """Read a ``Category,Message`` CSV and return clean ``text``/``label`` columns.

    Rows with missing text or unknown labels are rejected, and exact duplicate
    messages are dropped. Duplicates matter: the SMS corpus contains hundreds of
    repeated messages, and leaving them in lets copies of a test message sit in
    the training set, which inflates every reported metric.
    """
    if not path.is_file():
        raise DatasetError(f"Dataset not found: {path}")

    # utf-8-sig strips the byte-order mark some spreadsheet tools write.
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = set(_SOURCE_COLUMNS) - set(frame.columns)
    if missing:
        raise DatasetError(f"Dataset is missing column(s): {', '.join(sorted(missing))}")

    frame = frame[list(_SOURCE_COLUMNS)].rename(columns=_SOURCE_COLUMNS)
    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].astype("string").str.strip()
    frame = frame[frame[TEXT_COLUMN].fillna("").str.len() > 0]

    labels = frame[LABEL_COLUMN].astype("string").str.strip().str.lower()
    unknown = sorted(set(labels.dropna()) - set(_LABEL_MAP))
    if unknown or labels.isna().any():
        raise DatasetError(
            f"Unexpected labels {unknown or ['<missing>']}; expected 'spam' or 'ham'"
        )
    frame[LABEL_COLUMN] = labels.map(_LABEL_MAP).astype(int)

    before = len(frame)
    frame = frame.drop_duplicates(subset=TEXT_COLUMN).reset_index(drop=True)
    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].astype(str)
    logger.info(
        "Loaded %d messages (%d duplicates removed): %d spam, %d ham",
        len(frame),
        before - len(frame),
        int((frame[LABEL_COLUMN] == SPAM).sum()),
        int((frame[LABEL_COLUMN] == HAM).sum()),
    )

    if frame[LABEL_COLUMN].nunique() < 2:
        raise DatasetError("Dataset must contain both spam and ham examples")
    return frame
