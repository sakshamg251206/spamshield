"""Classifying a whole mailbox."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from spamshield.classifier import SpamClassifier
from spamshield.mailbox_reader import read_mbox

RESULT_COLUMNS = ["Verdict", "Spam probability", "Date", "From", "Subject", "Folder", "Preview"]
_PREVIEW_CHARS = 160


@dataclass(frozen=True)
class ScanResult:
    frame: pd.DataFrame
    skipped: int
    truncated: bool

    @property
    def total(self) -> int:
        return len(self.frame)

    @property
    def spam_count(self) -> int:
        return int((self.frame["Verdict"] == "Spam").sum())


def scan_mbox(classifier: SpamClassifier, path: Path, *, max_messages: int = 5_000) -> ScanResult:
    """Read an mbox file and classify every message in one batch."""
    read = read_mbox(path, max_messages=max_messages)
    predictions = classifier.predict_many([record.text for record in read.records])
    rows = [
        {
            "Verdict": "Spam" if prediction.is_spam else "Not spam",
            "Spam probability": round(prediction.spam_probability, 4),
            "Date": record.date,
            "From": record.sender,
            "Subject": record.subject,
            "Folder": record.folder,
            "Preview": record.body[:_PREVIEW_CHARS],
        }
        for record, prediction in zip(read.records, predictions, strict=True)
    ]
    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    frame = frame.sort_values("Spam probability", ascending=False, kind="stable")
    return ScanResult(frame.reset_index(drop=True), read.skipped, read.truncated)
