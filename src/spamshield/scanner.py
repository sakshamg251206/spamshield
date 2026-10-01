"""Classifying a whole mailbox and exporting the results safely."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from spamshield.classifier import SpamClassifier
from spamshield.mailbox_reader import read_mbox

RESULT_COLUMNS = ["Verdict", "Spam probability", "Date", "From", "Subject", "Folder", "Preview"]
_PREVIEW_CHARS = 160
# Cells starting with these characters are executed as formulas by Excel,
# LibreOffice and Google Sheets (CSV injection).
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f​-‏﻿]")


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


def sanitize_cell(value: object) -> object:
    """Make a value safe to open in a spreadsheet."""
    if not isinstance(value, str):
        return value
    value = _CONTROL_CHARS_RE.sub("", value)[:32_767]  # Excel's per-cell limit
    return "'" + value if value.startswith(_FORMULA_PREFIXES) else value


def to_safe_csv(frame: pd.DataFrame) -> bytes:
    """Serialise ``frame`` as UTF-8 CSV with spreadsheet formula injection neutralised."""
    safe = frame.apply(lambda column: column.map(sanitize_cell))
    # utf-8-sig so Excel detects the encoding and shows non-ASCII text correctly.
    content: str = safe.to_csv(index=False)
    return content.encode("utf-8-sig")
