import csv
import io
from pathlib import Path

import pandas as pd
import pytest

from spamshield.classifier import SpamClassifier
from spamshield.scanner import RESULT_COLUMNS, sanitize_cell, scan_mbox, to_safe_csv


def test_scan_ranks_most_suspicious_first(
    shipped_classifier: SpamClassifier, sample_mbox: Path
) -> None:
    result = scan_mbox(shipped_classifier, sample_mbox)
    assert list(result.frame.columns) == RESULT_COLUMNS
    assert result.total == 5
    probabilities = result.frame["Spam probability"].tolist()
    assert probabilities == sorted(probabilities, reverse=True)
    flagged = set(result.frame.loc[result.frame["Verdict"] == "Spam", "From"])
    assert {"alerts@secure-bank-verify.net", "winner@prizes.example"} <= flagged
    assert "priya@example.com" not in flagged
    assert result.spam_count == len(flagged)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ('=HYPERLINK("x")', '\'=HYPERLINK("x")'),
        ("+1 call", "'+1 call"),
        ("-2", "'-2"),
        ("@SUM(A1)", "'@SUM(A1)"),
        ("normal", "normal"),
        ("zero​width\x00", "zerowidth"),
        (0.5, 0.5),
    ],
)
def test_sanitize_cell(value: object, expected: object) -> None:
    assert sanitize_cell(value) == expected


def test_csv_export_neutralises_formulas() -> None:
    frame = pd.DataFrame({"Subject": ["=cmd|' /C calc'!A0", "hi"], "Spam probability": [0.9, 0.1]})
    content = to_safe_csv(frame).decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(content)))
    assert rows[1][0].startswith("'=")
    assert rows[2] == ["hi", "0.1"]
