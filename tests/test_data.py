from pathlib import Path

import pytest

from spamshield.data import HAM, SPAM, DatasetError, load_dataset


def write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "data.csv"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_encodes_and_deduplicates(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "﻿Category,Message\nham,hello there\nspam,WIN NOW\nham,hello there\nham,  \n",
    )
    frame = load_dataset(path)
    assert list(frame.columns) == ["text", "label"]
    assert frame["text"].tolist() == ["hello there", "WIN NOW"]
    assert frame["label"].tolist() == [HAM, SPAM]


def test_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "nope.csv")


def test_rejects_missing_columns(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="missing column"):
        load_dataset(write(tmp_path, "label,text\nham,hi\n"))


def test_rejects_unknown_labels(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="Unexpected labels"):
        load_dataset(write(tmp_path, "Category,Message\nham,hi\nphishing,bad\n"))


def test_requires_both_classes(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="both spam and ham"):
        load_dataset(write(tmp_path, "Category,Message\nham,hi\nham,hello\n"))
