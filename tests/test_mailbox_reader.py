from email.message import EmailMessage
from pathlib import Path

import pytest

from spamshield.mailbox_reader import (
    MailboxError,
    decode_header_value,
    extract_body,
    read_mbox,
    to_record,
)


def test_reads_fixture(sample_mbox: Path) -> None:
    result = read_mbox(sample_mbox)
    assert len(result.records) == 5
    assert result.skipped == 0
    assert not result.truncated
    first = result.records[0]
    assert first.sender == "alerts@secure-bank-verify.net"
    assert first.folder == "Spam"
    assert first.date == "2025-03-01 09:00"


def test_decodes_encoded_subject_and_strips_html(sample_mbox: Path) -> None:
    record = read_mbox(sample_mbox).records[2]
    assert record.subject == "You WON a €500 gift card!"
    assert "<b>" not in record.body
    assert "£500" in record.body
    assert record.folder == "Promotions"


def test_prefers_plain_text_and_ignores_attachments(sample_mbox: Path) -> None:
    record = read_mbox(sample_mbox).records[3]
    assert record.body.startswith("Hi team")
    assert "<p>" not in record.body
    assert "PDF" not in record.body
    assert record.body.count("Hi team") == 1


def test_text_combines_subject_and_body(sample_mbox: Path) -> None:
    record = read_mbox(sample_mbox).records[1]
    assert record.text.startswith("Dinner on Friday?\nHi!")


def test_respects_message_limit(sample_mbox: Path) -> None:
    result = read_mbox(sample_mbox, max_messages=2)
    assert len(result.records) == 2
    assert result.truncated


def test_rejects_non_mbox_files(tmp_path: Path) -> None:
    path = tmp_path / "notes.mbox"
    path.write_text("just some text")
    with pytest.raises(MailboxError, match="does not look like an mbox"):
        read_mbox(path)
    with pytest.raises(MailboxError, match="not found"):
        read_mbox(tmp_path / "missing.mbox")


def test_empty_file_has_no_records(tmp_path: Path) -> None:
    path = tmp_path / "empty.mbox"
    path.write_bytes(b"")
    assert read_mbox(path).records == []


def test_unknown_charset_falls_back_to_utf8() -> None:
    message = EmailMessage()
    message["Content-Type"] = 'text/plain; charset="x-made-up"'
    message.set_payload("caf\xe9".encode("latin-1"))
    assert extract_body(message).startswith("caf")


def test_missing_headers_produce_empty_fields() -> None:
    message = EmailMessage()
    message.set_content("body only")
    record = to_record(message)
    assert (record.sender, record.subject, record.date, record.folder) == ("", "", "", "")
    assert record.body == "body only"


def test_decode_header_value_tolerates_garbage() -> None:
    assert decode_header_value(None) == ""
    assert decode_header_value("=?bogus?Q?abc?=") != ""
