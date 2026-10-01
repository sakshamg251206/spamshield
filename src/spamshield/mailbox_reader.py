"""Reading ``.mbox`` exports (Gmail Takeout, Thunderbird, Apple Mail...)."""

from __future__ import annotations

import logging
import mailbox
import re
from dataclasses import dataclass
from email.header import decode_header, make_header
from email.message import Message
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")


class MailboxError(ValueError):
    """Raised when a file cannot be read as a mailbox."""


@dataclass(frozen=True)
class EmailRecord:
    date: str
    sender: str
    subject: str
    body: str
    #: Folder or label the mail client had filed this message under, if known.
    folder: str

    @property
    def text(self) -> str:
        """Subject and body together: what the classifier reads."""
        return f"{self.subject}\n{self.body}".strip()


@dataclass(frozen=True)
class MailboxReadResult:
    records: list[EmailRecord]
    skipped: int
    truncated: bool


def decode_header_value(value: object) -> str:
    """Decode RFC 2047 headers such as ``=?UTF-8?B?...?=`` into plain text."""
    if value is None:
        return ""
    try:
        return str(make_header(decode_header(str(value)))).strip()
    except (UnicodeDecodeError, LookupError, ValueError):
        return str(value).strip()


def _decode_part(part: Message) -> str:
    payload = part.get_payload(decode=True)
    if not isinstance(payload, bytes):
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:  # unknown charset name in the header
        return payload.decode("utf-8", errors="replace")


def extract_body(message: Message) -> str:
    """Return readable body text, preferring ``text/plain`` over HTML.

    Attachments are ignored. HTML is converted to text so markup never reaches
    the model.
    """
    plain: list[str] = []
    html: list[str] = []
    for part in message.walk():
        if part.is_multipart() or part.get_content_disposition() == "attachment":
            continue
        content_type = part.get_content_type()
        if content_type == "text/plain":
            plain.append(_decode_part(part))
        elif content_type == "text/html":
            html.append(_decode_part(part))

    if any(p.strip() for p in plain):
        text = " ".join(plain)
    else:
        text = " ".join(BeautifulSoup(h, "html.parser").get_text(" ") for h in html)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _folder(message: Message) -> str:
    labels = decode_header_value(message.get("X-Gmail-Labels")).lower()
    if not labels:
        return ""
    for needle, name in (
        ("spam", "Spam"),
        ("category promotions", "Promotions"),
        ("category_promotions", "Promotions"),
        ("category social", "Social"),
        ("category_social", "Social"),
        ("category updates", "Updates"),
        ("category_updates", "Updates"),
        ("sent", "Sent"),
    ):
        if needle in labels:
            return name
    return "Inbox"


def _date(message: Message) -> str:
    raw = message.get("Date")
    if not raw:
        return ""
    try:
        return parsedate_to_datetime(str(raw)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, IndexError):
        return str(raw)


def to_record(message: Message) -> EmailRecord:
    name, address = parseaddr(decode_header_value(message.get("From")))
    return EmailRecord(
        date=_date(message),
        sender=address or name,
        subject=decode_header_value(message.get("Subject")),
        body=extract_body(message),
        folder=_folder(message),
    )


def read_mbox(path: Path, *, max_messages: int = 5_000) -> MailboxReadResult:
    """Parse up to ``max_messages`` emails from an mbox file.

    A single malformed message is skipped and counted rather than aborting the
    whole scan.
    """
    if not path.is_file():
        raise MailboxError(f"File not found: {path}")
    with path.open("rb") as handle:
        head = handle.read(5)
    if head and head != b"From ":
        raise MailboxError(
            "This does not look like an mbox file (it should start with 'From '). "
            "Export your mailbox in mbox format, e.g. via Google Takeout."
        )

    box = mailbox.mbox(path, create=False)
    records: list[EmailRecord] = []
    skipped = 0
    truncated = False
    try:
        for index, message in enumerate(box):
            if index >= max_messages:
                truncated = True
                break
            try:
                records.append(to_record(message))
            except Exception:
                skipped += 1
                logger.warning("Skipped unreadable message #%d", index, exc_info=True)
    finally:
        box.close()

    logger.info("Read %d messages from %s (%d skipped)", len(records), path.name, skipped)
    return MailboxReadResult(records, skipped, truncated)
