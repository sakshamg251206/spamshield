"""Text normalisation shared by training and inference.

Spam is full of details that are unique per message but mean the same thing:
a specific URL, phone number or prize amount. Replacing them with placeholder
words lets the model learn "contains a phone number" instead of memorising
individual numbers it will never see again.

The same function runs inside the saved scikit-learn pipeline, so training and
prediction can never drift apart.
"""

from __future__ import annotations

import html
import re

URL_PLACEHOLDER = "urltoken"
EMAIL_PLACEHOLDER = "emailtoken"
PHONE_PLACEHOLDER = "phonetoken"
MONEY_PLACEHOLDER = "moneytoken"
NUMBER_PLACEHOLDER = "numbertoken"

#: Human-readable names for placeholders, used when explaining a prediction.
PLACEHOLDER_LABELS = {
    URL_PLACEHOLDER: "a link",
    EMAIL_PLACEHOLDER: "an email address",
    PHONE_PLACEHOLDER: "a phone number",
    MONEY_PLACEHOLDER: "a money amount",
    NUMBER_PLACEHOLDER: "a number",
}

_URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[a-z0-9-]+\.(?:com|net|org|co\.uk|biz|info)\b\S*")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_MONEY_RE = re.compile(r"[£$€]\s?\d[\d,.]*|\b\d[\d,.]*\s?(?:pounds?|dollars?|usd|gbp|eur)\b")
# Seven or more digits, optionally separated by spaces, dots or dashes.
_PHONE_RE = re.compile(r"\+?\d(?:[\s.-]?\d){6,}")
_NUMBER_RE = re.compile(r"\d+")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lower-case ``text`` and replace volatile details with placeholder words.

    The function is idempotent: normalising already-normalised text returns it
    unchanged, which the prediction explainer relies on.
    """
    text = html.unescape(text).lower()
    # Emails first: the bare-domain URL pattern would otherwise eat "gmail.com".
    text = _EMAIL_RE.sub(f" {EMAIL_PLACEHOLDER} ", text)
    text = _URL_RE.sub(f" {URL_PLACEHOLDER} ", text)
    text = _MONEY_RE.sub(f" {MONEY_PLACEHOLDER} ", text)
    text = _PHONE_RE.sub(f" {PHONE_PLACEHOLDER} ", text)
    text = _NUMBER_RE.sub(f" {NUMBER_PLACEHOLDER} ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()
