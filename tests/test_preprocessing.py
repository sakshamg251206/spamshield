import pytest

from spamshield.preprocessing import normalize_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Visit https://example.com/win now", "visit urltoken now"),
        ("go to www.free-stuff.biz today", "go to urltoken today"),
        ("mail me at jane.doe@gmail.com", "mail me at emailtoken"),
        ("Win £1,000 cash", "win moneytoken cash"),
        ("worth 500 pounds", "worth moneytoken"),
        ("Call 0906 170 1461 now", "call phonetoken now"),
        ("Call +44 7911 123456", "call phonetoken"),
        ("see you at 7", "see you at numbertoken"),
        ("Fish &amp; Chips", "fish & chips"),
        ("  lots\n\tof   space ", "lots of space"),
    ],
)
def test_normalize_text(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


def test_email_is_not_mistaken_for_url() -> None:
    assert "urltoken" not in normalize_text("write to support@company.com")


@pytest.mark.parametrize(
    "raw",
    ["WIN a £900 prize! Call 09061701461 or visit www.win.com, mail a@b.co", "plain words"],
)
def test_normalize_text_is_idempotent(raw: str) -> None:
    once = normalize_text(raw)
    assert normalize_text(once) == once
