"""Plain-language wording for predictions, shared by the web app and tests."""

from __future__ import annotations

from dataclasses import dataclass

from spamshield.classifier import Prediction

#: Ready-made messages so first-time visitors can try the app in one click.
#: They are written for this demo and are not taken from the training data.
EXAMPLES: dict[str, str] = {
    "Prize scam": (
        "CONGRATULATIONS! You've been selected to receive a £1,000 gift card. "
        "To claim your reward call 0906 170 1461 or visit www.claim-prize-now.com "
        "before midnight. T&Cs apply."
    ),
    "Fake bank alert": (
        "URGENT: Your account has been suspended due to unusual activity. "
        "Verify your details within 24 hours at http://secure-verify-account.net "
        "or your card will be blocked."
    ),
    "Friend": (
        "Hey! Are we still on for dinner tomorrow? I can pick you up around 7, "
        "just text me when you're leaving work."
    ),
    "Work update": (
        "Hi team, the quarterly report draft is in the shared folder. "
        "Please add your comments by Thursday so we can finalise it on Friday. Thanks!"
    ),
}


@dataclass(frozen=True)
class Verdict:
    title: str
    summary: str
    #: "spam", "ham" or "uncertain"; drives the colour of the result card.
    tone: str


def describe(prediction: Prediction, threshold: float = 0.5) -> Verdict:
    """Turn a prediction into a headline and one sentence a non-expert understands."""
    probability = prediction.spam_probability
    percent = f"{probability:.0%}"
    if abs(probability - threshold) < 0.15:
        return Verdict(
            "Could go either way",
            f"The model gives this a {percent} chance of being spam, close to its "
            f"{threshold:.0%} cut-off. Use your own judgement.",
            "uncertain",
        )
    if prediction.is_spam:
        strength = "Very likely" if probability >= 0.9 else "Likely"
        return Verdict(
            f"{strength} spam",
            f"The model gives this a {percent} chance of being spam. "
            "Don't click links or reply with personal details.",
            "spam",
        )
    strength = "Very likely" if probability <= 0.1 else "Probably"
    return Verdict(
        f"{strength} not spam",
        f"The model gives this a {percent} chance of being spam. It looks like a normal message.",
        "ham",
    )
