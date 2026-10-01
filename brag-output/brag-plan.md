# /brag-slim plan: SpamShield

**Angle:** the model's own verdicts, shown working. A scam text arrives, SpamShield says how likely it is spam *and which words gave it away*, then the same for clean messages, a whole mailbox, and the honest held-out numbers.
**Tone:** `default` (punchy, clean). 20 s, 1920×1080, 30 fps. Light theme, the app's own palette (primary #2563eb, red/green tints from its CSS) and its font (Source Sans).
**Hook:** "Prize or scam?" over a typing prize-scam text (the app's built-in example).
**Punchline:** 96.1% of spam caught, 0.77% of normal messages wrongly flagged, on 1,032 messages the model never saw; "Nothing you paste is stored."

All verdicts, probabilities, signal words and the mailbox table are real output from `models/spamshield.joblib` (built-in `EXAMPLES`, `tests/fixtures/sample.mbox`). Test-set numbers are from `models/spamshield.card.json`. Nothing is invented.

## Storyboard (sums to 20 s)

| Time | Scene | What happens |
|---|---|---|
| 0.0–2.0 | Hook | "Prize or scam?" + prize-scam SMS types in |
| 2.0–7.5 | Reveal + why | Message slides left; verdict card "Very likely spam", probability bar fills to >99%; chips (a phone number, a link, a money amount, claim) pop in and light up the same words in the message; tagline "See the words that gave it away." |
| 7.5–11.0 | Contrast | "Spam flagged. Friends left alone." Fake bank alert 87%, Friend <1%, Work update 1% |
| 11.0–14.5 | Mailbox | "Or scan a whole mailbox." sample.mbox drops in; metrics; table sorted by risk |
| 14.5–17.2 | Proof | 96.1% caught / 0.77% wrongly flagged / "On 1,032 messages it never saw." |
| 17.2–20.0 | Outro | 🛡️ SpamShield, "Nothing you paste is stored.", `uv run streamlit run app.py`, repo URL |

Transitions dip through the background (old content out, then new in), so there is no muddy double exposure.

## Sound
120 bpm, Am–F–C–G–Am–F–C–G–F–C, bar-aligned to the scenes. Pad, bass, pluck arp, soft kick/hat. SFX (typing ticks, chip pops, row ticks, counter run, whooshes) are drawn from the current chord's notes and share one reverb with the music. Pad/arp are ducked by the kick; no spiky highs. Measured: −14.8 LUFS integrated, −1.5 dBFS peak, 0 clipped samples.
