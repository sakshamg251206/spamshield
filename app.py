"""SpamShield web app. Run with ``streamlit run app.py``."""

from __future__ import annotations

import html
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from spamshield.classifier import ModelLoadError, SpamClassifier
from spamshield.config import ConfigError, Settings
from spamshield.logging_setup import configure_logging
from spamshield.mailbox_reader import MailboxError
from spamshield.model_card import ModelCard
from spamshield.presentation import EXAMPLES, describe, format_probability
from spamshield.scanner import ScanResult, scan_mbox, to_safe_csv

logger = logging.getLogger("spamshield.app")

st.set_page_config(page_title="SpamShield", page_icon="🛡️", layout="wide")

# Colours are translucent tints so the page reads well in light and dark themes.
st.markdown(
    """
    <style>
      .block-container { max-width: 1080px; padding-top: 2.5rem; }
      .hero h1 { font-size: 2.3rem; margin-bottom: .2rem; }
      .hero p { font-size: 1.08rem; opacity: .8; margin-top: 0; }
      .steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
               gap: .75rem; margin: 1rem 0 1.5rem; }
      .step { border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: .8rem 1rem; }
      .step b { display: block; margin-bottom: .15rem; }
      .step span { font-size: .9rem; opacity: .75; }
      .verdict { border-radius: 14px; padding: 1.1rem 1.3rem; margin: .5rem 0 1rem;
                 border: 1px solid; }
      .verdict h3 { margin: 0 0 .25rem; font-size: 1.35rem; }
      .verdict p { margin: 0; opacity: .85; }
      .tone-spam { background: rgba(220,38,38,.10); border-color: rgba(220,38,38,.45); }
      .tone-ham { background: rgba(22,163,74,.10); border-color: rgba(22,163,74,.45); }
      .tone-uncertain { background: rgba(217,119,6,.10); border-color: rgba(217,119,6,.45); }
      .chip { display: inline-block; padding: .2rem .65rem; margin: 0 .35rem .4rem 0;
              border-radius: 999px; font-size: .9rem; border: 1px solid; }
      .chip-spam { background: rgba(220,38,38,.10); border-color: rgba(220,38,38,.35); }
      .chip-ham { background: rgba(22,163,74,.10); border-color: rgba(22,163,74,.35); }
      .muted { opacity: .7; font-size: .9rem; }
      .cm { display: grid; grid-template-columns: auto 1fr 1fr; gap: 6px; max-width: 420px;
            text-align: center; font-size: .9rem; }
      .cm div { padding: .55rem; border-radius: 8px; }
      .cm .good { background: rgba(22,163,74,.12); }
      .cm .bad { background: rgba(220,38,38,.12); }
      .cm .head { opacity: .7; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------- setup


@st.cache_resource(show_spinner="Loading the spam model…")
def load_classifier(settings: Settings) -> SpamClassifier:
    configure_logging(settings.log_level)
    return SpamClassifier.load(
        settings.model_path, threshold=settings.threshold, max_chars=settings.max_chars
    )


def startup() -> tuple[Settings, SpamClassifier]:
    try:
        settings = Settings.from_env()
        return settings, load_classifier(settings)
    except (ConfigError, ModelLoadError) as exc:
        st.error(f"**SpamShield could not start.** {exc}")
        st.markdown(
            "Train a model with `uv run spamshield train` (about a minute), "
            "check your `.env` against `.env.example`, then reload this page."
        )
        st.stop()
        raise  # unreachable; st.stop() raises, this satisfies the type checker


settings, classifier = startup()
card = classifier.card


def chips(signals: list[tuple[str, float]], tone: str) -> str:
    return "".join(
        f'<span class="chip chip-{tone}">{html.escape(token)}</span>' for token, _ in signals
    )


# -------------------------------------------------------------------- header

trained_on = f"{card.n_train + card.n_test:,}" if card else "thousands of"
st.markdown(
    f"""
    <div class="hero">
      <h1>🛡️ SpamShield</h1>
      <p>Not sure whether a message is a scam? Paste it here or upload a whole mailbox.
      SpamShield tells you how likely it is to be spam and which words gave it away.</p>
    </div>
    <div class="steps">
      <div class="step"><b>1 · Paste or upload</b>
        <span>A single message, or an .mbox export of your inbox.</span></div>
      <div class="step"><b>2 · The model reads it</b>
        <span>A machine-learning model trained on {trained_on} labelled messages.</span></div>
      <div class="step"><b>3 · Get a clear answer</b>
        <span>A spam probability, a verdict, and the words behind it.</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

check_tab, scan_tab, about_tab = st.tabs(
    ["✉️  Check a message", "📥  Scan a mailbox", "📊  Accuracy"]
)

# ------------------------------------------------------------- check a message

with check_tab:
    if "message" not in st.session_state:
        st.session_state.message = ""

    def use_example(name: str) -> None:
        st.session_state.message = EXAMPLES[name]

    st.markdown("**Try an example**, or paste your own message below.")
    example_columns = st.columns(len(EXAMPLES))
    for column, name in zip(example_columns, EXAMPLES, strict=True):
        column.button(name, on_click=use_example, args=(name,), width="stretch")

    text = st.text_area(
        "Message",
        key="message",
        height=180,
        max_chars=settings.max_chars,
        placeholder="Paste the subject and body of an email, or the text of an SMS…",
    )
    submitted = st.button("Check message", type="primary", disabled=not text.strip())

    if submitted and text.strip():
        with st.spinner("Analysing…"):
            prediction = classifier.predict(text)
            signals = classifier.explain(text)
        verdict = describe(prediction, classifier.threshold)
        st.markdown(
            f'<div class="verdict tone-{verdict.tone}"><h3>{verdict.title}</h3>'
            f"<p>{html.escape(verdict.summary)}</p></div>",
            unsafe_allow_html=True,
        )
        st.progress(
            prediction.spam_probability,
            text=f"Spam probability: {format_probability(prediction.spam_probability)}",
        )

        spammy = [(s.display, s.impact) for s in signals if s.impact > 0]
        normal = [(s.display, s.impact) for s in signals if s.impact < 0]
        left, right = st.columns(2)
        with left:
            st.markdown("**Looks like spam**")
            st.markdown(
                chips(spammy, "spam") or '<span class="muted">Nothing stood out.</span>',
                unsafe_allow_html=True,
            )
        with right:
            st.markdown("**Looks like a normal message**")
            st.markdown(
                chips(normal, "ham") or '<span class="muted">Nothing stood out.</span>',
                unsafe_allow_html=True,
            )
        st.caption(
            "Words are ranked by how much the verdict changes when each one is removed. "
            "Links, phone numbers and money amounts are grouped, because their exact value "
            "rarely matters."
        )
    elif not text.strip():
        st.info("Your result will appear here. Nothing you paste is stored.", icon="💡")

# ------------------------------------------------------------- scan a mailbox


def run_scan(file_bytes: bytes) -> ScanResult:
    # mailbox.mbox needs a real path; the file is deleted as soon as it is parsed.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "upload.mbox"
        path.write_bytes(file_bytes)
        return scan_mbox(classifier, path, max_messages=settings.max_messages)


with scan_tab:
    st.markdown(
        "Upload a mailbox export and SpamShield will check every email in it, "
        "listing the most suspicious first."
    )
    with st.expander("How do I get an .mbox file?"):
        st.markdown(
            "- **Gmail:** go to [Google Takeout](https://takeout.google.com), select only "
            "*Mail*, and download the archive. The `.mbox` file is inside.\n"
            "- **Thunderbird:** install the *ImportExportTools NG* add-on, right-click a "
            "folder and choose *Export folder*.\n"
            "- **Apple Mail:** select a mailbox, then *Mailbox → Export Mailbox…*.\n\n"
            f"Up to {settings.max_messages:,} emails are scanned per file. The file is "
            "processed in memory and deleted straight away."
        )

    upload = st.file_uploader("Mailbox file", type=["mbox"], label_visibility="collapsed")

    if upload is None:
        st.session_state.pop("scan", None)
        st.info("Upload an .mbox file to get started.", icon="📂")
    else:
        scan_key = upload.file_id
        cached = st.session_state.get("scan")
        if cached is None or cached[0] != scan_key:
            with st.spinner(f"Scanning {upload.name}…"):
                try:
                    st.session_state.scan = (scan_key, run_scan(upload.getvalue()))
                except MailboxError as exc:
                    st.session_state.pop("scan", None)
                    st.error(str(exc))
                except Exception:
                    logger.exception("Mailbox scan failed")
                    st.session_state.pop("scan", None)
                    st.error("Something went wrong while reading this file. Is it a valid mbox?")

        if "scan" in st.session_state:
            result: ScanResult = st.session_state.scan[1]
            if result.total == 0:
                st.warning("This mailbox has no readable emails.")
            else:
                share = result.spam_count / result.total
                m1, m2, m3 = st.columns(3)
                m1.metric("Emails scanned", f"{result.total:,}")
                m2.metric("Flagged as spam", f"{result.spam_count:,}")
                m3.metric("Spam share", f"{share:.0%}")
                if result.truncated:
                    st.warning(f"Only the first {settings.max_messages:,} emails were scanned.")
                if result.skipped:
                    st.caption(f"{result.skipped} malformed email(s) could not be read.")

                f1, f2 = st.columns([1, 2])
                show = f1.segmented_control(
                    "Show", ["All", "Spam", "Not spam"], default="All", key="scan_filter"
                )
                query = f2.text_input("Search sender or subject", key="scan_query")

                frame = result.frame
                if show in ("Spam", "Not spam"):
                    frame = frame[frame["Verdict"] == show]
                if query:
                    needle = query.lower()
                    frame = frame[
                        frame["From"].str.lower().str.contains(needle, regex=False)
                        | frame["Subject"].str.lower().str.contains(needle, regex=False)
                    ]

                if frame.empty:
                    st.info("No emails match these filters.")
                else:
                    st.dataframe(
                        frame,
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "Spam probability": st.column_config.ProgressColumn(
                                format="percent", min_value=0.0, max_value=1.0
                            ),
                            "Preview": st.column_config.TextColumn(width="large"),
                        },
                    )
                st.download_button(
                    "Download results as CSV",
                    data=to_safe_csv(result.frame),
                    file_name=f"spamshield-{datetime.now():%Y%m%d-%H%M}.csv",
                    mime="text/csv",
                )

# ------------------------------------------------------------- about the model


def render_about(card: ModelCard) -> None:
    metrics = card.test_metrics
    st.markdown(
        f"The current model is a **{card.model_display_name}** trained on "
        f"{card.n_train + card.n_test:,} unique labelled text messages. It was tested on "
        f"**{card.n_test:,} messages it never saw during training**:"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spam caught", f"{metrics.recall:.1%}", help="Recall: share of spam detected")
    c2.metric(
        "Flags that were right",
        f"{metrics.precision:.1%}",
        help="Precision: share of spam verdicts that really were spam",
    )
    c3.metric(
        "False alarms",
        f"{metrics.false_positive_rate:.1%}",
        help="False-positive rate: share of normal messages wrongly flagged as spam",
    )
    c4.metric("F1 score", f"{metrics.f1:.3f}", help="Balance of precision and recall")

    (tn, fp), (fn, tp) = metrics.confusion_matrix
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Test results in detail**")
        st.markdown(
            f"""<div class="cm">
              <div></div><div class="head">Predicted normal</div>
              <div class="head">Predicted spam</div>
              <div class="head">Actually normal</div><div class="good">{tn}</div>
              <div class="bad">{fp}</div>
              <div class="head">Actually spam</div><div class="bad">{fn}</div>
              <div class="good">{tp}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("**How the model was chosen**")
        st.markdown(
            f"{len(card.candidates)} model types were tuned with {card.cv_folds}-fold "
            "cross-validation on the training data. The one with the best spam F1 score won."
        )

    comparison = pd.DataFrame(
        {
            "Model": [c.display_name for c in card.candidates],
            "Cross-validated F1": [c.cv_f1 for c in card.candidates],
            "Precision": [c.cv_precision for c in card.candidates],
            "Recall": [c.cv_recall for c in card.candidates],
        }
    )
    st.dataframe(
        comparison,
        hide_index=True,
        width="stretch",
        column_config={
            "Cross-validated F1": st.column_config.ProgressColumn(
                format="%.3f", min_value=0.0, max_value=1.0
            ),
            "Precision": st.column_config.NumberColumn(format="%.3f"),
            "Recall": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    st.markdown("**Good to know**")
    st.markdown(
        "- The training data is the public *SMS Spam Collection*. Emails are longer and "
        "worded differently, so treat results on emails as a strong hint, not a guarantee.\n"
        "- Spam keeps changing. A model trained on older messages can miss new tricks.\n"
        "- Everything runs locally in this app: messages are never sent to a third party "
        "or saved."
    )
    st.caption(
        f"Trained {card.trained_at[:10]} · scikit-learn {card.sklearn_version} · "
        f"spam threshold {classifier.threshold:.0%}"
    )


with about_tab:
    if card is None:
        st.info("No model card is available for this model.")
    else:
        render_about(card)
