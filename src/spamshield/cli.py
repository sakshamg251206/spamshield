"""Command-line interface: ``spamshield train | predict | scan``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from spamshield.config import DEFAULT_DATASET_PATH, ConfigError, Settings
from spamshield.logging_setup import configure_logging

logger = logging.getLogger("spamshield")


def _build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spamshield", description="Explainable spam detection for messages and mailboxes."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=settings.model_path,
        help="model file to load or write (default: %(default)s)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train, compare and save a model")
    train.add_argument("--data", type=Path, default=DEFAULT_DATASET_PATH, help="labelled CSV")
    train.add_argument("--cv-folds", type=int, default=5, help="cross-validation folds")
    train.add_argument(
        "--only",
        nargs="+",
        metavar="NAME",
        help="restrict to these candidates, e.g. logistic_regression naive_bayes",
    )

    predict = commands.add_parser("predict", help="classify one message")
    predict.add_argument("text", nargs="?", help="message text (reads stdin when omitted)")
    predict.add_argument("--explain", action="store_true", help="show the most influential words")

    scan = commands.add_parser("scan", help="classify every message in an .mbox file")
    scan.add_argument("mbox", type=Path)
    scan.add_argument("-o", "--output", type=Path, help="write results to this CSV file")
    return parser


def _train(args: argparse.Namespace) -> int:
    from spamshield.training import train

    card = train(args.data, args.model, cv_folds=args.cv_folds, only=args.only)
    metrics = card.test_metrics
    print(f"\nCandidates (ranked by {card.cv_folds}-fold cross-validated spam F1):")
    for result in card.candidates:
        print(f"  {result.display_name:<26} F1 {result.cv_f1:.4f} ± {result.cv_f1_std:.4f}")
    print(f"\nSelected: {card.model_display_name}  {card.best_params}")
    print(
        f"Held-out test set ({card.n_test} messages): "
        f"precision {metrics.precision:.3f}, recall {metrics.recall:.3f}, "
        f"F1 {metrics.f1:.3f}, false-positive rate {metrics.false_positive_rate:.2%}"
    )
    print(f"Saved to {args.model}")
    return 0


def _predict(args: argparse.Namespace, settings: Settings) -> int:
    from spamshield.classifier import SpamClassifier
    from spamshield.presentation import format_probability

    text = args.text if args.text is not None else sys.stdin.read()
    if not text.strip():
        print("error: message is empty", file=sys.stderr)
        return 2
    classifier = SpamClassifier.load(
        args.model, threshold=settings.threshold, max_chars=settings.max_chars
    )
    prediction = classifier.predict(text)
    verdict = "SPAM" if prediction.is_spam else "NOT SPAM"
    print(f"{verdict}  (spam probability {format_probability(prediction.spam_probability)})")
    if args.explain:
        for signal in classifier.explain(text):
            direction = "spam" if signal.impact > 0 else "ham"
            print(
                f"  {signal.display:<20} pushes towards {direction} ({signal.impact:+.2f} log-odds)"
            )
    return 0


def _scan(args: argparse.Namespace, settings: Settings) -> int:
    from spamshield.classifier import SpamClassifier
    from spamshield.scanner import scan_mbox, to_safe_csv

    classifier = SpamClassifier.load(
        args.model, threshold=settings.threshold, max_chars=settings.max_chars
    )
    result = scan_mbox(classifier, args.mbox, max_messages=settings.max_messages)
    print(f"Scanned {result.total} messages: {result.spam_count} flagged as spam.")
    if result.skipped:
        print(f"Skipped {result.skipped} unreadable message(s).")
    if result.truncated:
        print(f"Stopped after {settings.max_messages} messages (SPAMSHIELD_MAX_MESSAGES).")
    if args.output:
        args.output.write_bytes(to_safe_csv(result.frame))
        print(f"Results written to {args.output}")
    else:
        print(result.frame[["Verdict", "Spam probability", "From", "Subject"]].head(20).to_string())
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    configure_logging(settings.log_level)
    args = _build_parser(settings).parse_args(argv)

    from spamshield.classifier import ModelLoadError
    from spamshield.data import DatasetError
    from spamshield.mailbox_reader import MailboxError

    try:
        if args.command == "train":
            return _train(args)
        if args.command == "predict":
            return _predict(args, settings)
        return _scan(args, settings)
    except (ModelLoadError, DatasetError, MailboxError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
