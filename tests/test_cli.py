from pathlib import Path

import pytest

from spamshield.cli import main


def test_predict_with_explanation(trained_model: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "--model",
            str(trained_model),
            "predict",
            "--explain",
            "WIN a FREE prize, call 09061701461",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("SPAM")
    assert "pushes towards" in out


def test_predict_reads_stdin(
    trained_model: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("see you at lunch"))
    assert main(["--model", str(trained_model), "predict"]) == 0
    assert capsys.readouterr().out.startswith("NOT SPAM")


def test_predict_rejects_empty(trained_model: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--model", str(trained_model), "predict", "  "]) == 2
    assert "empty" in capsys.readouterr().err


def test_scan_writes_csv(
    trained_model: Path, sample_mbox: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "out.csv"
    assert main(["--model", str(trained_model), "scan", str(sample_mbox), "-o", str(output)]) == 0
    assert "Scanned 5 messages" in capsys.readouterr().out
    assert output.read_text(encoding="utf-8-sig").startswith("Verdict,Spam probability")


def test_missing_model_is_a_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--model", str(tmp_path / "none.joblib"), "predict", "hi"]) == 1
    assert "spamshield train" in capsys.readouterr().err


def test_bad_configuration_is_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SPAMSHIELD_THRESHOLD", "nope")
    assert main(["predict", "hi"]) == 2
    assert "SPAMSHIELD_THRESHOLD" in capsys.readouterr().err


@pytest.mark.slow
def test_train_command(
    small_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "cli.joblib"
    code = main(
        [
            "--model",
            str(model),
            "train",
            "--data",
            str(small_dataset),
            "--cv-folds",
            "3",
            "--only",
            "naive_bayes",
        ]
    )
    assert code == 0
    assert model.is_file()
    assert "Selected: Complement Naive Bayes" in capsys.readouterr().out


def test_train_rejects_unknown_candidate(small_dataset: Path, tmp_path: Path) -> None:
    code = main(
        [
            "--model",
            str(tmp_path / "m.joblib"),
            "train",
            "--data",
            str(small_dataset),
            "--only",
            "transformer",
        ]
    )
    assert code == 1
