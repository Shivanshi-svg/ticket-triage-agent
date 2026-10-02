import sys

import pytest

from triage import train


def test_train_cli_writes_model(tmp_path, monkeypatch, capsys):
    out = tmp_path / "model.joblib"
    monkeypatch.setattr(sys, "argv", ["train", "--out", str(out), "--n", "300"])
    train.main()
    assert out.exists() and "macro_f1" in capsys.readouterr().out


def test_train_cli_logs_to_mlflow(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)  # keep default artifact root (./mlruns) inside the temp dir
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(
        sys, "argv", ["train", "--out", str(tmp_path / "m.joblib"), "--n", "300", "--mlflow"]
    )
    train.main()
    runs = mlflow.search_runs(experiment_ids=["0"])
    assert len(runs) == 1 and runs.iloc[0]["metrics.macro_f1"] > 0.5
