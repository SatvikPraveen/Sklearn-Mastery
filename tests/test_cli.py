"""End-to-end tests for the command-line interface."""

import json

import pandas as pd
import pytest
from click.testing import CliRunner

from sklearn_mastery.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_version(runner):
    out = runner.invoke(cli, ["--version"])
    assert out.exit_code == 0 and "sklearn-mastery" in out.output


def test_info(runner):
    out = runner.invoke(cli, ["info"])
    assert out.exit_code == 0
    data = json.loads(out.output[out.output.index("{") :])
    assert "python" in data and "packages" in data


def test_generate_and_train(runner, tmp_path):
    csv = tmp_path / "data.csv"
    out = runner.invoke(
        cli, ["generate-data", "--dataset-type", "classification", "--n-samples", "200", "-o", str(csv)]
    )
    assert out.exit_code == 0, out.output
    df = pd.read_csv(csv)
    assert "target" in df.columns and len(df) == 200

    out = runner.invoke(
        cli,
        [
            "train",
            str(csv),
            "--algorithm",
            "logistic_regression",
            "--cv-folds",
            "3",
            "-o",
            str(tmp_path / "run"),
        ],
    )
    assert out.exit_code == 0, out.output
    metrics = json.loads((tmp_path / "run" / "logistic_regression_metrics.json").read_text())
    assert 0.0 <= metrics["holdout_score"] <= 1.0
    assert (tmp_path / "run" / "logistic_regression.joblib").exists()

    out = runner.invoke(cli, ["train", str(csv), "--target", "missing"])
    assert out.exit_code != 0 and "not found" in out.output


def test_generate_regression_and_clustering(runner, tmp_path):
    out = runner.invoke(cli, ["generate-data", "--dataset-type", "regression", "-o", str(tmp_path / "r.csv")])
    assert out.exit_code == 0, out.output
    out = runner.invoke(cli, ["generate-data", "--dataset-type", "clustering", "-o", str(tmp_path / "c.csv")])
    assert out.exit_code == 0, out.output
    assert "target" not in pd.read_csv(tmp_path / "c.csv").columns


def test_benchmark_and_compare(runner, tmp_path):
    out_dir = tmp_path / "bench"
    out = runner.invoke(
        cli,
        ["benchmark", "--dataset", "iris", "--dataset", "wine", "--quick", "-o", str(out_dir)],
    )
    assert out.exit_code == 0, out.output
    assert "| dataset |" in out.output and "Friedman" in out.output
    assert (out_dir / "results.csv").exists() and (out_dir / "manifest.json").exists()

    png = tmp_path / "cd.png"
    out = runner.invoke(cli, ["compare", str(out_dir), "--cd-diagram", str(png)])
    assert out.exit_code == 0, out.output
    assert "Nemenyi critical difference" in out.output and png.exists()


def test_benchmark_single_dataset_bayesian(runner, tmp_path):
    out_dir = tmp_path / "bench1"
    out = runner.invoke(cli, ["benchmark", "--dataset", "iris", "--quick", "-o", str(out_dir)])
    assert out.exit_code == 0, out.output
    out = runner.invoke(cli, ["compare", str(out_dir)])
    assert out.exit_code == 0, out.output
    assert "Bayesian correlated t-test" in out.output


def test_benchmark_task_mismatch(runner):
    out = runner.invoke(
        cli, ["benchmark", "--dataset", "diabetes", "--task-type", "classification", "--quick"]
    )
    assert out.exit_code != 0 and "regression problem" in out.output
