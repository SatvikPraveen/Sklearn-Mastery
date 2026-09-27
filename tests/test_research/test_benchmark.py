"""Tests for the benchmark harness, reproducibility and reporting."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_wine, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn_mastery.research import (
    BenchmarkResult,
    BenchmarkSuite,
    RunManifest,
    capture_environment,
    config_hash,
    results_to_latex,
    results_to_markdown,
    set_global_seed,
    to_jsonable,
)


@pytest.fixture(scope="module")
def clf_suite():
    return BenchmarkSuite(
        estimators={
            "logreg": LogisticRegression(max_iter=1000),
            "tree": DecisionTreeClassifier(random_state=0),
        },
        datasets={
            "iris": load_iris(return_X_y=True),
            "wine": lambda: load_wine(return_X_y=True),
        },
        scoring=["accuracy", "f1_macro"],
        n_splits=3,
        n_repeats=2,
        random_state=0,
    )


@pytest.fixture(scope="module")
def clf_result(clf_suite):
    return clf_suite.run()


class TestBenchmarkSuite:
    def test_tidy_shape(self, clf_result):
        df = clf_result.results
        assert set(df.columns) == {
            "dataset",
            "estimator",
            "repeat",
            "fold",
            "metric",
            "value",
            "fit_time",
            "score_time",
        }
        assert len(df) == 2 * 2 * 2 * 3 * 2  # datasets * estimators * repeats * folds * metrics
        assert df["value"].between(0, 1).all()

    def test_score_matrix_and_ranks(self, clf_result):
        m = clf_result.score_matrix("accuracy")
        assert m.shape == (2, 2)
        assert list(m.index) == ["iris", "wine"]
        ranks = clf_result.rank_table("accuracy")
        assert "average_rank" in ranks.index
        assert ranks.loc["average_rank"].between(1, 2).all()
        with pytest.raises(KeyError):
            clf_result.score_matrix("nope")

    def test_paired_scores_aligned(self, clf_result):
        paired = clf_result.paired_scores("accuracy", "iris")
        assert paired.shape == (6, 2)
        assert not paired.isna().any().any()

    def test_summary(self, clf_result):
        s = clf_result.summary("accuracy")
        assert s.shape == (2, 2)
        full = clf_result.summary()
        assert {"mean", "std", "count"} <= set(full.columns)

    def test_deterministic(self, clf_suite):
        a = clf_suite.run().results["value"].to_numpy()
        b = clf_suite.run().results["value"].to_numpy()
        np.testing.assert_allclose(a, b)

    def test_regression_and_default_scoring(self):
        X, y = make_regression(n_samples=120, n_features=5, noise=1.0, random_state=0)
        suite = BenchmarkSuite(
            {"ridge": Ridge(), "tree": DecisionTreeRegressor(random_state=0)},
            {"synthetic": (X, y)},
            n_splits=4,
        )
        res = suite.run()
        assert set(res.results["metric"]) == {"score"}
        assert res.score_matrix("score").loc["synthetic", "ridge"] > 0.9

    def test_nested_tuning_records_params(self):
        X, y = load_iris(return_X_y=True)
        suite = BenchmarkSuite(
            {"tree": DecisionTreeClassifier(random_state=0)},
            {"iris": (X, y)},
            scoring="accuracy",
            n_splits=3,
            param_grids={"tree": {"max_depth": [1, 3]}},
            inner_cv=2,
        )
        res = suite.run()
        assert len(res.best_params["iris"]["tree"]) == 3
        assert all("max_depth" in p for p in res.best_params["iris"]["tree"])

    def test_explicit_cv_and_parallel(self):
        X, y = load_iris(return_X_y=True)
        suite = BenchmarkSuite(
            {"logreg": LogisticRegression(max_iter=500)},
            {"iris": (X, y)},
            scoring={"acc": "accuracy"},
            cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=1),
            n_jobs=2,
        )
        res = suite.run()
        assert res.results["fold"].nunique() == 4
        assert set(res.results["metric"]) == {"acc"}

    def test_validation(self):
        with pytest.raises(ValueError):
            BenchmarkSuite({}, {"d": (np.zeros((4, 2)), np.zeros(4))})
        with pytest.raises(ValueError):
            BenchmarkSuite({"a": Ridge()}, {})
        with pytest.raises(KeyError):
            BenchmarkSuite({"a": Ridge()}, {"d": (np.zeros((4, 2)), np.zeros(4))}, param_grids={"zzz": {}})

    def test_save_and_load_roundtrip(self, clf_result, tmp_path):
        out = clf_result.save(tmp_path / "run")
        assert (out / "results.csv").exists() and (out / "manifest.json").exists()
        loaded = BenchmarkResult.load(out)
        pd.testing.assert_frame_equal(
            loaded.results.sort_index(axis=1), clf_result.results.sort_index(axis=1), check_dtype=False
        )
        assert loaded.manifest.hash == clf_result.manifest.hash


class TestReproducibility:
    def test_set_global_seed(self):
        g1 = set_global_seed(7)
        x1 = np.random.rand(3)
        g2 = set_global_seed(7)
        x2 = np.random.rand(3)
        np.testing.assert_array_equal(x1, x2)
        assert g1.random() == g2.random()
        with pytest.raises(ValueError):
            set_global_seed(-1)

    def test_environment_and_hash(self):
        env = capture_environment()
        assert env["packages"]["scikit-learn"] is not None
        assert "python" in env
        h1 = config_hash({"a": 1, "b": [1, 2]})
        h2 = config_hash({"b": [1, 2], "a": 1})
        assert h1 == h2 and len(h1) == 12
        assert config_hash({"a": 2}) != h1

    def test_manifest_roundtrip(self, tmp_path):
        m = RunManifest(
            name="x", config={"est": Ridge(alpha=2.0), "arr": np.arange(3)}, seed=3, tags={"k": "v"}
        )
        path = m.save(tmp_path / "m.json")
        data = json.loads(path.read_text())
        assert data["config"]["est"]["params"]["alpha"] == 2.0
        loaded = RunManifest.load(path)
        assert loaded.hash == m.hash and loaded.tags == {"k": "v"}

    def test_to_jsonable(self):
        out = to_jsonable({"a": np.float32(1.5), "b": {1, 2}, "c": (np.array([1]),)})
        assert out == {"a": 1.5, "b": [1, 2], "c": [[1]]}


class TestReporting:
    def test_markdown_and_latex(self, clf_result):
        md = results_to_markdown(clf_result.results, "accuracy", caption="Accuracy")
        assert md.startswith("| dataset |") and "**" in md and "avg. rank" in md
        tex = results_to_latex(clf_result.results, "accuracy", caption="Acc", label="tab:acc")
        assert r"\begin{table}" in tex and r"\textbf{" in tex and r"\label{tab:acc}" in tex
        with pytest.raises(KeyError):
            results_to_markdown(clf_result.results, "missing")
