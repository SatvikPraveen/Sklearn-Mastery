"""Tests for calibration diagnostics and bias-variance decomposition."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn_mastery.research import (
    bias_variance_decomposition,
    brier_score_decomposition,
    compute_calibration_bins,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_diagram,
)


class TestCalibration:
    def test_perfectly_calibrated_is_zero(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 200_000)
        y = (rng.uniform(0, 1, p.size) < p).astype(int)
        assert expected_calibration_error(y, p, n_bins=10) < 0.01
        assert maximum_calibration_error(y, p, n_bins=10) < 0.02

    def test_overconfident_has_large_ece(self):
        y = np.array([0, 1] * 50)
        p = np.full(100, 0.95)  # says 95% but accuracy of positive class is 50%
        assert expected_calibration_error(y, p) == pytest.approx(0.45)

    def test_multiclass_top_label(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        P = np.eye(3)[y] * 0.8 + 0.1  # confident and correct
        assert expected_calibration_error(y, P) == pytest.approx(
            0.1, abs=1e-9
        )  # confidence 0.9, accuracy 1.0

    def test_bins_and_strategies(self):
        rng = np.random.default_rng(1)
        p = rng.beta(2, 5, 500)
        y = (rng.uniform(size=500) < p).astype(int)
        b_u = compute_calibration_bins(y, p, 5, "uniform")
        b_q = compute_calibration_bins(y, p, 5, "quantile")
        assert b_u.counts.sum() == 500 and b_q.counts.sum() == 500
        assert np.all(np.diff(b_q.counts) < 200)  # roughly equal-mass
        with pytest.raises(ValueError):
            compute_calibration_bins(y, p, 5, "bogus")

    def test_brier_decomposition_identity(self):
        rng = np.random.default_rng(2)
        p = rng.uniform(0, 1, 5000)
        y = (rng.uniform(size=5000) < p).astype(int)
        d = brier_score_decomposition(y, p, n_bins=20)
        assert d.brier == pytest.approx(d.reliability - d.resolution + d.uncertainty, abs=0.01)
        assert d.reliability < 0.01

    def test_reliability_diagram(self):
        rng = np.random.default_rng(3)
        p = rng.uniform(0, 1, 300)
        y = (rng.uniform(size=300) < p).astype(int)
        ax = reliability_diagram(y, p, label="m")
        assert ax.get_ylabel() == "empirical accuracy"


class TestBiasVariance:
    def test_mse_identity_and_model_ordering(self):
        X, y = make_regression(n_samples=400, n_features=5, noise=5.0, random_state=0)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)
        lin = bias_variance_decomposition(
            LinearRegression(), Xtr, ytr, Xte, yte, loss="mse", n_rounds=30, random_state=0
        )
        tree = bias_variance_decomposition(
            DecisionTreeRegressor(), Xtr, ytr, Xte, yte, loss="mse", n_rounds=30, random_state=0
        )
        for r in (lin, tree):
            assert r.expected_loss == pytest.approx(r.bias + r.variance, rel=1e-6)
        assert tree.variance > lin.variance  # trees are high-variance

    def test_zero_one_loss(self):
        X, y = make_classification(n_samples=400, n_features=8, n_informative=4, random_state=0)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)
        r = bias_variance_decomposition(
            DecisionTreeClassifier(),
            Xtr,
            ytr,
            Xte,
            yte,
            loss="0-1",
            n_rounds=25,
            random_state=1,
            return_predictions=True,
        )
        assert r.predictions.shape == (25, Xte.shape[0])
        assert 0 <= r.bias <= 1
        assert r.expected_loss == pytest.approx(r.bias + r.variance, abs=1e-9)
        assert r.variance_unbiased >= 0 and r.variance_biased >= 0
        lr = bias_variance_decomposition(
            LogisticRegression(max_iter=500), Xtr, ytr, Xte, yte, loss="0-1", n_rounds=25, random_state=1
        )
        assert lr.variance_unbiased < r.variance_unbiased

    def test_validation(self):
        X = np.zeros((10, 2))
        y = np.zeros(10)
        with pytest.raises(ValueError):
            bias_variance_decomposition(LinearRegression(), X, y, X, y, loss="huber")
        with pytest.raises(ValueError):
            bias_variance_decomposition(LinearRegression(), X, y, X, y, n_rounds=1)
