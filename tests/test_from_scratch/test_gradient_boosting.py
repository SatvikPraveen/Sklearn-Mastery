"""Tests for the from-scratch gradient boosting machines."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss, r2_score

from sklearn_mastery.from_scratch.gradient_boosting import (
    GradientBoostingClassifierScratch,
    GradientBoostingRegressorScratch,
)


class TestAPI:
    def test_classifier_smoke(self, api_smoke, binary_data):
        X_tr, _, y_tr, _ = binary_data
        api_smoke(
            GradientBoostingClassifierScratch(n_estimators=10, random_state=0), X_tr, y_tr, classifier=True
        )

    def test_regressor_smoke(self, api_smoke, regression_data):
        X_tr, _, y_tr, _ = regression_data
        api_smoke(
            GradientBoostingRegressorScratch(n_estimators=10, random_state=0), X_tr, y_tr, classifier=False
        )

    def test_determinism_with_subsampling(self, regression_data):
        X_tr, X_te, y_tr, _ = regression_data
        a = GradientBoostingRegressorScratch(n_estimators=10, subsample=0.6, random_state=9).fit(X_tr, y_tr)
        b = GradientBoostingRegressorScratch(n_estimators=10, subsample=0.6, random_state=9).fit(X_tr, y_tr)
        c = GradientBoostingRegressorScratch(n_estimators=10, subsample=0.6, random_state=10).fit(X_tr, y_tr)
        np.testing.assert_array_equal(a.predict(X_te), b.predict(X_te))
        assert not np.array_equal(a.predict(X_te), c.predict(X_te))

    @pytest.mark.parametrize(
        "kwargs", [{"loss": "poisson"}, {"n_estimators": 0}, {"learning_rate": 0}, {"subsample": 1.5}]
    )
    def test_invalid_options(self, kwargs, regression_data):
        X_tr, _, y_tr, _ = regression_data
        with pytest.raises(ValueError):
            GradientBoostingRegressorScratch(**kwargs).fit(X_tr, y_tr)


class TestRegressorHandChecks:
    def test_first_tree_fits_residuals_of_the_mean(self):
        rng = np.random.RandomState(0)
        X = rng.rand(16, 2)
        y = rng.rand(16) * 10
        gb = GradientBoostingRegressorScratch(n_estimators=1, learning_rate=1.0, max_depth=None).fit(X, y)
        assert gb.init_ == pytest.approx(y.mean())
        np.testing.assert_allclose(gb.estimators_[0].predict(X), y - y.mean())
        np.testing.assert_allclose(gb.predict(X), y)

    def test_shrinkage_scales_the_update(self):
        rng = np.random.RandomState(1)
        X = rng.rand(30, 2)
        y = rng.rand(30)
        gb = GradientBoostingRegressorScratch(n_estimators=1, learning_rate=0.25, max_depth=2).fit(X, y)
        np.testing.assert_allclose(gb.predict(X), gb.init_ + 0.25 * gb.estimators_[0].predict(X))

    def test_absolute_error_uses_median_init_and_leaf_medians(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        y = np.array([1.0, 2.0, 30.0, 10.0, 11.0, 12.0])
        gb = GradientBoostingRegressorScratch(
            loss="absolute_error", n_estimators=1, learning_rate=1.0, max_depth=1
        )
        gb.fit(X, y)
        assert gb.init_ == pytest.approx(np.median(y))
        # a single split -> each leaf value is the median residual of its members
        leaf_ids = gb.estimators_[0].apply(X)
        for leaf in np.unique(leaf_ids):
            members = leaf_ids == leaf
            expected = np.median(y[members] - gb.init_)
            assert gb.estimators_[0].predict(X[members])[0] == pytest.approx(expected)

    def test_huber_leaf_update_formula(self):
        rng = np.random.RandomState(2)
        X = rng.rand(40, 1)
        y = 5 * X[:, 0] + rng.randn(40)
        y[:3] += 30  # outliers
        gb = GradientBoostingRegressorScratch(
            loss="huber", n_estimators=1, learning_rate=1.0, max_depth=1, alpha=0.9
        )
        gb.fit(X, y)
        diff = y - gb.init_
        delta = np.percentile(np.abs(diff), 90)
        leaf_ids = gb.estimators_[0].apply(X)
        for leaf in np.unique(leaf_ids):
            members = leaf_ids == leaf
            med = np.median(diff[members])
            dev = diff[members] - med
            expected = med + np.mean(np.sign(dev) * np.minimum(delta, np.abs(dev)))
            assert gb.estimators_[0].predict(X[members])[0] == pytest.approx(expected)

    def test_train_score_decreases_and_staged_predict(self, regression_data):
        X_tr, X_te, y_tr, _ = regression_data
        gb = GradientBoostingRegressorScratch(n_estimators=20, random_state=0).fit(X_tr, y_tr)
        assert gb.train_score_.shape == (20,)
        assert gb.train_score_[-1] < gb.train_score_[0]
        stages = list(gb.staged_predict(X_te))
        assert len(stages) == 20
        np.testing.assert_allclose(stages[-1], gb.predict(X_te))


class TestClassifierHandChecks:
    def test_binary_init_is_log_odds_of_prior(self, binary_data):
        X_tr, _, y_tr, _ = binary_data
        gb = GradientBoostingClassifierScratch(n_estimators=1, random_state=0).fit(X_tr, y_tr)
        p = y_tr.mean()
        assert gb.init_[0] == pytest.approx(np.log(p / (1 - p)))

    def test_binary_newton_leaf_update(self, binary_data):
        X_tr, _, y_tr, _ = binary_data
        gb = GradientBoostingClassifierScratch(n_estimators=1, learning_rate=1.0, max_depth=1, random_state=0)
        gb.fit(X_tr, y_tr)
        p = np.full(len(y_tr), y_tr.mean())
        residual = y_tr - p
        tree = gb.estimators_[0][0]
        leaf_ids = tree.apply(X_tr)
        for leaf in np.unique(leaf_ids):
            members = leaf_ids == leaf
            expected = residual[members].sum() / (p[members] * (1 - p[members])).sum()
            assert tree.predict(X_tr[members])[0] == pytest.approx(expected)

    def test_multiclass_leaf_update_uses_k_minus_one_over_k(self, multiclass_data):
        X_tr, _, y_tr, _ = multiclass_data
        gb = GradientBoostingClassifierScratch(n_estimators=1, learning_rate=1.0, max_depth=1, random_state=0)
        gb.fit(X_tr, y_tr)
        K = 3
        prior = np.bincount(y_tr) / len(y_tr)
        assert len(gb.estimators_[0]) == K
        for k, tree in enumerate(gb.estimators_[0]):
            r = (y_tr == k).astype(float) - prior[k]
            leaf_ids = tree.apply(X_tr)
            for leaf in np.unique(leaf_ids):
                m = leaf_ids == leaf
                expected = (K - 1) / K * r[m].sum() / (np.abs(r[m]) * (1 - np.abs(r[m]))).sum()
                assert tree.predict(X_tr[m])[0] == pytest.approx(expected)

    def test_staged_outputs(self, multiclass_data):
        X_tr, X_te, y_tr, _ = multiclass_data
        gb = GradientBoostingClassifierScratch(n_estimators=6, random_state=0).fit(X_tr, y_tr)
        probas = list(gb.staged_predict_proba(X_te))
        assert len(probas) == 6
        np.testing.assert_allclose(probas[-1].sum(axis=1), 1.0)
        np.testing.assert_allclose(probas[-1], gb.predict_proba(X_te))
        preds = list(gb.staged_predict(X_te))
        np.testing.assert_array_equal(preds[-1], gb.predict(X_te))
        assert gb.decision_function(X_te).shape == (X_te.shape[0], 3)


class TestAgainstSklearn:
    def test_regressor_close_to_sklearn(self, diabetes_data):
        X_tr, X_te, y_tr, y_te = diabetes_data
        ours = GradientBoostingRegressorScratch(n_estimators=60, random_state=0).fit(X_tr, y_tr)
        ref = GradientBoostingRegressor(n_estimators=60, random_state=0).fit(X_tr, y_tr)
        assert r2_score(y_te, ours.predict(X_te)) >= r2_score(y_te, ref.predict(X_te)) - 0.05

    @pytest.mark.parametrize("loss", ["absolute_error", "huber"])
    def test_robust_losses_close_to_sklearn(self, loss, regression_data):
        X_tr, X_te, y_tr, y_te = regression_data
        ours = GradientBoostingRegressorScratch(loss=loss, n_estimators=40, random_state=0).fit(X_tr, y_tr)
        ref = GradientBoostingRegressor(loss=loss, n_estimators=40, random_state=0).fit(X_tr, y_tr)
        assert r2_score(y_te, ours.predict(X_te)) >= r2_score(y_te, ref.predict(X_te)) - 0.05

    def test_binary_classifier_close_to_sklearn(self, cancer_data):
        X_tr, X_te, y_tr, y_te = cancer_data
        ours = GradientBoostingClassifierScratch(n_estimators=40, random_state=0).fit(X_tr, y_tr)
        ref = GradientBoostingClassifier(n_estimators=40, random_state=0).fit(X_tr, y_tr)
        assert accuracy_score(y_te, ours.predict(X_te)) >= accuracy_score(y_te, ref.predict(X_te)) - 0.05
        assert (
            log_loss(y_te, ours.predict_proba(X_te)) <= 1.2 * log_loss(y_te, ref.predict_proba(X_te)) + 0.02
        )

    def test_multiclass_classifier_close_to_sklearn(self, multiclass_data):
        X_tr, X_te, y_tr, y_te = multiclass_data
        ours = GradientBoostingClassifierScratch(n_estimators=30, random_state=0).fit(X_tr, y_tr)
        ref = GradientBoostingClassifier(n_estimators=30, random_state=0).fit(X_tr, y_tr)
        assert accuracy_score(y_te, ours.predict(X_te)) >= accuracy_score(y_te, ref.predict(X_te)) - 0.05

    def test_stochastic_boosting_still_fits(self, regression_data):
        X_tr, X_te, y_tr, y_te = regression_data
        gb = GradientBoostingRegressorScratch(n_estimators=40, subsample=0.5, random_state=0).fit(X_tr, y_tr)
        assert r2_score(y_te, gb.predict(X_te)) > 0.6
