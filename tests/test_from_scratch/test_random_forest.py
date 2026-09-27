"""Tests for the from-scratch random forests."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

from sklearn_mastery.from_scratch.random_forest import (
    RandomForestClassifierScratch,
    RandomForestRegressorScratch,
)


class TestAPI:
    def test_classifier_smoke(self, api_smoke, binary_data):
        X_tr, _, y_tr, _ = binary_data
        api_smoke(RandomForestClassifierScratch(n_estimators=5, random_state=0), X_tr, y_tr, classifier=True)

    def test_regressor_smoke(self, api_smoke, regression_data):
        X_tr, _, y_tr, _ = regression_data
        api_smoke(RandomForestRegressorScratch(n_estimators=5, random_state=0), X_tr, y_tr, classifier=False)

    def test_determinism_and_threads(self, multiclass_data):
        X_tr, X_te, y_tr, _ = multiclass_data
        a = RandomForestClassifierScratch(n_estimators=6, random_state=11).fit(X_tr, y_tr)
        b = RandomForestClassifierScratch(n_estimators=6, random_state=11, n_jobs=2).fit(X_tr, y_tr)
        np.testing.assert_array_equal(a.predict_proba(X_te), b.predict_proba(X_te))
        np.testing.assert_array_equal(a.feature_importances_, b.feature_importances_)

    def test_invalid_options(self, binary_data):
        X_tr, _, y_tr, _ = binary_data
        with pytest.raises(ValueError):
            RandomForestClassifierScratch(n_estimators=0).fit(X_tr, y_tr)
        with pytest.raises(ValueError):
            RandomForestClassifierScratch(oob_score=True, bootstrap=False).fit(X_tr, y_tr)


class TestForestMechanics:
    def test_per_split_feature_subsampling_defaults(self, binary_data, regression_data):
        X_tr, _, y_tr, _ = binary_data
        forest = RandomForestClassifierScratch(n_estimators=3, random_state=0).fit(X_tr, y_tr)
        assert all(t.max_features_ == int(np.sqrt(X_tr.shape[1])) for t in forest.estimators_)
        Xr, _, yr, _ = regression_data
        reg = RandomForestRegressorScratch(n_estimators=3, random_state=0).fit(Xr, yr)
        assert all(t.max_features_ == Xr.shape[1] for t in reg.estimators_)

    def test_trees_differ_from_each_other(self, binary_data):
        X_tr, X_te, y_tr, _ = binary_data
        forest = RandomForestClassifierScratch(n_estimators=4, random_state=0).fit(X_tr, y_tr)
        texts = {t.export_text() for t in forest.estimators_}
        assert len(texts) == 4
        assert len(forest.estimators_samples_) == 4

    def test_no_bootstrap_uses_all_rows(self, binary_data):
        X_tr, _, y_tr, _ = binary_data
        forest = RandomForestClassifierScratch(n_estimators=2, bootstrap=False, random_state=0).fit(
            X_tr, y_tr
        )
        for samples in forest.estimators_samples_:
            np.testing.assert_array_equal(samples, np.arange(len(y_tr)))

    def test_oob_score(self, binary_data):
        X_tr, X_te, y_tr, y_te = binary_data
        forest = RandomForestClassifierScratch(n_estimators=30, oob_score=True, random_state=0).fit(
            X_tr, y_tr
        )
        assert 0.0 <= forest.oob_score_ <= 1.0
        assert forest.oob_decision_function_.shape == (len(y_tr), 2)
        assert abs(forest.oob_score_ - accuracy_score(y_te, forest.predict(X_te))) < 0.1

    def test_oob_regression(self, regression_data):
        X_tr, X_te, y_tr, y_te = regression_data
        forest = RandomForestRegressorScratch(n_estimators=25, oob_score=True, random_state=0).fit(X_tr, y_tr)
        assert abs(forest.oob_score_ - r2_score(y_te, forest.predict(X_te))) < 0.15


class TestImportances:
    def test_mdi_sums_to_one_and_finds_informative_features(self):
        rng = np.random.RandomState(0)
        X = rng.randn(400, 6)
        y = (X[:, 0] + 0.8 * X[:, 1] > 0).astype(int)
        forest = RandomForestClassifierScratch(n_estimators=20, random_state=0).fit(X, y)
        mdi = forest.feature_importances_
        assert mdi.sum() == pytest.approx(1.0)
        assert set(np.argsort(mdi)[-2:]) == {0, 1}

    def test_permutation_importance(self):
        rng = np.random.RandomState(1)
        X = rng.randn(400, 6)
        y = 3 * X[:, 2] + X[:, 4] + 0.1 * rng.randn(400)
        forest = RandomForestRegressorScratch(n_estimators=15, max_depth=6, random_state=0).fit(X, y)
        result = forest.permutation_importance(X, y, n_repeats=3, random_state=0)
        assert result.importances.shape == (6, 3)
        assert result.importances_mean.shape == (6,)
        assert result.importances_std.shape == (6,)
        assert np.argmax(result.importances_mean) == 2
        assert result.importances_mean[2] > result.importances_mean[4] > 0
        noise = np.delete(result.importances_mean, [2, 4])
        assert np.all(np.abs(noise) < 0.05)
        again = forest.permutation_importance(X, y, n_repeats=3, random_state=0)
        np.testing.assert_array_equal(result.importances, again.importances)


class TestAgainstSklearn:
    def test_classifier_close_to_sklearn(self, cancer_data):
        X_tr, X_te, y_tr, y_te = cancer_data
        ours = RandomForestClassifierScratch(n_estimators=30, random_state=0).fit(X_tr, y_tr)
        ref = RandomForestClassifier(n_estimators=30, random_state=0).fit(X_tr, y_tr)
        assert accuracy_score(y_te, ours.predict(X_te)) >= accuracy_score(y_te, ref.predict(X_te)) - 0.05

    def test_multiclass_close_to_sklearn(self, multiclass_data):
        X_tr, X_te, y_tr, y_te = multiclass_data
        ours = RandomForestClassifierScratch(n_estimators=30, random_state=0).fit(X_tr, y_tr)
        ref = RandomForestClassifier(n_estimators=30, random_state=0).fit(X_tr, y_tr)
        assert accuracy_score(y_te, ours.predict(X_te)) >= accuracy_score(y_te, ref.predict(X_te)) - 0.05

    def test_regressor_close_to_sklearn(self, diabetes_data):
        X_tr, X_te, y_tr, y_te = diabetes_data
        ours = RandomForestRegressorScratch(n_estimators=30, random_state=0).fit(X_tr, y_tr)
        ref = RandomForestRegressor(n_estimators=30, random_state=0).fit(X_tr, y_tr)
        assert r2_score(y_te, ours.predict(X_te)) >= r2_score(y_te, ref.predict(X_te)) - 0.05
