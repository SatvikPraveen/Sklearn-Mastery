"""Tests for the from-scratch bagging ensembles."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier

from sklearn_mastery.from_scratch.bagging import BaggingClassifierScratch, BaggingRegressorScratch
from sklearn_mastery.from_scratch.decision_tree import DecisionTreeClassifierScratch


class TestAPI:
    def test_classifier_smoke(self, api_smoke, binary_data):
        X_tr, _, y_tr, _ = binary_data
        api_smoke(BaggingClassifierScratch(n_estimators=5, random_state=0), X_tr, y_tr, classifier=True)

    def test_regressor_smoke(self, api_smoke, regression_data):
        X_tr, _, y_tr, _ = regression_data
        api_smoke(BaggingRegressorScratch(n_estimators=5, random_state=0), X_tr, y_tr, classifier=False)

    def test_determinism_and_thread_independence(self, binary_data):
        X_tr, X_te, y_tr, _ = binary_data
        a = BaggingClassifierScratch(n_estimators=8, random_state=3).fit(X_tr, y_tr)
        b = BaggingClassifierScratch(n_estimators=8, random_state=3, n_jobs=2).fit(X_tr, y_tr)
        c = BaggingClassifierScratch(n_estimators=8, random_state=4).fit(X_tr, y_tr)
        np.testing.assert_array_equal(a.predict_proba(X_te), b.predict_proba(X_te))
        assert not np.array_equal(a.predict_proba(X_te), c.predict_proba(X_te))
        for s_a, s_b in zip(a.estimators_samples_, b.estimators_samples_):
            np.testing.assert_array_equal(s_a, s_b)

    def test_invalid_options(self, binary_data):
        X_tr, _, y_tr, _ = binary_data
        with pytest.raises(ValueError):
            BaggingClassifierScratch(oob_score=True, bootstrap=False).fit(X_tr, y_tr)
        with pytest.raises(ValueError):
            BaggingClassifierScratch(voting="mean").fit(X_tr, y_tr)
        with pytest.raises(ValueError):
            BaggingClassifierScratch(max_samples=0).fit(X_tr, y_tr)


class TestResampling:
    def test_bootstrap_draws_with_replacement(self, binary_data):
        X_tr, _, y_tr, _ = binary_data
        bag = BaggingClassifierScratch(n_estimators=4, random_state=0).fit(X_tr, y_tr)
        for samples in bag.estimators_samples_:
            assert len(samples) == len(y_tr)
            assert len(np.unique(samples)) < len(y_tr)  # duplicates exist

    def test_no_bootstrap_draws_without_replacement(self, binary_data):
        X_tr, _, y_tr, _ = binary_data
        bag = BaggingClassifierScratch(n_estimators=4, bootstrap=False, max_samples=0.5, random_state=0).fit(
            X_tr, y_tr
        )
        for samples in bag.estimators_samples_:
            assert len(samples) == len(y_tr) // 2
            assert len(np.unique(samples)) == len(samples)

    def test_feature_bagging(self, binary_data):
        X_tr, X_te, y_tr, _ = binary_data
        bag = BaggingClassifierScratch(n_estimators=4, max_features=0.5, random_state=0).fit(X_tr, y_tr)
        for feats in bag.estimators_features_:
            assert len(feats) == X_tr.shape[1] // 2
            assert len(np.unique(feats)) == len(feats)
        assert bag.predict(X_te).shape == (X_te.shape[0],)
        with_replacement = BaggingClassifierScratch(
            n_estimators=6, max_features=1.0, bootstrap_features=True, random_state=0
        ).fit(X_tr, y_tr)
        assert any(len(np.unique(f)) < len(f) for f in with_replacement.estimators_features_)

    def test_custom_base_estimators(self, binary_data):
        X_tr, X_te, y_tr, y_te = binary_data
        for base in (DecisionTreeClassifier(max_depth=3), LogisticRegression(max_iter=200)):
            bag = BaggingClassifierScratch(estimator=base, n_estimators=5, random_state=0).fit(X_tr, y_tr)
            assert accuracy_score(y_te, bag.predict(X_te)) > 0.7

    def test_hard_voting(self, binary_data):
        X_tr, X_te, y_tr, y_te = binary_data
        hard = BaggingClassifierScratch(n_estimators=7, voting="hard", random_state=0).fit(X_tr, y_tr)
        proba = hard.predict_proba(X_te)
        # vote fractions are multiples of 1/7
        np.testing.assert_allclose(proba * 7, np.round(proba * 7), atol=1e-9)
        assert accuracy_score(y_te, hard.predict(X_te)) > 0.75


class TestOOB:
    def test_oob_classifier(self, binary_data):
        X_tr, X_te, y_tr, y_te = binary_data
        bag = BaggingClassifierScratch(n_estimators=25, oob_score=True, random_state=0).fit(X_tr, y_tr)
        assert 0.0 <= bag.oob_score_ <= 1.0
        assert bag.oob_decision_function_.shape == (len(y_tr), 2)
        assert abs(bag.oob_score_ - accuracy_score(y_te, bag.predict(X_te))) < 0.1

    def test_oob_regressor(self, regression_data):
        X_tr, X_te, y_tr, y_te = regression_data
        bag = BaggingRegressorScratch(n_estimators=25, oob_score=True, random_state=0).fit(X_tr, y_tr)
        assert bag.oob_prediction_.shape == (len(y_tr),)
        assert abs(bag.oob_score_ - r2_score(y_te, bag.predict(X_te))) < 0.15

    def test_oob_warns_when_some_rows_never_oob(self, binary_data):
        X_tr, _, y_tr, _ = binary_data
        with pytest.warns(UserWarning, match="never out-of-bag"):
            bag = BaggingClassifierScratch(n_estimators=2, oob_score=True, random_state=0).fit(X_tr, y_tr)
        assert np.isnan(bag.oob_decision_function_).any()


class TestAgainstSklearn:
    def test_classifier_close_to_sklearn(self, cancer_data):
        X_tr, X_te, y_tr, y_te = cancer_data
        ours = BaggingClassifierScratch(n_estimators=15, random_state=0).fit(X_tr, y_tr)
        ref = BaggingClassifier(n_estimators=15, random_state=0).fit(X_tr, y_tr)
        assert accuracy_score(y_te, ours.predict(X_te)) >= accuracy_score(y_te, ref.predict(X_te)) - 0.05

    def test_regressor_close_to_sklearn(self, diabetes_data):
        X_tr, X_te, y_tr, y_te = diabetes_data
        ours = BaggingRegressorScratch(n_estimators=15, random_state=0).fit(X_tr, y_tr)
        ref = BaggingRegressor(n_estimators=15, random_state=0).fit(X_tr, y_tr)
        assert r2_score(y_te, ours.predict(X_te)) >= r2_score(y_te, ref.predict(X_te)) - 0.05

    def test_bagging_beats_single_tree(self, binary_data):
        X_tr, X_te, y_tr, y_te = binary_data
        single = DecisionTreeClassifierScratch(random_state=0).fit(X_tr, y_tr)
        bag = BaggingClassifierScratch(n_estimators=20, random_state=0).fit(X_tr, y_tr)
        assert accuracy_score(y_te, bag.predict(X_te)) >= accuracy_score(y_te, single.predict(X_te))
