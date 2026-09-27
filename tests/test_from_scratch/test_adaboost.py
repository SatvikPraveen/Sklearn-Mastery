"""Tests for the from-scratch AdaBoost (SAMME and AdaBoost.R2)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn_mastery.from_scratch.adaboost import AdaBoostClassifierScratch, AdaBoostRegressorScratch
from sklearn_mastery.from_scratch.decision_tree import DecisionTreeClassifierScratch


class _FixedPredictor(ClassifierMixin, BaseEstimator):
    """Weak learner that ignores the data and returns a fixed prediction vector."""

    def __init__(self, prediction=None):
        self.prediction = prediction

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.asarray(self.prediction)[: X.shape[0]]


class TestAPI:
    def test_classifier_smoke(self, api_smoke, binary_data):
        X_tr, _, y_tr, _ = binary_data
        api_smoke(AdaBoostClassifierScratch(n_estimators=10, random_state=0), X_tr, y_tr, classifier=True)

    def test_regressor_smoke(self, api_smoke, regression_data):
        X_tr, _, y_tr, _ = regression_data
        api_smoke(AdaBoostRegressorScratch(n_estimators=10, random_state=0), X_tr, y_tr, classifier=False)

    def test_determinism(self, regression_data):
        X_tr, X_te, y_tr, _ = regression_data
        a = AdaBoostRegressorScratch(n_estimators=8, random_state=5).fit(X_tr, y_tr)
        b = AdaBoostRegressorScratch(n_estimators=8, random_state=5).fit(X_tr, y_tr)
        np.testing.assert_array_equal(a.predict(X_te), b.predict(X_te))
        np.testing.assert_array_equal(a.estimator_weights_, b.estimator_weights_)

    def test_learner_without_sample_weight_is_rejected(self, binary_data):
        X_tr, _, y_tr, _ = binary_data
        with pytest.raises(ValueError, match="sample_weight"):
            AdaBoostClassifierScratch(estimator=KNeighborsClassifier()).fit(X_tr, y_tr)

    def test_invalid_options(self, binary_data):
        X_tr, _, y_tr, _ = binary_data
        with pytest.raises(ValueError):
            AdaBoostClassifierScratch(n_estimators=0).fit(X_tr, y_tr)
        with pytest.raises(ValueError):
            AdaBoostRegressorScratch(loss="cubic").fit(X_tr, y_tr.astype(float))


class TestSAMMEHandChecks:
    def test_alpha_for_known_weighted_error_binary(self):
        # 1 of 4 uniformly weighted samples wrong -> err = 0.25, alpha = log(0.75/0.25) + log(K-1) = log 3
        X = np.zeros((4, 1))
        y = np.array([0, 1, 1, 0])
        learner = _FixedPredictor(prediction=[0, 1, 1, 1])
        ada = AdaBoostClassifierScratch(estimator=learner, n_estimators=5).fit(X, y)
        assert ada.estimator_errors_[0] == pytest.approx(0.25)
        assert ada.estimator_weights_[0] == pytest.approx(np.log(3.0))
        # after re-weighting the wrong sample carries weight 3/6 -> err = 0.5 >= 1 - 1/K: boosting stops
        assert len(ada.estimators_) == 1

    def test_alpha_includes_log_k_minus_one_for_multiclass(self):
        X = np.zeros((4, 1))
        y = np.array([0, 1, 2, 0])
        learner = _FixedPredictor(prediction=[0, 1, 2, 1])
        ada = AdaBoostClassifierScratch(estimator=learner, n_estimators=3, learning_rate=0.5).fit(X, y)
        assert ada.estimator_weights_[0] == pytest.approx(0.5 * (np.log(3.0) + np.log(2.0)))

    def test_learner_worse_than_chance_on_first_round_raises(self):
        X = np.zeros((4, 1))
        y = np.array([0, 1, 1, 0])
        with pytest.raises(ValueError, match="random guessing"):
            AdaBoostClassifierScratch(estimator=_FixedPredictor(prediction=[1, 0, 0, 1])).fit(X, y)

    def test_perfect_learner_stops_with_unit_weight(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])
        ada = AdaBoostClassifierScratch(n_estimators=10).fit(X, y)
        assert len(ada.estimators_) == 1
        np.testing.assert_array_equal(ada.estimator_weights_, [1.0])
        np.testing.assert_array_equal(ada.estimator_errors_, [0.0])

    def test_sample_weights_concentrate_on_misclassified(self, binary_data):
        # every kept learner must have error < 1 - 1/K and positive alpha
        X_tr, _, y_tr, _ = binary_data
        ada = AdaBoostClassifierScratch(n_estimators=15, random_state=0).fit(X_tr, y_tr)
        assert np.all(ada.estimator_errors_ < 0.5)
        assert np.all(ada.estimator_weights_ > 0)


class TestSAMMEPrediction:
    def test_staged_and_decision_function_shapes(self, binary_data, multiclass_data):
        X_tr, X_te, y_tr, _ = binary_data
        ada = AdaBoostClassifierScratch(n_estimators=8, random_state=0).fit(X_tr, y_tr)
        stages = list(ada.staged_predict(X_te))
        assert len(stages) == len(ada.estimators_)
        np.testing.assert_array_equal(stages[-1], ada.predict(X_te))
        assert ada.decision_function(X_te).shape == (X_te.shape[0],)
        probs = list(ada.staged_predict_proba(X_te))
        np.testing.assert_allclose(probs[-1], ada.predict_proba(X_te))

        X_tr, X_te, y_tr, _ = multiclass_data
        ada = AdaBoostClassifierScratch(n_estimators=8, random_state=0).fit(X_tr, y_tr)
        assert ada.decision_function(X_te).shape == (X_te.shape[0], 3)
        assert len(list(ada.staged_decision_function(X_te))) == len(ada.estimators_)

    def test_binary_decision_sign_matches_prediction(self, binary_data):
        X_tr, X_te, y_tr, _ = binary_data
        ada = AdaBoostClassifierScratch(n_estimators=8, random_state=0).fit(X_tr, y_tr)
        decision = ada.decision_function(X_te)
        expected = ada.classes_[(decision > 0).astype(int)]
        mask = decision != 0
        np.testing.assert_array_equal(ada.predict(X_te)[mask], expected[mask])

    def test_close_to_sklearn_binary(self, cancer_data):
        X_tr, X_te, y_tr, y_te = cancer_data
        ours = AdaBoostClassifierScratch(n_estimators=30, random_state=0).fit(X_tr, y_tr)
        ref = AdaBoostClassifier(n_estimators=30, random_state=0).fit(X_tr, y_tr)
        assert accuracy_score(y_te, ours.predict(X_te)) >= accuracy_score(y_te, ref.predict(X_te)) - 0.05

    def test_close_to_sklearn_multiclass(self, multiclass_data):
        X_tr, X_te, y_tr, y_te = multiclass_data
        stump = DecisionTreeClassifierScratch(max_depth=2)
        ours = AdaBoostClassifierScratch(estimator=stump, n_estimators=30, random_state=0).fit(X_tr, y_tr)
        ref = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2), n_estimators=30, random_state=0
        )
        ref.fit(X_tr, y_tr)
        assert accuracy_score(y_te, ours.predict(X_te)) >= accuracy_score(y_te, ref.predict(X_te)) - 0.05

    def test_sklearn_base_learner_works(self, binary_data):
        X_tr, X_te, y_tr, y_te = binary_data
        ada = AdaBoostClassifierScratch(estimator=LogisticRegression(max_iter=300), n_estimators=5).fit(
            X_tr, y_tr
        )
        assert accuracy_score(y_te, ada.predict(X_te)) > 0.7


class TestAdaBoostR2:
    def test_weighted_median_hand_check(self):
        reg = AdaBoostRegressorScratch()
        preds = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
        np.testing.assert_array_equal(reg._weighted_median(preds, np.array([1.0, 1.0, 1.0])), [2.0, 2.0])
        # the member with weight 3 dominates: cumulative weight reaches half at its prediction
        np.testing.assert_array_equal(reg._weighted_median(preds, np.array([3.0, 1.0, 1.0])), [1.0, 3.0])

    def test_beta_and_weights_hand_check(self):
        # with a 1-D linear target and depth-1 stumps the first round's average loss is in (0, 0.5)
        X = np.linspace(0, 1, 20).reshape(-1, 1)
        y = X[:, 0].copy()
        reg = AdaBoostRegressorScratch(n_estimators=1, random_state=0).fit(X, y)
        avg_loss = reg.estimator_errors_[0]
        assert 0.0 < avg_loss < 0.5
        beta = avg_loss / (1.0 - avg_loss)
        assert reg.estimator_weights_[0] == pytest.approx(np.log(1.0 / beta))

    @pytest.mark.parametrize("loss", ["linear", "square", "exponential"])
    def test_losses_run_and_fit(self, loss, regression_data):
        X_tr, X_te, y_tr, y_te = regression_data
        reg = AdaBoostRegressorScratch(n_estimators=15, loss=loss, random_state=0).fit(X_tr, y_tr)
        assert r2_score(y_te, reg.predict(X_te)) > 0.5
        assert np.all(reg.estimator_errors_ < 0.5)

    def test_staged_predict(self, regression_data):
        X_tr, X_te, y_tr, _ = regression_data
        reg = AdaBoostRegressorScratch(n_estimators=6, random_state=0).fit(X_tr, y_tr)
        stages = list(reg.staged_predict(X_te))
        assert len(stages) == len(reg.estimators_)
        np.testing.assert_array_equal(stages[-1], reg.predict(X_te))

    def test_close_to_sklearn(self, diabetes_data):
        X_tr, X_te, y_tr, y_te = diabetes_data
        ours = AdaBoostRegressorScratch(n_estimators=30, random_state=0).fit(X_tr, y_tr)
        ref = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3), n_estimators=30, random_state=0)
        ref.fit(X_tr, y_tr)
        assert r2_score(y_te, ours.predict(X_te)) >= r2_score(y_te, ref.predict(X_te)) - 0.05
