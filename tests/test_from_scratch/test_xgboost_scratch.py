"""Tests for the from-scratch second-order (XGBoost-style) booster."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, r2_score

from sklearn_mastery.from_scratch.xgboost_scratch import XGBoostClassifierScratch, XGBoostRegressorScratch

try:
    import xgboost

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False


class TestAPI:
    def test_classifier_smoke(self, api_smoke, binary_data):
        X_tr, _, y_tr, _ = binary_data
        api_smoke(
            XGBoostClassifierScratch(n_estimators=10, max_depth=3, random_state=0),
            X_tr,
            y_tr,
            classifier=True,
        )

    def test_regressor_smoke(self, api_smoke, regression_data):
        X_tr, _, y_tr, _ = regression_data
        api_smoke(
            XGBoostRegressorScratch(n_estimators=10, max_depth=3, random_state=0),
            X_tr,
            y_tr,
            classifier=False,
        )

    def test_determinism_with_subsampling(self, binary_data):
        X_tr, X_te, y_tr, _ = binary_data
        kw = dict(n_estimators=8, max_depth=3, subsample=0.7, colsample_bytree=0.7, colsample_bynode=0.8)
        a = XGBoostClassifierScratch(random_state=1, **kw).fit(X_tr, y_tr)
        b = XGBoostClassifierScratch(random_state=1, **kw).fit(X_tr, y_tr)
        c = XGBoostClassifierScratch(random_state=2, **kw).fit(X_tr, y_tr)
        np.testing.assert_array_equal(a.predict_proba(X_te), b.predict_proba(X_te))
        assert not np.array_equal(a.predict_proba(X_te), c.predict_proba(X_te))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_estimators": 0},
            {"max_depth": 0},
            {"reg_lambda": -1},
            {"subsample": 0},
            {"early_stopping_rounds": 5},
        ],
    )
    def test_invalid_options(self, kwargs, regression_data):
        X_tr, _, y_tr, _ = regression_data
        with pytest.raises(ValueError):
            XGBoostRegressorScratch(**kwargs).fit(X_tr, y_tr)

    def test_infinite_values_rejected(self):
        X = np.array([[1.0], [np.inf], [3.0], [4.0]])
        with pytest.raises(ValueError, match="infinit"):
            XGBoostRegressorScratch(n_estimators=1).fit(X, np.array([1.0, 2.0, 3.0, 4.0]))


class TestHandCheckedNumerics:
    def test_regression_gain_and_leaf_weights(self):
        # squared error, F0 = 0: g = -y, h = 1.  Split {1,2} | {10,11}:
        # G_L=-3, H_L=2, G_R=-21, H_R=2, G=-24, H=4, lambda=1
        # gain = 0.5 * (9/3 + 441/3 - 576/5) = 17.4 ; w_L = 3/3 = 1, w_R = 21/3 = 7
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([1.0, 2.0, 10.0, 11.0])
        model = XGBoostRegressorScratch(
            n_estimators=1,
            learning_rate=1.0,
            max_depth=1,
            reg_lambda=1.0,
            gamma=0.0,
            min_child_weight=0.0,
            base_score=0.0,
        ).fit(X, y)
        root = model.estimators_[0][0].root
        assert root.threshold == pytest.approx(2.5)
        assert root.gain == pytest.approx(17.4)
        assert root.left.weight == pytest.approx(1.0)
        assert root.right.weight == pytest.approx(7.0)
        np.testing.assert_allclose(model.predict(X), [1.0, 1.0, 7.0, 7.0])
        assert root.sum_grad == pytest.approx(-24.0)
        assert root.sum_hess == pytest.approx(4.0)

    def test_logistic_gain_and_leaf_weights(self):
        # base_score 0.5 -> p = 0.5, g = p - y, h = 0.25.  y = [0,0,1,1]:
        # G_L = 1, H_L = 0.5, G_R = -1, H_R = 0.5, G = 0
        # gain = 0.5 * (1/1.5 + 1/1.5 - 0) = 2/3 ; w = -G/(H+lambda) = -/+ 2/3 ; times eta = 0.3
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 1, 1])
        model = XGBoostClassifierScratch(
            n_estimators=1,
            learning_rate=0.3,
            max_depth=1,
            reg_lambda=1.0,
            min_child_weight=0.0,
            base_score=0.5,
        ).fit(X, y)
        root = model.estimators_[0][0].root
        assert root.gain == pytest.approx(2.0 / 3.0)
        assert root.left.weight == pytest.approx(-0.2)
        assert root.right.weight == pytest.approx(0.2)
        np.testing.assert_allclose(model.decision_function(X), [-0.2, -0.2, 0.2, 0.2])

    def test_gamma_blocks_low_gain_splits(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([1.0, 2.0, 10.0, 11.0])
        blocked = XGBoostRegressorScratch(
            n_estimators=1, max_depth=2, gamma=20.0, min_child_weight=0.0, base_score=0.0
        )
        blocked.fit(X, y)
        assert blocked.estimators_[0][0].root.is_leaf
        # a root leaf still carries its optimal weight: eta * -G/(H+lambda) = 0.3 * 24/5
        np.testing.assert_allclose(blocked.predict(X), 0.3 * 24.0 / 5.0)
        allowed = XGBoostRegressorScratch(
            n_estimators=1, max_depth=2, gamma=17.0, min_child_weight=0.0, base_score=0.0
        )
        allowed.fit(X, y)
        assert not allowed.estimators_[0][0].root.is_leaf

    def test_min_child_weight_blocks_splits(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([1.0, 2.0, 10.0, 11.0])
        model = XGBoostRegressorScratch(
            n_estimators=1, max_depth=2, min_child_weight=3.0, base_score=0.0
        ).fit(X, y)
        assert model.estimators_[0][0].root.is_leaf

    def test_importance_types(self, binary_data):
        X_tr, _, y_tr, _ = binary_data
        model = XGBoostClassifierScratch(n_estimators=5, max_depth=3, random_state=0).fit(X_tr, y_tr)
        gain = model.get_feature_importance("gain")
        weight = model.get_feature_importance("weight")
        cover = model.get_feature_importance("cover")
        np.testing.assert_allclose(model.feature_importances_, gain / gain.sum())
        assert np.all(weight == np.round(weight)) and weight.sum() > 0
        assert np.all(cover >= 0)
        with pytest.raises(ValueError):
            model.get_feature_importance("shap")


class TestMissingValues:
    def test_nan_features_are_handled(self, binary_data):
        X_tr, X_te, y_tr, y_te = binary_data
        rng = np.random.RandomState(0)
        X_tr = X_tr.copy()
        X_te = X_te.copy()
        X_tr[rng.rand(*X_tr.shape) < 0.15] = np.nan
        X_te[rng.rand(*X_te.shape) < 0.15] = np.nan
        model = XGBoostClassifierScratch(n_estimators=20, max_depth=3, random_state=0).fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        assert np.isfinite(proba).all()
        assert accuracy_score(y_te, model.predict(X_te)) > 0.75

    def test_learned_default_direction(self):
        # Missing values only ever occur on class-1 rows, so the learned default direction
        # must route NaN to the class-1 side at prediction time.
        rng = np.random.RandomState(0)
        n = 200
        x = rng.rand(n)
        y = (x > 0.5).astype(int)
        X = x.reshape(-1, 1).copy()
        X[(y == 1) & (rng.rand(n) < 0.5), 0] = np.nan
        model = XGBoostClassifierScratch(n_estimators=5, max_depth=1, random_state=0).fit(X, y)
        root = model.estimators_[0][0].root
        assert not root.is_leaf
        assert model.predict(np.array([[np.nan]]))[0] == 1

    def test_all_nan_column_is_ignored(self, regression_data):
        X_tr, X_te, y_tr, y_te = regression_data
        X_tr = np.column_stack([X_tr, np.full(len(y_tr), np.nan)])
        X_te = np.column_stack([X_te, np.full(len(y_te), np.nan)])
        model = XGBoostRegressorScratch(n_estimators=15, max_depth=3, random_state=0).fit(X_tr, y_tr)
        assert model.get_feature_importance("weight")[-1] == 0
        assert r2_score(y_te, model.predict(X_te)) > 0.5


class TestEarlyStopping:
    def test_stops_before_n_estimators(self, binary_data):
        X_tr, X_te, y_tr, y_te = binary_data
        model = XGBoostClassifierScratch(
            n_estimators=400, learning_rate=0.5, max_depth=4, early_stopping_rounds=5, random_state=0
        ).fit(X_tr, y_tr, eval_set=(X_te, y_te))
        val = model.evals_result_["validation"]["logloss"]
        assert model.n_estimators_ < 400
        assert len(model.estimators_) == model.n_estimators_ == model.best_iteration_ + 1
        assert model.best_iteration_ == int(np.argmin(val))
        assert model.best_score_ == pytest.approx(min(val))
        assert len(val) == model.best_iteration_ + 1 + 5
        assert len(model.evals_result_["train"]["logloss"]) == len(val)
        # predictions use the best iteration
        np.testing.assert_allclose(log_loss(y_te, model.predict_proba(X_te)), model.best_score_, rtol=1e-6)

    def test_eval_set_without_early_stopping_records_metrics(self, regression_data):
        X_tr, X_te, y_tr, y_te = regression_data
        model = XGBoostRegressorScratch(n_estimators=12, max_depth=3, random_state=0).fit(
            X_tr, y_tr, eval_set=(X_te, y_te)
        )
        assert len(model.evals_result_["validation"]["rmse"]) == 12
        assert model.n_estimators_ == 12
        assert not hasattr(model, "best_iteration_")
        stages = list(model.staged_predict(X_te))
        rmse_last = np.sqrt(np.mean((stages[-1] - y_te) ** 2))
        assert rmse_last == pytest.approx(model.evals_result_["validation"]["rmse"][-1])


class TestAgainstReferenceImplementations:
    def test_multiclass_matches_sklearn_gb_accuracy(self, multiclass_data):
        X_tr, X_te, y_tr, y_te = multiclass_data
        ours = XGBoostClassifierScratch(n_estimators=30, max_depth=3, random_state=0).fit(X_tr, y_tr)
        ref = GradientBoostingClassifier(n_estimators=30, random_state=0).fit(X_tr, y_tr)
        proba = ours.predict_proba(X_te)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        assert len(ours.estimators_[0]) == 3
        assert accuracy_score(y_te, ours.predict(X_te)) >= accuracy_score(y_te, ref.predict(X_te)) - 0.05

    def test_regressor_fits_diabetes(self, diabetes_data):
        X_tr, X_te, y_tr, y_te = diabetes_data
        model = XGBoostRegressorScratch(n_estimators=60, learning_rate=0.1, max_depth=3, random_state=0).fit(
            X_tr, y_tr
        )
        assert r2_score(y_te, model.predict(X_te)) > 0.3

    @pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost is not installed")
    def test_matches_real_xgboost_classifier(self, cancer_data):
        X_tr, X_te, y_tr, y_te = cancer_data
        params = dict(
            n_estimators=30, max_depth=3, learning_rate=0.3, reg_lambda=1.0, gamma=0.0, min_child_weight=1.0
        )
        ours = XGBoostClassifierScratch(base_score=0.5, **params).fit(X_tr, y_tr)
        ref = xgboost.XGBClassifier(
            tree_method="exact", base_score=0.5, subsample=1.0, colsample_bytree=1.0, **params
        )
        ref.fit(X_tr, y_tr)
        ll_ours = log_loss(y_te, ours.predict_proba(X_te))
        ll_ref = log_loss(y_te, ref.predict_proba(X_te))
        assert abs(ll_ours - ll_ref) <= 0.2 * ll_ref
        assert abs(accuracy_score(y_te, ours.predict(X_te)) - accuracy_score(y_te, ref.predict(X_te))) <= 0.03

    @pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost is not installed")
    def test_first_tree_margins_match_real_xgboost(self, cancer_data):
        X_tr, X_te, y_tr, _ = cancer_data
        ours = XGBoostClassifierScratch(n_estimators=1, max_depth=2, base_score=0.5).fit(X_tr, y_tr)
        ref = xgboost.XGBClassifier(n_estimators=1, max_depth=2, tree_method="exact", base_score=0.5).fit(
            X_tr, y_tr
        )
        np.testing.assert_allclose(
            ours.decision_function(X_te), ref.predict(X_te, output_margin=True), atol=1e-5
        )

    @pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost is not installed")
    def test_matches_real_xgboost_regressor(self, diabetes_data):
        X_tr, X_te, y_tr, y_te = diabetes_data
        params = dict(n_estimators=30, max_depth=3, learning_rate=0.3, reg_lambda=1.0)
        ours = XGBoostRegressorScratch(base_score=float(y_tr.mean()), **params).fit(X_tr, y_tr)
        ref = xgboost.XGBRegressor(tree_method="exact", base_score=float(y_tr.mean()), **params).fit(
            X_tr, y_tr
        )
        assert abs(r2_score(y_te, ours.predict(X_te)) - r2_score(y_te, ref.predict(X_te))) < 0.05
