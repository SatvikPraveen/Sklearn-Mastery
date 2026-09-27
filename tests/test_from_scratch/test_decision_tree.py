"""Tests for the from-scratch CART trees."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn_mastery.from_scratch.decision_tree import (
    DecisionTreeClassifierScratch,
    DecisionTreeRegressorScratch,
    _entropy,
    _gini,
    _resolve_max_features,
)


class TestAPI:
    def test_classifier_smoke(self, api_smoke, binary_data):
        X_tr, _, y_tr, _ = binary_data
        api_smoke(DecisionTreeClassifierScratch(max_depth=4), X_tr, y_tr, classifier=True)

    def test_regressor_smoke(self, api_smoke, regression_data):
        X_tr, _, y_tr, _ = regression_data
        api_smoke(DecisionTreeRegressorScratch(max_depth=4), X_tr, y_tr, classifier=False)

    def test_string_labels(self):
        X = np.arange(8, dtype=float).reshape(-1, 1)
        y = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
        clf = DecisionTreeClassifierScratch().fit(X, y)
        assert list(clf.classes_) == ["a", "b"]
        assert list(clf.predict(X)) == list(y)

    @pytest.mark.parametrize(
        "bad_kwargs",
        [
            {"criterion": "nope"},
            {"max_depth": 0},
            {"min_samples_split": 1},
            {"min_samples_leaf": 0},
            {"ccp_alpha": -1},
        ],
    )
    def test_invalid_hyperparameters(self, bad_kwargs, binary_data):
        X_tr, _, y_tr, _ = binary_data
        with pytest.raises(ValueError):
            DecisionTreeClassifierScratch(**bad_kwargs).fit(X_tr, y_tr)

    def test_determinism_with_feature_subsampling(self, binary_data):
        X_tr, X_te, y_tr, _ = binary_data
        a = DecisionTreeClassifierScratch(max_features="sqrt", random_state=7).fit(X_tr, y_tr)
        b = DecisionTreeClassifierScratch(max_features="sqrt", random_state=7).fit(X_tr, y_tr)
        np.testing.assert_array_equal(a.predict_proba(X_te), b.predict_proba(X_te))
        assert a.export_text() == b.export_text()


class TestHandCheckedNumerics:
    def test_gini_and_entropy_functions(self):
        assert _gini(np.array([2.0, 2.0])) == pytest.approx(0.5)
        assert _gini(np.array([4.0, 0.0])) == pytest.approx(0.0)
        assert _entropy(np.array([2.0, 2.0])) == pytest.approx(1.0)
        assert _entropy(np.array([3.0, 1.0])) == pytest.approx(0.8112781)

    def test_gini_of_known_split(self):
        # y = [0,0,0,1,1,1]; x separates {0,0,0,1} | {1,1}
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        tree = DecisionTreeClassifierScratch(max_depth=1).fit(X, y)
        root = tree.tree_
        assert root.impurity == pytest.approx(0.5)
        # the optimal single split is between 3 and 4 (pure children)
        assert root.threshold == pytest.approx(3.5)
        assert root.left.impurity == pytest.approx(0.0)
        assert root.right.impurity == pytest.approx(0.0)
        np.testing.assert_allclose(root.left.value, [1.0, 0.0])
        # a forced impure split: min_samples_leaf=2 with x separating {0,0,0,1} | {1,1}
        y2 = np.array([0, 0, 0, 1, 1, 1])
        X2 = np.array([[1.0], [1.0], [1.0], [1.0], [2.0], [2.0]])
        tree2 = DecisionTreeClassifierScratch(max_depth=1).fit(X2, y2)
        left = tree2.tree_.left
        assert left.n_samples == 4
        assert left.impurity == pytest.approx(1 - (0.75**2 + 0.25**2))  # 0.375
        # impurity decrease = 0.5 - (4/6 * 0.375 + 2/6 * 0) = 0.25 -> importance normalises to 1
        np.testing.assert_allclose(tree2.feature_importances_, [1.0])

    def test_entropy_criterion_selects_same_pure_split(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 1, 1])
        tree = DecisionTreeClassifierScratch(criterion="entropy").fit(X, y)
        assert tree.tree_.impurity == pytest.approx(1.0)
        assert tree.tree_.threshold == pytest.approx(2.5)
        assert tree.get_n_leaves() == 2

    def test_regression_root_impurity_is_variance_and_leaf_is_mean(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([1.0, 2.0, 10.0, 11.0])
        tree = DecisionTreeRegressorScratch(max_depth=1).fit(X, y)
        assert tree.tree_.impurity == pytest.approx(np.var(y))
        assert tree.tree_.threshold == pytest.approx(2.5)
        np.testing.assert_allclose(tree.predict(X), [1.5, 1.5, 10.5, 10.5])

    def test_friedman_criterion_agrees_on_simple_split(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        y = np.array([1.0, 1.5, 1.2, 8.0, 8.5, 8.2])
        a = DecisionTreeRegressorScratch(criterion="friedman_mse", max_depth=1).fit(X, y)
        b = DecisionTreeRegressorScratch(criterion="squared_error", max_depth=1).fit(X, y)
        assert a.tree_.threshold == b.tree_.threshold == pytest.approx(3.5)

    def test_full_depth_tree_interpolates_distinct_inputs(self):
        rng = np.random.RandomState(0)
        X = rng.rand(30, 2)
        y = rng.rand(30)
        tree = DecisionTreeRegressorScratch().fit(X, y)
        np.testing.assert_allclose(tree.predict(X), y)


class TestSampleWeight:
    def test_zero_weight_equals_removal(self, binary_data):
        X_tr, X_te, y_tr, _ = binary_data
        w = np.ones(len(y_tr))
        w[::3] = 0.0
        weighted = DecisionTreeClassifierScratch(max_depth=5).fit(X_tr, y_tr, sample_weight=w)
        # zero-weight rows still count for min_samples_*; compare predictions with the tree fitted
        # on the non-zero rows, which has identical impurities and therefore identical splits
        removed = DecisionTreeClassifierScratch(max_depth=5).fit(X_tr[w > 0], y_tr[w > 0])
        np.testing.assert_allclose(weighted.predict_proba(X_te), removed.predict_proba(X_te))

    def test_integer_weights_equal_duplication(self):
        rng = np.random.RandomState(1)
        X = rng.rand(40, 3)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        w = rng.randint(1, 4, size=40)
        weighted = DecisionTreeClassifierScratch(max_depth=3).fit(X, y, sample_weight=w.astype(float))
        duplicated = DecisionTreeClassifierScratch(max_depth=3).fit(np.repeat(X, w, axis=0), np.repeat(y, w))
        np.testing.assert_allclose(weighted.predict_proba(X), duplicated.predict_proba(X))


class TestStructureAndOptions:
    @pytest.mark.parametrize(
        "option, expected",
        [(None, 10), ("sqrt", 3), ("log2", 3), (4, 4), (0.5, 5), (1.0, 10)],
    )
    def test_max_features_resolution(self, option, expected):
        assert _resolve_max_features(option, 10) == expected

    @pytest.mark.parametrize("option", ["auto", 0, 11, 0.0, 1.5])
    def test_max_features_invalid(self, option):
        with pytest.raises((ValueError, TypeError)):
            _resolve_max_features(option, 10)

    def test_depth_leaf_constraints(self, binary_data):
        X_tr, _, y_tr, _ = binary_data
        tree = DecisionTreeClassifierScratch(max_depth=3, min_samples_leaf=7).fit(X_tr, y_tr)
        assert tree.get_depth() <= 3
        leaves = [n for n in tree._nodes if n.is_leaf]
        assert all(n.n_samples >= 7 for n in leaves)
        assert tree.get_n_leaves() == len(leaves)
        assert tree.node_count_ == len(tree._nodes)

    def test_apply_returns_leaf_ids(self, binary_data):
        X_tr, X_te, y_tr, _ = binary_data
        tree = DecisionTreeClassifierScratch(max_depth=4).fit(X_tr, y_tr)
        leaf_ids = tree.apply(X_te)
        leaf_set = {n.node_id for n in tree._nodes if n.is_leaf}
        assert set(leaf_ids) <= leaf_set
        # samples in the same leaf get the same probabilities
        proba = tree.predict_proba(X_te)
        for leaf in np.unique(leaf_ids):
            assert np.allclose(proba[leaf_ids == leaf], proba[leaf_ids == leaf][0])

    def test_export_text(self, binary_data):
        X_tr, _, y_tr, _ = binary_data
        tree = DecisionTreeClassifierScratch(max_depth=2).fit(X_tr, y_tr)
        text = tree.export_text(feature_names=[f"f{i}" for i in range(X_tr.shape[1])])
        assert "<=" in text and ">" in text and "class:" in text
        assert text.count("class:") == tree.get_n_leaves()
        reg = DecisionTreeRegressorScratch(max_depth=1).fit(X_tr, y_tr.astype(float))
        assert "value:" in reg.export_text()

    def test_pruning_reduces_leaves_monotonically(self, cancer_data):
        X_tr, _, y_tr, _ = cancer_data
        leaves = [
            DecisionTreeClassifierScratch(ccp_alpha=a).fit(X_tr, y_tr).get_n_leaves()
            for a in (0.0, 0.001, 0.005, 0.02, 0.1, 1.0)
        ]
        assert leaves == sorted(leaves, reverse=True)
        assert leaves[0] > leaves[-2]
        assert leaves[-1] == 1  # everything pruned back to the root

    @pytest.mark.parametrize("alpha", [0.002, 0.01, 0.05])
    def test_pruning_matches_sklearn_leaf_count(self, cancer_data, alpha):
        X_tr, _, y_tr, _ = cancer_data
        ours = DecisionTreeClassifierScratch(ccp_alpha=alpha).fit(X_tr, y_tr)
        ref = DecisionTreeClassifier(ccp_alpha=alpha, random_state=0).fit(X_tr, y_tr)
        assert ours.get_n_leaves() == ref.get_n_leaves()

    def test_effective_alpha_hand_check(self):
        # Root with impurity 0.5 (4 samples) split into two pure leaves: R(t)=0.5, R(T_t)=0,
        # |T_t|=2 -> alpha_eff = 0.5. ccp_alpha just below keeps the split, at/above prunes it.
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 1, 1])
        assert DecisionTreeClassifierScratch(ccp_alpha=0.49).fit(X, y).get_n_leaves() == 2
        assert DecisionTreeClassifierScratch(ccp_alpha=0.5).fit(X, y).get_n_leaves() == 1


class TestAgainstSklearn:
    def test_classifier_accuracy_close_to_sklearn(self, cancer_data):
        X_tr, X_te, y_tr, y_te = cancer_data
        ours = DecisionTreeClassifierScratch(max_depth=5).fit(X_tr, y_tr)
        ref = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X_tr, y_tr)
        assert accuracy_score(y_te, ours.predict(X_te)) >= accuracy_score(y_te, ref.predict(X_te)) - 0.05

    def test_regressor_r2_close_to_sklearn(self, diabetes_data):
        X_tr, X_te, y_tr, y_te = diabetes_data
        ours = DecisionTreeRegressorScratch(max_depth=4).fit(X_tr, y_tr)
        ref = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X_tr, y_tr)
        assert r2_score(y_te, ours.predict(X_te)) >= r2_score(y_te, ref.predict(X_te)) - 0.05

    def test_exact_agreement_on_shallow_tree(self, diabetes_data):
        # With few candidate ties a depth-2 regression tree should coincide with sklearn's.
        X_tr, X_te, y_tr, _ = diabetes_data
        ours = DecisionTreeRegressorScratch(max_depth=2).fit(X_tr, y_tr)
        ref = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X_tr, y_tr)
        np.testing.assert_allclose(ours.predict(X_te), ref.predict(X_te), rtol=1e-6)
        np.testing.assert_allclose(ours.feature_importances_, ref.feature_importances_, atol=1e-6)
