"""Random forests (Breiman, 2001) implemented from first principles.

A random forest is bagging of CART trees with one extra source of
randomness: at *every split* of every tree only a random subset of
``max_features`` features is eligible. Bagging alone leaves the trees highly
correlated because a few dominant features get chosen at the top of almost
every bootstrap tree; the average of ``B`` predictors with variance
$\\sigma^2$ and pairwise correlation $\\rho$ has variance

$$
\\rho\\sigma^2 + \\frac{1 - \\rho}{B}\\sigma^2,
$$

so once $B$ is large the *correlation* term dominates and adding
trees no longer helps. Restricting each split to a random feature subset
forces different trees to use different features, which lowers $\\rho$
at the price of a modest increase in each tree's own variance; the net effect
is a lower ensemble variance (Breiman, 2001; Hastie, Tibshirani and
Friedman, 2009, sec. 15.4). The subset is redrawn at each split, which is
what distinguishes a random forest from bagging with feature bagging
(``BaggingClassifierScratch(max_features=...)``), where a single feature
subset is drawn once *per tree*.

Defaults follow the literature and scikit-learn: $\\sqrt{p}$ features
per split for classification, all $p$ features for regression.

Feature importance
------------------
* ``feature_importances_`` is the mean decrease in impurity (MDI): the
  average over trees of each tree's normalised impurity decrease per feature.
  It is cheap but biased towards high-cardinality features and is computed
  on the training data.
* :meth:`permutation_importance` (Breiman, 2001) measures how much the score
  on a dataset drops when the values of one feature are randomly shuffled,
  breaking its relationship with the target while keeping its marginal
  distribution. It is model-agnostic and can be evaluated on held-out data.

References
----------
Breiman, L. (2001). Random forests. *Machine Learning* 45, 5-32.

Hastie, T., Tibshirani, R. and Friedman, J. (2009). *The Elements of
Statistical Learning*, 2nd ed., chapter 15. Springer.
"""

from __future__ import annotations

import numbers
import warnings
from typing import List, Optional, Union

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import Bunch, check_random_state
from sklearn.utils.validation import _check_sample_weight, check_array, check_is_fitted, check_X_y

from sklearn_mastery.from_scratch.decision_tree import (
    DecisionTreeClassifierScratch,
    DecisionTreeRegressorScratch,
    MaxFeatures,
    _BaseTree,
)

__all__ = ["RandomForestClassifierScratch", "RandomForestRegressorScratch"]

_MAX_SEED = np.iinfo(np.int32).max


class _BaseForestScratch(BaseEstimator):
    """Shared forest logic: bootstrap, parallel tree growth, OOB and importances."""

    _is_classifier: bool = False

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: MaxFeatures = "sqrt",
        bootstrap: bool = True,
        oob_score: bool = False,
        ccp_alpha: float = 0.0,
        n_jobs: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.ccp_alpha = ccp_alpha
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _make_tree(self, seed: int) -> _BaseTree:
        cls = DecisionTreeClassifierScratch if self._is_classifier else DecisionTreeRegressorScratch
        return cls(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=seed,
            ccp_alpha=self.ccp_alpha,
        )

    def _validate_targets(self, y: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> _BaseForestScratch:
        """Grow the forest.

        Each tree is fitted on a bootstrap sample expressed as *weights*: the
        distinct drawn rows are passed with ``sample_weight`` equal to their
        multiplicity (times any user weights), which is equivalent to fitting
        on the duplicated rows but cheaper.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Targets of shape ``(n_samples,)``.
            sample_weight: Optional per-sample weights.

        Returns:
            The fitted forest.
        """
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=not self._is_classifier)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64)
        if not isinstance(self.n_estimators, numbers.Integral) or self.n_estimators < 1:
            raise ValueError(f"n_estimators must be a positive integer; got {self.n_estimators!r}")
        if self.oob_score and not self.bootstrap:
            raise ValueError("oob_score=True requires bootstrap=True")
        n_samples = X.shape[0]
        self.n_features_in_ = X.shape[1]
        y_enc = self._validate_targets(y)
        rng = check_random_state(self.random_state)

        seeds = rng.randint(_MAX_SEED, size=self.n_estimators)
        self.estimators_samples_: List[np.ndarray] = []
        for seed in seeds:
            if self.bootstrap:
                self.estimators_samples_.append(
                    np.random.RandomState(seed).randint(n_samples, size=n_samples)
                )
            else:
                self.estimators_samples_.append(np.arange(n_samples))

        def grow(seed: int, samples: np.ndarray) -> _BaseTree:
            rows, counts = np.unique(samples, return_counts=True)
            return self._make_tree(int(seed)).fit(
                X[rows], y_enc[rows], sample_weight=counts * sample_weight[rows]
            )

        self.estimators_: List[_BaseTree] = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(grow)(seed, samples) for seed, samples in zip(seeds, self.estimators_samples_)
        )
        self.feature_importances_ = self._mean_decrease_impurity()
        if self.oob_score:
            self._compute_oob(X, y_enc)
        return self

    def _mean_decrease_impurity(self) -> np.ndarray:
        importances = np.mean([tree.feature_importances_ for tree in self.estimators_], axis=0)
        total = importances.sum()
        return importances / total if total > 0 else importances

    # ------------------------------------------------------------ helpers
    def _check_X(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the forest was fitted with {self.n_features_in_}"
            )
        return X

    def _tree_output(self, tree: _BaseTree, X: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _aggregate(self, X: np.ndarray) -> np.ndarray:
        return np.mean([self._tree_output(tree, X) for tree in self.estimators_], axis=0)

    def _compute_oob(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples = X.shape[0]
        accumulator: Optional[np.ndarray] = None
        n_votes = np.zeros(n_samples)
        for tree, samples in zip(self.estimators_, self.estimators_samples_):
            oob = np.ones(n_samples, dtype=bool)
            oob[samples] = False
            if not oob.any():
                continue
            out = self._tree_output(tree, X[oob])
            if accumulator is None:
                accumulator = np.zeros((n_samples,) + out.shape[1:])
            accumulator[oob] += out
            n_votes[oob] += 1
        if accumulator is None:
            raise ValueError("No out-of-bag samples; increase n_estimators")
        if (n_votes == 0).any():
            warnings.warn(
                "Some training samples were never out-of-bag; their OOB prediction is NaN. "
                "Increase n_estimators for a reliable OOB estimate.",
                UserWarning,
                stacklevel=3,
            )
        has_vote = n_votes > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            aggregated = accumulator / n_votes.reshape((-1,) + (1,) * (accumulator.ndim - 1))
        self._store_oob(aggregated, y, has_vote)

    def _store_oob(
        self, aggregated: np.ndarray, y: np.ndarray, has_vote: np.ndarray
    ) -> None:  # pragma: no cover
        raise NotImplementedError

    def permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 5,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> Bunch:
        """Permutation feature importance (Breiman, 2001).

        For each feature ``j`` and repeat ``r`` the column ``X[:, j]`` is
        shuffled and the drop in score ``s(X, y) - s(X_perm_j, y)`` is
        recorded, where ``s`` is accuracy for classifiers and $R^2$ for
        regressors. Positive values mean the model relies on the feature;
        values near zero (or negative) mean it does not.

        Args:
            X: Feature matrix, ideally held-out data.
            y: True targets.
            n_repeats: Number of random shuffles per feature.
            random_state: Seed for the shuffles.

        Returns:
            A :class:`sklearn.utils.Bunch` with ``importances_mean`` ``(p,)``,
            ``importances_std`` ``(p,)`` and ``importances`` ``(p, n_repeats)``.
        """
        X = self._check_X(X)
        y = np.asarray(y)
        rng = check_random_state(random_state)
        baseline = self.score(X, y)  # type: ignore[attr-defined]
        importances = np.empty((self.n_features_in_, n_repeats))
        for j in range(self.n_features_in_):
            X_perm = X.copy()
            for r in range(n_repeats):
                X_perm[:, j] = X[rng.permutation(X.shape[0]), j]
                importances[j, r] = baseline - self.score(X_perm, y)  # type: ignore[attr-defined]
        return Bunch(
            importances_mean=importances.mean(axis=1),
            importances_std=importances.std(axis=1),
            importances=importances,
        )


class RandomForestClassifierScratch(ClassifierMixin, _BaseForestScratch):
    """Random forest classifier built from scratch trees.

    Args:
        n_estimators: Number of trees.
        criterion: ``"gini"`` or ``"entropy"``.
        max_depth: Maximum tree depth (``None`` = fully grown).
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples in each child.
        max_features: Features drawn at random *at each split*; default
            ``"sqrt"`` (Breiman's recommendation for classification).
        bootstrap: Fit each tree on a bootstrap resample.
        oob_score: Compute the out-of-bag accuracy.
        ccp_alpha: Cost-complexity pruning parameter passed to each tree.
        n_jobs: Number of joblib threads.
        random_state: Seed controlling bootstraps and feature subsampling.

    Attributes:
        classes_: Sorted unique class labels.
        estimators_: Fitted trees.
        estimators_samples_: Bootstrap indices for each tree.
        feature_importances_: Mean decrease in impurity, normalised to 1.
        oob_decision_function_: OOB class probabilities (when ``oob_score``).
        oob_score_: OOB accuracy (when ``oob_score``).
    """

    _is_classifier = True

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: MaxFeatures = "sqrt",
        bootstrap: bool = True,
        oob_score: bool = False,
        ccp_alpha: float = 0.0,
        n_jobs: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            oob_score=oob_score,
            ccp_alpha=ccp_alpha,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def _validate_targets(self, y: np.ndarray) -> np.ndarray:
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return y_enc.astype(np.intp)

    def _tree_output(self, tree: _BaseTree, X: np.ndarray) -> np.ndarray:
        proba = np.zeros((X.shape[0], self.n_classes_))
        proba[:, np.asarray(tree.classes_, dtype=np.intp)] = tree.predict_proba(X)  # type: ignore[attr-defined]
        return proba

    def _store_oob(self, aggregated: np.ndarray, y: np.ndarray, has_vote: np.ndarray) -> None:
        self.oob_decision_function_ = aggregated
        self.oob_score_ = float(accuracy_score(y[has_vote], np.argmax(aggregated[has_vote], axis=1)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average the leaf class proportions over all trees (soft vote)."""
        return self._aggregate(self._check_X(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class with the highest averaged probability."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class RandomForestRegressorScratch(RegressorMixin, _BaseForestScratch):
    """Random forest regressor built from scratch trees.

    Args:
        n_estimators: Number of trees.
        criterion: ``"squared_error"`` or ``"friedman_mse"``.
        max_depth: Maximum tree depth (``None`` = fully grown).
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples in each child.
        max_features: Features drawn at random at each split; default ``1.0``
            (all features, as in scikit-learn's regressor).
        bootstrap: Fit each tree on a bootstrap resample.
        oob_score: Compute the out-of-bag $R^2$.
        ccp_alpha: Cost-complexity pruning parameter passed to each tree.
        n_jobs: Number of joblib threads.
        random_state: Seed controlling bootstraps and feature subsampling.

    Attributes:
        estimators_: Fitted trees.
        estimators_samples_: Bootstrap indices for each tree.
        feature_importances_: Mean decrease in impurity, normalised to 1.
        oob_prediction_: OOB predictions (when ``oob_score``).
        oob_score_: OOB $R^2$ (when ``oob_score``).
    """

    _is_classifier = False

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "squared_error",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: MaxFeatures = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        ccp_alpha: float = 0.0,
        n_jobs: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            oob_score=oob_score,
            ccp_alpha=ccp_alpha,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def _validate_targets(self, y: np.ndarray) -> np.ndarray:
        return y.astype(np.float64)

    def _tree_output(self, tree: _BaseTree, X: np.ndarray) -> np.ndarray:
        return tree.predict(X)  # type: ignore[attr-defined]

    def _store_oob(self, aggregated: np.ndarray, y: np.ndarray, has_vote: np.ndarray) -> None:
        self.oob_prediction_ = aggregated
        self.oob_score_ = float(r2_score(y[has_vote], aggregated[has_vote]))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Average the trees' predictions."""
        return self._aggregate(self._check_X(X))
