"""Gradient boosting machines (Friedman, 2001) implemented from first principles.

Gradient boosting builds an additive model $F_M(x) = F_0 + \\eta
\\sum_{m=1}^{M} h_m(x)$ by *functional gradient descent* on a loss
$L(y, F)$. Starting from a constant $F_0 = \\arg\\min_c \\sum_i
L(y_i, c)$, round $m$ computes the negative gradient (the
*pseudo-residuals*)

$$
r_{im} = -\\left.\\frac{\\partial L(y_i, F)}{\\partial F}\\right|_{F = F_{m-1}(x_i)},
$$

fits a regression tree $h_m$ to $\\{(x_i, r_{im})\\}$ with
terminal regions $R_{jm}$, and then replaces each leaf's value by the
*line-search* optimum

$$
\\gamma_{jm} = \\arg\\min_\\gamma \\sum_{x_i \\in R_{jm}} L\\big(y_i, F_{m-1}(x_i) + \\gamma\\big),
$$

before updating $F_m(x) = F_{m-1}(x) + \\eta \\sum_j \\gamma_{jm}
\\mathbb{1}[x \\in R_{jm}]$. The learning rate $\\eta$ (shrinkage) and,
optionally, fitting each tree on a random *subsample* of the rows
(stochastic gradient boosting, Friedman 2002) regularise the fit.

Regression losses (``GradientBoostingRegressorScratch``)
--------------------------------------------------------
* ``squared_error``: $L = \\tfrac12 (y - F)^2$; $r = y - F$,
  $F_0 = \\bar y$; the leaf mean of the residuals *is* the line-search
  optimum so no extra step is needed.
* ``absolute_error``: $L = |y - F|$; $r = \\operatorname{sign}(y - F)$,
  $F_0 = \\operatorname{median}(y)$; $\\gamma_{jm} =
  \\operatorname{median}_{i \\in R_{jm}}(y_i - F_{m-1}(x_i))$.
* ``huber``: $L = \\tfrac12 (y-F)^2$ for $|y - F| \\le \\delta$
  and $\\delta(|y - F| - \\delta/2)$ otherwise, with $\\delta$ the
  ``alpha``-quantile of the current absolute residuals;
  $r = y - F$ clipped to $[-\\delta, \\delta]$; the leaf update
  is Friedman's one-step approximation
  $\\gamma_{jm} = \\tilde r_{jm} + \\frac{1}{n_{jm}} \\sum_{i \\in R_{jm}}
  \\operatorname{sign}(r_i - \\tilde r_{jm}) \\min(\\delta, |r_i - \\tilde r_{jm}|)$
  where $\\tilde r_{jm}$ is the leaf median of the raw residuals.

Classification (``GradientBoostingClassifierScratch``)
------------------------------------------------------
*Binary* problems model the log-odds $F(x) = \\log \\frac{p}{1-p}$
under the binomial deviance $L = -[y \\log p + (1 - y)\\log(1 - p)]$
with $y \\in \\{0, 1\\}$. Then $r = y - p$, $F_0 =
\\log \\frac{\\bar y}{1 - \\bar y}$, and one Newton step in each leaf gives

$$
\\gamma_{jm} = \\frac{\\sum_{i \\in R_{jm}} (y_i - p_i)}{\\sum_{i \\in R_{jm}} p_i (1 - p_i)}.
$$

*Multiclass* problems fit $K$ trees per round (one per class) to the
residuals $r_{ik} = y_{ik} - p_k(x_i)$ of the multinomial deviance,
where $p_k = \\operatorname{softmax}(F)_k$. Friedman's Algorithm 6
uses the leaf update

$$
\\gamma_{jkm} = \\frac{K - 1}{K}\\;
\\frac{\\sum_{i \\in R_{jkm}} r_{ik}}{\\sum_{i \\in R_{jkm}} |r_{ik}| (1 - |r_{ik}|)},
$$

a diagonal Newton step whose $(K-1)/K$ factor accounts for the
sum-to-zero constraint on the $F_k$. Initial scores are the log
class priors.

References
----------
Friedman, J. H. (2001). Greedy function approximation: a gradient boosting
machine. *Annals of Statistics* 29(5), 1189-1232.

Friedman, J. H. (2002). Stochastic gradient boosting. *Computational
Statistics & Data Analysis* 38(4), 367-378.
"""

from __future__ import annotations

import numbers
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from sklearn_mastery.from_scratch.decision_tree import DecisionTreeRegressorScratch, MaxFeatures

__all__ = ["GradientBoostingClassifierScratch", "GradientBoostingRegressorScratch"]

_MAX_SEED = np.iinfo(np.int32).max


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(F: np.ndarray) -> np.ndarray:
    z = F - F.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class _BaseGradientBoostingScratch(BaseEstimator):
    """Shared boosting loop; subclasses define the loss-specific pieces."""

    _is_classifier: bool = False

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: MaxFeatures = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def _validate_common(self) -> None:
        if not isinstance(self.n_estimators, numbers.Integral) or self.n_estimators < 1:
            raise ValueError(f"n_estimators must be a positive integer; got {self.n_estimators!r}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive; got {self.learning_rate}")
        if not 0.0 < self.subsample <= 1.0:
            raise ValueError(f"subsample must lie in (0, 1]; got {self.subsample}")

    def _make_tree(self, seed: int) -> DecisionTreeRegressorScratch:
        return DecisionTreeRegressorScratch(
            criterion="friedman_mse",
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=seed,
        )

    def _fit_stage_tree(
        self, X: np.ndarray, residual: np.ndarray, rows: np.ndarray, seed: int, leaf_value_fn
    ) -> DecisionTreeRegressorScratch:
        """Fit one tree to the pseudo-residuals of ``rows`` and apply the leaf line search.

        Args:
            X: Full feature matrix.
            residual: Pseudo-residuals for all samples.
            rows: Indices of the (sub)sampled rows used for this stage.
            seed: Seed for the tree's feature subsampling.
            leaf_value_fn: Callable ``(member_rows) -> gamma`` computing the
                line-search optimum for one leaf from the in-bag rows it
                contains, or ``None`` to keep the leaf mean of residuals.
        """
        tree = self._make_tree(seed).fit(X[rows], residual[rows])
        if leaf_value_fn is not None:
            leaf_ids = tree.apply(X[rows])
            updates: Dict[int, float] = {}
            for leaf in np.unique(leaf_ids):
                updates[int(leaf)] = float(leaf_value_fn(rows[leaf_ids == leaf]))
            tree._set_leaf_values(updates)
        return tree

    def _check_X(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was fitted with {self.n_features_in_}"
            )
        return X

    def _draw_rows(self, n_samples: int, rng: np.random.RandomState) -> np.ndarray:
        if self.subsample >= 1.0:
            return np.arange(n_samples)
        n_draw = max(1, int(self.subsample * n_samples))
        return np.sort(rng.choice(n_samples, n_draw, replace=False))


class GradientBoostingRegressorScratch(RegressorMixin, _BaseGradientBoostingScratch):
    """Gradient boosting regressor built from scratch (see module docstring).

    Args:
        loss: ``"squared_error"``, ``"absolute_error"`` or ``"huber"``.
        learning_rate: Shrinkage $\\eta$.
        n_estimators: Number of boosting rounds ``M``.
        subsample: Fraction of rows drawn (without replacement) for each tree;
            ``< 1`` gives stochastic gradient boosting.
        max_depth: Depth of each regression tree.
        min_samples_split: Passed to the trees.
        min_samples_leaf: Passed to the trees.
        max_features: Passed to the trees.
        alpha: Quantile used for the Huber $\\delta$ (ignored otherwise).
        random_state: Seed for subsampling and feature subsampling.

    Attributes:
        init_: The constant $F_0$.
        estimators_: The fitted trees, one per round.
        train_score_: Training loss after each round (on the in-bag rows when
            ``subsample < 1``, as in scikit-learn).
        n_features_in_: Number of features seen during fit.
        feature_importances_: Mean MDI of the trees, normalised to one.
    """

    def __init__(
        self,
        loss: str = "squared_error",
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: MaxFeatures = None,
        alpha: float = 0.9,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
        )
        self.loss = loss
        self.alpha = alpha

    # ------------------------------------------------------------- losses
    def _loss_value(self, y: np.ndarray, F: np.ndarray, delta: float = np.nan) -> float:
        diff = y - F
        if self.loss == "squared_error":
            return float(np.mean(diff**2) / 2.0)
        if self.loss == "absolute_error":
            return float(np.mean(np.abs(diff)))
        absdiff = np.abs(diff)
        quad = 0.5 * diff**2
        lin = delta * (absdiff - 0.5 * delta)
        return float(np.mean(np.where(absdiff <= delta, quad, lin)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> GradientBoostingRegressorScratch:
        """Fit ``n_estimators`` stages of functional gradient descent.

        Args:
            X: Feature matrix ``(n_samples, n_features)``.
            y: Continuous targets ``(n_samples,)``.

        Returns:
            The fitted model.
        """
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        self._validate_common()
        if self.loss not in ("squared_error", "absolute_error", "huber"):
            raise ValueError(f"loss must be 'squared_error', 'absolute_error' or 'huber'; got {self.loss!r}")
        if self.loss == "huber" and not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1); got {self.alpha}")
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]
        rng = check_random_state(self.random_state)

        self.init_ = float(np.mean(y) if self.loss == "squared_error" else np.median(y))
        F = np.full(n_samples, self.init_)
        self.estimators_: List[DecisionTreeRegressorScratch] = []
        self.train_score_ = np.empty(self.n_estimators)

        for m in range(self.n_estimators):
            rows = self._draw_rows(n_samples, rng)
            diff = y - F
            delta = np.nan
            if self.loss == "squared_error":
                residual = diff
                leaf_value_fn = None  # leaf mean of residuals is already optimal
            elif self.loss == "absolute_error":
                residual = np.sign(diff)

                def leaf_value_fn(idx, diff=diff):
                    return np.median(diff[idx])

            else:  # huber
                delta = float(np.percentile(np.abs(diff), self.alpha * 100.0))
                residual = np.clip(diff, -delta, delta)

                def leaf_value_fn(idx, diff=diff, delta=delta):
                    med = np.median(diff[idx])
                    dev = diff[idx] - med
                    return med + np.mean(np.sign(dev) * np.minimum(delta, np.abs(dev)))

            tree = self._fit_stage_tree(X, residual, rows, int(rng.randint(_MAX_SEED)), leaf_value_fn)
            F = F + self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)
            self.train_score_[m] = self._loss_value(y[rows], F[rows], delta)

        self.feature_importances_ = self._mean_importances()
        return self

    def _mean_importances(self) -> np.ndarray:
        importances = np.mean([t.feature_importances_ for t in self.estimators_], axis=0)
        total = importances.sum()
        return importances / total if total > 0 else importances

    def staged_predict(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield $F_m(X)$ after each boosting round."""
        X = self._check_X(X)
        F = np.full(X.shape[0], self.init_)
        for tree in self.estimators_:
            F = F + self.learning_rate * tree.predict(X)
            yield F

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return $F_M(X)$."""
        prediction = None
        for prediction in self.staged_predict(X):
            pass
        return prediction  # type: ignore[return-value]


class GradientBoostingClassifierScratch(ClassifierMixin, _BaseGradientBoostingScratch):
    """Gradient boosting classifier built from scratch (see module docstring).

    Binary problems boost the log-odds with the binomial deviance; problems
    with ``K > 2`` classes fit ``K`` trees per round under the multinomial
    deviance.

    Args:
        learning_rate: Shrinkage $\\eta$.
        n_estimators: Number of boosting rounds ``M``.
        subsample: Row fraction per round (stochastic gradient boosting).
        max_depth: Depth of each regression tree.
        min_samples_split: Passed to the trees.
        min_samples_leaf: Passed to the trees.
        max_features: Passed to the trees.
        random_state: Seed for subsampling and feature subsampling.

    Attributes:
        classes_: Sorted unique class labels.
        n_classes_: Number of classes ``K``.
        init_: Initial raw scores ``(n_trees_per_round,)``.
        estimators_: List of length ``n_estimators``; each entry is a list of
            ``1`` (binary) or ``K`` trees.
        train_score_: Training deviance after each round.
        n_features_in_: Number of features seen during fit.
        feature_importances_: Mean MDI over all trees, normalised to one.
    """

    _is_classifier = True

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: MaxFeatures = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
        )

    # --------------------------------------------------------- raw scores
    def _proba_from_raw(self, F: np.ndarray) -> np.ndarray:
        if self.n_classes_ == 2:
            p = _sigmoid(F[:, 0])
            return np.column_stack([1.0 - p, p])
        return _softmax(F)

    def _deviance(self, y_onehot: np.ndarray, F: np.ndarray) -> float:
        proba = np.clip(self._proba_from_raw(F), 1e-15, 1.0)
        return float(-np.mean(np.sum(y_onehot * np.log(proba), axis=1)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> GradientBoostingClassifierScratch:
        """Fit the boosted classifier.

        Args:
            X: Feature matrix ``(n_samples, n_features)``.
            y: Class labels ``(n_samples,)``.

        Returns:
            The fitted model.
        """
        X, y = check_X_y(X, y, dtype=np.float64)
        self._validate_common()
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.n_classes_ = K = len(self.classes_)
        if K < 2:
            raise ValueError("Need at least two classes")
        rng = check_random_state(self.random_state)

        y_onehot = np.zeros((n_samples, K))
        y_onehot[np.arange(n_samples), y_enc] = 1.0
        prior = np.clip(y_onehot.mean(axis=0), 1e-12, 1.0)
        n_trees = 1 if K == 2 else K
        if K == 2:
            self.init_ = np.array([np.log(prior[1] / prior[0])])
        else:
            self.init_ = np.log(prior)
        F = np.tile(self.init_, (n_samples, 1))

        self.estimators_: List[List[DecisionTreeRegressorScratch]] = []
        self.train_score_ = np.empty(self.n_estimators)
        for m in range(self.n_estimators):
            rows = self._draw_rows(n_samples, rng)
            proba = self._proba_from_raw(F)
            stage: List[DecisionTreeRegressorScratch] = []
            for k in range(n_trees):
                target_col = 1 if K == 2 else k
                residual = y_onehot[:, target_col] - proba[:, target_col]
                if K == 2:

                    def leaf_value_fn(idx, r=residual, p=proba[:, 1]):
                        denom = np.sum(p[idx] * (1.0 - p[idx]))
                        return np.sum(r[idx]) / denom if denom > 0 else 0.0

                else:

                    def leaf_value_fn(idx, r=residual):
                        a = np.abs(r[idx])
                        denom = np.sum(a * (1.0 - a))
                        return (K - 1.0) / K * np.sum(r[idx]) / denom if denom > 0 else 0.0

                tree = self._fit_stage_tree(X, residual, rows, int(rng.randint(_MAX_SEED)), leaf_value_fn)
                stage.append(tree)
            # All K trees of a round are fitted against the *same* F_{m-1}
            # (Friedman's Algorithm 6); F is updated only after the round.
            for k, tree in enumerate(stage):
                F[:, k] += self.learning_rate * tree.predict(X)
            self.estimators_.append(stage)
            self.train_score_[m] = self._deviance(y_onehot[rows], F[rows])

        all_trees = [t for stage in self.estimators_ for t in stage]
        importances = np.mean([t.feature_importances_ for t in all_trees], axis=0)
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
        return self

    def staged_decision_function(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield raw scores after each round: ``(n_samples,)`` for binary, else ``(n_samples, K)``."""
        X = self._check_X(X)
        F = np.tile(self.init_, (X.shape[0], 1))
        for stage in self.estimators_:
            for k, tree in enumerate(stage):
                F[:, k] += self.learning_rate * tree.predict(X)
            yield F[:, 0].copy() if self.n_classes_ == 2 else F.copy()

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Raw additive scores (log-odds for binary problems)."""
        out = None
        for out in self.staged_decision_function(X):
            pass
        return out  # type: ignore[return-value]

    def staged_predict_proba(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield class probabilities after each round."""
        for raw in self.staged_decision_function(X):
            yield self._proba_from_raw(raw.reshape(raw.shape[0], -1))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Sigmoid (binary) or softmax (multiclass) of the raw scores."""
        raw = self.decision_function(X)
        return self._proba_from_raw(raw.reshape(raw.shape[0], -1))

    def staged_predict(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield class predictions after each round."""
        for proba in self.staged_predict_proba(X):
            yield self.classes_[np.argmax(proba, axis=1)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the most probable class."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
