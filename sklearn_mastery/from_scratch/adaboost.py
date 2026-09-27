"""AdaBoost implemented from first principles: SAMME and AdaBoost.R2.

Multiclass classification - SAMME (Hastie, Rosset, Zhu and Zou, 2009)
--------------------------------------------------------------------
Boosting fits an additive model $F(x) = \\sum_m \\alpha_m T_m(x)$ by
forward stagewise minimisation of the multiclass exponential loss. With
$K$ classes and sample weights $w_i$ (initialised to
$1/n$), round $m$ proceeds as

1. fit a weak learner $T_m$ to the weighted data;
2. compute the weighted error
   $\\varepsilon_m = \\sum_i w_i \\mathbb{1}[T_m(x_i) \\ne y_i] / \\sum_i w_i$;
3. set the learner weight

   $$
   \\alpha_m = \\eta \\left[ \\log\\frac{1 - \\varepsilon_m}{\\varepsilon_m}
               + \\log(K - 1) \\right];
   $$

4. re-weight, $w_i \\leftarrow w_i \\exp(\\alpha_m \\mathbb{1}[T_m(x_i) \\ne y_i])$,
   and renormalise so the weights sum to one.

The $\\log(K-1)$ term is what makes SAMME a proper multiclass
generalisation: $\\alpha_m$ is positive as long as
$\\varepsilon_m < 1 - 1/K$, i.e. the weak learner beats random guessing
among $K$ classes (for $K = 2$ it reduces to Freund and
Schapire's original AdaBoost.M1). Boosting stops early when
$\\varepsilon_m = 0$ (the learner is perfect, no further reweighting is
possible) or $\\varepsilon_m \\ge 1 - 1/K$ (the learner is no better
than chance, $\\alpha_m \\le 0$).

Prediction is the weighted vote $\\arg\\max_k \\sum_m \\alpha_m
\\mathbb{1}[T_m(x) = k]$. Class probabilities follow Zhu et al.'s
population minimiser, $p_k(x) \\propto \\exp\\!\\big(\\frac{1}{K-1} f_k(x)\\big)$
with $f_k$ the normalised vote for class $k$.

Regression - AdaBoost.R2 (Drucker, 1997)
-----------------------------------------
Each round draws a bootstrap sample according to the current weights, fits
the learner, and computes the relative error of every training point,
$e_i = |T_m(x_i) - y_i| / \\max_j |T_m(x_j) - y_j| \\in [0, 1]$, passed
through a loss $L(e)$ (``linear``: $e$; ``square``: $e^2$;
``exponential``: $1 - \\exp(-e)$). With the weighted average loss
$\\bar L = \\sum_i w_i L_i$ the round is kept only if
$\\bar L < 1/2$, and

$$
\\beta_m = \\frac{\\bar L}{1 - \\bar L}, \\qquad
w_i \\leftarrow w_i\\, \\beta_m^{\\,\\eta (1 - L_i)}, \\qquad
\\alpha_m = \\eta \\log(1/\\beta_m).
$$

Small-error points are down-weighted (their exponent $1 - L_i$ is
large and $\\beta_m < 1$). The ensemble predicts the *weighted median*
of the members' predictions: sort the ``M`` predictions for a sample, and
return the first one at which the cumulative $\\alpha$ reaches half of
$\\sum_m \\alpha_m$. The median is robust to the occasional wildly wrong
member in a way that a weighted mean is not.

References
----------
Freund, Y. and Schapire, R. (1997). A decision-theoretic generalization of
on-line learning and an application to boosting. *JCSS* 55, 119-139.

Hastie, T., Rosset, S., Zhu, J. and Zou, H. (2009). Multi-class AdaBoost.
*Statistics and Its Interface* 2, 349-360.

Drucker, H. (1997). Improving regressors using boosting techniques. *ICML*.
"""

from __future__ import annotations

import numbers
from typing import Iterator, List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.utils import check_random_state
from sklearn.utils.validation import (
    _check_sample_weight,
    check_array,
    check_is_fitted,
    check_X_y,
    has_fit_parameter,
)

from sklearn_mastery.from_scratch.decision_tree import (
    DecisionTreeClassifierScratch,
    DecisionTreeRegressorScratch,
)

__all__ = ["AdaBoostClassifierScratch", "AdaBoostRegressorScratch"]

_MAX_SEED = np.iinfo(np.int32).max


def _check_n_estimators(n_estimators: int, learning_rate: float) -> None:
    if not isinstance(n_estimators, numbers.Integral) or n_estimators < 1:
        raise ValueError(f"n_estimators must be a positive integer; got {n_estimators!r}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive; got {learning_rate}")


class AdaBoostClassifierScratch(ClassifierMixin, BaseEstimator):
    """SAMME AdaBoost classifier built from scratch (see module docstring).

    Args:
        estimator: Weak learner supporting ``fit(X, y, sample_weight=...)``;
            defaults to a decision stump
            (``DecisionTreeClassifierScratch(max_depth=1)``).
        n_estimators: Maximum number of boosting rounds.
        learning_rate: Shrinkage $\\eta$ applied to every $\\alpha_m$.
        random_state: Seed forwarded to the weak learners (if they accept one).

    Attributes:
        classes_: Sorted unique class labels.
        n_classes_: Number of classes ``K``.
        estimators_: Fitted weak learners (may be fewer than ``n_estimators``
            if boosting stopped early).
        estimator_weights_: $\\alpha_m$ for each kept learner.
        estimator_errors_: Weighted error $\\varepsilon_m$ of each kept learner.
        n_features_in_: Number of features seen during fit.
    """

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> AdaBoostClassifierScratch:
        """Run the SAMME boosting rounds.

        Args:
            X: Feature matrix ``(n_samples, n_features)``.
            y: Class labels ``(n_samples,)``.
            sample_weight: Optional initial weights (normalised to sum to one).

        Returns:
            The fitted ensemble.

        Raises:
            ValueError: If the weak learner does not support ``sample_weight``
                or the very first learner is no better than chance.
        """
        X, y = check_X_y(X, y, dtype=np.float64)
        _check_n_estimators(self.n_estimators, self.learning_rate)
        self.n_features_in_ = X.shape[1]
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.n_classes_ = K = len(self.classes_)
        base = self.estimator if self.estimator is not None else DecisionTreeClassifierScratch(max_depth=1)
        if not has_fit_parameter(base, "sample_weight"):
            raise ValueError("The weak learner must accept sample_weight in fit()")
        rng = check_random_state(self.random_state)

        w = _check_sample_weight(sample_weight, X, dtype=np.float64)
        w = w / w.sum()

        self.estimators_: List[BaseEstimator] = []
        self.estimator_weights_: List[float] = []
        self.estimator_errors_: List[float] = []
        for _m in range(self.n_estimators):
            learner = clone(base)
            if "random_state" in learner.get_params():
                learner.set_params(random_state=int(rng.randint(_MAX_SEED)))
            learner.fit(X, y_enc, sample_weight=w)
            incorrect = learner.predict(X) != y_enc
            error = float(np.dot(w, incorrect) / w.sum())

            if error <= 0.0:  # perfect learner: keep it with unit weight and stop
                self.estimators_.append(learner)
                self.estimator_weights_.append(1.0)
                self.estimator_errors_.append(0.0)
                break
            if error >= 1.0 - 1.0 / K:  # no better than chance: alpha would be <= 0
                if not self.estimators_:
                    raise ValueError(
                        "The first weak learner is no better than random guessing; boosting cannot start"
                    )
                break

            alpha = self.learning_rate * (np.log((1.0 - error) / error) + np.log(K - 1.0))
            self.estimators_.append(learner)
            self.estimator_weights_.append(float(alpha))
            self.estimator_errors_.append(error)

            w = w * np.exp(alpha * incorrect)
            w = w / w.sum()

        self.estimator_weights_ = np.asarray(self.estimator_weights_)  # type: ignore[assignment]
        self.estimator_errors_ = np.asarray(self.estimator_errors_)  # type: ignore[assignment]
        return self

    # ---------------------------------------------------------- inference
    def _check_X(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the ensemble was fitted with {self.n_features_in_}"
            )
        return X

    def _staged_votes(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield the cumulative weighted vote matrix ``(n_samples, K)`` after each round."""
        votes = np.zeros((X.shape[0], self.n_classes_))
        rows = np.arange(X.shape[0])
        for learner, alpha in zip(self.estimators_, self.estimator_weights_):
            votes[rows, np.asarray(learner.predict(X), dtype=np.intp)] += alpha
            yield votes

    def _votes_to_decision(self, votes: np.ndarray) -> np.ndarray:
        """Normalise votes by the total alpha; collapse to 1-D for binary problems."""
        decision = votes / np.sum(self.estimator_weights_)
        if self.n_classes_ == 2:
            return decision[:, 1] - decision[:, 0]
        return decision

    def _decision_to_proba(self, decision: np.ndarray) -> np.ndarray:
        """Zhu et al. (2009) probability map: softmax of ``decision / (K - 1)``."""
        if self.n_classes_ == 2:
            decision = np.column_stack([-decision, decision]) / 2.0
        scaled = decision / (self.n_classes_ - 1)
        scaled -= scaled.max(axis=1, keepdims=True)
        proba = np.exp(scaled)
        return proba / proba.sum(axis=1, keepdims=True)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Normalised weighted votes ``(n_samples, K)``; for two classes a 1-D margin."""
        X = self._check_X(X)
        votes = None
        for votes in self._staged_votes(X):
            pass
        return self._votes_to_decision(votes)  # type: ignore[arg-type]

    def staged_decision_function(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield the decision function after each boosting round."""
        X = self._check_X(X)
        for votes in self._staged_votes(X):
            yield self._votes_to_decision(votes)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities derived from the weighted votes."""
        return self._decision_to_proba(self.decision_function(X))

    def staged_predict_proba(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield class probabilities after each boosting round."""
        for decision in self.staged_decision_function(X):
            yield self._decision_to_proba(decision)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class with the largest weighted vote."""
        X = self._check_X(X)
        votes = None
        for votes in self._staged_votes(X):
            pass
        return self.classes_[np.argmax(votes, axis=1)]

    def staged_predict(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield predictions after each boosting round."""
        X = self._check_X(X)
        for votes in self._staged_votes(X):
            yield self.classes_[np.argmax(votes, axis=1)]


class AdaBoostRegressorScratch(RegressorMixin, BaseEstimator):
    """AdaBoost.R2 regressor built from scratch (see module docstring).

    Args:
        estimator: Weak learner; defaults to
            ``DecisionTreeRegressorScratch(max_depth=3)``.
        n_estimators: Maximum number of boosting rounds.
        learning_rate: Shrinkage $\\eta$.
        loss: ``"linear"``, ``"square"`` or ``"exponential"``.
        random_state: Seed for the weighted bootstrap draws.

    Attributes:
        estimators_: Fitted weak learners.
        estimator_weights_: $\\alpha_m = \\eta \\log(1/\\beta_m)$.
        estimator_errors_: Average loss $\\bar L_m$ of each learner.
        n_features_in_: Number of features seen during fit.
    """

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        loss: str = "linear",
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> AdaBoostRegressorScratch:
        """Run the AdaBoost.R2 boosting rounds.

        Args:
            X: Feature matrix ``(n_samples, n_features)``.
            y: Continuous targets ``(n_samples,)``.
            sample_weight: Optional initial weights (normalised to sum to one).

        Returns:
            The fitted ensemble.
        """
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        _check_n_estimators(self.n_estimators, self.learning_rate)
        if self.loss not in ("linear", "square", "exponential"):
            raise ValueError(f"loss must be 'linear', 'square' or 'exponential'; got {self.loss!r}")
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]
        base = self.estimator if self.estimator is not None else DecisionTreeRegressorScratch(max_depth=3)
        rng = check_random_state(self.random_state)

        w = _check_sample_weight(sample_weight, X, dtype=np.float64)
        w = w / w.sum()

        self.estimators_: List[BaseEstimator] = []
        self.estimator_weights_: List[float] = []
        self.estimator_errors_: List[float] = []
        for _m in range(self.n_estimators):
            # Drucker's algorithm samples the training set according to w.
            draw = rng.choice(n_samples, size=n_samples, replace=True, p=w)
            learner = clone(base)
            if "random_state" in learner.get_params():
                learner.set_params(random_state=int(rng.randint(_MAX_SEED)))
            learner.fit(X[draw], y[draw])

            abs_error = np.abs(learner.predict(X) - y)
            max_error = abs_error.max()
            if max_error > 0:
                abs_error = abs_error / max_error
            if self.loss == "square":
                loss_vec = abs_error**2
            elif self.loss == "exponential":
                loss_vec = 1.0 - np.exp(-abs_error)
            else:
                loss_vec = abs_error
            avg_loss = float(np.dot(w, loss_vec))

            if avg_loss <= 0.0:  # perfect fit: keep with unit weight and stop
                self.estimators_.append(learner)
                self.estimator_weights_.append(1.0)
                self.estimator_errors_.append(0.0)
                break
            if avg_loss >= 0.5:  # worse than the median-error threshold: discard and stop
                if not self.estimators_:
                    raise ValueError("The first weak learner has average loss >= 0.5; boosting cannot start")
                break

            beta = avg_loss / (1.0 - avg_loss)
            self.estimators_.append(learner)
            self.estimator_weights_.append(float(self.learning_rate * np.log(1.0 / beta)))
            self.estimator_errors_.append(avg_loss)

            w = w * np.power(beta, (1.0 - loss_vec) * self.learning_rate)
            w = w / w.sum()

        self.estimator_weights_ = np.asarray(self.estimator_weights_)  # type: ignore[assignment]
        self.estimator_errors_ = np.asarray(self.estimator_errors_)  # type: ignore[assignment]
        return self

    def _check_X(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the ensemble was fitted with {self.n_features_in_}"
            )
        return X

    def _weighted_median(self, predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Weighted median across members for each sample.

        Args:
            predictions: Array ``(n_samples, n_members)``.
            weights: Member weights ``(n_members,)``.
        """
        order = np.argsort(predictions, axis=1)
        cdf = np.cumsum(weights[order], axis=1)
        at_or_above_half = cdf >= 0.5 * cdf[:, -1][:, None]
        median_pos = np.argmax(at_or_above_half, axis=1)
        rows = np.arange(predictions.shape[0])
        return predictions[rows, order[rows, median_pos]]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted median of the members' predictions."""
        X = self._check_X(X)
        predictions = np.column_stack([learner.predict(X) for learner in self.estimators_])
        return self._weighted_median(predictions, np.asarray(self.estimator_weights_))

    def staged_predict(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield the weighted-median prediction using the first ``m`` members."""
        X = self._check_X(X)
        predictions = np.column_stack([learner.predict(X) for learner in self.estimators_])
        weights = np.asarray(self.estimator_weights_)
        for m in range(1, predictions.shape[1] + 1):
            yield self._weighted_median(predictions[:, :m], weights[:m])
