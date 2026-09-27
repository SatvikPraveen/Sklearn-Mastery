"""Bootstrap aggregating (bagging) implemented from first principles.

Bagging (Breiman, 1996) reduces the variance of an unstable learner by
training ``B`` copies of it on bootstrap resamples of the training set and
averaging their predictions:

$$
\\hat f_{\\text{bag}}(x) = \\frac{1}{B} \\sum_{b=1}^{B} \\hat f^{*b}(x),
$$

where $\\hat f^{*b}$ is fitted on a sample $\\mathcal{D}^{*b}$
drawn *with replacement* from $\\mathcal{D}$. If the individual
predictors have variance $\\sigma^2$ and pairwise correlation
$\\rho$, the variance of the average is
$\\rho\\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2$: averaging removes the
independent part of the variance but not the correlated part, which is why
random forests (:mod:`sklearn_mastery.from_scratch.random_forest`) go further
and decorrelate the trees.

For classification the ensemble votes: *soft* voting averages the members'
class-probability estimates, *hard* voting counts predicted labels.

Out-of-bag (OOB) estimation
---------------------------
A bootstrap sample of size ``n`` leaves each observation out with
probability $(1 - 1/n)^n \\to e^{-1} \\approx 0.368$. Every
observation is therefore *out-of-bag* for roughly 37% of the members, and
aggregating only those members' predictions yields an honest estimate of
generalisation error at no extra cost (Breiman, 1996b).

References
----------
Breiman, L. (1996). Bagging predictors. *Machine Learning* 24, 123-140.

Breiman, L. (1996b). Out-of-bag estimation. Technical report, UC Berkeley.
"""

from __future__ import annotations

import numbers
import warnings
from typing import List, Optional, Union

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from sklearn_mastery.from_scratch.decision_tree import (
    DecisionTreeClassifierScratch,
    DecisionTreeRegressorScratch,
)

__all__ = ["BaggingClassifierScratch", "BaggingRegressorScratch"]

_MAX_SEED = np.iinfo(np.int32).max


def _resolve_draw_size(value: Union[int, float], total: int, name: str) -> int:
    """Translate an int / fraction ``max_samples``/``max_features`` into a count."""
    if isinstance(value, numbers.Integral):
        if not 1 <= int(value) <= total:
            raise ValueError(f"{name}={value} must be in [1, {total}]")
        return int(value)
    if isinstance(value, numbers.Real):
        if not 0.0 < float(value) <= 1.0:
            raise ValueError(f"A float {name} must lie in (0, 1]; got {value}")
        return max(1, int(float(value) * total))
    raise TypeError(f"{name} must be an int or a float; got {type(value).__name__}")


class _BaseBaggingScratch(BaseEstimator):
    """Shared bagging logic: resampling, parallel fitting and OOB bookkeeping."""

    _is_classifier: bool = False

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        n_estimators: int = 10,
        max_samples: Union[int, float] = 1.0,
        max_features: Union[int, float] = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        n_jobs: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _default_estimator(self) -> BaseEstimator:  # pragma: no cover - abstract
        raise NotImplementedError

    def _validate_targets(self, y: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> _BaseBaggingScratch:
        """Fit ``n_estimators`` base estimators on bootstrap resamples.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Targets of shape ``(n_samples,)``.

        Returns:
            The fitted ensemble.
        """
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=not self._is_classifier)
        if not isinstance(self.n_estimators, numbers.Integral) or self.n_estimators < 1:
            raise ValueError(f"n_estimators must be a positive integer; got {self.n_estimators!r}")
        if self.oob_score and not self.bootstrap:
            raise ValueError("oob_score=True requires bootstrap=True")
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        y_enc = self._validate_targets(y)
        rng = check_random_state(self.random_state)

        n_draw_samples = _resolve_draw_size(self.max_samples, n_samples, "max_samples")
        n_draw_features = _resolve_draw_size(self.max_features, n_features, "max_features")

        # Draw every resample up front from a single stream so that results
        # do not depend on the number of worker threads.
        seeds = rng.randint(_MAX_SEED, size=self.n_estimators)
        self.estimators_samples_: List[np.ndarray] = []
        self.estimators_features_: List[np.ndarray] = []
        for seed in seeds:
            member_rng = np.random.RandomState(seed)
            self.estimators_features_.append(
                member_rng.choice(n_features, n_draw_features, replace=self.bootstrap_features)
            )
            self.estimators_samples_.append(
                member_rng.choice(n_samples, n_draw_samples, replace=self.bootstrap)
            )

        base = self.estimator if self.estimator is not None else self._default_estimator()
        self.estimator_ = base
        self.estimators_: List[BaseEstimator] = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_fit_member)(base, X, y_enc, samples, features, int(seed))
            for samples, features, seed in zip(self.estimators_samples_, self.estimators_features_, seeds)
        )

        if self.oob_score:
            self._compute_oob(X, y_enc)
        return self

    # ------------------------------------------------------------ helpers
    def _check_X(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the ensemble was fitted with {self.n_features_in_}"
            )
        return X

    def _member_outputs(self, X: np.ndarray, members: Optional[np.ndarray] = None) -> np.ndarray:
        """Stack every member's output on ``X`` (rows) along axis 0."""
        outputs = []
        for k, (est, features) in enumerate(zip(self.estimators_, self.estimators_features_)):
            if members is not None and not members[k]:
                continue
            outputs.append(self._member_output(est, X[:, features]))
        return np.stack(outputs)

    def _member_output(self, est: BaseEstimator, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _compute_oob(self, X: np.ndarray, y: np.ndarray) -> None:
        """Aggregate, for every training row, only the members that did not see it."""
        n_samples = X.shape[0]
        accumulator: Optional[np.ndarray] = None
        n_votes = np.zeros(n_samples)
        for est, samples, features in zip(
            self.estimators_, self.estimators_samples_, self.estimators_features_
        ):
            oob = np.ones(n_samples, dtype=bool)
            oob[samples] = False
            if not oob.any():
                continue
            out = self._member_output(est, X[oob][:, features])
            if accumulator is None:
                accumulator = np.zeros((n_samples,) + out.shape[1:])
            accumulator[oob] += out
            n_votes[oob] += 1
        if accumulator is None or not n_votes.any():
            raise ValueError("No out-of-bag samples; increase n_estimators or reduce max_samples")
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


def _fit_member(
    base: BaseEstimator, X: np.ndarray, y: np.ndarray, samples: np.ndarray, features: np.ndarray, seed: int
) -> BaseEstimator:
    """Clone ``base``, seed it if possible and fit it on the resampled data."""
    est = clone(base)
    if "random_state" in est.get_params():
        est.set_params(random_state=seed)
    return est.fit(X[samples][:, features], y[samples])


class BaggingClassifierScratch(ClassifierMixin, _BaseBaggingScratch):
    """Bagging classifier built from scratch.

    Args:
        estimator: Base estimator to bag; defaults to a fully grown
            :class:`~sklearn_mastery.from_scratch.decision_tree.DecisionTreeClassifierScratch`.
        n_estimators: Number of members ``B``.
        max_samples: Size of each resample (int, or fraction of ``n_samples``).
        max_features: Number of features per member (int or fraction).
        bootstrap: Sample rows with replacement (``True``) or without.
        bootstrap_features: Sample features with replacement.
        oob_score: Compute the out-of-bag accuracy after fitting.
        voting: ``"soft"`` averages ``predict_proba`` of the members (falls
            back to hard voting if the base estimator has none); ``"hard"``
            counts predicted labels.
        n_jobs: Number of joblib threads used to fit members in parallel.
        random_state: Seed for resampling and member seeding.

    Attributes:
        classes_: Sorted unique class labels.
        estimators_: Fitted members.
        estimators_samples_: Row indices drawn for each member.
        estimators_features_: Feature indices drawn for each member.
        oob_decision_function_: OOB class probabilities ``(n_samples, n_classes)``
            (only when ``oob_score=True``; rows without OOB votes are NaN).
        oob_score_: OOB accuracy (only when ``oob_score=True``).
    """

    _is_classifier = True

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        n_estimators: int = 10,
        max_samples: Union[int, float] = 1.0,
        max_features: Union[int, float] = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        voting: str = "soft",
        n_jobs: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.voting = voting

    def _default_estimator(self) -> BaseEstimator:
        return DecisionTreeClassifierScratch()

    def _validate_targets(self, y: np.ndarray) -> np.ndarray:
        if self.voting not in ("soft", "hard"):
            raise ValueError(f"voting must be 'soft' or 'hard'; got {self.voting!r}")
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return y_enc.astype(np.intp)

    def _member_output(self, est: BaseEstimator, X: np.ndarray) -> np.ndarray:
        """Class-probability matrix of one member, aligned to ``self.classes_``.

        A member fitted on a bootstrap sample may have missed some classes,
        so its ``predict_proba`` columns are scattered into the full width.
        """
        proba = np.zeros((X.shape[0], self.n_classes_))
        member_classes = np.asarray(est.classes_, dtype=np.intp)
        if self.voting == "soft" and hasattr(est, "predict_proba"):
            proba[:, member_classes] = est.predict_proba(X)
        else:
            proba[np.arange(X.shape[0]), np.asarray(est.predict(X), dtype=np.intp)] = 1.0
        return proba

    def _store_oob(self, aggregated: np.ndarray, y: np.ndarray, has_vote: np.ndarray) -> None:
        self.oob_decision_function_ = aggregated
        self.oob_score_ = float(accuracy_score(y[has_vote], np.argmax(aggregated[has_vote], axis=1)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average (soft) or vote-fraction (hard) class probabilities."""
        X = self._check_X(X)
        return self._member_outputs(X).mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class with the highest aggregated probability / vote count."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class BaggingRegressorScratch(RegressorMixin, _BaseBaggingScratch):
    """Bagging regressor built from scratch.

    Args:
        estimator: Base estimator to bag; defaults to a fully grown
            :class:`~sklearn_mastery.from_scratch.decision_tree.DecisionTreeRegressorScratch`.
        n_estimators: Number of members ``B``.
        max_samples: Size of each resample (int, or fraction of ``n_samples``).
        max_features: Number of features per member (int or fraction).
        bootstrap: Sample rows with replacement (``True``) or without.
        bootstrap_features: Sample features with replacement.
        oob_score: Compute the out-of-bag $R^2$ after fitting.
        n_jobs: Number of joblib threads used to fit members in parallel.
        random_state: Seed for resampling and member seeding.

    Attributes:
        estimators_: Fitted members.
        estimators_samples_: Row indices drawn for each member.
        estimators_features_: Feature indices drawn for each member.
        oob_prediction_: OOB predictions ``(n_samples,)`` (NaN where no vote).
        oob_score_: OOB $R^2$ (only when ``oob_score=True``).
    """

    _is_classifier = False

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        n_estimators: int = 10,
        max_samples: Union[int, float] = 1.0,
        max_features: Union[int, float] = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        n_jobs: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def _default_estimator(self) -> BaseEstimator:
        return DecisionTreeRegressorScratch()

    def _validate_targets(self, y: np.ndarray) -> np.ndarray:
        return y.astype(np.float64)

    def _member_output(self, est: BaseEstimator, X: np.ndarray) -> np.ndarray:
        return np.asarray(est.predict(X), dtype=np.float64)

    def _store_oob(self, aggregated: np.ndarray, y: np.ndarray, has_vote: np.ndarray) -> None:
        self.oob_prediction_ = aggregated
        self.oob_score_ = float(r2_score(y[has_vote], aggregated[has_vote]))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Average the members' predictions."""
        X = self._check_X(X)
        return self._member_outputs(X).mean(axis=0)
