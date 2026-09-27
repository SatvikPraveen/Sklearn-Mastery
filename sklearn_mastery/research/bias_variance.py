"""Bias–variance decomposition via bootstrap resampling.

Implements the decompositions of Domingos (2000) for squared loss and 0-1 loss:

* **Squared loss**: ``E[(y - ŷ)²] = bias² + variance + noise``. Without access
  to the noise-free target the reported ``bias²`` absorbs the irreducible
  noise term, as is standard when the Bayes-optimal predictor is unknown.
* **0-1 loss**: main prediction is the majority vote; bias is the 0-1 loss of
  the main prediction; variance is the probability that a bootstrap model
  disagrees with the main prediction. Loss = bias + (net) variance where the
  net variance subtracts variance on biased points (Domingos, 2000, Thm 1).

References
----------
Domingos, P. (2000). A unified bias-variance decomposition and its
applications. *ICML*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.utils import check_random_state

__all__ = ["BiasVarianceResult", "bias_variance_decomposition"]


@dataclass
class BiasVarianceResult:
    """Decomposition summary. ``loss ≈ bias + variance`` (0-1 loss uses net variance)."""

    loss: str
    expected_loss: float
    bias: float
    variance: float
    n_rounds: int
    variance_unbiased: Optional[float] = None
    variance_biased: Optional[float] = None
    predictions: Optional[np.ndarray] = None  # shape (n_rounds, n_test)

    def as_dict(self) -> dict:
        out = {
            "loss": self.loss,
            "expected_loss": self.expected_loss,
            "bias": self.bias,
            "variance": self.variance,
            "n_rounds": self.n_rounds,
        }
        if self.variance_unbiased is not None:
            out["variance_unbiased"] = self.variance_unbiased
            out["variance_biased"] = self.variance_biased
        return out


def _fit_predict(estimator, X_train, y_train, X_test, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, X_train.shape[0], X_train.shape[0])
    est = clone(estimator)
    if "random_state" in est.get_params():
        est.set_params(random_state=seed)
    est.fit(X_train[idx], y_train[idx])
    return est.predict(X_test)


def bias_variance_decomposition(
    estimator: BaseEstimator,
    X_train,
    y_train,
    X_test,
    y_test,
    loss: str = "mse",
    n_rounds: int = 100,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
    return_predictions: bool = False,
) -> BiasVarianceResult:
    """Estimate bias and variance of ``estimator`` with bootstrap training sets.

    Args:
        estimator: Unfitted estimator (cloned every round).
        X_train: Training features, resampled with replacement each round.
        y_train: Training targets.
        X_test: Fixed evaluation features.
        y_test: Fixed evaluation targets.
        loss: ``'mse'`` (regression) or ``'0-1'`` (classification).
        n_rounds: Bootstrap rounds.
        random_state: Seed for the bootstrap draws.
        n_jobs: Parallel jobs.
        return_predictions: Keep the ``(n_rounds, n_test)`` prediction matrix.

    Returns:
        :class:`BiasVarianceResult`.
    """
    if loss not in {"mse", "0-1"}:
        raise ValueError("loss must be 'mse' or '0-1'")
    if n_rounds < 2:
        raise ValueError("n_rounds must be >= 2")
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    rng = check_random_state(random_state)
    seeds = rng.randint(0, np.iinfo(np.int32).max, size=n_rounds)

    preds = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict)(estimator, X_train, y_train, X_test, int(s)) for s in seeds
    )
    P = np.stack(preds)  # (n_rounds, n_test)

    if loss == "mse":
        P = P.astype(float)
        main = P.mean(axis=0)
        expected_loss = float(np.mean((P - y_test[None, :]) ** 2))
        bias = float(np.mean((main - y_test) ** 2))
        variance = float(np.mean(P.var(axis=0)))
        return BiasVarianceResult(
            loss=loss,
            expected_loss=expected_loss,
            bias=bias,
            variance=variance,
            n_rounds=n_rounds,
            predictions=P if return_predictions else None,
        )

    # 0-1 loss (Domingos 2000)
    classes = np.unique(np.concatenate([y_train, y_test, P.ravel()]))
    enc = {c: i for i, c in enumerate(classes)}
    Pi = np.vectorize(enc.get)(P)
    counts = np.apply_along_axis(
        lambda col: np.bincount(col, minlength=len(classes)), 0, Pi
    )  # (n_classes, n_test)
    main_idx = counts.argmax(axis=0)
    main = classes[main_idx]
    expected_loss = float(np.mean(P != y_test[None, :]))
    biased = main != y_test
    bias = float(np.mean(biased))
    disagree = np.mean(P != main[None, :], axis=0)  # per-point variance
    var_unbiased = float(np.mean(disagree[~biased])) * float(np.mean(~biased)) if (~biased).any() else 0.0
    var_biased = float(np.mean(disagree[biased])) * float(np.mean(biased)) if biased.any() else 0.0
    net_variance = var_unbiased - var_biased
    return BiasVarianceResult(
        loss=loss,
        expected_loss=expected_loss,
        bias=bias,
        variance=net_variance,
        n_rounds=n_rounds,
        variance_unbiased=var_unbiased,
        variance_biased=var_biased,
        predictions=P if return_predictions else None,
    )
