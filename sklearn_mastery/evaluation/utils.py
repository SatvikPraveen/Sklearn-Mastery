"""Shared utilities for the evaluation package.

Small, dependency-light helpers used across the metrics, statistical-test,
cross-validation and analyzer modules: input validation, bootstrap
confidence intervals, effect sizes, multiple-comparison corrections and
conversion of numpy objects to JSON-serialisable builtins.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error  # noqa: F401 - re-exported for compat

ArrayLike = Union[np.ndarray, list, tuple, pd.Series, pd.DataFrame]

#: Metrics for which a *lower* value is better. Everything else is assumed
#: to be "higher is better".
LOWER_IS_BETTER_METRICS = frozenset(
    {
        "mse",
        "rmse",
        "mae",
        "mape",
        "median_absolute_error",
        "max_error",
        "log_loss",
        "brier_score",
        "ece",
        "mce",
        "davies_bouldin_score",
        "neg_mean_squared_error",
    }
)


# --------------------------------------------------------------------------- #
# Random state / conversions
# --------------------------------------------------------------------------- #
def make_rng(random_state: Optional[Union[int, np.random.Generator]] = None) -> np.random.Generator:
    """Return a :class:`numpy.random.Generator` for ``random_state``.

    Args:
        random_state: ``None`` (fresh entropy), an integer seed, or an existing
            generator which is returned unchanged.

    Returns:
        A numpy random generator.
    """
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, np.random.RandomState):
        return np.random.default_rng(random_state.randint(0, 2**31 - 1))
    return np.random.default_rng(random_state)


def to_builtin(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to JSON-serialisable builtins.

    Args:
        obj: Any object; dicts, lists and tuples are traversed recursively.

    Returns:
        The same structure with ``np.generic`` scalars converted to Python
        scalars and ``np.ndarray`` converted to lists.
    """
    if isinstance(obj, dict):
        return {str(k) if isinstance(k, np.generic) else k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_builtin(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def to_scalar(value: Any) -> Any:
    """Convert a numpy scalar to the matching Python scalar, leaving others untouched."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def is_higher_better(metric_name: str) -> bool:
    """Return ``True`` when a larger value of ``metric_name`` is better.

    Args:
        metric_name: Metric key such as ``"accuracy"`` or ``"rmse"``.

    Returns:
        ``False`` for error/loss-type metrics, ``True`` otherwise.
    """
    name = metric_name.lower()
    if name.startswith("neg_"):
        return True
    return name not in LOWER_IS_BETTER_METRICS


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def ensure_numpy_array(data: ArrayLike) -> np.ndarray:
    """Ensure data is a numpy array.

    Args:
        data: Input data as array, list, tuple, Series or DataFrame.

    Returns:
        Data as a numpy array (no copy when already an array).
    """
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return data.to_numpy()
    return np.asarray(data)


def check_consistent_length(*arrays: Optional[ArrayLike]) -> None:
    """Check that all non-``None`` arrays share the same first dimension.

    Args:
        *arrays: Arrays to compare.

    Raises:
        ValueError: If the lengths differ.
    """
    lengths = [len(arr) for arr in arrays if arr is not None]
    if len(set(lengths)) > 1:
        raise ValueError(f"Inconsistent array lengths: {lengths}")


def validate_targets(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    allow_empty: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and coerce a pair of target arrays.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values (labels, scores or probabilities).
        allow_empty: Whether zero-length inputs are acceptable.

    Returns:
        ``(y_true, y_pred)`` as numpy arrays.

    Raises:
        ValueError: If lengths differ, inputs are empty (unless allowed) or
            numeric inputs contain NaN/inf.
    """
    y_true = ensure_numpy_array(y_true)
    y_pred = ensure_numpy_array(y_pred)
    if y_true.ndim == 0 or y_pred.ndim == 0:
        raise ValueError("Targets must be at least 1-dimensional")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Found input variables with inconsistent numbers of samples: [{len(y_true)}, {len(y_pred)}]"
        )
    if not allow_empty and len(y_true) == 0:
        raise ValueError("Found empty input arrays; at least one sample is required")
    for name, arr in (("y_true", y_true), ("y_pred", y_pred)):
        if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or infinite values")
    return y_true, y_pred


def validate_binary_probabilities(
    y_true: ArrayLike,
    y_proba: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate labels and scores for binary threshold/calibration analysis.

    Two-column probability matrices are reduced to the positive-class column.

    Args:
        y_true: Binary labels.
        y_proba: 1-D scores/probabilities or an ``(n_samples, 2)`` matrix.

    Returns:
        ``(y_true, y_proba)`` with ``y_proba`` one-dimensional.

    Raises:
        ValueError: If shapes are inconsistent or ``y_true`` has more than two classes.
    """
    y_true = ensure_numpy_array(y_true)
    y_proba = ensure_numpy_array(y_proba)
    if y_proba.ndim == 2:
        if y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]
        elif y_proba.shape[1] == 1:
            y_proba = y_proba[:, 0]
        else:
            raise ValueError(f"Expected 1-D scores or an (n_samples, 2) matrix, got shape {y_proba.shape}")
    y_true, y_proba = validate_targets(y_true, y_proba)
    if len(np.unique(y_true)) > 2:
        raise ValueError("Binary analysis requires at most two classes in y_true")
    return y_true, y_proba


def ensure_binary_classification(y_true: np.ndarray, y_pred_proba: np.ndarray) -> bool:
    """Return ``True`` when inputs describe a binary problem with a 2-column probability matrix."""
    y_pred_proba = ensure_numpy_array(y_pred_proba)
    return len(np.unique(y_true)) == 2 and y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2


def validate_evaluation_inputs(
    X: ArrayLike,
    y: ArrayLike,
    task_type: str,
    allow_empty: bool = False,
) -> None:
    """Validate a feature matrix / target pair for a given task type.

    Args:
        X: Feature matrix.
        y: Target vector.
        task_type: One of ``"classification"``, ``"regression"``, ``"clustering"``.
        allow_empty: Whether zero-length datasets are acceptable.

    Raises:
        ValueError: If inputs are inconsistent or unsuitable for the task.
    """
    valid_tasks = ("classification", "regression", "clustering")
    if task_type not in valid_tasks:
        raise ValueError(f"Unknown task type: {task_type!r}. Must be one of {list(valid_tasks)}")

    X = ensure_numpy_array(X)
    if not allow_empty and len(X) == 0:
        raise ValueError("Empty dataset provided")
    if y is None:
        return
    y = ensure_numpy_array(y)
    if len(X) != len(y):
        raise ValueError(f"X and y must have the same length: {len(X)} vs {len(y)}")
    if np.issubdtype(X.dtype, np.number) and np.any(np.isnan(X)):
        raise ValueError("Input features contain NaN values")
    if np.issubdtype(y.dtype, np.number) and np.any(np.isnan(y)):
        raise ValueError("Target contains NaN values")
    if task_type == "classification" and len(y) > 0 and len(np.unique(y)) < 2:
        raise ValueError("Classification requires at least 2 classes")


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide ``numerator`` by ``denominator``, returning ``default`` for a zero denominator."""
    return numerator / denominator if denominator != 0 else default


def safe_metric_computation(
    metric_func: Callable[..., float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    default_value: float = np.nan,
    **kwargs: Any,
) -> float:
    """Compute a metric, returning ``default_value`` on ``ValueError``/``ZeroDivisionError``.

    Args:
        metric_func: Metric callable ``f(y_true, y_pred, **kwargs)``.
        y_true: Ground truth.
        y_pred: Predictions.
        default_value: Value returned when the metric is undefined.
        **kwargs: Passed through to ``metric_func``.

    Returns:
        The metric value as a Python float, or ``default_value``.
    """
    try:
        return float(metric_func(y_true, y_pred, **kwargs))
    except (ValueError, ZeroDivisionError, TypeError):
        return default_value


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_confidence_interval(
    data: ArrayLike,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    method: str = "percentile",
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Dict[str, Any]:
    """Non-parametric bootstrap confidence interval for a statistic.

    Resamples ``data`` with replacement ``n_bootstrap`` times and forms an
    interval from the empirical distribution of the statistic.

    Methods (Efron & Tibshirani, 1993, chs. 13-14; DiCiccio & Efron, 1996):

    * ``"percentile"``: ``[q_{alpha/2}, q_{1-alpha/2}]`` of the bootstrap
      distribution.
    * ``"basic"``: reverse-percentile ``[2t - q_{1-alpha/2}, 2t - q_{alpha/2}]``.
    * ``"bca"``: bias-corrected and accelerated interval via
      :func:`scipy.stats.bootstrap`.

    Args:
        data: One-dimensional sample.
        statistic: Function mapping a sample to a scalar (must accept ``axis``
            for ``method="bca"`` vectorisation, e.g. ``np.mean``).
        n_bootstrap: Number of resamples.
        confidence_level: Coverage probability in ``(0, 1)``.
        method: ``"percentile"``, ``"basic"`` or ``"bca"``.
        random_state: Seed or generator for reproducibility.

    Returns:
        Dict with ``statistic`` (point estimate), ``ci_lower``, ``ci_upper``,
        ``confidence_level``, ``method``, ``n_bootstrap``, ``bootstrap_std`` and
        the raw ``bootstrap_distribution`` (``None`` for ``"bca"``).

    Raises:
        ValueError: For an empty sample, invalid level or unknown method.

    References:
        Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
        DiCiccio, T. J. & Efron, B. (1996). Bootstrap confidence intervals.
        *Statistical Science*, 11(3), 189-228.
    """
    data = ensure_numpy_array(data).ravel()
    if len(data) == 0:
        raise ValueError("Cannot bootstrap an empty sample")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie in (0, 1)")
    if method not in {"percentile", "basic", "bca"}:
        raise ValueError(f"Unknown bootstrap method: {method!r}")

    rng = make_rng(random_state)
    point = float(statistic(data))
    alpha = 1.0 - confidence_level

    if method == "bca":
        from scipy import stats

        res = stats.bootstrap(
            (data,),
            statistic,
            n_resamples=n_bootstrap,
            confidence_level=confidence_level,
            method="BCa",
            random_state=rng,
        )
        return {
            "statistic": point,
            "ci_lower": float(res.confidence_interval.low),
            "ci_upper": float(res.confidence_interval.high),
            "bootstrap_std": float(res.standard_error),
            "confidence_level": confidence_level,
            "method": method,
            "n_bootstrap": n_bootstrap,
            "bootstrap_distribution": None,
        }

    n = len(data)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot = np.array([statistic(data[row]) for row in idx], dtype=float)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    if method == "basic":
        lo, hi = 2 * point - hi, 2 * point - lo
    return {
        "statistic": point,
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "bootstrap_std": float(np.std(boot, ddof=1)) if n_bootstrap > 1 else float("nan"),
        "confidence_level": confidence_level,
        "method": method,
        "n_bootstrap": n_bootstrap,
        "bootstrap_distribution": boot,
    }


def bootstrap_metric(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """Percentile-bootstrap statistics for a paired ``metric(y_true, y_pred)``.

    Args:
        y_true: Ground truth.
        y_pred: Predictions aligned with ``y_true``.
        metric_func: Metric callable.
        n_bootstrap: Number of resamples.
        random_state: Seed or generator for reproducibility.
        confidence_level: Coverage of the returned interval.

    Returns:
        Dict with ``mean``, ``std``, ``ci_lower``, ``ci_upper`` and
        ``n_valid_samples`` (resamples for which the metric was defined).
    """
    y_true, y_pred = validate_targets(y_true, y_pred)
    rng = make_rng(random_state)
    n = len(y_true)
    alpha = 1.0 - confidence_level
    scores: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            scores.append(float(metric_func(y_true[idx], y_pred[idx])))
        except (ValueError, ZeroDivisionError):
            continue
    if not scores:
        return {"mean": np.nan, "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n_valid_samples": 0}
    arr = np.asarray(scores)
    lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "n_valid_samples": len(arr),
    }


# --------------------------------------------------------------------------- #
# Effect sizes and multiple comparisons
# --------------------------------------------------------------------------- #
def compute_effect_size(
    group1_scores: ArrayLike,
    group2_scores: ArrayLike,
    method: str = "cohens_d",
) -> float:
    """Standardised mean difference between two groups.

    * ``"cohens_d"``: pooled-standard-deviation *d* (Cohen, 1988).
    * ``"hedges_g"``: Cohen's *d* with the small-sample bias correction
      ``J = 1 - 3 / (4(n1 + n2) - 9)`` (Hedges, 1981).
    * ``"glass_delta"``: mean difference over the *second* (control) group's SD.
    * ``"cohens_dz"``: paired-samples *d* = mean(diff) / sd(diff); requires equal lengths.

    Args:
        group1_scores: First sample.
        group2_scores: Second (control) sample.
        method: Effect-size definition.

    Returns:
        The effect size (``0.0`` when the relevant standard deviation is zero).

    Raises:
        ValueError: For an unknown method or mismatched lengths with ``"cohens_dz"``.
    """
    g1 = ensure_numpy_array(group1_scores).astype(float).ravel()
    g2 = ensure_numpy_array(group2_scores).astype(float).ravel()
    mean_diff = float(np.mean(g1) - np.mean(g2))
    n1, n2 = len(g1), len(g2)

    if method in {"cohens_d", "hedges_g"}:
        var1 = np.var(g1, ddof=1) if n1 > 1 else 0.0
        var2 = np.var(g2, ddof=1) if n2 > 1 else 0.0
        dof = n1 + n2 - 2
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / dof) if dof > 0 else 0.0
        d = mean_diff / pooled_std if pooled_std > 0 else 0.0
        if method == "hedges_g" and (4 * (n1 + n2) - 9) > 0:
            d *= 1.0 - 3.0 / (4 * (n1 + n2) - 9)
        return float(d)
    if method == "glass_delta":
        control_std = np.std(g2, ddof=1) if n2 > 1 else 0.0
        return float(mean_diff / control_std) if control_std > 0 else 0.0
    if method == "cohens_dz":
        if n1 != n2:
            raise ValueError("cohens_dz requires paired samples of equal length")
        diff = g1 - g2
        sd = np.std(diff, ddof=1) if n1 > 1 else 0.0
        return float(np.mean(diff) / sd) if sd > 0 else 0.0
    raise ValueError(f"Unknown effect size method: {method!r}")


def interpret_effect_size(effect_size: float, method: str = "cohens_d") -> str:
    """Label an effect size using Cohen's (1988) conventions (0.2 / 0.5 / 0.8)."""
    abs_effect = abs(float(effect_size))
    if abs_effect < 0.2:
        return "negligible"
    if abs_effect < 0.5:
        return "small"
    if abs_effect < 0.8:
        return "medium"
    return "large"


def holm_bonferroni(p_values: Sequence[float]) -> np.ndarray:
    """Holm's step-down adjustment of p-values for multiple comparisons.

    Controls the family-wise error rate and is uniformly more powerful than
    the plain Bonferroni correction (Holm, 1979); recommended by Demšar (2006)
    for post-hoc classifier comparisons.

    Args:
        p_values: Raw p-values.

    Returns:
        Adjusted p-values (same order as the input), clipped to ``[0, 1]``.

    References:
        Holm, S. (1979). A simple sequentially rejective multiple test
        procedure. *Scandinavian Journal of Statistics*, 6(2), 65-70.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running_max = max(running_max, val)
        adjusted[idx] = running_max
    return adjusted


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
def format_metric_value(value: Optional[float], metric_name: str) -> str:
    """Format a metric value for display (``"N/A"`` for missing values)."""
    if value is None or pd.isna(value):
        return "N/A"
    value = float(value)
    error_metrics = ("mse", "rmse", "mae", "mape")
    if any(m in metric_name.lower() for m in error_metrics):
        if abs(value) < 0.001:
            return f"{value:.2e}"
        if abs(value) < 1:
            return f"{value:.6f}"
        return f"{value:.4f}"
    if abs(value) < 0.001 and value != 0:
        return f"{value:.2e}"
    return f"{value:.4f}"


def create_evaluation_summary_dict(
    model_name: str,
    task_type: str,
    metrics: Dict[str, float],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a standardised evaluation summary dictionary."""
    summary: Dict[str, Any] = {
        "model_name": model_name,
        "task_type": task_type,
        "metrics": metrics,
        "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        "n_metrics": len(metrics),
    }
    if metadata:
        summary["metadata"] = metadata
    return summary


def filter_valid_scores(scores: ArrayLike) -> np.ndarray:
    """Drop NaN/inf entries from an array of scores."""
    arr = ensure_numpy_array(scores).astype(float)
    return arr[np.isfinite(arr)]


def compute_metric_stability(scores: ArrayLike) -> Dict[str, float]:
    """Summary statistics (mean, std, coefficient of variation, range) of fold scores."""
    valid = filter_valid_scores(scores)
    if len(valid) == 0:
        return {k: np.nan for k in ("mean", "std", "cv", "min", "max", "range")}
    mean_score = float(np.mean(valid))
    std_score = float(np.std(valid))
    return {
        "mean": mean_score,
        "std": std_score,
        "cv": std_score / abs(mean_score) if mean_score != 0 else np.inf,
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "range": float(np.max(valid) - np.min(valid)),
    }


def rank_models_by_metric(
    results: List[Dict[str, Any]],
    metric_name: str,
    ascending: bool = False,
) -> List[Dict[str, Any]]:
    """Sort evaluation-result dicts by a metric found at ``metric``, ``test_<metric>`` or ``metrics.<metric>``."""

    def get_metric_value(result: Dict[str, Any]) -> float:
        for key_path in (metric_name, f"test_{metric_name}", f"metrics.{metric_name}"):
            value: Any = result
            for key in key_path.split("."):
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    value = None
                    break
            if value is not None and not pd.isna(value):
                return float(value)
        return np.inf if ascending else -np.inf

    return sorted(results, key=get_metric_value, reverse=not ascending)


__all__ = [
    "LOWER_IS_BETTER_METRICS",
    "bootstrap_confidence_interval",
    "bootstrap_metric",
    "check_consistent_length",
    "compute_effect_size",
    "compute_metric_stability",
    "create_evaluation_summary_dict",
    "ensure_binary_classification",
    "ensure_numpy_array",
    "filter_valid_scores",
    "format_metric_value",
    "holm_bonferroni",
    "interpret_effect_size",
    "is_higher_better",
    "make_rng",
    "rank_models_by_metric",
    "safe_division",
    "safe_metric_computation",
    "to_builtin",
    "to_scalar",
    "validate_binary_probabilities",
    "validate_evaluation_inputs",
    "validate_targets",
]
