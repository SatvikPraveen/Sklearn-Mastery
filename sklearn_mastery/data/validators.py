"""Data quality and schema validation utilities.

This module provides two complementary validators:

* :class:`DataValidator` runs a battery of data-quality checks (missing values,
  duplicates, constant features, outliers, multicollinearity, target sanity,
  task-specific requirements, ...) on a feature matrix and optional target and
  produces a structured report made of :class:`ValidationIssue` records.
* :class:`SchemaValidator` checks a DataFrame against a declarative schema
  (required columns, column dtypes, value ranges, allowed values).

Both accept numpy arrays and pandas DataFrames.  Arrays are wrapped in a
DataFrame with ``feature_<i>`` column names so that every message can refer to
a column by name.

Example:
    >>> import numpy as np, pandas as pd
    >>> from sklearn_mastery.data.validators import DataValidator
    >>> X = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [1, 1, 1, 1]})
    >>> report = DataValidator().validate_dataset(X)
    >>> report["validation_status"]
    'FAILED'
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, TextIO, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import (
    CategoricalDtype,
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)
from scipy import stats

from sklearn_mastery.config.logging_config import LoggerMixin

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series, Sequence[Any]]
CustomRule = Callable[[pd.DataFrame, Optional[pd.Series]], Any]

VALID_TASK_TYPES: Tuple[str, ...] = ("classification", "regression", "clustering")
VALID_STATUSES: Tuple[str, ...] = ("PASSED", "PASSED_WITH_WARNINGS", "FAILED")


# --------------------------------------------------------------------------- #
# Issue model
# --------------------------------------------------------------------------- #
class ValidationSeverity(str, Enum):
    """Severity levels of a validation issue, ordered from least to most severe."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Integer rank (``info`` = 0 ... ``critical`` = 3) for ordering."""
        return _SEVERITY_RANK[self]

    @property
    def icon(self) -> str:
        """Console glyph used by :meth:`DataValidator.print_summary`."""
        return _SEVERITY_ICON[self]


_SEVERITY_RANK: Dict[ValidationSeverity, int] = {
    ValidationSeverity.INFO: 0,
    ValidationSeverity.WARNING: 1,
    ValidationSeverity.ERROR: 2,
    ValidationSeverity.CRITICAL: 3,
}
_SEVERITY_ICON: Dict[ValidationSeverity, str] = {
    ValidationSeverity.INFO: "ℹ️",  # noqa: RUF001 - deliberate console glyph
    ValidationSeverity.WARNING: "⚠️",
    ValidationSeverity.ERROR: "❌",
    ValidationSeverity.CRITICAL: "🔴",
}
_SEVERITY_PENALTY: Dict[ValidationSeverity, float] = {
    ValidationSeverity.INFO: 1.0,
    ValidationSeverity.WARNING: 5.0,
    ValidationSeverity.ERROR: 15.0,
    ValidationSeverity.CRITICAL: 30.0,
}


@dataclass
class ValidationIssue:
    """A single finding produced by a validation check.

    Attributes:
        severity: How serious the issue is.
        category: Machine-readable check family (``"missing_values"``, ``"target"``, ...).
        message: Human-readable description.
        column: Column the issue refers to, if any.
        details: Structured, JSON-serialisable context (counts, fractions, names).
    """

    severity: ValidationSeverity
    category: str
    message: str
    column: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.severity, ValidationSeverity):
            self.severity = ValidationSeverity(str(self.severity).lower())

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict representation with the severity as a string."""
        return {
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
            "column": self.column,
            "details": _to_builtin(self.details),
        }


@dataclass
class ValidationReport:
    """Structured result of :meth:`DataValidator.validate_dataset`.

    Attributes:
        issues: All issues found, sorted from most to least severe.
        validation_status: ``"PASSED"``, ``"PASSED_WITH_WARNINGS"`` or ``"FAILED"``.
        dataset_statistics: Shape, dtype composition, missing/duplicate counts, target summary.
        quality_metrics: Normalised quality indicators and an ``overall_score`` in ``[0, 100]``.
        recommendations: Actionable next steps derived from the issues.
        task_type: Task type used for target/task checks (explicit or inferred).
    """

    issues: List[ValidationIssue]
    validation_status: str
    dataset_statistics: Dict[str, Any]
    quality_metrics: Dict[str, float]
    recommendations: List[str]
    task_type: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """``True`` unless the status is ``FAILED``."""
        return self.validation_status != "FAILED"

    @property
    def severity_counts(self) -> Dict[str, int]:
        """Number of issues per severity level (all levels present)."""
        counts = {severity.value: 0 for severity in ValidationSeverity}
        for issue in self.issues:
            counts[issue.severity.value] += 1
        return counts

    @property
    def issues_by_category(self) -> Dict[str, List[Dict[str, Any]]]:
        """Issue dicts grouped by category, preserving severity order."""
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for issue in self.issues:
            grouped.setdefault(issue.category, []).append(issue.to_dict())
        return grouped

    def to_dict(self) -> Dict[str, Any]:
        """Return the report as a nested, JSON-serialisable dictionary."""
        return {
            "validation_status": self.validation_status,
            "is_valid": self.is_valid,
            "task_type": self.task_type,
            "severity_counts": self.severity_counts,
            "total_issues": len(self.issues),
            "dataset_statistics": _to_builtin(self.dataset_statistics),
            "quality_metrics": _to_builtin(self.quality_metrics),
            "issues_by_category": self.issues_by_category,
            "recommendations": list(self.recommendations),
            "detailed_issues": [issue.to_dict() for issue in self.issues],
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_builtin(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars and containers to built-in types."""
    if isinstance(value, dict):
        return {
            str(k) if not isinstance(k, (str, int, float, bool)) else k: _to_builtin(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_builtin(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Series):
        return _to_builtin(value.to_dict())
    return value


def _to_frame(X: ArrayLike, name: str = "X") -> pd.DataFrame:
    """Coerce ``X`` to a DataFrame, naming array columns ``feature_<i>``.

    Raises:
        ValueError: If ``X`` has more than two dimensions or is a scalar.
    """
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, pd.Series):
        return X.to_frame()
    arr = np.asarray(X)
    if arr.ndim == 0:
        raise ValueError(f"{name} must be 1-D or 2-D, got a scalar.")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim > 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got {arr.ndim}-D.")
    return pd.DataFrame(arr, columns=[f"feature_{i}" for i in range(arr.shape[1])])


def _to_target(y: Optional[ArrayLike]) -> Optional[Union[pd.Series, pd.DataFrame]]:
    """Coerce ``y`` to a Series (single output) or DataFrame (multi-output)."""
    if y is None:
        return None
    if isinstance(y, pd.Series):
        return y.reset_index(drop=True)
    if isinstance(y, pd.DataFrame):
        return y.iloc[:, 0].reset_index(drop=True) if y.shape[1] == 1 else y.reset_index(drop=True)
    arr = np.asarray(y)
    if arr.ndim == 0:
        raise ValueError("y must be 1-D or 2-D, got a scalar.")
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if arr.ndim == 2:
        return pd.DataFrame(arr, columns=[f"target_{i}" for i in range(arr.shape[1])])
    if arr.ndim > 2:
        raise ValueError(f"y must be 1-D or 2-D, got {arr.ndim}-D.")
    return pd.Series(arr, name="target")


def _numeric_columns(X: pd.DataFrame) -> List[Any]:
    """Columns with a real numeric dtype (booleans excluded)."""
    return [c for c in X.columns if is_numeric_dtype(X[c]) and not is_bool_dtype(X[c])]


def _categorical_columns(X: pd.DataFrame) -> List[Any]:
    """Columns holding object, string, categorical or boolean data."""
    return [
        c
        for c in X.columns
        if is_object_dtype(X[c])
        or is_string_dtype(X[c])
        or isinstance(X[c].dtype, CategoricalDtype)
        or is_bool_dtype(X[c])
    ]


def _datetime_columns(X: pd.DataFrame) -> List[Any]:
    return [c for c in X.columns if is_datetime64_any_dtype(X[c])]


def _finite_values(series: pd.Series) -> np.ndarray:
    """Float array of the finite, non-missing values of a numeric series."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    return values[np.isfinite(values)]


def _dtype_matches(dtype: Any, spec: Any) -> bool:
    """Return whether ``dtype`` satisfies a schema type specification.

    ``spec`` may be a friendly alias (``"int"``, ``"float"``, ``"numeric"``,
    ``"object"``/``"string"``, ``"bool"``, ``"datetime"``, ``"category"``),
    a numpy/pandas dtype, or a Python type such as ``int``.
    """
    if isinstance(spec, str):
        alias = spec.strip().lower()
        if alias in {"int", "integer"}:
            return bool(is_integer_dtype(dtype))
        if alias in {"float", "floating", "double"}:
            return bool(is_float_dtype(dtype))
        if alias in {"numeric", "number"}:
            return bool(is_numeric_dtype(dtype)) and not is_bool_dtype(dtype)
        if alias in {"object", "str", "string", "text"}:
            return bool(is_object_dtype(dtype) or is_string_dtype(dtype))
        if alias in {"bool", "boolean"}:
            return bool(is_bool_dtype(dtype))
        if alias in {"datetime", "datetime64", "timestamp"}:
            return bool(is_datetime64_any_dtype(dtype))
        if alias in {"category", "categorical"}:
            return isinstance(dtype, CategoricalDtype)
    if spec is int:
        return bool(is_integer_dtype(dtype))
    if spec is float:
        return bool(is_float_dtype(dtype))
    if spec is bool:
        return bool(is_bool_dtype(dtype))
    if spec is str:
        return bool(is_object_dtype(dtype) or is_string_dtype(dtype))
    try:
        return bool(pd.api.types.is_dtype_equal(dtype, spec))
    except TypeError:
        return False


def _psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index between two numeric samples using reference quantile bins."""
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        return 0.0
    inner = edges[1:-1]
    ref_idx = np.searchsorted(inner, reference, side="right")
    cur_idx = np.searchsorted(inner, current, side="right")
    n_bins_eff = inner.size + 1
    ref_frac = np.bincount(ref_idx, minlength=n_bins_eff) / reference.size
    cur_frac = np.bincount(cur_idx, minlength=n_bins_eff) / current.size
    ref_frac = np.clip(ref_frac, eps, None)
    cur_frac = np.clip(cur_frac, eps, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def _psi_categorical(reference: pd.Series, current: pd.Series, eps: float = 1e-6) -> float:
    """Population Stability Index over category frequencies."""
    ref_frac = reference.value_counts(normalize=True)
    cur_frac = current.value_counts(normalize=True)
    categories = ref_frac.index.union(cur_frac.index)
    ref = np.clip(ref_frac.reindex(categories, fill_value=0.0).to_numpy(dtype=float), eps, None)
    cur = np.clip(cur_frac.reindex(categories, fill_value=0.0).to_numpy(dtype=float), eps, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


# --------------------------------------------------------------------------- #
# DataValidator
# --------------------------------------------------------------------------- #
class DataValidator(LoggerMixin):
    """Run data-quality checks on a feature matrix and optional target.

    Every check emits zero or more :class:`ValidationIssue` records.  The
    overall status is ``FAILED`` when any ``critical`` or ``error`` issue is
    present, ``PASSED_WITH_WARNINGS`` when only warnings/info remain and
    ``PASSED`` otherwise.  In ``strict_mode`` warnings also fail validation.

    Args:
        strict_mode: Treat warnings as failures when determining the status.
        min_samples: Datasets smaller than this trigger a "small dataset" warning.
        missing_thresholds: ``(warning, error, critical)`` per-column missing
            fractions.  Below the first value the issue is informational.
        duplicate_thresholds: ``(warning, error)`` duplicate-row fractions.
        low_variance_threshold: Share of rows a single value may cover before a
            feature is flagged as low-variance.
        outlier_iqr_factor: IQR multiplier for the outlier fences (3.0 = "extreme").
        outlier_fraction_warning: Outlier share above which the issue is a warning.
        outlier_min_samples: Minimum finite values required to run outlier detection.
        high_cardinality_threshold: Absolute unique-count threshold for categoricals.
        high_cardinality_ratio: Unique/non-null ratio above which a categorical with
            more than ten levels is flagged.
        correlation_threshold: Absolute Pearson correlation flagged as multicollinearity.
        max_corr_features: Skip the pairwise correlation check above this width.
        scale_ratio_threshold: Max/min standard-deviation ratio flagged as needing scaling.
        imbalance_ratio: Minority/majority ratio below which classes are imbalanced.
        severe_imbalance_ratio: Ratio below which the imbalance is an error.
        min_samples_per_class: Classes with fewer samples are flagged.
        min_unique_regression: Regression targets with fewer unique values are flagged.
        target_skew_threshold: Absolute skewness above which a regression target is flagged.

    Attributes:
        report_: The :class:`ValidationReport` from the last ``validate_dataset`` call.
        issues_: Issues from the last call, sorted from most to least severe.
    """

    CHECK_NAMES: Tuple[str, ...] = (
        "sample_size",
        "missing_values",
        "duplicates",
        "infinite_values",
        "constant_features",
        "outliers",
        "high_cardinality",
        "data_types",
        "multicollinearity",
        "feature_scaling",
        "target",
        "task_requirements",
        "custom_rules",
    )

    def __init__(
        self,
        strict_mode: bool = False,
        min_samples: int = 30,
        missing_thresholds: Tuple[float, float, float] = (0.05, 0.2, 0.5),
        duplicate_thresholds: Tuple[float, float] = (0.01, 0.1),
        low_variance_threshold: float = 0.98,
        outlier_iqr_factor: float = 3.0,
        outlier_fraction_warning: float = 0.05,
        outlier_min_samples: int = 5,
        high_cardinality_threshold: int = 50,
        high_cardinality_ratio: float = 0.5,
        correlation_threshold: float = 0.95,
        max_corr_features: int = 500,
        scale_ratio_threshold: float = 100.0,
        imbalance_ratio: float = 0.3,
        severe_imbalance_ratio: float = 0.1,
        min_samples_per_class: int = 5,
        min_unique_regression: int = 10,
        target_skew_threshold: float = 2.0,
    ) -> None:
        self.strict_mode = strict_mode
        self.min_samples = min_samples
        self.missing_thresholds = missing_thresholds
        self.duplicate_thresholds = duplicate_thresholds
        self.low_variance_threshold = low_variance_threshold
        self.outlier_iqr_factor = outlier_iqr_factor
        self.outlier_fraction_warning = outlier_fraction_warning
        self.outlier_min_samples = outlier_min_samples
        self.high_cardinality_threshold = high_cardinality_threshold
        self.high_cardinality_ratio = high_cardinality_ratio
        self.correlation_threshold = correlation_threshold
        self.max_corr_features = max_corr_features
        self.scale_ratio_threshold = scale_ratio_threshold
        self.imbalance_ratio = imbalance_ratio
        self.severe_imbalance_ratio = severe_imbalance_ratio
        self.min_samples_per_class = min_samples_per_class
        self.min_unique_regression = min_unique_regression
        self.target_skew_threshold = target_skew_threshold

        self.custom_rules: Dict[str, CustomRule] = {}
        self.report_: Optional[ValidationReport] = None
        self.issues_: List[ValidationIssue] = []

    # ------------------------------------------------------------------ API
    def add_custom_rule(self, name: str, rule: CustomRule) -> DataValidator:
        """Register a custom rule executed by :meth:`validate_dataset`.

        Args:
            name: Unique rule name (re-registering replaces the previous rule).
            rule: Callable ``rule(X, y)`` receiving a DataFrame and an optional
                Series and returning a :class:`ValidationIssue`, an iterable of
                issues, or ``None``.

        Returns:
            ``self`` to allow chaining.

        Raises:
            TypeError: If ``rule`` is not callable.
        """
        if not callable(rule):
            raise TypeError(f"Custom rule '{name}' must be callable, got {type(rule).__name__}.")
        self.custom_rules[name] = rule
        return self

    def remove_custom_rule(self, name: str) -> None:
        """Remove a previously registered custom rule (no-op if absent)."""
        self.custom_rules.pop(name, None)

    def validate_dataset(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        task_type: Optional[str] = None,
        checks: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Run all (or selected) checks and return a structured report dictionary.

        Args:
            X: Feature matrix (DataFrame, 2-D array or array-like).
            y: Optional target vector/matrix.
            task_type: ``"classification"``, ``"regression"`` or ``"clustering"``.
                When ``None`` and ``y`` is given, the task is inferred from ``y``.
            checks: Subset of :attr:`CHECK_NAMES` to run; ``None`` runs all.
                Structural checks (empty data, X/y length mismatch) always run.

        Returns:
            Dictionary with keys ``validation_status``, ``is_valid``, ``task_type``,
            ``severity_counts``, ``total_issues``, ``dataset_statistics``,
            ``quality_metrics``, ``issues_by_category``, ``recommendations`` and
            ``detailed_issues`` (list of issue dicts sorted by severity).

        Raises:
            ValueError: For an unknown ``task_type`` or check name.
        """
        if task_type is not None and task_type not in VALID_TASK_TYPES:
            raise ValueError(f"task_type must be one of {VALID_TASK_TYPES} or None, got {task_type!r}.")
        selected = self._resolve_checks(checks)

        X_df = _to_frame(X)
        y_t = _to_target(y)
        issues: List[ValidationIssue] = []

        structural_ok = self._check_structure(X_df, y_t, issues)
        resolved_task = task_type
        if structural_ok:
            if "sample_size" in selected:
                self._check_sample_size(X_df, issues)
            if "missing_values" in selected:
                self._check_missing_values(X_df, issues)
            if "duplicates" in selected:
                self._check_duplicates(X_df, issues)
            if "infinite_values" in selected:
                self._check_infinite_values(X_df, issues)
            if "constant_features" in selected:
                self._check_constant_features(X_df, issues)
            if "outliers" in selected:
                self._check_outliers(X_df, issues)
            if "high_cardinality" in selected:
                self._check_high_cardinality(X_df, issues)
            if "data_types" in selected:
                self._check_data_types(X_df, issues)
            if "multicollinearity" in selected:
                self._check_multicollinearity(X_df, issues)
            if "feature_scaling" in selected:
                self._check_feature_scaling(X_df, issues)

            lengths_match = y_t is None or len(y_t) == len(X_df)
            if y_t is not None and lengths_match and resolved_task is None:
                resolved_task = self.infer_task_type(y_t)
            if "target" in selected and y_t is not None and lengths_match:
                self._check_target(y_t, resolved_task, issues)
            if "task_requirements" in selected:
                self._check_task_requirements(X_df, y_t if lengths_match else None, resolved_task, issues)
            if "custom_rules" in selected:
                self._run_custom_rules(X_df, y_t if lengths_match else None, issues)

        issues.sort(key=lambda issue: issue.severity.rank, reverse=True)
        status = self._determine_status(issues)
        statistics = self._dataset_statistics(X_df, y_t, resolved_task)
        metrics = self._quality_metrics(X_df, issues)
        recommendations = self._recommendations(issues)

        report = ValidationReport(
            issues=issues,
            validation_status=status,
            dataset_statistics=statistics,
            quality_metrics=metrics,
            recommendations=recommendations,
            task_type=resolved_task,
        )
        self.report_ = report
        self.issues_ = issues
        self.logger.info(
            "Validation %s: %d issue(s) on %d x %d dataset",
            status,
            len(issues),
            X_df.shape[0],
            X_df.shape[1],
        )
        return report.to_dict()

    def validate(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        task_type: Optional[str] = None,
        checks: Optional[Iterable[str]] = None,
    ) -> bool:
        """Run :meth:`validate_dataset` and return whether the data passed.

        Returns:
            ``True`` unless the validation status is ``FAILED``.
        """
        report = self.validate_dataset(X, y, task_type=task_type, checks=checks)
        return bool(report["is_valid"])

    def validate_classification_data(
        self,
        X: ArrayLike,
        y: ArrayLike,
        check_missing: bool = True,
        check_duplicates: bool = True,
        check_outliers: bool = True,
        check_class_balance: bool = True,
        check_feature_correlation: bool = True,
    ) -> Dict[str, Any]:
        """Validate a classification dataset and return a flat summary.

        Args:
            X: Feature matrix.
            y: Class labels.
            check_missing: Run the missing-value check.
            check_duplicates: Run the duplicate-row check.
            check_outliers: Run the outlier check.
            check_class_balance: Run target/class-balance checks.
            check_feature_correlation: Run the multicollinearity check.

        Returns:
            Dictionary with ``validation_status``, ``is_valid``, ``overall_score``,
            ``missing_values_percent``, ``duplicate_rows``, ``class_imbalance_ratio``,
            ``class_counts``, ``high_correlation_pairs`` and the full ``report``.
        """
        checks = self._checks_from_flags(
            {
                "missing_values": check_missing,
                "duplicates": check_duplicates,
                "outliers": check_outliers,
                "target": check_class_balance,
                "multicollinearity": check_feature_correlation,
            }
        )
        report = self.validate_dataset(X, y, task_type="classification", checks=checks)
        y_series = _to_target(y)
        counts = (
            y_series.value_counts(dropna=True) if isinstance(y_series, pd.Series) else pd.Series(dtype=int)
        )
        ratio = float(counts.min() / counts.max()) if len(counts) > 1 and counts.max() > 0 else float("nan")
        summary = self._flat_summary(report)
        summary["class_imbalance_ratio"] = ratio
        summary["class_counts"] = _to_builtin(counts.to_dict())
        return summary

    def validate_regression_data(
        self,
        X: ArrayLike,
        y: ArrayLike,
        check_missing: bool = True,
        check_duplicates: bool = True,
        check_outliers: bool = True,
        check_feature_correlation: bool = True,
        check_target_distribution: bool = True,
    ) -> Dict[str, Any]:
        """Validate a regression dataset and return a flat summary.

        Args:
            X: Feature matrix.
            y: Continuous target.
            check_missing: Run the missing-value check.
            check_duplicates: Run the duplicate-row check.
            check_outliers: Run the outlier check.
            check_feature_correlation: Run the multicollinearity check.
            check_target_distribution: Run target-distribution checks.

        Returns:
            Dictionary with ``validation_status``, ``is_valid``, ``overall_score``,
            ``missing_values_percent``, ``duplicate_rows``, ``target_skewness``,
            ``high_correlation_pairs`` and the full ``report``.
        """
        checks = self._checks_from_flags(
            {
                "missing_values": check_missing,
                "duplicates": check_duplicates,
                "outliers": check_outliers,
                "multicollinearity": check_feature_correlation,
                "target": check_target_distribution,
            }
        )
        report = self.validate_dataset(X, y, task_type="regression", checks=checks)
        summary = self._flat_summary(report)
        y_series = _to_target(y)
        values = _finite_values(y_series) if isinstance(y_series, pd.Series) else np.array([])
        summary["target_skewness"] = float(stats.skew(values)) if values.size >= 3 else float("nan")
        return summary

    def validate_pipeline_input(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        expected_features: Optional[Union[int, Sequence[str]]] = None,
        expected_dtypes: Optional[Mapping[str, Any]] = None,
        allow_missing: bool = False,
    ) -> Tuple[bool, List[str]]:
        """Cheap structural checks for data about to enter a fitted pipeline.

        Args:
            X: Feature matrix.
            y: Optional target.
            expected_features: Expected feature count, or the expected column
                names (order-sensitive for DataFrames).
            expected_dtypes: Mapping ``column -> type spec`` (see :class:`SchemaValidator`).
            allow_missing: Whether missing values are acceptable.

        Returns:
            ``(is_valid, problems)`` where ``problems`` is a list of messages.
        """
        problems: List[str] = []
        X_df = _to_frame(X)
        if X_df.shape[0] == 0 or X_df.shape[1] == 0:
            problems.append(f"X is empty (shape {X_df.shape}).")
            return False, problems

        if isinstance(expected_features, int):
            if X_df.shape[1] != expected_features:
                problems.append(f"Expected {expected_features} features, got {X_df.shape[1]}.")
        elif expected_features is not None:
            expected = [str(c) for c in expected_features]
            actual = [str(c) for c in X_df.columns]
            missing = [c for c in expected if c not in actual]
            extra = [c for c in actual if c not in expected]
            if missing:
                problems.append(f"Missing expected feature(s): {missing}.")
            if extra:
                problems.append(f"Unexpected feature(s): {extra}.")
            if not missing and not extra and actual != expected:
                problems.append("Feature order differs from the expected order.")

        for column, spec in (expected_dtypes or {}).items():
            if column not in X_df.columns:
                problems.append(f"Column '{column}' required by expected_dtypes is missing.")
            elif not _dtype_matches(X_df[column].dtype, spec):
                problems.append(f"Column '{column}' has dtype {X_df[column].dtype}, expected {spec}.")

        if not allow_missing:
            missing_counts = X_df.isna().sum()
            bad = missing_counts[missing_counts > 0]
            if len(bad) > 0:
                problems.append(f"Missing values in {len(bad)} column(s): {[str(c) for c in bad.index]}.")

        for column in _numeric_columns(X_df):
            values = X_df[column].to_numpy(dtype="float64", na_value=np.nan)
            n_inf = int(np.isinf(values).sum())
            if n_inf:
                problems.append(f"Column '{column}' contains {n_inf} infinite value(s).")

        if y is not None:
            y_t = _to_target(y)
            if len(y_t) != len(X_df):
                problems.append(f"X has {len(X_df)} rows but y has {len(y_t)}.")
            elif not allow_missing and int(
                pd.isna(y_t).sum().sum() if isinstance(y_t, pd.DataFrame) else y_t.isna().sum()
            ):
                problems.append("y contains missing values.")

        return len(problems) == 0, problems

    def detect_data_drift(
        self,
        X_reference: ArrayLike,
        X_current: ArrayLike,
        method: str = "ks_test",
        threshold: float = 0.05,
        columns: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Detect distribution drift between a reference and a current dataset.

        Numeric columns use a two-sample Kolmogorov-Smirnov test (``"ks_test"``)
        or the Population Stability Index (``"psi"``).  Categorical columns use a
        chi-square test on category frequencies for ``"ks_test"`` and a
        frequency-based PSI for ``"psi"``.

        Args:
            X_reference: Reference (e.g. training) data.
            X_current: Current (e.g. production) data.
            method: ``"ks_test"`` or ``"psi"``.
            threshold: Significance level for ``"ks_test"`` (drift when p-value
                is below it) or PSI cutoff for ``"psi"`` (drift when PSI exceeds
                it; 0.1 = moderate, 0.25 = severe).
            columns: Columns to compare; defaults to the columns present in both.

        Returns:
            Dictionary with ``drift_detected``, ``n_drifted_features``,
            ``drifted_features``, ``feature_results`` (per-column statistics),
            ``method``, ``threshold`` and ``skipped_features``.

        Raises:
            ValueError: For an unknown method or when no column can be compared.
        """
        if method not in ("ks_test", "psi"):
            raise ValueError(f"method must be 'ks_test' or 'psi', got {method!r}.")
        ref = _to_frame(X_reference, "X_reference")
        cur = _to_frame(X_current, "X_current")
        shared = [c for c in ref.columns if c in cur.columns] if columns is None else list(columns)
        missing = [c for c in shared if c not in ref.columns or c not in cur.columns]
        if missing:
            raise ValueError(f"Column(s) {missing} are not present in both datasets.")
        if not shared:
            raise ValueError("No common columns to compare for drift detection.")

        feature_results: Dict[str, Dict[str, Any]] = {}
        drifted: List[str] = []
        skipped: List[str] = []
        for column in shared:
            key = str(column)
            ref_col, cur_col = ref[column], cur[column]
            numeric = is_numeric_dtype(ref_col) and not is_bool_dtype(ref_col)
            if numeric:
                r, c = _finite_values(ref_col), _finite_values(cur_col)
                if r.size < 2 or c.size < 2:
                    skipped.append(key)
                    continue
                if method == "ks_test":
                    statistic, p_value = stats.ks_2samp(r, c)
                    is_drift = bool(p_value < threshold)
                    result = {"test": "ks_2samp", "statistic": float(statistic), "p_value": float(p_value)}
                else:
                    psi = _psi(r, c)
                    is_drift = bool(psi > threshold)
                    result = {"test": "psi", "statistic": psi, "p_value": None}
                result.update(
                    {
                        "reference_mean": float(r.mean()),
                        "current_mean": float(c.mean()),
                        "reference_std": float(r.std(ddof=1)),
                        "current_std": float(c.std(ddof=1)),
                    }
                )
            else:
                r_cat, c_cat = ref_col.dropna(), cur_col.dropna()
                if r_cat.empty or c_cat.empty:
                    skipped.append(key)
                    continue
                if method == "ks_test":
                    categories = (
                        r_cat.astype(str).value_counts().index.union(c_cat.astype(str).value_counts().index)
                    )
                    table = np.vstack(
                        [
                            r_cat.astype(str).value_counts().reindex(categories, fill_value=0).to_numpy(),
                            c_cat.astype(str).value_counts().reindex(categories, fill_value=0).to_numpy(),
                        ]
                    )
                    if table.shape[1] < 2:
                        statistic, p_value = 0.0, 1.0
                    else:
                        statistic, p_value = stats.chi2_contingency(table)[:2]
                    is_drift = bool(p_value < threshold)
                    result = {
                        "test": "chi2_contingency",
                        "statistic": float(statistic),
                        "p_value": float(p_value),
                    }
                else:
                    psi = _psi_categorical(r_cat.astype(str), c_cat.astype(str))
                    is_drift = bool(psi > threshold)
                    result = {"test": "psi", "statistic": psi, "p_value": None}
                new_levels = sorted(set(c_cat.astype(str)) - set(r_cat.astype(str)))
                result["new_categories"] = new_levels[:20]
            result["drift_detected"] = is_drift
            feature_results[key] = result
            if is_drift:
                drifted.append(key)

        self.logger.info(
            "Drift check (%s): %d/%d feature(s) drifted", method, len(drifted), len(feature_results)
        )
        return {
            "drift_detected": len(drifted) > 0,
            "n_features_compared": len(feature_results),
            "n_drifted_features": len(drifted),
            "drift_fraction": len(drifted) / len(feature_results) if feature_results else 0.0,
            "drifted_features": drifted,
            "feature_results": feature_results,
            "skipped_features": skipped,
            "method": method,
            "threshold": threshold,
        }

    def format_summary(self) -> str:
        """Return a human-readable summary of the last validation run.

        Raises:
            RuntimeError: If :meth:`validate_dataset` has not been called yet.
        """
        if self.report_ is None:
            raise RuntimeError("No validation report available; call validate_dataset() first.")
        report = self.report_
        stats_ = report.dataset_statistics
        lines = [
            "=" * 60,
            "Validation Summary",
            "=" * 60,
            f"Status: {report.validation_status} | {len(report.issues)} issues found",
            f"Dataset: {stats_.get('n_samples', 0)} samples x {stats_.get('n_features', 0)} features"
            + (f" | task: {report.task_type}" if report.task_type else ""),
            f"Quality score: {report.quality_metrics.get('overall_score', float('nan')):.1f}/100",
            "",
            "Severity counts:",
        ]
        counts = report.severity_counts
        for severity in reversed(list(ValidationSeverity)):
            lines.append(f"  {severity.icon} {severity.value:<9} {counts[severity.value]}")
        if report.issues:
            lines.extend(["", "Issues:"])
            for issue in report.issues:
                where = f" [{issue.column}]" if issue.column else ""
                lines.append(f"  {issue.severity.icon} ({issue.category}){where} {issue.message}")
        if report.recommendations:
            lines.extend(["", "Recommendations:"])
            lines.extend(f"  - {rec}" for rec in report.recommendations)
        lines.append("=" * 60)
        return "\n".join(lines)

    def print_summary(self, stream: Optional[TextIO] = None) -> None:
        """Write :meth:`format_summary` to ``stream`` (default: ``sys.stdout``)."""
        target = stream if stream is not None else sys.stdout
        target.write(self.format_summary() + "\n")

    # ------------------------------------------------------------ inference
    @staticmethod
    def infer_task_type(y: Union[pd.Series, pd.DataFrame, ArrayLike], max_classes: int = 20) -> str:
        """Guess ``"classification"`` or ``"regression"`` from a target.

        Non-numeric, boolean or categorical targets are classification.  Numeric
        targets are classification when they take at most ``max_classes``
        integer-valued levels, otherwise regression.
        """
        y_t = _to_target(y)
        if isinstance(y_t, pd.DataFrame):
            y_t = y_t.iloc[:, 0]
        if y_t is None:
            raise ValueError("y is required to infer the task type.")
        if not is_numeric_dtype(y_t) or is_bool_dtype(y_t):
            return "classification"
        values = _finite_values(y_t)
        if values.size == 0:
            return "classification"
        n_unique = np.unique(values).size
        if n_unique <= max_classes and np.all(np.mod(values, 1) == 0):
            return "classification"
        return "regression"

    # --------------------------------------------------------------- checks
    def _resolve_checks(self, checks: Optional[Iterable[str]]) -> set:
        if checks is None:
            return set(self.CHECK_NAMES)
        selected = set(checks)
        unknown = selected - set(self.CHECK_NAMES)
        if unknown:
            raise ValueError(f"Unknown check name(s) {sorted(unknown)}; valid names: {self.CHECK_NAMES}.")
        return selected

    def _checks_from_flags(self, flags: Mapping[str, bool]) -> List[str]:
        return [name for name in self.CHECK_NAMES if flags.get(name, True)]

    def _check_structure(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]],
        issues: List[ValidationIssue],
    ) -> bool:
        """Empty-data and X/y length checks. Returns ``False`` when content checks must be skipped."""
        if X.shape[0] == 0 or X.shape[1] == 0:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.CRITICAL,
                    "structure",
                    f"Empty dataset: X has shape {tuple(X.shape)} (no {'rows' if X.shape[0] == 0 else 'columns'}).",
                    details={"shape": list(X.shape)},
                )
            )
            return False
        if y is not None and len(y) != len(X):
            issues.append(
                ValidationIssue(
                    ValidationSeverity.CRITICAL,
                    "structure",
                    f"X/y length mismatch: X has {len(X)} rows but y has {len(y)} entries.",
                    details={"n_samples_X": len(X), "n_samples_y": len(y)},
                )
            )
        return True

    def _check_sample_size(self, X: pd.DataFrame, issues: List[ValidationIssue]) -> None:
        n, p = X.shape
        if n < 10:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    "sample_size",
                    f"Very small dataset ({n} sample(s)); no reliable model or validation is possible.",
                    details={"n_samples": n, "min_samples": self.min_samples},
                )
            )
        elif n < self.min_samples:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.WARNING,
                    "sample_size",
                    f"Small dataset ({n} samples, fewer than {self.min_samples}); expect high variance in estimates.",
                    details={"n_samples": n, "min_samples": self.min_samples},
                )
            )
        if p > n:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.WARNING,
                    "sample_size",
                    f"More features ({p}) than samples ({n}); high overfitting risk.",
                    details={"n_samples": n, "n_features": p},
                )
            )

    def _check_missing_values(self, X: pd.DataFrame, issues: List[ValidationIssue]) -> None:
        n = len(X)
        warn_t, err_t, crit_t = self.missing_thresholds
        for column, count in X.isna().sum().items():
            if count == 0:
                continue
            fraction = float(count) / n
            if fraction >= crit_t:
                severity = ValidationSeverity.CRITICAL
            elif fraction >= err_t:
                severity = ValidationSeverity.ERROR
            elif fraction >= warn_t:
                severity = ValidationSeverity.WARNING
            else:
                severity = ValidationSeverity.INFO
            issues.append(
                ValidationIssue(
                    severity,
                    "missing_values",
                    f"Column '{column}' has {int(count)} missing values ({fraction:.1%}).",
                    column=str(column),
                    details={"missing_count": int(count), "missing_fraction": fraction},
                )
            )

    def _check_duplicates(self, X: pd.DataFrame, issues: List[ValidationIssue]) -> None:
        try:
            dup_mask = X.duplicated()
        except TypeError:  # unhashable cell values (lists, dicts)
            self.logger.debug("Duplicate check skipped: unhashable values present")
            return
        count = int(dup_mask.sum())
        if count == 0:
            return
        fraction = count / len(X)
        warn_t, err_t = self.duplicate_thresholds
        if fraction >= err_t:
            severity = ValidationSeverity.ERROR
        elif fraction >= warn_t:
            severity = ValidationSeverity.WARNING
        else:
            severity = ValidationSeverity.INFO
        issues.append(
            ValidationIssue(
                severity,
                "duplicates",
                f"Found {count} duplicate rows ({fraction:.1%} of the data); duplicates can leak across splits.",
                details={
                    "duplicate_count": count,
                    "duplicate_fraction": fraction,
                    "duplicate_indices": [int(i) for i in np.flatnonzero(dup_mask.to_numpy())[:50]],
                },
            )
        )

    def _check_infinite_values(self, X: pd.DataFrame, issues: List[ValidationIssue]) -> None:
        for column in _numeric_columns(X):
            values = X[column].to_numpy(dtype="float64", na_value=np.nan)
            count = int(np.isinf(values).sum())
            if count == 0:
                continue
            issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    "infinite_values",
                    f"Column '{column}' contains {count} infinite values ({count / len(X):.1%}).",
                    column=str(column),
                    details={"infinite_count": count, "infinite_fraction": count / len(X)},
                )
            )

    def _check_constant_features(self, X: pd.DataFrame, issues: List[ValidationIssue]) -> None:
        try:
            n_unique = X.nunique(dropna=True)
        except TypeError:
            self.logger.debug("Constant-feature check skipped: unhashable values present")
            return
        constant = [str(c) for c in X.columns if n_unique[c] <= 1]
        if constant:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.WARNING,
                    "constant_features",
                    f"Found {len(constant)} constant feature(s) with no predictive value: {constant}.",
                    details={"constant_features": constant, "n_constant": len(constant)},
                )
            )
        for column in X.columns:
            if n_unique[column] <= 1:
                continue
            non_null = X[column].dropna()
            if non_null.empty:
                continue
            top_fraction = float(non_null.value_counts(normalize=True).iloc[0])
            if top_fraction >= self.low_variance_threshold:
                issues.append(
                    ValidationIssue(
                        ValidationSeverity.INFO,
                        "low_variance",
                        f"Low-variance feature '{column}': a single value covers {top_fraction:.1%} of rows.",
                        column=str(column),
                        details={"dominant_value_fraction": top_fraction, "n_unique": int(n_unique[column])},
                    )
                )

    def _check_outliers(self, X: pd.DataFrame, issues: List[ValidationIssue]) -> None:
        for column in _numeric_columns(X):
            values = _finite_values(X[column])
            if values.size < self.outlier_min_samples:
                continue
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            if iqr <= 0:
                continue
            lower, upper = q1 - self.outlier_iqr_factor * iqr, q3 + self.outlier_iqr_factor * iqr
            count = int(((values < lower) | (values > upper)).sum())
            if count == 0:
                continue
            fraction = count / values.size
            severity = (
                ValidationSeverity.WARNING
                if fraction > self.outlier_fraction_warning
                else ValidationSeverity.INFO
            )
            issues.append(
                ValidationIssue(
                    severity,
                    "outliers",
                    f"Column '{column}' has {count} extreme outlier(s) ({fraction:.1%}) beyond "
                    f"{self.outlier_iqr_factor:g}x IQR fences [{lower:.4g}, {upper:.4g}].",
                    column=str(column),
                    details={
                        "outlier_count": count,
                        "outlier_fraction": fraction,
                        "lower_fence": float(lower),
                        "upper_fence": float(upper),
                        "method": f"iqr_{self.outlier_iqr_factor:g}",
                    },
                )
            )

    def _check_high_cardinality(self, X: pd.DataFrame, issues: List[ValidationIssue]) -> None:
        for column in _categorical_columns(X):
            if is_bool_dtype(X[column]):
                continue
            non_null = X[column].dropna()
            if non_null.empty:
                continue
            try:
                n_unique = int(non_null.nunique())
            except TypeError:
                continue
            ratio = n_unique / len(non_null)
            if n_unique > self.high_cardinality_threshold or (
                ratio > self.high_cardinality_ratio and n_unique > 10
            ):
                identifier = ratio >= 0.99
                message = (
                    f"High cardinality categorical feature '{column}': {n_unique} unique values "
                    f"({ratio:.0%} of rows)"
                )
                message += (
                    "; it looks like an identifier column."
                    if identifier
                    else "; one-hot encoding will explode."
                )
                issues.append(
                    ValidationIssue(
                        ValidationSeverity.WARNING,
                        "high_cardinality",
                        message,
                        column=str(column),
                        details={
                            "n_unique": n_unique,
                            "unique_ratio": ratio,
                            "looks_like_identifier": identifier,
                        },
                    )
                )

    def _check_data_types(self, X: pd.DataFrame, issues: List[ValidationIssue]) -> None:
        for column in X.columns:
            series = X[column]
            if not (is_object_dtype(series) or is_string_dtype(series)):
                continue
            non_null = series.dropna()
            if non_null.empty:
                continue
            if is_object_dtype(series):
                type_names = sorted({type(v).__name__ for v in non_null})
                if len(type_names) > 1:
                    issues.append(
                        ValidationIssue(
                            ValidationSeverity.WARNING,
                            "data_types",
                            f"Column '{column}' mixes Python types {type_names}; clean or cast it to one type.",
                            column=str(column),
                            details={"types": type_names},
                        )
                    )
            try:
                converted = pd.to_numeric(non_null, errors="coerce")
            except (TypeError, ValueError):
                continue
            parseable = float(converted.notna().mean())
            if parseable >= 0.95:
                issues.append(
                    ValidationIssue(
                        ValidationSeverity.WARNING,
                        "data_types",
                        f"Column '{column}' stores numeric data as strings ({parseable:.0%} parseable); "
                        "type conversion to a numeric dtype is recommended.",
                        column=str(column),
                        details={"parseable_fraction": parseable, "suggested_dtype": "float"},
                    )
                )

    def _check_multicollinearity(self, X: pd.DataFrame, issues: List[ValidationIssue]) -> None:
        numeric = [c for c in _numeric_columns(X) if _finite_values(X[c]).size >= 3]
        numeric = [c for c in numeric if np.std(_finite_values(X[c])) > 0]
        if len(numeric) < 2:
            return
        if len(numeric) > self.max_corr_features:
            self.logger.debug("Multicollinearity check skipped: %d numeric features", len(numeric))
            return
        frame = X[numeric].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        corr = frame.corr().abs().to_numpy()
        iu = np.triu_indices_from(corr, k=1)
        values = corr[iu]
        mask = values > self.correlation_threshold
        if not mask.any():
            return
        pairs = [
            {"feature_1": str(numeric[i]), "feature_2": str(numeric[j]), "correlation": float(r)}
            for i, j, r in zip(iu[0][mask], iu[1][mask], values[mask])
        ]
        pairs.sort(key=lambda d: d["correlation"], reverse=True)
        issues.append(
            ValidationIssue(
                ValidationSeverity.WARNING,
                "multicollinearity",
                f"Multicollinearity detected: {len(pairs)} feature pair(s) with |correlation| > "
                f"{self.correlation_threshold} (strongest: '{pairs[0]['feature_1']}' vs '{pairs[0]['feature_2']}', "
                f"r={pairs[0]['correlation']:.3f}).",
                details={"n_pairs": len(pairs), "pairs": pairs[:20], "threshold": self.correlation_threshold},
            )
        )

    def _check_feature_scaling(self, X: pd.DataFrame, issues: List[ValidationIssue]) -> None:
        stds: Dict[str, float] = {}
        for column in _numeric_columns(X):
            values = _finite_values(X[column])
            if values.size >= 2:
                std = float(values.std(ddof=1))
                if np.isfinite(std) and std > 0:
                    stds[str(column)] = std
        if len(stds) < 2:
            return
        min_col = min(stds, key=stds.get)
        max_col = max(stds, key=stds.get)
        ratio = stds[max_col] / stds[min_col]
        if ratio > self.scale_ratio_threshold:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.WARNING,
                    "feature_scaling",
                    f"Features are on very different scales (std ratio {ratio:.1e}: '{max_col}' vs '{min_col}'); "
                    "scale-sensitive models need feature scaling.",
                    details={
                        "std_ratio": ratio,
                        "smallest_scale_feature": min_col,
                        "largest_scale_feature": max_col,
                        "feature_std": stds,
                    },
                )
            )

    def _check_target(
        self,
        y: Union[pd.Series, pd.DataFrame],
        task_type: Optional[str],
        issues: List[ValidationIssue],
    ) -> None:
        if isinstance(y, pd.DataFrame):
            issues.append(
                ValidationIssue(
                    ValidationSeverity.INFO,
                    "target",
                    f"Multi-output target with {y.shape[1]} outputs; per-output target checks were skipped.",
                    details={"n_outputs": int(y.shape[1])},
                )
            )
            return
        n_missing = int(y.isna().sum())
        if n_missing:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    "target",
                    f"Target contains {n_missing} missing values ({n_missing / len(y):.1%}); drop or impute them.",
                    details={"missing_count": n_missing},
                )
            )
        y_valid = y.dropna()
        if y_valid.empty or task_type == "clustering":
            return
        if task_type == "classification":
            self._check_classification_target(y_valid, issues)
        elif task_type == "regression":
            self._check_regression_target(y_valid, issues)

    def _check_classification_target(self, y: pd.Series, issues: List[ValidationIssue]) -> None:
        counts = y.value_counts()
        n_classes = len(counts)
        class_counts = {str(k): int(v) for k, v in counts.items()}
        if n_classes < 2:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.CRITICAL,
                    "target",
                    f"Target has only one class ({counts.index[0]!r}); classification needs at least two classes.",
                    details={"n_classes": n_classes, "class_counts": class_counts},
                )
            )
            return
        if n_classes > len(y) / 2 and len(y) > 10:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.WARNING,
                    "target",
                    f"Target has {n_classes} classes for {len(y)} samples; check that this is really a "
                    "classification task and not a continuous target.",
                    details={"n_classes": n_classes, "n_samples": len(y)},
                )
            )
        ratio = float(counts.min() / counts.max())
        majority, minority = counts.index[0], counts.index[-1]
        if ratio < self.severe_imbalance_ratio:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    "class_balance",
                    f"Severe class imbalance: minority/majority ratio {ratio:.3f} "
                    f"(class {minority!r}: {int(counts.min())} vs class {majority!r}: {int(counts.max())}).",
                    details={"imbalance_ratio": ratio, "class_counts": class_counts},
                )
            )
        elif ratio < self.imbalance_ratio:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.WARNING,
                    "class_balance",
                    f"Class imbalance: minority/majority ratio {ratio:.3f} "
                    f"(class {minority!r}: {int(counts.min())} vs class {majority!r}: {int(counts.max())}).",
                    details={"imbalance_ratio": ratio, "class_counts": class_counts},
                )
            )
        small = {str(k): int(v) for k, v in counts.items() if v < self.min_samples_per_class}
        if small:
            severity = ValidationSeverity.ERROR if min(small.values()) < 2 else ValidationSeverity.WARNING
            issues.append(
                ValidationIssue(
                    severity,
                    "class_balance",
                    f"Insufficient samples in {len(small)} class(es) (fewer than {self.min_samples_per_class}): "
                    f"{small}; stratified cross-validation may fail.",
                    details={"small_classes": small, "min_samples_per_class": self.min_samples_per_class},
                )
            )

    def _check_regression_target(self, y: pd.Series, issues: List[ValidationIssue]) -> None:
        if not is_numeric_dtype(y) or is_bool_dtype(y):
            issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    "target",
                    f"Regression target has non-numeric dtype {y.dtype}; convert it or use classification.",
                    details={"dtype": str(y.dtype)},
                )
            )
            return
        values = _finite_values(y)
        n_inf = int(len(y) - values.size)
        if n_inf:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    "target",
                    f"Regression target contains {n_inf} infinite value(s).",
                    details={"infinite_count": n_inf},
                )
            )
        if values.size == 0:
            return
        n_unique = int(np.unique(values).size)
        if n_unique < 2:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.CRITICAL,
                    "target",
                    "Regression target is constant; nothing to predict.",
                    details={"n_unique": n_unique},
                )
            )
            return
        if n_unique < self.min_unique_regression:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.WARNING,
                    "target",
                    f"Regression target has only {n_unique} unique values; consider treating it as classification.",
                    details={"n_unique": n_unique, "min_unique_regression": self.min_unique_regression},
                )
            )
        if values.size >= 3 and values.std() > 0:
            skewness = float(stats.skew(values))
            if abs(skewness) > self.target_skew_threshold:
                issues.append(
                    ValidationIssue(
                        ValidationSeverity.INFO,
                        "target",
                        f"Regression target is highly skewed (skewness {skewness:.2f}); "
                        "a log or Box-Cox transform may help.",
                        details={"skewness": skewness},
                    )
                )
        if values.size >= self.outlier_min_samples:
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                lower, upper = q1 - self.outlier_iqr_factor * iqr, q3 + self.outlier_iqr_factor * iqr
                count = int(((values < lower) | (values > upper)).sum())
                if count:
                    issues.append(
                        ValidationIssue(
                            ValidationSeverity.INFO,
                            "target",
                            f"Regression target has {count} extreme outlier(s) beyond "
                            f"{self.outlier_iqr_factor:g}x IQR fences; consider robust losses.",
                            details={
                                "outlier_count": count,
                                "lower_fence": float(lower),
                                "upper_fence": float(upper),
                            },
                        )
                    )

    def _check_task_requirements(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]],
        task_type: Optional[str],
        issues: List[ValidationIssue],
    ) -> None:
        if task_type is None:
            return
        n_numeric = len(_numeric_columns(X))
        n_categorical = len(_categorical_columns(X))
        if task_type == "clustering":
            if n_numeric == 0:
                issues.append(
                    ValidationIssue(
                        ValidationSeverity.ERROR,
                        "task_requirements",
                        "Clustering requires numerical features but none were found; "
                        "encode categorical columns before clustering.",
                        details={"n_numeric_features": n_numeric, "n_categorical_features": n_categorical},
                    )
                )
            elif n_categorical > 0:
                issues.append(
                    ValidationIssue(
                        ValidationSeverity.INFO,
                        "task_requirements",
                        f"Clustering on mixed types: {n_categorical} categorical column(s) must be encoded "
                        "(or use a distance that supports them) before distance-based clustering.",
                        details={"n_numeric_features": n_numeric, "n_categorical_features": n_categorical},
                    )
                )
            if y is not None:
                issues.append(
                    ValidationIssue(
                        ValidationSeverity.INFO,
                        "task_requirements",
                        "A target was provided for clustering; it is only useful for external evaluation.",
                    )
                )
        elif y is None:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.ERROR,
                    "task_requirements",
                    f"Supervised task '{task_type}' requires a target, but y was not provided.",
                    details={"task_type": task_type},
                )
            )
        elif n_categorical > 0:
            issues.append(
                ValidationIssue(
                    ValidationSeverity.INFO,
                    "task_requirements",
                    f"{n_categorical} categorical column(s) need encoding before most sklearn estimators can fit.",
                    details={"categorical_features": [str(c) for c in _categorical_columns(X)][:50]},
                )
            )

    def _run_custom_rules(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]],
        issues: List[ValidationIssue],
    ) -> None:
        for name, rule in self.custom_rules.items():
            result = rule(X, y)
            if result is None:
                continue
            produced = [result] if isinstance(result, ValidationIssue) else list(result)
            for item in produced:
                if not isinstance(item, ValidationIssue):
                    raise TypeError(
                        f"Custom rule '{name}' must return ValidationIssue objects, got {type(item).__name__}."
                    )
                item.details.setdefault("rule", name)
                issues.append(item)

    # ---------------------------------------------------------- reporting
    def _determine_status(self, issues: List[ValidationIssue]) -> str:
        if not issues:
            return "PASSED"
        worst = max(issue.severity.rank for issue in issues)
        fail_rank = ValidationSeverity.WARNING.rank if self.strict_mode else ValidationSeverity.ERROR.rank
        if worst >= fail_rank:
            return "FAILED"
        if worst >= ValidationSeverity.WARNING.rank:
            return "PASSED_WITH_WARNINGS"
        return "PASSED"

    def _dataset_statistics(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]],
        task_type: Optional[str],
    ) -> Dict[str, Any]:
        n, p = X.shape
        n_cells = n * p
        missing_cells = int(X.isna().sum().sum()) if n_cells else 0
        try:
            duplicate_rows = int(X.duplicated().sum()) if n_cells else 0
        except TypeError:
            duplicate_rows = -1
        stats_: Dict[str, Any] = {
            "n_samples": int(n),
            "n_features": int(p),
            "n_numeric_features": len(_numeric_columns(X)),
            "n_categorical_features": len(_categorical_columns(X)),
            "n_datetime_features": len(_datetime_columns(X)),
            "missing_cells": missing_cells,
            "missing_fraction": missing_cells / n_cells if n_cells else 0.0,
            "duplicate_rows": duplicate_rows,
            "memory_usage_mb": float(X.memory_usage(deep=True).sum() / 1e6) if p else 0.0,
            "feature_names": [str(c) for c in X.columns[:100]],
            "task_type": task_type,
        }
        if y is None:
            stats_["target"] = None
        elif isinstance(y, pd.DataFrame):
            stats_["target"] = {"n_samples": len(y), "n_outputs": int(y.shape[1])}
        else:
            target: Dict[str, Any] = {
                "n_samples": len(y),
                "dtype": str(y.dtype),
                "n_missing": int(y.isna().sum()),
            }
            y_valid = y.dropna()
            try:
                target["n_unique"] = int(y_valid.nunique())
            except TypeError:
                target["n_unique"] = None
            if task_type == "classification" and target["n_unique"] is not None and target["n_unique"] <= 50:
                target["class_counts"] = {str(k): int(v) for k, v in y_valid.value_counts().items()}
            if is_numeric_dtype(y_valid) and not is_bool_dtype(y_valid) and len(y_valid):
                values = _finite_values(y_valid)
                if values.size:
                    target.update(
                        {
                            "mean": float(values.mean()),
                            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                            "min": float(values.min()),
                            "max": float(values.max()),
                        }
                    )
            stats_["target"] = target
        return stats_

    def _quality_metrics(self, X: pd.DataFrame, issues: List[ValidationIssue]) -> Dict[str, float]:
        n, p = X.shape
        n_cells = n * p
        completeness = 1.0 - (float(X.isna().sum().sum()) / n_cells) if n_cells else 0.0
        try:
            uniqueness = 1.0 - float(X.duplicated().sum()) / n if n_cells else 0.0
        except TypeError:
            uniqueness = float("nan")
        numeric = _numeric_columns(X)
        if numeric and n:
            arr = X[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
            validity = 1.0 - float(np.isinf(arr).sum()) / arr.size
        else:
            validity = 1.0 if n_cells else 0.0
        n_constant = sum(
            issue.details.get("n_constant", 0) for issue in issues if issue.category == "constant_features"
        )
        usefulness = 1.0 - n_constant / p if p else 0.0
        penalty = sum(_SEVERITY_PENALTY[issue.severity] for issue in issues)
        overall = float(np.clip(100.0 - penalty, 0.0, 100.0))
        return {
            "completeness": float(completeness),
            "uniqueness": float(uniqueness),
            "validity": float(validity),
            "feature_usefulness": float(usefulness),
            "overall_score": overall,
        }

    @staticmethod
    def _recommendations(issues: List[ValidationIssue]) -> List[str]:
        categories = {issue.category for issue in issues}
        recs: List[str] = []
        if "structure" in categories:
            recs.append("Fix the dataset structure first: provide non-empty X and a y of matching length.")
        if "sample_size" in categories:
            recs.append(
                "Sample size: collect more data, prefer simpler/regularised models and use repeated CV."
            )
        if "missing_values" in categories:
            recs.append(
                "Data cleaning: impute missing values (SimpleImputer/KNNImputer) or drop columns that are mostly empty."
            )
        if "duplicates" in categories:
            recs.append("Data cleaning: remove duplicate rows before splitting to avoid train/test leakage.")
        if "infinite_values" in categories:
            recs.append(
                "Data cleaning: replace infinite values (np.inf) with NaN or clip them before imputation."
            )
        if "outliers" in categories:
            recs.append(
                "Data cleaning: inspect extreme outliers; winsorise, clip, or use robust scalers/losses."
            )
        if "constant_features" in categories or "low_variance" in categories:
            recs.append("Feature selection: drop constant/low-variance features (VarianceThreshold).")
        if "high_cardinality" in categories:
            recs.append(
                "Encoding: use target/frequency/hash encoding for high-cardinality categoricals; drop identifier columns."
            )
        if "data_types" in categories:
            recs.append(
                "Type conversion: cast numeric-looking string columns with pd.to_numeric and fix mixed types."
            )
        if "multicollinearity" in categories:
            recs.append(
                "Feature selection: remove one feature from each highly correlated pair, or use PCA/regularisation."
            )
        if "feature_scaling" in categories:
            recs.append(
                "Preprocessing: apply StandardScaler/RobustScaler before scale-sensitive models (SVM, kNN, NN)."
            )
        if "class_balance" in categories:
            recs.append(
                "Class balance: use class_weight='balanced', resampling (SMOTE), stratified splits and "
                "imbalance-aware metrics (F1, balanced accuracy, PR-AUC)."
            )
        if "target" in categories:
            recs.append(
                "Target: review the target definition, its missing values and distribution before modelling."
            )
        if "task_requirements" in categories:
            recs.append("Task setup: encode categorical features and supply a target for supervised tasks.")
        if "custom" in categories or any(issue.details.get("rule") for issue in issues):
            recs.append("Custom rules: address the findings reported by the registered custom rules.")
        if not recs and issues:
            recs.append("Review the reported issues; no automated recommendation is available.")
        return recs

    @staticmethod
    def _flat_summary(report: Dict[str, Any]) -> Dict[str, Any]:
        stats_ = report["dataset_statistics"]
        pairs: List[Dict[str, Any]] = []
        for issue in report["issues_by_category"].get("multicollinearity", []):
            pairs.extend(issue["details"].get("pairs", []))
        return {
            "validation_status": report["validation_status"],
            "is_valid": report["is_valid"],
            "overall_score": report["quality_metrics"]["overall_score"],
            "missing_values_percent": 100.0 * stats_["missing_fraction"],
            "duplicate_rows": stats_["duplicate_rows"],
            "high_correlation_pairs": pairs,
            "n_issues": report["total_issues"],
            "severity_counts": report["severity_counts"],
            "recommendations": report["recommendations"],
            "report": report,
        }


# --------------------------------------------------------------------------- #
# SchemaValidator
# --------------------------------------------------------------------------- #
class SchemaValidator(LoggerMixin):
    """Validate a DataFrame against a declarative schema.

    Supported schema keys (all optional):

    * ``required_columns``: list of column names that must be present.
    * ``column_types``: mapping ``column -> type spec`` where the spec is an
      alias (``"int"``, ``"float"``, ``"numeric"``, ``"object"``/``"string"``,
      ``"bool"``, ``"datetime"``, ``"category"``), a numpy/pandas dtype or a
      Python type.
    * ``value_ranges``: mapping ``column -> (min, max)``; ``None`` leaves a bound open.
    * ``allowed_values``: mapping ``column -> iterable`` of permitted values.
    * ``non_nullable``: list of columns that must not contain missing values.

    Args:
        schema: Schema dictionary.

    Raises:
        ValueError: If the schema contains unknown keys or malformed ranges.

    Example:
        >>> validator = SchemaValidator({"required_columns": ["a"], "value_ranges": {"a": (0, 1)}})
        >>> validator.validate(pd.DataFrame({"a": [0.2, 0.9]}))
        True
    """

    SUPPORTED_KEYS: Tuple[str, ...] = (
        "required_columns",
        "column_types",
        "value_ranges",
        "allowed_values",
        "non_nullable",
    )

    def __init__(self, schema: Optional[Mapping[str, Any]] = None) -> None:
        self.schema: Dict[str, Any] = dict(schema or {})
        unknown = set(self.schema) - set(self.SUPPORTED_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown schema key(s) {sorted(unknown)}; supported keys: {self.SUPPORTED_KEYS}."
            )
        for column, bounds in (self.schema.get("value_ranges") or {}).items():
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError(f"value_ranges['{column}'] must be a (min, max) pair, got {bounds!r}.")
        self.validation_errors_: List[str] = []

    @classmethod
    def infer_schema(cls, X: ArrayLike, include_ranges: bool = True) -> Dict[str, Any]:
        """Build a schema describing ``X`` (all columns required, dtypes and numeric ranges).

        Args:
            X: Reference data (typically the training set).
            include_ranges: Record ``(min, max)`` for numeric columns.

        Returns:
            Schema dictionary usable with :class:`SchemaValidator`.
        """
        X_df = _to_frame(X)
        column_types: Dict[str, str] = {}
        value_ranges: Dict[str, Tuple[float, float]] = {}
        for column in X_df.columns:
            dtype = X_df[column].dtype
            if is_bool_dtype(dtype):
                column_types[str(column)] = "bool"
            elif is_integer_dtype(dtype):
                column_types[str(column)] = "int"
            elif is_float_dtype(dtype):
                column_types[str(column)] = "float"
            elif is_datetime64_any_dtype(dtype):
                column_types[str(column)] = "datetime"
            elif isinstance(dtype, CategoricalDtype):
                column_types[str(column)] = "category"
            else:
                column_types[str(column)] = "object"
            if include_ranges and column_types[str(column)] in ("int", "float"):
                values = _finite_values(X_df[column])
                if values.size:
                    value_ranges[str(column)] = (float(values.min()), float(values.max()))
        schema: Dict[str, Any] = {
            "required_columns": [str(c) for c in X_df.columns],
            "column_types": column_types,
        }
        if value_ranges:
            schema["value_ranges"] = value_ranges
        return schema

    def validate(self, X: ArrayLike) -> bool:
        """Validate ``X`` against the schema.

        Args:
            X: DataFrame (or array, wrapped with ``feature_<i>`` names).

        Returns:
            ``True`` when no constraint is violated.  Violations are available
            through :meth:`get_validation_errors`.
        """
        self.validation_errors_ = []
        X_df = _to_frame(X)
        errors = self.validation_errors_

        required = [str(c) for c in (self.schema.get("required_columns") or [])]
        present = {str(c) for c in X_df.columns}
        missing_required = [c for c in required if c not in present]
        for column in missing_required:
            errors.append(f"Required column '{column}' is missing.")
        reported_missing = set(missing_required)

        for column, spec in (self.schema.get("column_types") or {}).items():
            if str(column) in reported_missing:
                continue
            if column not in X_df.columns:
                errors.append(f"Column '{column}' referenced in column_types is missing.")
                reported_missing.add(str(column))
            elif not _dtype_matches(X_df[column].dtype, spec):
                errors.append(f"Column '{column}' has dtype {X_df[column].dtype}, expected {spec!r}.")

        for column, (lower, upper) in (self.schema.get("value_ranges") or {}).items():
            if str(column) in reported_missing:
                continue
            if column not in X_df.columns:
                errors.append(f"Column '{column}' referenced in value_ranges is missing.")
                reported_missing.add(str(column))
                continue
            series = X_df[column]
            if not is_numeric_dtype(series) or is_bool_dtype(series):
                errors.append(
                    f"Column '{column}' has a value range but is not numeric (dtype {series.dtype})."
                )
                continue
            values = pd.to_numeric(series, errors="coerce").dropna()
            if values.empty:
                continue
            mask = pd.Series(False, index=values.index)
            if lower is not None:
                mask |= values < lower
            if upper is not None:
                mask |= values > upper
            n_bad = int(mask.sum())
            if n_bad:
                lo = "-inf" if lower is None else f"{lower}"
                hi = "inf" if upper is None else f"{upper}"
                errors.append(
                    f"Column '{column}' has {n_bad} value(s) outside range [{lo}, {hi}] "
                    f"(observed min={values.min()}, max={values.max()})."
                )

        for column, allowed in (self.schema.get("allowed_values") or {}).items():
            if str(column) in reported_missing:
                continue
            if column not in X_df.columns:
                errors.append(f"Column '{column}' referenced in allowed_values is missing.")
                reported_missing.add(str(column))
                continue
            allowed_set = set(allowed)
            observed = X_df[column].dropna()
            invalid = sorted({str(v) for v in observed if v not in allowed_set})
            if invalid:
                errors.append(
                    f"Column '{column}' contains {len(invalid)} disallowed value(s): {invalid[:10]}."
                )

        for column in self.schema.get("non_nullable") or []:
            if str(column) in reported_missing:
                continue
            if column not in X_df.columns:
                errors.append(f"Column '{column}' referenced in non_nullable is missing.")
                reported_missing.add(str(column))
                continue
            n_null = int(X_df[column].isna().sum())
            if n_null:
                errors.append(f"Column '{column}' must not contain missing values but has {n_null}.")

        if errors:
            self.logger.info("Schema validation failed with %d error(s)", len(errors))
        return not errors

    def validate_or_raise(self, X: ArrayLike) -> None:
        """Validate ``X`` and raise ``ValueError`` listing every violation."""
        if not self.validate(X):
            raise ValueError("Schema validation failed:\n- " + "\n- ".join(self.validation_errors_))

    def get_validation_errors(self) -> List[str]:
        """Return the error messages from the most recent :meth:`validate` call."""
        return list(self.validation_errors_)


__all__ = [
    "DataValidator",
    "SchemaValidator",
    "ValidationIssue",
    "ValidationReport",
    "ValidationSeverity",
]
