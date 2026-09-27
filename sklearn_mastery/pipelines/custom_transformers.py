"""Custom scikit-learn compatible transformers for feature-engineering pipelines.

Every transformer in this module follows the scikit-learn estimator contract:

* constructor arguments are stored unchanged, so ``sklearn.base.clone`` works;
* everything learned in :meth:`fit` lives in attributes with a trailing
  underscore and :meth:`fit` returns ``self``;
* :meth:`transform` never mutates its input;
* :meth:`get_feature_names_out` reports the produced feature names.

DataFrame inputs produce DataFrame outputs (column names are preserved or
derived from the input names); ``ndarray`` inputs produce ``ndarray`` outputs
with synthetic ``x0, x1, ...`` names used for bookkeeping.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_array, check_is_fitted

from sklearn_mastery.config.logging_config import LoggerMixin

__all__ = [
    "AdvancedImputer",
    "BinningTransformer",
    "CategoricalEncoder",
    "CustomScaler",
    "DataValidator",
    "DateTimeTransformer",
    "DomainSpecificEncoder",
    "FeatureInteractionCreator",
    "FeatureScaler",
    "FeatureSelector",
    "FeatureUnion",
    "MissingValueHandler",
    "NumericTransformer",
    "OutlierRemover",
    "PipelineDebugger",
    "PolynomialFeatureCreator",
    "TargetEncoder",
    "TextFeatureExtractor",
    "TextTransformer",
    "TimeSeriesFeatureCreator",
]

ArrayLike = Union[np.ndarray, pd.DataFrame]
ColumnKey = Union[str, int]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _column_kind(series: pd.Series) -> str:
    """Classify a column as ``'bool'``, ``'datetime'``, ``'numeric'`` or ``'categorical'``."""
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _columns_of_kind(df: pd.DataFrame, kinds: Iterable[str]) -> List[str]:
    """Return the columns of ``df`` whose kind is in ``kinds`` (input order preserved)."""
    wanted = set(kinds)
    return [col for col in df.columns if _column_kind(df[col]) in wanted]


def _check_columns(df: pd.DataFrame, columns: Iterable[ColumnKey], owner: str) -> List[ColumnKey]:
    """Validate that ``columns`` exist in ``df``.

    Raises:
        KeyError: If any requested column is absent.
    """
    columns = list(columns)
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"{owner}: columns not found in input: {missing}")
    return columns


def _check_numeric_columns(df: pd.DataFrame, columns: Iterable[ColumnKey], owner: str) -> List[ColumnKey]:
    """Validate that ``columns`` exist and hold numeric (or boolean) data.

    Raises:
        KeyError: If any column is absent.
        TypeError: If any column is not numeric.
    """
    columns = _check_columns(df, columns, owner)
    bad = [col for col in columns if _column_kind(df[col]) not in ("numeric", "bool")]
    if bad:
        dtypes = {col: str(df[col].dtype) for col in bad}
        raise TypeError(f"{owner}: columns must be numeric, got non-numeric dtypes {dtypes}")
    return columns


def _require_dataframe(X: Any, owner: str) -> pd.DataFrame:
    """Return ``X`` if it is a DataFrame, otherwise raise ``TypeError``."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"{owner} requires a pandas DataFrame input, got {type(X).__name__}")
    return X


def _iqr_bounds(values: np.ndarray, factor: float) -> Tuple[np.ndarray, np.ndarray]:
    """Per-column Tukey fences ``(Q1 - factor * IQR, Q3 + factor * IQR)`` (NaN-aware)."""
    q1, q3 = np.nanpercentile(values, [25, 75], axis=0)
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr


def _infer_task(y: np.ndarray) -> str:
    """Infer ``'classification'`` or ``'regression'`` from a target vector.

    Raises:
        ValueError: If the target type is not supported.
    """
    target_type = type_of_target(y)
    if target_type in ("binary", "multiclass"):
        return "classification"
    if target_type == "continuous":
        return "regression"
    raise ValueError(f"Unsupported target type '{target_type}'")


def _target_mean_mapping(
    categories: pd.Series, y: np.ndarray, smoothing: float
) -> Tuple[Dict[Any, float], float]:
    """Smoothed per-category target means.

    Args:
        categories: Categorical values (missing values are ignored).
        y: Numeric target aligned positionally with ``categories``.
        smoothing: Pseudo-count ``m`` in ``(sum + m * prior) / (count + m)``.

    Returns:
        Tuple of ``(mapping, prior)`` where ``prior`` is the global target mean.
    """
    prior = float(np.mean(y))
    frame = pd.DataFrame(
        {"category": categories.to_numpy(dtype=object), "target": np.asarray(y, dtype=float)}
    )
    grouped = (
        frame.dropna(subset=["category"]).groupby("category", sort=False)["target"].agg(["sum", "count"])
    )
    encoded = (grouped["sum"] + smoothing * prior) / (grouped["count"] + smoothing)
    return encoded.to_dict(), prior


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise division that yields ``0`` where the denominator is ``0``."""
    out = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out


class _BaseTransformer(TransformerMixin, BaseEstimator, LoggerMixin):
    """Shared plumbing for the transformers in this module.

    Subclasses set ``n_features_in_`` (and ``feature_names_in_`` for DataFrame
    input) through :meth:`_record_input` and ``feature_names_out_`` at the end
    of ``fit``.
    """

    def _to_frame(self, X: Any, use_fitted_names: bool = True) -> pd.DataFrame:
        """View ``X`` as a DataFrame without copying DataFrame input.

        ``ndarray`` input is named ``x0, x1, ...`` unless the transformer was
        fitted on a DataFrame with a matching number of columns.
        """
        if isinstance(X, pd.DataFrame):
            return X
        array = check_array(X, dtype=None, ensure_all_finite=False)
        names_in = getattr(self, "feature_names_in_", None) if use_fitted_names else None
        if names_in is not None and len(names_in) == array.shape[1]:
            names: List[ColumnKey] = list(names_in)
        else:
            names = [f"x{i}" for i in range(array.shape[1])]
        return pd.DataFrame(array, columns=names)

    def _record_input(self, X: Any, frame: pd.DataFrame) -> None:
        """Store ``n_features_in_`` and, for DataFrame input, ``feature_names_in_``."""
        self.n_features_in_ = int(frame.shape[1])
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            delattr(self, "feature_names_in_")

    @staticmethod
    def _like_input(result: pd.DataFrame, X: Any) -> ArrayLike:
        """Return ``result`` as a DataFrame if ``X`` was one, else as an ``ndarray``."""
        return result if isinstance(X, pd.DataFrame) else result.to_numpy()

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "n_features_in_")

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        """Output feature names.

        Args:
            input_features: Ignored; kept for API compatibility.

        Returns:
            Object array of output feature names.
        """
        check_is_fitted(self)
        return np.asarray([str(name) for name in self.feature_names_out_], dtype=object)


# --------------------------------------------------------------------------- #
# Row filtering
# --------------------------------------------------------------------------- #
class OutlierRemover(_BaseTransformer):
    """Detect outlying rows and drop them in :meth:`transform`.

    Args:
        method: ``'iqr'`` (Tukey fences), ``'zscore'``, ``'isolation_forest'`` or ``'lof'``.
        threshold: Strictness of the statistical methods: the IQR multiplier
            for ``'iqr'`` (default ``1.5``) or the absolute z-score cut-off for
            ``'zscore'`` (default ``3.0``). Ignored by the model-based methods.
        contamination: Expected outlier fraction for ``'isolation_forest'`` and ``'lof'``.
        n_neighbors: Neighbourhood size for ``'lof'``.
        random_state: Seed for ``'isolation_forest'``.
        columns: Numeric columns to inspect (DataFrame input). Defaults to all
            numeric columns; non-numeric columns are passed through untouched.

    Attributes:
        outlier_bounds_: ``(lower, upper)`` per-feature bounds for the statistical
            methods, ``None`` for the model-based ones (and before fitting).
        threshold_: Effective threshold.
        columns_: Inspected columns.
        detector_: Fitted ``IsolationForest``/``LocalOutlierFactor`` or ``None``.
        outlier_mask_: Boolean mask over the training rows (``True`` = outlier).
        n_outliers_: Number of outliers in the training data.

    Note:
        ``transform`` removes rows, so the target is not filtered when the
        transformer sits in front of a supervised estimator inside a
        ``Pipeline``. Use it on the training data before model fitting.
    """

    _DEFAULT_THRESHOLDS = {"iqr": 1.5, "zscore": 3.0}
    _METHOD_ALIASES = {
        "z_score": "zscore",
        "isolationforest": "isolation_forest",
        "local_outlier_factor": "lof",
    }
    _METHODS = ("iqr", "zscore", "isolation_forest", "lof")

    def __init__(
        self,
        method: str = "iqr",
        threshold: Optional[float] = None,
        contamination: float = 0.1,
        n_neighbors: int = 20,
        random_state: Optional[int] = 42,
        columns: Optional[List[ColumnKey]] = None,
    ):
        self.method = method
        self.threshold = threshold
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.columns = columns

    @property
    def outlier_bounds_(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Per-feature ``(lower, upper)`` bounds learned by the statistical methods."""
        return getattr(self, "_outlier_bounds", None)

    def _resolved_method(self) -> str:
        method = str(self.method).lower()
        method = self._METHOD_ALIASES.get(method, method)
        if method not in self._METHODS:
            raise ValueError(
                f"Unknown outlier detection method '{self.method}'. Choose from {self._METHODS}."
            )
        return method

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> OutlierRemover:
        """Learn the outlier bounds or fit the outlier model.

        Args:
            X: Feature matrix.
            y: Ignored.

        Returns:
            Fitted transformer.

        Raises:
            ValueError: If no numeric columns are available or the method is unknown.
        """
        method = self._resolved_method()
        frame = self._to_frame(X, use_fitted_names=False)
        self._record_input(X, frame)

        if self.columns is not None:
            columns = _check_numeric_columns(frame, self.columns, "OutlierRemover")
        else:
            columns = _columns_of_kind(frame, ("numeric", "bool"))
        if not columns:
            raise ValueError("OutlierRemover needs at least one numeric column.")
        values = frame[columns].to_numpy(dtype=float)

        self.method_ = method
        self.columns_ = columns
        self.threshold_ = (
            self.threshold if self.threshold is not None else self._DEFAULT_THRESHOLDS.get(method)
        )
        self.detector_ = None
        self._outlier_bounds = None

        if method == "iqr":
            self._outlier_bounds = _iqr_bounds(values, float(self.threshold_))
        elif method == "zscore":
            mean = np.nanmean(values, axis=0)
            std = np.nanstd(values, axis=0)
            self._outlier_bounds = (mean - self.threshold_ * std, mean + self.threshold_ * std)
        elif method == "isolation_forest":
            self.detector_ = IsolationForest(contamination=self.contamination, random_state=self.random_state)
            self.detector_.fit(values)
        else:  # lof
            n_neighbors = max(1, min(self.n_neighbors, len(values) - 1))
            self.detector_ = LocalOutlierFactor(
                n_neighbors=n_neighbors, contamination=self.contamination, novelty=True
            )
            self.detector_.fit(values)

        self.outlier_mask_ = self._flag(values)
        self.n_outliers_ = int(self.outlier_mask_.sum())
        self.feature_names_out_ = list(frame.columns)
        self.logger.info("Detected %d outliers in %d rows using %s", self.n_outliers_, len(values), method)
        return self

    def _flag(self, values: np.ndarray) -> np.ndarray:
        if self.detector_ is not None:
            return self.detector_.predict(values) == -1
        lower, upper = self._outlier_bounds
        with np.errstate(invalid="ignore"):
            return np.any((values < lower) | (values > upper), axis=1)

    def get_outlier_mask(self, X: ArrayLike) -> np.ndarray:
        """Flag the outlying rows of ``X`` using the fitted state.

        Args:
            X: Feature matrix.

        Returns:
            Boolean array with ``True`` for rows considered outliers.
        """
        check_is_fitted(self)
        frame = self._to_frame(X)
        _check_columns(frame, self.columns_, "OutlierRemover")
        return self._flag(frame[self.columns_].to_numpy(dtype=float))

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Drop the rows flagged as outliers.

        Args:
            X: Feature matrix.

        Returns:
            ``X`` without the outlying rows (same container type as the input).
        """
        keep = ~self.get_outlier_mask(X)
        self.logger.debug(
            "%s: removing %d of %d rows", self.__class__.__name__, int((~keep).sum()), len(keep)
        )
        if isinstance(X, pd.DataFrame):
            return X[keep]
        return np.asarray(X)[keep]


# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #
class FeatureSelector(_BaseTransformer):
    """Select numeric features with univariate, model-based or variance criteria.

    Args:
        method: ``'univariate'`` (``SelectKBest``), ``'rfe'``, ``'from_model'``
            (``SelectFromModel``) or ``'variance_threshold'``.
        k: Number of features to keep. ``None`` keeps every feature that passes
            the criterion; values larger than the number of candidates keep all.
        threshold: Variance cut-off for ``'variance_threshold'`` (default ``0.0``)
            or importance cut-off for ``'from_model'``.
        score_func: Scoring function for ``'univariate'``; defaults to
            ``f_classif``/``f_regression`` depending on the target.
        estimator: Estimator for ``'rfe'``/``'from_model'``. Defaults to a linear
            model for RFE and a random forest for ``from_model``.
        task: ``'auto'``, ``'classification'`` or ``'regression'``.
        step: Features removed per RFE iteration.
        random_state: Seed for the default estimators.

    Attributes:
        selected_features_: Positional indices (into the input columns) of the kept features.
        selected_columns_: Names of the kept features.
        scores_: Per-input-feature score (``NaN`` for non-numeric columns that were skipped).
        support_: Boolean mask over the numeric candidate columns.
        selector_: The underlying scikit-learn selector.
    """

    _METHODS = ("univariate", "rfe", "from_model", "variance_threshold")

    def __init__(
        self,
        method: str = "univariate",
        k: Optional[int] = 10,
        threshold: Optional[float] = None,
        score_func: Optional[Callable] = None,
        estimator: Optional[BaseEstimator] = None,
        task: str = "auto",
        step: Union[int, float] = 1,
        random_state: Optional[int] = 42,
    ):
        self.method = method
        self.k = k
        self.threshold = threshold
        self.score_func = score_func
        self.estimator = estimator
        self.task = task
        self.step = step
        self.random_state = random_state

    @property
    def selected_features_(self) -> Optional[List[int]]:
        """Positional indices of the selected input features (``None`` before fitting)."""
        return getattr(self, "_selected_features", None)

    def _resolve_task(self, y: Optional[np.ndarray]) -> str:
        if self.task != "auto":
            if self.task not in ("classification", "regression"):
                raise ValueError("task must be 'auto', 'classification' or 'regression'")
            return self.task
        return "unsupervised" if y is None else _infer_task(y)

    def _default_estimator(self, task: str) -> BaseEstimator:
        if self.method == "rfe":
            if task == "classification":
                return LogisticRegression(max_iter=1000, random_state=self.random_state)
            return LinearRegression()
        if task == "classification":
            return RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        return RandomForestRegressor(n_estimators=100, random_state=self.random_state)

    @staticmethod
    def _importances(estimator: BaseEstimator, n_features: int) -> np.ndarray:
        if hasattr(estimator, "feature_importances_"):
            return np.asarray(estimator.feature_importances_, dtype=float)
        if hasattr(estimator, "coef_"):
            return np.abs(np.atleast_2d(estimator.coef_)).sum(axis=0).astype(float)
        return np.full(n_features, np.nan)

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> FeatureSelector:
        """Score the numeric columns and choose the ones to keep.

        Args:
            X: Feature matrix. Non-numeric columns are ignored (and dropped on transform).
            y: Target; required for every method except ``'variance_threshold'``.

        Returns:
            Fitted selector.

        Raises:
            ValueError: If the method is unknown, ``y`` is missing or no numeric columns exist.
        """
        if self.method not in self._METHODS:
            raise ValueError(
                f"Unknown feature selection method '{self.method}'. Choose from {self._METHODS}."
            )
        frame = self._to_frame(X, use_fitted_names=False)
        self._record_input(X, frame)

        candidates = _columns_of_kind(frame, ("numeric", "bool"))
        if not candidates:
            raise ValueError("FeatureSelector: the input has no numeric columns to select from.")
        skipped = [col for col in frame.columns if col not in candidates]
        if skipped:
            self.logger.warning(
                "FeatureSelector: ignoring %d non-numeric column(s): %s", len(skipped), skipped
            )
        values = frame[candidates].to_numpy(dtype=float)
        n_candidates = values.shape[1]

        if self.method != "variance_threshold" and y is None:
            raise ValueError(f"FeatureSelector(method='{self.method}') requires a target vector y.")
        y_array = None if y is None else np.asarray(y)
        task = self._resolve_task(y_array)

        k: Optional[int] = None
        if self.k is not None:
            k = int(min(self.k, n_candidates))
            if k < self.k:
                self.logger.warning(
                    "FeatureSelector: k=%d exceeds %d candidate features; keeping all", self.k, k
                )

        if self.method == "univariate":
            score_func = self.score_func or (f_classif if task == "classification" else f_regression)
            selector = SelectKBest(score_func=score_func, k=k if k is not None else "all").fit(
                values, y_array
            )
            scores = np.asarray(selector.scores_, dtype=float)
            support = selector.get_support()
        elif self.method == "rfe":
            estimator = self.estimator if self.estimator is not None else self._default_estimator(task)
            selector = RFE(clone(estimator), n_features_to_select=k, step=self.step).fit(values, y_array)
            scores = 1.0 / np.asarray(selector.ranking_, dtype=float)
            support = selector.get_support()
        elif self.method == "from_model":
            estimator = self.estimator if self.estimator is not None else self._default_estimator(task)
            threshold = self.threshold if self.threshold is not None else -np.inf
            selector = SelectFromModel(clone(estimator), max_features=k, threshold=threshold).fit(
                values, y_array
            )
            scores = self._importances(selector.estimator_, n_candidates)
            support = selector.get_support()
        else:
            selector = VarianceThreshold(threshold=self.threshold if self.threshold is not None else 0.0).fit(
                values
            )
            scores = np.asarray(selector.variances_, dtype=float)
            support = selector.get_support()
            if k is not None and support.sum() > k:
                ranked = np.argsort(-np.where(support, scores, -np.inf), kind="stable")[:k]
                support = np.zeros(n_candidates, dtype=bool)
                support[ranked] = True

        positions = [int(frame.columns.get_loc(col)) for col in candidates]
        full_scores = np.full(frame.shape[1], np.nan)
        full_scores[positions] = scores

        self.task_ = task
        self.selector_ = selector
        self.support_ = np.asarray(support, dtype=bool)
        self.scores_ = full_scores
        chosen = np.flatnonzero(self.support_)
        self._selected_features = [positions[i] for i in chosen]
        self.selected_columns_ = [candidates[i] for i in chosen]
        self.feature_names_out_ = list(self.selected_columns_)
        self.logger.info("FeatureSelector(%s) kept %d of %d features", self.method, len(chosen), n_candidates)
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Keep only the selected columns.

        Args:
            X: Feature matrix.

        Returns:
            Selected features (same container type as the input).
        """
        check_is_fitted(self)
        frame = self._to_frame(X)
        _check_columns(frame, self.selected_columns_, "FeatureSelector")
        return self._like_input(frame[self.selected_columns_], X)

    def get_selected_features(self) -> List[int]:
        """Positional indices of the selected input features."""
        check_is_fitted(self)
        return list(self._selected_features)

    def get_feature_scores(self) -> np.ndarray:
        """Per-input-feature scores (``NaN`` for skipped non-numeric columns)."""
        check_is_fitted(self)
        return self.scores_.copy()


# --------------------------------------------------------------------------- #
# Datetime features
# --------------------------------------------------------------------------- #
class DateTimeTransformer(_BaseTransformer):
    """Expand datetime columns into calendar features.

    Args:
        datetime_columns: Columns to expand. Defaults to every datetime column.
        extract_features: Features to create per column; any of ``'year'``,
            ``'month'``, ``'day'``, ``'dayofweek'``, ``'dayofyear'``,
            ``'weekofyear'``, ``'quarter'``, ``'hour'``, ``'minute'``,
            ``'second'``, ``'is_weekend'``, ``'is_month_start'``,
            ``'is_month_end'``, ``'days_since_epoch'``. Defaults to
            ``['year', 'month', 'day', 'dayofweek']``.
        cyclical_encoding: Also emit ``<col>_<feature>_sin``/``_cos`` for periodic features.
        drop_original: Remove the source datetime columns from the output.
        errors: Passed to ``pandas.to_datetime`` when a column needs parsing.

    Attributes:
        datetime_columns_: Columns that are expanded.
        extract_features_: Features that are created.
    """

    _DEFAULT_FEATURES = ("year", "month", "day", "dayofweek")
    _CYCLE_PERIODS = {
        "month": 12,
        "day": 31,
        "dayofweek": 7,
        "dayofyear": 366,
        "weekofyear": 53,
        "quarter": 4,
        "hour": 24,
        "minute": 60,
        "second": 60,
    }
    _SUPPORTED_FEATURES = (
        "year",
        "month",
        "day",
        "dayofweek",
        "dayofyear",
        "weekofyear",
        "quarter",
        "hour",
        "minute",
        "second",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "days_since_epoch",
    )

    def __init__(
        self,
        datetime_columns: Optional[List[str]] = None,
        extract_features: Optional[List[str]] = None,
        cyclical_encoding: bool = False,
        drop_original: bool = False,
        errors: str = "raise",
    ):
        self.datetime_columns = datetime_columns
        self.extract_features = extract_features
        self.cyclical_encoding = cyclical_encoding
        self.drop_original = drop_original
        self.errors = errors

    @property
    def datetime_columns_(self) -> Optional[List[str]]:
        """Columns expanded by the fitted transformer (``None`` before fitting)."""
        return getattr(self, "_datetime_columns", None)

    def _to_datetime(self, series: pd.Series) -> pd.Series:
        if pd.api.types.is_datetime64_any_dtype(series):
            return series
        return pd.to_datetime(series, errors=self.errors)

    @staticmethod
    def _extract(dt: pd.Series, feature: str) -> pd.Series:
        accessor = dt.dt
        if feature == "weekofyear":
            return accessor.isocalendar().week.astype("int64")
        if feature == "is_weekend":
            return (accessor.dayofweek >= 5).astype("int64")
        if feature in ("is_month_start", "is_month_end"):
            return getattr(accessor, feature).astype("int64")
        if feature == "days_since_epoch":
            return (dt - pd.Timestamp("1970-01-01")).dt.days
        return getattr(accessor, feature)

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> DateTimeTransformer:
        """Resolve the datetime columns and the output feature names.

        Args:
            X: DataFrame containing the datetime columns.
            y: Ignored.

        Returns:
            Fitted transformer.

        Raises:
            TypeError: If ``X`` is not a DataFrame.
            ValueError: If an unsupported feature name is requested.
        """
        frame = _require_dataframe(X, "DateTimeTransformer")
        features = (
            list(self.extract_features) if self.extract_features is not None else list(self._DEFAULT_FEATURES)
        )
        unknown = [f for f in features if f not in self._SUPPORTED_FEATURES]
        if unknown:
            raise ValueError(
                f"Unsupported datetime features {unknown}. Choose from {self._SUPPORTED_FEATURES}."
            )

        if self.datetime_columns is None:
            columns = _columns_of_kind(frame, ("datetime",))
        else:
            columns = _check_columns(frame, self.datetime_columns, "DateTimeTransformer")
        for col in columns:
            self._to_datetime(frame[col])  # validates that the column can be parsed

        self._record_input(X, frame)
        self._datetime_columns = list(columns)
        self.extract_features_ = features

        names: List[str] = [col for col in frame.columns if not (self.drop_original and col in columns)]
        for col in columns:
            for feature in features:
                names.append(f"{col}_{feature}")
                if self.cyclical_encoding and feature in self._CYCLE_PERIODS:
                    names.extend([f"{col}_{feature}_sin", f"{col}_{feature}_cos"])
        self.feature_names_out_ = names
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Append the calendar features.

        Args:
            X: DataFrame containing the datetime columns.

        Returns:
            Copy of ``X`` with the new columns appended.
        """
        check_is_fitted(self)
        frame = _require_dataframe(X, "DateTimeTransformer")
        _check_columns(frame, self._datetime_columns, "DateTimeTransformer")
        out = frame.copy()
        for col in self._datetime_columns:
            dt = self._to_datetime(frame[col])
            for feature in self.extract_features_:
                values = self._extract(dt, feature)
                out[f"{col}_{feature}"] = values
                if self.cyclical_encoding and feature in self._CYCLE_PERIODS:
                    angle = 2.0 * np.pi * values.astype(float) / self._CYCLE_PERIODS[feature]
                    out[f"{col}_{feature}_sin"] = np.sin(angle)
                    out[f"{col}_{feature}_cos"] = np.cos(angle)
        if self.drop_original:
            out = out.drop(columns=self._datetime_columns)
        return out


# --------------------------------------------------------------------------- #
# Categorical encoding
# --------------------------------------------------------------------------- #
class TargetEncoder(_BaseTransformer):
    """Replace categories by their (smoothed) mean target value.

    Args:
        categorical_columns: Columns to encode. Defaults to all non-numeric columns.
        smoothing: Pseudo-count pulling category means towards the global mean
            (``0`` gives raw means).
        cv_folds: When set, :meth:`fit_transform` produces out-of-fold encodings
            for the training data to limit target leakage.
        handle_unknown: ``'global_mean'`` maps unseen (or missing) categories to
            the global target mean; ``'error'`` raises.
        random_state: Seed for the fold shuffling.
        shuffle: Whether folds are shuffled.

    Attributes:
        columns_: Encoded columns.
        mappings_: ``{column: {category: encoded_value}}`` learned on the full training data.
        prior_: Global target mean.
    """

    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        smoothing: float = 1.0,
        cv_folds: Optional[int] = None,
        handle_unknown: str = "global_mean",
        random_state: Optional[int] = 42,
        shuffle: bool = True,
    ):
        self.categorical_columns = categorical_columns
        self.smoothing = smoothing
        self.cv_folds = cv_folds
        self.handle_unknown = handle_unknown
        self.random_state = random_state
        self.shuffle = shuffle

    @staticmethod
    def _target_array(y: Any) -> np.ndarray:
        if y is None:
            raise ValueError("TargetEncoder requires a target vector y.")
        try:
            return np.asarray(y, dtype=float).ravel()
        except (TypeError, ValueError) as exc:
            raise ValueError("TargetEncoder requires a numeric (or binary 0/1) target.") from exc

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> TargetEncoder:
        """Learn the per-category target means.

        Args:
            X: DataFrame holding the categorical columns.
            y: Numeric target.

        Returns:
            Fitted encoder.
        """
        frame = _require_dataframe(X, "TargetEncoder")
        if self.handle_unknown not in ("global_mean", "error"):
            raise ValueError("handle_unknown must be 'global_mean' or 'error'")
        target = self._target_array(y)
        if len(target) != len(frame):
            raise ValueError(f"X has {len(frame)} rows but y has {len(target)} values.")
        if self.categorical_columns is None:
            columns = _columns_of_kind(frame, ("categorical",))
        else:
            columns = _check_columns(frame, self.categorical_columns, "TargetEncoder")

        self._record_input(X, frame)
        self.columns_ = list(columns)
        self.prior_ = float(np.mean(target))
        self.mappings_ = {col: _target_mean_mapping(frame[col], target, self.smoothing)[0] for col in columns}
        self.feature_names_out_ = list(frame.columns)
        return self

    def _encode(self, series: pd.Series, mapping: Dict[Any, float], prior: float) -> pd.Series:
        encoded = series.map(mapping).astype(float)
        if self.handle_unknown == "error":
            unseen = series[encoded.isna() & series.notna()].unique()
            if len(unseen):
                raise ValueError(
                    f"TargetEncoder: unseen categories in column '{series.name}': {list(unseen)}"
                )
        return encoded.fillna(prior)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode the categorical columns with the learned means.

        Args:
            X: DataFrame holding the categorical columns.

        Returns:
            Copy of ``X`` with the encoded columns as floats.
        """
        check_is_fitted(self)
        frame = _require_dataframe(X, "TargetEncoder")
        _check_columns(frame, self.columns_, "TargetEncoder")
        out = frame.copy()
        for col in self.columns_:
            out[col] = self._encode(frame[col], self.mappings_[col], self.prior_)
        return out

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[np.ndarray] = None, **fit_params: Any
    ) -> pd.DataFrame:
        """Fit and encode; uses out-of-fold encodings when ``cv_folds`` is set.

        Args:
            X: DataFrame holding the categorical columns.
            y: Numeric target.
            **fit_params: Ignored.

        Returns:
            Encoded copy of ``X``.
        """
        self.fit(X, y)
        if not self.cv_folds:
            return self.transform(X)
        frame = X
        target = self._target_array(y)
        n_splits = min(int(self.cv_folds), len(frame))
        folds = KFold(
            n_splits=n_splits, shuffle=self.shuffle, random_state=self.random_state if self.shuffle else None
        )
        out = frame.copy()
        for col in self.columns_:
            encoded = np.full(len(frame), self.prior_, dtype=float)
            for train_idx, valid_idx in folds.split(frame):
                mapping, prior = _target_mean_mapping(
                    frame[col].iloc[train_idx], target[train_idx], self.smoothing
                )
                encoded[valid_idx] = self._encode(frame[col].iloc[valid_idx], mapping, prior).to_numpy()
            out[col] = encoded
        return out


class CategoricalEncoder(_BaseTransformer):
    """Encode categorical columns in place, keeping the other columns untouched.

    Args:
        categorical_columns: Columns to encode. Defaults to all non-numeric columns.
        encoding_method: ``'onehot'``, ``'label'`` (alias ``'ordinal'``), ``'target'`` or ``'frequency'``.
        handle_unknown: ``'ignore'`` (unseen categories become all-zero one-hot rows,
            ``-1`` labels, the global target mean or frequency ``0``) or ``'error'``.
        max_cardinality: For one-hot encoding, keep at most this many categories per
            column and group the remaining infrequent ones.
        smoothing: Smoothing used by ``'target'`` encoding.

    Attributes:
        columns_: Encoded columns.
        encoders_: Per-column fitted encoder (``dict`` mapping for ``'frequency'``).
        target_encoder_: Fitted :class:`TargetEncoder` for ``'target'`` encoding.
    """

    _METHODS = ("onehot", "label", "ordinal", "target", "frequency")

    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        encoding_method: str = "onehot",
        handle_unknown: str = "ignore",
        max_cardinality: Optional[int] = None,
        smoothing: float = 1.0,
    ):
        self.categorical_columns = categorical_columns
        self.encoding_method = encoding_method
        self.handle_unknown = handle_unknown
        self.max_cardinality = max_cardinality
        self.smoothing = smoothing

    def _make_encoder(self, series: pd.Series) -> Any:
        if self.encoding_method == "onehot":
            if self.handle_unknown == "error":
                handle = "error"
            else:
                handle = "infrequent_if_exist" if self.max_cardinality is not None else "ignore"
            encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown=handle,
                max_categories=self.max_cardinality,
                dtype=np.int64,
            )
            return encoder.fit(series.to_frame())
        if self.encoding_method in ("label", "ordinal"):
            if self.handle_unknown == "error":
                encoder = OrdinalEncoder(handle_unknown="error", encoded_missing_value=-2, dtype=np.int64)
            else:
                encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-2,
                    dtype=np.int64,
                )
            return encoder.fit(series.to_frame())
        return series.value_counts(normalize=True, dropna=True).to_dict()  # frequency

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> CategoricalEncoder:
        """Fit one encoder per categorical column.

        Args:
            X: DataFrame holding the categorical columns.
            y: Target; required for ``'target'`` encoding.

        Returns:
            Fitted encoder.

        Raises:
            ValueError: If the method or ``handle_unknown`` option is unknown.
        """
        frame = _require_dataframe(X, "CategoricalEncoder")
        if self.encoding_method not in self._METHODS:
            raise ValueError(
                f"Unknown encoding method '{self.encoding_method}'. Choose from {self._METHODS}."
            )
        if self.handle_unknown not in ("ignore", "error"):
            raise ValueError("handle_unknown must be 'ignore' or 'error'")
        if self.categorical_columns is None:
            columns = _columns_of_kind(frame, ("categorical",))
        else:
            columns = _check_columns(frame, self.categorical_columns, "CategoricalEncoder")

        self._record_input(X, frame)
        self.columns_ = list(columns)
        self.encoders_ = {}
        self.target_encoder_ = None
        if self.encoding_method == "target":
            self.target_encoder_ = TargetEncoder(
                categorical_columns=self.columns_,
                smoothing=self.smoothing,
                handle_unknown="error" if self.handle_unknown == "error" else "global_mean",
            ).fit(frame, y)
        else:
            self.encoders_ = {col: self._make_encoder(frame[col]) for col in columns}

        names: List[str] = []
        for col in frame.columns:
            if col in self.columns_ and self.encoding_method == "onehot":
                names.extend(str(n) for n in self.encoders_[col].get_feature_names_out([str(col)]))
            else:
                names.append(str(col))
        self.feature_names_out_ = names
        return self

    def _encode_column(self, col: str, series: pd.Series) -> pd.DataFrame:
        encoder = self.encoders_[col]
        if self.encoding_method == "onehot":
            values = encoder.transform(series.to_frame())
            return pd.DataFrame(values, columns=encoder.get_feature_names_out([str(col)]), index=series.index)
        if self.encoding_method in ("label", "ordinal"):
            values = encoder.transform(series.to_frame()).ravel()
            return pd.DataFrame({col: values}, index=series.index)
        encoded = series.map(encoder).astype(float)
        if self.handle_unknown == "error" and (encoded.isna() & series.notna()).any():
            raise ValueError(f"CategoricalEncoder: unseen categories in column '{col}'")
        return pd.DataFrame({col: encoded.fillna(0.0)}, index=series.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode the categorical columns.

        Args:
            X: DataFrame holding the categorical columns.

        Returns:
            DataFrame with encoded columns replacing the originals (in place).
        """
        check_is_fitted(self)
        frame = _require_dataframe(X, "CategoricalEncoder")
        _check_columns(frame, self.columns_, "CategoricalEncoder")
        target_frame = self.target_encoder_.transform(frame[self.columns_]) if self.target_encoder_ else None
        pieces: List[pd.DataFrame] = []
        for col in frame.columns:
            if col not in self.columns_:
                pieces.append(frame[[col]])
            elif target_frame is not None:
                pieces.append(target_frame[[col]])
            else:
                pieces.append(self._encode_column(col, frame[col]))
        return pd.concat(pieces, axis=1)


# --------------------------------------------------------------------------- #
# Numeric preprocessing
# --------------------------------------------------------------------------- #
class NumericTransformer(_BaseTransformer):
    """Impute, clip, transform and scale numeric columns in place.

    Processing order per column: missing-value fill, outlier treatment,
    derived transforms (appended as new columns) and finally scaling of the
    original and derived numeric columns.

    Args:
        numeric_columns: Columns to process. Defaults to all numeric columns.
        scaling_method: ``'standard'``, ``'minmax'``, ``'robust'``, ``'maxabs'``,
            ``'quantile'``, ``'power'`` or ``'none'``.
        handle_outliers: Treat values outside the Tukey fences learned at fit time.
        outlier_method: ``'clip'`` (winsorise to the fences) or ``'median'``
            (replace by the training median).
        outlier_threshold: IQR multiplier for the fences.
        apply_transforms: Derived columns to add: any of ``'log'``, ``'sqrt'``,
            ``'square'``, ``'yeo_johnson'`` (named ``<col>_<transform>``).
        handle_missing: ``'median'``, ``'mean'``, ``'constant'`` or ``None`` (leave NaN).
        fill_value: Value used by ``handle_missing='constant'``.
        n_quantiles: Quantile count for ``'quantile'`` scaling (capped at ``n_samples``).
        random_state: Seed for the quantile transformer.

    Attributes:
        columns_: Processed columns.
        fill_values_: Per-column fill values (``None`` when missing values are left as is).
        outlier_bounds_: ``(lower, upper)`` Series per column, or ``None``.
        derived_columns_: Names of the appended transform columns.
        scaler_: Fitted scaler (``None`` for ``'none'``).
    """

    _SCALERS = ("standard", "minmax", "robust", "maxabs", "quantile", "power", "none")
    _TRANSFORMS = ("log", "sqrt", "square", "yeo_johnson")
    _OUTLIER_METHODS = ("clip", "median")
    _MISSING = ("median", "mean", "constant")

    def __init__(
        self,
        numeric_columns: Optional[List[str]] = None,
        scaling_method: Optional[str] = "standard",
        handle_outliers: bool = False,
        outlier_method: str = "clip",
        outlier_threshold: float = 1.5,
        apply_transforms: Optional[List[str]] = None,
        handle_missing: Optional[str] = "median",
        fill_value: float = 0.0,
        n_quantiles: int = 1000,
        random_state: Optional[int] = 42,
    ):
        self.numeric_columns = numeric_columns
        self.scaling_method = scaling_method
        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.apply_transforms = apply_transforms
        self.handle_missing = handle_missing
        self.fill_value = fill_value
        self.n_quantiles = n_quantiles
        self.random_state = random_state

    def _validate_params(self) -> None:
        scaling = "none" if self.scaling_method is None else self.scaling_method
        if scaling not in self._SCALERS:
            raise ValueError(f"Unknown scaling method '{self.scaling_method}'. Choose from {self._SCALERS}.")
        if self.outlier_method not in self._OUTLIER_METHODS:
            raise ValueError(
                f"Unknown outlier method '{self.outlier_method}'. Choose from {self._OUTLIER_METHODS}."
            )
        if self.handle_missing is not None and self.handle_missing not in self._MISSING:
            raise ValueError(
                f"Unknown missing strategy '{self.handle_missing}'. Choose from {self._MISSING}."
            )
        unknown = [t for t in (self.apply_transforms or []) if t not in self._TRANSFORMS]
        if unknown:
            raise ValueError(f"Unknown transforms {unknown}. Choose from {self._TRANSFORMS}.")

    def _make_scaler(self, n_samples: int) -> Optional[BaseEstimator]:
        method = "none" if self.scaling_method is None else self.scaling_method
        if method == "standard":
            return StandardScaler()
        if method == "minmax":
            return MinMaxScaler()
        if method == "robust":
            return RobustScaler()
        if method == "maxabs":
            return MaxAbsScaler()
        if method == "quantile":
            return QuantileTransformer(
                n_quantiles=min(self.n_quantiles, n_samples), random_state=self.random_state
            )
        if method == "power":
            return PowerTransformer()
        return None

    def _fill(self, values: pd.DataFrame) -> pd.DataFrame:
        return values if self.fill_values_ is None else values.fillna(self.fill_values_)

    def _treat_outliers(self, values: pd.DataFrame) -> pd.DataFrame:
        if self.outlier_bounds_ is None:
            return values
        lower, upper = self.outlier_bounds_
        if self.outlier_method == "clip":
            return values.clip(lower=lower, upper=upper, axis=1)
        return values.mask((values < lower) | (values > upper), self.medians_, axis=1)

    def _derive(self, values: pd.DataFrame) -> pd.DataFrame:
        derived: Dict[str, np.ndarray] = {}
        for col in self.columns_:
            column = values[col].to_numpy(dtype=float)
            shifted = np.clip(column + self.shifts_[col], 0.0, None)
            for transform in self.transforms_:
                name = f"{col}_{transform}"
                if transform == "log":
                    derived[name] = np.log1p(shifted)
                elif transform == "sqrt":
                    derived[name] = np.sqrt(shifted)
                elif transform == "square":
                    derived[name] = np.square(column)
                else:
                    derived[name] = self.power_transformers_[col].transform(column.reshape(-1, 1)).ravel()
        return pd.DataFrame(derived, index=values.index)

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> NumericTransformer:
        """Learn fill values, outlier fences, transform shifts and the scaler.

        Args:
            X: Feature matrix.
            y: Ignored.

        Returns:
            Fitted transformer.

        Raises:
            TypeError: If a requested column is not numeric.
            ValueError: If no numeric columns are available or an option is unknown.
        """
        self._validate_params()
        frame = self._to_frame(X, use_fitted_names=False)
        self._record_input(X, frame)
        if self.numeric_columns is not None:
            columns = _check_numeric_columns(frame, self.numeric_columns, "NumericTransformer")
        else:
            columns = _columns_of_kind(frame, ("numeric",))
        if not columns:
            raise ValueError("NumericTransformer needs at least one numeric column.")
        self.columns_ = list(columns)

        values = frame[self.columns_].astype(float)
        if self.handle_missing == "mean":
            self.fill_values_ = values.mean()
        elif self.handle_missing == "median":
            self.fill_values_ = values.median()
        elif self.handle_missing == "constant":
            self.fill_values_ = pd.Series(float(self.fill_value), index=self.columns_)
        else:
            self.fill_values_ = None
        values = self._fill(values)

        if self.handle_outliers:
            lower, upper = _iqr_bounds(values.to_numpy(dtype=float), float(self.outlier_threshold))
            self.outlier_bounds_ = (
                pd.Series(lower, index=self.columns_),
                pd.Series(upper, index=self.columns_),
            )
            self.medians_ = values.median()
            values = self._treat_outliers(values)
        else:
            self.outlier_bounds_ = None
            self.medians_ = None

        self.transforms_ = list(self.apply_transforms or [])
        self.shifts_ = {
            col: float(max(0.0, -np.nanmin(values[col].to_numpy(dtype=float)))) for col in self.columns_
        }
        self.power_transformers_ = {}
        if "yeo_johnson" in self.transforms_:
            for col in self.columns_:
                self.power_transformers_[col] = PowerTransformer().fit(values[[col]].to_numpy(dtype=float))
        derived = self._derive(values)
        self.derived_columns_ = list(derived.columns)

        full = pd.concat([values, derived], axis=1)
        self.scaler_ = self._make_scaler(len(full))
        if self.scaler_ is not None:
            self.scaler_.fit(full.to_numpy(dtype=float))
        self.feature_names_out_ = list(frame.columns) + self.derived_columns_
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Apply the learned numeric processing.

        Args:
            X: Feature matrix.

        Returns:
            Transformed data (same container type as the input) with derived columns appended.
        """
        check_is_fitted(self)
        frame = self._to_frame(X)
        _check_columns(frame, self.columns_, "NumericTransformer")
        values = self._treat_outliers(self._fill(frame[self.columns_].astype(float)))
        full = pd.concat([values, self._derive(values)], axis=1)
        if self.scaler_ is not None:
            full = pd.DataFrame(
                self.scaler_.transform(full.to_numpy(dtype=float)), columns=full.columns, index=full.index
            )
        out = frame.copy()
        out[self.columns_] = full[self.columns_]
        for col in self.derived_columns_:
            out[col] = full[col]
        return self._like_input(out, X)


class MissingValueHandler(_BaseTransformer):
    """Impute missing values with per-column strategies.

    Args:
        strategy: ``'auto'`` (median for numeric, most frequent otherwise),
            ``'mean'``, ``'median'``, ``'mode'``/``'most_frequent'``,
            ``'constant'``, ``'knn'`` or a ``{column: strategy}`` mapping
            (unlisted columns use ``'auto'``). Strategies that need numeric data
            fall back to the most frequent value on non-numeric columns.
        fill_value: Value for ``'constant'``; defaults to ``0`` for numeric and
            ``'missing'`` for other columns.
        add_indicator: Append ``<col>_missing`` indicator columns for columns
            that had missing values during fit.
        n_neighbors: Neighbourhood size for ``'knn'``.
        columns: Columns to impute. Defaults to every column.
        indicator_suffix: Suffix for indicator columns.

    Attributes:
        columns_: Imputed columns.
        strategies_: Resolved strategy per column.
        fill_values_: Fill value per non-KNN column.
        knn_columns_: Columns imputed with KNN.
        indicator_columns_: Columns that receive an indicator.
    """

    _STRATEGIES = ("auto", "mean", "median", "mode", "most_frequent", "constant", "knn")

    def __init__(
        self,
        strategy: Union[str, Dict[str, str]] = "auto",
        fill_value: Optional[Any] = None,
        add_indicator: bool = False,
        n_neighbors: int = 5,
        columns: Optional[List[str]] = None,
        indicator_suffix: str = "_missing",
    ):
        self.strategy = strategy
        self.fill_value = fill_value
        self.add_indicator = add_indicator
        self.n_neighbors = n_neighbors
        self.columns = columns
        self.indicator_suffix = indicator_suffix

    def _requested_strategy(self, col: Any) -> str:
        requested = self.strategy.get(col, "auto") if isinstance(self.strategy, dict) else self.strategy
        if requested not in self._STRATEGIES:
            raise ValueError(
                f"Unknown imputation strategy '{requested}' for column '{col}'. Choose from {self._STRATEGIES}."
            )
        return requested

    def _resolve_strategy(self, requested: str, kind: str, col: Any) -> str:
        numeric = kind == "numeric"
        if requested == "auto":
            return "median" if numeric else "most_frequent"
        if requested == "mode":
            return "most_frequent"
        if requested in ("mean", "median", "knn") and not numeric:
            self.logger.debug(
                "MissingValueHandler: '%s' is not numeric; using most_frequent for %s", col, requested
            )
            return "most_frequent"
        return requested

    def _fill_value_for(self, series: pd.Series, resolved: str, kind: str) -> Any:
        numeric = kind == "numeric"
        if resolved == "mean":
            return float(series.mean())
        if resolved == "median":
            return float(series.median())
        if resolved == "constant":
            if self.fill_value is not None:
                return self.fill_value
            return 0.0 if numeric else "missing"
        modes = series.mode(dropna=True)
        if len(modes):
            return modes.iloc[0]
        return 0.0 if numeric else "missing"

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> MissingValueHandler:
        """Learn the fill values (and the KNN imputer when requested).

        Args:
            X: Feature matrix.
            y: Ignored.

        Returns:
            Fitted handler.
        """
        frame = self._to_frame(X, use_fitted_names=False)
        self._record_input(X, frame)
        columns = (
            list(frame.columns)
            if self.columns is None
            else _check_columns(frame, self.columns, "MissingValueHandler")
        )

        strategies: Dict[Any, str] = {}
        fill_values: Dict[Any, Any] = {}
        knn_columns: List[Any] = []
        for col in columns:
            kind = _column_kind(frame[col])
            resolved = self._resolve_strategy(self._requested_strategy(col), kind, col)
            strategies[col] = resolved
            if resolved == "knn":
                knn_columns.append(col)
            else:
                fill_values[col] = self._fill_value_for(frame[col], resolved, kind)

        self.columns_ = columns
        self.strategies_ = strategies
        self.fill_values_ = fill_values
        self.knn_columns_ = knn_columns
        self.knn_imputer_ = None
        self.knn_feature_columns_ = []
        if knn_columns:
            self.knn_feature_columns_ = _columns_of_kind(frame, ("numeric", "bool"))
            self.knn_imputer_ = KNNImputer(n_neighbors=self.n_neighbors)
            self.knn_imputer_.fit(frame[self.knn_feature_columns_].astype(float))
        self.indicator_columns_ = (
            [col for col in columns if frame[col].isna().any()] if self.add_indicator else []
        )
        self.feature_names_out_ = list(frame.columns) + [
            f"{col}{self.indicator_suffix}" for col in self.indicator_columns_
        ]
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Fill the missing values.

        Args:
            X: Feature matrix.

        Returns:
            Imputed data (same container type as the input).
        """
        check_is_fitted(self)
        frame = self._to_frame(X)
        _check_columns(frame, self.columns_, "MissingValueHandler")
        out = frame.copy()
        for col in self.indicator_columns_:
            out[f"{col}{self.indicator_suffix}"] = frame[col].isna().astype(int)
        for col, value in self.fill_values_.items():
            out[col] = frame[col].fillna(value)
        if self.knn_imputer_ is not None:
            _check_columns(frame, self.knn_feature_columns_, "MissingValueHandler")
            imputed = self.knn_imputer_.transform(frame[self.knn_feature_columns_].astype(float))
            for col in self.knn_columns_:
                out[col] = imputed[:, self.knn_feature_columns_.index(col)]
        return self._like_input(out, X)


class CustomScaler(_BaseTransformer):
    """Scale numeric columns with a selectable scikit-learn scaler.

    Args:
        method: ``'standard'``, ``'minmax'``, ``'robust'``, ``'maxabs'``,
            ``'quantile'``, ``'power'`` or ``'unit_vector'`` (row-wise L2 normalisation).
        columns: Columns to scale. Defaults to all numeric columns.
        n_quantiles: Quantile count for ``'quantile'`` (capped at ``n_samples``).
        output_distribution: ``'uniform'`` or ``'normal'`` for ``'quantile'``.
        feature_range: Target range for ``'minmax'``.
        norm: Norm for ``'unit_vector'``.
        random_state: Seed for the quantile transformer.

    Attributes:
        columns_: Scaled columns.
        scaler_: The fitted scikit-learn scaler.
    """

    _METHODS = ("standard", "minmax", "robust", "maxabs", "quantile", "power", "unit_vector")

    def __init__(
        self,
        method: str = "standard",
        columns: Optional[List[str]] = None,
        n_quantiles: int = 1000,
        output_distribution: str = "uniform",
        feature_range: Tuple[float, float] = (0.0, 1.0),
        norm: str = "l2",
        random_state: Optional[int] = 42,
    ):
        self.method = method
        self.columns = columns
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.feature_range = feature_range
        self.norm = norm
        self.random_state = random_state

    def _make_scaler(self, n_samples: int) -> BaseEstimator:
        if self.method == "standard":
            return StandardScaler()
        if self.method == "minmax":
            return MinMaxScaler(feature_range=self.feature_range)
        if self.method == "robust":
            return RobustScaler()
        if self.method == "maxabs":
            return MaxAbsScaler()
        if self.method == "quantile":
            return QuantileTransformer(
                n_quantiles=min(self.n_quantiles, n_samples),
                output_distribution=self.output_distribution,
                random_state=self.random_state,
            )
        if self.method == "power":
            return PowerTransformer()
        if self.method == "unit_vector":
            return Normalizer(norm=self.norm)
        raise ValueError(f"Unknown scaling method '{self.method}'. Choose from {self._METHODS}.")

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> CustomScaler:
        """Fit the scaler on the numeric columns.

        Args:
            X: Feature matrix.
            y: Ignored.

        Returns:
            Fitted scaler.
        """
        frame = self._to_frame(X, use_fitted_names=False)
        self._record_input(X, frame)
        if self.columns is not None:
            columns = _check_numeric_columns(frame, self.columns, "CustomScaler")
        else:
            columns = _columns_of_kind(frame, ("numeric",))
        if not columns:
            raise ValueError("CustomScaler needs at least one numeric column.")
        self.columns_ = list(columns)
        self.scaler_ = self._make_scaler(len(frame)).fit(frame[self.columns_].to_numpy(dtype=float))
        self.feature_names_out_ = list(frame.columns)
        return self

    def _apply(self, X: ArrayLike, func: Callable[[np.ndarray], np.ndarray]) -> ArrayLike:
        frame = self._to_frame(X)
        _check_columns(frame, self.columns_, "CustomScaler")
        out = frame.copy()
        out[self.columns_] = pd.DataFrame(
            func(frame[self.columns_].to_numpy(dtype=float)), columns=self.columns_, index=frame.index
        )
        return self._like_input(out, X)

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Scale the numeric columns.

        Args:
            X: Feature matrix.

        Returns:
            Scaled data (same container type as the input).
        """
        check_is_fitted(self)
        return self._apply(X, self.scaler_.transform)

    def inverse_transform(self, X: ArrayLike) -> ArrayLike:
        """Undo the scaling.

        Args:
            X: Scaled feature matrix.

        Returns:
            Data on the original scale.

        Raises:
            NotImplementedError: For ``'unit_vector'`` scaling, which is not invertible.
        """
        check_is_fitted(self)
        if not hasattr(self.scaler_, "inverse_transform"):
            raise NotImplementedError(f"CustomScaler(method='{self.method}') is not invertible.")
        return self._apply(X, self.scaler_.inverse_transform)


class BinningTransformer(_BaseTransformer):
    """Discretise numeric columns into bins.

    Args:
        columns: Columns to bin. Defaults to all numeric columns.
        n_bins: Number of bins for the learned strategies.
        strategy: ``'uniform'`` (equal width), ``'quantile'`` (equal frequency) or ``'kmeans'``.
        bins: Explicit bin edges (sequence). An ``int`` is treated as ``n_bins``.
            Values outside explicit edges are unassigned (``-1``/missing).
        labels: Bin labels; implies ``encode='label'`` unless ``encode`` is set.
        encode: ``'ordinal'`` (integer bin index), ``'label'`` (bin labels or
            interval strings), ``'onehot'`` or ``'auto'`` (label when ``labels`` is given).
        drop_original: Remove the source columns.
        suffix: Suffix of the binned column (``<col><suffix>``).

    Attributes:
        columns_: Binned columns.
        bin_edges_: ``{column: ndarray}`` of finite edges.
        bin_labels_: ``{column: list}`` of bin labels.
        encode_: Resolved encoding.
    """

    _STRATEGIES = ("uniform", "quantile", "kmeans")
    _ENCODINGS = ("auto", "ordinal", "label", "onehot")

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        n_bins: int = 5,
        strategy: str = "uniform",
        bins: Optional[Union[int, Sequence[float]]] = None,
        labels: Optional[Sequence[Any]] = None,
        encode: str = "auto",
        drop_original: bool = False,
        suffix: str = "_binned",
    ):
        self.columns = columns
        self.n_bins = n_bins
        self.strategy = strategy
        self.bins = bins
        self.labels = labels
        self.encode = encode
        self.drop_original = drop_original
        self.suffix = suffix

    def _learn_edges(self, values: np.ndarray, n_bins: int) -> np.ndarray:
        values = values[~np.isnan(values)]
        if values.size == 0:
            raise ValueError("BinningTransformer: cannot bin a column with only missing values.")
        if self.strategy == "uniform":
            return np.linspace(values.min(), values.max(), n_bins + 1)
        if self.strategy == "quantile":
            edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
            if len(edges) - 1 < n_bins:
                self.logger.warning(
                    "BinningTransformer: duplicate quantile edges reduced bins to %d", len(edges) - 1
                )
            return edges
        if self.strategy == "kmeans":
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="kmeans")
            discretizer.fit(values.reshape(-1, 1))
            return np.asarray(discretizer.bin_edges_[0], dtype=float)
        raise ValueError(f"Unknown binning strategy '{self.strategy}'. Choose from {self._STRATEGIES}.")

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> BinningTransformer:
        """Learn the bin edges.

        Args:
            X: Feature matrix.
            y: Ignored.

        Returns:
            Fitted transformer.
        """
        if self.encode not in self._ENCODINGS:
            raise ValueError(f"Unknown encode option '{self.encode}'. Choose from {self._ENCODINGS}.")
        frame = self._to_frame(X, use_fitted_names=False)
        self._record_input(X, frame)
        if self.columns is not None:
            columns = _check_numeric_columns(frame, self.columns, "BinningTransformer")
        else:
            columns = _columns_of_kind(frame, ("numeric",))
        if not columns:
            raise ValueError("BinningTransformer needs at least one numeric column.")

        explicit_edges = None
        n_bins = int(self.n_bins)
        if isinstance(self.bins, (int, np.integer)) and not isinstance(self.bins, bool):
            n_bins = int(self.bins)
        elif self.bins is not None:
            explicit_edges = np.asarray(list(self.bins), dtype=float)
            if explicit_edges.ndim != 1 or len(explicit_edges) < 2 or np.any(np.diff(explicit_edges) <= 0):
                raise ValueError("bins must be a strictly increasing sequence of at least two edges.")

        self.columns_ = list(columns)
        self.explicit_edges_ = explicit_edges is not None
        self.encode_ = (
            self.encode if self.encode != "auto" else ("label" if self.labels is not None else "ordinal")
        )
        self.bin_edges_ = {}
        self.bin_labels_ = {}
        for col in self.columns_:
            edges = (
                explicit_edges
                if explicit_edges is not None
                else self._learn_edges(frame[col].to_numpy(dtype=float), n_bins)
            )
            if self.labels is not None and len(self.labels) != len(edges) - 1:
                raise ValueError(
                    f"labels has {len(self.labels)} entries but column '{col}' has {len(edges) - 1} bins."
                )
            self.bin_edges_[col] = edges
            self.bin_labels_[col] = (
                list(self.labels)
                if self.labels is not None
                else [f"({lo:.4g}, {hi:.4g}]" for lo, hi in zip(edges[:-1], edges[1:])]
            )

        names = [col for col in frame.columns if not (self.drop_original and col in self.columns_)]
        for col in self.columns_:
            if self.encode_ == "onehot":
                names.extend(f"{col}{self.suffix}_{label}" for label in self.bin_labels_[col])
            else:
                names.append(f"{col}{self.suffix}")
        self.feature_names_out_ = names
        return self

    def _codes(self, series: pd.Series, edges: np.ndarray) -> np.ndarray:
        cut_edges = edges.copy()
        if not self.explicit_edges_:
            cut_edges[0], cut_edges[-1] = -np.inf, np.inf
        codes = pd.cut(series.astype(float), bins=cut_edges, labels=False, include_lowest=True)
        return np.asarray(codes, dtype=float)

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Bin the numeric columns.

        Args:
            X: Feature matrix.

        Returns:
            Data with binned columns appended (same container type as the input).
        """
        check_is_fitted(self)
        frame = self._to_frame(X)
        _check_columns(frame, self.columns_, "BinningTransformer")
        out = frame.copy()
        for col in self.columns_:
            codes = self._codes(frame[col], self.bin_edges_[col])
            labels = self.bin_labels_[col]
            name = f"{col}{self.suffix}"
            if self.encode_ == "ordinal":
                out[name] = np.where(np.isnan(codes), -1, codes).astype(np.int64)
            elif self.encode_ == "label":
                out[name] = pd.Series(codes, index=frame.index).map(dict(enumerate(labels)))
            else:
                for i, label in enumerate(labels):
                    out[f"{name}_{label}"] = (codes == i).astype(np.int64)
        if self.drop_original:
            out = out.drop(columns=self.columns_)
        return self._like_input(out, X)


class PolynomialFeatureCreator(_BaseTransformer):
    """Polynomial and interaction terms with readable column names.

    Args:
        degree: Maximum polynomial degree.
        include_bias: Prepend a constant ``1`` column.
        interaction_only: Only products of distinct features (no powers).
        columns: Columns to expand. Defaults to all numeric columns; other
            columns are passed through after the polynomial block.

    Attributes:
        columns_: Expanded columns.
        poly_: The fitted ``PolynomialFeatures``.
        polynomial_feature_names_: Names of the polynomial block (``x1^2``, ``x1 x2``, ...).
    """

    def __init__(
        self,
        degree: int = 2,
        include_bias: bool = False,
        interaction_only: bool = False,
        columns: Optional[List[str]] = None,
    ):
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.columns = columns

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> PolynomialFeatureCreator:
        """Fit the polynomial expansion.

        Args:
            X: Feature matrix.
            y: Ignored.

        Returns:
            Fitted transformer.
        """
        frame = self._to_frame(X, use_fitted_names=False)
        self._record_input(X, frame)
        if self.columns is not None:
            columns = _check_numeric_columns(frame, self.columns, "PolynomialFeatureCreator")
        else:
            columns = _columns_of_kind(frame, ("numeric",))
        if not columns:
            raise ValueError("PolynomialFeatureCreator needs at least one numeric column.")
        self.columns_ = list(columns)
        self.passthrough_columns_ = [col for col in frame.columns if col not in self.columns_]
        self.poly_ = PolynomialFeatures(
            degree=self.degree, include_bias=self.include_bias, interaction_only=self.interaction_only
        ).fit(frame[self.columns_].to_numpy(dtype=float))
        self.polynomial_feature_names_ = [
            str(n) for n in self.poly_.get_feature_names_out([str(c) for c in self.columns_])
        ]
        self.feature_names_out_ = self.polynomial_feature_names_ + [str(c) for c in self.passthrough_columns_]
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Create the polynomial features.

        Args:
            X: Feature matrix.

        Returns:
            Polynomial block followed by pass-through columns (same container type as the input).
        """
        check_is_fitted(self)
        frame = self._to_frame(X)
        _check_columns(frame, self.columns_, "PolynomialFeatureCreator")
        poly = pd.DataFrame(
            self.poly_.transform(frame[self.columns_].to_numpy(dtype=float)),
            columns=self.polynomial_feature_names_,
            index=frame.index,
        )
        if self.passthrough_columns_:
            poly = pd.concat([poly, frame[self.passthrough_columns_]], axis=1)
        return self._like_input(poly, X)


class FeatureInteractionCreator(_BaseTransformer):
    """Create interaction features, either pairwise or via polynomial expansion.

    Two modes are available and can be combined:

    * **pairwise** (``interaction_pairs`` given or ``auto_detect=True``): one
      column per pair and operation, named ``<a>_<op>_<b>``, appended to the
      original columns;
    * **polynomial** (default when no pairs are requested, or when
      ``polynomial_degree`` is set): ``PolynomialFeatures`` over the numeric
      columns (which replaces them; non-numeric columns pass through first).

    Args:
        degree: Polynomial degree (legacy name; ``polynomial_degree`` wins when set).
        interaction_only: Polynomial products of distinct features only.
        include_bias: Add a constant column to the polynomial block.
        max_features: Cap on the number of generated numeric features. Ranked by
            univariate score when ``y`` is given, by variance otherwise.
        interaction_pairs: Explicit ``(a, b)`` column pairs (names or positions).
        interaction_types: Operations per pair: ``'multiply'``, ``'add'``,
            ``'subtract'``, ``'divide'`` (``a / b``), ``'ratio'`` (``b / a``),
            ``'max'``, ``'min'``. Defaults to ``['multiply']``.
        auto_detect: Build pairs from all numeric column combinations, keeping the
            ``max_interactions`` pairs whose product correlates most with ``y``
            (input order when ``y`` is absent).
        max_interactions: Number of auto-detected pairs.
        numeric_columns: Numeric columns considered. Defaults to all numeric columns.
        polynomial_degree: Explicit polynomial degree (enables the polynomial block).
        random_state: Reserved for reproducibility of future stochastic options.

    Attributes:
        numeric_columns_: Numeric columns considered.
        pairs_: Resolved column pairs.
        interaction_types_: Operations applied.
        poly_: Fitted ``PolynomialFeatures`` or ``None``.
        generated_columns_: Names of every generated column before capping.
        support_: Boolean mask over ``generated_columns_`` after ``max_features``.
    """

    _OPERATIONS: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
        "multiply": lambda a, b: a * b,
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "divide": _safe_divide,
        "ratio": lambda a, b: _safe_divide(b, a),
        "max": np.maximum,
        "min": np.minimum,
    }

    def __init__(
        self,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False,
        max_features: Optional[int] = None,
        interaction_pairs: Optional[List[Tuple[ColumnKey, ColumnKey]]] = None,
        interaction_types: Optional[List[str]] = None,
        auto_detect: bool = False,
        max_interactions: int = 10,
        numeric_columns: Optional[List[str]] = None,
        polynomial_degree: Optional[int] = None,
        random_state: Optional[int] = 42,
    ):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.max_features = max_features
        self.interaction_pairs = interaction_pairs
        self.interaction_types = interaction_types
        self.auto_detect = auto_detect
        self.max_interactions = max_interactions
        self.numeric_columns = numeric_columns
        self.polynomial_degree = polynomial_degree
        self.random_state = random_state

    def _resolve_pair(self, pair: Sequence[ColumnKey], frame: pd.DataFrame) -> Tuple[ColumnKey, ColumnKey]:
        if len(pair) != 2:
            raise ValueError(f"interaction pairs must have exactly two entries, got {pair!r}")
        resolved = []
        for item in pair:
            if (
                isinstance(item, (int, np.integer))
                and not isinstance(item, bool)
                and item not in frame.columns
            ):
                item = frame.columns[int(item)]
            resolved.append(item)
        _check_numeric_columns(frame, resolved, "FeatureInteractionCreator")
        return resolved[0], resolved[1]

    def _auto_pairs(self, frame: pd.DataFrame, y: Optional[np.ndarray]) -> List[Tuple[ColumnKey, ColumnKey]]:
        candidates = list(itertools.combinations(self.numeric_columns_, 2))
        if y is not None and len(candidates) > self.max_interactions:
            target = np.asarray(y, dtype=float)
            scores = []
            for a, b in candidates:
                product = frame[a].to_numpy(dtype=float) * frame[b].to_numpy(dtype=float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    corr = np.corrcoef(product, target)[0, 1]
                scores.append(0.0 if np.isnan(corr) else abs(corr))
            order = np.argsort(-np.asarray(scores), kind="stable")
            candidates = [candidates[i] for i in order]
        return candidates[: self.max_interactions]

    def _generate(self, frame: pd.DataFrame) -> pd.DataFrame:
        pieces: List[pd.DataFrame] = []
        if self.poly_ is not None:
            values = self.poly_.transform(frame[self.numeric_columns_].to_numpy(dtype=float))
            pieces.append(pd.DataFrame(values, columns=self.polynomial_feature_names_, index=frame.index))
        interactions: Dict[str, np.ndarray] = {}
        for a, b in self.pairs_:
            left = frame[a].to_numpy(dtype=float)
            right = frame[b].to_numpy(dtype=float)
            for op in self.interaction_types_:
                interactions[f"{a}_{op}_{b}"] = self._OPERATIONS[op](left, right)
        if interactions:
            pieces.append(pd.DataFrame(interactions, index=frame.index))
        if not pieces:
            return pd.DataFrame(index=frame.index)
        return pd.concat(pieces, axis=1)

    def _select(self, generated: pd.DataFrame, y: Optional[np.ndarray]) -> np.ndarray:
        n_generated = generated.shape[1]
        if self.max_features is None or n_generated <= self.max_features:
            return np.ones(n_generated, dtype=bool)
        values = generated.to_numpy(dtype=float)
        if y is not None:
            task = _infer_task(y)
            score_func = f_classif if task == "classification" else f_regression
            support = (
                SelectKBest(score_func=score_func, k=int(self.max_features)).fit(values, y).get_support()
            )
        else:
            ranked = np.argsort(-np.nanvar(values, axis=0), kind="stable")[: int(self.max_features)]
            support = np.zeros(n_generated, dtype=bool)
            support[ranked] = True
        self.logger.info(
            "FeatureInteractionCreator kept %d of %d generated features", int(support.sum()), n_generated
        )
        return support

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> FeatureInteractionCreator:
        """Resolve pairs, fit the polynomial expansion and select features.

        Args:
            X: Feature matrix.
            y: Optional target used to rank interactions.

        Returns:
            Fitted transformer.
        """
        frame = self._to_frame(X, use_fitted_names=False)
        self._record_input(X, frame)
        if self.numeric_columns is not None:
            numeric = _check_numeric_columns(frame, self.numeric_columns, "FeatureInteractionCreator")
        else:
            numeric = _columns_of_kind(frame, ("numeric",))
        if not numeric:
            raise ValueError("FeatureInteractionCreator needs at least one numeric column.")
        self.numeric_columns_ = list(numeric)

        types = list(self.interaction_types) if self.interaction_types is not None else ["multiply"]
        unknown = [t for t in types if t not in self._OPERATIONS]
        if unknown:
            raise ValueError(f"Unknown interaction types {unknown}. Choose from {tuple(self._OPERATIONS)}.")
        self.interaction_types_ = types

        y_array = None if y is None else np.asarray(y)
        pairwise = self.interaction_pairs is not None or self.auto_detect
        if self.interaction_pairs is not None:
            self.pairs_ = [self._resolve_pair(pair, frame) for pair in self.interaction_pairs]
        elif self.auto_detect:
            self.pairs_ = self._auto_pairs(frame, y_array)
        else:
            self.pairs_ = []

        self.use_polynomial_ = self.polynomial_degree is not None or not pairwise
        if self.use_polynomial_:
            degree = self.polynomial_degree if self.polynomial_degree is not None else self.degree
            self.poly_ = PolynomialFeatures(
                degree=degree, interaction_only=self.interaction_only, include_bias=self.include_bias
            ).fit(frame[self.numeric_columns_].to_numpy(dtype=float))
            self.polynomial_feature_names_ = [
                str(n) for n in self.poly_.get_feature_names_out([str(c) for c in self.numeric_columns_])
            ]
            self.base_columns_ = [col for col in frame.columns if col not in self.numeric_columns_]
        else:
            self.poly_ = None
            self.polynomial_feature_names_ = []
            self.base_columns_ = list(frame.columns)

        generated = self._generate(frame)
        self.generated_columns_ = [str(c) for c in generated.columns]
        self.support_ = self._select(generated, y_array)
        self.kept_columns_ = [col for col, keep in zip(generated.columns, self.support_) if keep]
        self.feature_names_out_ = [str(c) for c in self.base_columns_] + [str(c) for c in self.kept_columns_]
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Create the interaction features.

        Args:
            X: Feature matrix.

        Returns:
            Base columns followed by the generated features (same container type as the input).
        """
        check_is_fitted(self)
        frame = self._to_frame(X)
        _check_columns(frame, self.numeric_columns_, "FeatureInteractionCreator")
        _check_columns(frame, self.base_columns_, "FeatureInteractionCreator")
        generated = self._generate(frame)[self.kept_columns_]
        out = pd.concat([frame[self.base_columns_], generated], axis=1)
        return self._like_input(out, X)


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #
class TextTransformer(_BaseTransformer):
    """Vectorise text columns into dense term features appended to the frame.

    Args:
        text_columns: Columns to vectorise. Defaults to all non-numeric columns.
        method: ``'tfidf'`` or ``'count'``.
        max_features: Vocabulary cap per column.
        ngram_range: N-gram range for the vectoriser.
        drop_original: Remove the raw text columns.
        lowercase: Lowercase the text before tokenising.
        stop_words: Passed to the vectoriser (``'english'`` or a list).

    Attributes:
        columns_: Vectorised columns.
        vectorizers_: Fitted vectoriser per column.
    """

    def __init__(
        self,
        text_columns: Optional[List[str]] = None,
        method: str = "tfidf",
        max_features: Optional[int] = 100,
        ngram_range: Tuple[int, int] = (1, 1),
        drop_original: bool = True,
        lowercase: bool = True,
        stop_words: Optional[Union[str, List[str]]] = None,
    ):
        self.text_columns = text_columns
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.drop_original = drop_original
        self.lowercase = lowercase
        self.stop_words = stop_words

    def _make_vectorizer(self) -> BaseEstimator:
        kwargs = {
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "lowercase": self.lowercase,
            "stop_words": self.stop_words,
        }
        if self.method == "tfidf":
            return TfidfVectorizer(**kwargs)
        if self.method == "count":
            return CountVectorizer(**kwargs)
        raise ValueError("method must be 'tfidf' or 'count'")

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> TextTransformer:
        """Fit one vectoriser per text column.

        Args:
            X: DataFrame holding the text columns.
            y: Ignored.

        Returns:
            Fitted transformer.
        """
        frame = _require_dataframe(X, "TextTransformer")
        if self.text_columns is None:
            columns = _columns_of_kind(frame, ("categorical",))
        else:
            columns = _check_columns(frame, self.text_columns, "TextTransformer")
        self._record_input(X, frame)
        self.columns_ = list(columns)
        self.vectorizers_ = {
            col: self._make_vectorizer().fit(frame[col].fillna("").astype(str)) for col in columns
        }
        names = [str(col) for col in frame.columns if not (self.drop_original and col in self.columns_)]
        for col in self.columns_:
            names.extend(f"{col}_{term}" for term in self.vectorizers_[col].get_feature_names_out())
        self.feature_names_out_ = names
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Vectorise the text columns.

        Args:
            X: DataFrame holding the text columns.

        Returns:
            DataFrame with dense term columns appended.
        """
        check_is_fitted(self)
        frame = _require_dataframe(X, "TextTransformer")
        _check_columns(frame, self.columns_, "TextTransformer")
        keep = [col for col in frame.columns if not (self.drop_original and col in self.columns_)]
        pieces = [frame[keep]]
        for col in self.columns_:
            vectorizer = self.vectorizers_[col]
            matrix = vectorizer.transform(frame[col].fillna("").astype(str)).toarray()
            names = [f"{col}_{term}" for term in vectorizer.get_feature_names_out()]
            pieces.append(pd.DataFrame(matrix, columns=names, index=frame.index))
        return pd.concat(pieces, axis=1)


# --------------------------------------------------------------------------- #
# Validation and debugging
# --------------------------------------------------------------------------- #
class DataValidator(_BaseTransformer):
    """Validate a DataFrame against a column schema and pass it through.

    Args:
        schema: ``{column: rules}`` where rules may contain ``'type'``
            (``'numeric'``, ``'categorical'``/``'string'``, ``'datetime'``,
            ``'bool'``, ``'any'``), ``'min'``, ``'max'``, ``'categories'``,
            ``'nullable'`` (default ``True``) and ``'unique'``.
        strict: Reject columns that are not described by the schema.
        raise_on_error: Raise ``ValueError`` on violations; otherwise log them
            and store them in ``validation_errors_``.

    Attributes:
        validation_errors_: Violations found during the last validation.
    """

    _TYPES = ("numeric", "categorical", "string", "datetime", "bool", "any")

    def __init__(
        self,
        schema: Optional[Dict[str, Dict[str, Any]]] = None,
        strict: bool = True,
        raise_on_error: bool = True,
    ):
        self.schema = schema
        self.strict = strict
        self.raise_on_error = raise_on_error

    def _validate(self, frame: pd.DataFrame) -> List[str]:
        schema = self.schema or {}
        errors: List[str] = []
        if self.strict:
            extra = [col for col in frame.columns if col not in schema]
            if extra:
                errors.append(f"unexpected columns not in schema: {extra}")
        for col, rules in schema.items():
            if col not in frame.columns:
                errors.append(f"missing required column '{col}'")
                continue
            series = frame[col]
            kind = _column_kind(series)
            expected = rules.get("type", "any")
            if expected not in self._TYPES:
                raise ValueError(f"Unknown type '{expected}' for column '{col}'. Choose from {self._TYPES}.")
            type_ok = {
                "numeric": kind in ("numeric", "bool"),
                "categorical": kind in ("categorical", "bool"),
                "string": kind == "categorical",
                "datetime": kind == "datetime",
                "bool": kind == "bool",
                "any": True,
            }[expected]
            if not type_ok:
                errors.append(f"column '{col}' expected type '{expected}' but has dtype {series.dtype}")
                continue
            non_null = series.dropna()
            if not rules.get("nullable", True) and len(non_null) < len(series):
                errors.append(f"column '{col}' contains missing values")
            if "min" in rules and len(non_null) and (non_null < rules["min"]).any():
                errors.append(f"column '{col}' has values below the minimum {rules['min']}")
            if "max" in rules and len(non_null) and (non_null > rules["max"]).any():
                errors.append(f"column '{col}' has values above the maximum {rules['max']}")
            if "categories" in rules:
                invalid = sorted({str(v) for v in non_null.unique() if v not in set(rules["categories"])})
                if invalid:
                    errors.append(f"column '{col}' has unexpected categories {invalid}")
            if rules.get("unique", False) and non_null.duplicated().any():
                errors.append(f"column '{col}' has duplicate values")
        return errors

    def _run(self, frame: pd.DataFrame) -> None:
        errors = self._validate(frame)
        self.validation_errors_ = errors
        if errors:
            message = f"DataValidator found {len(errors)} violation(s): {'; '.join(errors)}"
            if self.raise_on_error:
                raise ValueError(message)
            self.logger.warning(message)

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> DataValidator:
        """Validate ``X`` against the schema.

        Args:
            X: DataFrame to validate.
            y: Ignored.

        Returns:
            Fitted validator.

        Raises:
            ValueError: If the data violates the schema and ``raise_on_error`` is set.
        """
        frame = _require_dataframe(X, "DataValidator")
        self._run(frame)
        self._record_input(X, frame)
        self.feature_names_out_ = list(frame.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate ``X`` and return it unchanged.

        Args:
            X: DataFrame to validate.

        Returns:
            ``X`` itself.
        """
        check_is_fitted(self)
        frame = _require_dataframe(X, "DataValidator")
        self._run(frame)
        return frame


class PipelineDebugger(_BaseTransformer):
    """Pass-through step that logs and records data characteristics.

    Args:
        step_name: Label used in the log messages.
        log_level: Logging level name for the messages.
        log_shape: Record the data shape.
        log_dtypes: Record dtype counts (DataFrame) or the array dtype.
        log_missing: Record the number of missing values.
        log_stats: Record summary statistics of numeric data.

    Attributes:
        debug_info_: Information captured during the last ``fit``/``transform`` call.
        history_: List of every captured snapshot.
    """

    def __init__(
        self,
        step_name: str = "debug",
        log_level: str = "INFO",
        log_shape: bool = True,
        log_dtypes: bool = True,
        log_missing: bool = True,
        log_stats: bool = False,
    ):
        self.step_name = step_name
        self.log_level = log_level
        self.log_shape = log_shape
        self.log_dtypes = log_dtypes
        self.log_missing = log_missing
        self.log_stats = log_stats

    def _inspect(self, X: Any, y: Optional[Any], phase: str) -> Dict[str, Any]:
        info: Dict[str, Any] = {"step_name": self.step_name, "phase": phase, "type": type(X).__name__}
        shape = getattr(X, "shape", None)
        if self.log_shape:
            info["shape"] = tuple(shape) if shape is not None else None
        if self.log_dtypes:
            if isinstance(X, pd.DataFrame):
                info["dtypes"] = {str(k): int(v) for k, v in X.dtypes.astype(str).value_counts().items()}
            elif hasattr(X, "dtype"):
                info["dtypes"] = {str(X.dtype): int(shape[1]) if shape is not None and len(shape) > 1 else 1}
        if self.log_missing:
            if isinstance(X, pd.DataFrame):
                info["missing"] = int(X.isnull().sum().sum())
            elif isinstance(X, np.ndarray) and X.dtype.kind in "fc":
                info["missing"] = int(np.isnan(X).sum())
        if self.log_stats:
            numeric = X.select_dtypes(include=[np.number]) if isinstance(X, pd.DataFrame) else X
            if isinstance(numeric, pd.DataFrame) and numeric.shape[1]:
                info["stats"] = numeric.describe().to_dict()
            elif isinstance(numeric, np.ndarray) and numeric.dtype.kind in "biufc" and numeric.size:
                info["stats"] = {
                    "mean": float(np.nanmean(numeric)),
                    "std": float(np.nanstd(numeric)),
                    "min": float(np.nanmin(numeric)),
                    "max": float(np.nanmax(numeric)),
                }
        if y is not None:
            y_array = np.asarray(y)
            target: Dict[str, Any] = {"shape": y_array.shape}
            try:
                n_unique = len(np.unique(y_array))
                target["n_unique"] = n_unique
                if n_unique <= 10:
                    target["distribution"] = {
                        str(k): int(v) for k, v in pd.Series(y_array).value_counts().items()
                    }
            except TypeError:
                pass
            info["target"] = target
        level = logging.getLevelName(str(self.log_level).upper())
        if not isinstance(level, int):
            level = logging.INFO
        self.logger.log(
            level, "[%s] %s: %s", self.step_name, phase, {k: v for k, v in info.items() if k != "stats"}
        )
        return info

    def fit(self, X: Any, y: Optional[Any] = None) -> PipelineDebugger:
        """Record information about ``X`` (and ``y``).

        Args:
            X: Data at this pipeline position.
            y: Optional target.

        Returns:
            Fitted debugger.
        """
        self.history_ = []
        self.debug_info_ = self._inspect(X, y, "fit")
        self.history_.append(self.debug_info_)
        shape = getattr(X, "shape", None)
        self.n_features_in_ = int(shape[1]) if shape is not None and len(shape) > 1 else 0
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            self.feature_names_out_ = list(X.columns)
        else:
            self.feature_names_out_ = [f"x{i}" for i in range(self.n_features_in_)]
        return self

    def transform(self, X: Any) -> Any:
        """Record information about ``X`` and return it unchanged.

        Args:
            X: Data at this pipeline position.

        Returns:
            ``X`` itself.
        """
        check_is_fitted(self)
        self.debug_info_ = self._inspect(X, None, "transform")
        self.history_.append(self.debug_info_)
        return X


# --------------------------------------------------------------------------- #
# Legacy pipeline-factory transformers
# --------------------------------------------------------------------------- #
class DomainSpecificEncoder(_BaseTransformer):
    """Encode a mixed DataFrame into a numeric matrix with cardinality-aware strategies.

    Low-cardinality categorical columns use ``categorical_strategy``; high
    cardinality columns use target encoding when ``y`` is available and
    feature hashing otherwise. Numeric columns are passed through first.

    Args:
        categorical_strategy: ``'onehot'``, ``'target'`` or ``'ordinal'`` for low-cardinality columns.
        ordinal_strategy: Kept for API compatibility (ordinal encoding is always integer codes).
        high_cardinality_threshold: Distinct-value count above which a column is high cardinality.
        rare_category_threshold: Minimum category frequency (fraction) before a
            category is grouped as infrequent by the one-hot/ordinal encoders.
        n_hash_features: Maximum hashed features for high-cardinality columns without a target.

    Attributes:
        categorical_features_: Encoded categorical columns.
        numerical_features_: Pass-through numeric columns.
        encoders_: Fitted encoder per categorical column.
        encoder_kinds_: ``'onehot'``, ``'ordinal'``, ``'target'`` or ``'hash'`` per column.
    """

    def __init__(
        self,
        categorical_strategy: str = "onehot",
        ordinal_strategy: str = "ordinal",
        high_cardinality_threshold: int = 10,
        rare_category_threshold: float = 0.01,
        n_hash_features: int = 32,
    ):
        self.categorical_strategy = categorical_strategy
        self.ordinal_strategy = ordinal_strategy
        self.high_cardinality_threshold = high_cardinality_threshold
        self.rare_category_threshold = rare_category_threshold
        self.n_hash_features = n_hash_features

    def _fit_column(self, series: pd.Series, y: Optional[np.ndarray]) -> Tuple[str, Any, List[str]]:
        col = str(series.name)
        n_unique = series.nunique()
        min_frequency = self.rare_category_threshold if 0 < self.rare_category_threshold < 1 else None
        low_cardinality = n_unique <= self.high_cardinality_threshold
        if low_cardinality and self.categorical_strategy == "onehot":
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", min_frequency=min_frequency)
            encoder.fit(series.to_frame())
            return "onehot", encoder, [str(n) for n in encoder.get_feature_names_out([col])]
        if y is not None and (not low_cardinality or self.categorical_strategy == "target"):
            mapping, prior = _target_mean_mapping(series, y, smoothing=0.0)
            return "target", (mapping, prior), [f"{col}_target"]
        if low_cardinality:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-2,
                min_frequency=min_frequency,
            )
            encoder.fit(series.to_frame())
            return "ordinal", encoder, [f"{col}_ordinal"]
        n_features = max(1, min(self.n_hash_features, int(n_unique)))
        hasher = FeatureHasher(n_features=n_features, input_type="string")
        return "hash", hasher, [f"{col}_hash_{i}" for i in range(n_features)]

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> DomainSpecificEncoder:
        """Fit one encoder per categorical column.

        Args:
            X: DataFrame with mixed column types.
            y: Optional numeric target enabling target encoding.

        Returns:
            Fitted encoder.

        Raises:
            TypeError: If ``X`` is not a DataFrame.
        """
        frame = _require_dataframe(X, "DomainSpecificEncoder")
        self._record_input(X, frame)
        target: Optional[np.ndarray] = None
        if y is not None:
            try:
                target = np.asarray(y, dtype=float).ravel()
            except (TypeError, ValueError):
                self.logger.warning("DomainSpecificEncoder: non-numeric target; target encoding disabled")
        self.categorical_features_ = _columns_of_kind(frame, ("categorical",))
        self.numerical_features_ = [col for col in frame.columns if col not in self.categorical_features_]
        self.encoders_ = {}
        self.encoder_kinds_ = {}
        names = [str(col) for col in self.numerical_features_]
        for col in self.categorical_features_:
            kind, encoder, col_names = self._fit_column(frame[col], target)
            self.encoder_kinds_[col] = kind
            self.encoders_[col] = encoder
            names.extend(col_names)
        self.feature_names_out_ = names
        self.logger.info(
            "DomainSpecificEncoder: %d categorical and %d numerical features",
            len(self.categorical_features_),
            len(self.numerical_features_),
        )
        return self

    def _transform_column(self, series: pd.Series) -> np.ndarray:
        kind = self.encoder_kinds_[series.name]
        encoder = self.encoders_[series.name]
        if kind in ("onehot", "ordinal"):
            return np.asarray(encoder.transform(series.to_frame()), dtype=float)
        if kind == "target":
            mapping, prior = encoder
            return series.map(mapping).astype(float).fillna(prior).to_numpy().reshape(-1, 1)
        return encoder.transform([[str(v)] for v in series]).toarray()

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Encode the DataFrame into a dense numeric matrix.

        Args:
            X: DataFrame with the fitted columns.

        Returns:
            Numeric matrix ``(n_samples, n_output_features)``.
        """
        check_is_fitted(self)
        frame = _require_dataframe(X, "DomainSpecificEncoder")
        _check_columns(frame, self.numerical_features_ + self.categorical_features_, "DomainSpecificEncoder")
        blocks: List[np.ndarray] = []
        if self.numerical_features_:
            blocks.append(frame[self.numerical_features_].to_numpy(dtype=float))
        blocks.extend(self._transform_column(frame[col]) for col in self.categorical_features_)
        if not blocks:
            return np.empty((len(frame), 0))
        return np.hstack(blocks)


class AdvancedImputer(_BaseTransformer):
    """Impute numeric and categorical columns with separate strategies.

    Args:
        numerical_strategy: ``'mean'``, ``'median'``, ``'most_frequent'``, ``'knn'`` or ``'iterative'``.
        categorical_strategy: ``'mode'`` or ``'constant'`` (fills ``'missing'``).
        n_neighbors: Neighbourhood size for ``'knn'``.
        random_state: Seed for ``'iterative'``.

    Attributes:
        numerical_features_: Numeric columns (``None`` for ndarray input).
        categorical_features_: Categorical columns.
        numerical_imputer_: Fitted scikit-learn imputer for numeric data.
        categorical_imputer_: ``{column: fill_value}`` for categorical data.
    """

    def __init__(
        self,
        numerical_strategy: str = "knn",
        categorical_strategy: str = "mode",
        n_neighbors: int = 5,
        random_state: Optional[int] = 42,
    ):
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def _make_numerical_imputer(self) -> BaseEstimator:
        if self.numerical_strategy == "knn":
            return KNNImputer(n_neighbors=self.n_neighbors)
        if self.numerical_strategy == "iterative":
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer

            return IterativeImputer(random_state=self.random_state)
        return SimpleImputer(strategy=self.numerical_strategy)

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> AdvancedImputer:
        """Fit the imputers.

        Args:
            X: Feature matrix (ndarray input is treated as fully numeric).
            y: Ignored.

        Returns:
            Fitted imputer.
        """
        self.numerical_imputer_ = None
        self.categorical_imputer_ = {}
        if isinstance(X, pd.DataFrame):
            self._record_input(X, X)
            self.numerical_features_ = _columns_of_kind(X, ("numeric",))
            self.categorical_features_ = _columns_of_kind(X, ("categorical",))
            if self.numerical_features_:
                self.numerical_imputer_ = self._make_numerical_imputer().fit(
                    X[self.numerical_features_].astype(float)
                )
            for col in self.categorical_features_:
                if self.categorical_strategy == "mode":
                    modes = X[col].mode(dropna=True)
                    self.categorical_imputer_[col] = modes.iloc[0] if len(modes) else "unknown"
                else:
                    self.categorical_imputer_[col] = "missing"
            self.feature_names_out_ = list(X.columns)
        else:
            array = check_array(X, ensure_all_finite=False)
            self._record_input(X, pd.DataFrame(array))
            self.numerical_features_ = None
            self.categorical_features_ = []
            self.numerical_imputer_ = self._make_numerical_imputer().fit(array)
            self.feature_names_out_ = [f"x{i}" for i in range(array.shape[1])]
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Impute missing values.

        Args:
            X: Feature matrix.

        Returns:
            Imputed data (same container type as the input).
        """
        check_is_fitted(self)
        if isinstance(X, pd.DataFrame):
            out = X.copy()
            if self.numerical_features_ and self.numerical_imputer_ is not None:
                _check_columns(X, self.numerical_features_, "AdvancedImputer")
                out[self.numerical_features_] = self.numerical_imputer_.transform(
                    X[self.numerical_features_].astype(float)
                )
            for col, value in self.categorical_imputer_.items():
                if col in out.columns:
                    out[col] = out[col].fillna(value)
            return out
        array = check_array(X, ensure_all_finite=False)
        return self.numerical_imputer_.transform(array) if self.numerical_imputer_ is not None else array


class FeatureScaler(_BaseTransformer):
    """Scale each feature with a strategy chosen from its distribution.

    Args:
        strategy: ``'auto'``, ``'standard'``, ``'robust'``, ``'minmax'``,
            ``'quantile'`` or ``'none'``. ``'auto'`` picks robust scaling for
            outlier-heavy features, quantile scaling for strongly skewed ones,
            no scaling for features already in ``[0, 1]`` and standard scaling otherwise.
        robust_threshold: Outlier fraction above which ``'auto'`` chooses robust scaling.
        random_state: Seed for the quantile transformer.

    Attributes:
        scalers_: ``{feature_index: fitted scaler}``.
        feature_strategies_: ``{feature_index: strategy}``.
    """

    _STRATEGIES = ("auto", "standard", "robust", "minmax", "quantile", "none")

    def __init__(
        self, strategy: str = "auto", robust_threshold: float = 0.1, random_state: Optional[int] = 42
    ):
        self.strategy = strategy
        self.robust_threshold = robust_threshold
        self.random_state = random_state

    def _numeric_values(self, X: ArrayLike) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X[_columns_of_kind(X, ("numeric", "bool"))].to_numpy(dtype=float)
        return check_array(X, dtype=float, ensure_all_finite=False)

    def _choose_strategy(self, feature: np.ndarray) -> str:
        finite = feature[~np.isnan(feature)]
        if finite.size == 0:
            return "none"
        lower, upper = _iqr_bounds(finite.reshape(-1, 1), 1.5)
        outlier_fraction = float(np.mean((finite < lower[0]) | (finite > upper[0])))
        if outlier_fraction > self.robust_threshold:
            return "robust"
        if finite.min() >= 0 and finite.max() <= 1:
            return "none"
        if finite.size > 2 and abs(float(scipy_stats.skew(finite))) > 2:
            return "quantile"
        return "standard"

    def _make_scaler(self, strategy: str, n_samples: int) -> Optional[BaseEstimator]:
        if strategy == "standard":
            return StandardScaler()
        if strategy == "robust":
            return RobustScaler()
        if strategy == "minmax":
            return MinMaxScaler()
        if strategy == "quantile":
            return QuantileTransformer(n_quantiles=min(1000, n_samples), random_state=self.random_state)
        return None

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> FeatureScaler:
        """Fit one scaler per feature.

        Args:
            X: Feature matrix (numeric columns only are used for DataFrame input).
            y: Ignored.

        Returns:
            Fitted scaler.
        """
        if self.strategy not in self._STRATEGIES:
            raise ValueError(f"Unknown scaling strategy '{self.strategy}'. Choose from {self._STRATEGIES}.")
        values = self._numeric_values(X)
        self._record_input(X, pd.DataFrame(values))
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            self.feature_names_out_ = list(_columns_of_kind(X, ("numeric", "bool")))
        else:
            self.feature_names_out_ = [f"x{i}" for i in range(values.shape[1])]
        self.scalers_ = {}
        self.feature_strategies_ = {}
        for i in range(values.shape[1]):
            feature = values[:, i]
            strategy = self._choose_strategy(feature) if self.strategy == "auto" else self.strategy
            self.feature_strategies_[i] = strategy
            scaler = self._make_scaler(strategy, len(feature))
            if scaler is not None:
                self.scalers_[i] = scaler.fit(feature.reshape(-1, 1))
        self.logger.debug("FeatureScaler strategies: %s", self.feature_strategies_)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Scale the numeric features.

        Args:
            X: Feature matrix.

        Returns:
            Scaled numeric matrix.
        """
        check_is_fitted(self)
        values = self._numeric_values(X)
        if values.shape[1] != len(self.feature_strategies_):
            raise ValueError(
                f"FeatureScaler was fitted on {len(self.feature_strategies_)} features, got {values.shape[1]}."
            )
        scaled = values.copy()
        for i, scaler in self.scalers_.items():
            scaled[:, i] = scaler.transform(values[:, i].reshape(-1, 1)).ravel()
        return scaled


class TimeSeriesFeatureCreator(_BaseTransformer):
    """Create calendar, cyclical, lag and rolling features from a datetime column.

    Rows are assumed to be ordered in time; lag and rolling features of the
    first rows are ``NaN``/partial by construction.

    Args:
        datetime_column: Name of the datetime column.
        create_lags: Add ``<col>_lag_<p>`` for every value column and lag period.
        lag_periods: Lag periods; defaults to ``[1, 7, 30]``.
        create_rolling: Add rolling mean/std ``<col>_rolling_{mean,std}_<w>``.
        rolling_windows: Rolling windows; defaults to ``[7, 30]``.
        create_cyclical: Add sin/cos encodings of month, weekday and day of year.
        value_columns: Numeric columns receiving lag/rolling features; defaults
            to every numeric column except the datetime column.

    Attributes:
        lag_periods_, rolling_windows_, value_columns_: Resolved settings.
    """

    _CALENDAR = ("year", "month", "day", "dayofweek", "dayofyear", "quarter")

    def __init__(
        self,
        datetime_column: str = "date",
        create_lags: bool = True,
        lag_periods: Optional[List[int]] = None,
        create_rolling: bool = True,
        rolling_windows: Optional[List[int]] = None,
        create_cyclical: bool = True,
        value_columns: Optional[List[str]] = None,
    ):
        self.datetime_column = datetime_column
        self.create_lags = create_lags
        self.lag_periods = lag_periods
        self.create_rolling = create_rolling
        self.rolling_windows = rolling_windows
        self.create_cyclical = create_cyclical
        self.value_columns = value_columns

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> TimeSeriesFeatureCreator:
        """Resolve settings and output names.

        Args:
            X: DataFrame with the datetime column.
            y: Ignored.

        Returns:
            Fitted transformer.
        """
        frame = _require_dataframe(X, "TimeSeriesFeatureCreator")
        _check_columns(frame, [self.datetime_column], "TimeSeriesFeatureCreator")
        self._record_input(X, frame)
        self.lag_periods_ = list(self.lag_periods) if self.lag_periods is not None else [1, 7, 30]
        self.rolling_windows_ = list(self.rolling_windows) if self.rolling_windows is not None else [7, 30]
        if self.value_columns is not None:
            self.value_columns_ = _check_numeric_columns(
                frame, self.value_columns, "TimeSeriesFeatureCreator"
            )
        else:
            self.value_columns_ = [
                c for c in _columns_of_kind(frame, ("numeric",)) if c != self.datetime_column
            ]

        names = [str(col) for col in frame.columns]
        names.extend(self._CALENDAR)
        names.extend(["is_weekend", "is_month_start", "is_month_end"])
        if self.create_cyclical:
            names.extend(
                ["month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos", "dayofyear_sin", "dayofyear_cos"]
            )
        names.append("days_since_epoch")
        for col in self.value_columns_:
            if self.create_lags:
                names.extend(f"{col}_lag_{lag}" for lag in self.lag_periods_)
            if self.create_rolling:
                for window in self.rolling_windows_:
                    names.extend([f"{col}_rolling_mean_{window}", f"{col}_rolling_std_{window}"])
        self.feature_names_out_ = names
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Append the time-series features.

        Args:
            X: DataFrame with the datetime column.

        Returns:
            Copy of ``X`` with the new columns appended.
        """
        check_is_fitted(self)
        frame = _require_dataframe(X, "TimeSeriesFeatureCreator")
        _check_columns(frame, [self.datetime_column, *self.value_columns_], "TimeSeriesFeatureCreator")
        out = frame.copy()
        dt = frame[self.datetime_column]
        if not pd.api.types.is_datetime64_any_dtype(dt):
            dt = pd.to_datetime(dt)
        for feature in self._CALENDAR:
            out[feature] = getattr(dt.dt, feature)
        out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        out["is_month_start"] = dt.dt.is_month_start.astype(int)
        out["is_month_end"] = dt.dt.is_month_end.astype(int)
        if self.create_cyclical:
            for feature, period in (("month", 12), ("dayofweek", 7), ("dayofyear", 365.25)):
                angle = 2.0 * np.pi * out[feature].astype(float) / period
                out[f"{feature}_sin"] = np.sin(angle)
                out[f"{feature}_cos"] = np.cos(angle)
        out["days_since_epoch"] = (dt - pd.Timestamp("1970-01-01")).dt.days
        for col in self.value_columns_:
            series = frame[col].astype(float)
            if self.create_lags:
                for lag in self.lag_periods_:
                    out[f"{col}_lag_{lag}"] = series.shift(lag)
            if self.create_rolling:
                for window in self.rolling_windows_:
                    rolling = series.rolling(window=window, min_periods=1)
                    out[f"{col}_rolling_mean_{window}"] = rolling.mean()
                    out[f"{col}_rolling_std_{window}"] = rolling.std()
        return out


class TextFeatureExtractor(_BaseTransformer):
    """Turn text columns into a numeric matrix (vectorised terms plus length features).

    Args:
        text_columns: Text columns; defaults to every non-numeric column.
        max_features: Vocabulary cap per column.
        ngram_range: N-gram range.
        use_tfidf: TF-IDF weighting instead of raw counts.
        extract_length_features: Append character and word counts per column.

    Attributes:
        text_columns_: Vectorised columns.
        vectorizers_: Fitted vectoriser per column.
        numeric_columns_: Non-text numeric columns passed through first.
    """

    def __init__(
        self,
        text_columns: Optional[List[str]] = None,
        max_features: int = 1000,
        ngram_range: Tuple[int, int] = (1, 2),
        use_tfidf: bool = True,
        extract_length_features: bool = True,
    ):
        self.text_columns = text_columns
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.use_tfidf = use_tfidf
        self.extract_length_features = extract_length_features

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> TextFeatureExtractor:
        """Fit one vectoriser per text column.

        Args:
            X: DataFrame holding the text columns.
            y: Ignored.

        Returns:
            Fitted extractor.
        """
        frame = _require_dataframe(X, "TextFeatureExtractor")
        if self.text_columns is None:
            columns = _columns_of_kind(frame, ("categorical",))
        else:
            columns = _check_columns(frame, self.text_columns, "TextFeatureExtractor")
        self._record_input(X, frame)
        self.text_columns_ = list(columns)
        self.numeric_columns_ = [
            c for c in _columns_of_kind(frame, ("numeric", "bool")) if c not in self.text_columns_
        ]
        vectorizer_cls = TfidfVectorizer if self.use_tfidf else CountVectorizer
        self.vectorizers_ = {}
        names = [str(c) for c in self.numeric_columns_]
        for col in self.text_columns_:
            vectorizer = vectorizer_cls(
                max_features=self.max_features, ngram_range=self.ngram_range, stop_words="english"
            )
            self.vectorizers_[col] = vectorizer.fit(frame[col].fillna("").astype(str))
            names.extend(f"{col}_{term}" for term in vectorizer.get_feature_names_out())
            if self.extract_length_features:
                names.extend([f"{col}_char_count", f"{col}_word_count"])
        self.feature_names_out_ = names
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Vectorise the text columns.

        Args:
            X: DataFrame holding the text columns.

        Returns:
            Dense numeric matrix.
        """
        check_is_fitted(self)
        frame = _require_dataframe(X, "TextFeatureExtractor")
        _check_columns(frame, self.numeric_columns_ + self.text_columns_, "TextFeatureExtractor")
        blocks: List[np.ndarray] = []
        if self.numeric_columns_:
            blocks.append(frame[self.numeric_columns_].to_numpy(dtype=float))
        for col in self.text_columns_:
            text = frame[col].fillna("").astype(str)
            blocks.append(self.vectorizers_[col].transform(text).toarray())
            if self.extract_length_features:
                blocks.append(text.str.len().to_numpy(dtype=float).reshape(-1, 1))
                blocks.append(text.str.split().str.len().fillna(0).to_numpy(dtype=float).reshape(-1, 1))
        if not blocks:
            return np.empty((len(frame), 0))
        return np.hstack(blocks)
