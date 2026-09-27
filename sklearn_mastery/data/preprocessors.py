"""Preprocessing transformers for tabular data.

This module provides sklearn-compatible transformers that cover the common
preprocessing chores of a modelling workflow:

* :class:`DataPreprocessor` - an end-to-end, configurable pipeline
  (imputation, categorical encoding, outlier detection, scaling, feature
  selection and dimensionality reduction) that returns a dense ``ndarray``.
* :class:`CategoricalEncoder` - cardinality-aware encoding of categorical
  columns (one-hot, ordinal/label, smoothed target and binary encoding).
* :class:`NumericalTransformer` - skewness-driven log / sqrt / Box-Cox
  transforms plus polynomial and pairwise interaction features.
* :class:`ImbalancedDataHandler` - resampling helper (SMOTE / random
  over- and under-sampling) with an ``imbalanced-learn`` optional dependency.

All transformers follow the scikit-learn estimator contract: constructor
arguments are stored verbatim, ``fit`` returns ``self``, learned state ends in
a trailing underscore and ``get_feature_names_out`` is available.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import stats
from scipy.special import boxcox as _boxcox_transform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils.multiclass import type_of_target

from sklearn_mastery.config.logging_config import LoggerMixin

try:  # optional dependency
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    HAS_IMBLEARN = True
except ImportError:  # pragma: no cover - exercised only without imblearn installed
    HAS_IMBLEARN = False

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series, Sequence[Sequence[Any]]]

__all__ = [
    "HAS_IMBLEARN",
    "CategoricalEncoder",
    "DataPreprocessor",
    "ImbalancedDataHandler",
    "NumericalTransformer",
]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _to_dataframe(X: ArrayLike, feature_prefix: str = "feature") -> pd.DataFrame:
    """Coerce array-like input to a DataFrame with string column names.

    Args:
        X: 2-D array-like, DataFrame or Series.
        feature_prefix: Prefix used to name columns of array inputs.

    Returns:
        A copy of the input as a DataFrame whose column labels are strings.

    Raises:
        ValueError: If the input is not 1-D or 2-D.
    """
    if isinstance(X, pd.DataFrame):
        frame = X.copy()
    elif isinstance(X, pd.Series):
        frame = X.to_frame()
    else:
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2-D input, got array with ndim={arr.ndim}.")
        frame = pd.DataFrame(arr, columns=[f"{feature_prefix}_{i}" for i in range(arr.shape[1])])
    frame.columns = [str(c) for c in frame.columns]
    return frame


def _check_non_empty(X: pd.DataFrame, estimator_name: str) -> None:
    """Raise if ``X`` has no rows or no columns."""
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"{estimator_name} requires a non-empty 2-D input; got shape {X.shape}.")


def _split_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split columns into numeric and categorical (everything else) groups."""
    numeric = [c for c in X.columns if is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def _check_columns_present(X: pd.DataFrame, required: Sequence[str], estimator_name: str) -> None:
    """Raise if any of ``required`` columns is missing from ``X``."""
    missing = [c for c in required if c not in X.columns]
    if missing:
        raise ValueError(f"{estimator_name}.transform is missing columns seen during fit: {missing}")


def _target_to_float(y: ArrayLike, n_samples: int) -> np.ndarray:
    """Return ``y`` as a 1-D float array, label-encoding non-numeric targets."""
    y_arr = np.asarray(y).ravel()
    if y_arr.shape[0] != n_samples:
        raise ValueError(f"y has {y_arr.shape[0]} samples but X has {n_samples}.")
    if np.issubdtype(y_arr.dtype, np.number) or np.issubdtype(y_arr.dtype, np.bool_):
        return y_arr.astype(float)
    return LabelEncoder().fit_transform(y_arr.astype(str)).astype(float)


# --------------------------------------------------------------------------- #
# Private per-column encoders used by CategoricalEncoder
# --------------------------------------------------------------------------- #
class _SmoothedTargetEncoder:
    """Mean target encoding with additive (m-estimate) smoothing.

    ``encoding = (sum_y_in_category + m * prior) / (count_in_category + m)``.
    """

    def __init__(self, smoothing: float, handle_unknown: str) -> None:
        self.smoothing = smoothing
        self.handle_unknown = handle_unknown

    def fit(self, values: np.ndarray, y: np.ndarray) -> _SmoothedTargetEncoder:
        frame = pd.DataFrame({"category": values.ravel(), "target": y})
        self.prior_ = float(frame["target"].mean())
        grouped = frame.groupby("category", sort=True)["target"].agg(["sum", "count"])
        smoothed = (grouped["sum"] + self.smoothing * self.prior_) / (grouped["count"] + self.smoothing)
        self.mapping_ = {str(k): float(v) for k, v in smoothed.items()}
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        flat = values.ravel()
        if self.handle_unknown == "error":
            unknown = sorted({str(v) for v in flat if str(v) not in self.mapping_})
            if unknown:
                raise ValueError(f"Found unknown categories {unknown} during transform.")
        return np.array([self.mapping_.get(str(v), self.prior_) for v in flat], dtype=float).reshape(-1, 1)


class _BinaryEncoder:
    """Binary encoding: ordinal code (1-based) written in ``ceil(log2(n + 1))`` bits.

    Unknown categories receive code 0, i.e. an all-zero bit pattern.
    """

    def __init__(self, handle_unknown: str) -> None:
        self.handle_unknown = handle_unknown

    def fit(self, values: np.ndarray) -> _BinaryEncoder:
        self.ordinal_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(values)
        n_categories = len(self.ordinal_.categories_[0])
        self.n_bits_ = max(1, math.ceil(math.log2(n_categories + 1)))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        codes = self.ordinal_.transform(values).ravel().astype(int)
        if self.handle_unknown == "error" and np.any(codes < 0):
            unknown = sorted({str(v) for v, c in zip(values.ravel(), codes) if c < 0})
            raise ValueError(f"Found unknown categories {unknown} during transform.")
        codes = codes + 1
        bit_positions = np.arange(self.n_bits_)
        return ((codes[:, None] >> bit_positions) & 1).astype(np.int64)


# --------------------------------------------------------------------------- #
# CategoricalEncoder
# --------------------------------------------------------------------------- #
class CategoricalEncoder(BaseEstimator, TransformerMixin, LoggerMixin):
    """Encode categorical columns with a per-column strategy.

    With ``strategy="auto"`` the strategy is chosen from each column's
    cardinality: at most ``onehot_max_categories`` unique values -> one-hot;
    at most ``max_cardinality`` unique values -> target encoding when ``y`` is
    given (label encoding otherwise); anything larger -> label encoding.

    Missing values are treated as their own category (``missing_value``).
    Non-categorical columns are passed through untouched, and the output is
    always a DataFrame.

    Args:
        strategy: ``"auto"``, ``"onehot"``, ``"label"``, ``"target"`` or ``"binary"``.
        handle_unknown: ``"ignore"`` maps unseen categories to a neutral code
            (all-zeros for one-hot/binary, ``-1`` for label, the prior for
            target encoding); ``"error"`` raises.
        max_cardinality: Upper bound for target encoding under ``"auto"``.
        onehot_max_categories: Upper bound for one-hot encoding under ``"auto"``.
        columns: Explicit list of columns to encode; ``None`` auto-detects
            non-numeric columns.
        target_smoothing: Smoothing weight ``m`` for target encoding.
        missing_value: Category label substituted for missing values.

    Attributes:
        encoders_: Mapping column -> fitted per-column encoder.
        strategies_: Mapping column -> resolved strategy.
        feature_names_out_: Output column names in order.
        is_fitted_: Whether :meth:`fit` has been called.
    """

    _VALID_STRATEGIES = ("auto", "onehot", "label", "target", "binary")
    _VALID_UNKNOWN = ("ignore", "error")

    def __init__(
        self,
        strategy: str = "auto",
        handle_unknown: str = "ignore",
        max_cardinality: int = 50,
        onehot_max_categories: int = 10,
        columns: Optional[Sequence[str]] = None,
        target_smoothing: float = 10.0,
        missing_value: str = "missing",
    ) -> None:
        self.strategy = strategy
        self.handle_unknown = handle_unknown
        self.max_cardinality = max_cardinality
        self.onehot_max_categories = onehot_max_categories
        self.columns = columns
        self.target_smoothing = target_smoothing
        self.missing_value = missing_value
        self.encoders_: Dict[str, Any] = {}
        self.strategies_: Dict[str, str] = {}
        self.is_fitted_ = False

    def __sklearn_is_fitted__(self) -> bool:
        return self.is_fitted_

    # ----------------------------------------------------------------- utils
    def _validate_params(self) -> None:
        if self.strategy not in self._VALID_STRATEGIES:
            raise ValueError(f"strategy must be one of {self._VALID_STRATEGIES}, got {self.strategy!r}.")
        if self.handle_unknown not in self._VALID_UNKNOWN:
            raise ValueError(
                f"handle_unknown must be one of {self._VALID_UNKNOWN}, got {self.handle_unknown!r}."
            )
        if self.max_cardinality < 1 or self.onehot_max_categories < 1:
            raise ValueError("max_cardinality and onehot_max_categories must be >= 1.")

    def _prepare_values(self, series: pd.Series) -> np.ndarray:
        """Return a ``(n, 1)`` object array of strings with missing values filled."""
        values = series.astype(object)
        values = values.where(values.notna(), self.missing_value)
        return values.map(str).to_numpy(dtype=object).reshape(-1, 1)

    def _resolve_strategy(self, n_unique: int, has_target: bool) -> str:
        if self.strategy != "auto":
            return self.strategy
        if n_unique <= self.onehot_max_categories:
            return "onehot"
        if n_unique <= self.max_cardinality and has_target:
            return "target"
        return "label"

    def _build_encoder(self, strategy: str) -> Any:
        if strategy == "onehot":
            return OneHotEncoder(handle_unknown=self.handle_unknown, sparse_output=False, dtype=np.int64)
        if strategy == "label":
            if self.handle_unknown == "ignore":
                return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            return OrdinalEncoder(handle_unknown="error")
        if strategy == "target":
            return _SmoothedTargetEncoder(self.target_smoothing, self.handle_unknown)
        return _BinaryEncoder(self.handle_unknown)

    def _output_names(self, column: str, strategy: str, encoder: Any) -> List[str]:
        if strategy == "onehot":
            return [f"{column}_{category}" for category in encoder.categories_[0]]
        if strategy == "binary":
            return [f"{column}_bin{i}" for i in range(encoder.n_bits_)]
        return [column]

    # ------------------------------------------------------------------ API
    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> CategoricalEncoder:
        """Learn per-column strategies and encoders.

        Args:
            X: Input features.
            y: Optional target, required for target encoding.

        Returns:
            The fitted encoder.
        """
        self._validate_params()
        frame = _to_dataframe(X)
        _check_non_empty(frame, self.__class__.__name__)
        if self.columns is not None:
            columns = [str(c) for c in self.columns]
            _check_columns_present(frame, columns, self.__class__.__name__)
        else:
            columns = _split_columns(frame)[1]

        y_float = None if y is None else _target_to_float(y, frame.shape[0])
        if self.strategy == "target" and y_float is None:
            raise ValueError("strategy='target' requires y to be passed to fit.")

        self.encoders_ = {}
        self.strategies_ = {}
        names_out: List[str] = []
        for column in frame.columns:
            if column not in columns:
                names_out.append(column)
                continue
            values = self._prepare_values(frame[column])
            strategy = self._resolve_strategy(len(np.unique(values)), y_float is not None)
            encoder = self._build_encoder(strategy)
            if strategy == "target":
                encoder.fit(values, y_float)
            else:
                encoder.fit(values)
            self.encoders_[column] = encoder
            self.strategies_[column] = strategy
            names_out.extend(self._output_names(column, strategy, encoder))
            self.logger.debug("Column %r encoded with strategy %r", column, strategy)

        self.encoded_columns_ = list(columns)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        self.feature_names_out_ = names_out
        self.is_fitted_ = True
        return self

    def transform(self, X: ArrayLike) -> pd.DataFrame:
        """Encode the categorical columns of ``X``.

        Args:
            X: Input features with the columns seen during fit.

        Returns:
            DataFrame with encoded columns (and pass-through columns) in fit order.

        Raises:
            NotFittedError: If called before :meth:`fit`.
            ValueError: If required columns are missing or unknown categories
                are found with ``handle_unknown="error"``.
        """
        if not self.is_fitted_:
            raise NotFittedError(f"{self.__class__.__name__} must be fitted before calling transform.")
        frame = _to_dataframe(X)
        _check_columns_present(frame, list(self.feature_names_in_), self.__class__.__name__)

        parts: List[pd.DataFrame] = []
        for column in self.feature_names_in_:
            if column not in self.encoders_:
                parts.append(frame[[column]])
                continue
            encoder = self.encoders_[column]
            strategy = self.strategies_[column]
            encoded = encoder.transform(self._prepare_values(frame[column]))
            if strategy == "label":
                encoded = encoded.astype(np.int64)
            names = self._output_names(column, strategy, encoder)
            parts.append(pd.DataFrame(encoded, columns=names, index=frame.index))
        return pd.concat(parts, axis=1)

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        """Return output feature names (``input_features`` is accepted for API parity)."""
        if not self.is_fitted_:
            raise NotFittedError(f"{self.__class__.__name__} must be fitted before get_feature_names_out.")
        return np.asarray(self.feature_names_out_, dtype=object)


# --------------------------------------------------------------------------- #
# NumericalTransformer
# --------------------------------------------------------------------------- #
class NumericalTransformer(BaseEstimator, TransformerMixin, LoggerMixin):
    """Skewness-driven variance-stabilising transforms and feature expansion.

    During ``fit`` each numeric column's sample skewness is measured. Columns
    whose skewness exceeds ``skewness_threshold`` (positive skew) receive the
    first enabled transform in priority order Box-Cox (strictly positive data
    only) > log1p > sqrt; log/sqrt inputs are shifted by the fitted minimum
    when negative values are present. Polynomial powers and pairwise
    interactions are then built from the (transformed) numeric columns.

    Args:
        apply_log: Enable ``log1p`` for skewed columns.
        apply_sqrt: Enable square root for skewed columns.
        apply_boxcox: Enable Box-Cox (lambda fitted by maximum likelihood).
        create_interactions: Add ``a_x_b`` pairwise products.
        create_polynomials: Add ``col_poly{d}`` powers for ``d`` in ``2..degree``.
        polynomial_degree: Highest polynomial degree (>= 2).
        skewness_threshold: Skewness above which a column is transformed.
        columns: Explicit numeric columns to consider; ``None`` uses all numeric.

    Attributes:
        transformations_: Mapping column -> ``"log"``, ``"sqrt"``, ``"boxcox"`` or ``"none"``.
        shifts_: Mapping column -> additive shift applied before log/sqrt.
        boxcox_lambdas_: Mapping column -> fitted Box-Cox lambda.
        feature_names_out_: Output column names.
    """

    def __init__(
        self,
        apply_log: bool = False,
        apply_sqrt: bool = False,
        apply_boxcox: bool = False,
        create_interactions: bool = False,
        create_polynomials: bool = False,
        polynomial_degree: int = 2,
        skewness_threshold: float = 0.5,
        columns: Optional[Sequence[str]] = None,
    ) -> None:
        self.apply_log = apply_log
        self.apply_sqrt = apply_sqrt
        self.apply_boxcox = apply_boxcox
        self.create_interactions = create_interactions
        self.create_polynomials = create_polynomials
        self.polynomial_degree = polynomial_degree
        self.skewness_threshold = skewness_threshold
        self.columns = columns
        self.transformations_: Dict[str, str] = {}
        self.is_fitted_ = False

    def __sklearn_is_fitted__(self) -> bool:
        return self.is_fitted_

    def _select_transform(self, series: pd.Series) -> Tuple[str, float, Optional[float]]:
        """Pick a transform for one column; returns ``(name, shift, lambda)``."""
        skew = series.skew()
        skew = 0.0 if pd.isna(skew) else float(skew)
        minimum = series.min()
        minimum = 0.0 if pd.isna(minimum) else float(minimum)
        if skew <= self.skewness_threshold:
            return "none", 0.0, None
        if self.apply_boxcox and minimum > 0:
            _, lmbda = stats.boxcox(series.dropna().to_numpy(dtype=float))
            return "boxcox", minimum, float(lmbda)
        shift = -minimum if minimum < 0 else 0.0
        if self.apply_log:
            return "log", shift, None
        if self.apply_sqrt:
            return "sqrt", shift, None
        return "none", 0.0, None

    def _apply_transform(self, column: str, values: np.ndarray) -> np.ndarray:
        kind = self.transformations_[column]
        if kind == "log":
            return np.log1p(values + self.shifts_[column])
        if kind == "sqrt":
            return np.sqrt(values + self.shifts_[column])
        if kind == "boxcox":
            clipped = np.clip(values, self.shifts_[column], None)  # shift_ holds the fitted minimum (> 0)
            return _boxcox_transform(clipped, self.boxcox_lambdas_[column])
        return values

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> NumericalTransformer:
        """Measure skewness and decide the transform for each numeric column.

        Args:
            X: Input features.
            y: Ignored; present for API compatibility.

        Returns:
            The fitted transformer.
        """
        if self.create_polynomials and int(self.polynomial_degree) < 2:
            raise ValueError("polynomial_degree must be >= 2 when create_polynomials=True.")
        frame = _to_dataframe(X)
        _check_non_empty(frame, self.__class__.__name__)
        if self.columns is not None:
            columns = [str(c) for c in self.columns]
            _check_columns_present(frame, columns, self.__class__.__name__)
        else:
            columns = _split_columns(frame)[0]

        self.transformations_ = {}
        self.shifts_: Dict[str, float] = {}
        self.boxcox_lambdas_: Dict[str, float] = {}
        for column in columns:
            kind, shift, lmbda = self._select_transform(frame[column].astype(float))
            self.transformations_[column] = kind
            self.shifts_[column] = shift
            if lmbda is not None:
                self.boxcox_lambdas_[column] = lmbda

        self.numeric_columns_ = list(columns)
        self.interaction_pairs_ = list(combinations(columns, 2)) if self.create_interactions else []
        self.polynomial_terms_ = (
            [(c, d) for c in columns for d in range(2, int(self.polynomial_degree) + 1)]
            if self.create_polynomials
            else []
        )
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        self.feature_names_out_ = (
            list(frame.columns)
            + [f"{c}_poly{d}" for c, d in self.polynomial_terms_]
            + [f"{a}_x_{b}" for a, b in self.interaction_pairs_]
        )
        self.is_fitted_ = True
        return self

    def transform(self, X: ArrayLike) -> pd.DataFrame:
        """Apply the fitted transforms and feature expansions.

        Args:
            X: Input features with the columns seen during fit.

        Returns:
            DataFrame with transformed columns followed by polynomial and
            interaction features. Columns without a transform keep their dtype.
        """
        if not self.is_fitted_:
            raise NotFittedError(f"{self.__class__.__name__} must be fitted before calling transform.")
        frame = _to_dataframe(X)
        _check_columns_present(frame, list(self.feature_names_in_), self.__class__.__name__)
        out = frame[list(self.feature_names_in_)].copy()
        for column in self.numeric_columns_:
            if self.transformations_[column] != "none":
                out[column] = self._apply_transform(column, out[column].to_numpy(dtype=float))

        extra: Dict[str, np.ndarray] = {}
        for column, degree in self.polynomial_terms_:
            extra[f"{column}_poly{degree}"] = np.power(out[column].to_numpy(dtype=float), degree)
        for left, right in self.interaction_pairs_:
            extra[f"{left}_x_{right}"] = out[left].to_numpy(dtype=float) * out[right].to_numpy(dtype=float)
        if extra:
            out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)
        return out

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        """Return output feature names (``input_features`` is accepted for API parity)."""
        if not self.is_fitted_:
            raise NotFittedError(f"{self.__class__.__name__} must be fitted before get_feature_names_out.")
        return np.asarray(self.feature_names_out_, dtype=object)


# --------------------------------------------------------------------------- #
# DataPreprocessor
# --------------------------------------------------------------------------- #
class DataPreprocessor(BaseEstimator, TransformerMixin, LoggerMixin):
    """Configurable end-to-end preprocessing pipeline returning a dense array.

    Steps run in this fixed order, each one optional:

    1. ``handle_missing``: median/most-frequent imputation (``"imputer"``).
    2. ``encode_categoricals``: :class:`CategoricalEncoder` (``"encoder"``);
       when disabled, categorical columns are dropped with a warning.
    3. ``create_interactions``: pairwise products via
       :class:`NumericalTransformer` (``"feature_engineer"``).
    4. ``remove_outliers``: :class:`~sklearn.ensemble.IsolationForest`
       (``"outlier_detector"``). Rows flagged at fit time are *excluded from
       fitting* the downstream steps but are never dropped, so ``transform``
       always preserves the number of rows; see ``outlier_mask_``.
    5. ``scale_features``: standard / min-max / robust scaling (``"scaler"``).
    6. ``feature_selection``: variance threshold plus optional univariate or
       mutual-information ``SelectKBest`` when ``y`` is given (``"feature_selector"``).
    7. ``dimensionality_reduction``: PCA (``"dim_reducer"``).

    The names in parentheses are the keys of ``preprocessors_``.

    Args:
        handle_missing: Impute missing values.
        scale_features: Scale numeric features.
        remove_outliers: Detect outliers and fit downstream steps on inliers.
        feature_selection: Apply feature selection.
        dimensionality_reduction: Apply PCA.
        encode_categoricals: Encode non-numeric columns.
        create_interactions: Add pairwise interaction features.
        missing_strategy: Numeric imputation strategy (``"mean"``, ``"median"``,
            ``"most_frequent"``); categorical columns always use most-frequent.
        scaling_method: ``"standard"``, ``"minmax"`` or ``"robust"``.
        outlier_contamination: Expected outlier fraction or ``"auto"``.
        feature_selection_method: ``"univariate"``, ``"mutual_info"`` or ``"variance"``.
        feature_selection_k: Number of features kept by ``SelectKBest`` when
            ``y`` is given; ``None`` keeps at most ``n_features_in_`` features
            (so encoding/interaction expansion is undone by the ranking).
            Without ``y`` only the variance threshold is applied.
        variance_threshold: Minimum variance for a feature to be kept.
        n_components: PCA components (``int``) or explained-variance ratio (``float``).
        handle_outliers: Alias for ``remove_outliers`` (overrides when not ``None``).
        normalize_features: Alias for ``scale_features`` (overrides when not ``None``).
        random_state: Seed for the stochastic steps.

    Attributes:
        preprocessors_: Mapping step key -> fitted transformer.
        steps_applied_: Human-readable list of applied step names.
        is_fitted_: Whether :meth:`fit` has been called.
        feature_names_out_: Output feature names.
        outlier_mask_: Boolean array, ``True`` for inlier rows of the fit data.
        n_outliers_: Number of rows flagged at fit time.
        dropped_columns_: Categorical columns dropped when encoding is disabled.
    """

    _STEP_NAMES = {
        "imputer": "missing_value_imputation",
        "encoder": "categorical_encoding",
        "feature_engineer": "feature_engineering",
        "outlier_detector": "outlier_detection",
        "scaler": "feature_scaling",
        "feature_selector": "feature_selection",
        "dim_reducer": "dimensionality_reduction",
    }

    def __init__(
        self,
        handle_missing: bool = True,
        scale_features: bool = True,
        remove_outliers: bool = False,
        feature_selection: bool = False,
        dimensionality_reduction: bool = False,
        encode_categoricals: bool = True,
        create_interactions: bool = False,
        missing_strategy: str = "median",
        scaling_method: str = "standard",
        outlier_contamination: Union[float, str] = 0.05,
        feature_selection_method: str = "univariate",
        feature_selection_k: Optional[int] = None,
        variance_threshold: float = 0.0,
        n_components: Union[int, float] = 0.95,
        handle_outliers: Optional[bool] = None,
        normalize_features: Optional[bool] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.handle_missing = handle_missing
        self.scale_features = scale_features
        self.remove_outliers = remove_outliers
        self.feature_selection = feature_selection
        self.dimensionality_reduction = dimensionality_reduction
        self.encode_categoricals = encode_categoricals
        self.create_interactions = create_interactions
        self.missing_strategy = missing_strategy
        self.scaling_method = scaling_method
        self.outlier_contamination = outlier_contamination
        self.feature_selection_method = feature_selection_method
        self.feature_selection_k = feature_selection_k
        self.variance_threshold = variance_threshold
        self.n_components = n_components
        self.handle_outliers = handle_outliers
        self.normalize_features = normalize_features
        self.random_state = random_state
        self.preprocessors_: Dict[str, Any] = {}
        self.steps_applied_: List[str] = []
        self.is_fitted_ = False

    def __sklearn_is_fitted__(self) -> bool:
        return self.is_fitted_

    # ----------------------------------------------------------------- utils
    def _check_fitted(self, method: str) -> None:
        if not self.is_fitted_:
            raise NotFittedError(f"{self.__class__.__name__} must be fitted before calling {method}.")

    def _validate_params(self) -> None:
        valid_missing = ("mean", "median", "most_frequent")
        if self.missing_strategy not in valid_missing:
            raise ValueError(
                f"missing_strategy must be one of {valid_missing}, got {self.missing_strategy!r}."
            )
        valid_scaling = ("standard", "minmax", "robust")
        if self.scaling_method not in valid_scaling:
            raise ValueError(f"scaling_method must be one of {valid_scaling}, got {self.scaling_method!r}.")
        valid_selection = ("univariate", "mutual_info", "variance")
        if self.feature_selection_method not in valid_selection:
            raise ValueError(
                f"feature_selection_method must be one of {valid_selection}, got {self.feature_selection_method!r}."
            )
        if self.outlier_contamination != "auto" and not 0 < float(self.outlier_contamination) <= 0.5:
            raise ValueError("outlier_contamination must be in (0, 0.5] or 'auto'.")

    def _resolved_flags(self) -> Dict[str, bool]:
        """Resolve alias parameters into the effective boolean switches."""
        return {
            "handle_missing": bool(self.handle_missing),
            "encode_categoricals": bool(self.encode_categoricals),
            "create_interactions": bool(self.create_interactions),
            "remove_outliers": bool(
                self.remove_outliers if self.handle_outliers is None else self.handle_outliers
            ),
            "scale_features": bool(
                self.scale_features if self.normalize_features is None else self.normalize_features
            ),
            "feature_selection": bool(self.feature_selection),
            "dimensionality_reduction": bool(self.dimensionality_reduction),
        }

    def _build_imputer(self, numeric: List[str], categorical: List[str]) -> ColumnTransformer:
        transformers = []
        if numeric:
            transformers.append(
                ("numeric", SimpleImputer(strategy=self.missing_strategy, keep_empty_features=True), numeric)
            )
        if categorical:
            transformers.append(
                (
                    "categorical",
                    SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                    categorical,
                )
            )
        imputer = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
        return imputer.set_output(transform="pandas")

    def _build_scaler(self) -> Any:
        scalers = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
        return scalers[self.scaling_method]().set_output(transform="pandas")

    def _build_selector(self, n_features: int, y: Optional[np.ndarray]) -> Pipeline:
        steps: List[Tuple[str, Any]] = [("variance", VarianceThreshold(threshold=self.variance_threshold))]
        k: int
        if y is not None and self.feature_selection_method != "variance":
            is_classification = self.target_type_ in ("binary", "multiclass")
            if self.feature_selection_method == "univariate":
                score_func = f_classif if is_classification else f_regression
            else:
                mi = mutual_info_classif if is_classification else mutual_info_regression

                def score_func(X_, y_, _mi=mi, _seed=self.random_state):  # type: ignore[misc]
                    return _mi(X_, y_, random_state=_seed)

            if self.feature_selection_k is None:
                # Default: never return more features than the raw input had, so that
                # one-hot / interaction expansion is undone by the ranking step.
                k = min(n_features, int(self.n_features_in_))
            else:
                k = min(int(self.feature_selection_k), n_features)
            steps.append(("kbest", SelectKBest(score_func=score_func, k=k)))
        return Pipeline(steps).set_output(transform="pandas")

    def _build_reducer(self, n_samples: int, n_features: int) -> PCA:
        upper = min(n_samples, n_features)
        if isinstance(self.n_components, float) and 0 < self.n_components < 1:
            n_components: Union[int, float] = self.n_components
        else:
            n_components = max(1, min(int(self.n_components), upper))
        return PCA(n_components=n_components, random_state=self.random_state).set_output(transform="pandas")

    # ------------------------------------------------------------------ API
    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> DataPreprocessor:
        """Fit every enabled preprocessing step sequentially.

        Args:
            X: Input features (array or DataFrame; mixed dtypes allowed).
            y: Optional target used for target encoding and feature selection.

        Returns:
            The fitted preprocessor.

        Raises:
            ValueError: On empty input or invalid configuration.
        """
        self._validate_params()
        flags = self._resolved_flags()
        frame = _to_dataframe(X)
        _check_non_empty(frame, self.__class__.__name__)
        y_arr = None if y is None else np.asarray(y).ravel()
        if y_arr is not None and y_arr.shape[0] != frame.shape[0]:
            raise ValueError(f"y has {y_arr.shape[0]} samples but X has {frame.shape[0]}.")
        self.target_type_ = None if y_arr is None else type_of_target(y_arr)

        self.preprocessors_ = {}
        self.steps_applied_ = []
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        self.numeric_columns_, self.categorical_columns_ = _split_columns(frame)
        self.dropped_columns_: List[str] = []
        self.outlier_mask_ = np.ones(frame.shape[0], dtype=bool)
        self.n_outliers_ = 0
        mask = self.outlier_mask_
        work = frame

        if flags["handle_missing"]:
            imputer = self._build_imputer(self.numeric_columns_, self.categorical_columns_)
            work = imputer.fit_transform(work)[list(frame.columns)]
            self._register("imputer", imputer)
        elif work.isna().any().any():
            self.logger.warning("Input contains missing values but handle_missing=False.")

        if self.categorical_columns_:
            if flags["encode_categoricals"]:
                encoder = CategoricalEncoder(columns=self.categorical_columns_)
                work = encoder.fit_transform(work, y_arr)
                self._register("encoder", encoder)
            else:
                self.dropped_columns_ = list(self.categorical_columns_)
                work = work.drop(columns=self.dropped_columns_)
                self.logger.warning(
                    "Dropping categorical columns %s (encode_categoricals=False).", self.dropped_columns_
                )
                if work.shape[1] == 0:
                    raise ValueError("No numeric columns remain after dropping categorical columns.")
        work = work.astype(float)

        if flags["create_interactions"]:
            engineer = NumericalTransformer(create_interactions=True)
            work = engineer.fit_transform(work)
            self._register("feature_engineer", engineer)

        if flags["remove_outliers"]:
            detector = IsolationForest(
                contamination=self.outlier_contamination, random_state=self.random_state
            )
            detector.fit(work)
            inliers = detector.predict(work) == 1
            if inliers.sum() >= 2:
                mask = inliers
            else:
                self.logger.warning(
                    "Outlier detector flagged nearly all rows; fitting downstream steps on all rows."
                )
            self.outlier_mask_ = mask
            self.n_outliers_ = int((~mask).sum())
            self._register("outlier_detector", detector)
            self.logger.info("Outlier detection flagged %d of %d rows.", self.n_outliers_, frame.shape[0])

        if flags["scale_features"]:
            scaler = self._build_scaler()
            scaler.fit(work[mask])
            work = scaler.transform(work)
            self._register("scaler", scaler)

        if flags["feature_selection"]:
            selector = self._build_selector(work.shape[1], y_arr)
            selector.fit(work[mask], None if y_arr is None else y_arr[mask])
            work = selector.transform(work)
            self._register("feature_selector", selector)

        if flags["dimensionality_reduction"]:
            reducer = self._build_reducer(int(mask.sum()), work.shape[1])
            reducer.fit(work[mask])
            work = reducer.transform(work)
            self._register("dim_reducer", reducer)

        self.feature_names_out_ = [str(c) for c in work.columns]
        self.is_fitted_ = True
        self.logger.info(
            "Fitted %s: %d -> %d features, steps=%s",
            self.__class__.__name__,
            self.n_features_in_,
            len(self.feature_names_out_),
            self.steps_applied_,
        )
        return self

    def _register(self, key: str, transformer: Any) -> None:
        self.preprocessors_[key] = transformer
        self.steps_applied_.append(self._STEP_NAMES[key])

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Apply the fitted steps to ``X``; rows are never removed.

        Args:
            X: Input features with the columns seen during fit.

        Returns:
            Dense ``float`` array of shape ``(n_samples, n_features_out)``.

        Raises:
            NotFittedError: If called before :meth:`fit` (a ``ValueError`` subclass).
        """
        self._check_fitted("transform")
        frame = _to_dataframe(X)
        _check_columns_present(frame, list(self.feature_names_in_), self.__class__.__name__)
        work = frame[list(self.feature_names_in_)]

        if "imputer" in self.preprocessors_:
            work = self.preprocessors_["imputer"].transform(work)[list(self.feature_names_in_)]
        if "encoder" in self.preprocessors_:
            work = self.preprocessors_["encoder"].transform(work)
        elif self.dropped_columns_:
            work = work.drop(columns=self.dropped_columns_)
        work = work.astype(float)
        for key in ("feature_engineer", "scaler", "feature_selector", "dim_reducer"):
            if key in self.preprocessors_:
                work = self.preprocessors_[key].transform(work)
        return np.asarray(work, dtype=float)

    def inverse_transform(self, X: ArrayLike) -> pd.DataFrame:
        """Map preprocessed data back to the (imputed, encoded) feature space.

        Only the scaling and PCA steps are invertible; the method raises when
        feature selection, feature engineering or categorical encoding changed
        the feature set.

        Args:
            X: Array as returned by :meth:`transform`.

        Returns:
            DataFrame in the scaler's input space.

        Raises:
            ValueError: If a non-invertible step is part of the fitted pipeline.
        """
        self._check_fitted("inverse_transform")
        blocking = [
            k for k in ("feature_selector", "feature_engineer", "encoder") if k in self.preprocessors_
        ]
        if blocking:
            raise ValueError(f"inverse_transform is not available with steps {blocking}.")
        arr = np.asarray(X, dtype=float)
        if "dim_reducer" in self.preprocessors_:
            arr = self.preprocessors_["dim_reducer"].inverse_transform(arr)
        if "scaler" in self.preprocessors_:
            arr = self.preprocessors_["scaler"].inverse_transform(arr)
        columns = [c for c in self.feature_names_in_ if c not in self.dropped_columns_]
        return pd.DataFrame(np.asarray(arr), columns=columns)

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        """Return output feature names (``input_features`` is accepted for API parity)."""
        self._check_fitted("get_feature_names_out")
        return np.asarray(self.feature_names_out_, dtype=object)

    def get_applied_steps(self) -> List[str]:
        """Return the names of the steps applied during fit, in order."""
        self._check_fitted("get_applied_steps")
        return list(self.steps_applied_)

    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Summarise the fitted pipeline.

        Returns:
            Dictionary with ``steps_applied``, ``parameters`` (constructor
            params), feature counts and names, outlier statistics and any
            dropped columns.
        """
        self._check_fitted("get_preprocessing_summary")
        return {
            "steps_applied": list(self.steps_applied_),
            "parameters": self.get_params(deep=False),
            "n_features_in": int(self.n_features_in_),
            "n_features_out": len(self.feature_names_out_),
            "feature_names_out": list(self.feature_names_out_),
            "numeric_columns": list(self.numeric_columns_),
            "categorical_columns": list(self.categorical_columns_),
            "dropped_columns": list(self.dropped_columns_),
            "n_outliers_detected": int(self.n_outliers_),
        }

    @staticmethod
    def analyze_data(X: ArrayLike, skewness_threshold: float = 1.0) -> Dict[str, Any]:
        """Profile a dataset before preprocessing.

        Args:
            X: Input features.
            skewness_threshold: Absolute skewness above which a column is reported as skewed.

        Returns:
            Dictionary with sample/feature counts, dtype split, missing-value
            statistics, constant and skewed columns, IQR outlier counts,
            high-cardinality categoricals and duplicate-row count.
        """
        frame = _to_dataframe(X)
        numeric, categorical = _split_columns(frame)
        missing_per_column = frame.isna().sum()
        outlier_counts: Dict[str, int] = {}
        skewed: List[str] = []
        constant: List[str] = []
        for column in numeric:
            series = frame[column].dropna().astype(float)
            if series.nunique() <= 1:
                constant.append(column)
                continue
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            n_out = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
            if n_out:
                outlier_counts[column] = n_out
            skew = series.skew()
            if not pd.isna(skew) and abs(float(skew)) > skewness_threshold:
                skewed.append(column)
        constant.extend(c for c in categorical if frame[c].nunique(dropna=True) <= 1)
        n_cells = max(1, frame.shape[0] * frame.shape[1])
        return {
            "n_samples": int(frame.shape[0]),
            "n_features": int(frame.shape[1]),
            "numeric_features": len(numeric),
            "categorical_features": len(categorical),
            "missing_values": int(missing_per_column.sum()),
            "missing_ratio": float(missing_per_column.sum() / n_cells),
            "columns_with_missing": {c: int(v) for c, v in missing_per_column.items() if v > 0},
            "constant_features": constant,
            "skewed_features": skewed,
            "outlier_counts": outlier_counts,
            "high_cardinality_features": [c for c in categorical if frame[c].nunique(dropna=True) > 50],
            "duplicate_rows": int(frame.duplicated().sum()),
        }


# --------------------------------------------------------------------------- #
# ImbalancedDataHandler
# --------------------------------------------------------------------------- #
class ImbalancedDataHandler(LoggerMixin):
    """Resample imbalanced classification data.

    Uses ``imbalanced-learn`` when installed. Without it, random over- and
    under-sampling to a balanced (``"auto"``) class distribution are provided
    by a NumPy fallback; SMOTE always requires ``imbalanced-learn``.

    Args:
        method: ``"smote"``, ``"oversampling"`` or ``"undersampling"``.
        sampling_strategy: Passed to the imblearn sampler (``"auto"`` balances classes).
        random_state: Seed for the resampling.
        k_neighbors: Neighbourhood size for SMOTE.
    """

    _ALIASES = {
        "smote": "smote",
        "oversampling": "oversampling",
        "oversample": "oversampling",
        "random_over": "oversampling",
        "undersampling": "undersampling",
        "undersample": "undersampling",
        "random_under": "undersampling",
    }

    def __init__(
        self,
        method: str = "smote",
        sampling_strategy: Union[str, float, Dict[Any, int]] = "auto",
        random_state: Optional[int] = None,
        k_neighbors: int = 5,
    ) -> None:
        self.method = method
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    @staticmethod
    def class_distribution(y: ArrayLike) -> Dict[Any, int]:
        """Return ``{class: count}`` for ``y``."""
        labels, counts = np.unique(np.asarray(y).ravel(), return_counts=True)
        return {label.item() if hasattr(label, "item") else label: int(c) for label, c in zip(labels, counts)}

    def handle_imbalance(
        self,
        X: ArrayLike,
        y: ArrayLike,
        method: Optional[str] = None,
        ratio: Optional[Union[str, float, Dict[Any, int]]] = None,
    ) -> Tuple[Any, Any]:
        """Resample ``(X, y)`` to reduce class imbalance.

        Args:
            X: Feature matrix (DataFrame type is preserved).
            y: Class labels.
            method: Override the configured method for this call.
            ratio: Override ``sampling_strategy`` for this call.

        Returns:
            Tuple ``(X_resampled, y_resampled)``.

        Raises:
            ValueError: For an unknown method.
            ImportError: If SMOTE (or a non-``"auto"`` strategy) is requested
                without ``imbalanced-learn`` installed.
        """
        resolved = self._ALIASES.get(str(method or self.method).lower())
        if resolved is None:
            raise ValueError(
                f"Unknown method {method or self.method!r}; expected one of {sorted(self._ALIASES)}."
            )
        strategy = self.sampling_strategy if ratio is None else ratio
        before = self.class_distribution(y)

        if HAS_IMBLEARN:
            if resolved == "smote":
                sampler = SMOTE(
                    sampling_strategy=strategy, random_state=self.random_state, k_neighbors=self.k_neighbors
                )
            elif resolved == "oversampling":
                sampler = RandomOverSampler(sampling_strategy=strategy, random_state=self.random_state)
            else:
                sampler = RandomUnderSampler(sampling_strategy=strategy, random_state=self.random_state)
            X_res, y_res = sampler.fit_resample(X, y)
        else:
            if resolved == "smote" or strategy != "auto":
                raise ImportError(
                    "SMOTE and custom sampling strategies require 'imbalanced-learn' (pip install imbalanced-learn)."
                )
            X_res, y_res = self._random_resample(X, y, oversample=resolved == "oversampling")

        self.logger.info("Resampled with %s: %s -> %s", resolved, before, self.class_distribution(y_res))
        return X_res, y_res

    fit_resample = handle_imbalance

    def _random_resample(self, X: ArrayLike, y: ArrayLike, oversample: bool) -> Tuple[Any, Any]:
        """NumPy fallback that balances every class to the majority (or minority) size."""
        rng = np.random.default_rng(self.random_state)
        y_arr = np.asarray(y).ravel()
        labels, counts = np.unique(y_arr, return_counts=True)
        target = int(counts.max() if oversample else counts.min())
        indices: List[np.ndarray] = []
        for label in labels:
            members = np.flatnonzero(y_arr == label)
            indices.append(rng.choice(members, size=target, replace=oversample and members.size < target))
        idx = np.concatenate(indices)
        X_res = X.iloc[idx].reset_index(drop=True) if isinstance(X, pd.DataFrame) else np.asarray(X)[idx]
        y_res = y.iloc[idx].reset_index(drop=True) if isinstance(y, pd.Series) else y_arr[idx]
        return X_res, y_res
