"""Advanced feature-union and pipeline-composition utilities.

This module provides sklearn-compatible building blocks for combining several
feature-processing branches into a single transformer:

* :class:`AdvancedFeatureUnion` - parallel branches stacked horizontally.
* :class:`WeightedFeatureUnion` - branches scaled by per-branch weights.
* :class:`ConditionalFeatureUnion` - branches activated by data predicates.
* :class:`DynamicFeatureUnion` - branches chosen by a strategy callable.
* :class:`ParallelFeatureProcessor` - column-group processing, feature
  engineering functions and conditional processing on DataFrames.
* :class:`ColumnSelector` / :class:`ColumnTransformer` - column selection and
  column-wise transformation helpers.
* :class:`FeatureStacker` - stacking of pre-computed feature sets.
* :class:`FeaturePipelineBuilder` / :class:`PipelineComposer` - declarative
  pipeline construction and composition.

Every transformer follows the scikit-learn API (``BaseEstimator`` +
``TransformerMixin``, ``fit`` returns ``self``, fitted attributes end in ``_``,
``get_feature_names_out`` is available after fitting) and is ``clone``-safe.

Branches and pipeline steps may be given as *placeholder strings* (for example
``"onehot"``, ``"standard_scaling"`` or ``"simple_imputer"``) which are
resolved through a registry, see :func:`resolve_transformer` and
:func:`register_transformer`.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer as _SkColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from sklearn_mastery.config.logging_config import LoggerMixin, get_logger

try:  # optional dependency used for bayesian pipeline optimisation
    import optuna

    HAS_OPTUNA = True
except ImportError:  # pragma: no cover - exercised only without optuna installed
    optuna = None
    HAS_OPTUNA = False

logger = get_logger(__name__)

ArrayLike = Union[np.ndarray, pd.DataFrame]
TransformerSpec = Union[str, BaseEstimator]
Condition = Callable[[Any], bool]


def _is_categorical_dtype(dtype: Any) -> bool:
    """True for object, string (pandas 2 and 3), categorical and boolean dtypes."""
    types = pd.api.types
    return bool(
        types.is_object_dtype(dtype)
        or types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
        or types.is_bool_dtype(dtype)
    )


def _is_categorical_series(series: pd.Series) -> bool:
    """Column predicate for :class:`ColumnSelector`: keep categorical-like columns."""
    return _is_categorical_dtype(series.dtype)


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #


def _to_frame(X: Any) -> pd.DataFrame:
    """Return ``X`` as a DataFrame (arrays get ``x0, x1, ...`` column names)."""
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, pd.Series):
        return X.to_frame()
    if sparse.issparse(X):
        X = X.toarray()
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return pd.DataFrame(arr, columns=[f"x{i}" for i in range(arr.shape[1])])


def _to_block(Xt: Any, sparse_output: bool = False) -> Any:
    """Coerce a transformer output into a 2-D block ready for stacking."""
    if sparse.issparse(Xt):
        return Xt.tocsr() if sparse_output else Xt.toarray()
    if isinstance(Xt, (pd.DataFrame, pd.Series)):
        Xt = Xt.to_numpy()
    arr = np.asarray(Xt)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return sparse.csr_matrix(arr) if sparse_output else arr


def _block_feature_names(name: str, estimator: Any, n_features: int) -> List[str]:
    """Feature names of a fitted branch, prefixed with the branch name."""
    names: Optional[Sequence[str]] = None
    if hasattr(estimator, "get_feature_names_out"):
        try:
            names = list(estimator.get_feature_names_out())
        except Exception:  # fall back to generic names
            names = None
    if names is None or len(names) != n_features:
        names = [f"x{i}" for i in range(n_features)]
    return [f"{name}__{feature}" for feature in names]


def _select_columns(frame: pd.DataFrame, columns: Sequence[Any]) -> pd.DataFrame:
    """Select ``columns`` from ``frame`` raising ``KeyError`` for missing ones."""
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise KeyError(f"Columns not found in input data: {missing}")
    return frame.loc[:, list(columns)]


def _stack_blocks(blocks: Sequence[Any], sparse_output: bool = False) -> Any:
    """Horizontally stack blocks (dense ``ndarray`` or sparse CSR)."""
    if sparse_output:
        return sparse.hstack([sparse.csr_matrix(b) for b in blocks]).tocsr()
    return np.hstack([np.asarray(b) for b in blocks])


# --------------------------------------------------------------------------- #
# Private dtype-aware transformers used by the placeholder registry
# --------------------------------------------------------------------------- #


class _CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode the categorical columns of a DataFrame and pass numeric ones through.

    If the input has no categorical (object/category/bool/string) columns all
    columns are encoded. Array inputs are always encoded entirely.

    Args:
        method: ``"onehot"``, ``"ordinal"`` or ``"target"``.
        sparse_output: Emit a sparse matrix for one-hot encoding.
    """

    def __init__(self, method: str = "onehot", sparse_output: bool = False):
        self.method = method
        self.sparse_output = sparse_output

    def _make_encoder(self) -> BaseEstimator:
        if self.method == "onehot":
            return OneHotEncoder(handle_unknown="ignore", sparse_output=self.sparse_output)
        if self.method == "ordinal":
            return OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2
            )
        if self.method == "target":
            return TargetEncoder()
        raise ValueError(f"Unknown encoding method {self.method!r}; use 'onehot', 'ordinal' or 'target'.")

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> _CategoricalEncoder:
        frame = _to_frame(X)
        categorical = [c for c in frame.columns if _is_categorical_dtype(frame[c].dtype)]
        if not categorical or not isinstance(X, pd.DataFrame):
            categorical = list(frame.columns)
        self.encoded_columns_ = categorical
        self.passthrough_columns_ = [c for c in frame.columns if c not in categorical]
        self.encoder_ = self._make_encoder()
        subset = (
            frame[self.encoded_columns_].astype(object).where(frame[self.encoded_columns_].notna(), np.nan)
        )
        if self.method == "target":
            if y is None:
                raise ValueError("Target encoding requires `y`.")
            self.encoder_.fit(subset, y)
        else:
            self.encoder_.fit(subset)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X: ArrayLike) -> Any:
        check_is_fitted(self, "encoder_")
        frame = _to_frame(X)
        subset = _select_columns(frame, self.encoded_columns_)
        subset = subset.astype(object).where(subset.notna(), np.nan)
        encoded = self.encoder_.transform(subset)
        if not self.passthrough_columns_:
            return encoded
        passthrough = frame[self.passthrough_columns_].to_numpy()
        if sparse.issparse(encoded):
            return sparse.hstack([sparse.csr_matrix(passthrough.astype(float)), encoded]).tocsr()
        return np.hstack([passthrough, np.asarray(encoded)])

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        check_is_fitted(self, "encoder_")
        encoded = list(self.encoder_.get_feature_names_out([str(c) for c in self.encoded_columns_]))
        return np.asarray([str(c) for c in self.passthrough_columns_] + encoded, dtype=object)


class _DTypeAwareImputer(BaseEstimator, TransformerMixin):
    """Impute numeric columns with ``numeric_strategy`` and others with the mode.

    Args:
        numeric_strategy: ``SimpleImputer`` strategy for numeric columns.
    """

    def __init__(self, numeric_strategy: str = "median"):
        self.numeric_strategy = numeric_strategy

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> _DTypeAwareImputer:
        frame = _to_frame(X)
        self.numeric_columns_ = list(frame.select_dtypes(include=[np.number]).columns)
        self.other_columns_ = [c for c in frame.columns if c not in self.numeric_columns_]
        self.columns_ = list(frame.columns)
        self.numeric_imputer_ = (
            SimpleImputer(strategy=self.numeric_strategy).fit(frame[self.numeric_columns_])
            if self.numeric_columns_
            else None
        )
        self.other_imputer_ = (
            SimpleImputer(strategy="most_frequent").fit(frame[self.other_columns_].astype(object))
            if self.other_columns_
            else None
        )
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X: ArrayLike) -> Any:
        check_is_fitted(self, "columns_")
        frame = _select_columns(_to_frame(X), self.columns_).copy()
        if self.numeric_imputer_ is not None:
            frame[self.numeric_columns_] = self.numeric_imputer_.transform(frame[self.numeric_columns_])
        if self.other_imputer_ is not None:
            frame[self.other_columns_] = self.other_imputer_.transform(
                frame[self.other_columns_].astype(object)
            )
        return frame if isinstance(X, pd.DataFrame) else frame.to_numpy()

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        check_is_fitted(self, "columns_")
        return np.asarray([str(c) for c in self.columns_], dtype=object)


class _OutlierClipper(BaseEstimator, TransformerMixin):
    """Winsorise numeric columns to ``[Q1 - factor*IQR, Q3 + factor*IQR]``.

    Rows are never dropped, which keeps the transformer usable inside
    pipelines. Non-numeric columns are passed through unchanged.

    Args:
        factor: IQR multiplier defining the clipping bounds.
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> _OutlierClipper:
        frame = _to_frame(X)
        self.columns_ = list(frame.columns)
        self.numeric_columns_ = list(frame.select_dtypes(include=[np.number]).columns)
        numeric = frame[self.numeric_columns_]
        q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
        iqr = q3 - q1
        self.lower_bounds_ = (q1 - self.factor * iqr).to_numpy()
        self.upper_bounds_ = (q3 + self.factor * iqr).to_numpy()
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X: ArrayLike) -> Any:
        check_is_fitted(self, "columns_")
        frame = _select_columns(_to_frame(X), self.columns_).copy()
        if self.numeric_columns_:
            clipped = np.clip(
                frame[self.numeric_columns_].to_numpy(dtype=float), self.lower_bounds_, self.upper_bounds_
            )
            frame[self.numeric_columns_] = clipped
        return frame if isinstance(X, pd.DataFrame) else frame.to_numpy()

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        check_is_fitted(self, "columns_")
        return np.asarray([str(c) for c in self.columns_], dtype=object)


class _DateTimeFeatures(BaseEstimator, TransformerMixin):
    """Expand datetime columns into calendar components."""

    _COMPONENTS: Tuple[str, ...] = ("year", "month", "day", "dayofweek", "dayofyear")

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> _DateTimeFeatures:
        frame = _to_frame(X)
        self.columns_ = list(frame.columns)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "columns_")
        frame = _select_columns(_to_frame(X), self.columns_)
        blocks = []
        for col in self.columns_:
            series = pd.to_datetime(frame[col], errors="coerce")
            blocks.append(
                np.column_stack([getattr(series.dt, comp).to_numpy(dtype=float) for comp in self._COMPONENTS])
            )
        return np.hstack(blocks)

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        check_is_fitted(self, "columns_")
        return np.asarray([f"{c}_{comp}" for c in self.columns_ for comp in self._COMPONENTS], dtype=object)


class _TextVectorizer(BaseEstimator, TransformerMixin):
    """TF-IDF vectorise one or more text columns (dense output).

    Args:
        max_features: Vocabulary cap per column.
    """

    def __init__(self, max_features: Optional[int] = 200):
        self.max_features = max_features

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> _TextVectorizer:
        frame = _to_frame(X)
        self.columns_ = list(frame.columns)
        self.vectorizers_ = {
            col: TfidfVectorizer(max_features=self.max_features).fit(frame[col].fillna("").astype(str))
            for col in self.columns_
        }
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "vectorizers_")
        frame = _select_columns(_to_frame(X), self.columns_)
        return np.hstack(
            [
                self.vectorizers_[col].transform(frame[col].fillna("").astype(str)).toarray()
                for col in self.columns_
            ]
        )

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        check_is_fitted(self, "vectorizers_")
        names = [
            f"{col}_{term}"
            for col in self.columns_
            for term in self.vectorizers_[col].get_feature_names_out()
        ]
        return np.asarray(names, dtype=object)


class _UnivariateSelector(BaseEstimator, TransformerMixin):
    """``SelectKBest`` choosing the score function from the target type at fit time.

    Args:
        k: Number of features to keep (``"all"`` keeps every feature).
    """

    def __init__(self, k: Union[int, str] = 10):
        self.k = k

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> _UnivariateSelector:
        if y is None:
            raise ValueError("Univariate feature selection requires `y`.")
        score_func = f_classif if type_of_target(y) in {"binary", "multiclass"} else f_regression
        self.selector_ = SelectKBest(score_func=score_func, k=self.k).fit(X, y)
        self.n_features_in_ = self.selector_.n_features_in_
        return self

    def transform(self, X: ArrayLike) -> Any:
        check_is_fitted(self, "selector_")
        return self.selector_.transform(X)

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        check_is_fitted(self, "selector_")
        return self.selector_.get_feature_names_out(input_features)


class _AutoPreprocessor(BaseEstimator, TransformerMixin):
    """Type-driven preprocessing: impute+scale numerics, impute+one-hot categoricals, expand datetimes.

    Columns of other dtypes are dropped. The concrete ``ColumnTransformer`` is
    built at fit time once column dtypes are known.
    """

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> _AutoPreprocessor:
        frame = _to_frame(X)
        self.transformer_ = build_type_aware_column_transformer(frame)
        self.transformer_.fit(frame, y)
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, X: ArrayLike) -> Any:
        check_is_fitted(self, "transformer_")
        return self.transformer_.transform(_to_frame(X))

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        check_is_fitted(self, "transformer_")
        return self.transformer_.get_feature_names_out()


def build_type_aware_column_transformer(frame: pd.DataFrame) -> _SkColumnTransformer:
    """Build a scikit-learn ``ColumnTransformer`` from the dtypes of ``frame``.

    Args:
        frame: DataFrame whose column dtypes drive the branch selection.

    Returns:
        Unfitted ``sklearn.compose.ColumnTransformer`` with ``numeric``,
        ``categorical`` and ``datetime`` branches (only those that apply).

    Raises:
        ValueError: If no column is numeric, categorical or datetime.
    """
    numeric = list(frame.select_dtypes(include=[np.number]).columns)
    datetime_cols = list(frame.select_dtypes(include=["datetime", "datetimetz"]).columns)
    categorical = [c for c in frame.columns if _is_categorical_dtype(frame[c].dtype)]
    branches: List[Tuple[str, Any, List[Any]]] = []
    if numeric:
        branches.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric,
            )
        )
    if categorical:
        branches.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    if datetime_cols:
        branches.append(("datetime", _DateTimeFeatures(), datetime_cols))
    if not branches:
        raise ValueError("No numeric, categorical or datetime columns found to preprocess.")
    return _SkColumnTransformer(branches, remainder="drop", sparse_threshold=0.0)


# --------------------------------------------------------------------------- #
# Placeholder registry
# --------------------------------------------------------------------------- #

_REGISTRY: Dict[str, Callable[[], Any]] = {}


def register_transformer(name: str, factory: Callable[[], Any], *aliases: str) -> None:
    """Register a placeholder ``name`` (and ``aliases``) resolving to ``factory()``.

    Args:
        name: Canonical placeholder name.
        factory: Zero-argument callable returning a fresh transformer.
        *aliases: Additional names resolving to the same factory.
    """
    for key in (name, *aliases):
        _REGISTRY[key.lower()] = factory


def registered_transformers() -> List[str]:
    """Return the sorted list of registered placeholder names."""
    return sorted(_REGISTRY)


def resolve_transformer(spec: TransformerSpec, **params: Any) -> Any:
    """Resolve a placeholder string (or pass through an estimator) into a transformer.

    Args:
        spec: Placeholder name, ``"passthrough"`` or an estimator instance.
        **params: Parameters applied through ``set_params`` on the result.

    Returns:
        A fresh transformer instance (``"passthrough"`` is returned verbatim).

    Raises:
        ValueError: If ``spec`` is an unknown placeholder string.
    """
    if isinstance(spec, str):
        key = spec.lower()
        if key == "passthrough":
            return "passthrough"
        if key not in _REGISTRY:
            raise ValueError(
                f"Unknown transformer placeholder {spec!r}. Registered: {registered_transformers()}"
            )
        estimator = _REGISTRY[key]()
    else:
        estimator = spec
    if params and hasattr(estimator, "set_params"):
        estimator.set_params(**params)
    return estimator


def _numeric_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("selector", ColumnSelector(dtype_include=["number"])),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _categorical_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("selector", ColumnSelector(selector_function=_is_categorical_series)),
            ("imputer", _DTypeAwareImputer()),
            ("encoder", _CategoricalEncoder(method="onehot")),
        ]
    )


def _register_defaults() -> None:
    register_transformer("standard_scaler", StandardScaler, "standard_scaling", "scaler", "standard")
    register_transformer("minmax_scaler", MinMaxScaler, "minmax_scaling", "minmax")
    register_transformer("robust_scaler", RobustScaler, "robust_scaling", "robust")
    register_transformer(
        "onehot_encoder", lambda: _CategoricalEncoder(method="onehot"), "onehot", "onehot_encoding", "one_hot"
    )
    register_transformer(
        "ordinal_encoder",
        lambda: _CategoricalEncoder(method="ordinal"),
        "ordinal",
        "label",
        "label_encoder",
        "label_encoding",
    )
    register_transformer("target_encoder", lambda: _CategoricalEncoder(method="target"), "target_encoding")
    register_transformer(
        "simple_imputer",
        _DTypeAwareImputer,
        "simple",
        "imputer",
        "missing_imputation",
        "missing_value_imputation",
        "imputation",
    )
    register_transformer("outlier_removal", _OutlierClipper, "outlier_clipping", "outlier_handling")
    register_transformer(
        "polynomial_features", lambda: PolynomialFeatures(degree=2, include_bias=False), "polynomial"
    )
    register_transformer(
        "interactions",
        lambda: PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
        "feature_interactions",
        "interaction_features",
    )
    register_transformer("variance_threshold", VarianceThreshold, "variance")
    register_transformer("univariate_selection", _UnivariateSelector, "univariate", "select_k_best", "k_best")
    register_transformer("datetime_features_extraction", _DateTimeFeatures, "datetime_features", "datetime")
    register_transformer("tfidf_vectorization", _TextVectorizer, "tfidf", "text_vectorization")
    register_transformer("standard_numeric_pipeline", _numeric_pipeline, "numeric_pipeline")
    register_transformer("onehot_pipeline", _categorical_pipeline, "categorical_pipeline")
    register_transformer("auto_preprocessing", _AutoPreprocessor, "auto")


_register_defaults()


def materialize(spec: TransformerSpec, *, unknown: str = "raise") -> Any:
    """Return a fresh transformer for ``spec`` with placeholder strings resolved recursively.

    Estimators are cloned (``deepcopy`` for objects that do not implement
    ``get_params``); pipelines are rebuilt with each step materialised.

    Args:
        spec: Placeholder string, estimator or pipeline.
        unknown: ``"raise"`` to fail on unknown placeholders, ``"keep"`` to leave them untouched.

    Returns:
        A new, unfitted transformer.

    Raises:
        ValueError: If ``spec`` contains an unknown placeholder and ``unknown="raise"``.
    """
    if isinstance(spec, str):
        try:
            return resolve_transformer(spec)
        except ValueError:
            if unknown == "keep":
                return spec
            raise
    if isinstance(spec, Pipeline):
        params = spec.get_params(deep=False)
        params["steps"] = [(name, materialize(step, unknown=unknown)) for name, step in spec.steps]
        return spec.__class__(**params)
    return clone(spec, safe=False)


def _materialize_branch(spec: TransformerSpec) -> Any:
    """Materialise a union branch; ``"passthrough"`` becomes an identity transformer."""
    estimator = materialize(spec)
    return FunctionTransformer(feature_names_out="one-to-one") if estimator == "passthrough" else estimator


def _fit_transform_branch(name: str, estimator: Any, X: Any, y: Any) -> Tuple[str, Any, Any]:
    if hasattr(estimator, "fit_transform"):
        Xt = estimator.fit_transform(X, y)
    else:
        Xt = estimator.fit(X, y).transform(X)
    return name, estimator, Xt


# --------------------------------------------------------------------------- #
# Column selection / transformation
# --------------------------------------------------------------------------- #


class ColumnSelector(BaseEstimator, TransformerMixin, LoggerMixin):
    """Select DataFrame columns by name, dtype, regex pattern or predicate.

    Exactly one of ``columns``, ``dtype_include``, ``pattern`` or
    ``selector_function`` should be given; when none is given every column is
    selected. With ``inverse=True`` the complement is returned. DataFrame
    inputs yield DataFrames; array inputs (columns addressed positionally
    or as ``x0, x1, ...``) yield arrays.

    Args:
        columns: Column labels (or positions for array input) to keep.
        dtype_include: Dtypes accepted by ``DataFrame.select_dtypes``.
        pattern: Regular expression matched (``re.fullmatch``) against column names.
        selector_function: Predicate ``f(series) -> bool`` evaluated per column at fit time.
        inverse: Return the columns *not* selected.

    Raises:
        KeyError: If a requested column is missing from the data.
    """

    def __init__(
        self,
        columns: Optional[Sequence[Any]] = None,
        dtype_include: Optional[Union[str, Sequence[Any]]] = None,
        pattern: Optional[str] = None,
        selector_function: Optional[Callable[[pd.Series], bool]] = None,
        inverse: bool = False,
    ):
        self.columns = columns
        self.dtype_include = dtype_include
        self.pattern = pattern
        self.selector_function = selector_function
        self.inverse = inverse

    def _frame(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        frame = _to_frame(X)
        if self.columns is not None and all(isinstance(c, (int, np.integer)) for c in self.columns):
            frame.columns = list(range(frame.shape[1]))
        return frame

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> ColumnSelector:
        frame = self._frame(X)
        if self.columns is not None:
            selected = [c for c in _select_columns(frame, list(self.columns)).columns]
        elif self.dtype_include is not None:
            include = (
                [self.dtype_include] if isinstance(self.dtype_include, str) else list(self.dtype_include)
            )
            selected = list(frame.select_dtypes(include=include).columns)
        elif self.pattern is not None:
            regex = re.compile(self.pattern)
            selected = [c for c in frame.columns if regex.fullmatch(str(c))]
        elif self.selector_function is not None:
            selected = [c for c in frame.columns if bool(self.selector_function(frame[c]))]
        else:
            selected = list(frame.columns)
        if self.inverse:
            selected = [c for c in frame.columns if c not in set(selected)]
        self.columns_ = selected
        self.n_features_in_ = frame.shape[1]
        self.feature_names_in_ = np.asarray([str(c) for c in frame.columns], dtype=object)
        self.logger.debug("ColumnSelector kept %d of %d columns", len(selected), frame.shape[1])
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        check_is_fitted(self, "columns_")
        frame = _select_columns(self._frame(X), self.columns_)
        return frame if isinstance(X, pd.DataFrame) else frame.to_numpy()

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        check_is_fitted(self, "columns_")
        return np.asarray([str(c) for c in self.columns_], dtype=object)


class ColumnTransformer(BaseEstimator, TransformerMixin, LoggerMixin):
    """Apply different transformers to different column groups.

    A thin wrapper around ``sklearn.compose.ColumnTransformer`` that accepts
    placeholder strings (``"onehot_encoder"``, ``"label_encoder"``,
    ``"passthrough"``, ...) and offers a ``set_transformations`` builder API.

    Args:
        transformations: Sequence of ``(columns, transformer)`` pairs.
        remainder: ``"drop"``, ``"passthrough"`` or an estimator for unlisted columns.
        sparse_output: Always return a CSR matrix.
        n_jobs: Number of threads used to fit/transform the groups.
        verbose_feature_names_out: Prefix output names with the group name.
    """

    def __init__(
        self,
        transformations: Optional[Sequence[Tuple[Sequence[Any], TransformerSpec]]] = None,
        remainder: Union[str, BaseEstimator] = "drop",
        sparse_output: bool = False,
        n_jobs: Optional[int] = None,
        verbose_feature_names_out: bool = True,
    ):
        self.transformations = transformations
        self.remainder = remainder
        self.sparse_output = sparse_output
        self.n_jobs = n_jobs
        self.verbose_feature_names_out = verbose_feature_names_out

    def set_transformations(
        self, transformations: Sequence[Tuple[Sequence[Any], TransformerSpec]]
    ) -> ColumnTransformer:
        """Replace the ``(columns, transformer)`` specification and return ``self``."""
        self.transformations = list(transformations)
        return self

    def _build(self) -> _SkColumnTransformer:
        if not self.transformations:
            raise ValueError("No transformations defined; call `set_transformations` first.")
        branches = []
        for i, (columns, spec) in enumerate(self.transformations):
            estimator = materialize(spec)
            if self.sparse_output and hasattr(estimator, "sparse_output"):
                estimator.set_params(sparse_output=True)
            branches.append((f"group_{i}", estimator, list(columns)))
        return _SkColumnTransformer(
            branches,
            remainder=self.remainder,
            sparse_threshold=1.0 if self.sparse_output else 0.0,
            n_jobs=self.n_jobs,
            verbose_feature_names_out=self.verbose_feature_names_out,
        )

    def _finalize(self, Xt: Any) -> Any:
        if self.sparse_output and not sparse.issparse(Xt):
            return sparse.csr_matrix(np.asarray(Xt, dtype=float))
        if not self.sparse_output and sparse.issparse(Xt):
            return Xt.toarray()
        return Xt

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> ColumnTransformer:
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X: ArrayLike, y: Optional[np.ndarray] = None, **fit_params: Any) -> Any:
        self.transformer_ = self._build()
        frame = _to_frame(X)
        with _thread_backend():
            Xt = self.transformer_.fit_transform(frame, y, **fit_params)
        self.n_features_in_ = frame.shape[1]
        return self._finalize(Xt)

    def transform(self, X: ArrayLike) -> Any:
        check_is_fitted(self, "transformer_")
        with _thread_backend():
            Xt = self.transformer_.transform(_to_frame(X))
        return self._finalize(Xt)

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        check_is_fitted(self, "transformer_")
        return self.transformer_.get_feature_names_out(input_features)


def _thread_backend():
    """Context manager forcing joblib onto the threading backend.

    Threads avoid process start-up cost and pickling restrictions (lambdas,
    locally defined classes) for the lightweight, GIL-releasing transformers
    used here.
    """
    return parallel_config(backend="threading")


# --------------------------------------------------------------------------- #
# Feature unions
# --------------------------------------------------------------------------- #


class _BaseUnion(BaseEstimator, TransformerMixin, LoggerMixin):
    """Shared machinery for the union transformers.

    Subclasses implement :meth:`_select_branches` returning the ``(name, spec)``
    pairs to fit, and may override :meth:`_branch_weight`.
    """

    def __init__(
        self,
        pipelines: Optional[Sequence[Tuple[str, TransformerSpec]]] = None,
        n_jobs: Optional[int] = None,
        sparse_output: bool = False,
    ):
        self.pipelines = pipelines
        self.n_jobs = n_jobs
        self.sparse_output = sparse_output

    # -- builder API ------------------------------------------------------- #
    def _append_pipeline(self, name: str, pipeline: TransformerSpec) -> None:
        existing = list(self.pipelines or [])
        if any(existing_name == name for existing_name, _ in existing):
            raise ValueError(f"A pipeline named {name!r} already exists.")
        self.pipelines = [*existing, (name, pipeline)]

    def add_pipeline(self, name: str, pipeline: TransformerSpec) -> _BaseUnion:
        """Append a branch and return ``self`` for chaining.

        Args:
            name: Unique branch name.
            pipeline: Transformer, pipeline or placeholder string.

        Raises:
            ValueError: If ``name`` is already used.
        """
        self._append_pipeline(name, pipeline)
        return self

    # -- hooks ------------------------------------------------------------- #
    def _select_branches(self, X: Any, y: Any) -> List[Tuple[str, TransformerSpec]]:
        return list(self.pipelines or [])

    def _branch_weight(self, name: str) -> float:
        return 1.0

    def _handle_no_branches(self, X: Any) -> Any:
        raise ValueError(f"{self.__class__.__name__} has no pipelines to fit; add at least one branch.")

    # -- sklearn API ------------------------------------------------------- #
    def _stack(self, named_blocks: Sequence[Tuple[str, Any]]) -> Any:
        blocks = []
        for name, block in named_blocks:
            block = _to_block(block, sparse_output=self.sparse_output)
            weight = self._branch_weight(name)
            if weight != 1.0:
                block = (
                    block.astype(float) if self.sparse_output else np.asarray(block, dtype=float)
                ) * weight
            blocks.append(block)
        return _stack_blocks(blocks, sparse_output=self.sparse_output)

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None, **fit_params: Any) -> _BaseUnion:
        self.fit_transform(X, y, **fit_params)
        return self

    def fit_transform(self, X: ArrayLike, y: Optional[np.ndarray] = None, **fit_params: Any) -> Any:
        branches = self._select_branches(X, y)
        if not branches:
            return self._handle_no_branches(X)
        with _thread_backend():
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_transform_branch)(name, _materialize_branch(spec), X, y)
                for name, spec in branches
            )
        self.pipelines_ = [(name, est) for name, est, _ in results]
        Xt = self._stack([(name, block) for name, _, block in results])
        self.n_features_out_per_pipeline_ = {name: _to_block(block).shape[1] for name, _, block in results}
        self.n_features_in_ = X.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.logger.debug(
            "%s fitted %d branches -> %d features", self.__class__.__name__, len(results), Xt.shape[1]
        )
        return Xt

    def transform(self, X: ArrayLike) -> Any:
        check_is_fitted(self, "pipelines_")
        with _thread_backend():
            blocks = Parallel(n_jobs=self.n_jobs)(delayed(est.transform)(X) for _, est in self.pipelines_)
        return self._stack(list(zip([name for name, _ in self.pipelines_], blocks)))

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        check_is_fitted(self, "pipelines_")
        names: List[str] = []
        for name, est in self.pipelines_:
            names.extend(_block_feature_names(name, est, self.n_features_out_per_pipeline_[name]))
        return np.asarray(names, dtype=object)


class AdvancedFeatureUnion(_BaseUnion):
    """Horizontally stack the outputs of several parallel branches.

    Branches receive the full input and may be pipelines, transformers or
    placeholder strings (see :func:`resolve_transformer`). Steps inside
    pipelines may also be placeholders. Output is a dense ``ndarray`` (or CSR
    matrix when ``sparse_output=True``).

    Args:
        pipelines: Sequence of ``(name, branch)`` pairs; usually built with :meth:`add_pipeline`.
        n_jobs: Number of threads used to fit/transform branches.
        sparse_output: Stack into a CSR matrix instead of a dense array.

    Attributes:
        pipelines_: Fitted ``(name, transformer)`` pairs.
        n_features_out_per_pipeline_: Output width of every branch.
    """


class WeightedFeatureUnion(_BaseUnion):
    """Feature union whose branch outputs are multiplied by per-branch weights.

    Args:
        pipelines: Sequence of ``(name, branch)`` pairs.
        weights: Mapping ``name -> weight``; missing branches use ``1.0``.
        n_jobs: Number of threads used to fit/transform branches.
        sparse_output: Stack into a CSR matrix instead of a dense array.

    Attributes:
        feature_weights_: Weight applied to each fitted branch.
    """

    def __init__(
        self,
        pipelines: Optional[Sequence[Tuple[str, TransformerSpec]]] = None,
        weights: Optional[Mapping[str, float]] = None,
        n_jobs: Optional[int] = None,
        sparse_output: bool = False,
    ):
        super().__init__(pipelines=pipelines, n_jobs=n_jobs, sparse_output=sparse_output)
        self.weights = weights

    def add_pipeline(self, name: str, pipeline: TransformerSpec, weight: float = 1.0) -> WeightedFeatureUnion:
        """Append a weighted branch and return ``self``.

        Args:
            name: Unique branch name.
            pipeline: Transformer, pipeline or placeholder string.
            weight: Multiplicative factor applied to the branch output.

        Raises:
            ValueError: If ``name`` already exists or ``weight`` is negative.
        """
        if weight < 0:
            raise ValueError("weight must be non-negative.")
        self._append_pipeline(name, pipeline)
        self.weights = {**(self.weights or {}), name: float(weight)}
        return self

    def _branch_weight(self, name: str) -> float:
        return float((self.weights or {}).get(name, 1.0))

    def fit_transform(self, X: ArrayLike, y: Optional[np.ndarray] = None, **fit_params: Any) -> Any:
        Xt = super().fit_transform(X, y, **fit_params)
        self.feature_weights_ = {name: self._branch_weight(name) for name, _ in self.pipelines_}
        return Xt


class ConditionalFeatureUnion(_BaseUnion):
    """Feature union that only fits branches whose condition holds on the training data.

    Conditions are evaluated once at fit time so that the same branches are
    applied at transform time.

    Args:
        pipelines: Sequence of ``(name, branch)`` pairs.
        conditions: Mapping ``name -> predicate(X) -> bool``; branches without a
            predicate are always active.
        on_no_match: ``"passthrough"`` (default, input returned unchanged with a
            warning) or ``"raise"`` when no branch condition holds at fit time.
        n_jobs: Number of threads used to fit/transform branches.
        sparse_output: Stack into a CSR matrix instead of a dense array.

    Attributes:
        active_pipelines_: Names of the branches selected at fit time.
    """

    def __init__(
        self,
        pipelines: Optional[Sequence[Tuple[str, TransformerSpec]]] = None,
        conditions: Optional[Mapping[str, Condition]] = None,
        on_no_match: str = "passthrough",
        n_jobs: Optional[int] = None,
        sparse_output: bool = False,
    ):
        super().__init__(pipelines=pipelines, n_jobs=n_jobs, sparse_output=sparse_output)
        self.conditions = conditions
        self.on_no_match = on_no_match

    def add_conditional_pipeline(
        self, name: str, pipeline: TransformerSpec, condition: Optional[Condition] = None
    ) -> ConditionalFeatureUnion:
        """Append a branch guarded by ``condition`` and return ``self``.

        Args:
            name: Unique branch name.
            pipeline: Transformer, pipeline or placeholder string.
            condition: Predicate ``f(X) -> bool``; ``None`` means always active.
        """
        self._append_pipeline(name, pipeline)
        if condition is not None:
            self.conditions = {**(self.conditions or {}), name: condition}
        return self

    def _select_branches(self, X: Any, y: Any) -> List[Tuple[str, TransformerSpec]]:
        conditions = self.conditions or {}
        active = [
            (name, spec)
            for name, spec in (self.pipelines or [])
            if name not in conditions or bool(conditions[name](X))
        ]
        self.active_pipelines_ = [name for name, _ in active]
        self.logger.debug("Active conditional branches: %s", self.active_pipelines_)
        return active

    def _handle_no_branches(self, X: Any) -> Any:
        if not self.pipelines:
            raise ValueError("ConditionalFeatureUnion has no pipelines; add at least one branch.")
        if self.on_no_match == "passthrough":
            self.logger.warning(
                "No branch condition was satisfied; ConditionalFeatureUnion acts as identity."
            )
            self.pipelines_ = []
            self.n_features_out_per_pipeline_ = {}
            self.n_features_in_ = X.shape[1]
            return X
        raise ValueError("No branch condition was satisfied by the training data.")

    def transform(self, X: ArrayLike) -> Any:
        check_is_fitted(self, "pipelines_")
        if not self.pipelines_:
            return X
        return super().transform(X)


class DynamicFeatureUnion(_BaseUnion):
    """Feature union whose branches are chosen by a strategy callable at fit time.

    The strategy receives ``(X, y)`` and returns ``[(name, spec), ...]`` where
    ``spec`` is a transformer, a pipeline or a placeholder string such as
    ``"standard_numeric_pipeline"``, ``"onehot_pipeline"`` or
    ``"univariate_selection"``.

    Args:
        strategy: Callable ``f(X, y) -> list[(name, spec)]``; set via :meth:`set_adaptive_strategy`.
        n_jobs: Number of threads used to fit/transform branches.
        sparse_output: Stack into a CSR matrix instead of a dense array.

    Attributes:
        selected_pipelines_: Fitted ``(name, transformer)`` pairs chosen by the strategy.
        selected_names_: Names of the selected branches.
    """

    def __init__(
        self,
        strategy: Optional[Callable[[Any, Any], Sequence[Tuple[str, TransformerSpec]]]] = None,
        n_jobs: Optional[int] = None,
        sparse_output: bool = False,
    ):
        super().__init__(pipelines=None, n_jobs=n_jobs, sparse_output=sparse_output)
        self.strategy = strategy

    def set_adaptive_strategy(
        self, strategy: Callable[[Any, Any], Sequence[Tuple[str, TransformerSpec]]]
    ) -> DynamicFeatureUnion:
        """Set the branch-selection strategy and return ``self``."""
        self.strategy = strategy
        return self

    def _select_branches(self, X: Any, y: Any) -> List[Tuple[str, TransformerSpec]]:
        if self.strategy is None:
            raise ValueError("No strategy set; call `set_adaptive_strategy` first.")
        selected = list(self.strategy(X, y))
        self.selected_names_ = [name for name, _ in selected]
        return selected

    def fit_transform(self, X: ArrayLike, y: Optional[np.ndarray] = None, **fit_params: Any) -> Any:
        Xt = super().fit_transform(X, y, **fit_params)
        self.selected_pipelines_ = list(self.pipelines_)
        return Xt


# --------------------------------------------------------------------------- #
# Parallel column-group processing
# --------------------------------------------------------------------------- #


class ParallelFeatureProcessor(BaseEstimator, TransformerMixin, LoggerMixin):
    """Apply per-column-group processors, feature-engineering functions and conditional processors.

    Stages run in this order on a DataFrame view of the input:

    1. **Conditional strategies** ``{name: {"condition": f(X)->bool, "processor": spec}}``;
       active processors are applied (in order) to all numeric columns.
    2. **Processing strategies** ``{name: {"columns": [...], "processor": spec}}``;
       each group is transformed independently, unlisted columns are passed through.
    3. **Engineering strategies** ``{name: {"columns": [...], "function": f(frame)->array}}``;
       results are appended as new columns.

    Groups are processed with joblib threads (``n_jobs``, or ``max_workers``
    after :meth:`enable_async_processing`). DataFrame input yields a DataFrame.

    Args:
        processing_strategies: Column-group processors.
        engineering_strategies: Feature engineering functions.
        conditional_strategies: Conditionally applied processors.
        n_jobs: Number of threads.
        async_processing: Use ``max_workers`` threads instead of ``n_jobs``.
        max_workers: Thread count for asynchronous processing.

    Attributes:
        processors_: Fitted processors for each processing strategy.
        conditional_processors_: Fitted processors of the active conditional strategies.
        active_conditions_: Names of conditional strategies whose condition held at fit.
        feature_names_out_: Output column names.
    """

    def __init__(
        self,
        processing_strategies: Optional[Mapping[str, Mapping[str, Any]]] = None,
        engineering_strategies: Optional[Mapping[str, Mapping[str, Any]]] = None,
        conditional_strategies: Optional[Mapping[str, Mapping[str, Any]]] = None,
        n_jobs: Optional[int] = None,
        async_processing: bool = False,
        max_workers: Optional[int] = None,
    ):
        self.processing_strategies = processing_strategies
        self.engineering_strategies = engineering_strategies
        self.conditional_strategies = conditional_strategies
        self.n_jobs = n_jobs
        self.async_processing = async_processing
        self.max_workers = max_workers

    # -- builder API ------------------------------------------------------- #
    def set_processing_strategies(
        self, strategies: Mapping[str, Mapping[str, Any]]
    ) -> ParallelFeatureProcessor:
        """Set the column-group processing strategies and return ``self``."""
        self.processing_strategies = dict(strategies)
        return self

    def set_engineering_strategies(
        self, strategies: Mapping[str, Mapping[str, Any]]
    ) -> ParallelFeatureProcessor:
        """Set the feature engineering strategies and return ``self``."""
        self.engineering_strategies = dict(strategies)
        return self

    def set_conditional_strategies(
        self, strategies: Mapping[str, Mapping[str, Any]]
    ) -> ParallelFeatureProcessor:
        """Set the conditional processing strategies and return ``self``."""
        self.conditional_strategies = dict(strategies)
        return self

    def enable_async_processing(self, max_workers: Optional[int] = None) -> ParallelFeatureProcessor:
        """Run groups on a thread pool of ``max_workers`` threads and return ``self``."""
        self.async_processing = True
        self.max_workers = max_workers
        return self

    # -- internals --------------------------------------------------------- #
    @property
    def _n_workers(self) -> Optional[int]:
        return self.max_workers if self.async_processing else self.n_jobs

    @staticmethod
    def _output_frame(
        name: str, columns: Sequence[Any], estimator: Any, Xt: Any, index: pd.Index
    ) -> pd.DataFrame:
        block = _to_block(Xt)
        if block.shape[1] == len(columns):
            names: List[Any] = list(columns)
        else:
            names = _block_feature_names(name, estimator, block.shape[1])
        return pd.DataFrame(block, columns=names, index=index)

    @staticmethod
    def _engineered_frame(name: str, result: Any, index: pd.Index) -> pd.DataFrame:
        if isinstance(result, pd.DataFrame):
            return pd.DataFrame(
                result.to_numpy(), columns=[f"{name}_{c}" for c in result.columns], index=index
            )
        if isinstance(result, pd.Series):
            return pd.DataFrame({name: result.to_numpy()}, index=index)
        block = _to_block(result)
        names = [name] if block.shape[1] == 1 else [f"{name}_{i}" for i in range(block.shape[1])]
        return pd.DataFrame(block, columns=names, index=index)

    def _run_conditional(self, frame: pd.DataFrame, y: Any, fitting: bool) -> pd.DataFrame:
        strategies = self.conditional_strategies or {}
        if fitting:
            self.active_conditions_ = [
                name for name, spec in strategies.items() if bool(spec["condition"](frame))
            ]
            self.conditional_processors_ = {}
        for name in self.active_conditions_:
            numeric = list(frame.select_dtypes(include=[np.number]).columns)
            if not numeric:
                continue
            if fitting:
                estimator = materialize(strategies[name]["processor"])
                Xt = estimator.fit_transform(frame[numeric], y)
                self.conditional_processors_[name] = (numeric, estimator)
            else:
                numeric, estimator = self.conditional_processors_[name]
                Xt = estimator.transform(_select_columns(frame, numeric))
            out = self._output_frame(name, numeric, estimator, Xt, frame.index)
            if list(out.columns) == numeric:
                frame = frame.copy()
                frame[numeric] = out.to_numpy()
            else:
                frame = pd.concat([frame.drop(columns=numeric), out], axis=1)
        return frame

    def _run_processing(self, frame: pd.DataFrame, y: Any, fitting: bool) -> pd.DataFrame:
        strategies = self.processing_strategies or {}
        if not strategies:
            return frame
        if fitting:
            jobs = [
                (name, list(spec["columns"]), materialize(spec["processor"]))
                for name, spec in strategies.items()
            ]
            with _thread_backend():
                results = Parallel(n_jobs=self._n_workers)(
                    delayed(_fit_transform_branch)(name, est, _select_columns(frame, cols), y)
                    for name, cols, est in jobs
                )
            self.processors_ = {name: (cols, est) for (name, cols, _), (_, est, _) in zip(jobs, results)}
            blocks = [(name, cols, est, Xt) for (name, cols, _), (_, est, Xt) in zip(jobs, results)]
        else:
            items = list(self.processors_.items())
            with _thread_backend():
                outputs = Parallel(n_jobs=self._n_workers)(
                    delayed(est.transform)(_select_columns(frame, cols)) for _, (cols, est) in items
                )
            blocks = [(name, cols, est, Xt) for (name, (cols, est)), Xt in zip(items, outputs)]
        used = {c for _, cols, _, _ in blocks for c in cols}
        parts = [self._output_frame(name, cols, est, Xt, frame.index) for name, cols, est, Xt in blocks]
        remainder = [c for c in frame.columns if c not in used]
        if remainder:
            parts.append(frame[remainder])
        return pd.concat(parts, axis=1)

    def _run_engineering(self, frame: pd.DataFrame) -> pd.DataFrame:
        strategies = self.engineering_strategies or {}
        if not strategies:
            return frame
        jobs = [
            (name, list(spec.get("columns", frame.columns)), spec["function"])
            for name, spec in strategies.items()
        ]
        with _thread_backend():
            results = Parallel(n_jobs=self._n_workers)(
                delayed(func)(_select_columns(frame, cols)) for _, cols, func in jobs
            )
        parts = [frame] + [
            self._engineered_frame(name, res, frame.index) for (name, _, _), res in zip(jobs, results)
        ]
        return pd.concat(parts, axis=1)

    def _run(self, X: ArrayLike, y: Any, fitting: bool) -> pd.DataFrame:
        frame = _to_frame(X)
        frame = self._run_conditional(frame, y, fitting)
        frame = self._run_processing(frame, y, fitting)
        return self._run_engineering(frame)

    # -- sklearn API ------------------------------------------------------- #
    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> ParallelFeatureProcessor:
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X: ArrayLike, y: Optional[np.ndarray] = None, **fit_params: Any) -> ArrayLike:
        out = self._run(X, y, fitting=True)
        self.n_features_in_ = _to_frame(X).shape[1]
        self.feature_names_out_ = np.asarray([str(c) for c in out.columns], dtype=object)
        self.logger.debug("ParallelFeatureProcessor produced %d features", out.shape[1])
        return out if isinstance(X, pd.DataFrame) else out.to_numpy()

    def transform(self, X: ArrayLike) -> ArrayLike:
        check_is_fitted(self, "feature_names_out_")
        out = self._run(X, None, fitting=False)
        return out if isinstance(X, pd.DataFrame) else out.to_numpy()

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_


# --------------------------------------------------------------------------- #
# Feature stacking of pre-computed feature sets
# --------------------------------------------------------------------------- #


class FeatureStacker(BaseEstimator, TransformerMixin, LoggerMixin):
    """Combine several pre-computed feature sets (``{name: DataFrame}``) into one frame.

    Besides the explicit ``*_stack`` methods the class works as a transformer:
    ``transform(feature_sets)`` dispatches on ``mode``.

    Args:
        mode: ``"horizontal"``, ``"weighted"``, ``"selective"`` or ``"hierarchical"``.
        weights: Per-set weights for ``"weighted"`` mode.
        feature_importance: ``{set: {feature: score}}`` for ``"selective"`` mode.
        threshold: Minimum importance kept in ``"selective"`` mode.
        hierarchy: Ordered set names for ``"hierarchical"`` mode.
        cumulative: In hierarchical mode, level ``i`` holds sets ``hierarchy[:i+1]``.

    Attributes:
        feature_weights_: Weights used by the last weighted stack.
        selected_features_: ``{set: [features]}`` kept by the last selective stack.
        feature_names_out_: Column names of the last stacked frame.
    """

    def __init__(
        self,
        mode: str = "horizontal",
        weights: Optional[Mapping[str, float]] = None,
        feature_importance: Optional[Mapping[str, Mapping[str, float]]] = None,
        threshold: float = 0.5,
        hierarchy: Optional[Sequence[str]] = None,
        cumulative: bool = True,
    ):
        self.mode = mode
        self.weights = weights
        self.feature_importance = feature_importance
        self.threshold = threshold
        self.hierarchy = hierarchy
        self.cumulative = cumulative

    @staticmethod
    def _normalise(feature_sets: Union[Mapping[str, Any], Sequence[Any]]) -> Dict[str, pd.DataFrame]:
        if not isinstance(feature_sets, Mapping):
            feature_sets = {f"set_{i}": fs for i, fs in enumerate(feature_sets)}
        if not feature_sets:
            raise ValueError("At least one feature set is required.")
        frames = {name: _to_frame(fs).reset_index(drop=True) for name, fs in feature_sets.items()}
        lengths = {name: len(f) for name, f in frames.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"All feature sets must have the same number of rows, got {lengths}.")
        return frames

    def _finish(self, stacked: pd.DataFrame) -> pd.DataFrame:
        self.feature_names_out_ = np.asarray([str(c) for c in stacked.columns], dtype=object)
        return stacked

    def horizontal_stack(self, feature_sets: Union[Mapping[str, Any], Sequence[Any]]) -> pd.DataFrame:
        """Concatenate all sets column-wise; columns are renamed ``{set}_{feature}``.

        Raises:
            ValueError: If the sets differ in length.
        """
        frames = self._normalise(feature_sets)
        parts = [f.rename(columns=lambda c, n=name: f"{n}_{c}") for name, f in frames.items()]
        self.feature_set_names_ = list(frames)
        return self._finish(pd.concat(parts, axis=1))

    def weighted_stack(
        self, feature_sets: Union[Mapping[str, Any], Sequence[Any]], weights: Mapping[str, float]
    ) -> pd.DataFrame:
        """Horizontal stack with each set multiplied by ``weights[set]`` (default ``1.0``).

        Raises:
            ValueError: If the sets differ in length or a weight is negative.
        """
        frames = self._normalise(feature_sets)
        self.feature_weights_ = {name: float(weights.get(name, 1.0)) for name in frames}
        if any(w < 0 for w in self.feature_weights_.values()):
            raise ValueError("Weights must be non-negative.")
        weighted = {name: f.astype(float) * self.feature_weights_[name] for name, f in frames.items()}
        return self.horizontal_stack(weighted)

    def selective_stack(
        self,
        feature_sets: Union[Mapping[str, Any], Sequence[Any]],
        feature_importance: Mapping[str, Mapping[str, float]],
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Keep only features whose importance is ``>= threshold``.

        Features absent from ``feature_importance`` are dropped.

        Raises:
            ValueError: If no feature survives the threshold.
        """
        frames = self._normalise(feature_sets)
        selected = {
            name: [c for c in f.columns if feature_importance.get(name, {}).get(c, -np.inf) >= threshold]
            for name, f in frames.items()
        }
        self.selected_features_ = {name: cols for name, cols in selected.items() if cols}
        if not self.selected_features_:
            raise ValueError(f"No feature has importance >= {threshold}.")
        return self.horizontal_stack(
            {name: frames[name][cols] for name, cols in self.selected_features_.items()}
        )

    def hierarchical_stack(
        self,
        feature_sets: Union[Mapping[str, Any], Sequence[Any]],
        hierarchy: Sequence[str],
        cumulative: bool = True,
    ) -> pd.DataFrame:
        """Stack sets level by level with ``MultiIndex`` columns ``(level_i, set_feature)``.

        With ``cumulative=True`` level ``i`` contains the sets ``hierarchy[:i+1]``;
        otherwise it contains ``hierarchy[i]`` only.

        Raises:
            KeyError: If a hierarchy entry is not a known feature set.
        """
        frames = self._normalise(feature_sets)
        unknown = [name for name in hierarchy if name not in frames]
        if unknown:
            raise KeyError(f"Unknown feature sets in hierarchy: {unknown}")
        parts = []
        for level, name in enumerate(hierarchy):
            members = list(hierarchy[: level + 1]) if cumulative else [name]
            block = self.horizontal_stack({m: frames[m] for m in members})
            block.columns = pd.MultiIndex.from_product(
                [[f"level_{level}"], block.columns], names=["level", "feature"]
            )
            parts.append(block)
        self.hierarchy_levels_ = [f"level_{i}" for i in range(len(hierarchy))]
        return self._finish(pd.concat(parts, axis=1))

    def fit(
        self, X: Union[Mapping[str, Any], Sequence[Any]], y: Optional[np.ndarray] = None
    ) -> FeatureStacker:
        self.feature_set_names_ = list(self._normalise(X))
        return self

    def transform(self, X: Union[Mapping[str, Any], Sequence[Any]]) -> pd.DataFrame:
        check_is_fitted(self, "feature_set_names_")
        if self.mode == "horizontal":
            return self.horizontal_stack(X)
        if self.mode == "weighted":
            return self.weighted_stack(X, self.weights or {})
        if self.mode == "selective":
            return self.selective_stack(X, self.feature_importance or {}, self.threshold)
        if self.mode == "hierarchical":
            return self.hierarchical_stack(X, self.hierarchy or self.feature_set_names_, self.cumulative)
        raise ValueError(f"Unknown mode {self.mode!r}.")

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_


# --------------------------------------------------------------------------- #
# Pipeline building and composition
# --------------------------------------------------------------------------- #


def _has_predictor(estimator: Any) -> bool:
    final = estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator
    return hasattr(final, "predict")


class FeaturePipelineBuilder(BaseEstimator, LoggerMixin):
    """Declaratively build feature-processing pipelines.

    Column modules are implemented with :class:`ParallelFeatureProcessor`
    (group columns are transformed, the rest pass through), so modules chain
    naturally on DataFrames. Step names are placeholders resolved through
    :func:`resolve_transformer`.

    Args:
        random_state: Seed for pipeline optimisation.
        cv: Cross-validation folds used by :meth:`optimize_pipeline`.
        scoring: Scoring name for :meth:`optimize_pipeline` (estimator default when ``None``).
        n_jobs: Threads used by generated column modules and searches.

    Attributes:
        optimization_results_: Summary of the last :meth:`optimize_pipeline` call.
    """

    def __init__(
        self,
        random_state: Optional[int] = None,
        cv: int = 3,
        scoring: Optional[str] = None,
        n_jobs: Optional[int] = None,
    ):
        self.random_state = random_state
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self._modules: List[Tuple[str, Any]] = []

    # -- modular API ------------------------------------------------------- #
    def _column_module(
        self, name: str, columns: Sequence[Any], steps: Optional[Sequence[TransformerSpec]]
    ) -> ParallelFeatureProcessor:
        if steps:
            processor: Any = Pipeline([(f"step_{i}", materialize(step)) for i, step in enumerate(steps)])
        else:
            processor = _AutoPreprocessor()
        return ParallelFeatureProcessor(
            processing_strategies={name: {"columns": list(columns), "processor": processor}},
            n_jobs=self.n_jobs,
        )

    def add_preprocessing_module(
        self, name: str, columns: Sequence[Any], steps: Optional[Sequence[TransformerSpec]] = None
    ) -> FeaturePipelineBuilder:
        """Add a column-group module (type-aware preprocessing when ``steps`` is ``None``)."""
        self._modules.append((name, self._column_module(name, columns, steps)))
        return self

    def add_feature_engineering_module(
        self, method: TransformerSpec, **params: Any
    ) -> FeaturePipelineBuilder:
        """Add a global feature-engineering step (for example ``"interactions"``)."""
        name = method if isinstance(method, str) else type(method).__name__.lower()
        self._modules.append((f"engineering_{name}", resolve_transformer(method, **params)))
        return self

    def add_selection_module(self, method: TransformerSpec, **params: Any) -> FeaturePipelineBuilder:
        """Add a feature-selection step (for example ``"variance_threshold"`` or ``"univariate"``)."""
        name = method if isinstance(method, str) else type(method).__name__.lower()
        self._modules.append((f"selection_{name}", resolve_transformer(method, **params)))
        return self

    def add_step(self, name: str, transformer: TransformerSpec, **params: Any) -> FeaturePipelineBuilder:
        """Add an arbitrary named step."""
        self._modules.append((name, resolve_transformer(transformer, **params)))
        return self

    def reset(self) -> FeaturePipelineBuilder:
        """Discard all modules added so far."""
        self._modules = []
        return self

    def build(self) -> Pipeline:
        """Assemble the added modules into a ``Pipeline``.

        Raises:
            ValueError: If no module was added.
        """
        if not self._modules:
            raise ValueError("No modules added; use the `add_*_module` methods first.")
        return Pipeline([(name, clone(est, safe=False)) for name, est in self._modules])

    # -- one-shot builders ------------------------------------------------- #
    def build_auto_pipeline(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> Pipeline:
        """Build a preprocessing pipeline from the dtypes of ``X``.

        Numeric columns are median-imputed and standardised, categoricals are
        mode-imputed and one-hot encoded, datetimes are expanded.
        """
        frame = _to_frame(X)
        self.logger.info("Auto pipeline for %d columns", frame.shape[1])
        return Pipeline([("preprocessing", build_type_aware_column_transformer(frame))])

    def build_custom_pipeline(self, spec: Mapping[str, Mapping[str, Any]]) -> Pipeline:
        """Build a pipeline from a declarative specification.

        Each entry of ``spec`` is one step. Entries with ``"columns"`` become
        column modules applying ``"steps"`` to those columns; entries with only
        ``"steps"`` apply them globally; entries with ``"method"`` resolve that
        placeholder with the remaining keys as parameters (for example
        ``{"method": "univariate", "k": 5}``).

        Raises:
            ValueError: If ``spec`` is empty or an entry has no recognised keys.
        """
        if not spec:
            raise ValueError("Pipeline specification is empty.")
        steps: List[Tuple[str, Any]] = []
        for name, entry in spec.items():
            if "columns" in entry:
                steps.append((name, self._column_module(name, entry["columns"], entry.get("steps"))))
            elif "steps" in entry:
                inner = [(f"step_{i}", materialize(s)) for i, s in enumerate(entry["steps"])]
                steps.append((name, inner[0][1] if len(inner) == 1 else Pipeline(inner)))
            elif "method" in entry:
                params = {k: v for k, v in entry.items() if k != "method"}
                steps.append((name, resolve_transformer(entry["method"], **params)))
            else:
                raise ValueError(f"Specification entry {name!r} needs 'columns', 'steps' or 'method'.")
        return Pipeline(steps)

    def build_conditional_pipeline(
        self,
        X: ArrayLike,
        conditions: Mapping[str, Condition],
        conditional_steps: Mapping[str, Sequence[TransformerSpec]],
        base_steps: Optional[Sequence[TransformerSpec]] = ("auto_preprocessing",),
    ) -> Pipeline:
        """Build a pipeline containing the steps whose condition holds on ``X``.

        Args:
            X: Data on which the conditions are evaluated.
            conditions: ``{name: predicate(X) -> bool}``.
            conditional_steps: ``{name: [placeholders]}`` added when ``conditions[name]`` holds.
            base_steps: Steps always appended at the end (``None`` for none).

        Returns:
            Pipeline; a single ``"passthrough"`` step if nothing applies.
        """
        frame = _to_frame(X)
        steps: List[Tuple[str, Any]] = []
        for name, condition in conditions.items():
            if name in conditional_steps and bool(condition(frame)):
                for step in conditional_steps[name]:
                    steps.append((f"{name}_{step}" if isinstance(step, str) else name, materialize(step)))
        for step in base_steps or ():
            steps.append((step if isinstance(step, str) else type(step).__name__.lower(), materialize(step)))
        self.logger.info("Conditional pipeline steps: %s", [s for s, _ in steps])
        return Pipeline(steps or [("passthrough", "passthrough")])

    # -- optimisation ------------------------------------------------------ #
    @staticmethod
    def default_search_space(pipeline: Pipeline) -> Dict[str, List[Any]]:
        """Derive a small categorical search space from a pipeline's parameters."""
        space: Dict[str, List[Any]] = {}
        for key, value in pipeline.get_params(deep=True).items():
            leaf = key.rsplit("__", 1)[-1]
            if leaf == "strategy" and value in {"mean", "median"}:
                space[key] = ["mean", "median"]
            elif leaf == "k" and isinstance(value, (int, np.integer)):
                space[key] = sorted({max(1, value // 2), int(value), int(value) * 2})
            elif leaf == "degree" and isinstance(value, (int, np.integer)):
                space[key] = [1, 2]
            elif leaf == "threshold" and isinstance(value, float):
                space[key] = [0.0, 0.01, 0.05]
            elif leaf == "C" and isinstance(value, float):
                space[key] = [0.1, 1.0, 10.0]
            elif leaf == "n_estimators" and isinstance(value, (int, np.integer)):
                space[key] = [50, 100, 200]
        return space

    def optimize_pipeline(
        self,
        pipeline: Pipeline,
        X: ArrayLike,
        y: np.ndarray,
        optimization_method: str = "random",
        n_iterations: int = 10,
        param_space: Optional[Mapping[str, Sequence[Any]]] = None,
    ) -> Pipeline:
        """Search pipeline hyper-parameters and return the best (unfitted) pipeline.

        Transformer-only pipelines are scored through a probe estimator
        (logistic regression or ridge, chosen from the target type).

        Args:
            pipeline: Pipeline to optimise (not modified).
            X: Training features.
            y: Training target.
            optimization_method: ``"grid"``, ``"random"`` or ``"bayesian"`` (optuna, falls back to random).
            n_iterations: Candidates evaluated for random/bayesian search.
            param_space: ``{param: [choices]}``; derived from the pipeline when ``None``.

        Returns:
            ``clone(pipeline)`` with the best parameters set.

        Raises:
            ValueError: If ``optimization_method`` is unknown.
        """
        if optimization_method not in {"grid", "random", "bayesian"}:
            raise ValueError("optimization_method must be 'grid', 'random' or 'bayesian'.")
        space = dict(param_space) if param_space is not None else self.default_search_space(pipeline)
        if not space:
            self.logger.warning("No tunable parameters found; returning the pipeline unchanged.")
            self.optimization_results_ = {
                "method": optimization_method,
                "best_params": {},
                "best_score": None,
            }
            return clone(pipeline)

        classification = type_of_target(y) in {"binary", "multiclass"}
        if _has_predictor(pipeline):
            search_est: Any = clone(pipeline)
            prefix = ""
        else:
            probe = LogisticRegression(max_iter=500) if classification else Ridge()
            search_est = Pipeline([("features", clone(pipeline)), ("probe", probe)])
            prefix = "features__"
        search_space = {f"{prefix}{k}": list(v) for k, v in space.items()}

        method = optimization_method
        if method == "bayesian" and not HAS_OPTUNA:
            self.logger.warning("optuna is not installed; falling back to random search.")
            method = "random"

        if method == "bayesian":
            best_params, best_score = self._optuna_search(search_est, search_space, X, y, n_iterations)
        else:
            if method == "grid":
                search = GridSearchCV(
                    search_est, search_space, cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs
                )
            else:
                search = RandomizedSearchCV(
                    search_est,
                    search_space,
                    n_iter=n_iterations,
                    cv=self.cv,
                    scoring=self.scoring,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                )
            with _thread_backend():
                search.fit(X, y)
            best_params, best_score = search.best_params_, float(search.best_score_)

        stripped = {k[len(prefix) :]: v for k, v in best_params.items()}
        self.optimization_results_ = {"method": method, "best_params": stripped, "best_score": best_score}
        self.logger.info("Best %s-search score %.4f with %s", method, best_score, stripped)
        return clone(pipeline).set_params(**stripped)

    def _optuna_search(
        self, estimator: Any, space: Mapping[str, Sequence[Any]], X: ArrayLike, y: np.ndarray, n_trials: int
    ) -> Tuple[Dict[str, Any], float]:
        def objective(trial: optuna.Trial) -> float:
            params = {k: trial.suggest_categorical(k, list(v)) for k, v in space.items()}
            candidate = clone(estimator).set_params(**params)
            with _thread_backend():
                return float(np.mean(cross_val_score(candidate, X, y, cv=self.cv, scoring=self.scoring)))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=n_trials)
        return dict(study.best_params), float(study.best_value)


class PipelineComposer(BaseEstimator, LoggerMixin):
    """Compose steps, sub-pipelines and unions into ``Pipeline`` objects.

    Args:
        flatten: Inline the steps of nested pipelines instead of nesting them.
        memory: ``Pipeline`` caching argument.
        verbose: ``Pipeline`` verbosity flag.
    """

    def __init__(self, flatten: bool = False, memory: Any = None, verbose: bool = False):
        self.flatten = flatten
        self.memory = memory
        self.verbose = verbose

    @staticmethod
    def validate_steps(steps: Sequence[Tuple[str, Any]]) -> None:
        """Check that steps form a valid pipeline.

        Raises:
            ValueError: On duplicate names, unknown placeholders or an
                intermediate step without ``fit``/``transform``.
            TypeError: If the final step cannot ``fit``.
        """
        names = [name for name, _ in steps]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate step names: {names}")
        for i, (name, step) in enumerate(steps):
            if isinstance(step, str):
                if step != "passthrough":
                    raise ValueError(f"Step {name!r} uses unknown placeholder {step!r}.")
                continue
            is_last = i == len(steps) - 1
            if not hasattr(step, "fit"):
                raise TypeError(f"Step {name!r} must implement `fit`.")
            if not is_last and not (hasattr(step, "transform") or hasattr(step, "fit_transform")):
                raise ValueError(f"Intermediate step {name!r} must implement `transform`.")

    def compose(self, steps: Sequence[Tuple[str, Any]], validate: bool = False) -> Pipeline:
        """Compose ``(name, step)`` pairs into a ``Pipeline``.

        Known placeholder strings are resolved; estimator objects are used as
        given (not cloned).

        Args:
            steps: Sequence of ``(name, estimator | Pipeline | placeholder)``.
            validate: Run :meth:`validate_steps` before building.

        Raises:
            ValueError: If ``steps`` is empty or validation fails.
        """
        if not steps:
            raise ValueError("At least one step is required.")
        resolved: List[Tuple[str, Any]] = []
        for name, step in steps:
            step = materialize(step, unknown="keep") if isinstance(step, str) else step
            if self.flatten and isinstance(step, Pipeline):
                resolved.extend((f"{name}_{inner}", est) for inner, est in step.steps)
            else:
                resolved.append((name, step))
        if validate:
            self.validate_steps(resolved)
        return Pipeline(resolved, memory=self.memory, verbose=self.verbose)

    def dynamic_compose(
        self, X: ArrayLike, y: Optional[np.ndarray], rules: Mapping[str, Mapping[str, Any]]
    ) -> Pipeline:
        """Compose the steps of the rules whose ``condition(X, y)`` holds.

        Args:
            X: Data the conditions inspect.
            y: Target the conditions inspect (may be ``None``).
            rules: ``{rule: {"condition": f(X, y) -> bool, "pipeline": (name, step)}}``.

        Raises:
            ValueError: If no rule applies.
        """
        steps = [tuple(rule["pipeline"]) for rule in rules.values() if bool(rule["condition"](X, y))]
        if not steps:
            raise ValueError("No composition rule matched the data.")
        self.logger.info("Dynamic composition selected steps: %s", [s for s, _ in steps])
        return self.compose(steps)

    def create_branched_pipeline(
        self, branches: Mapping[str, TransformerSpec], n_jobs: Optional[int] = None
    ) -> AdvancedFeatureUnion:
        """Run ``branches`` in parallel on the same input and stack their outputs."""
        union = AdvancedFeatureUnion(n_jobs=n_jobs)
        for name, branch in branches.items():
            union.add_pipeline(name, branch)
        return union


__all__ = [
    "HAS_OPTUNA",
    "AdvancedFeatureUnion",
    "ColumnSelector",
    "ColumnTransformer",
    "ConditionalFeatureUnion",
    "DynamicFeatureUnion",
    "FeaturePipelineBuilder",
    "FeatureStacker",
    "ParallelFeatureProcessor",
    "PipelineComposer",
    "WeightedFeatureUnion",
    "build_type_aware_column_transformer",
    "materialize",
    "register_transformer",
    "registered_transformers",
    "resolve_transformer",
]
