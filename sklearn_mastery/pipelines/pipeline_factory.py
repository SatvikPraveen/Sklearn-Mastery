"""Pipeline factories, builders, configuration objects and optimisers.

This module offers several complementary ways of assembling scikit-learn
pipelines:

* :class:`PipelineFactory` -- level-based and configuration-driven creation
  of complete classification / regression pipelines, plus data-adaptive
  preprocessing pipelines.
* :class:`AutoMLPipelineBuilder` -- data profiling and recommendation of
  preprocessing, feature-engineering and model choices.
* :class:`ClassificationPipelineFactory` / :class:`RegressionPipelineFactory`
  -- task-specific convenience factories (binary, multiclass, imbalanced,
  ensemble, text, polynomial, ...).
* :class:`CustomPipelineBuilder` -- fluent step-by-step builder with
  conditional, parallel and branched steps.
* :class:`PipelineConfig` -- validated, mergeable pipeline configuration with
  templates.
* :class:`PipelineOptimizer` -- hyper-parameter, feature-selection,
  preprocessing and end-to-end pipeline optimisation.

Optional dependencies (``imbalanced-learn``, ``xgboost``, ``lightgbm``) are
guarded by ``HAS_IMBLEARN``, ``HAS_XGBOOST`` and ``HAS_LIGHTGBM`` flags; the
module imports without them.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from sklearn_mastery.config.logging_config import LoggerMixin, get_logger
from sklearn_mastery.config.settings import settings

try:  # optional: imbalanced-learn
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.under_sampling import RandomUnderSampler

    HAS_IMBLEARN = True
except ImportError:  # pragma: no cover - exercised only without imblearn
    ImbPipeline = None
    HAS_IMBLEARN = False

try:  # optional: xgboost
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False

try:  # optional: lightgbm
    from lightgbm import LGBMClassifier, LGBMRegressor

    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover
    HAS_LIGHTGBM = False

logger = get_logger(__name__)

ArrayLike = Union[np.ndarray, pd.DataFrame]
Step = Tuple[str, BaseEstimator]

CLASSIFICATION = "classification"
REGRESSION = "regression"

__all__ = [
    "HAS_IMBLEARN",
    "HAS_LIGHTGBM",
    "HAS_XGBOOST",
    "AutoMLPipelineBuilder",
    "ClassificationPipelineFactory",
    "CustomPipelineBuilder",
    "PipelineConfig",
    "PipelineFactory",
    "PipelineOptimizer",
    "RegressionPipelineFactory",
    "available_classifiers",
    "available_preprocessing_steps",
    "available_regressors",
    "infer_task_type",
    "profile_data",
]


# --------------------------------------------------------------------------- #
# Private helper transformers
# --------------------------------------------------------------------------- #


def _signed_log1p(X: ArrayLike) -> np.ndarray:
    """Sign-preserving ``log1p`` usable on negative values."""
    arr = np.asarray(X, dtype=float)
    return np.sign(arr) * np.log1p(np.abs(arr))


def _pure_powers(X: ArrayLike, degree: int = 2) -> np.ndarray:
    """Stack ``X, X**2, ..., X**degree`` without interaction terms."""
    arr = np.asarray(X, dtype=float)
    return np.hstack([arr**d for d in range(1, degree + 1)])


class _MixedTypeImputer(BaseEstimator, TransformerMixin):
    """Impute numeric and non-numeric columns with separate strategies.

    DataFrames keep their column names, order and (categorical) dtypes so that
    downstream encoders can still detect column types. Array input is imputed
    with a single :class:`~sklearn.impute.SimpleImputer`.

    Args:
        numeric_strategy: Strategy for numeric columns (``mean``, ``median``,
            ``most_frequent`` or ``constant``).
        categorical_strategy: Strategy for non-numeric columns.
        fill_value: Fill value used with the ``constant`` strategy.
    """

    def __init__(
        self,
        numeric_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
        fill_value: Any = None,
    ):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.fill_value = fill_value

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> _MixedTypeImputer:
        """Fit the per-type imputers.

        Args:
            X: Feature matrix.
            y: Ignored.

        Returns:
            The fitted transformer.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            self.numeric_columns_ = list(X.select_dtypes(include=[np.number]).columns)
            self.other_columns_ = [c for c in X.columns if c not in self.numeric_columns_]
            self.numeric_imputer_ = (
                SimpleImputer(
                    strategy=self.numeric_strategy, fill_value=self.fill_value, keep_empty_features=True
                ).fit(X[self.numeric_columns_])
                if self.numeric_columns_
                else None
            )
            self.categorical_imputer_ = (
                SimpleImputer(
                    strategy=self.categorical_strategy, fill_value=self.fill_value, keep_empty_features=True
                ).fit(X[self.other_columns_].astype(object))
                if self.other_columns_
                else None
            )
        else:
            arr = np.asarray(X)
            self.numeric_columns_ = None
            self.other_columns_ = None
            strategy = self.numeric_strategy if arr.dtype.kind in "fiub" else self.categorical_strategy
            self.numeric_imputer_ = SimpleImputer(
                strategy=strategy, fill_value=self.fill_value, keep_empty_features=True
            ).fit(arr)
            self.categorical_imputer_ = None
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Impute missing values.

        Args:
            X: Feature matrix.

        Returns:
            Imputed data (DataFrame in, DataFrame out).
        """
        check_is_fitted(self, "n_features_in_")
        if isinstance(X, pd.DataFrame) and self.numeric_columns_ is not None:
            parts = []
            if self.numeric_imputer_ is not None:
                parts.append(
                    pd.DataFrame(
                        self.numeric_imputer_.transform(X[self.numeric_columns_]),
                        columns=self.numeric_columns_,
                        index=X.index,
                    )
                )
            if self.categorical_imputer_ is not None:
                parts.append(
                    pd.DataFrame(
                        self.categorical_imputer_.transform(X[self.other_columns_].astype(object)),
                        columns=self.other_columns_,
                        index=X.index,
                        dtype=object,
                    )
                )
            return pd.concat(parts, axis=1)[list(X.columns)]
        return self.numeric_imputer_.transform(np.asarray(X))

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        """Return output feature names (identical to the input names)."""
        check_is_fitted(self, "n_features_in_")
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


class _ColumnTypeEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode categorical columns, vectorise text columns, pass numeric through.

    Column types are detected at ``fit`` time from the DataFrame dtypes, so no
    column names need to be known when the pipeline is built. Array input is
    passed through unchanged (or one-hot encoded when it has object dtype).
    The output is always a dense ``numpy`` array.

    Args:
        text_columns: Columns to vectorise with TF-IDF instead of one-hot
            encoding.
        handle_unknown: ``OneHotEncoder`` behaviour for unseen categories.
        max_categories: Cap on categories per column (``None`` = unlimited).
        min_frequency: Minimum frequency for a category to get its own column.
        text_max_features: Vocabulary cap for the TF-IDF vectorisers.
    """

    def __init__(
        self,
        text_columns: Optional[Sequence[str]] = None,
        handle_unknown: str = "ignore",
        max_categories: Optional[int] = None,
        min_frequency: Optional[Union[int, float]] = None,
        text_max_features: Optional[int] = 1000,
    ):
        self.text_columns = text_columns
        self.handle_unknown = handle_unknown
        self.max_categories = max_categories
        self.min_frequency = min_frequency
        self.text_max_features = text_max_features

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> _ColumnTypeEncoder:
        """Detect column types and fit the underlying encoders."""
        if isinstance(X, pd.DataFrame):
            text_cols = [c for c in (self.text_columns or []) if c in X.columns]
            cat_cols = [
                c
                for c in X.select_dtypes(include=["object", "category", "bool", "string"]).columns
                if c not in text_cols
            ]
            dt_cols = list(X.select_dtypes(include=["datetime", "datetimetz", "timedelta"]).columns)
            num_cols = [c for c in X.columns if c not in set(text_cols) | set(cat_cols) | set(dt_cols)]
            transformers: List[Tuple[str, Any, Any]] = []
            if num_cols:
                transformers.append(("numeric", "passthrough", num_cols))
            if cat_cols:
                transformers.append(
                    (
                        "categorical",
                        OneHotEncoder(
                            handle_unknown=self.handle_unknown,
                            sparse_output=False,
                            max_categories=self.max_categories,
                            min_frequency=self.min_frequency,
                        ),
                        cat_cols,
                    )
                )
            for i, col in enumerate(text_cols):
                transformers.append((f"text_{i}", TfidfVectorizer(max_features=self.text_max_features), col))
            if not transformers:
                raise ValueError("No usable columns found for encoding.")
            self.encoder_ = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0).fit(X)
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            arr = np.asarray(X)
            if arr.dtype.kind in "OUS":
                self.encoder_ = OneHotEncoder(handle_unknown=self.handle_unknown, sparse_output=False).fit(
                    arr
                )
            else:
                self.encoder_ = FunctionTransformer(feature_names_out="one-to-one").fit(arr)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Encode ``X`` into a dense numeric array."""
        check_is_fitted(self, "encoder_")
        out = self.encoder_.transform(X)
        if hasattr(out, "toarray"):
            out = out.toarray()
        return np.asarray(out)

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        """Return the encoded feature names."""
        check_is_fitted(self, "encoder_")
        return self.encoder_.get_feature_names_out()


class _DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Replace datetime columns by numeric calendar components.

    Args:
        columns: Columns to expand; ``None`` auto-detects datetime dtypes.
        components: ``pandas`` ``.dt`` accessor attributes to extract.
        drop_original: Whether to drop the source datetime column.
    """

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        components: Sequence[str] = ("year", "month", "day", "dayofweek", "dayofyear"),
        drop_original: bool = True,
    ):
        self.columns = columns
        self.components = components
        self.drop_original = drop_original

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> _DatetimeFeatureExtractor:
        """Record which columns to expand."""
        if isinstance(X, pd.DataFrame):
            if self.columns is None:
                self.columns_ = list(X.select_dtypes(include=["datetime", "datetimetz"]).columns)
            else:
                self.columns_ = [c for c in self.columns if c in X.columns]
        else:
            self.columns_ = []
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Expand datetime columns into numeric components."""
        check_is_fitted(self, "columns_")
        if not self.columns_ or not isinstance(X, pd.DataFrame):
            return X
        out = X.copy()
        for col in self.columns_:
            dt = pd.to_datetime(out[col], errors="coerce")
            for comp in self.components:
                out[f"{col}_{comp}"] = getattr(dt.dt, comp).astype(float)
            if self.drop_original:
                out = out.drop(columns=[col])
        return out

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        """Return output feature names."""
        check_is_fitted(self, "columns_")
        names = (
            list(input_features)
            if input_features is not None
            else [f"x{i}" for i in range(self.n_features_in_)]
        )
        for col in self.columns_:
            if self.drop_original and col in names:
                names.remove(col)
            names.extend(f"{col}_{comp}" for comp in self.components)
        return np.asarray(names, dtype=object)


class _OutlierClipper(BaseEstimator, TransformerMixin):
    """Winsorise numeric features to bounds learned at fit time.

    Rows are never dropped (which would break the transformer contract);
    values outside the learned bounds are clipped instead. Columns with zero
    spread (e.g. one-hot indicators) are left untouched.

    Args:
        method: ``iqr`` (Tukey fences) or ``zscore``.
        factor: IQR multiplier for the fences.
        z_threshold: Standard-deviation multiplier for ``zscore``.
    """

    def __init__(self, method: str = "iqr", factor: float = 1.5, z_threshold: float = 3.0):
        self.method = method
        self.factor = factor
        self.z_threshold = z_threshold

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> _OutlierClipper:
        """Learn per-column clipping bounds."""
        if self.method not in ("iqr", "zscore"):
            raise ValueError(f"Unknown outlier method '{self.method}'. Use 'iqr' or 'zscore'.")
        if isinstance(X, pd.DataFrame):
            self.numeric_columns_ = list(X.select_dtypes(include=[np.number]).columns)
            arr = X[self.numeric_columns_].to_numpy(dtype=float)
        else:
            self.numeric_columns_ = None
            arr = np.asarray(X, dtype=float)
        if arr.shape[1] == 0:
            self.lower_ = np.empty(0)
            self.upper_ = np.empty(0)
        elif self.method == "iqr":
            q1, q3 = np.nanpercentile(arr, [25, 75], axis=0)
            iqr = q3 - q1
            spread = iqr > 0
            self.lower_ = np.where(spread, q1 - self.factor * iqr, -np.inf)
            self.upper_ = np.where(spread, q3 + self.factor * iqr, np.inf)
        else:
            mean, std = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
            spread = std > 0
            self.lower_ = np.where(spread, mean - self.z_threshold * std, -np.inf)
            self.upper_ = np.where(spread, mean + self.z_threshold * std, np.inf)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Clip values to the learned bounds."""
        check_is_fitted(self, "lower_")
        if isinstance(X, pd.DataFrame) and self.numeric_columns_ is not None:
            out = X.copy()
            for i, col in enumerate(self.numeric_columns_):
                out[col] = np.clip(X[col].astype(float), self.lower_[i], self.upper_[i])
            return out
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        """Return output feature names (identical to input names)."""
        check_is_fitted(self, "lower_")
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


class _ConditionalTransformer(BaseEstimator, TransformerMixin):
    """Apply ``transformer`` only when ``condition(X, y)`` holds at fit time.

    Args:
        transformer: Transformer to apply conditionally.
        condition: Callable ``(X, y) -> bool`` evaluated during ``fit``.
    """

    def __init__(self, transformer: BaseEstimator, condition: Callable[[ArrayLike, Any], bool]):
        self.transformer = transformer
        self.condition = condition

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> _ConditionalTransformer:
        """Evaluate the condition and fit the wrapped transformer if active."""
        self.active_ = bool(self.condition(X, y))
        if self.active_:
            self.transformer_ = clone(self.transformer).fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Transform ``X`` when active, otherwise pass it through."""
        check_is_fitted(self, "active_")
        return self.transformer_.transform(X) if self.active_ else X


class _ShapeLogger(BaseEstimator, TransformerMixin, LoggerMixin):
    """Identity transformer that logs the data shape (debugging aid)."""

    def __init__(self, step_name: str = "debug"):
        self.step_name = step_name

    def fit(self, X: ArrayLike, y: Optional[np.ndarray] = None) -> _ShapeLogger:
        """No-op fit."""
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Log the shape and return ``X`` unchanged."""
        self.logger.info("[%s] shape=%s", self.step_name, getattr(X, "shape", None))
        return X


# --------------------------------------------------------------------------- #
# Registries
# --------------------------------------------------------------------------- #

_CLASSIFIERS: Dict[str, Tuple[type, Dict[str, Any]]] = {
    "logistic_regression": (LogisticRegression, {"max_iter": 1000}),
    "random_forest": (RandomForestClassifier, {"n_estimators": 100}),
    "gradient_boosting": (GradientBoostingClassifier, {}),
    "hist_gradient_boosting": (HistGradientBoostingClassifier, {}),
    "extra_trees": (ExtraTreesClassifier, {"n_estimators": 100}),
    "adaboost": (AdaBoostClassifier, {}),
    "svm": (SVC, {}),
    "knn": (KNeighborsClassifier, {}),
    "decision_tree": (DecisionTreeClassifier, {}),
    "naive_bayes": (GaussianNB, {}),
    "mlp": (MLPClassifier, {"max_iter": 500}),
}
_REGRESSORS: Dict[str, Tuple[type, Dict[str, Any]]] = {
    "linear_regression": (LinearRegression, {}),
    "ridge": (Ridge, {}),
    "lasso": (Lasso, {"max_iter": 10000}),
    "elastic_net": (ElasticNet, {"max_iter": 10000}),
    "random_forest": (RandomForestRegressor, {"n_estimators": 100}),
    "gradient_boosting": (GradientBoostingRegressor, {}),
    "hist_gradient_boosting": (HistGradientBoostingRegressor, {}),
    "extra_trees": (ExtraTreesRegressor, {"n_estimators": 100}),
    "adaboost": (AdaBoostRegressor, {}),
    "svr": (SVR, {}),
    "knn": (KNeighborsRegressor, {}),
    "decision_tree": (DecisionTreeRegressor, {}),
    "mlp": (MLPRegressor, {"max_iter": 500}),
}
if HAS_XGBOOST:
    _CLASSIFIERS["xgboost"] = (XGBClassifier, {"n_estimators": 100, "verbosity": 0})
    _REGRESSORS["xgboost"] = (XGBRegressor, {"n_estimators": 100, "verbosity": 0})
if HAS_LIGHTGBM:
    _CLASSIFIERS["lightgbm"] = (LGBMClassifier, {"n_estimators": 100, "verbose": -1})
    _REGRESSORS["lightgbm"] = (LGBMRegressor, {"n_estimators": 100, "verbose": -1})

_CLASSIFIER_ALIASES: Dict[str, str] = {
    "logistic": "logistic_regression",
    "logreg": "logistic_regression",
    "lr": "logistic_regression",
    "rf": "random_forest",
    "gb": "gradient_boosting",
    "gbm": "gradient_boosting",
    "gbc": "gradient_boosting",
    "hgb": "hist_gradient_boosting",
    "et": "extra_trees",
    "ada": "adaboost",
    "svc": "svm",
    "support_vector_machine": "svm",
    "k_neighbors": "knn",
    "kneighbors": "knn",
    "dt": "decision_tree",
    "nb": "naive_bayes",
    "gaussian_nb": "naive_bayes",
    "neural_network": "mlp",
    "xgb": "xgboost",
    "lgbm": "lightgbm",
}
_REGRESSOR_ALIASES: Dict[str, str] = {
    "linear": "linear_regression",
    "ols": "linear_regression",
    "elasticnet": "elastic_net",
    "rf": "random_forest",
    "gb": "gradient_boosting",
    "gbm": "gradient_boosting",
    "gbr": "gradient_boosting",
    "hgb": "hist_gradient_boosting",
    "et": "extra_trees",
    "ada": "adaboost",
    "svm": "svr",
    "support_vector_regression": "svr",
    "k_neighbors": "knn",
    "kneighbors": "knn",
    "dt": "decision_tree",
    "neural_network": "mlp",
    "xgb": "xgboost",
    "lgbm": "lightgbm",
}
_OPTIONAL_MODELS = {"xgboost": ("xgboost", HAS_XGBOOST), "lightgbm": ("lightgbm", HAS_LIGHTGBM)}

_SCALERS: Dict[str, Optional[type]] = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "maxabs": MaxAbsScaler,
    "none": None,
}
_SCALER_ALIASES: Dict[str, str] = {
    "standard_scaler": "standard",
    "standardscaler": "standard",
    "zscore": "standard",
    "minmax_scaler": "minmax",
    "min_max": "minmax",
    "robust_scaler": "robust",
    "maxabs_scaler": "maxabs",
    "max_abs": "maxabs",
    "passthrough": "none",
}

_STEP_ALIASES: Dict[str, str] = {
    "scaler": "standard_scaler",
    "scaling": "standard_scaler",
    "standard": "standard_scaler",
    "minmax": "minmax_scaler",
    "robust": "robust_scaler",
    "maxabs": "maxabs_scaler",
    "imputer": "missing_value_imputer",
    "imputation": "missing_value_imputer",
    "simple_imputer": "missing_value_imputer",
    "handle_missing": "missing_value_imputer",
    "missing_values": "missing_value_imputer",
    "missing_value_handler": "missing_value_imputer",
    "missing_imputer": "missing_value_imputer",
    "encoder": "onehot_encoder",
    "one_hot": "onehot_encoder",
    "onehot": "onehot_encoder",
    "one_hot_encoder": "onehot_encoder",
    "categorical_encoder": "onehot_encoder",
    "categorical_encoding": "onehot_encoder",
    "datetime": "datetime_features",
    "datetime_feature_extraction": "datetime_features",
    "outlier_removal": "outlier_clipper",
    "outlier_handling": "outlier_clipper",
    "outliers": "outlier_clipper",
    "polynomial": "polynomial_features",
    "interactions": "interaction_features",
    "feature_interactions": "interaction_features",
    "log": "log_transform",
    "yeo_johnson": "power_transform",
    "power": "power_transform",
    "quantile": "quantile_transform",
    "kbins": "binning",
    "discretizer": "binning",
    "select_k_best": "feature_selection",
    "univariate": "feature_selection",
    "selectkbest": "feature_selection",
    "rfe": "rfe_selection",
    "model_based": "model_based_selection",
    "select_from_model": "model_based_selection",
    "smote": "handle_imbalance",
    "oversample": "handle_imbalance",
    "undersample": "handle_imbalance",
    "balance": "handle_imbalance",
    "debug": "debug_logger",
}
_PREPROCESSING_STEPS: Tuple[str, ...] = (
    "standard_scaler",
    "minmax_scaler",
    "robust_scaler",
    "maxabs_scaler",
    "missing_value_imputer",
    "knn_imputer",
    "onehot_encoder",
    "ordinal_encoder",
    "datetime_features",
    "outlier_clipper",
    "variance_threshold",
    "polynomial_features",
    "interaction_features",
    "log_transform",
    "power_transform",
    "quantile_transform",
    "binning",
    "pca",
    "feature_selection",
    "rfe_selection",
    "model_based_selection",
    "handle_imbalance",
    "debug_logger",
)


def _normalize_name(name: str, aliases: Optional[Dict[str, str]] = None) -> str:
    """Normalise a user-supplied name (case, separators, aliases)."""
    if not isinstance(name, str):
        raise ValueError(f"Expected a string name, got {type(name).__name__}.")
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    return (aliases or {}).get(key, key)


def available_classifiers() -> List[str]:
    """Return the names of all registered classifiers."""
    return sorted(_CLASSIFIERS)


def available_regressors() -> List[str]:
    """Return the names of all registered regressors."""
    return sorted(_REGRESSORS)


def available_preprocessing_steps() -> List[str]:
    """Return the canonical names of all registered preprocessing steps."""
    return list(_PREPROCESSING_STEPS)


# Estimators whose ``n_jobs`` is deprecated (no effect since scikit-learn 1.8).
_N_JOBS_DEPRECATED: Tuple[type, ...] = (LogisticRegression,)


def _instantiate(
    cls: type,
    defaults: Dict[str, Any],
    params: Dict[str, Any],
    random_state: Optional[int],
    n_jobs: Optional[int],
) -> BaseEstimator:
    """Instantiate ``cls`` with defaults, overrides and injected ``random_state``/``n_jobs``."""
    est = cls(**{**defaults, **params})
    available = est.get_params()
    if "random_state" in available and "random_state" not in params:
        est.set_params(random_state=random_state)
    if (
        n_jobs is not None
        and "n_jobs" in available
        and "n_jobs" not in params
        and not isinstance(est, _N_JOBS_DEPRECATED)
    ):
        est.set_params(n_jobs=n_jobs)
    return est


def _unknown_model_error(name: str, task_type: str) -> ValueError:
    key = _normalize_name(name)
    if key in _OPTIONAL_MODELS and not _OPTIONAL_MODELS[key][1]:
        return ValueError(f"Model '{name}' requires the optional package '{_OPTIONAL_MODELS[key][0]}'.")
    registry = available_classifiers() if task_type == CLASSIFICATION else available_regressors()
    return ValueError(f"Unknown {task_type} model '{name}'. Available: {registry}")


def _make_classifier(
    name: str, random_state: Optional[int] = None, n_jobs: Optional[int] = None, /, **params: Any
) -> BaseEstimator:
    """Build a classifier from its registry name.

    Args:
        name: Registry name or alias (e.g. ``'random_forest'``, ``'svm'``).
        random_state: Injected when the estimator supports it (positional-only so
            that ``params`` may carry its own ``random_state``, which wins).
        n_jobs: Injected when the estimator supports it (positional-only).
        **params: Estimator hyper-parameters (override the defaults).

    Returns:
        Unfitted classifier.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    key = _normalize_name(name, _CLASSIFIER_ALIASES)
    if key not in _CLASSIFIERS:
        raise _unknown_model_error(name, CLASSIFICATION)
    cls, defaults = _CLASSIFIERS[key]
    return _instantiate(cls, defaults, params, random_state, n_jobs)


def _make_regressor(
    name: str, random_state: Optional[int] = None, n_jobs: Optional[int] = None, /, **params: Any
) -> BaseEstimator:
    """Build a regressor from its registry name (see :func:`_make_classifier`)."""
    key = _normalize_name(name, _REGRESSOR_ALIASES)
    if key not in _REGRESSORS:
        raise _unknown_model_error(name, REGRESSION)
    cls, defaults = _REGRESSORS[key]
    return _instantiate(cls, defaults, params, random_state, n_jobs)


def _make_model(
    name: str,
    task_type: str,
    random_state: Optional[int] = None,
    n_jobs: Optional[int] = None,
    /,
    **params: Any,
) -> BaseEstimator:
    """Build a classifier or regressor depending on ``task_type``."""
    if task_type == CLASSIFICATION:
        return _make_classifier(name, random_state, n_jobs, **params)
    if task_type == REGRESSION:
        return _make_regressor(name, random_state, n_jobs, **params)
    raise ValueError(f"task_type must be 'classification' or 'regression', got '{task_type}'.")


def _with_probabilities(estimator: BaseEstimator) -> BaseEstimator:
    """Ensure ``estimator`` exposes ``predict_proba``.

    Classifiers without probability outputs (e.g. ``SVC``) are wrapped in
    ``CalibratedClassifierCV(ensemble=False)``, the replacement scikit-learn
    recommends for the deprecated ``SVC(probability=True)``.
    """
    if hasattr(estimator, "predict_proba"):
        return estimator
    return CalibratedClassifierCV(estimator, ensemble=False)


def _model_is_known(name: str, task_type: Optional[str] = None) -> bool:
    """Return whether ``name`` resolves to a registered model for ``task_type`` (or any)."""
    in_c = _normalize_name(name, _CLASSIFIER_ALIASES) in _CLASSIFIERS
    in_r = _normalize_name(name, _REGRESSOR_ALIASES) in _REGRESSORS
    if task_type == CLASSIFICATION:
        return in_c
    if task_type == REGRESSION:
        return in_r
    return in_c or in_r


def _resolve_model_task(name: str, task_type: Optional[str] = None) -> str:
    """Resolve the task type implied by a model name and an optional hint."""
    if task_type in (CLASSIFICATION, REGRESSION):
        if not _model_is_known(name, task_type):
            raise _unknown_model_error(name, task_type)
        return task_type
    if task_type not in (None, "auto"):
        raise ValueError(f"task_type must be 'classification', 'regression' or 'auto', got '{task_type}'.")
    in_c, in_r = _model_is_known(name, CLASSIFICATION), _model_is_known(name, REGRESSION)
    if in_c:
        return CLASSIFICATION
    if in_r:
        return REGRESSION
    raise ValueError(
        f"Unknown model '{name}'. Available classifiers: {available_classifiers()}; "
        f"regressors: {available_regressors()}"
    )


def _model_name_from_estimator(estimator: BaseEstimator) -> Optional[str]:
    """Reverse-lookup the registry name of an estimator instance."""
    for registry in (_CLASSIFIERS, _REGRESSORS):
        for name, (cls, _) in registry.items():
            if type(estimator) is cls:
                return name
    return None


def _estimator_step_name(task_type: str) -> str:
    return "classifier" if task_type == CLASSIFICATION else "regressor"


def _default_scoring(task_type: str) -> str:
    return "accuracy" if task_type == CLASSIFICATION else "r2"


def _default_estimator(task_type: str, random_state: Optional[int]) -> BaseEstimator:
    """Cheap, deterministic estimator used to score preprocessing variants."""
    if task_type == CLASSIFICATION:
        return LogisticRegression(max_iter=1000, random_state=random_state)
    return Ridge(random_state=random_state)


def infer_task_type(y: Any) -> str:
    """Infer ``'classification'`` or ``'regression'`` from a target vector.

    Args:
        y: Target values.

    Returns:
        Task type string.

    Raises:
        ValueError: If ``y`` is ``None`` or has an unsupported target type
            (e.g. multilabel).
    """
    if y is None:
        raise ValueError("A target vector is required to infer the task type.")
    kind = type_of_target(np.asarray(y))
    if kind in ("binary", "multiclass"):
        return CLASSIFICATION
    if kind == "continuous":
        return REGRESSION
    raise ValueError(f"Unsupported target type '{kind}' for pipeline creation.")


def _cv_splitter(cv: Optional[Union[int, Any]], task_type: str, random_state: Optional[int]) -> Any:
    """Turn an integer ``cv`` into a shuffled (stratified) K-fold splitter."""
    if cv is None:
        cv = settings.DEFAULT_CV_FOLDS
    if isinstance(cv, (int, np.integer)):
        if task_type == CLASSIFICATION:
            return StratifiedKFold(n_splits=int(cv), shuffle=True, random_state=random_state)
        return KFold(n_splits=int(cv), shuffle=True, random_state=random_state)
    return cv


def _make_scaler(name: Optional[str], **params: Any) -> Optional[BaseEstimator]:
    """Build a scaler by name (``standard``, ``minmax``, ``robust``, ``maxabs``, ``none``)."""
    if name is None or name is False:
        return None
    if name is True:
        name = "standard"
    key = _normalize_name(name, _SCALER_ALIASES)
    if key not in _SCALERS:
        raise ValueError(f"Unknown scaler '{name}'. Available: {sorted(_SCALERS)}")
    cls = _SCALERS[key]
    return cls(**params) if cls is not None else None


def _make_feature_selector(
    method: Union[str, bool, None],
    task_type: str,
    *,
    k: Optional[int] = None,
    percentile: Optional[float] = None,
    estimator: Optional[BaseEstimator] = None,
    n_features_to_select: Optional[Union[int, float]] = None,
    threshold: Optional[Union[str, float]] = None,
    random_state: Optional[int] = None,
    **params: Any,
) -> BaseEstimator:
    """Build a feature-selection transformer.

    Args:
        method: ``'univariate'`` (default, ``SelectKBest``/``SelectPercentile``),
            ``'mutual_info'``, ``'rfe'``, ``'model_based'`` or ``'variance'``.
            ``True`` maps to ``'univariate'``.
        task_type: ``'classification'`` or ``'regression'``.
        k: Number of features for ``SelectKBest`` / ``SelectFromModel``.
        percentile: Percentile of features to keep (univariate only).
        estimator: Estimator for ``rfe`` / ``model_based``.
        n_features_to_select: Features kept by RFE (int or fraction).
        threshold: ``SelectFromModel`` threshold.
        random_state: Seed for randomised default estimators.
        **params: Extra keyword arguments passed to the selector.

    Returns:
        Unfitted selector.

    Raises:
        ValueError: If ``method`` is unknown.
    """
    if method is True or method is None:
        method = "univariate"
    key = _normalize_name(str(method))
    key = {
        "kbest": "univariate",
        "select_k_best": "univariate",
        "selectkbest": "univariate",
        "percentile": "univariate",
        "from_model": "model_based",
        "select_from_model": "model_based",
        "variance_threshold": "variance",
    }.get(key, key)
    if key in ("univariate", "mutual_info"):
        if key == "mutual_info":
            score_func = mutual_info_classif if task_type == CLASSIFICATION else mutual_info_regression
        else:
            score_func = f_classif if task_type == CLASSIFICATION else f_regression
        if k is not None:
            return SelectKBest(score_func=score_func, k=k, **params)
        return SelectPercentile(
            score_func=score_func, percentile=50 if percentile is None else percentile, **params
        )
    if key == "rfe":
        base = (
            estimator
            if estimator is not None
            else (
                LogisticRegression(max_iter=1000, random_state=random_state)
                if task_type == CLASSIFICATION
                else LinearRegression()
            )
        )
        n_select = n_features_to_select if n_features_to_select is not None else (k if k is not None else 0.5)
        return RFE(clone(base), n_features_to_select=n_select, **params)
    if key == "model_based":
        base = (
            estimator
            if estimator is not None
            else (
                RandomForestClassifier(n_estimators=100, random_state=random_state)
                if task_type == CLASSIFICATION
                else RandomForestRegressor(n_estimators=100, random_state=random_state)
            )
        )
        return SelectFromModel(clone(base), threshold=threshold, max_features=k, **params)
    if key == "variance":
        return VarianceThreshold(**params)
    raise ValueError(
        f"Unknown feature selection method '{method}'. Use 'univariate', 'mutual_info', 'rfe', 'model_based' or 'variance'."
    )


def _make_sampler(
    strategy: str, random_state: Optional[int] = None, sampling_strategy: Any = "auto", **params: Any
) -> BaseEstimator:
    """Build an ``imbalanced-learn`` resampler.

    Raises:
        ImportError: If ``imbalanced-learn`` is not installed.
        ValueError: If ``strategy`` is unknown.
    """
    if not HAS_IMBLEARN:
        raise ImportError(
            "imbalanced-learn is required for resampling strategies (pip install imbalanced-learn)."
        )
    key = _normalize_name(strategy)
    if key == "smote":
        return SMOTE(random_state=random_state, sampling_strategy=sampling_strategy, **params)
    if key in ("oversample", "random_oversample", "random_over_sampler"):
        return RandomOverSampler(random_state=random_state, sampling_strategy=sampling_strategy, **params)
    if key in ("undersample", "random_undersample", "random_under_sampler"):
        return RandomUnderSampler(random_state=random_state, sampling_strategy=sampling_strategy, **params)
    raise ValueError(
        f"Unknown balancing strategy '{strategy}'. Use 'smote', 'oversample', 'undersample' or 'class_weight'."
    )


def _make_preprocessing_step(
    name: str,
    *,
    task_type: str = CLASSIFICATION,
    random_state: Optional[int] = None,
    **params: Any,
) -> Step:
    """Build a named preprocessing step from the registry.

    Args:
        name: Canonical step name or alias (see :func:`available_preprocessing_steps`).
        task_type: Task type used by feature selectors.
        random_state: Seed for randomised steps.
        **params: Keyword arguments for the underlying transformer.

    Returns:
        ``(step_name, transformer)`` tuple.

    Raises:
        ValueError: If the step is unknown.
    """
    key = _normalize_name(name, _STEP_ALIASES)
    if key in ("standard_scaler", "minmax_scaler", "robust_scaler", "maxabs_scaler"):
        return key, _make_scaler(key.split("_")[0], **params)
    if key == "missing_value_imputer":
        if "strategy" in params:
            params["numeric_strategy"] = params.pop("strategy")
        return key, _MixedTypeImputer(**params)
    if key == "knn_imputer":
        return key, KNNImputer(**params)
    if key == "onehot_encoder":
        return key, _ColumnTypeEncoder(**params)
    if key == "ordinal_encoder":
        return key, OrdinalEncoder(**{"handle_unknown": "use_encoded_value", "unknown_value": -1, **params})
    if key == "datetime_features":
        return key, _DatetimeFeatureExtractor(**params)
    if key == "outlier_clipper":
        return key, _OutlierClipper(**params)
    if key == "variance_threshold":
        return key, VarianceThreshold(**params)
    if key == "polynomial_features":
        return key, PolynomialFeatures(**{"degree": 2, "include_bias": False, **params})
    if key == "interaction_features":
        return key, PolynomialFeatures(
            **{"degree": 2, "interaction_only": True, "include_bias": False, **params}
        )
    if key == "log_transform":
        return key, FunctionTransformer(_signed_log1p, feature_names_out="one-to-one", **params)
    if key == "power_transform":
        return key, PowerTransformer(**params)
    if key == "quantile_transform":
        return key, QuantileTransformer(**{"n_quantiles": 100, "random_state": random_state, **params})
    if key == "binning":
        return key, KBinsDiscretizer(**{"n_bins": 5, "encode": "ordinal", "strategy": "uniform", **params})
    if key == "pca":
        return key, PCA(**{"random_state": random_state, **params})
    if key == "feature_selection":
        return key, _make_feature_selector(
            params.pop("method", "univariate"), task_type, random_state=random_state, **params
        )
    if key == "rfe_selection":
        return "feature_selection", _make_feature_selector(
            "rfe", task_type, random_state=random_state, **params
        )
    if key == "model_based_selection":
        return "feature_selection", _make_feature_selector(
            "model_based", task_type, random_state=random_state, **params
        )
    if key == "handle_imbalance":
        method = params.pop("method", params.pop("strategy", "smote"))
        return "sampler", _make_sampler(method, random_state=random_state, **params)
    if key == "debug_logger":
        return key, _ShapeLogger(**params)
    raise ValueError(f"Unknown preprocessing step '{name}'. Available: {available_preprocessing_steps()}")


def _route_params(preprocessing_params: Optional[Dict[str, Any]], step_name: str) -> Dict[str, Any]:
    """Extract the parameters addressed to ``step_name``.

    Supports both ``{'step__param': value}`` and ``{'step': {'param': value}}``
    layouts; aliases of the step name are honoured.
    """
    if not preprocessing_params:
        return {}
    canonical = _normalize_name(step_name, _STEP_ALIASES)
    out: Dict[str, Any] = {}
    for key, value in preprocessing_params.items():
        if "__" in key:
            step, param = key.split("__", 1)
            if _normalize_name(step, _STEP_ALIASES) == canonical:
                out[param] = value
        elif _normalize_name(key, _STEP_ALIASES) == canonical and isinstance(value, dict):
            out.update(value)
    return out


def _unique_step_name(name: str, existing: Sequence[str]) -> str:
    """Return ``name`` suffixed so that it does not clash with ``existing`` names."""
    if name not in existing:
        return name
    i = 2
    while f"{name}_{i}" in existing:
        i += 1
    return f"{name}_{i}"


def _is_sampler(estimator: Any) -> bool:
    return hasattr(estimator, "fit_resample")


def _assemble_pipeline(steps: List[Step], memory: Any = None, verbose: bool = False) -> Pipeline:
    """Build a ``Pipeline`` (or ``imblearn`` pipeline when a sampler is present)."""
    if any(_is_sampler(est) for _, est in steps):
        if not HAS_IMBLEARN:  # pragma: no cover - samplers cannot exist without imblearn
            raise ImportError("imbalanced-learn is required for pipelines containing samplers.")
        return ImbPipeline(steps, memory=memory, verbose=verbose)
    return Pipeline(steps, memory=memory, verbose=verbose)


def _default_param_grid(model_name: Optional[str], task_type: str, prefix: str) -> Dict[str, List[Any]]:
    """Return a sensible default hyper-parameter grid for a registered model."""
    p = f"{prefix}__"
    grids: Dict[str, Dict[str, List[Any]]] = {
        "random_forest": {
            f"{p}n_estimators": [100, 200],
            f"{p}max_depth": [None, 10, 20],
            f"{p}min_samples_split": [2, 5],
        },
        "extra_trees": {f"{p}n_estimators": [100, 200], f"{p}max_depth": [None, 10, 20]},
        "gradient_boosting": {
            f"{p}n_estimators": [100, 200],
            f"{p}learning_rate": [0.05, 0.1],
            f"{p}max_depth": [3, 5],
        },
        "hist_gradient_boosting": {f"{p}learning_rate": [0.05, 0.1], f"{p}max_iter": [100, 200]},
        "logistic_regression": {f"{p}C": [0.01, 0.1, 1.0, 10.0]},
        "svm": {f"{p}C": [0.1, 1.0, 10.0], f"{p}gamma": ["scale", "auto"]},
        "svr": {f"{p}C": [0.1, 1.0, 10.0], f"{p}epsilon": [0.05, 0.1, 0.2]},
        "knn": {f"{p}n_neighbors": [3, 5, 7, 11], f"{p}weights": ["uniform", "distance"]},
        "decision_tree": {f"{p}max_depth": [None, 5, 10, 20], f"{p}min_samples_split": [2, 5, 10]},
        "ridge": {f"{p}alpha": [0.1, 1.0, 10.0, 100.0]},
        "lasso": {f"{p}alpha": [0.001, 0.01, 0.1, 1.0]},
        "elastic_net": {f"{p}alpha": [0.01, 0.1, 1.0], f"{p}l1_ratio": [0.2, 0.5, 0.8]},
        "mlp": {f"{p}hidden_layer_sizes": [(50,), (100,)], f"{p}alpha": [0.0001, 0.001]},
        "adaboost": {f"{p}n_estimators": [50, 100], f"{p}learning_rate": [0.5, 1.0]},
        "xgboost": {
            f"{p}n_estimators": [100, 200],
            f"{p}max_depth": [3, 6],
            f"{p}learning_rate": [0.05, 0.1],
        },
        "lightgbm": {
            f"{p}n_estimators": [100, 200],
            f"{p}num_leaves": [15, 31],
            f"{p}learning_rate": [0.05, 0.1],
        },
    }
    if model_name is None:
        return {}
    key = _normalize_name(
        model_name, _CLASSIFIER_ALIASES if task_type == CLASSIFICATION else _REGRESSOR_ALIASES
    )
    return dict(grids.get(key, {}))


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# --------------------------------------------------------------------------- #
# Data profiling
# --------------------------------------------------------------------------- #


def _is_text_series(series: pd.Series) -> bool:
    """Heuristically decide whether an object column holds free text."""
    values = series.dropna()
    if values.empty:
        return False
    sample = values.astype(str).head(500)
    avg_tokens = sample.str.split().str.len().mean()
    avg_len = sample.str.len().mean()
    return bool(avg_tokens >= 4 or avg_len > 30)


def profile_data(X: ArrayLike, y: Optional[Any] = None) -> Dict[str, Any]:
    """Profile a dataset to drive pipeline construction.

    Args:
        X: Feature matrix (DataFrame or array).
        y: Optional target vector.

    Returns:
        Dictionary with column-type lists (``numeric_columns``,
        ``categorical_columns``, ``datetime_columns``, ``text_columns``,
        ``boolean_columns``), missing-value statistics (``missing_percentage``,
        ``missing_by_column``, ``has_missing``), outlier and skewness
        information, size/dimensionality categories and target statistics
        (``target_type``, ``task_type``, ``n_classes``, ``class_balance``,
        ``imbalance_ratio``, ``is_imbalanced``).
    """
    df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    n_samples, n_features = df.shape

    datetime_cols = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)
    bool_cols = list(df.select_dtypes(include=["bool"]).columns)
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in bool_cols]
    object_cols = [
        c
        for c in df.select_dtypes(include=["object", "category", "string"]).columns
        if c not in datetime_cols
    ]
    text_cols = [c for c in object_cols if df[c].dtype == object and _is_text_series(df[c])]
    categorical_cols = [c for c in object_cols if c not in text_cols]
    high_cardinality = [c for c in categorical_cols if df[c].nunique(dropna=True) > 50]

    missing_by_column = df.isna().mean().mul(100.0)
    missing_percentage = float(df.isna().to_numpy().mean() * 100.0) if df.size else 0.0

    outlier_cols: List[Any] = []
    skewed_cols: List[Any] = []
    for col in numeric_cols:
        values = df[col].to_numpy(dtype=float)
        values = values[~np.isnan(values)]
        if values.size < 10:
            continue
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        if iqr > 0 and (np.any(values < q1 - 3 * iqr) or np.any(values > q3 + 3 * iqr)):
            outlier_cols.append(col)
        if values.std() > 0 and abs(float(pd.Series(values).skew())) > 2.0:
            skewed_cols.append(col)

    profile: Dict[str, Any] = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "text_columns": text_cols,
        "boolean_columns": bool_cols,
        "high_cardinality_columns": high_cardinality,
        "missing_percentage": missing_percentage,
        "missing_by_column": {c: float(v) for c, v in missing_by_column.items() if v > 0},
        "has_missing": missing_percentage > 0,
        "outlier_columns": outlier_cols,
        "has_outliers": bool(outlier_cols),
        "skewed_columns": skewed_cols,
        "duplicate_rows": int(df.duplicated().sum()) if not datetime_cols else 0,
        "size_category": "small" if n_samples < 1000 else ("medium" if n_samples < 10000 else "large"),
        "dimensionality": "high" if n_features > 100 else ("medium" if n_features > 20 else "low"),
        "target_type": None,
        "task_type": None,
        "n_classes": None,
        "class_balance": None,
        "imbalance_ratio": None,
        "is_imbalanced": False,
    }

    if y is not None:
        y_arr = np.asarray(y)
        target_type = type_of_target(y_arr)
        profile["target_type"] = target_type
        if target_type in ("binary", "multiclass"):
            profile["task_type"] = CLASSIFICATION
            classes, counts = np.unique(y_arr, return_counts=True)
            profile["n_classes"] = len(classes)
            profile["class_balance"] = {
                cls.item() if hasattr(cls, "item") else cls: int(c) for cls, c in zip(classes, counts)
            }
            ratio = float(counts.max() / counts.min()) if counts.min() > 0 else float("inf")
            profile["imbalance_ratio"] = ratio
            profile["is_imbalanced"] = ratio >= 3.0
        elif target_type == "continuous":
            profile["task_type"] = REGRESSION
    return profile


# --------------------------------------------------------------------------- #
# PipelineConfig
# --------------------------------------------------------------------------- #


class PipelineConfig(LoggerMixin):
    """Validated, mergeable pipeline configuration.

    The configuration is a nested dictionary with the optional sections
    ``task_type``, ``preprocessing``, ``feature_engineering``,
    ``feature_selection``, ``model``, ``evaluation`` and ``pipeline``.

    Example::

        PipelineConfig({
            "preprocessing": {"scaling": "standard", "feature_selection": "univariate", "k_features": 3},
            "model": {"type": "logistic_regression", "C": 1.0},
        })

    Args:
        config: Configuration dictionary (deep-copied) or another
            :class:`PipelineConfig`.
        validate: Validate on construction.

    Raises:
        ValueError: If validation fails (unknown model, scaler, section, ...).
    """

    KNOWN_SECTIONS: Tuple[str, ...] = (
        "task_type",
        "preprocessing",
        "feature_engineering",
        "feature_selection",
        "model",
        "evaluation",
        "pipeline",
        "name",
        "description",
    )
    MODEL_META_KEYS: Tuple[str, ...] = ("type", "name", "hyperparameters", "params")

    def __init__(self, config: Optional[Union[Dict[str, Any], PipelineConfig]] = None, validate: bool = True):
        if isinstance(config, PipelineConfig):
            config = config.to_dict()
        if config is not None and not isinstance(config, dict):
            raise ValueError(f"config must be a dict or PipelineConfig, got {type(config).__name__}.")
        self.config: Dict[str, Any] = copy.deepcopy(config) if config else {}
        if validate:
            self.validate()

    # ---- accessors -------------------------------------------------------
    @property
    def task_type(self) -> Optional[str]:
        """Explicit task type, or the one implied by the model type."""
        explicit = self.config.get("task_type")
        if explicit in (CLASSIFICATION, REGRESSION):
            return explicit
        model_type = self.get_model_config().get("type")
        if model_type:
            try:
                return _resolve_model_task(model_type, None)
            except ValueError:
                return None
        return None

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Return the ``preprocessing`` section (copy)."""
        return dict(self.config.get("preprocessing", {}) or {})

    def get_feature_engineering_config(self) -> Dict[str, Any]:
        """Return the ``feature_engineering`` section (copy)."""
        return dict(self.config.get("feature_engineering", {}) or {})

    def get_model_config(self) -> Dict[str, Any]:
        """Return the ``model`` section (copy)."""
        return dict(self.config.get("model", {}) or {})

    def get_evaluation_config(self) -> Dict[str, Any]:
        """Return the ``evaluation`` section (copy)."""
        return dict(self.config.get("evaluation", {}) or {})

    def get_model_params(self) -> Dict[str, Any]:
        """Return the model hyper-parameters (nested ``hyperparameters`` merged with inline keys)."""
        model = self.get_model_config()
        params = {k: v for k, v in model.items() if k not in self.MODEL_META_KEYS}
        params.update(model.get("hyperparameters") or {})
        params.update(model.get("params") or {})
        return params

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style ``get`` on the top-level sections."""
        return self.config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.config[key]

    def __contains__(self, key: str) -> bool:
        return key in self.config

    def __repr__(self) -> str:
        return f"PipelineConfig({self.config!r})"

    # ---- legacy flat-option API -----------------------------------------
    def set_option(self, key: str, value: Any) -> PipelineConfig:
        """Set an option; dotted keys (``'preprocessing.scaling'``) address nested sections."""
        target = self.config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
        return self

    def get_option(self, key: str, default: Any = None) -> Any:
        """Get an option set via :meth:`set_option` (dotted keys supported)."""
        target: Any = self.config
        for part in key.split("."):
            if not isinstance(target, dict) or part not in target:
                return default
            target = target[part]
        return target

    # ---- validation / merging -------------------------------------------
    def validate(self) -> PipelineConfig:
        """Validate the configuration.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: On unknown sections, models, scalers, feature-selection
                methods or invalid evaluation settings.
        """
        unknown = sorted(set(self.config) - set(self.KNOWN_SECTIONS))
        if unknown:
            raise ValueError(
                f"Unknown configuration sections {unknown}. Allowed: {list(self.KNOWN_SECTIONS)}"
            )
        task_type = self.config.get("task_type")
        if task_type not in (None, "auto", CLASSIFICATION, REGRESSION):
            raise ValueError(
                f"task_type must be 'classification', 'regression' or 'auto', got '{task_type}'."
            )

        model = self.config.get("model")
        if model is not None:
            if not isinstance(model, dict):
                raise ValueError("The 'model' section must be a dictionary.")
            model_type = model.get("type", model.get("name"))
            if model_type is not None:
                hint = task_type if task_type in (CLASSIFICATION, REGRESSION) else None
                if not _model_is_known(model_type, hint):
                    raise _unknown_model_error(model_type, hint or CLASSIFICATION)

        pre = self.config.get("preprocessing")
        if pre is not None:
            if not isinstance(pre, dict):
                raise ValueError("The 'preprocessing' section must be a dictionary.")
            scaler = pre.get("scaler", pre.get("scaling"))
            if isinstance(scaler, str) and _normalize_name(scaler, _SCALER_ALIASES) not in _SCALERS:
                raise ValueError(f"Unknown scaler '{scaler}'. Available: {sorted(_SCALERS)}")
            selection = pre.get("feature_selection")
            if isinstance(selection, str) and selection.lower() not in (
                "univariate",
                "kbest",
                "select_k_best",
                "mutual_info",
                "rfe",
                "model_based",
                "variance",
                "none",
                "auto",
            ):
                raise ValueError(f"Unknown feature selection method '{selection}'.")

        evaluation = self.config.get("evaluation")
        if evaluation is not None:
            if not isinstance(evaluation, dict):
                raise ValueError("The 'evaluation' section must be a dictionary.")
            folds = evaluation.get("cv_folds")
            if folds is not None and (not isinstance(folds, (int, np.integer)) or folds < 2):
                raise ValueError(f"evaluation.cv_folds must be an integer >= 2, got {folds!r}.")
        return self

    def merge(self, override: Union[Dict[str, Any], PipelineConfig]) -> PipelineConfig:
        """Return a new configuration with ``override`` deep-merged on top of this one."""
        other = override.to_dict() if isinstance(override, PipelineConfig) else (override or {})
        return PipelineConfig(_deep_merge(self.config, other))

    def to_dict(self) -> Dict[str, Any]:
        """Return a deep copy of the underlying dictionary."""
        return copy.deepcopy(self.config)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> PipelineConfig:
        """Construct from a dictionary (alias of the constructor)."""
        return cls(config)

    # ---- templates -------------------------------------------------------
    @classmethod
    def quick_start_classification(cls) -> PipelineConfig:
        """Fast, robust classification template (imputer, scaler, random forest)."""
        return cls(
            {
                "task_type": CLASSIFICATION,
                "preprocessing": {
                    "handle_missing": "auto",
                    "encode_categorical": True,
                    "scaling": "standard",
                },
                "model": {"type": "random_forest", "n_estimators": 100},
                "evaluation": {"cv_folds": settings.DEFAULT_CV_FOLDS, "scoring": "accuracy"},
            }
        )

    @classmethod
    def advanced_classification(cls) -> PipelineConfig:
        """Classification template with feature engineering and selection."""
        return cls(
            {
                "task_type": CLASSIFICATION,
                "preprocessing": {
                    "handle_missing": "auto",
                    "encode_categorical": True,
                    "handle_outliers": "iqr",
                    "scaling": "standard",
                    "feature_selection": "univariate",
                    "percentile": 50,
                },
                "feature_engineering": {"interactions": True},
                "model": {"type": "gradient_boosting"},
                "evaluation": {"cv_folds": settings.DEFAULT_CV_FOLDS, "scoring": "f1_weighted"},
            }
        )

    @classmethod
    def quick_start_regression(cls) -> PipelineConfig:
        """Fast, robust regression template."""
        return cls(
            {
                "task_type": REGRESSION,
                "preprocessing": {
                    "handle_missing": "auto",
                    "encode_categorical": True,
                    "scaling": "standard",
                },
                "model": {"type": "random_forest", "n_estimators": 100},
                "evaluation": {"cv_folds": settings.DEFAULT_CV_FOLDS, "scoring": "r2"},
            }
        )

    @classmethod
    def advanced_regression(cls) -> PipelineConfig:
        """Regression template with feature engineering and selection."""
        return cls(
            {
                "task_type": REGRESSION,
                "preprocessing": {
                    "handle_missing": "auto",
                    "encode_categorical": True,
                    "handle_outliers": "iqr",
                    "scaling": "standard",
                    "feature_selection": "univariate",
                    "percentile": 50,
                },
                "feature_engineering": {"polynomial_features": {"degree": 2}},
                "model": {"type": "gradient_boosting"},
                "evaluation": {
                    "cv_folds": settings.DEFAULT_CV_FOLDS,
                    "scoring": "neg_root_mean_squared_error",
                },
            }
        )


# --------------------------------------------------------------------------- #
# PipelineFactory
# --------------------------------------------------------------------------- #

_LEVEL_STEPS: Dict[str, List[str]] = {
    "minimal": ["missing_value_imputer", "standard_scaler"],
    "standard": ["missing_value_imputer", "onehot_encoder", "standard_scaler"],
    "advanced": [
        "missing_value_imputer",
        "onehot_encoder",
        "outlier_clipper",
        "interaction_features",
        "standard_scaler",
        "feature_selection",
    ],
}


class PipelineFactory(LoggerMixin):
    """Factory for complete and data-adaptive scikit-learn pipelines.

    Args:
        random_state: Seed injected into every randomised component. Defaults
            to ``settings.RANDOM_SEED``.
        n_jobs: Default parallelism injected into estimators that support it.
        memory: ``joblib.Memory`` or cache directory for transformer caching.
    """

    def __init__(self, random_state: Optional[int] = None, n_jobs: Optional[int] = None, memory: Any = None):
        self.random_state = settings.RANDOM_SEED if random_state is None else random_state
        self.n_jobs = n_jobs
        self.memory = memory

    # ---- internal helpers ------------------------------------------------
    def _build_steps(
        self,
        names: Sequence[Union[str, Step]],
        task_type: str,
        preprocessing_params: Optional[Dict[str, Any]] = None,
    ) -> List[Step]:
        """Instantiate registry steps (or pass through ``(name, estimator)`` tuples)."""
        steps: List[Step] = []
        for item in names:
            if isinstance(item, tuple):
                name, est = item
            else:
                name, est = _make_preprocessing_step(
                    item,
                    task_type=task_type,
                    random_state=self.random_state,
                    **_route_params(preprocessing_params, item),
                )
            steps.append((_unique_step_name(name, [s for s, _ in steps]), est))
        return steps

    def _preprocessing_steps(
        self,
        task_type: str,
        preprocessing_level: str,
        preprocessing_steps: Optional[Sequence[Union[str, Step]]],
        preprocessing_params: Optional[Dict[str, Any]],
        feature_engineering: bool,
    ) -> List[Step]:
        if preprocessing_steps is not None:
            names: List[Union[str, Step]] = list(preprocessing_steps)
        else:
            if preprocessing_level not in _LEVEL_STEPS:
                raise ValueError(
                    f"preprocessing_level must be one of {sorted(_LEVEL_STEPS)}, got '{preprocessing_level}'."
                )
            names = list(_LEVEL_STEPS[preprocessing_level])
        if feature_engineering and not any(
            isinstance(n, str)
            and _normalize_name(n, _STEP_ALIASES) in ("interaction_features", "polynomial_features")
            for n in names
        ):
            scaler_idx = next(
                (
                    i
                    for i, n in enumerate(names)
                    if isinstance(n, str) and _normalize_name(n, _STEP_ALIASES).endswith("_scaler")
                ),
                len(names),
            )
            names.insert(scaler_idx, "interaction_features")
        return self._build_steps(names, task_type, preprocessing_params)

    def _finalize(
        self,
        steps: List[Step],
        pipeline_name: Optional[str] = None,
        memory: Any = None,
        verbose: bool = False,
    ) -> Pipeline:
        pipeline = _assemble_pipeline(
            steps, memory=self.memory if memory is None else memory, verbose=verbose
        )
        self.logger.info(
            "Created pipeline%s with %d steps: %s",
            f" '{pipeline_name}'" if pipeline_name else "",
            len(steps),
            [name for name, _ in steps],
        )
        return pipeline

    @staticmethod
    def _resolve_model_name(model_type: Optional[str], algorithm: Optional[str], default: str) -> str:
        if model_type is not None and algorithm is not None and model_type != algorithm:
            raise ValueError(
                "Pass either 'model_type' or its alias 'algorithm', not both with different values."
            )
        return model_type or algorithm or default

    def _apply_imbalance_handling(self, steps: List[Step], classifier: BaseEstimator, strategy: str) -> None:
        """Insert a resampler (imblearn) or fall back to ``class_weight='balanced'``."""
        key = _normalize_name(strategy)
        if key in ("class_weight", "balanced", "none"):
            use_sampler = False
        elif HAS_IMBLEARN:
            use_sampler = True
        else:
            self.logger.warning(
                "imbalanced-learn is not installed; falling back to class_weight='balanced' instead of '%s'.",
                strategy,
            )
            use_sampler = False
        if use_sampler:
            steps.append(("sampler", _make_sampler(key, random_state=self.random_state)))
        elif "class_weight" in classifier.get_params():
            classifier.set_params(class_weight="balanced")
        else:
            self.logger.warning(
                "%s does not support class_weight; imbalance handling skipped.", type(classifier).__name__
            )

    # ---- public API ------------------------------------------------------
    def create_classification_pipeline(
        self,
        model_type: Optional[str] = None,
        task_type: Optional[str] = CLASSIFICATION,
        *,
        algorithm: Optional[str] = None,
        preprocessing_level: str = "standard",
        preprocessing_steps: Optional[Sequence[Union[str, Step]]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        preprocessing_params: Optional[Dict[str, Any]] = None,
        feature_selection: bool = False,
        feature_selection_params: Optional[Dict[str, Any]] = None,
        feature_engineering: bool = False,
        handle_imbalance: bool = False,
        balancing_strategy: str = "smote",
        n_jobs: Optional[int] = None,
        pipeline_name: Optional[str] = None,
    ) -> Pipeline:
        """Create a complete classification pipeline.

        Args:
            model_type: Registered classifier name (``'random_forest'``,
                ``'logistic_regression'``, ``'gradient_boosting'``, ``'svm'``,
                ``'knn'``, ...). Defaults to ``'random_forest'``.
            task_type: Must be ``'classification'`` (kept for API symmetry).
            algorithm: Alias of ``model_type``.
            preprocessing_level: ``'minimal'`` (imputer + scaler, numeric data
                only), ``'standard'`` (adds categorical encoding) or
                ``'advanced'`` (adds outlier clipping, interaction features and
                univariate feature selection).
            preprocessing_steps: Explicit list of registry step names (or
                ``(name, transformer)`` tuples) overriding ``preprocessing_level``.
            model_params: Classifier hyper-parameters.
            preprocessing_params: Step parameters as ``{'step__param': value}``
                or ``{'step': {...}}``.
            feature_selection: Append a univariate feature selector.
            feature_selection_params: Arguments for :func:`_make_feature_selector`
                (e.g. ``{'k': 10}`` or ``{'method': 'rfe'}``).
            feature_engineering: Add pairwise interaction features.
            handle_imbalance: Resample with ``balancing_strategy`` (requires
                ``imbalanced-learn``) or fall back to ``class_weight='balanced'``.
            balancing_strategy: ``'smote'``, ``'oversample'``, ``'undersample'``
                or ``'class_weight'``.
            n_jobs: Parallelism for the classifier (overrides the factory default).
            pipeline_name: Optional label used in log messages.

        Returns:
            Unfitted pipeline ending in a ``'classifier'`` step.

        Raises:
            ValueError: If the model, level or step names are unknown.
        """
        if task_type not in (None, CLASSIFICATION):
            raise ValueError(
                f"create_classification_pipeline only builds classification pipelines, got task_type='{task_type}'."
            )
        name = self._resolve_model_name(model_type, algorithm, "random_forest")
        steps = self._preprocessing_steps(
            CLASSIFICATION,
            preprocessing_level,
            preprocessing_steps,
            preprocessing_params,
            feature_engineering,
        )
        if feature_selection and not any(s == "feature_selection" for s, _ in steps):
            params = dict(feature_selection_params or {})
            method = params.pop("method", "univariate")
            steps.append(
                (
                    "feature_selection",
                    _make_feature_selector(method, CLASSIFICATION, random_state=self.random_state, **params),
                )
            )
        classifier = _make_classifier(
            name, self.random_state, self.n_jobs if n_jobs is None else n_jobs, **(model_params or {})
        )
        if handle_imbalance:
            self._apply_imbalance_handling(steps, classifier, balancing_strategy)
        steps.append(("classifier", classifier))
        return self._finalize(steps, pipeline_name)

    def create_regression_pipeline(
        self,
        model_type: Optional[str] = None,
        task_type: Optional[str] = REGRESSION,
        *,
        algorithm: Optional[str] = None,
        preprocessing_level: str = "standard",
        preprocessing_steps: Optional[Sequence[Union[str, Step]]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        preprocessing_params: Optional[Dict[str, Any]] = None,
        feature_selection: bool = False,
        feature_selection_params: Optional[Dict[str, Any]] = None,
        feature_engineering: bool = False,
        n_jobs: Optional[int] = None,
        pipeline_name: Optional[str] = None,
    ) -> Pipeline:
        """Create a complete regression pipeline.

        See :meth:`create_classification_pipeline` for the shared arguments.

        Args:
            model_type: Registered regressor name (``'random_forest'``,
                ``'linear_regression'``, ``'ridge'``, ``'lasso'``, ...).
            task_type: Must be ``'regression'``.

        Returns:
            Unfitted pipeline ending in a ``'regressor'`` step.

        Raises:
            ValueError: If the model, level or step names are unknown.
        """
        if task_type not in (None, REGRESSION):
            raise ValueError(
                f"create_regression_pipeline only builds regression pipelines, got task_type='{task_type}'."
            )
        name = self._resolve_model_name(model_type, algorithm, "random_forest")
        steps = self._preprocessing_steps(
            REGRESSION, preprocessing_level, preprocessing_steps, preprocessing_params, feature_engineering
        )
        if feature_selection and not any(s == "feature_selection" for s, _ in steps):
            params = dict(feature_selection_params or {})
            method = params.pop("method", "univariate")
            steps.append(
                (
                    "feature_selection",
                    _make_feature_selector(method, REGRESSION, random_state=self.random_state, **params),
                )
            )
        regressor = _make_regressor(
            name, self.random_state, self.n_jobs if n_jobs is None else n_jobs, **(model_params or {})
        )
        steps.append(("regressor", regressor))
        return self._finalize(steps, pipeline_name)

    def create_pipeline(self, config: Union[Dict[str, Any], PipelineConfig]) -> Pipeline:
        """Create a pipeline from a configuration dictionary or :class:`PipelineConfig`.

        Recognised keys::

            task_type: 'classification' | 'regression' | 'auto'
            preprocessing:
                handle_missing: True | 'auto' | 'mean' | 'median' | 'most_frequent' | 'knn' | False
                encode_categorical: bool (default True)
                handle_outliers / outlier_handling: 'iqr' | 'zscore' | False
                scaler / scaling: 'standard' | 'minmax' | 'robust' | 'maxabs' | 'none'
                feature_selection: True | 'univariate' | 'mutual_info' | 'rfe' | 'model_based' | False
                k_features / k, percentile: selector size
            feature_engineering:
                polynomial_features: True | {'degree': 2, ...}
                interactions: bool
                log_transform / power_transform: bool
                pca: {'n_components': ...}
            model:
                type: registered model name
                hyperparameters: {...}  (inline keys are treated as hyper-parameters too)
            pipeline:
                memory: cache dir, verbose: bool

        Args:
            config: Configuration.

        Returns:
            Unfitted pipeline.

        Raises:
            ValueError: If the configuration is invalid or the model unknown.
        """
        cfg = config if isinstance(config, PipelineConfig) else PipelineConfig(config)
        model_cfg = cfg.get_model_config()
        model_type = model_cfg.get("type", model_cfg.get("name"))
        if not model_type:
            raise ValueError("Configuration must define model.type.")
        task_type = _resolve_model_task(model_type, cfg.get("task_type"))
        pre = cfg.get_preprocessing_config()
        fe = cfg.get_feature_engineering_config()
        rs = self.random_state
        steps: List[Step] = []

        handle_missing = pre.get("handle_missing", True)
        if handle_missing not in (False, None, "none"):
            strategy = (
                handle_missing
                if isinstance(handle_missing, str) and handle_missing not in ("auto", "true")
                else "median"
            )
            if strategy == "knn":
                steps.append(("missing_value_imputer", KNNImputer()))
            else:
                steps.append(("missing_value_imputer", _MixedTypeImputer(numeric_strategy=strategy)))
        if pre.get("encode_categorical", True):
            steps.append(("onehot_encoder", _ColumnTypeEncoder()))
        outliers = pre.get("handle_outliers", pre.get("outlier_handling", False))
        if outliers:
            steps.append(
                ("outlier_clipper", _OutlierClipper(method=outliers if isinstance(outliers, str) else "iqr"))
            )

        poly = fe.get("polynomial_features")
        if poly:
            steps.append(
                (
                    "polynomial_features",
                    PolynomialFeatures(
                        **{"degree": 2, "include_bias": False, **(poly if isinstance(poly, dict) else {})}
                    ),
                )
            )
        elif fe.get("interactions"):
            steps.append(
                (
                    "interaction_features",
                    PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
                )
            )
        if fe.get("log_transform"):
            steps.append(
                ("log_transform", FunctionTransformer(_signed_log1p, feature_names_out="one-to-one"))
            )
        if fe.get("power_transform"):
            steps.append(("power_transform", PowerTransformer()))

        scaler = _make_scaler(pre.get("scaler", pre.get("scaling", "standard")))
        if scaler is not None:
            steps.append(("scaler", scaler))

        pca = fe.get("pca")
        if pca:
            steps.append(("pca", PCA(**{"random_state": rs, **(pca if isinstance(pca, dict) else {})})))

        selection = pre.get("feature_selection", cfg.get("feature_selection"))
        if selection not in (None, False, "none"):
            sel_params: Dict[str, Any] = {}
            method: Any = selection
            if isinstance(selection, dict):
                sel_params = dict(selection)
                method = sel_params.pop("method", "univariate")
            k = pre.get("k_features", pre.get("k", sel_params.pop("k_features", None)))
            if k is not None:
                sel_params["k"] = k
            if "percentile" in pre:
                sel_params["percentile"] = pre["percentile"]
            steps.append(
                (
                    "feature_selection",
                    _make_feature_selector(method, task_type, random_state=rs, **sel_params),
                )
            )

        estimator = _make_model(model_type, task_type, rs, self.n_jobs, **cfg.get_model_params())
        steps.append((_estimator_step_name(task_type), estimator))
        pipe_cfg = cfg.get("pipeline", {}) or {}
        return self._finalize(
            steps,
            cfg.get("name"),
            memory=pipe_cfg.get("memory"),
            verbose=bool(pipe_cfg.get("verbose", False)),
        )

    def create_from_config(self, config: Union[Dict[str, Any], PipelineConfig]) -> Pipeline:
        """Alias of :meth:`create_pipeline`."""
        return self.create_pipeline(config)

    def create_custom_pipeline(
        self,
        custom_steps: Dict[str, Any],
        task_type: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> Pipeline:
        """Create a pipeline from grouped step names.

        Args:
            custom_steps: Dictionary with optional ``'preprocessing'``,
                ``'feature_engineering'`` and ``'feature_selection'`` lists of
                registry step names and a ``'model'`` entry (name, or dict with
                ``'type'`` and hyper-parameters).
            task_type: ``'classification'``/``'regression'``; inferred from the
                model name when ``None`` (ambiguous names default to classification).
            model_params: Hyper-parameters for the model.

        Returns:
            Unfitted pipeline.

        Raises:
            ValueError: If no model is given or a step is unknown.
        """
        model = custom_steps.get("model")
        if model is None:
            raise ValueError("custom_steps must include a 'model' entry.")
        params = dict(model_params or {})
        if isinstance(model, dict):
            model = dict(model)
            model_name = model.pop("type", model.pop("name", None))
            params.update(model.pop("hyperparameters", {}) or {})
            params.update(model)
        else:
            model_name = model
        if not model_name:
            raise ValueError("custom_steps['model'] must name a model type.")
        task = _resolve_model_task(model_name, task_type)

        names: List[Union[str, Step]] = []
        for section in ("preprocessing", "feature_engineering", "feature_selection"):
            value = custom_steps.get(section)
            if value is None or value is False:
                continue
            if isinstance(value, (str, tuple)) or not isinstance(value, (list, tuple)):
                value = [value]
            names.extend(value)
        steps = self._build_steps(names, task)
        steps.append(
            (
                _estimator_step_name(task),
                _make_model(model_name, task, self.random_state, self.n_jobs, **params),
            )
        )
        return self._finalize(steps, custom_steps.get("name"))

    def _adaptive_steps(
        self,
        profile: Dict[str, Any],
        task_type: str,
        complexity_level: str,
        include_feature_engineering: bool,
        include_outlier_removal: bool,
        debug_mode: bool,
    ) -> List[Step]:
        """Derive preprocessing steps from a data profile."""
        if complexity_level not in ("simple", "medium", "advanced"):
            raise ValueError(
                f"complexity_level must be 'simple', 'medium' or 'advanced', got '{complexity_level}'."
            )
        steps: List[Step] = []
        if debug_mode:
            steps.append(("debug_input", _ShapeLogger("input")))
        if profile["datetime_columns"]:
            steps.append(("datetime_features", _DatetimeFeatureExtractor()))
        steps.append(("missing_value_imputer", _MixedTypeImputer()))
        if include_outlier_removal and profile["has_outliers"]:
            steps.append(("outlier_clipper", _OutlierClipper(method="iqr")))
        if (
            profile["categorical_columns"]
            or profile["text_columns"]
            or profile["boolean_columns"]
            or profile["datetime_columns"]
        ):
            steps.append(
                (
                    "onehot_encoder",
                    _ColumnTypeEncoder(
                        text_columns=profile["text_columns"] or None,
                        max_categories=50 if profile["high_cardinality_columns"] else None,
                    ),
                )
            )
        if include_feature_engineering:
            if profile["skewed_columns"] and not profile["categorical_columns"]:
                steps.append(("power_transform", PowerTransformer()))
            if (
                complexity_level in ("medium", "advanced")
                and profile["dimensionality"] == "low"
                and len(profile["numeric_columns"]) >= 2
            ):
                steps.append(
                    (
                        "interaction_features",
                        PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
                    )
                )
        steps.append(
            (
                "scaler",
                RobustScaler()
                if profile["has_outliers"] and not include_outlier_removal
                else StandardScaler(),
            )
        )
        if complexity_level == "advanced" or profile["dimensionality"] == "high":
            steps.append(
                (
                    "feature_selection",
                    _make_feature_selector(
                        "univariate", task_type, percentile=50, random_state=self.random_state
                    ),
                )
            )
        if debug_mode:
            steps.append(("debug_output", _ShapeLogger("features")))
        return steps

    def create_adaptive_pipeline(
        self,
        X: ArrayLike,
        y: Optional[Any] = None,
        task_type: str = "auto",
        complexity_level: str = "medium",
        include_feature_engineering: bool = True,
        include_outlier_removal: bool = True,
        debug_mode: bool = False,
    ) -> Pipeline:
        """Create a preprocessing-only pipeline adapted to the data profile.

        Args:
            X: Feature matrix used for profiling.
            y: Optional target (used for task inference and feature selection).
            task_type: ``'auto'``, ``'classification'`` or ``'regression'``.
            complexity_level: ``'simple'``, ``'medium'`` or ``'advanced'``.
            include_feature_engineering: Add interaction / power transforms.
            include_outlier_removal: Clip outliers when detected.
            debug_mode: Add shape-logging steps.

        Returns:
            Preprocessing pipeline (no final estimator).
        """
        profile = profile_data(X, y)
        task = (
            infer_task_type(y)
            if task_type == "auto" and y is not None
            else (task_type if task_type != "auto" else CLASSIFICATION)
        )
        steps = self._adaptive_steps(
            profile, task, complexity_level, include_feature_engineering, include_outlier_removal, debug_mode
        )
        return self._finalize(steps, f"adaptive-{complexity_level}")

    def create_preprocessing_pipeline(
        self, X: ArrayLike, y: Optional[Any] = None, pipeline_type: str = "basic"
    ) -> Pipeline:
        """Create a preprocessing pipeline of a given richness.

        Args:
            X: Feature matrix used for profiling.
            y: Optional target.
            pipeline_type: ``'basic'``, ``'advanced'`` or ``'full'``.

        Returns:
            Preprocessing pipeline (no final estimator).

        Raises:
            ValueError: If ``pipeline_type`` is unknown.
        """
        mapping = {"basic": "simple", "advanced": "medium", "full": "advanced"}
        if pipeline_type not in mapping:
            raise ValueError(f"pipeline_type must be one of {sorted(mapping)}, got '{pipeline_type}'.")
        return self.create_adaptive_pipeline(
            X,
            y,
            complexity_level=mapping[pipeline_type],
            include_feature_engineering=pipeline_type != "basic",
            include_outlier_removal=pipeline_type != "basic",
        )

    def auto_create_pipeline(
        self,
        X: ArrayLike,
        y: Any,
        task_type: str = "auto",
        model_type: Optional[str] = None,
        complexity_level: str = "medium",
        model_params: Optional[Dict[str, Any]] = None,
    ) -> Pipeline:
        """Create a complete pipeline whose preprocessing is derived from the data.

        Missing values, categorical / datetime / text columns and outliers are
        detected automatically and handled with dedicated steps.

        Args:
            X: Feature matrix.
            y: Target vector.
            task_type: ``'auto'``, ``'classification'`` or ``'regression'``.
            model_type: Model name; defaults to a random forest.
            complexity_level: ``'simple'``, ``'medium'`` or ``'advanced'``.
            model_params: Model hyper-parameters.

        Returns:
            Unfitted pipeline with a final estimator step.
        """
        task = infer_task_type(y) if task_type == "auto" else task_type
        profile = profile_data(X, y)
        steps = self._adaptive_steps(profile, task, complexity_level, True, True, False)
        model_name = model_type or "random_forest"
        estimator = _make_model(model_name, task, self.random_state, self.n_jobs, **(model_params or {}))
        if (
            task == CLASSIFICATION
            and profile["is_imbalanced"]
            and "class_weight" in estimator.get_params()
            and "class_weight" not in (model_params or {})
        ):
            estimator.set_params(class_weight="balanced")
        steps.append((_estimator_step_name(task), estimator))
        return self._finalize(steps, "auto")

    def create_pipeline_with_auto_tuning(
        self,
        algorithm: str,
        task_type: str,
        preprocessing_level: str = "standard",
        param_grid: Optional[Dict[str, Any]] = None,
        cv: Optional[Union[int, Any]] = None,
        scoring: Optional[str] = None,
        search: str = "grid",
        n_iter: int = 20,
        n_jobs: Optional[int] = None,
    ) -> Union[GridSearchCV, RandomizedSearchCV]:
        """Create a pipeline wrapped in a cross-validated hyper-parameter search.

        The returned search object exposes ``fit``/``predict``/``score`` and,
        after fitting, ``best_estimator_``, ``best_params_`` and ``best_score_``.

        Args:
            algorithm: Registered model name.
            task_type: ``'classification'`` or ``'regression'``.
            preprocessing_level: See :meth:`create_classification_pipeline`.
            param_grid: Grid keyed by ``'classifier__...'``/``'regressor__...'``;
                a default grid is used when ``None``.
            cv: Folds (int) or splitter.
            scoring: Scoring name; defaults to accuracy / r2.
            search: ``'grid'`` or ``'random'``.
            n_iter: Candidates for randomised search.
            n_jobs: Parallelism of the search.

        Returns:
            Unfitted ``GridSearchCV`` or ``RandomizedSearchCV``.

        Raises:
            ValueError: If no parameter grid is available for ``algorithm``.
        """
        if task_type == CLASSIFICATION:
            pipeline = self.create_classification_pipeline(algorithm, preprocessing_level=preprocessing_level)
        elif task_type == REGRESSION:
            pipeline = self.create_regression_pipeline(algorithm, preprocessing_level=preprocessing_level)
        else:
            raise ValueError(f"task_type must be 'classification' or 'regression', got '{task_type}'.")
        grid = (
            param_grid
            if param_grid is not None
            else _default_param_grid(algorithm, task_type, _estimator_step_name(task_type))
        )
        if not grid:
            raise ValueError(f"No default parameter grid for '{algorithm}'; pass param_grid explicitly.")
        splitter = _cv_splitter(cv, task_type, self.random_state)
        scoring = scoring or _default_scoring(task_type)
        n_jobs = self.n_jobs if n_jobs is None else n_jobs
        if search == "random":
            return RandomizedSearchCV(
                pipeline,
                grid,
                n_iter=n_iter,
                cv=splitter,
                scoring=scoring,
                n_jobs=n_jobs,
                random_state=self.random_state,
                refit=True,
            )
        if search != "grid":
            raise ValueError(f"search must be 'grid' or 'random', got '{search}'.")
        return GridSearchCV(pipeline, grid, cv=splitter, scoring=scoring, n_jobs=n_jobs, refit=True)

    def create_production_pipeline(
        self,
        algorithm: str = "random_forest",
        task_type: str = CLASSIFICATION,
        *,
        preprocessing_level: str = "standard",
        model_params: Optional[Dict[str, Any]] = None,
        enable_monitoring: bool = True,
        cache_transformations: bool = True,
        cache_dir: Optional[str] = None,
        parallel_preprocessing: bool = True,
    ) -> Pipeline:
        """Create a pipeline configured for production use.

        Args:
            algorithm: Registered model name.
            task_type: ``'classification'`` or ``'regression'``.
            preprocessing_level: See :meth:`create_classification_pipeline`.
            model_params: Model hyper-parameters.
            enable_monitoring: Log each fitted step (``verbose=True``) and add a
                shape-logging step before the estimator.
            cache_transformations: Cache fitted transformers with ``joblib``.
            cache_dir: Cache directory (defaults to ``settings.CACHE_DIR`` or a
                temporary directory).
            parallel_preprocessing: Use all cores for estimators that support it.

        Returns:
            Unfitted pipeline.
        """
        n_jobs = -1 if parallel_preprocessing else None
        if task_type == CLASSIFICATION:
            pipeline = self.create_classification_pipeline(
                algorithm, preprocessing_level=preprocessing_level, model_params=model_params, n_jobs=n_jobs
            )
        elif task_type == REGRESSION:
            pipeline = self.create_regression_pipeline(
                algorithm, preprocessing_level=preprocessing_level, model_params=model_params, n_jobs=n_jobs
            )
        else:
            raise ValueError(f"task_type must be 'classification' or 'regression', got '{task_type}'.")
        steps = list(pipeline.steps)
        if enable_monitoring:
            steps.insert(len(steps) - 1, ("monitor", _ShapeLogger("pre-estimator")))
        memory = None
        if cache_transformations:
            import tempfile

            memory = cache_dir or str(
                getattr(settings, "CACHE_DIR", None) or tempfile.mkdtemp(prefix="sklearn_mastery_cache_")
            )
        return _assemble_pipeline(steps, memory=memory, verbose=enable_monitoring)

    def profile_data(self, X: ArrayLike, y: Optional[Any] = None) -> Dict[str, Any]:
        """Profile a dataset (see :func:`profile_data`)."""
        return profile_data(X, y)


# --------------------------------------------------------------------------- #
# AutoMLPipelineBuilder
# --------------------------------------------------------------------------- #


class AutoMLPipelineBuilder(LoggerMixin):
    """Profile data and recommend preprocessing, feature engineering and models.

    Args:
        task_type: ``'auto'`` (inferred from ``y``), ``'classification'`` or
            ``'regression'``.
        random_state: Seed for all randomised components.
        n_jobs: Parallelism for estimators that support it.
    """

    def __init__(
        self, task_type: str = "auto", random_state: Optional[int] = None, n_jobs: Optional[int] = None
    ):
        if task_type not in ("auto", CLASSIFICATION, REGRESSION):
            raise ValueError(
                f"task_type must be 'auto', 'classification' or 'regression', got '{task_type}'."
            )
        self.task_type = task_type
        self.random_state = settings.RANDOM_SEED if random_state is None else random_state
        self.n_jobs = n_jobs
        self.factory = PipelineFactory(random_state=self.random_state, n_jobs=n_jobs)

    def _task(self, y: Any, profile: Optional[Dict[str, Any]] = None) -> str:
        if self.task_type != "auto":
            return self.task_type
        if profile is not None and profile.get("task_type"):
            return profile["task_type"]
        return infer_task_type(y)

    def profile_data(self, X: ArrayLike, y: Optional[Any] = None) -> Dict[str, Any]:
        """Profile the dataset (see :func:`profile_data`)."""
        return profile_data(X, y)

    def recommend_preprocessing(self, X: ArrayLike, y: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Recommend ordered preprocessing steps.

        Each recommendation is a dict with ``name`` (a registry step name usable
        with :class:`CustomPipelineBuilder`), ``reason``, ``params`` and
        ``priority`` (1 = apply first).

        Args:
            X: Feature matrix.
            y: Optional target.

        Returns:
            List of recommendation dictionaries in pipeline order.
        """
        p = profile_data(X, y)
        recs: List[Dict[str, Any]] = []
        if p["datetime_columns"]:
            recs.append(
                {
                    "name": "datetime_features",
                    "reason": f"datetime columns {p['datetime_columns']}",
                    "params": {},
                }
            )
        if p["has_missing"]:
            recs.append(
                {
                    "name": "missing_value_imputer",
                    "reason": f"{p['missing_percentage']:.1f}% missing values in {sorted(p['missing_by_column'])}",
                    "params": {"numeric_strategy": "median", "categorical_strategy": "most_frequent"},
                }
            )
        if p["has_outliers"]:
            recs.append(
                {
                    "name": "outlier_clipper",
                    "reason": f"extreme values in {p['outlier_columns']}",
                    "params": {"method": "iqr"},
                }
            )
        if p["categorical_columns"] or p["text_columns"] or p["boolean_columns"] or p["datetime_columns"]:
            params: Dict[str, Any] = {}
            if p["text_columns"]:
                params["text_columns"] = list(p["text_columns"])
            if p["high_cardinality_columns"]:
                params["max_categories"] = 50
            recs.append(
                {
                    "name": "onehot_encoder",
                    "reason": f"non-numeric columns {p['categorical_columns'] + p['text_columns'] + p['boolean_columns']}",
                    "params": params,
                }
            )
        if p["skewed_columns"] and not (p["categorical_columns"] or p["text_columns"]):
            recs.append(
                {"name": "power_transform", "reason": f"skewed columns {p['skewed_columns']}", "params": {}}
            )
        recs.append(
            {
                "name": "standard_scaler",
                "reason": "standardise features for scale-sensitive models",
                "params": {},
            }
        )
        if p["dimensionality"] == "high":
            recs.append(
                {
                    "name": "feature_selection",
                    "reason": f"{p['n_features']} features (high-dimensional)",
                    "params": {"percentile": 50},
                }
            )
        for i, rec in enumerate(recs, start=1):
            rec["priority"] = i
        return recs

    def recommend_models(self, X: ArrayLike, y: Any) -> List[Dict[str, Any]]:
        """Recommend candidate models ranked by a heuristic suitability score.

        Args:
            X: Feature matrix.
            y: Target vector.

        Returns:
            List of dicts with ``name``, ``score`` (0-1), ``priority`` (1 = best)
            and ``reason``, sorted best first.
        """
        p = profile_data(X, y)
        task = self._task(y, p)
        n = p["n_samples"]
        small, large = n < 5000, n >= 50000
        if task == CLASSIFICATION:
            candidates = {
                "random_forest": (0.90, "robust default for tabular data"),
                "gradient_boosting": (0.85 if not large else 0.70, "strong on structured data"),
                "logistic_regression": (
                    0.80 if p["n_classes"] == 2 else 0.72,
                    "fast, interpretable linear baseline",
                ),
                "svm": (0.70 if small else 0.30, "kernel model, best on small datasets"),
                "knn": (0.60 if small else 0.35, "non-parametric baseline"),
                "naive_bayes": (0.50, "very fast probabilistic baseline"),
                "mlp": (0.65 if n >= 1000 else 0.40, "neural network, needs more data"),
            }
            if p["is_imbalanced"]:
                candidates["random_forest"] = (0.93, "robust default; supports class weights for imbalance")
        else:
            candidates = {
                "random_forest": (0.90, "robust default for tabular data"),
                "gradient_boosting": (0.85 if not large else 0.70, "strong on structured data"),
                "ridge": (0.80, "regularised linear baseline"),
                "linear_regression": (0.75, "interpretable linear baseline"),
                "lasso": (0.70, "sparse linear model for feature selection"),
                "svr": (0.60 if small else 0.30, "kernel model, best on small datasets"),
                "knn": (0.55 if small else 0.30, "non-parametric baseline"),
                "mlp": (0.65 if n >= 1000 else 0.40, "neural network, needs more data"),
            }
        if HAS_XGBOOST:
            candidates["xgboost"] = (0.88, "gradient boosting, strong general performer")
        if HAS_LIGHTGBM:
            candidates["lightgbm"] = (0.88 if large else 0.84, "fast gradient boosting")
        ranked = sorted(candidates.items(), key=lambda kv: (-kv[1][0], kv[0]))
        return [
            {"name": name, "score": float(score), "priority": i, "reason": reason, "task_type": task}
            for i, (name, (score, reason)) in enumerate(ranked, start=1)
        ]

    def recommend_feature_engineering(self, X: ArrayLike, y: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Recommend feature-engineering transformations.

        Args:
            X: Feature matrix.
            y: Optional target.

        Returns:
            List of dicts with ``name``, ``reason``, ``params`` and ``columns``.
        """
        p = profile_data(X, y)
        recs: List[Dict[str, Any]] = []
        if p["datetime_columns"]:
            recs.append(
                {
                    "name": "datetime_feature_extraction",
                    "reason": "expand datetime columns into calendar components",
                    "params": {},
                    "columns": p["datetime_columns"],
                }
            )
        if p["text_columns"]:
            recs.append(
                {
                    "name": "text_vectorization",
                    "reason": "TF-IDF features for free-text columns",
                    "params": {"vectorizer": "tfidf"},
                    "columns": p["text_columns"],
                }
            )
        if p["skewed_columns"]:
            recs.append(
                {
                    "name": "log_transform",
                    "reason": "reduce skewness",
                    "params": {},
                    "columns": p["skewed_columns"],
                }
            )
        if p["high_cardinality_columns"]:
            recs.append(
                {
                    "name": "frequency_encoding",
                    "reason": "high-cardinality categoricals",
                    "params": {},
                    "columns": p["high_cardinality_columns"],
                }
            )
        n_num = len(p["numeric_columns"])
        if 2 <= n_num <= 20 and p["n_samples"] >= 100:
            recs.append(
                {
                    "name": "polynomial_features",
                    "reason": "few numeric features; interactions may add signal",
                    "params": {"degree": 2, "interaction_only": True},
                    "columns": p["numeric_columns"],
                }
            )
        if p["dimensionality"] == "high":
            recs.append(
                {
                    "name": "pca",
                    "reason": "many features; reduce dimensionality",
                    "params": {"n_components": 0.95},
                    "columns": p["numeric_columns"],
                }
            )
        if p["categorical_columns"]:
            recs.append(
                {
                    "name": "onehot_encoder",
                    "reason": "encode categorical columns",
                    "params": {},
                    "columns": p["categorical_columns"],
                }
            )
        return recs

    def build_pipeline(
        self,
        X: ArrayLike,
        y: Any,
        max_time_mins: float = 5.0,
        task_type: Optional[str] = None,
        model: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> Pipeline:
        """Build a complete pipeline from the recommendations.

        Args:
            X: Feature matrix.
            y: Target vector.
            max_time_mins: Training-time budget; below one minute a fast linear
                model is chosen, otherwise the top recommended model.
            task_type: Override the task type.
            model: Override the model name.
            model_params: Model hyper-parameters.

        Returns:
            Unfitted pipeline.
        """
        p = profile_data(X, y)
        task = task_type or self._task(y, p)
        steps = [
            _make_preprocessing_step(
                rec["name"], task_type=task, random_state=self.random_state, **rec.get("params", {})
            )
            for rec in self.recommend_preprocessing(X, y)
        ]
        if model is None:
            if max_time_mins < 1:
                model = "logistic_regression" if task == CLASSIFICATION else "ridge"
            else:
                model = self.recommend_models(X, y)[0]["name"]
        estimator = _make_model(model, task, self.random_state, self.n_jobs, **(model_params or {}))
        if (
            task == CLASSIFICATION
            and p["is_imbalanced"]
            and "class_weight" in estimator.get_params()
            and "class_weight" not in (model_params or {})
        ):
            estimator.set_params(class_weight="balanced")
        steps.append((_estimator_step_name(task), estimator))
        self.logger.info("AutoML pipeline: %s -> %s", [s for s, _ in steps[:-1]], model)
        return _assemble_pipeline(steps)

    def build(self, X: ArrayLike, y: Any, **kwargs: Any) -> Pipeline:
        """Alias of :meth:`build_pipeline` (kept for backwards compatibility)."""
        return self.build_pipeline(X, y, **kwargs)

    def suggest_hyperparameter_tuning(self, pipeline: Pipeline, X: ArrayLike, y: Any) -> Dict[str, Any]:
        """Suggest a hyper-parameter search for the pipeline's final estimator.

        Args:
            pipeline: Pipeline whose last step is the estimator.
            X: Feature matrix (used for sizing the search).
            y: Target vector.

        Returns:
            Dict with ``param_grid``, ``param_distributions``, ``search_method``,
            ``n_iter``, ``cv``, ``scoring``, ``estimator_step`` and ``model_name``.
        """
        step_name, estimator = pipeline.steps[-1]
        task = self._task(y)
        model_name = _model_name_from_estimator(estimator)
        grid = _default_param_grid(model_name, task, step_name)
        n_candidates = int(np.prod([len(v) for v in grid.values()])) if grid else 0
        n_samples = X.shape[0]
        return {
            "param_grid": grid,
            "param_distributions": grid,
            "search_method": "grid" if n_candidates <= 30 else "random",
            "n_iter": min(30, n_candidates) if n_candidates else 0,
            "cv": 5 if n_samples >= 100 else 3,
            "scoring": _default_scoring(task),
            "estimator_step": step_name,
            "model_name": model_name,
            "n_candidates": n_candidates,
        }


# --------------------------------------------------------------------------- #
# Task-specific factories
# --------------------------------------------------------------------------- #


class ClassificationPipelineFactory(LoggerMixin):
    """Convenience factory for common classification scenarios.

    Args:
        random_state: Seed for randomised components.
        preprocessing_level: Default preprocessing level (see
            :meth:`PipelineFactory.create_classification_pipeline`).
        n_jobs: Parallelism for estimators that support it.
    """

    def __init__(
        self,
        random_state: Optional[int] = None,
        preprocessing_level: str = "standard",
        n_jobs: Optional[int] = None,
    ):
        self.random_state = settings.RANDOM_SEED if random_state is None else random_state
        self.preprocessing_level = preprocessing_level
        self.n_jobs = n_jobs
        self.factory = PipelineFactory(random_state=self.random_state, n_jobs=n_jobs)

    def _base_steps(self, preprocessing_level: Optional[str] = None) -> List[Step]:
        pipeline = self.factory.create_classification_pipeline(
            "logistic_regression", preprocessing_level=preprocessing_level or self.preprocessing_level
        )
        return list(pipeline.steps[:-1])

    def create_standard_pipeline(self, algorithm: str = "random_forest", **model_params: Any) -> Pipeline:
        """Standard-level classification pipeline."""
        return self.factory.create_classification_pipeline(
            algorithm, preprocessing_level="standard", model_params=model_params
        )

    def create_advanced_pipeline(self, algorithm: str = "gradient_boosting", **model_params: Any) -> Pipeline:
        """Advanced-level classification pipeline."""
        return self.factory.create_classification_pipeline(
            algorithm, preprocessing_level="advanced", model_params=model_params
        )

    def create_binary_classification_pipeline(
        self, algorithm: str = "logistic_regression", **model_params: Any
    ) -> Pipeline:
        """Binary classification pipeline with probability outputs.

        Args:
            algorithm: Registered classifier name.
            **model_params: Classifier hyper-parameters.

        Returns:
            Unfitted pipeline supporting ``predict_proba``.
        """
        clf = _make_classifier(algorithm, self.random_state, self.n_jobs, **model_params)
        clf = _with_probabilities(clf)
        steps = self._base_steps() + [("classifier", clf)]
        return _assemble_pipeline(steps)

    def create_multiclass_pipeline(
        self, algorithm: str = "random_forest", strategy: str = "auto", **model_params: Any
    ) -> Pipeline:
        """Multiclass classification pipeline.

        Args:
            algorithm: Registered classifier name.
            strategy: ``'auto'``/``'native'`` (estimator handles multiclass),
                ``'ovr'`` (one-vs-rest) or ``'ovo'`` (one-vs-one).
            **model_params: Classifier hyper-parameters.

        Returns:
            Unfitted pipeline.

        Raises:
            ValueError: If ``strategy`` is unknown.
        """
        clf = _make_classifier(algorithm, self.random_state, self.n_jobs, **model_params)
        if strategy == "ovr":
            clf = OneVsRestClassifier(clf, n_jobs=self.n_jobs)
        elif strategy == "ovo":
            clf = OneVsOneClassifier(clf, n_jobs=self.n_jobs)
        elif strategy not in ("auto", "native"):
            raise ValueError(f"strategy must be 'auto', 'native', 'ovr' or 'ovo', got '{strategy}'.")
        return _assemble_pipeline(self._base_steps() + [("classifier", clf)])

    def create_imbalanced_pipeline(
        self,
        algorithm: str = "random_forest",
        balancing_strategy: str = "smote",
        sampling_strategy: Any = "auto",
        **model_params: Any,
    ) -> Pipeline:
        """Pipeline for imbalanced classes.

        With ``imbalanced-learn`` installed a resampler (``'smote'``,
        ``'oversample'`` or ``'undersample'``) is inserted before the classifier;
        otherwise (or with ``'class_weight'``) ``class_weight='balanced'`` is
        used when the estimator supports it.

        Args:
            algorithm: Registered classifier name.
            balancing_strategy: Resampling / weighting strategy.
            sampling_strategy: ``imbalanced-learn`` ``sampling_strategy``.
            **model_params: Classifier hyper-parameters.

        Returns:
            Unfitted (``imblearn`` or sklearn) pipeline.
        """
        clf = _make_classifier(algorithm, self.random_state, self.n_jobs, **model_params)
        steps = self._base_steps()
        key = _normalize_name(balancing_strategy)
        if key not in ("class_weight", "balanced", "none") and HAS_IMBLEARN:
            steps.append(
                (
                    "sampler",
                    _make_sampler(key, random_state=self.random_state, sampling_strategy=sampling_strategy),
                )
            )
        else:
            if key not in ("class_weight", "balanced", "none"):
                self.logger.warning(
                    "imbalanced-learn not installed; using class_weight='balanced' instead of '%s'.",
                    balancing_strategy,
                )
            if "class_weight" in clf.get_params():
                clf.set_params(class_weight="balanced")
            else:
                self.logger.warning(
                    "%s does not support class_weight; no balancing applied.", type(clf).__name__
                )
        steps.append(("classifier", clf))
        return _assemble_pipeline(steps)

    def create_ensemble_pipeline(
        self,
        base_models: Sequence[str] = ("logistic_regression", "random_forest", "gradient_boosting"),
        ensemble_method: str = "voting",
        voting: str = "soft",
        final_estimator: Optional[BaseEstimator] = None,
        model_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Pipeline:
        """Ensemble classification pipeline.

        Args:
            base_models: Registered classifier names.
            ensemble_method: ``'voting'``, ``'stacking'`` or ``'bagging'`` (bags
                the first base model).
            voting: ``'soft'`` or ``'hard'`` for voting ensembles.
            final_estimator: Meta-learner for stacking (default logistic regression).
            model_params: Per-model hyper-parameters keyed by model name.

        Returns:
            Unfitted pipeline supporting ``predict_proba`` (soft voting / stacking).

        Raises:
            ValueError: If ``base_models`` is empty or the method is unknown.
        """
        if not base_models:
            raise ValueError("base_models must contain at least one model name.")
        params = model_params or {}
        estimators: List[Tuple[str, BaseEstimator]] = []
        for name in base_models:
            est = _make_classifier(name, self.random_state, self.n_jobs, **params.get(name, {}))
            est = _with_probabilities(est)
            estimators.append(
                (
                    _unique_step_name(_normalize_name(name, _CLASSIFIER_ALIASES), [n for n, _ in estimators]),
                    est,
                )
            )
        if ensemble_method == "voting":
            if voting == "soft" and not all(hasattr(est, "predict_proba") for _, est in estimators):
                voting = "hard"
            ensemble: BaseEstimator = VotingClassifier(estimators, voting=voting, n_jobs=self.n_jobs)
        elif ensemble_method == "stacking":
            ensemble = StackingClassifier(
                estimators,
                final_estimator=final_estimator
                or LogisticRegression(max_iter=1000, random_state=self.random_state),
                cv=_cv_splitter(settings.DEFAULT_CV_FOLDS, CLASSIFICATION, self.random_state),
                n_jobs=self.n_jobs,
            )
        elif ensemble_method == "bagging":
            ensemble = BaggingClassifier(
                estimators[0][1], n_estimators=10, random_state=self.random_state, n_jobs=self.n_jobs
            )
        else:
            raise ValueError(
                f"ensemble_method must be 'voting', 'stacking' or 'bagging', got '{ensemble_method}'."
            )
        return _assemble_pipeline(self._base_steps() + [("classifier", ensemble)])

    def create_text_classification_pipeline(
        self,
        text_column: Optional[str] = "text",
        vectorizer: str = "tfidf",
        algorithm: str = "logistic_regression",
        numeric_columns: Optional[Sequence[str]] = None,
        vectorizer_params: Optional[Dict[str, Any]] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> Pipeline:
        """Text classification pipeline.

        Args:
            text_column: DataFrame column holding the documents. ``None`` means
                the input itself is a sequence of documents.
            vectorizer: ``'tfidf'``, ``'count'`` or ``'hashing'``.
            algorithm: Registered classifier name.
            numeric_columns: Extra numeric columns to scale and concatenate.
            vectorizer_params: Vectoriser keyword arguments.
            model_params: Classifier hyper-parameters.

        Returns:
            Unfitted pipeline.

        Raises:
            ValueError: If ``vectorizer`` is unknown.
        """
        vectorizers = {"tfidf": TfidfVectorizer, "count": CountVectorizer, "hashing": HashingVectorizer}
        key = _normalize_name(vectorizer)
        if key not in vectorizers:
            raise ValueError(f"vectorizer must be one of {sorted(vectorizers)}, got '{vectorizer}'.")
        vec = vectorizers[key](**(vectorizer_params or {}))
        clf = _make_classifier(algorithm, self.random_state, self.n_jobs, **(model_params or {}))
        if text_column is None:
            steps: List[Step] = [("vectorizer", vec)]
        else:
            transformers: List[Tuple[str, Any, Any]] = [("text", vec, text_column)]
            if numeric_columns:
                transformers.append(
                    (
                        "numeric",
                        Pipeline(
                            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
                        ),
                        list(numeric_columns),
                    )
                )
            steps = [("features", ColumnTransformer(transformers, remainder="drop"))]
        steps.append(("classifier", clf))
        return _assemble_pipeline(steps)


class RegressionPipelineFactory(LoggerMixin):
    """Convenience factory for common regression scenarios.

    Args:
        random_state: Seed for randomised components.
        preprocessing_level: Default preprocessing level.
        n_jobs: Parallelism for estimators that support it.
    """

    def __init__(
        self,
        random_state: Optional[int] = None,
        preprocessing_level: str = "standard",
        n_jobs: Optional[int] = None,
    ):
        self.random_state = settings.RANDOM_SEED if random_state is None else random_state
        self.preprocessing_level = preprocessing_level
        self.n_jobs = n_jobs
        self.factory = PipelineFactory(random_state=self.random_state, n_jobs=n_jobs)

    def _base_steps(self, preprocessing_level: Optional[str] = None) -> List[Step]:
        pipeline = self.factory.create_regression_pipeline(
            "ridge", preprocessing_level=preprocessing_level or self.preprocessing_level
        )
        return list(pipeline.steps[:-1])

    def create_standard_pipeline(self, algorithm: str = "random_forest", **model_params: Any) -> Pipeline:
        """Standard-level regression pipeline."""
        return self.factory.create_regression_pipeline(
            algorithm, preprocessing_level="standard", model_params=model_params
        )

    def create_advanced_pipeline(self, algorithm: str = "gradient_boosting", **model_params: Any) -> Pipeline:
        """Advanced-level regression pipeline."""
        return self.factory.create_regression_pipeline(
            algorithm, preprocessing_level="advanced", model_params=model_params
        )

    def create_linear_regression_pipeline(
        self, regularization: Optional[str] = None, alpha: float = 1.0, **model_params: Any
    ) -> Pipeline:
        """Linear regression pipeline with optional regularisation.

        Args:
            regularization: ``None``/``'none'`` (OLS), ``'ridge'``, ``'lasso'``
                or ``'elastic_net'``.
            alpha: Regularisation strength.
            **model_params: Extra estimator hyper-parameters.

        Returns:
            Unfitted pipeline.

        Raises:
            ValueError: If ``regularization`` is unknown.
        """
        key = _normalize_name(
            regularization or "none", {"l2": "ridge", "l1": "lasso", "elasticnet": "elastic_net"}
        )
        if key == "none":
            reg = _make_regressor("linear_regression", self.random_state, self.n_jobs, **model_params)
        elif key in ("ridge", "lasso", "elastic_net"):
            reg = _make_regressor(key, self.random_state, self.n_jobs, alpha=alpha, **model_params)
        else:
            raise ValueError(
                f"regularization must be None, 'ridge', 'lasso' or 'elastic_net', got '{regularization}'."
            )
        return _assemble_pipeline(self._base_steps() + [("regressor", reg)])

    def create_nonlinear_regression_pipeline(
        self, algorithm: str = "random_forest", **model_params: Any
    ) -> Pipeline:
        """Non-linear regression pipeline (tree ensembles, SVR, kNN, MLP, ...).

        Args:
            algorithm: Registered regressor name.
            **model_params: Estimator hyper-parameters.

        Returns:
            Unfitted pipeline.
        """
        reg = _make_regressor(algorithm, self.random_state, self.n_jobs, **model_params)
        return _assemble_pipeline(self._base_steps() + [("regressor", reg)])

    def create_polynomial_regression_pipeline(
        self,
        degree: int = 2,
        include_interactions: bool = True,
        regularization: Optional[str] = "ridge",
        alpha: float = 1.0,
    ) -> Pipeline:
        """Polynomial regression pipeline.

        Args:
            degree: Polynomial degree.
            include_interactions: Include cross terms; when ``False`` only pure
                powers ``x, x**2, ...`` are generated.
            regularization: Regulariser for the linear model (see
                :meth:`create_linear_regression_pipeline`).
            alpha: Regularisation strength.

        Returns:
            Unfitted pipeline.
        """
        if degree < 1:
            raise ValueError("degree must be >= 1.")
        if include_interactions:
            poly: BaseEstimator = PolynomialFeatures(degree=degree, include_bias=False)
        else:
            poly = FunctionTransformer(_pure_powers, kw_args={"degree": degree})
        linear = self.create_linear_regression_pipeline(regularization, alpha=alpha)
        steps = self._base_steps() + [("polynomial_features", poly), ("regressor", linear.steps[-1][1])]
        return _assemble_pipeline(steps)

    def create_ensemble_regression_pipeline(
        self,
        base_models: Sequence[str] = ("linear_regression", "random_forest", "gradient_boosting"),
        ensemble_method: str = "stacking",
        final_estimator: Optional[BaseEstimator] = None,
        model_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Pipeline:
        """Ensemble regression pipeline.

        Args:
            base_models: Registered regressor names.
            ensemble_method: ``'stacking'``, ``'voting'`` or ``'bagging'``.
            final_estimator: Meta-learner for stacking (default ``RidgeCV``).
            model_params: Per-model hyper-parameters keyed by model name.

        Returns:
            Unfitted pipeline.

        Raises:
            ValueError: If ``base_models`` is empty or the method is unknown.
        """
        if not base_models:
            raise ValueError("base_models must contain at least one model name.")
        params = model_params or {}
        estimators: List[Tuple[str, BaseEstimator]] = []
        for name in base_models:
            est = _make_regressor(name, self.random_state, self.n_jobs, **params.get(name, {}))
            estimators.append(
                (
                    _unique_step_name(_normalize_name(name, _REGRESSOR_ALIASES), [n for n, _ in estimators]),
                    est,
                )
            )
        if ensemble_method == "stacking":
            ensemble: BaseEstimator = StackingRegressor(
                estimators,
                final_estimator=final_estimator or RidgeCV(),
                cv=_cv_splitter(settings.DEFAULT_CV_FOLDS, REGRESSION, self.random_state),
                n_jobs=self.n_jobs,
            )
        elif ensemble_method == "voting":
            ensemble = VotingRegressor(estimators, n_jobs=self.n_jobs)
        elif ensemble_method == "bagging":
            ensemble = BaggingRegressor(
                estimators[0][1], n_estimators=10, random_state=self.random_state, n_jobs=self.n_jobs
            )
        else:
            raise ValueError(
                f"ensemble_method must be 'stacking', 'voting' or 'bagging', got '{ensemble_method}'."
            )
        return _assemble_pipeline(self._base_steps() + [("regressor", ensemble)])


# --------------------------------------------------------------------------- #
# CustomPipelineBuilder
# --------------------------------------------------------------------------- #


class CustomPipelineBuilder(LoggerMixin):
    """Fluent, step-by-step pipeline builder.

    Every ``add_*`` method returns ``self`` so calls can be chained; the
    resulting pipeline is produced by :meth:`build`.

    Args:
        task_type: ``'classification'`` or ``'regression'``; used to resolve
            ambiguous model names (e.g. ``'random_forest'``) and feature
            selectors.
        random_state: Seed for randomised components.
        n_jobs: Parallelism for estimators that support it.
    """

    def __init__(
        self,
        task_type: str = CLASSIFICATION,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ):
        if task_type not in (CLASSIFICATION, REGRESSION):
            raise ValueError(f"task_type must be 'classification' or 'regression', got '{task_type}'.")
        self.task_type = task_type
        self.random_state = settings.RANDOM_SEED if random_state is None else random_state
        self.n_jobs = n_jobs
        self.steps: List[Step] = []

    # ---- helpers ---------------------------------------------------------
    def _append(self, name: str, estimator: BaseEstimator) -> CustomPipelineBuilder:
        self.steps.append((_unique_step_name(name, [s for s, _ in self.steps]), estimator))
        return self

    def _registry_step(self, name: str, **params: Any) -> Step:
        return _make_preprocessing_step(
            name, task_type=self.task_type, random_state=self.random_state, **params
        )

    @staticmethod
    def _selector_for_branch(branch_name: str, columns: Any = None) -> Any:
        from sklearn.compose import make_column_selector

        if columns is not None:
            return list(columns) if not callable(columns) else columns
        key = branch_name.lower()
        if "numeric" in key or "number" in key or "num" in key:
            return make_column_selector(dtype_include=np.number)
        if "categor" in key or "cat" in key or "text" in key or "object" in key:
            return make_column_selector(dtype_include=["object", "category", "bool", "string"])
        if "date" in key or "time" in key:
            return make_column_selector(dtype_include=["datetime", "datetimetz"])
        raise ValueError(
            f"Cannot infer the columns for branch '{branch_name}'; include 'numeric', 'categorical' or 'datetime' in the name or pass 'columns'."
        )

    # ---- public API ------------------------------------------------------
    def add_preprocessing_step(self, name: str, **params: Any) -> CustomPipelineBuilder:
        """Add a registry preprocessing step (see :func:`available_preprocessing_steps`)."""
        return self._append(*self._registry_step(name, **params))

    def add_feature_engineering_step(self, name: str, **params: Any) -> CustomPipelineBuilder:
        """Add a feature-engineering step (``polynomial_features``, ``interaction_features``, ``pca``, ...)."""
        return self._append(*self._registry_step(name, **params))

    def add_feature_selection_step(self, method: str = "univariate", **params: Any) -> CustomPipelineBuilder:
        """Add a feature selector (``univariate``, ``mutual_info``, ``rfe``, ``model_based``, ``variance``)."""
        selector = _make_feature_selector(method, self.task_type, random_state=self.random_state, **params)
        return self._append("feature_selection", selector)

    def add_model_step(self, name: str, **params: Any) -> CustomPipelineBuilder:
        """Add the final estimator (registered model name)."""
        task = _resolve_model_task(name, self.task_type if _model_is_known(name, self.task_type) else None)
        estimator = _make_model(name, task, self.random_state, self.n_jobs, **params)
        return self._append(_estimator_step_name(task), estimator)

    def add_custom_step(self, name: str, transformer: BaseEstimator) -> CustomPipelineBuilder:
        """Add an arbitrary transformer / estimator instance under ``name``."""
        if not hasattr(transformer, "fit"):
            raise ValueError(f"Step '{name}' must be an estimator with a 'fit' method.")
        return self._append(name, transformer)

    def add_step(self, name: str, transformer: BaseEstimator) -> CustomPipelineBuilder:
        """Alias of :meth:`add_custom_step` (kept for backwards compatibility)."""
        return self.add_custom_step(name, transformer)

    def add_conditional_step(
        self,
        name: str,
        condition: Callable[[ArrayLike, Any], bool],
        step_config: Optional[Dict[str, Any]] = None,
        transformer: Optional[BaseEstimator] = None,
    ) -> CustomPipelineBuilder:
        """Add a step applied only when ``condition(X, y)`` holds at fit time.

        Args:
            name: Registry step name (or label when ``transformer`` is given).
            condition: Callable ``(X, y) -> bool``.
            step_config: Keyword arguments for the registry step.
            transformer: Explicit transformer instead of a registry step.

        Returns:
            ``self``.
        """
        if transformer is None:
            step_name, transformer = self._registry_step(name, **(step_config or {}))
        else:
            step_name = name
        return self._append(step_name, _ConditionalTransformer(transformer, condition))

    def add_parallel_steps(
        self, steps: Sequence[Tuple[str, Union[str, BaseEstimator]]], name: str = "parallel_features"
    ) -> CustomPipelineBuilder:
        """Add several transformers applied in parallel (``FeatureUnion``).

        Args:
            steps: ``(label, registry_name_or_transformer)`` pairs.
            name: Name of the union step.

        Returns:
            ``self``.
        """
        if not steps:
            raise ValueError("steps must not be empty.")
        transformers = []
        for label, spec in steps:
            est = self._registry_step(spec)[1] if isinstance(spec, str) else spec
            transformers.append((label, est))
        return self._append(name, FeatureUnion(transformers, n_jobs=self.n_jobs))

    def add_branched_processing(
        self,
        branches: Dict[str, Union[Sequence[Union[str, BaseEstimator]], Dict[str, Any]]],
        name: str = "branched_processing",
        remainder: str = "drop",
    ) -> CustomPipelineBuilder:
        """Add a ``ColumnTransformer`` with one sub-pipeline per column group.

        Args:
            branches: Mapping ``branch_name -> [step names]`` or
                ``branch_name -> {'steps': [...], 'columns': [...]}``. The
                columns are inferred from the branch name (``numeric``,
                ``categorical``/``text``, ``datetime``) when not given.
            name: Name of the column-transformer step.
            remainder: ``ColumnTransformer`` remainder handling.

        Returns:
            ``self``.
        """
        if not branches:
            raise ValueError("branches must not be empty.")
        transformers = []
        for branch_name, spec in branches.items():
            columns = None
            if isinstance(spec, dict):
                columns = spec.get("columns")
                spec = spec.get("steps", [])
            sub_steps: List[Step] = []
            for item in spec:
                s_name, est = (
                    self._registry_step(item)
                    if isinstance(item, str)
                    else (type(item).__name__.lower(), item)
                )
                sub_steps.append((_unique_step_name(s_name, [s for s, _ in sub_steps]), est))
            branch_est: Any = Pipeline(sub_steps) if sub_steps else "passthrough"
            transformers.append((branch_name, branch_est, self._selector_for_branch(branch_name, columns)))
        return self._append(name, ColumnTransformer(transformers, remainder=remainder, sparse_threshold=0))

    def reset(self) -> CustomPipelineBuilder:
        """Remove all steps."""
        self.steps = []
        return self

    def build(self, memory: Any = None, verbose: bool = False) -> Pipeline:
        """Build the pipeline from the added steps.

        Args:
            memory: ``joblib`` cache for transformers.
            verbose: Log fitting of each step.

        Returns:
            Unfitted pipeline.

        Raises:
            ValueError: If no steps were added.
        """
        if not self.steps:
            raise ValueError("No steps added to pipeline.")
        self.logger.info("Building custom pipeline with steps %s", [s for s, _ in self.steps])
        return _assemble_pipeline(list(self.steps), memory=memory, verbose=verbose)


# --------------------------------------------------------------------------- #
# PipelineOptimizer
# --------------------------------------------------------------------------- #


class PipelineOptimizer(LoggerMixin):
    """Optimise pipelines: hyper-parameters, feature selection, preprocessing, models.

    Args:
        random_state: Seed for CV splits and randomised components.
        n_jobs: Parallelism for searches and cross-validation.
        cv: Default number of folds (or splitter).
        scoring: Default scoring (``accuracy`` / ``r2`` when ``None``).
        verbose: Verbosity passed to the search objects.
        pipeline: Optional pipeline for the legacy :meth:`optimize` API.
        param_grid: Optional grid for the legacy :meth:`optimize` API.
    """

    def __init__(
        self,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        cv: Optional[Union[int, Any]] = None,
        scoring: Optional[str] = None,
        verbose: int = 0,
        *,
        pipeline: Optional[Pipeline] = None,
        param_grid: Optional[Dict[str, Any]] = None,
    ):
        self.random_state = settings.RANDOM_SEED if random_state is None else random_state
        self.n_jobs = n_jobs
        self.cv = settings.DEFAULT_CV_FOLDS if cv is None else cv
        self.scoring = scoring
        self.verbose = verbose
        self.pipeline = pipeline
        self.param_grid = param_grid
        self.factory = PipelineFactory(random_state=self.random_state, n_jobs=n_jobs)
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None

    def _resolve(
        self, y: Any, task_type: Optional[str], cv: Any, scoring: Optional[str]
    ) -> Tuple[str, Any, str]:
        task = task_type or infer_task_type(y)
        splitter = _cv_splitter(self.cv if cv is None else cv, task, self.random_state)
        return task, splitter, scoring or self.scoring or _default_scoring(task)

    def _cv_score(self, estimator: BaseEstimator, X: ArrayLike, y: Any, splitter: Any, scoring: str) -> float:
        scores = cross_val_score(
            estimator, X, y, cv=splitter, scoring=scoring, n_jobs=self.n_jobs, error_score="raise"
        )
        return float(np.mean(scores))

    def optimize_hyperparameters(
        self,
        pipeline: Pipeline,
        X: ArrayLike,
        y: Any,
        param_grid: Dict[str, Any],
        cv: Optional[Union[int, Any]] = None,
        scoring: Optional[str] = None,
        search: str = "grid",
        n_iter: int = 20,
        task_type: Optional[str] = None,
    ) -> Union[GridSearchCV, RandomizedSearchCV]:
        """Run a cross-validated hyper-parameter search over ``pipeline``.

        Args:
            pipeline: Pipeline to tune.
            X: Feature matrix.
            y: Target vector.
            param_grid: Grid keyed by ``'<step>__<param>'``.
            cv: Folds or splitter (defaults to the optimizer setting).
            scoring: Scoring name.
            search: ``'grid'`` or ``'random'``.
            n_iter: Candidates for randomised search.
            task_type: Override the inferred task type.

        Returns:
            Fitted search object (``best_params_``, ``best_score_``,
            ``best_estimator_``; supports ``predict``/``score``).

        Raises:
            ValueError: If ``search`` is unknown or the grid is empty.
        """
        if not param_grid:
            raise ValueError("param_grid must not be empty.")
        _task, splitter, scoring = self._resolve(y, task_type, cv, scoring)
        if search == "grid":
            searcher: Union[GridSearchCV, RandomizedSearchCV] = GridSearchCV(
                pipeline,
                param_grid,
                cv=splitter,
                scoring=scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                refit=True,
            )
        elif search == "random":
            searcher = RandomizedSearchCV(
                pipeline,
                param_grid,
                n_iter=n_iter,
                cv=splitter,
                scoring=scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                random_state=self.random_state,
                refit=True,
            )
        else:
            raise ValueError(f"search must be 'grid' or 'random', got '{search}'.")
        searcher.fit(X, y)
        self.best_params = dict(searcher.best_params_)
        self.best_score = float(searcher.best_score_)
        self.hyperparameter_results_ = searcher.cv_results_
        self.logger.info("Best %s=%.4f with %s", scoring, self.best_score, self.best_params)
        return searcher

    def optimize(
        self, X: ArrayLike, y: Any, cv: Optional[Union[int, Any]] = None, scoring: Optional[str] = None
    ) -> Pipeline:
        """Legacy API: tune ``self.pipeline`` with ``self.param_grid`` and return it with the best parameters set.

        Raises:
            ValueError: If the optimizer was created without ``pipeline``/``param_grid``.
        """
        if self.pipeline is None or not self.param_grid:
            raise ValueError(
                "optimize() requires 'pipeline' and 'param_grid' to be passed to the constructor."
            )
        searcher = self.optimize_hyperparameters(self.pipeline, X, y, self.param_grid, cv=cv, scoring=scoring)
        return searcher.best_estimator_

    def optimize_feature_selection(
        self,
        X: ArrayLike,
        y: Any,
        max_features: int = 10,
        method: str = "univariate",
        estimator: Optional[BaseEstimator] = None,
        cv: Optional[Union[int, Any]] = None,
        scoring: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> np.ndarray:
        """Choose the number of features (<= ``max_features``) that maximises CV score.

        Args:
            X: Feature matrix.
            y: Target vector.
            max_features: Upper bound on the number of selected features.
            method: ``'univariate'``, ``'mutual_info'``, ``'rfe'`` or ``'model_based'``.
            estimator: Estimator used for scoring (and for RFE / model-based
                selection); defaults to logistic regression / ridge.
            cv: Folds or splitter.
            scoring: Scoring name.
            task_type: Override the inferred task type.

        Returns:
            Sorted integer indices of the selected features.
        """
        task, splitter, scoring = self._resolve(y, task_type, cv, scoring)
        n_features = X.shape[1]
        max_k = int(min(max_features, n_features))
        if max_k < 1:
            raise ValueError("max_features must be >= 1.")
        base = estimator if estimator is not None else _default_estimator(task, self.random_state)
        candidates = sorted(set(int(k) for k in np.linspace(1, max_k, num=min(max_k, 8))))
        results: Dict[int, float] = {}
        best_k, best_score = candidates[0], -np.inf
        for k in candidates:
            selector = _make_feature_selector(
                method, task, k=k, n_features_to_select=k, estimator=base, random_state=self.random_state
            )
            score = self._cv_score(
                Pipeline([("selector", selector), ("estimator", clone(base))]), X, y, splitter, scoring
            )
            results[k] = score
            if score > best_score:
                best_k, best_score = k, score
        self.feature_selection_results_ = results
        selector = _make_feature_selector(
            method,
            task,
            k=best_k,
            n_features_to_select=best_k,
            estimator=base,
            random_state=self.random_state,
        )
        selector.fit(X, y)
        indices = np.sort(selector.get_support(indices=True)).astype(int)
        self.logger.info(
            "Selected %d features (%s=%.4f): %s", len(indices), scoring, best_score, indices.tolist()
        )
        return indices

    def optimize_preprocessing(
        self,
        X: ArrayLike,
        y: Any,
        scalers: Sequence[str] = ("standard", "minmax", "robust"),
        feature_selectors: Sequence[Optional[str]] = ("none", "univariate"),
        estimator: Optional[BaseEstimator] = None,
        cv: Optional[Union[int, Any]] = None,
        scoring: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Grid-evaluate scaler x feature-selector combinations.

        Args:
            X: Feature matrix.
            y: Target vector.
            scalers: Scaler names (``'none'`` allowed).
            feature_selectors: Selector methods (``'none'``/``None`` allowed).
            estimator: Estimator used for scoring (default logistic regression / ridge).
            cv: Folds or splitter.
            scoring: Scoring name.
            task_type: Override the inferred task type.

        Returns:
            Dict with ``best_scaler``, ``best_selector``, ``score``,
            ``best_pipeline`` (unfitted) and ``results`` (all combinations).
        """
        task, splitter, scoring = self._resolve(y, task_type, cv, scoring)
        base = estimator if estimator is not None else _default_estimator(task, self.random_state)
        results: List[Dict[str, Any]] = []
        best: Optional[Dict[str, Any]] = None
        best_pipeline: Optional[Pipeline] = None
        for scaler_name in scalers:
            for selector_name in feature_selectors:
                steps: List[Step] = [("imputer", SimpleImputer(strategy="median"))]
                scaler = _make_scaler(scaler_name)
                if scaler is not None:
                    steps.append(("scaler", scaler))
                if selector_name not in (None, "none", False):
                    steps.append(
                        (
                            "feature_selection",
                            _make_feature_selector(
                                selector_name, task, estimator=base, random_state=self.random_state
                            ),
                        )
                    )
                steps.append(("estimator", clone(base)))
                pipeline = Pipeline(steps)
                score = self._cv_score(pipeline, X, y, splitter, scoring)
                entry = {"scaler": scaler_name, "selector": selector_name, "score": score}
                results.append(entry)
                if best is None or score > best["score"]:
                    best, best_pipeline = entry, pipeline
        self.preprocessing_results_ = results
        assert best is not None  # scalers / feature_selectors are non-empty
        self.logger.info(
            "Best preprocessing: scaler=%s selector=%s (%s=%.4f)",
            best["scaler"],
            best["selector"],
            scoring,
            best["score"],
        )
        return {
            "best_scaler": best["scaler"],
            "best_selector": best["selector"],
            "score": best["score"],
            "scoring": scoring,
            "best_pipeline": best_pipeline,
            "results": results,
        }

    def optimize_full_pipeline(
        self,
        X: ArrayLike,
        y: Any,
        algorithms: Sequence[str] = ("logistic_regression", "random_forest"),
        preprocessing_options: Sequence[str] = ("standard", "minmax"),
        max_time_minutes: float = 5.0,
        cv: Optional[Union[int, Any]] = None,
        scoring: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> Pipeline:
        """Search algorithm x preprocessing combinations and return the best pipeline, fitted on all data.

        Args:
            X: Feature matrix.
            y: Target vector.
            algorithms: Registered model names.
            preprocessing_options: Scaler names (``'standard'``, ``'minmax'``,
                ``'robust'``, ``'none'``) or registry step names.
            max_time_minutes: Wall-clock budget; the search stops early once
                exceeded (at least one combination is always evaluated).
            cv: Folds or splitter.
            scoring: Scoring name.
            task_type: Override the inferred task type.

        Returns:
            Best pipeline, fitted on ``X, y``.
        """
        if not algorithms or not preprocessing_options:
            raise ValueError("algorithms and preprocessing_options must not be empty.")
        task, splitter, scoring = self._resolve(y, task_type, cv, scoring)
        budget = max_time_minutes * 60.0
        start = time.perf_counter()
        results: List[Dict[str, Any]] = []
        best: Optional[Dict[str, Any]] = None
        best_pipeline: Optional[Pipeline] = None
        stop = False
        for algorithm in algorithms:
            for option in preprocessing_options:
                if _normalize_name(option, _SCALER_ALIASES) in _SCALERS:
                    steps: List[Union[str, Step]] = ["missing_value_imputer", "onehot_encoder"]
                    if _normalize_name(option, _SCALER_ALIASES) != "none":
                        steps.append(f"{_normalize_name(option, _SCALER_ALIASES)}_scaler")
                else:
                    steps = ["missing_value_imputer", "onehot_encoder", option]
                if task == CLASSIFICATION:
                    pipeline = self.factory.create_classification_pipeline(
                        algorithm, preprocessing_steps=steps
                    )
                else:
                    pipeline = self.factory.create_regression_pipeline(algorithm, preprocessing_steps=steps)
                score = self._cv_score(pipeline, X, y, splitter, scoring)
                entry = {
                    "algorithm": algorithm,
                    "preprocessing": option,
                    "score": score,
                    "elapsed_s": time.perf_counter() - start,
                }
                results.append(entry)
                if best is None or score > best["score"]:
                    best, best_pipeline = entry, pipeline
                if time.perf_counter() - start > budget:
                    self.logger.warning(
                        "Time budget of %.1f min exceeded; stopping after %d combinations.",
                        max_time_minutes,
                        len(results),
                    )
                    stop = True
                    break
            if stop:
                break
        assert best is not None and best_pipeline is not None
        self.optimization_results_ = results
        self.best_params = {"algorithm": best["algorithm"], "preprocessing": best["preprocessing"]}
        self.best_score = best["score"]
        self.logger.info(
            "Best pipeline: %s + %s (%s=%.4f)",
            best["algorithm"],
            best["preprocessing"],
            scoring,
            best["score"],
        )
        best_pipeline.fit(X, y)
        return best_pipeline
