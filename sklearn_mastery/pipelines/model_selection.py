"""Model selection, hyperparameter optimisation and experiment tracking.

This module bundles the building blocks that sit between *having a dataset*
and *having a trained model you trust*:

* :class:`ModelSelectionPipeline` -- cross-validated comparison of candidate
  estimators with optional preprocessing, multi-metric scoring and early
  stopping.
* :class:`AutoModelSelector` -- dataset profiling, model recommendation and
  budgeted / progressive automatic selection.
* :class:`ModelComparator` -- statistical, runtime, robustness and calibration
  comparisons of fitted models plus a consolidated report.
* :class:`HyperparameterOptimizer`, :class:`GridSearchPipeline` and
  :class:`BayesianOptimizer` -- grid, random, adaptive, constrained,
  multi-objective and Bayesian search.  Bayesian optimisation uses Optuna when
  it is installed and falls back to a Gaussian-process surrogate built on
  scikit-learn otherwise.
* :class:`CrossValidationPipeline` -- stratified, time-series, nested, custom
  and multi-metric cross-validation helpers.
* :class:`ModelEnsemblePipeline` -- voting, stacking, bagging, dynamic and
  weight-optimised ensembles.
* :class:`PerformanceTracker` and :class:`ModelRegistry` -- lightweight
  in-memory experiment tracking and model versioning.

The legacy analysers (:class:`AdvancedModelSelector`,
:class:`MultiObjectiveSelector`, :class:`NestedCrossValidation`,
:class:`LearningCurveAnalyzer`, :class:`ValidationCurveAnalyzer` and
:class:`AutoMLSelector`) are kept for backwards compatibility.
"""

from __future__ import annotations

import copy
import fnmatch
import inspect
import itertools
import math
import pickle
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone, is_classifier
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    check_scoring,
    f1_score,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    ParameterGrid,
    ParameterSampler,
    RandomizedSearchCV,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_predict,
    cross_val_score,
    cross_validate,
    learning_curve,
    validation_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from sklearn_mastery.config.logging_config import LoggerMixin, get_logger
from sklearn_mastery.config.settings import settings

try:  # optional dependency
    import optuna

    HAS_OPTUNA = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:  # pragma: no cover - exercised only without optuna
    optuna = None
    HAS_OPTUNA = False

try:  # optional dependency used only for persistence in ModelRegistry
    import joblib

    HAS_JOBLIB = True
except ImportError:  # pragma: no cover
    joblib = None
    HAS_JOBLIB = False

_logger = get_logger(__name__)

ScoringLike = Union[str, Callable[..., float], None]
ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series, Sequence[Any]]
ParamDict = Dict[str, Any]

# Objectives that are *minimised* when used in multi-objective optimisation.
_MINIMISE_OBJECTIVES = frozenset(
    {"model_size", "model_complexity", "n_parameters", "training_time", "prediction_time"}
)
# Objectives that are maximised but are not scikit-learn scorers.
_SPECIAL_MAXIMISE_OBJECTIVES = frozenset({"model_simplicity"})
_SPECIAL_OBJECTIVES = _MINIMISE_OBJECTIVES | _SPECIAL_MAXIMISE_OBJECTIVES

_ACQUISITION_ALIASES = {
    "ei": "expected_improvement",
    "expected_improvement": "expected_improvement",
    "pi": "probability_of_improvement",
    "probability_of_improvement": "probability_of_improvement",
    "ucb": "upper_confidence_bound",
    "upper_confidence_bound": "upper_confidence_bound",
}


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def _resolve_random_state(random_state: Optional[int]) -> int:
    """Return ``random_state`` or the project-wide default seed."""
    return settings.RANDOM_SEED if random_state is None else int(random_state)


def _resolve_scorer(scoring: ScoringLike) -> Union[str, Callable[..., float], None]:
    """Normalise a scoring specification for use with scikit-learn.

    Strings and ``None`` pass through unchanged.  Plain metric functions with
    the ``(y_true, y_pred)`` signature are wrapped with
    :func:`sklearn.metrics.make_scorer`; scorer objects (``(estimator, X, y)``)
    are returned as-is.

    Args:
        scoring: Scorer name, metric function, scorer object or ``None``.

    Returns:
        A value accepted by ``cross_validate(scoring=...)``.
    """
    if scoring is None or isinstance(scoring, str):
        return scoring
    if not callable(scoring):
        raise TypeError(f"scoring must be a string, callable or None, got {type(scoring)!r}")
    try:
        signature = inspect.signature(scoring)
    except (TypeError, ValueError):
        return scoring
    positional = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    required = [
        p
        for p in signature.parameters.values()
        if p.default is inspect.Parameter.empty and p.kind in positional
    ]
    if len(required) == 2:
        return make_scorer(scoring)
    return scoring


def _scoring_name(scoring: ScoringLike) -> str:
    """Return a human readable name for a scoring specification."""
    if scoring is None:
        return "score"
    if isinstance(scoring, str):
        return scoring
    name = getattr(scoring, "__name__", None)
    if name is None:
        score_func = getattr(scoring, "_score_func", None)
        name = getattr(score_func, "__name__", None)
    return name or "custom_score"


def _make_cv(
    cv: Any,
    estimator: BaseEstimator,
    y: ArrayLike,
    random_state: int,
    shuffle: bool = True,
) -> Any:
    """Build a cross-validation splitter from an integer or pass one through.

    Classifiers get a shuffled :class:`StratifiedKFold`; other estimators a
    shuffled :class:`KFold`.  The number of folds is capped by the smallest
    class count so that stratification never fails.

    Args:
        cv: Number of folds or an existing splitter / iterable.
        estimator: Estimator used to decide between stratified and plain folds.
        y: Target vector.
        random_state: Seed used when shuffling.
        shuffle: Whether to shuffle before splitting.

    Returns:
        A cross-validation splitter.
    """
    if not isinstance(cv, (int, np.integer)):
        return cv
    n_splits = int(cv)
    seed = random_state if shuffle else None
    if is_classifier(estimator):
        _, counts = np.unique(np.asarray(y), return_counts=True)
        n_splits = max(2, min(n_splits, int(counts.min())))
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)


def _take(data: Any, indices: np.ndarray) -> Any:
    """Row-index ``data`` whether it is an array, a DataFrame or a Series."""
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return np.asarray(data)[indices]


def _like(template: Any, values: np.ndarray) -> Any:
    """Re-wrap ``values`` as a DataFrame when ``template`` is one."""
    if isinstance(template, pd.DataFrame):
        return pd.DataFrame(values, columns=template.columns, index=template.index)
    return values


def _is_fitted(estimator: BaseEstimator) -> bool:
    """Return ``True`` when ``estimator`` has been fitted."""
    try:
        check_is_fitted(estimator)
    except (NotFittedError, TypeError):
        return False
    return True


def _is_number(value: Any) -> bool:
    """Return ``True`` for real numbers (excluding booleans)."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _is_integer(value: Any) -> bool:
    """Return ``True`` for integers (excluding booleans)."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _to_python(value: Any) -> Any:
    """Convert numpy scalars to their Python equivalents."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _estimator_size_bytes(estimator: BaseEstimator) -> int:
    """Return the pickled size of ``estimator`` in bytes."""
    return len(pickle.dumps(estimator))


def _count_parameters(estimator: BaseEstimator) -> int:
    """Estimate the number of learned parameters of a fitted estimator.

    Tree ensembles count tree nodes, linear models count coefficients,
    kernel machines count support-vector entries.  Unknown estimators
    return ``0``.

    Args:
        estimator: A fitted estimator.

    Returns:
        Approximate parameter count.
    """
    if isinstance(estimator, Pipeline):
        estimator = estimator.steps[-1][1]
    total = 0
    if hasattr(estimator, "estimators_"):
        members = np.asarray(estimator.estimators_, dtype=object).ravel()
        total += int(sum(_count_parameters(member) for member in members))
        if hasattr(estimator, "final_estimator_"):
            total += _count_parameters(estimator.final_estimator_)
        return total
    if hasattr(estimator, "tree_"):
        return int(estimator.tree_.node_count)
    if hasattr(estimator, "coef_"):
        return int(np.size(estimator.coef_) + np.size(getattr(estimator, "intercept_", 0)))
    if hasattr(estimator, "support_vectors_"):
        return int(np.size(estimator.support_vectors_))
    if hasattr(estimator, "theta_"):
        return int(np.size(estimator.theta_) + np.size(getattr(estimator, "var_", 0)))
    return total


def _pareto_mask(points: np.ndarray) -> np.ndarray:
    """Return a boolean mask of non-dominated rows (all objectives maximised).

    Args:
        points: Array of shape ``(n_points, n_objectives)``.

    Returns:
        Boolean array; ``True`` marks Pareto-optimal points.
    """
    points = np.asarray(points, dtype=float)
    n_points = len(points)
    mask = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if not mask[i]:
            continue
        dominated = np.all(points >= points[i], axis=1) & np.any(points > points[i], axis=1)
        if dominated.any():
            mask[i] = False
    return mask


def _hypervolume(front: np.ndarray, reference: np.ndarray, random_state: int = 0) -> float:
    """Hypervolume dominated by ``front`` relative to ``reference`` (maximisation).

    Exact for one and two objectives; Monte-Carlo estimate for more.

    Args:
        front: Non-dominated points of shape ``(n, k)``.
        reference: Reference point of shape ``(k,)``.
        random_state: Seed for the Monte-Carlo estimate.

    Returns:
        The dominated hypervolume.
    """
    front = np.asarray(front, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if front.ndim != 2 or len(front) == 0:
        return 0.0
    front = front[np.all(front >= reference, axis=1)]
    if len(front) == 0:
        return 0.0
    n_objectives = front.shape[1]
    if n_objectives == 1:
        return float((front[:, 0] - reference[0]).max())
    if n_objectives == 2:
        order = np.argsort(-front[:, 0])
        volume = 0.0
        previous_y = reference[1]
        for x, y in front[order]:
            if y > previous_y:
                volume += (x - reference[0]) * (y - previous_y)
                previous_y = y
        return float(volume)
    rng = np.random.default_rng(random_state)
    upper = front.max(axis=0)
    box = np.prod(upper - reference)
    if box <= 0:
        return 0.0
    samples = rng.uniform(reference, upper, size=(20_000, n_objectives))
    dominated = np.any(np.all(samples[:, None, :] <= front[None, :, :], axis=2), axis=1)
    return float(dominated.mean() * box)


def _objective_direction(objective: str) -> float:
    """Return ``-1.0`` for minimised objectives and ``1.0`` otherwise."""
    return -1.0 if objective in _MINIMISE_OBJECTIVES else 1.0


def _evaluate_objectives(
    model: BaseEstimator,
    params: ParamDict,
    X: ArrayLike,
    y: ArrayLike,
    objectives: Sequence[str],
    cv: Any,
    random_state: int,
    n_jobs: Optional[int],
) -> Dict[str, float]:
    """Evaluate a parameter configuration on several objectives.

    Scoring objectives are evaluated with cross-validation.  Special objectives
    (``model_size``, ``model_complexity``/``n_parameters``, ``training_time``,
    ``prediction_time``, ``model_simplicity``) are measured on a single fit on
    the full data.

    Args:
        model: Estimator template.
        params: Parameters to set on a clone of ``model``.
        X: Feature matrix.
        y: Target vector.
        objectives: Objective names.
        cv: Folds or splitter for the scoring objectives.
        random_state: Seed used to build the splitter.
        n_jobs: Parallelism passed to ``cross_validate``.

    Returns:
        Mapping from objective name to its raw (un-signed) value.
    """
    estimator = clone(model).set_params(**params)
    values: Dict[str, float] = {}
    scoring_objectives = [o for o in objectives if o not in _SPECIAL_OBJECTIVES]
    if scoring_objectives:
        splitter = _make_cv(cv, estimator, y, random_state)
        cv_results = cross_validate(
            estimator,
            X,
            y,
            cv=splitter,
            scoring={o: _resolve_scorer(o) for o in scoring_objectives},
            n_jobs=n_jobs,
        )
        for objective in scoring_objectives:
            values[objective] = float(np.mean(cv_results[f"test_{objective}"]))
    special = [o for o in objectives if o in _SPECIAL_OBJECTIVES]
    if special:
        fitted = clone(estimator)
        start = time.perf_counter()
        fitted.fit(X, y)
        training_time = time.perf_counter() - start
        start = time.perf_counter()
        fitted.predict(X)
        prediction_time = time.perf_counter() - start
        n_parameters = _count_parameters(fitted)
        size_bytes = _estimator_size_bytes(fitted)
        for objective in special:
            if objective == "model_size":
                values[objective] = float(size_bytes)
            elif objective in ("model_complexity", "n_parameters"):
                values[objective] = float(n_parameters)
            elif objective == "training_time":
                values[objective] = float(training_time)
            elif objective == "prediction_time":
                values[objective] = float(prediction_time)
            elif objective == "model_simplicity":
                values[objective] = 1.0 / (1.0 + math.log1p(max(n_parameters, size_bytes / 1024.0)))
    return values


def _trade_off_analysis(
    solutions: List[Dict[str, Any]],
    pareto: List[Dict[str, Any]],
    objectives: Sequence[str],
) -> Dict[str, Any]:
    """Summarise trade-offs between objectives across evaluated solutions.

    Args:
        solutions: All evaluated solutions (``params`` + ``objectives``).
        pareto: The Pareto-optimal subset.
        objectives: Objective names.

    Returns:
        Dictionary with objective ranges, pairwise correlations, the best
        solution per objective and a balanced "knee" compromise.
    """
    if not solutions:
        return {
            "objective_ranges": {},
            "pairwise_correlation": {},
            "best_per_objective": {},
            "knee_point": None,
        }
    matrix = np.array([[sol["objectives"][o] for o in objectives] for sol in solutions], dtype=float)
    ranges = {o: (float(matrix[:, i].min()), float(matrix[:, i].max())) for i, o in enumerate(objectives)}
    correlations: Dict[str, float] = {}
    for i, j in itertools.combinations(range(len(objectives)), 2):
        col_i, col_j = matrix[:, i], matrix[:, j]
        if np.std(col_i) == 0 or np.std(col_j) == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(col_i, col_j)[0, 1])
        correlations[f"{objectives[i]}_vs_{objectives[j]}"] = corr
    best_per_objective = {}
    for i, objective in enumerate(objectives):
        signed = matrix[:, i] * _objective_direction(objective)
        best_per_objective[objective] = solutions[int(np.argmax(signed))]
    # Knee point: Pareto solution with the best normalised sum of signed objectives.
    knee = None
    if pareto:
        pareto_matrix = np.array([[sol["objectives"][o] for o in objectives] for sol in pareto], dtype=float)
        normalised = np.zeros_like(pareto_matrix)
        for i, objective in enumerate(objectives):
            low, high = ranges[objective]
            span = high - low
            column = (pareto_matrix[:, i] - low) / span if span > 0 else np.ones(len(pareto)) * 0.5
            normalised[:, i] = column if _objective_direction(objective) > 0 else 1.0 - column
        knee = pareto[int(np.argmax(normalised.sum(axis=1)))]
    return {
        "objective_ranges": ranges,
        "pairwise_correlation": correlations,
        "best_per_objective": best_per_objective,
        "knee_point": knee,
        "n_pareto_optimal": len(pareto),
    }


def _build_preprocessing_steps(
    preprocessing: Optional[Iterable[Any]],
    classification: bool,
) -> List[Tuple[str, BaseEstimator]]:
    """Translate a lightweight preprocessing specification into pipeline steps.

    Each item is either a transformer instance or a ``(name, spec)`` tuple where
    ``spec`` is a transformer or one of the string shortcuts ``"standard"``,
    ``"minmax"``, ``"robust"``, ``"univariate_<k>"``, ``"variance_<t>"``,
    ``"pca_<k>"`` or ``"none"``.

    Args:
        preprocessing: Iterable of steps or ``None``.
        classification: Whether the downstream task is classification.

    Returns:
        List of ``(name, transformer)`` pipeline steps.

    Raises:
        ValueError: If a string shortcut is not recognised.
    """
    if not preprocessing:
        return []
    steps: List[Tuple[str, BaseEstimator]] = []
    for index, item in enumerate(preprocessing):
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            name, spec = item
        else:
            name, spec = f"step_{index}", item
        if hasattr(spec, "fit") and hasattr(spec, "transform"):
            steps.append((name, clone(spec)))
            continue
        if not isinstance(spec, str):
            raise ValueError(f"Unsupported preprocessing spec for step '{name}': {spec!r}")
        key = spec.lower()
        suffix = key.split("_")[-1] if "_" in key else ""
        if key in ("standard", "standard_scaler", "standardize", "zscore"):
            transformer: BaseEstimator = StandardScaler()
        elif key in ("minmax", "min_max", "minmax_scaler"):
            transformer = MinMaxScaler()
        elif key in ("robust", "robust_scaler"):
            transformer = RobustScaler()
        elif key in ("none", "passthrough"):
            continue
        elif key.startswith("univariate") or key.startswith("selectkbest"):
            k = int(suffix) if suffix.isdigit() else 10
            transformer = SelectKBest(f_classif if classification else f_regression, k=k)
        elif key.startswith("variance"):
            threshold = float(suffix) if suffix.replace(".", "", 1).isdigit() else 0.0
            transformer = VarianceThreshold(threshold)
        elif key.startswith("pca"):
            n_components = int(suffix) if suffix.isdigit() else None
            transformer = PCA(n_components=n_components)
        else:
            raise ValueError(f"Unknown preprocessing shortcut '{spec}' for step '{name}'")
        steps.append((name, transformer))
    return steps


def _normalise_estimators(base_models: Any) -> List[Tuple[str, BaseEstimator]]:
    """Coerce a dict / list of estimators into a list of ``(name, estimator)``.

    Args:
        base_models: Mapping name -> estimator, list of ``(name, estimator)``
            tuples or a plain list of estimators.

    Returns:
        List of unique ``(name, estimator)`` pairs.
    """
    if isinstance(base_models, dict):
        return list(base_models.items())
    pairs: List[Tuple[str, BaseEstimator]] = []
    seen: Dict[str, int] = {}
    for item in base_models:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            name, estimator = item
        else:
            estimator = item
            name = type(estimator).__name__
        count = seen.get(name, 0)
        seen[name] = count + 1
        if count:
            name = f"{name}_{count}"
        pairs.append((name, estimator))
    return pairs


def _class_index(classes: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Map labels to positions in ``classes``."""
    lookup = {label: i for i, label in enumerate(classes.tolist())}
    return np.array([lookup[label] for label in np.asarray(y).tolist()], dtype=int)


# --------------------------------------------------------------------------- #
# Ensemble estimators
# --------------------------------------------------------------------------- #
class _WeightedEnsembleBase(BaseEstimator):
    """Shared implementation of weight-averaged ensembles."""

    def __init__(
        self,
        estimators: List[Tuple[str, BaseEstimator]],
        weights: Optional[Sequence[float]] = None,
        selection_threshold: float = 1e-3,
        n_jobs: Optional[int] = None,
    ):
        self.estimators = estimators
        self.weights = weights
        self.selection_threshold = selection_threshold
        self.n_jobs = n_jobs

    def _resolve_weights(self) -> np.ndarray:
        n_estimators = len(self.estimators)
        if self.weights is None:
            weights = np.ones(n_estimators, dtype=float)
        else:
            weights = np.asarray(self.weights, dtype=float)
            if weights.shape != (n_estimators,):
                raise ValueError(f"weights must have length {n_estimators}, got shape {weights.shape}")
        weights = np.clip(weights, 0.0, None)
        weights[weights < self.selection_threshold] = 0.0
        if weights.sum() <= 0:
            weights = np.ones(n_estimators, dtype=float)
        return weights / weights.sum()

    def fit(self, X: ArrayLike, y: ArrayLike) -> _WeightedEnsembleBase:
        """Fit every base estimator whose weight is non-zero.

        Args:
            X: Feature matrix.
            y: Target vector.

        Returns:
            The fitted ensemble.
        """
        weights = self._resolve_weights()
        self.weights_ = weights
        self.estimators_: List[Tuple[str, BaseEstimator]] = []
        self.selected_models_: List[str] = []
        for (name, estimator), weight in zip(self.estimators, weights):
            if weight <= 0:
                continue
            fitted = clone(estimator).fit(X, y)
            self.estimators_.append((name, fitted))
            self.selected_models_.append(name)
        self.selected_weights_ = weights[weights > 0]
        self.n_features_in_ = np.asarray(X).shape[1]
        return self


class WeightedEnsembleClassifier(ClassifierMixin, _WeightedEnsembleBase):
    """Soft-voting classifier with (optionally optimised) member weights.

    Attributes:
        weights_: Normalised weight of every estimator (zero for dropped ones).
        selected_models_: Names of estimators with a non-zero weight.
        classes_: Class labels seen during :meth:`fit`.
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> WeightedEnsembleClassifier:
        self.classes_ = np.unique(np.asarray(y))
        super().fit(X, y)
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Weighted average of the members' class probabilities."""
        check_is_fitted(self, "estimators_")
        proba = np.zeros((np.asarray(X).shape[0], len(self.classes_)), dtype=float)
        for (_, estimator), weight in zip(self.estimators_, self.selected_weights_):
            if hasattr(estimator, "predict_proba"):
                member = estimator.predict_proba(X)
                member_classes = getattr(estimator, "classes_", self.classes_)
                aligned = np.zeros_like(proba)
                aligned[:, _class_index(self.classes_, member_classes)] = member
            else:
                aligned = np.zeros_like(proba)
                aligned[np.arange(len(aligned)), _class_index(self.classes_, estimator.predict(X))] = 1.0
            proba += weight * aligned
        return proba

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the class with the highest weighted probability."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class WeightedEnsembleRegressor(RegressorMixin, _WeightedEnsembleBase):
    """Weighted-average regressor ensemble.

    Attributes:
        weights_: Normalised weight of every estimator (zero for dropped ones).
        selected_models_: Names of estimators with a non-zero weight.
    """

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Weighted average of the members' predictions."""
        check_is_fitted(self, "estimators_")
        prediction = np.zeros(np.asarray(X).shape[0], dtype=float)
        for (_, estimator), weight in zip(self.estimators_, self.selected_weights_):
            prediction += weight * np.asarray(estimator.predict(X), dtype=float)
        return prediction


class DynamicEnsembleSelector(ClassifierMixin, BaseEstimator):
    """Dynamic ensemble selection over pre-fitted classifiers.

    When a validation set is supplied through :meth:`fit`, every test sample is
    classified by the member(s) that perform best on its ``k_neighbors``
    nearest validation neighbours (dynamic classifier selection).  Without a
    validation set the selector degrades gracefully to (soft) majority voting.

    Args:
        estimators: List of ``(name, fitted_classifier)`` pairs.
        selection_strategy: ``"best_local_accuracy"`` (pick one member),
            ``"weighted_local_accuracy"`` (weight members by local accuracy) or
            ``"majority_vote"``.
        k_neighbors: Size of the competence region.
    """

    _STRATEGIES = ("best_local_accuracy", "weighted_local_accuracy", "majority_vote")

    def __init__(
        self,
        estimators: List[Tuple[str, BaseEstimator]],
        selection_strategy: str = "best_local_accuracy",
        k_neighbors: int = 7,
    ):
        self.estimators = estimators
        self.selection_strategy = selection_strategy
        self.k_neighbors = k_neighbors

    def __sklearn_is_fitted__(self) -> bool:
        return len(self.estimators) > 0

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(self.estimators[0][1].classes_)

    def fit(self, X: ArrayLike, y: ArrayLike) -> DynamicEnsembleSelector:
        """Store the dynamic-selection (validation) set.

        Args:
            X: Validation features.
            y: Validation labels.

        Returns:
            The selector.
        """
        if self.selection_strategy not in self._STRATEGIES:
            raise ValueError(f"selection_strategy must be one of {self._STRATEGIES}")
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        self.X_dsel_ = X_arr
        self.y_dsel_ = y_arr
        k = max(1, min(self.k_neighbors, len(X_arr)))
        self.nn_ = NearestNeighbors(n_neighbors=k).fit(X_arr)
        self.correctness_ = np.array(
            [np.asarray(est.predict(X)) == y_arr for _, est in self.estimators], dtype=float
        )
        return self

    def _local_competence(self, X: ArrayLike) -> Optional[np.ndarray]:
        if not hasattr(self, "nn_"):
            return None
        _, neighbours = self.nn_.kneighbors(np.asarray(X, dtype=float))
        # shape (n_estimators, n_samples)
        return self.correctness_[:, neighbours].mean(axis=2)

    def _member_proba(self, X: ArrayLike) -> np.ndarray:
        classes = self.classes_
        stack = []
        for _, estimator in self.estimators:
            if hasattr(estimator, "predict_proba"):
                proba = np.asarray(estimator.predict_proba(X))
                aligned = np.zeros((proba.shape[0], len(classes)))
                aligned[:, _class_index(classes, np.asarray(estimator.classes_))] = proba
            else:
                aligned = np.zeros((np.asarray(X).shape[0], len(classes)))
                aligned[np.arange(len(aligned)), _class_index(classes, estimator.predict(X))] = 1.0
            stack.append(aligned)
        return np.stack(stack)  # (n_estimators, n_samples, n_classes)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Class probabilities from the dynamically selected member(s)."""
        member_proba = self._member_proba(X)
        competence = self._local_competence(X)
        if competence is None or self.selection_strategy == "majority_vote":
            return member_proba.mean(axis=0)
        if self.selection_strategy == "best_local_accuracy":
            best = np.argmax(competence, axis=0)
            return member_proba[best, np.arange(member_proba.shape[1])]
        weights = competence + 1e-9
        weights /= weights.sum(axis=0, keepdims=True)
        return np.einsum("ms,msc->sc", weights, member_proba)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict labels for ``X``."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# --------------------------------------------------------------------------- #
# Model selection pipeline
# --------------------------------------------------------------------------- #
class ModelSelectionPipeline(LoggerMixin):
    """Cross-validated comparison of candidate estimators.

    Args:
        cv: Default number of folds.
        scoring: Default scoring (name, metric function, scorer or list).
        random_state: Seed for fold shuffling.
        n_jobs: Parallelism passed to ``cross_validate``.

    Attributes:
        results_: Results of the last :meth:`select_best_model` call.
        best_model_name_: Name of the winner of the last selection.
    """

    def __init__(
        self,
        cv: int = 5,
        scoring: Union[ScoringLike, Sequence[ScoringLike]] = "accuracy",
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ):
        self.cv = cv
        self.scoring = scoring
        self.random_state = _resolve_random_state(random_state)
        self.n_jobs = n_jobs
        self.results_: Dict[str, Any] = {}
        self.best_model_name_: Optional[str] = None

    def select_best_model(
        self,
        models: Dict[str, BaseEstimator],
        X: ArrayLike,
        y: ArrayLike,
        scoring: Union[ScoringLike, Sequence[ScoringLike], None] = None,
        cv: Any = None,
        preprocessing: Optional[Iterable[Any]] = None,
        early_stopping: bool = False,
        early_stopping_threshold: Optional[float] = None,
        return_train_score: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """Cross-validate every candidate and return the best one.

        Args:
            models: Mapping from model name to (unfitted) estimator.
            X: Feature matrix.
            y: Target vector.
            scoring: Scoring specification or list of them; the first entry is
                the primary metric used for ranking.  Defaults to the
                pipeline's ``scoring``.
            cv: Folds or splitter; defaults to the pipeline's ``cv``.
            preprocessing: Optional preprocessing specification (see
                :func:`_build_preprocessing_steps`) prepended to each model.
            early_stopping: Stop evaluating once a model's mean primary score
                reaches ``early_stopping_threshold``.
            early_stopping_threshold: Score threshold for early stopping.
            return_train_score: Also record training-fold scores.

        Returns:
            ``(best_model_name, results)`` where ``results`` contains
            ``scores`` (per-model fold scores of the primary metric),
            ``mean_score``, ``std_score``, ``detailed_scores`` (per metric),
            ``fit_time``, ``models_evaluated`` and ``early_stopped``.

        Raises:
            ValueError: If ``models`` is empty.
        """
        if not models:
            raise ValueError("models must contain at least one estimator")
        scoring = self.scoring if scoring is None else scoring
        cv = self.cv if cv is None else cv
        metrics = list(scoring) if isinstance(scoring, (list, tuple)) else [scoring]
        metric_names = [_scoring_name(metric) for metric in metrics]
        scorers = {name: _resolve_scorer(metric) for name, metric in zip(metric_names, metrics)}
        primary = metric_names[0]

        first_model = next(iter(models.values()))
        preprocessing_steps = _build_preprocessing_steps(preprocessing, is_classifier(first_model))

        scores: Dict[str, np.ndarray] = {}
        mean_score: Dict[str, float] = {}
        std_score: Dict[str, float] = {}
        fit_time: Dict[str, float] = {}
        train_scores: Dict[str, np.ndarray] = {}
        detailed: Dict[str, Dict[str, Dict[str, Any]]] = {name: {} for name in metric_names}
        evaluated: List[str] = []
        early_stopped = False

        self.logger.info("Selecting among %d models using primary metric '%s'", len(models), primary)
        for name, model in models.items():
            estimator: BaseEstimator = clone(model)
            if preprocessing_steps:
                estimator = Pipeline([*preprocessing_steps, ("model", estimator)])
            splitter = _make_cv(cv, model, y, self.random_state)
            cv_results = cross_validate(
                estimator,
                X,
                y,
                cv=splitter,
                scoring=scorers,
                n_jobs=self.n_jobs,
                return_train_score=return_train_score,
                error_score="raise",
            )
            for metric in metric_names:
                fold_scores = np.asarray(cv_results[f"test_{metric}"], dtype=float)
                detailed[metric][name] = {
                    "scores": fold_scores,
                    "mean": float(fold_scores.mean()),
                    "std": float(fold_scores.std()),
                }
            primary_scores = detailed[primary][name]["scores"]
            scores[name] = primary_scores
            mean_score[name] = float(primary_scores.mean())
            std_score[name] = float(primary_scores.std())
            fit_time[name] = float(np.sum(cv_results["fit_time"]))
            if return_train_score:
                train_scores[name] = np.asarray(cv_results[f"train_{primary}"], dtype=float)
            evaluated.append(name)
            self.logger.info("%s: %s = %.4f (+/- %.4f)", name, primary, mean_score[name], std_score[name])
            if (
                early_stopping
                and early_stopping_threshold is not None
                and mean_score[name] >= early_stopping_threshold
            ):
                self.logger.info("Early stopping: %s reached threshold %.4f", name, early_stopping_threshold)
                early_stopped = True
                break

        best_name = max(mean_score, key=mean_score.get)
        results: Dict[str, Any] = {
            "best_model": best_name,
            "best_score": mean_score[best_name],
            "scores": scores,
            "mean_score": mean_score,
            "std_score": std_score,
            "fit_time": fit_time,
            "detailed_scores": detailed,
            "primary_metric": primary,
            "metrics": metric_names,
            "models_evaluated": evaluated,
            "early_stopped": early_stopped,
            "cv": cv,
        }
        if return_train_score:
            results["train_scores"] = train_scores
        self.results_ = results
        self.best_model_name_ = best_name
        self.logger.info("Best model: %s (%s = %.4f)", best_name, primary, mean_score[best_name])
        return best_name, results

    def summary(self) -> pd.DataFrame:
        """Return the last selection results as a DataFrame sorted by score.

        Returns:
            DataFrame with one row per evaluated model.
        """
        if not self.results_:
            return pd.DataFrame()
        rows = []
        for name in self.results_["models_evaluated"]:
            row = {
                "model": name,
                "mean_score": self.results_["mean_score"][name],
                "std_score": self.results_["std_score"][name],
                "fit_time": self.results_["fit_time"][name],
                "is_best": name == self.results_["best_model"],
            }
            for metric, per_model in self.results_["detailed_scores"].items():
                row[f"mean_{metric}"] = per_model[name]["mean"]
            rows.append(row)
        return pd.DataFrame(rows).sort_values("mean_score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Automatic model selection
# --------------------------------------------------------------------------- #
class AutoModelSelector(LoggerMixin):
    """Profile a dataset, recommend candidate models and select automatically.

    Args:
        cv: Number of folds used for quick evaluations.
        scoring: Scoring used for evaluations; defaults to ``accuracy`` for
            classification and ``r2`` for regression.
        random_state: Seed for models and fold shuffling.
        n_jobs: Parallelism for cross-validation and parallel estimators.
        max_classes_for_classification: A target with at most this many
            unique values (and integer-like) is treated as classification.
    """

    def __init__(
        self,
        cv: int = 3,
        scoring: ScoringLike = None,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        max_classes_for_classification: int = 20,
    ):
        self.cv = cv
        self.scoring = scoring
        self.random_state = _resolve_random_state(random_state)
        self.n_jobs = n_jobs
        self.max_classes_for_classification = max_classes_for_classification
        self.profile_: Dict[str, Any] = {}
        self.results_: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    def infer_task_type(self, y: ArrayLike) -> str:
        """Infer ``"classification"`` or ``"regression"`` from the target.

        Args:
            y: Target vector.

        Returns:
            The task type.
        """
        y_arr = np.asarray(y)
        if y_arr.dtype.kind in "OUSb":
            return "classification"
        unique = np.unique(y_arr)
        integer_like = np.all(np.mod(unique, 1) == 0)
        if integer_like and len(unique) <= self.max_classes_for_classification:
            return "classification"
        return "regression"

    def _default_scoring(self, task_type: str) -> ScoringLike:
        if self.scoring is not None:
            return self.scoring
        return "accuracy" if task_type == "classification" else "r2"

    def profile_dataset(self, X: ArrayLike, y: ArrayLike) -> Dict[str, Any]:
        """Compute descriptive statistics that drive model recommendation.

        Args:
            X: Feature matrix.
            y: Target vector.

        Returns:
            Dictionary with sample/feature counts, class balance, feature
            correlation summary, missing-value statistics and a
            ``dataset_complexity`` score with a coarse label.
        """
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        y_arr = np.asarray(y)
        n_samples, n_features = X_df.shape
        task_type = self.infer_task_type(y_arr)
        numeric = X_df.select_dtypes(include=[np.number])
        n_numeric = numeric.shape[1]
        missing = int(X_df.isna().sum().sum())

        profile: Dict[str, Any] = {
            "task_type": task_type,
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "n_numeric_features": int(n_numeric),
            "n_categorical_features": int(n_features - n_numeric),
            "samples_per_feature": float(n_samples / max(n_features, 1)),
            "missing_values": missing,
            "missing_fraction": float(missing / max(n_samples * n_features, 1)),
        }

        if task_type == "classification":
            classes, counts = np.unique(y_arr, return_counts=True)
            profile["n_classes"] = len(classes)
            profile["class_balance"] = {
                _to_python(cls): float(c / n_samples) for cls, c in zip(classes, counts)
            }
            profile["imbalance_ratio"] = float(counts.max() / max(counts.min(), 1))
            profile["minority_class_fraction"] = float(counts.min() / n_samples)
        else:
            profile["n_classes"] = None
            profile["class_balance"] = None
            profile["imbalance_ratio"] = 1.0
            profile["target_stats"] = {
                "mean": float(np.mean(y_arr)),
                "std": float(np.std(y_arr)),
                "min": float(np.min(y_arr)),
                "max": float(np.max(y_arr)),
                "skewness": float(stats.skew(y_arr)) if n_samples > 2 else 0.0,
            }

        if n_numeric >= 2:
            corr = np.abs(np.nan_to_num(numeric.corr().to_numpy(), nan=0.0))
            upper = corr[np.triu_indices(n_numeric, k=1)]
            profile["feature_correlation"] = {
                "mean_abs_correlation": float(upper.mean()),
                "max_abs_correlation": float(upper.max()),
                "n_highly_correlated_pairs": int((upper > 0.9).sum()),
                "fraction_highly_correlated": float((upper > 0.9).mean()),
            }
        else:
            profile["feature_correlation"] = {
                "mean_abs_correlation": 0.0,
                "max_abs_correlation": 0.0,
                "n_highly_correlated_pairs": 0,
                "fraction_highly_correlated": 0.0,
            }
        if n_numeric:
            feature_std = numeric.std(ddof=0).to_numpy()
            profile["n_constant_features"] = int((feature_std == 0).sum())
            profile["feature_scale_ratio"] = float(
                np.max(feature_std) / max(np.min(feature_std[feature_std > 0]), 1e-12)
                if np.any(feature_std > 0)
                else 1.0
            )
        else:
            profile["n_constant_features"] = 0
            profile["feature_scale_ratio"] = 1.0

        components = [
            min(1.0, n_features / 50.0),
            min(1.0, 10.0 * n_features / max(n_samples, 1)),
            profile["feature_correlation"]["mean_abs_correlation"],
            min(1.0, (profile["imbalance_ratio"] - 1.0) / 9.0),
        ]
        if task_type == "classification":
            components.append(min(1.0, (profile["n_classes"] - 2) / 8.0))
        score = float(np.clip(np.mean(components), 0.0, 1.0))
        label = "low" if score < 0.3 else "medium" if score < 0.6 else "high"
        profile["dataset_complexity"] = {"score": score, "label": label, "components": components}
        self.profile_ = profile
        return profile

    # ------------------------------------------------------------------ #
    def recommend_models(
        self,
        X: ArrayLike,
        y: ArrayLike,
        max_models: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Recommend candidate estimators for the dataset.

        Each recommendation carries an unfitted ``model``, a ``reason`` and an
        integer ``priority`` (lower is evaluated first).

        Args:
            X: Feature matrix.
            y: Target vector.
            max_models: Optional cap on the number of recommendations.

        Returns:
            List of recommendation dictionaries sorted by priority.
        """
        profile = self.profile_dataset(X, y)
        n_samples = profile["n_samples"]
        n_features = profile["n_features"]
        imbalanced = profile["imbalance_ratio"] > 3.0
        correlated = profile["feature_correlation"]["n_highly_correlated_pairs"] > 0
        rs = self.random_state
        recommendations: List[Dict[str, Any]] = []

        def add(name: str, model: BaseEstimator, reason: str, priority: int) -> None:
            recommendations.append({"name": name, "model": model, "reason": reason, "priority": priority})

        if profile["task_type"] == "classification":
            class_weight = "balanced" if imbalanced else None
            add(
                "LogisticRegression",
                LogisticRegression(max_iter=1000, random_state=rs, class_weight=class_weight),
                "Fast, well-calibrated linear baseline"
                + ("; class_weight='balanced' compensates for class imbalance" if imbalanced else ""),
                1,
            )
            add(
                "RandomForestClassifier",
                RandomForestClassifier(
                    n_estimators=100, random_state=rs, n_jobs=self.n_jobs, class_weight=class_weight
                ),
                "Robust to feature scaling and interactions, exposes feature importances"
                + ("; tolerates the highly correlated features found in the data" if correlated else ""),
                2,
            )
            if n_samples <= 10_000:
                add(
                    "SVC",
                    SVC(random_state=rs, probability=True, class_weight=class_weight),
                    f"Kernel SVM captures non-linear boundaries; tractable for {n_samples} samples",
                    3,
                )
            add(
                "GradientBoostingClassifier",
                GradientBoostingClassifier(random_state=rs),
                "Boosted trees typically give the best accuracy on tabular data",
                4,
            )
            if n_samples <= 20_000 and n_features <= 100:
                add(
                    "KNeighborsClassifier",
                    KNeighborsClassifier(),
                    "Non-parametric baseline suited to low-dimensional, moderate-size data",
                    5,
                )
            add("GaussianNB", GaussianNB(), "Extremely fast probabilistic baseline", 6)
            if n_samples > 10_000:
                add(
                    "SVC",
                    SVC(random_state=rs, probability=True, class_weight=class_weight),
                    "Kernel SVM; lower priority because training scales super-linearly with sample count",
                    7,
                )
        else:
            add("LinearRegression", LinearRegression(), "Fast interpretable linear baseline", 1)
            add(
                "Ridge",
                Ridge(random_state=rs),
                "L2-regularised linear model; stable when features are correlated"
                if correlated
                else "L2 regularisation",
                2,
            )
            add(
                "RandomForestRegressor",
                RandomForestRegressor(n_estimators=100, random_state=rs, n_jobs=self.n_jobs),
                "Captures non-linearities and interactions without feature scaling",
                3,
            )
            add(
                "GradientBoostingRegressor",
                GradientBoostingRegressor(random_state=rs),
                "Boosted trees for strong tabular performance",
                4,
            )
            if n_samples <= 10_000:
                add("SVR", SVR(), "Kernel regression for smooth non-linear targets", 5)
            if n_samples <= 20_000 and n_features <= 100:
                add(
                    "KNeighborsRegressor", KNeighborsRegressor(), "Non-parametric local averaging baseline", 6
                )

        recommendations.sort(key=lambda rec: rec["priority"])
        if max_models is not None:
            recommendations = recommendations[:max_models]
        return recommendations

    # ------------------------------------------------------------------ #
    def _cv_score(self, model: BaseEstimator, X: ArrayLike, y: ArrayLike, scoring: ScoringLike) -> np.ndarray:
        splitter = _make_cv(self.cv, model, y, self.random_state)
        return np.asarray(
            cross_val_score(
                clone(model), X, y, cv=splitter, scoring=_resolve_scorer(scoring), n_jobs=self.n_jobs
            ),
            dtype=float,
        )

    def auto_select(
        self,
        X: ArrayLike,
        y: ArrayLike,
        time_budget_minutes: float = 5.0,
        max_models: Optional[int] = None,
        scoring: ScoringLike = None,
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Evaluate recommended models within a time budget and fit the best.

        Args:
            X: Feature matrix.
            y: Target vector.
            time_budget_minutes: Wall-clock budget; evaluation stops when it is
                exhausted (at least one model is always evaluated).
            max_models: Maximum number of models to evaluate.
            scoring: Scoring override.

        Returns:
            ``(fitted_best_model, results)`` with ``results`` containing
            ``evaluation_results``, ``models_evaluated``, ``best_model_name``,
            ``best_score``, ``elapsed_seconds`` and ``stopping_reason``.
        """
        start = time.perf_counter()
        budget = float(time_budget_minutes) * 60.0
        recommendations = self.recommend_models(X, y, max_models=max_models)
        scoring = scoring or self._default_scoring(self.profile_["task_type"])

        evaluation_results: Dict[str, Dict[str, Any]] = {}
        models_evaluated: List[str] = []
        best_name: Optional[str] = None
        best_score = -np.inf
        stopping_reason = "all_candidates_evaluated"
        last_duration = 0.0
        for rec in recommendations:
            elapsed = time.perf_counter() - start
            if models_evaluated and elapsed + last_duration > budget:
                stopping_reason = "time_budget_exhausted"
                break
            eval_start = time.perf_counter()
            try:
                fold_scores = self._cv_score(rec["model"], X, y, scoring)
            except Exception as exc:
                self.logger.warning("Skipping %s: %s", rec["name"], exc)
                evaluation_results[rec["name"]] = {"error": str(exc)}
                continue
            last_duration = time.perf_counter() - eval_start
            mean = float(fold_scores.mean())
            evaluation_results[rec["name"]] = {
                "mean_score": mean,
                "std_score": float(fold_scores.std()),
                "scores": fold_scores,
                "evaluation_time": last_duration,
                "reason": rec["reason"],
                "priority": rec["priority"],
            }
            models_evaluated.append(rec["name"])
            self.logger.info("%s: %.4f (%.1fs)", rec["name"], mean, last_duration)
            if mean > best_score:
                best_score, best_name = mean, rec["name"]

        if best_name is None:
            raise RuntimeError("No candidate model could be evaluated")
        best_template = next(rec["model"] for rec in recommendations if rec["name"] == best_name)
        best_model = clone(best_template).fit(X, y)
        results = {
            "best_model_name": best_name,
            "best_score": best_score,
            "evaluation_results": evaluation_results,
            "models_evaluated": models_evaluated,
            "elapsed_seconds": time.perf_counter() - start,
            "time_budget_seconds": budget,
            "stopping_reason": stopping_reason,
            "scoring": _scoring_name(scoring),
            "task_type": self.profile_["task_type"],
        }
        self.results_ = results
        return best_model, results

    # ------------------------------------------------------------------ #
    def select_interpretable_models(
        self,
        X: ArrayLike,
        y: ArrayLike,
        require_feature_importance: bool = True,
        scoring: ScoringLike = None,
    ) -> Dict[str, Any]:
        """Select the best model among interpretable candidates.

        Args:
            X: Feature matrix.
            y: Target vector.
            require_feature_importance: Restrict candidates to tree-based
                models exposing ``feature_importances_``; otherwise linear
                models (``coef_``) are included as well.
            scoring: Scoring override.

        Returns:
            Dictionary with ``best_model`` (fitted), ``best_model_name``,
            ``feature_importance`` (normalised, one entry per feature),
            ``feature_ranking`` and per-candidate ``scores``.
        """
        task_type = self.infer_task_type(y)
        scoring = scoring or self._default_scoring(task_type)
        rs = self.random_state
        if task_type == "classification":
            candidates: Dict[str, BaseEstimator] = {
                "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=6, random_state=rs),
                "RandomForestClassifier": RandomForestClassifier(
                    n_estimators=100, random_state=rs, n_jobs=self.n_jobs
                ),
                "ExtraTreesClassifier": ExtraTreesClassifier(
                    n_estimators=100, random_state=rs, n_jobs=self.n_jobs
                ),
                "GradientBoostingClassifier": GradientBoostingClassifier(random_state=rs),
            }
            if not require_feature_importance:
                candidates["LogisticRegression"] = LogisticRegression(max_iter=1000, random_state=rs)
        else:
            candidates = {
                "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=6, random_state=rs),
                "RandomForestRegressor": RandomForestRegressor(
                    n_estimators=100, random_state=rs, n_jobs=self.n_jobs
                ),
                "ExtraTreesRegressor": ExtraTreesRegressor(
                    n_estimators=100, random_state=rs, n_jobs=self.n_jobs
                ),
                "GradientBoostingRegressor": GradientBoostingRegressor(random_state=rs),
            }
            if not require_feature_importance:
                candidates["LinearRegression"] = LinearRegression()
                candidates["Ridge"] = Ridge(random_state=rs)

        scores: Dict[str, Dict[str, float]] = {}
        for name, model in candidates.items():
            fold_scores = self._cv_score(model, X, y, scoring)
            scores[name] = {"mean_score": float(fold_scores.mean()), "std_score": float(fold_scores.std())}
        best_name = max(scores, key=lambda name: scores[name]["mean_score"])
        best_model = clone(candidates[best_name]).fit(X, y)
        importance = self._extract_importance(best_model)
        return {
            "best_model": best_model,
            "best_model_name": best_name,
            "feature_importance": importance,
            "feature_ranking": np.argsort(-importance),
            "scores": scores,
            "candidates": candidates,
            "scoring": _scoring_name(scoring),
        }

    @staticmethod
    def _extract_importance(model: BaseEstimator) -> np.ndarray:
        if hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_, dtype=float)
            importance = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        else:
            raise AttributeError(f"{type(model).__name__} exposes neither feature_importances_ nor coef_")
        total = importance.sum()
        return importance / total if total > 0 else importance

    # ------------------------------------------------------------------ #
    def progressive_evaluation(
        self,
        X: ArrayLike,
        y: ArrayLike,
        start_simple: bool = True,
        performance_threshold: Optional[float] = None,
        scoring: ScoringLike = None,
    ) -> Dict[str, Any]:
        """Evaluate models in stages of increasing (or decreasing) complexity.

        Evaluation stops as soon as the best score so far reaches
        ``performance_threshold``.

        Args:
            X: Feature matrix.
            y: Target vector.
            start_simple: Begin with the cheapest models.
            performance_threshold: Optional score at which to stop.
            scoring: Scoring override.

        Returns:
            Dictionary with ``evaluation_order``, ``best_model_at_each_stage``,
            ``stage_results``, ``stopping_reason``, ``best_model_name`` and
            ``best_score``.
        """
        task_type = self.infer_task_type(y)
        scoring = scoring or self._default_scoring(task_type)
        rs = self.random_state
        if task_type == "classification":
            stages: List[Tuple[str, Dict[str, BaseEstimator]]] = [
                (
                    "simple",
                    {
                        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=rs),
                        "GaussianNB": GaussianNB(),
                    },
                ),
                (
                    "intermediate",
                    {
                        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=rs),
                        "KNeighborsClassifier": KNeighborsClassifier(),
                        "RandomForestClassifier": RandomForestClassifier(
                            n_estimators=100, random_state=rs, n_jobs=self.n_jobs
                        ),
                    },
                ),
                (
                    "complex",
                    {
                        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=rs),
                        "SVC": SVC(random_state=rs),
                    },
                ),
            ]
        else:
            stages = [
                ("simple", {"LinearRegression": LinearRegression(), "Ridge": Ridge(random_state=rs)}),
                (
                    "intermediate",
                    {
                        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=rs),
                        "KNeighborsRegressor": KNeighborsRegressor(),
                        "RandomForestRegressor": RandomForestRegressor(
                            n_estimators=100, random_state=rs, n_jobs=self.n_jobs
                        ),
                    },
                ),
                (
                    "complex",
                    {"GradientBoostingRegressor": GradientBoostingRegressor(random_state=rs), "SVR": SVR()},
                ),
            ]
        if not start_simple:
            stages = stages[::-1]

        evaluation_order: List[str] = []
        best_at_stage: List[Dict[str, Any]] = []
        stage_results: Dict[str, Dict[str, Any]] = {}
        best_name: Optional[str] = None
        best_score = -np.inf
        stopping_reason = "all_stages_evaluated"
        for stage_name, models in stages:
            stage_scores: Dict[str, float] = {}
            for name, model in models.items():
                fold_scores = self._cv_score(model, X, y, scoring)
                stage_scores[name] = float(fold_scores.mean())
                evaluation_order.append(name)
                if stage_scores[name] > best_score:
                    best_score, best_name = stage_scores[name], name
            stage_best = max(stage_scores, key=stage_scores.get)
            stage_results[stage_name] = stage_scores
            best_at_stage.append(
                {
                    "stage": stage_name,
                    "stage_best_model": stage_best,
                    "stage_best_score": stage_scores[stage_best],
                    "model": best_name,
                    "score": best_score,
                }
            )
            self.logger.info("Stage '%s': best so far %s (%.4f)", stage_name, best_name, best_score)
            if performance_threshold is not None and best_score >= performance_threshold:
                stopping_reason = (
                    f"performance_threshold_reached ({best_score:.4f} >= {performance_threshold})"
                )
                break
        return {
            "evaluation_order": evaluation_order,
            "best_model_at_each_stage": best_at_stage,
            "stage_results": stage_results,
            "stopping_reason": stopping_reason,
            "best_model_name": best_name,
            "best_score": best_score,
            "stages_completed": len(best_at_stage),
            "scoring": _scoring_name(scoring),
        }


# --------------------------------------------------------------------------- #
# Model comparison
# --------------------------------------------------------------------------- #
class ModelComparator(LoggerMixin):
    """Statistical, runtime, robustness and calibration comparison of models.

    All public methods accept a mapping ``name -> estimator``.  Fitted
    estimators are evaluated as-is; unfitted ones are fitted on the provided
    data where that is needed.

    Args:
        scoring: Default scoring; ``None`` uses each estimator's ``score``.
        random_state: Seed for fold shuffling and perturbations.
        n_jobs: Parallelism for cross-validation.
        alpha: Significance level for hypothesis tests.
    """

    def __init__(
        self,
        scoring: ScoringLike = None,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        alpha: float = 0.05,
    ):
        self.scoring = scoring
        self.random_state = _resolve_random_state(random_state)
        self.n_jobs = n_jobs
        self.alpha = alpha

    # ------------------------------------------------------------------ #
    def _scorer(self, model: BaseEstimator, scoring: ScoringLike) -> Callable[..., float]:
        return check_scoring(model, scoring=_resolve_scorer(self.scoring if scoring is None else scoring))

    def _fold_scores(
        self,
        model: BaseEstimator,
        X: ArrayLike,
        y: ArrayLike,
        cv: Any,
        scoring: ScoringLike,
        refit: bool,
    ) -> np.ndarray:
        """Per-fold scores; fitted models are scored fold-wise without refitting."""
        splitter = _make_cv(cv, model, y, self.random_state)
        scorer = self._scorer(model, scoring)
        if refit or not _is_fitted(model):
            return np.asarray(
                cross_val_score(clone(model), X, y, cv=splitter, scoring=scorer, n_jobs=self.n_jobs)
            )
        scores = []
        for _, test_idx in splitter.split(X, y):
            scores.append(float(scorer(model, _take(X, test_idx), _take(y, test_idx))))
        return np.asarray(scores, dtype=float)

    def statistical_comparison(
        self,
        models: Dict[str, BaseEstimator],
        X: ArrayLike,
        y: ArrayLike,
        cv: Any = 5,
        scoring: ScoringLike = None,
        test: str = "paired_t",
        refit: bool = False,
    ) -> Dict[str, Any]:
        """Rank models and test pairwise score differences for significance.

        Args:
            models: Mapping name -> estimator (fitted or not).
            X: Evaluation features (typically a held-out set).
            y: Evaluation targets.
            cv: Number of folds (or splitter) used to obtain paired scores.
            scoring: Scoring override.
            test: ``"paired_t"`` or ``"wilcoxon"``.
            refit: Refit clones on every fold instead of scoring the fitted
                model fold-wise.

        Returns:
            Dictionary with ``ranking`` (list sorted by mean score),
            ``pairwise_comparisons`` (with ``p_value``), a
            ``statistical_significance`` summary and raw ``fold_scores``.
        """
        if len(models) < 2:
            raise ValueError("statistical_comparison needs at least two models")
        names = list(models)
        fold_scores = {name: self._fold_scores(models[name], X, y, cv, scoring, refit) for name in names}
        ranking = sorted(
            (
                {
                    "model": name,
                    "mean_score": float(fold_scores[name].mean()),
                    "std_score": float(fold_scores[name].std()),
                    "scores": fold_scores[name],
                }
                for name in names
            ),
            key=lambda item: item["mean_score"],
            reverse=True,
        )
        for rank, item in enumerate(ranking, start=1):
            item["rank"] = rank

        pairwise: List[Dict[str, Any]] = []
        for name_a, name_b in itertools.combinations(names, 2):
            scores_a, scores_b = fold_scores[name_a], fold_scores[name_b]
            diff = scores_a - scores_b
            if np.allclose(diff, 0.0):
                statistic, p_value = 0.0, 1.0
                used_test = test
            else:
                used_test = test
                try:
                    if test == "wilcoxon":
                        statistic, p_value = stats.wilcoxon(scores_a, scores_b)
                    else:
                        statistic, p_value = stats.ttest_rel(scores_a, scores_b)
                except ValueError:
                    statistic, p_value = stats.ttest_rel(scores_a, scores_b)
                    used_test = "paired_t"
            p_value = float(np.nan_to_num(p_value, nan=1.0))
            p_value = float(min(max(p_value, 0.0), 1.0))
            mean_diff = float(diff.mean())
            pairwise.append(
                {
                    "model_a": name_a,
                    "model_b": name_b,
                    "statistic": float(np.nan_to_num(statistic, nan=0.0)),
                    "p_value": p_value,
                    "mean_difference": mean_diff,
                    "significant": p_value < self.alpha,
                    "better": name_a if mean_diff >= 0 else name_b,
                    "test": used_test,
                }
            )

        friedman = None
        if len(names) >= 3:
            try:
                f_stat, f_p = stats.friedmanchisquare(*[fold_scores[name] for name in names])
                friedman = {"statistic": float(f_stat), "p_value": float(np.nan_to_num(f_p, nan=1.0))}
            except ValueError:
                friedman = None
        best = ranking[0]["model"]
        beaten = [
            comp["model_b"] if comp["model_a"] == best else comp["model_a"]
            for comp in pairwise
            if best in (comp["model_a"], comp["model_b"]) and comp["significant"] and comp["better"] == best
        ]
        significance = {
            "alpha": self.alpha,
            "test": test,
            "n_significant_pairs": int(sum(comp["significant"] for comp in pairwise)),
            "friedman": friedman,
            "best_model": best,
            "best_significantly_better_than": beaten,
            "any_significant_difference": any(comp["significant"] for comp in pairwise),
        }
        return {
            "ranking": ranking,
            "pairwise_comparisons": pairwise,
            "statistical_significance": significance,
            "fold_scores": fold_scores,
            "scoring": _scoring_name(self.scoring if scoring is None else scoring),
        }

    # ------------------------------------------------------------------ #
    def profile_performance(
        self,
        models: Dict[str, BaseEstimator],
        X: ArrayLike,
        y: ArrayLike,
        n_repeats: int = 3,
        scoring: ScoringLike = None,
    ) -> Dict[str, Any]:
        """Measure training time, prediction latency, memory and complexity.

        Args:
            models: Mapping name -> estimator.
            X: Feature matrix (used to time a fresh fit and predictions).
            y: Target vector.
            n_repeats: Number of timed prediction runs (median is reported).
            scoring: Scoring override for the reported ``scores``.

        Returns:
            Dictionary with ``training_time``, ``prediction_time``,
            ``prediction_time_per_sample``, ``memory_usage`` (pickled bytes),
            ``model_complexity``, ``scores`` and a ``summary`` DataFrame.
        """
        n_samples = np.asarray(X).shape[0]
        training_time: Dict[str, float] = {}
        prediction_time: Dict[str, float] = {}
        memory_usage: Dict[str, int] = {}
        complexity: Dict[str, Dict[str, Any]] = {}
        scores: Dict[str, float] = {}
        for name, model in models.items():
            estimator = clone(model)
            start = time.perf_counter()
            estimator.fit(X, y)
            training_time[name] = time.perf_counter() - start
            timings = []
            for _ in range(max(1, n_repeats)):
                start = time.perf_counter()
                estimator.predict(X)
                timings.append(time.perf_counter() - start)
            prediction_time[name] = float(np.median(timings))
            memory_usage[name] = _estimator_size_bytes(estimator)
            complexity[name] = {
                "n_parameters": _count_parameters(estimator),
                "size_bytes": memory_usage[name],
                "estimator_type": type(estimator).__name__,
                "hyperparameters": estimator.get_params(deep=False),
            }
            scored_model = model if _is_fitted(model) else estimator
            scores[name] = float(self._scorer(scored_model, scoring)(scored_model, X, y))
        summary = pd.DataFrame(
            {
                "model": list(models),
                "score": [scores[n] for n in models],
                "training_time": [training_time[n] for n in models],
                "prediction_time": [prediction_time[n] for n in models],
                "memory_bytes": [memory_usage[n] for n in models],
                "n_parameters": [complexity[n]["n_parameters"] for n in models],
            }
        )
        return {
            "training_time": training_time,
            "prediction_time": prediction_time,
            "prediction_time_per_sample": {n: t / max(n_samples, 1) for n, t in prediction_time.items()},
            "memory_usage": memory_usage,
            "model_complexity": complexity,
            "scores": scores,
            "fastest_training": min(training_time, key=training_time.get),
            "fastest_prediction": min(prediction_time, key=prediction_time.get),
            "summary": summary,
        }

    # ------------------------------------------------------------------ #
    def test_robustness(
        self,
        models: Dict[str, BaseEstimator],
        X: ArrayLike,
        y: ArrayLike,
        noise_levels: Sequence[float] = (0.01, 0.05, 0.1),
        outlier_fraction: float = 0.05,
        outlier_magnitude: float = 5.0,
        n_repeats: int = 3,
        scoring: ScoringLike = None,
    ) -> Dict[str, Any]:
        """Measure sensitivity to Gaussian feature noise and injected outliers.

        Noise is scaled per feature by the feature's standard deviation.

        Args:
            models: Mapping name -> fitted estimator (unfitted ones are fitted
                on ``X``/``y``).
            X: Evaluation features.
            y: Evaluation targets.
            noise_levels: Noise standard deviations relative to feature scale.
            outlier_fraction: Fraction of rows turned into outliers.
            outlier_magnitude: Outlier offset in units of feature std.
            n_repeats: Number of random perturbations per level.
            scoring: Scoring override.

        Returns:
            Dictionary with ``baseline_scores``, ``noise_sensitivity``,
            ``outlier_sensitivity``, ``stability_scores`` (1 = unaffected) and
            ``most_robust_model``.
        """
        rng = np.random.default_rng(self.random_state)
        X_arr = np.asarray(X, dtype=float)
        feature_std = X_arr.std(axis=0)
        feature_std[feature_std == 0] = 1.0
        n_samples, n_features = X_arr.shape
        n_outliers = max(1, round(outlier_fraction * n_samples))

        baseline: Dict[str, float] = {}
        noise_sensitivity: Dict[str, Dict[float, Dict[str, float]]] = {}
        outlier_sensitivity: Dict[str, Dict[str, float]] = {}
        stability: Dict[str, float] = {}
        for name, model in models.items():
            estimator = model if _is_fitted(model) else clone(model).fit(X, y)
            scorer = self._scorer(estimator, scoring)
            base = float(scorer(estimator, X, y))
            baseline[name] = base
            scale = max(abs(base), 1e-12)
            degradations: List[float] = []
            noise_sensitivity[name] = {}
            for level in noise_levels:
                level_scores = []
                for _ in range(max(1, n_repeats)):
                    noisy = X_arr + rng.normal(0.0, float(level), size=X_arr.shape) * feature_std
                    level_scores.append(float(scorer(estimator, _like(X, noisy), y)))
                mean_score = float(np.mean(level_scores))
                degradation = base - mean_score
                degradations.append(max(0.0, degradation) / scale)
                noise_sensitivity[name][float(level)] = {
                    "mean_score": mean_score,
                    "std_score": float(np.std(level_scores)),
                    "degradation": float(degradation),
                }
            outlier_scores = []
            for _ in range(max(1, n_repeats)):
                corrupted = X_arr.copy()
                rows = rng.choice(n_samples, size=n_outliers, replace=False)
                cols = rng.choice(n_features, size=max(1, n_features // 3), replace=False)
                signs = rng.choice([-1.0, 1.0], size=(n_outliers, len(cols)))
                corrupted[np.ix_(rows, cols)] += signs * outlier_magnitude * feature_std[cols]
                outlier_scores.append(float(scorer(estimator, _like(X, corrupted), y)))
            outlier_mean = float(np.mean(outlier_scores))
            outlier_degradation = base - outlier_mean
            degradations.append(max(0.0, outlier_degradation) / scale)
            outlier_sensitivity[name] = {
                "score": outlier_mean,
                "degradation": float(outlier_degradation),
                "n_outliers": n_outliers,
            }
            stability[name] = float(np.clip(1.0 - np.mean(degradations), 0.0, 1.0))
        return {
            "baseline_scores": baseline,
            "noise_sensitivity": noise_sensitivity,
            "outlier_sensitivity": outlier_sensitivity,
            "stability_scores": stability,
            "noise_levels": [float(level) for level in noise_levels],
            "most_robust_model": max(stability, key=stability.get) if stability else None,
        }

    # ------------------------------------------------------------------ #
    def compare_calibration(
        self,
        models: Dict[str, BaseEstimator],
        X: ArrayLike,
        y: ArrayLike,
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        """Compare probability calibration of classifiers with ``predict_proba``.

        Binary problems use the positive-class probability; multiclass problems
        use the multiclass Brier score and confidence of the predicted class.

        Args:
            models: Mapping name -> fitted classifier.
            X: Evaluation features.
            y: Evaluation labels.
            n_bins: Number of equal-width confidence bins.

        Returns:
            Dictionary with ``brier_scores``, ``calibration_errors`` (expected
            and maximum calibration error), ``reliability_diagrams`` (binned
            confidence vs. accuracy), ``skipped_models`` and
            ``best_calibrated_model``.
        """
        y_arr = np.asarray(y)
        brier_scores: Dict[str, float] = {}
        calibration_errors: Dict[str, Dict[str, float]] = {}
        reliability: Dict[str, Dict[str, List[float]]] = {}
        skipped: List[str] = []
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        for name, model in models.items():
            if not hasattr(model, "predict_proba"):
                skipped.append(name)
                continue
            estimator = model if _is_fitted(model) else clone(model).fit(X, y)
            proba = np.asarray(estimator.predict_proba(X), dtype=float)
            classes = np.asarray(estimator.classes_)
            y_idx = _class_index(classes, y_arr)
            one_hot = np.eye(len(classes))[y_idx]
            if len(classes) == 2:
                confidence = proba[:, 1]
                correct = one_hot[:, 1]
                brier = float(np.mean((confidence - correct) ** 2))
            else:
                confidence = proba.max(axis=1)
                correct = (np.argmax(proba, axis=1) == y_idx).astype(float)
                brier = float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))
            bin_ids = np.clip(np.digitize(confidence, bin_edges[1:-1], right=True), 0, n_bins - 1)
            prob_pred, prob_true, counts, gaps = [], [], [], []
            for b in range(n_bins):
                mask = bin_ids == b
                if not mask.any():
                    continue
                mean_conf = float(confidence[mask].mean())
                mean_acc = float(correct[mask].mean())
                prob_pred.append(mean_conf)
                prob_true.append(mean_acc)
                counts.append(int(mask.sum()))
                gaps.append(abs(mean_acc - mean_conf))
            weights = np.asarray(counts, dtype=float) / len(confidence)
            ece = float(np.sum(weights * np.asarray(gaps))) if gaps else 0.0
            mce = float(np.max(gaps)) if gaps else 0.0
            brier_scores[name] = brier
            calibration_errors[name] = {"expected_calibration_error": ece, "maximum_calibration_error": mce}
            reliability[name] = {"prob_pred": prob_pred, "prob_true": prob_true, "counts": counts}
        return {
            "brier_scores": brier_scores,
            "calibration_errors": calibration_errors,
            "reliability_diagrams": reliability,
            "skipped_models": skipped,
            "n_bins": n_bins,
            "best_calibrated_model": min(brier_scores, key=brier_scores.get) if brier_scores else None,
        }

    # ------------------------------------------------------------------ #
    def generate_comprehensive_report(
        self,
        models: Dict[str, BaseEstimator],
        X: ArrayLike,
        y: ArrayLike,
        cv: Any = 5,
        scoring: ScoringLike = None,
    ) -> Dict[str, Any]:
        """Run every comparison and consolidate the findings.

        Args:
            models: Mapping name -> fitted estimator.
            X: Evaluation features.
            y: Evaluation targets.
            cv: Folds for the statistical comparison.
            scoring: Scoring override.

        Returns:
            Dictionary with ``summary``, ``detailed_metrics`` (per model),
            actionable ``recommendations`` (``action`` + ``reasoning``),
            ``visualizations`` (plot-ready data) and the raw sub-reports.
        """
        statistical = self.statistical_comparison(models, X, y, cv=cv, scoring=scoring)
        performance = self.profile_performance(models, X, y, scoring=scoring)
        robustness = self.test_robustness(models, X, y, scoring=scoring)
        classification = all(is_classifier(m) for m in models.values())
        calibration = self.compare_calibration(models, X, y) if classification else None

        detailed: Dict[str, Dict[str, Any]] = {}
        for item in statistical["ranking"]:
            name = item["model"]
            estimator = models[name] if _is_fitted(models[name]) else clone(models[name]).fit(X, y)
            prediction = estimator.predict(X)
            metrics: Dict[str, Any] = {
                "rank": item["rank"],
                "mean_cv_score": item["mean_score"],
                "std_cv_score": item["std_score"],
                "training_time": performance["training_time"][name],
                "prediction_time": performance["prediction_time"][name],
                "memory_bytes": performance["memory_usage"][name],
                "n_parameters": performance["model_complexity"][name]["n_parameters"],
                "stability_score": robustness["stability_scores"][name],
            }
            if classification:
                metrics.update(
                    {
                        "accuracy": float(accuracy_score(y, prediction)),
                        "precision_weighted": float(
                            precision_score(y, prediction, average="weighted", zero_division=0)
                        ),
                        "recall_weighted": float(
                            recall_score(y, prediction, average="weighted", zero_division=0)
                        ),
                        "f1_weighted": float(f1_score(y, prediction, average="weighted", zero_division=0)),
                    }
                )
                if calibration and name in calibration["brier_scores"]:
                    metrics["brier_score"] = calibration["brier_scores"][name]
                    metrics["expected_calibration_error"] = calibration["calibration_errors"][name][
                        "expected_calibration_error"
                    ]
            else:
                metrics.update(
                    {
                        "r2": float(r2_score(y, prediction)),
                        "mae": float(mean_absolute_error(y, prediction)),
                        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
                    }
                )
            detailed[name] = metrics

        best = statistical["ranking"][0]["model"]
        runner_up = statistical["ranking"][1]["model"] if len(statistical["ranking"]) > 1 else None
        fastest = performance["fastest_prediction"]
        most_robust = robustness["most_robust_model"]
        best_calibrated = calibration["best_calibrated_model"] if calibration else None
        summary = {
            "n_models": len(models),
            "best_model": best,
            "best_score": statistical["ranking"][0]["mean_score"],
            "runner_up": runner_up,
            "fastest_model": fastest,
            "most_robust_model": most_robust,
            "best_calibrated_model": best_calibrated,
            "scoring": statistical["scoring"],
            "task": "classification" if classification else "regression",
        }

        recommendations: List[Dict[str, str]] = [
            {
                "action": f"Use '{best}' as the primary model",
                "reasoning": (
                    f"Highest mean {statistical['scoring']} ({summary['best_score']:.4f}) across "
                    f"{len(statistical['ranking'][0]['scores'])} folds"
                ),
            }
        ]
        if runner_up is not None:
            pair = next(
                comp
                for comp in statistical["pairwise_comparisons"]
                if {comp["model_a"], comp["model_b"]} == {best, runner_up}
            )
            if not pair["significant"]:
                recommendations.append(
                    {
                        "action": f"Treat '{best}' and '{runner_up}' as interchangeable; prefer the cheaper one",
                        "reasoning": f"Score difference is not statistically significant (p = {pair['p_value']:.3f})",
                    }
                )
        if fastest != best:
            ratio = performance["prediction_time"][best] / max(performance["prediction_time"][fastest], 1e-12)
            if ratio > 2.0:
                recommendations.append(
                    {
                        "action": f"Use '{fastest}' where prediction latency matters",
                        "reasoning": f"It predicts {ratio:.1f}x faster than '{best}'",
                    }
                )
        if most_robust is not None and most_robust != best:
            recommendations.append(
                {
                    "action": f"Prefer '{most_robust}' for noisy production inputs",
                    "reasoning": (
                        f"Stability {robustness['stability_scores'][most_robust]:.3f} vs "
                        f"{robustness['stability_scores'][best]:.3f} for '{best}'"
                    ),
                }
            )
        if best_calibrated is not None and best_calibrated != best:
            recommendations.append(
                {
                    "action": f"Use '{best_calibrated}' (or calibrate '{best}') when probabilities are consumed",
                    "reasoning": (
                        f"Brier score {calibration['brier_scores'][best_calibrated]:.4f} vs "
                        f"{calibration['brier_scores'].get(best, float('nan')):.4f}"
                    ),
                }
            )
        if robustness["stability_scores"][best] < 0.8:
            recommendations.append(
                {
                    "action": f"Add input validation / noise augmentation before deploying '{best}'",
                    "reasoning": f"Its stability score is only {robustness['stability_scores'][best]:.3f}",
                }
            )

        names = [item["model"] for item in statistical["ranking"]]
        visualizations = {
            "score_comparison": {
                "models": names,
                "means": [item["mean_score"] for item in statistical["ranking"]],
                "stds": [item["std_score"] for item in statistical["ranking"]],
            },
            "timing": {
                "models": names,
                "training_time": [performance["training_time"][n] for n in names],
                "prediction_time": [performance["prediction_time"][n] for n in names],
            },
            "noise_sensitivity_curves": {
                n: {
                    "noise_levels": robustness["noise_levels"],
                    "scores": [
                        robustness["noise_sensitivity"][n][lvl]["mean_score"]
                        for lvl in robustness["noise_levels"]
                    ],
                }
                for n in names
            },
            "reliability_diagrams": calibration["reliability_diagrams"] if calibration else {},
            "fold_scores": {n: statistical["fold_scores"][n].tolist() for n in names},
        }
        return {
            "summary": summary,
            "detailed_metrics": detailed,
            "recommendations": recommendations,
            "visualizations": visualizations,
            "statistical_comparison": statistical,
            "performance_profile": performance,
            "robustness": robustness,
            "calibration": calibration,
        }


# --------------------------------------------------------------------------- #
# Hyperparameter optimisation
# --------------------------------------------------------------------------- #
def _cv_results_to_records(cv_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert ``GridSearchCV.cv_results_`` into a list of per-candidate dicts."""
    records = []
    n_candidates = len(cv_results["params"])
    split_keys = [key for key in cv_results if key.startswith("split") and key.endswith("_test_score")]
    for index in range(n_candidates):
        records.append(
            {
                "params": dict(cv_results["params"][index]),
                "mean_test_score": float(cv_results["mean_test_score"][index]),
                "std_test_score": float(cv_results["std_test_score"][index]),
                "rank_test_score": int(cv_results["rank_test_score"][index]),
                "mean_fit_time": float(cv_results["mean_fit_time"][index]),
                "scores": np.asarray([cv_results[key][index] for key in split_keys], dtype=float),
            }
        )
    return records


class HyperparameterOptimizer(LoggerMixin):
    """Grid, random, Bayesian and multi-objective hyperparameter search.

    Args:
        scoring: Default scoring specification.
        random_state: Seed for sampling and fold shuffling.
        n_jobs: Parallelism for cross-validation.
    """

    def __init__(
        self,
        scoring: ScoringLike = "accuracy",
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ):
        self.scoring = scoring
        self.random_state = _resolve_random_state(random_state)
        self.n_jobs = n_jobs
        self.results_: Dict[str, Any] = {}

    def _scoring(self, scoring: ScoringLike) -> Any:
        return _resolve_scorer(self.scoring if scoring is None else scoring)

    def grid_search(
        self,
        model: BaseEstimator,
        param_grid: Union[Dict[str, Sequence[Any]], List[Dict[str, Sequence[Any]]]],
        X: ArrayLike,
        y: ArrayLike,
        cv: Any = 3,
        scoring: ScoringLike = None,
        refit: bool = True,
    ) -> Tuple[ParamDict, Dict[str, Any]]:
        """Exhaustive grid search.

        Args:
            model: Estimator template.
            param_grid: Parameter grid (dict or list of dicts).
            X: Feature matrix.
            y: Target vector.
            cv: Folds or splitter.
            scoring: Scoring override.
            refit: Refit the best configuration on the full data.

        Returns:
            ``(best_params, results)`` where ``results`` holds ``best_score``,
            ``cv_results`` (per-candidate records), ``best_estimator``,
            ``n_candidates`` and ``execution_time``.
        """
        start = time.perf_counter()
        search = GridSearchCV(
            clone(model),
            param_grid,
            cv=_make_cv(cv, model, y, self.random_state),
            scoring=self._scoring(scoring),
            n_jobs=self.n_jobs,
            refit=refit,
        )
        search.fit(X, y)
        results = {
            "best_params": dict(search.best_params_),
            "best_score": float(search.best_score_),
            "best_estimator": search.best_estimator_ if refit else None,
            "cv_results": _cv_results_to_records(search.cv_results_),
            "n_candidates": len(search.cv_results_["params"]),
            "execution_time": time.perf_counter() - start,
            "scoring": _scoring_name(self.scoring if scoring is None else scoring),
        }
        self.results_ = results
        self.logger.info(
            "Grid search: best %s = %.4f with %s",
            results["scoring"],
            results["best_score"],
            results["best_params"],
        )
        return results["best_params"], results

    def random_search(
        self,
        model: BaseEstimator,
        param_distributions: Dict[str, Any],
        X: ArrayLike,
        y: ArrayLike,
        n_iter: int = 20,
        cv: Any = 3,
        scoring: ScoringLike = None,
        early_stopping: bool = False,
        early_stopping_rounds: int = 5,
        early_stopping_threshold: Optional[float] = None,
    ) -> Tuple[ParamDict, Dict[str, Any]]:
        """Randomised search with optional early stopping.

        Args:
            model: Estimator template.
            param_distributions: Lists or scipy distributions per parameter.
            X: Feature matrix.
            y: Target vector.
            n_iter: Number of sampled configurations.
            cv: Folds or splitter.
            scoring: Scoring override.
            early_stopping: Enable early stopping.
            early_stopping_rounds: Stop after this many non-improving rounds.
            early_stopping_threshold: Stop once the best score reaches this.

        Returns:
            ``(best_params, results)`` with ``cv_results`` (evaluated
            candidates), ``best_score``, ``best_estimator``, ``early_stopped``
            and ``stopping_reason``.
        """
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            candidates = list(
                ParameterSampler(param_distributions, n_iter=n_iter, random_state=self.random_state)
            )
        splitter = _make_cv(cv, model, y, self.random_state)
        scorer = self._scoring(scoring)
        records: List[Dict[str, Any]] = []
        best_score, best_params = -np.inf, None
        stale = 0
        early_stopped = False
        stopping_reason = "completed"
        for iteration, params in enumerate(candidates):
            estimator = clone(model).set_params(**params)
            fold_scores = np.asarray(
                cross_val_score(estimator, X, y, cv=splitter, scoring=scorer, n_jobs=self.n_jobs)
            )
            mean = float(fold_scores.mean())
            records.append(
                {
                    "iteration": iteration,
                    "params": dict(params),
                    "mean_test_score": mean,
                    "std_test_score": float(fold_scores.std()),
                    "scores": fold_scores,
                }
            )
            if mean > best_score + 1e-12:
                best_score, best_params = mean, dict(params)
                stale = 0
            else:
                stale += 1
            if early_stopping:
                if early_stopping_threshold is not None and best_score >= early_stopping_threshold:
                    early_stopped, stopping_reason = True, "threshold_reached"
                    break
                if stale >= early_stopping_rounds:
                    early_stopped, stopping_reason = True, "no_improvement"
                    break
        if best_params is None:
            raise RuntimeError("Random search evaluated no candidates")
        best_estimator = clone(model).set_params(**best_params).fit(X, y)
        results = {
            "best_params": best_params,
            "best_score": best_score,
            "best_estimator": best_estimator,
            "cv_results": records,
            "n_candidates_evaluated": len(records),
            "early_stopped": early_stopped,
            "stopping_reason": stopping_reason,
            "execution_time": time.perf_counter() - start,
            "scoring": _scoring_name(self.scoring if scoring is None else scoring),
        }
        self.results_ = results
        return best_params, results

    def bayesian_optimization(
        self,
        model: BaseEstimator,
        param_space: Dict[str, Any],
        X: ArrayLike,
        y: ArrayLike,
        n_iterations: int = 20,
        cv: Any = 3,
        scoring: ScoringLike = None,
        **kwargs: Any,
    ) -> Tuple[ParamDict, Dict[str, Any]]:
        """Bayesian optimisation via :class:`BayesianOptimizer`.

        Args:
            model: Estimator template.
            param_space: Search space (see :meth:`BayesianOptimizer.optimize`).
            X: Feature matrix.
            y: Target vector.
            n_iterations: Number of evaluations.
            cv: Folds or splitter.
            scoring: Scoring override.
            **kwargs: Forwarded to :meth:`BayesianOptimizer.optimize`.

        Returns:
            ``(best_params, results)`` including ``optimization_history``.
        """
        optimizer = BayesianOptimizer(
            scoring=self.scoring if scoring is None else scoring,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        best_params, results = optimizer.optimize(
            model, param_space, X, y, n_iterations=n_iterations, cv=cv, **kwargs
        )
        self.results_ = results
        return best_params, results

    def multi_objective_optimization(
        self,
        model: BaseEstimator,
        param_grid: Dict[str, Sequence[Any]],
        X: ArrayLike,
        y: ArrayLike,
        objectives: Sequence[str] = ("accuracy", "model_size"),
        cv: Any = 3,
        max_candidates: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Evaluate a grid on several objectives and return the Pareto front.

        Objectives are scikit-learn scorer names (maximised) or the special
        names ``model_size``, ``model_complexity``, ``training_time``,
        ``prediction_time`` (minimised) and ``model_simplicity`` (maximised).

        Args:
            model: Estimator template.
            param_grid: Parameter grid.
            X: Feature matrix.
            y: Target vector.
            objectives: Objective names.
            cv: Folds or splitter for scoring objectives.
            max_candidates: Randomly subsample the grid to this many points.

        Returns:
            ``(pareto_solutions, results)`` where each solution has ``params``
            and ``objectives`` and ``results`` holds ``pareto_front``,
            ``all_solutions`` and ``trade_off_analysis``.
        """
        objectives = list(objectives)
        candidates = list(ParameterGrid(param_grid))
        if max_candidates is not None and len(candidates) > max_candidates:
            rng = np.random.default_rng(self.random_state)
            candidates = [
                candidates[i] for i in rng.choice(len(candidates), size=max_candidates, replace=False)
            ]
        solutions = []
        for params in candidates:
            values = _evaluate_objectives(model, params, X, y, objectives, cv, self.random_state, self.n_jobs)
            solutions.append({"params": dict(params), "objectives": values})
        points = np.array(
            [[sol["objectives"][o] * _objective_direction(o) for o in objectives] for sol in solutions]
        )
        mask = _pareto_mask(points)
        pareto = [sol for sol, keep in zip(solutions, mask) if keep]
        results = {
            "pareto_front": pareto,
            "all_solutions": solutions,
            "trade_off_analysis": _trade_off_analysis(solutions, pareto, objectives),
            "objectives": objectives,
            "directions": {o: "minimize" if _objective_direction(o) < 0 else "maximize" for o in objectives},
            "n_candidates": len(solutions),
            "n_pareto_optimal": len(pareto),
        }
        self.results_ = results
        return pareto, results


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
class CrossValidationPipeline(LoggerMixin):
    """Collection of cross-validation strategies with rich result dictionaries.

    Args:
        random_state: Seed for fold shuffling.
        n_jobs: Parallelism passed to scikit-learn.
        shuffle: Shuffle samples before splitting (stratified / k-fold).
    """

    def __init__(
        self, random_state: Optional[int] = None, n_jobs: Optional[int] = None, shuffle: bool = True
    ):
        self.random_state = _resolve_random_state(random_state)
        self.n_jobs = n_jobs
        self.shuffle = shuffle

    @staticmethod
    def _summarise(cv_results: Dict[str, Any], key: str = "test_score") -> Dict[str, Any]:
        scores = np.asarray(cv_results[key], dtype=float)
        return {
            "scores": scores,
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std()),
            "min_score": float(scores.min()),
            "max_score": float(scores.max()),
            "fit_times": np.asarray(cv_results["fit_time"], dtype=float),
            "score_times": np.asarray(cv_results["score_time"], dtype=float),
        }

    def stratified_cross_validation(
        self,
        model: BaseEstimator,
        X: ArrayLike,
        y: ArrayLike,
        cv: int = 5,
        scoring: ScoringLike = "accuracy",
        return_train_score: bool = False,
    ) -> Dict[str, Any]:
        """Stratified k-fold cross-validation with per-fold class distributions.

        Args:
            model: Estimator template.
            X: Feature matrix.
            y: Target labels.
            cv: Number of folds.
            scoring: Scoring specification.
            return_train_score: Also compute training-fold scores.

        Returns:
            Dictionary with ``scores``, ``mean_score``, ``std_score``,
            ``fold_distributions`` (class proportions in every test fold) and
            timing information.
        """
        splitter = StratifiedKFold(
            n_splits=cv, shuffle=self.shuffle, random_state=self.random_state if self.shuffle else None
        )
        cv_results = cross_validate(
            clone(model),
            X,
            y,
            cv=splitter,
            scoring=_resolve_scorer(scoring),
            n_jobs=self.n_jobs,
            return_train_score=return_train_score,
        )
        y_arr = np.asarray(y)
        classes = np.unique(y_arr)
        fold_distributions = []
        for _, test_idx in splitter.split(X, y):
            fold_y = y_arr[test_idx]
            fold_distributions.append({_to_python(c): float(np.mean(fold_y == c)) for c in classes})
        results = self._summarise(cv_results)
        results.update(
            {"fold_distributions": fold_distributions, "n_splits": cv, "scoring": _scoring_name(scoring)}
        )
        if return_train_score:
            results["train_scores"] = np.asarray(cv_results["train_score"], dtype=float)
        return results

    def time_series_cross_validation(
        self,
        model: BaseEstimator,
        X: ArrayLike,
        y: ArrayLike,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
        max_train_size: Optional[int] = None,
        scoring: ScoringLike = None,
    ) -> Dict[str, Any]:
        """Forward-chaining cross-validation for ordered data.

        Args:
            model: Estimator template.
            X: Feature matrix in temporal order.
            y: Target vector.
            n_splits: Number of splits.
            test_size: Size of each test window.
            gap: Samples excluded between train and test.
            max_train_size: Cap on the training window.
            scoring: Scoring specification (``None`` uses ``model.score``).

        Returns:
            Dictionary with ``scores``, ``mean_score``, ``std_score`` and
            ``fold_boundaries``.
        """
        splitter = TimeSeriesSplit(
            n_splits=n_splits, test_size=test_size, gap=gap, max_train_size=max_train_size
        )
        cv_results = cross_validate(
            clone(model), X, y, cv=splitter, scoring=_resolve_scorer(scoring), n_jobs=self.n_jobs
        )
        boundaries = [
            {
                "train_start": int(tr[0]),
                "train_end": int(tr[-1]),
                "test_start": int(te[0]),
                "test_end": int(te[-1]),
            }
            for tr, te in splitter.split(X)
        ]
        results = self._summarise(cv_results)
        results.update(
            {"fold_boundaries": boundaries, "n_splits": n_splits, "scoring": _scoring_name(scoring)}
        )
        return results

    def nested_cross_validation(
        self,
        model: BaseEstimator,
        param_grid: Dict[str, Sequence[Any]],
        X: ArrayLike,
        y: ArrayLike,
        inner_cv: Any = 3,
        outer_cv: Any = 3,
        scoring: ScoringLike = "accuracy",
        search: str = "grid",
        n_iter: int = 10,
    ) -> Dict[str, Any]:
        """Nested cross-validation for an unbiased estimate of tuned performance.

        Args:
            model: Estimator template.
            param_grid: Grid (or distributions when ``search="random"``).
            X: Feature matrix.
            y: Target vector.
            inner_cv: Folds for hyperparameter search.
            outer_cv: Folds for performance estimation.
            scoring: Scoring specification.
            search: ``"grid"`` or ``"random"``.
            n_iter: Iterations for random search.

        Returns:
            Dictionary with ``outer_scores``, ``best_params_per_fold``,
            ``inner_best_scores``, ``unbiased_score``, ``std_score`` and
            ``optimism`` (inner minus outer mean).
        """
        outer = _make_cv(outer_cv, model, y, self.random_state, shuffle=self.shuffle)
        inner = _make_cv(inner_cv, model, y, self.random_state + 1, shuffle=self.shuffle)
        scorer_spec = _resolve_scorer(scoring)
        outer_scores: List[float] = []
        best_params: List[ParamDict] = []
        inner_scores: List[float] = []
        for fold, (train_idx, test_idx) in enumerate(outer.split(X, y)):
            X_train, X_test = _take(X, train_idx), _take(X, test_idx)
            y_train, y_test = _take(y, train_idx), _take(y, test_idx)
            if search == "random":
                searcher: Any = RandomizedSearchCV(
                    clone(model),
                    param_grid,
                    n_iter=n_iter,
                    cv=inner,
                    scoring=scorer_spec,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                )
            else:
                searcher = GridSearchCV(
                    clone(model), param_grid, cv=inner, scoring=scorer_spec, n_jobs=self.n_jobs
                )
            searcher.fit(X_train, y_train)
            scorer = check_scoring(searcher.best_estimator_, scoring=scorer_spec)
            outer_scores.append(float(scorer(searcher.best_estimator_, X_test, y_test)))
            best_params.append(dict(searcher.best_params_))
            inner_scores.append(float(searcher.best_score_))
            self.logger.info("Outer fold %d: %.4f with %s", fold + 1, outer_scores[-1], best_params[-1])
        param_counts: Dict[str, int] = {}
        for params in best_params:
            key = repr(sorted(params.items()))
            param_counts[key] = param_counts.get(key, 0) + 1
        most_common = best_params[
            int(np.argmax([param_counts[repr(sorted(p.items()))] for p in best_params]))
        ]
        outer_arr = np.asarray(outer_scores)
        return {
            "outer_scores": outer_arr,
            "best_params_per_fold": best_params,
            "inner_best_scores": np.asarray(inner_scores),
            "unbiased_score": float(outer_arr.mean()),
            "std_score": float(outer_arr.std()),
            "optimism": float(np.mean(inner_scores) - outer_arr.mean()),
            "most_frequent_params": most_common,
            "scoring": _scoring_name(scoring),
        }

    def custom_cross_validation(
        self,
        model: BaseEstimator,
        X: ArrayLike,
        y: ArrayLike,
        cv_splitter: Any,
        groups: Optional[ArrayLike] = None,
        scoring: ScoringLike = None,
        fit_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Cross-validate with an arbitrary splitter (e.g. ``GroupKFold``).

        Args:
            model: Estimator template.
            X: Feature matrix.
            y: Target vector.
            cv_splitter: Any scikit-learn splitter or iterable of splits.
            groups: Group labels forwarded to the splitter.
            scoring: Scoring specification.
            fit_params: Extra parameters passed to ``fit``.

        Returns:
            Dictionary with ``scores``, ``mean_score``, ``std_score`` and timings.
        """
        cv_results = cross_validate(
            clone(model),
            X,
            y,
            groups=groups,
            cv=cv_splitter,
            scoring=_resolve_scorer(scoring),
            n_jobs=self.n_jobs,
            params=fit_params,
        )
        results = self._summarise(cv_results)
        results.update({"n_splits": len(results["scores"]), "scoring": _scoring_name(scoring)})
        return results

    def cross_validate_pipeline(
        self,
        pipeline: BaseEstimator,
        X: ArrayLike,
        y: ArrayLike,
        cv: Any = 5,
        scoring: Union[ScoringLike, Sequence[ScoringLike]] = ("accuracy",),
        return_train_score: bool = False,
    ) -> Dict[str, Any]:
        """Multi-metric cross-validation of a (preprocessing + model) pipeline.

        Args:
            pipeline: Estimator or :class:`~sklearn.pipeline.Pipeline`.
            X: Feature matrix.
            y: Target vector.
            cv: Folds or splitter.
            scoring: One or several scoring specifications.
            return_train_score: Also compute training-fold scores.

        Returns:
            Dictionary keyed by metric name, each holding ``scores``,
            ``mean_score`` and ``std_score``; plus ``fit_time``,
            ``score_time`` and ``n_splits``.
        """
        metrics = list(scoring) if isinstance(scoring, (list, tuple)) else [scoring]
        names = [_scoring_name(metric) for metric in metrics]
        scorers = {name: _resolve_scorer(metric) for name, metric in zip(names, metrics)}
        splitter = _make_cv(cv, pipeline, y, self.random_state, shuffle=self.shuffle)
        cv_results = cross_validate(
            clone(pipeline),
            X,
            y,
            cv=splitter,
            scoring=scorers,
            n_jobs=self.n_jobs,
            return_train_score=return_train_score,
        )
        results: Dict[str, Any] = {}
        for name in names:
            scores = np.asarray(cv_results[f"test_{name}"], dtype=float)
            entry = {"scores": scores, "mean_score": float(scores.mean()), "std_score": float(scores.std())}
            if return_train_score:
                entry["train_scores"] = np.asarray(cv_results[f"train_{name}"], dtype=float)
            results[name] = entry
        for key in ("fit_time", "score_time"):
            values = np.asarray(cv_results[key], dtype=float)
            results[key] = {
                "scores": values,
                "mean_score": float(values.mean()),
                "std_score": float(values.std()),
            }
        results["n_splits"] = len(cv_results["fit_time"])
        results["metrics"] = names
        return results


# --------------------------------------------------------------------------- #
# Ensembles
# --------------------------------------------------------------------------- #
class ModelEnsemblePipeline(LoggerMixin):
    """Factory for voting, stacking, bagging, dynamic and optimised ensembles.

    Args:
        random_state: Seed for bagging and weight optimisation.
        n_jobs: Parallelism passed to the ensemble estimators.
    """

    def __init__(self, random_state: Optional[int] = None, n_jobs: Optional[int] = None):
        self.random_state = _resolve_random_state(random_state)
        self.n_jobs = n_jobs
        self.optimization_results_: Dict[str, Any] = {}

    def create_voting_ensemble(
        self,
        base_models: Any,
        voting: str = "soft",
        weights: Optional[Sequence[float]] = None,
    ) -> BaseEstimator:
        """Build a :class:`VotingClassifier` / :class:`VotingRegressor`.

        Args:
            base_models: Dict, list of ``(name, estimator)`` or list of estimators.
            voting: ``"soft"`` or ``"hard"`` (classifiers only).  Soft voting
                silently degrades to hard voting when a member lacks
                ``predict_proba``.
            weights: Optional member weights.

        Returns:
            An unfitted voting ensemble.
        """
        estimators = [(name, clone(est)) for name, est in _normalise_estimators(base_models)]
        if not estimators:
            raise ValueError("base_models must not be empty")
        if is_classifier(estimators[0][1]):
            if voting == "soft" and not all(hasattr(est, "predict_proba") for _, est in estimators):
                self.logger.warning("Not every member supports predict_proba; falling back to hard voting")
                voting = "hard"
            return VotingClassifier(estimators=estimators, voting=voting, weights=weights, n_jobs=self.n_jobs)
        return VotingRegressor(estimators=estimators, weights=weights, n_jobs=self.n_jobs)

    def create_stacking_ensemble(
        self,
        base_models: Any,
        meta_model: Optional[BaseEstimator] = None,
        cv: Any = 3,
        passthrough: bool = False,
    ) -> BaseEstimator:
        """Build a :class:`StackingClassifier` / :class:`StackingRegressor`.

        Args:
            base_models: Dict, list of ``(name, estimator)`` or list of estimators.
            meta_model: Final estimator; scikit-learn's default when ``None``.
            cv: Folds used to build out-of-fold meta-features.
            passthrough: Also feed raw features to the meta model.

        Returns:
            An unfitted stacking ensemble.
        """
        estimators = [(name, clone(est)) for name, est in _normalise_estimators(base_models)]
        if not estimators:
            raise ValueError("base_models must not be empty")
        final = clone(meta_model) if meta_model is not None else None
        cls = StackingClassifier if is_classifier(estimators[0][1]) else StackingRegressor
        return cls(
            estimators=estimators, final_estimator=final, cv=cv, passthrough=passthrough, n_jobs=self.n_jobs
        )

    def create_bagging_ensemble(
        self,
        base_model: BaseEstimator,
        n_estimators: int = 10,
        random_state: Optional[int] = None,
        max_samples: Union[int, float] = 1.0,
        max_features: Union[int, float] = 1.0,
        bootstrap: bool = True,
    ) -> BaseEstimator:
        """Build a :class:`BaggingClassifier` / :class:`BaggingRegressor`.

        Args:
            base_model: Estimator to bag.
            n_estimators: Number of bootstrap members.
            random_state: Seed (defaults to the pipeline seed).
            max_samples: Samples drawn per member.
            max_features: Features drawn per member.
            bootstrap: Sample with replacement.

        Returns:
            An unfitted bagging ensemble.
        """
        cls = BaggingClassifier if is_classifier(base_model) else BaggingRegressor
        return cls(
            estimator=clone(base_model),
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=self.random_state if random_state is None else random_state,
            n_jobs=self.n_jobs,
        )

    def create_dynamic_ensemble(
        self,
        base_models: Any,
        selection_strategy: str = "best_local_accuracy",
        k_neighbors: int = 7,
        X_val: Optional[ArrayLike] = None,
        y_val: Optional[ArrayLike] = None,
    ) -> DynamicEnsembleSelector:
        """Build a :class:`DynamicEnsembleSelector` over fitted classifiers.

        Args:
            base_models: Fitted classifiers (dict or list).
            selection_strategy: See :class:`DynamicEnsembleSelector`.
            k_neighbors: Competence-region size.
            X_val: Optional validation features defining competence regions.
            y_val: Validation labels.

        Returns:
            The selector (fitted on the validation set when one is given).
        """
        estimators = _normalise_estimators(base_models)
        unfitted = [name for name, est in estimators if not _is_fitted(est)]
        if unfitted:
            raise ValueError(f"Dynamic ensembles need fitted members; unfitted: {unfitted}")
        selector = DynamicEnsembleSelector(
            estimators, selection_strategy=selection_strategy, k_neighbors=k_neighbors
        )
        if X_val is not None and y_val is not None:
            selector.fit(X_val, y_val)
        else:
            self.logger.info(
                "No validation set given; dynamic ensemble will fall back to soft majority voting"
            )
        return selector

    # ------------------------------------------------------------------ #
    def _out_of_fold_predictions(
        self,
        estimators: List[Tuple[str, BaseEstimator]],
        X: ArrayLike,
        y: ArrayLike,
        cv: Any,
        classification: bool,
        classes: np.ndarray,
    ) -> np.ndarray:
        splitter = _make_cv(cv, estimators[0][1], y, self.random_state)
        stack = []
        for _, estimator in estimators:
            if classification:
                if hasattr(estimator, "predict_proba"):
                    proba = cross_val_predict(
                        clone(estimator), X, y, cv=splitter, method="predict_proba", n_jobs=self.n_jobs
                    )
                else:
                    labels = cross_val_predict(clone(estimator), X, y, cv=splitter, n_jobs=self.n_jobs)
                    proba = np.eye(len(classes))[_class_index(classes, labels)]
                stack.append(np.asarray(proba, dtype=float))
            else:
                stack.append(
                    np.asarray(
                        cross_val_predict(clone(estimator), X, y, cv=splitter, n_jobs=self.n_jobs),
                        dtype=float,
                    )[:, None]
                )
        return np.stack(stack)  # (n_estimators, n_samples, n_outputs)

    @staticmethod
    def _genetic_weights(
        fitness: Callable[[np.ndarray], float],
        n_members: int,
        population_size: int,
        n_generations: int,
        rng: np.random.Generator,
        mutation_rate: float = 0.2,
    ) -> Tuple[np.ndarray, List[float]]:
        """Simple real-valued genetic algorithm over the weight simplex."""
        population = rng.dirichlet(np.ones(n_members), size=population_size)
        population[0] = np.ones(n_members) / n_members  # always include uniform weights
        history: List[float] = []
        elite_count = max(1, population_size // 5)
        for _ in range(n_generations):
            scores = np.array([fitness(w) for w in population])
            history.append(float(scores.max()))
            order = np.argsort(-scores)
            elites = population[order[:elite_count]]
            children = [*elites]
            while len(children) < population_size:
                a, b = rng.choice(order[: max(2, population_size // 2)], size=2, replace=False)
                alpha = rng.uniform(0.0, 1.0)
                child = alpha * population[a] + (1.0 - alpha) * population[b]
                if rng.uniform() < mutation_rate:
                    child = child + rng.normal(0.0, 0.1, size=n_members)
                child = np.clip(child, 0.0, None)
                child = child / child.sum() if child.sum() > 0 else np.ones(n_members) / n_members
                children.append(child)
            population = np.array(children)
        scores = np.array([fitness(w) for w in population])
        history.append(float(scores.max()))
        return population[int(np.argmax(scores))], history

    @staticmethod
    def _greedy_weights(
        fitness: Callable[[np.ndarray], float],
        n_members: int,
        n_rounds: int,
    ) -> Tuple[np.ndarray, List[float]]:
        """Caruana-style forward selection with replacement."""
        counts = np.zeros(n_members)
        history: List[float] = []
        best_overall = -np.inf
        best_counts = counts.copy()
        for _ in range(n_rounds):
            candidate_scores = []
            for member in range(n_members):
                trial = counts.copy()
                trial[member] += 1
                candidate_scores.append(fitness(trial / trial.sum()))
            chosen = int(np.argmax(candidate_scores))
            counts[chosen] += 1
            history.append(float(candidate_scores[chosen]))
            if candidate_scores[chosen] > best_overall:
                best_overall = candidate_scores[chosen]
                best_counts = counts.copy()
        return best_counts / best_counts.sum(), history

    def optimize_ensemble(
        self,
        base_models: Any,
        X: ArrayLike,
        y: ArrayLike,
        optimization_method: str = "genetic_algorithm",
        cv: Any = 3,
        population_size: int = 20,
        n_generations: int = 15,
        selection_threshold: float = 1e-3,
    ) -> BaseEstimator:
        """Learn member weights on out-of-fold predictions.

        Args:
            base_models: Dict, list of ``(name, estimator)`` or list of estimators.
            X: Feature matrix.
            y: Target vector.
            optimization_method: ``"genetic_algorithm"``, ``"greedy"`` or
                ``"uniform"``.
            cv: Folds used to produce out-of-fold predictions.
            population_size: GA population size.
            n_generations: GA generations (or greedy rounds).
            selection_threshold: Members with a smaller weight are dropped.

        Returns:
            An unfitted :class:`WeightedEnsembleClassifier` /
            :class:`WeightedEnsembleRegressor` whose ``fit`` exposes
            ``weights_`` and ``selected_models_``.
        """
        estimators = _normalise_estimators(base_models)
        if not estimators:
            raise ValueError("base_models must not be empty")
        classification = is_classifier(estimators[0][1])
        y_arr = np.asarray(y)
        classes = np.unique(y_arr) if classification else np.array([])
        oof = self._out_of_fold_predictions(estimators, X, y, cv, classification, classes)
        n_members = len(estimators)
        if classification:
            y_idx = _class_index(classes, y_arr)

            def fitness(weights: np.ndarray) -> float:
                combined = np.tensordot(weights, oof, axes=1)
                return float(np.mean(np.argmax(combined, axis=1) == y_idx))

        else:
            target = y_arr.astype(float)

            def fitness(weights: np.ndarray) -> float:
                combined = np.tensordot(weights, oof, axes=1)[:, 0]
                return float(r2_score(target, combined))

        rng = np.random.default_rng(self.random_state)
        method = optimization_method.lower()
        if method in ("genetic_algorithm", "genetic", "ga"):
            weights, history = self._genetic_weights(fitness, n_members, population_size, n_generations, rng)
        elif method in ("greedy", "caruana", "forward_selection"):
            weights, history = self._greedy_weights(fitness, n_members, max(n_generations, n_members))
        elif method == "uniform":
            weights, history = np.ones(n_members) / n_members, []
        else:
            raise ValueError(f"Unknown optimization_method '{optimization_method}'")
        self.optimization_results_ = {
            "method": method,
            "weights": dict(zip([name for name, _ in estimators], weights.tolist())),
            "oof_score": fitness(weights),
            "uniform_oof_score": fitness(np.ones(n_members) / n_members),
            "history": history,
        }
        self.logger.info(
            "Optimised ensemble weights: %s (OOF score %.4f)",
            self.optimization_results_["weights"],
            self.optimization_results_["oof_score"],
        )
        members = [(name, clone(est)) for name, est in estimators]
        cls = WeightedEnsembleClassifier if classification else WeightedEnsembleRegressor
        return cls(
            estimators=members,
            weights=weights.tolist(),
            selection_threshold=selection_threshold,
            n_jobs=self.n_jobs,
        )


# --------------------------------------------------------------------------- #
# Experiment tracking
# --------------------------------------------------------------------------- #
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PerformanceTracker(LoggerMixin):
    """In-memory experiment, comparison and performance-history tracker.

    The tracker is deliberately dependency-free; :meth:`to_dataframe` and
    :meth:`export` make it easy to hand the data to external tools.
    """

    def __init__(self):
        self.experiments_: Dict[str, Dict[str, Any]] = {}
        self.comparisons_: Dict[str, Dict[str, Any]] = {}
        self.performance_history_: Dict[str, List[Dict[str, Any]]] = {}
        self.monitors_: Dict[str, Dict[str, Any]] = {}
        self.alerts_: List[Dict[str, Any]] = []

    # -- experiments ---------------------------------------------------- #
    def start_experiment(
        self,
        name: str,
        description: str = "",
        tags: Optional[Sequence[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create an experiment and return its identifier.

        Args:
            name: Experiment name.
            description: Free-text description.
            tags: Optional tags.
            parameters: Initial parameters.

        Returns:
            The experiment id.
        """
        experiment_id = uuid.uuid4().hex
        self.experiments_[experiment_id] = {
            "id": experiment_id,
            "name": name,
            "description": description,
            "tags": list(tags or []),
            "parameters": dict(parameters or {}),
            "metrics": {},
            "metric_history": {},
            "status": "running",
            "start_time": _utc_now(),
            "end_time": None,
        }
        self.logger.info("Started experiment '%s' (%s)", name, experiment_id)
        return experiment_id

    def _experiment(self, experiment_id: str) -> Dict[str, Any]:
        try:
            return self.experiments_[experiment_id]
        except KeyError as exc:
            raise KeyError(f"Unknown experiment id '{experiment_id}'") from exc

    def log_metric(self, experiment_id: str, name: str, value: float, step: Optional[int] = None) -> None:
        """Record a metric value (the latest value is kept in ``metrics``).

        Args:
            experiment_id: Experiment identifier.
            name: Metric name.
            value: Metric value.
            step: Optional step index.
        """
        experiment = self._experiment(experiment_id)
        experiment["metrics"][name] = float(value)
        experiment["metric_history"].setdefault(name, []).append(
            {"value": float(value), "step": step, "timestamp": _utc_now()}
        )

    def log_metrics(self, experiment_id: str, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Record several metrics at once."""
        for name, value in metrics.items():
            self.log_metric(experiment_id, name, value, step=step)

    def log_parameter(self, experiment_id: str, name: str, value: Any) -> None:
        """Record a parameter value."""
        self._experiment(experiment_id)["parameters"][name] = value

    def log_parameters(self, experiment_id: str, parameters: Dict[str, Any]) -> None:
        """Record several parameters at once."""
        self._experiment(experiment_id)["parameters"].update(parameters)

    def end_experiment(self, experiment_id: str, status: str = "completed") -> Dict[str, Any]:
        """Mark an experiment as finished and return its record."""
        experiment = self._experiment(experiment_id)
        experiment["status"] = status
        experiment["end_time"] = _utc_now()
        return experiment

    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Return a copy of the experiment record."""
        return copy.deepcopy(self._experiment(experiment_id))

    def list_experiments(self, tags: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        """List experiments, optionally filtered to those carrying all ``tags``."""
        wanted = set(tags or [])
        return [copy.deepcopy(exp) for exp in self.experiments_.values() if wanted.issubset(exp["tags"])]

    def compare_experiments(
        self,
        experiment_ids: Sequence[str],
        metric: str = "accuracy",
        higher_is_better: bool = True,
    ) -> Dict[str, Any]:
        """Rank experiments by a metric.

        Args:
            experiment_ids: Experiments to compare.
            metric: Metric used for ranking.
            higher_is_better: Ranking direction.

        Returns:
            Dictionary with ``experiments`` (id, name, value), ``ranking`` and
            ``best_experiment``.
        """
        rows = []
        for experiment_id in experiment_ids:
            experiment = self._experiment(experiment_id)
            rows.append(
                {
                    "id": experiment_id,
                    "name": experiment["name"],
                    "metric": metric,
                    "value": experiment["metrics"].get(metric),
                    "parameters": dict(experiment["parameters"]),
                }
            )
        scored = [row for row in rows if row["value"] is not None]
        ranking = sorted(scored, key=lambda row: row["value"], reverse=higher_is_better)
        for rank, row in enumerate(ranking, start=1):
            row["rank"] = rank
        return {
            "metric": metric,
            "experiments": rows,
            "ranking": ranking,
            "best_experiment": ranking[0] if ranking else None,
            "missing_metric": [row["id"] for row in rows if row["value"] is None],
        }

    # -- model comparisons --------------------------------------------- #
    def start_model_comparison(
        self, name: str, primary_metric: Optional[str] = None, higher_is_better: bool = True
    ) -> str:
        """Open a model-comparison record and return its identifier.

        Args:
            name: Comparison name.
            primary_metric: Metric used to pick the best model (defaults to
                the first metric logged).
            higher_is_better: Direction of the primary metric.

        Returns:
            The comparison id.
        """
        comparison_id = uuid.uuid4().hex
        self.comparisons_[comparison_id] = {
            "id": comparison_id,
            "name": name,
            "models": {},
            "primary_metric": primary_metric,
            "higher_is_better": higher_is_better,
            "status": "running",
            "start_time": _utc_now(),
            "end_time": None,
            "best_model": None,
        }
        return comparison_id

    def _comparison(self, comparison_id: str) -> Dict[str, Any]:
        try:
            return self.comparisons_[comparison_id]
        except KeyError as exc:
            raise KeyError(f"Unknown comparison id '{comparison_id}'") from exc

    def log_model_performance(
        self,
        comparison_id: str,
        model_name: str,
        metrics: Dict[str, float],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a model's metrics inside a comparison."""
        comparison = self._comparison(comparison_id)
        if comparison["primary_metric"] is None and metrics:
            comparison["primary_metric"] = next(iter(metrics))
        comparison["models"][model_name] = {
            "name": model_name,
            "metrics": {k: float(v) for k, v in metrics.items()},
            "parameters": dict(parameters or {}),
            "timestamp": _utc_now(),
        }

    def end_model_comparison(self, comparison_id: str) -> Dict[str, Any]:
        """Close a comparison, compute the ranking and return the record."""
        comparison = self._comparison(comparison_id)
        comparison["status"] = "completed"
        comparison["end_time"] = _utc_now()
        metric = comparison["primary_metric"]
        candidates = [m for m in comparison["models"].values() if metric in m["metrics"]]
        ranking = sorted(
            candidates, key=lambda m: m["metrics"][metric], reverse=comparison["higher_is_better"]
        )
        comparison["ranking"] = [m["name"] for m in ranking]
        comparison["best_model"] = ranking[0] if ranking else None
        return comparison

    def get_model_comparison(self, comparison_id: str) -> Dict[str, Any]:
        """Return a copy of the comparison record (ranking is refreshed)."""
        comparison = self._comparison(comparison_id)
        if comparison["status"] != "completed":
            metric = comparison["primary_metric"]
            candidates = [m for m in comparison["models"].values() if metric in m["metrics"]]
            ranking = sorted(
                candidates, key=lambda m: m["metrics"][metric], reverse=comparison["higher_is_better"]
            )
            comparison["ranking"] = [m["name"] for m in ranking]
            comparison["best_model"] = ranking[0] if ranking else None
        return copy.deepcopy(comparison)

    # -- performance history ------------------------------------------- #
    def log_performance_snapshot(
        self,
        model_name: str,
        metrics: Dict[str, float],
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a dated performance snapshot for ``model_name``."""
        self.performance_history_.setdefault(model_name, []).append(
            {
                "timestamp": timestamp or _utc_now(),
                "metrics": {k: float(v) for k, v in metrics.items()},
                "metadata": dict(metadata or {}),
            }
        )

    def get_performance_history(self, model_name: str) -> List[Dict[str, Any]]:
        """Return the snapshots recorded for ``model_name`` in insertion order."""
        return copy.deepcopy(self.performance_history_.get(model_name, []))

    def detect_performance_drift(
        self, model_name: str, metric: str, window: int = 3, tolerance: float = 0.0
    ) -> Dict[str, Any]:
        """Compare the latest ``window`` snapshots with the preceding ones.

        Args:
            model_name: Model whose history is analysed.
            metric: Metric to compare.
            window: Number of most recent snapshots.
            tolerance: Minimum drop treated as drift.

        Returns:
            Dictionary with ``drift_detected``, ``recent_mean``,
            ``baseline_mean`` and ``change``.
        """
        values = [
            snap["metrics"][metric]
            for snap in self.performance_history_.get(model_name, [])
            if metric in snap["metrics"]
        ]
        if len(values) <= window:
            return {"drift_detected": False, "recent_mean": None, "baseline_mean": None, "change": None}
        recent = float(np.mean(values[-window:]))
        baseline = float(np.mean(values[:-window]))
        change = recent - baseline
        return {
            "drift_detected": change < -abs(tolerance),
            "recent_mean": recent,
            "baseline_mean": baseline,
            "change": change,
        }

    # -- monitoring ---------------------------------------------------- #
    def setup_performance_monitoring(
        self,
        model_name: str,
        alert_threshold: Dict[str, float],
        alert_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        higher_is_better: bool = True,
    ) -> None:
        """Register minimum acceptable metric values for ``model_name``.

        Args:
            model_name: Monitored model.
            alert_threshold: Mapping metric -> threshold.
            alert_callback: Called with an alert dictionary on every breach.
            higher_is_better: Whether values below the threshold are breaches.
        """
        self.monitors_[model_name] = {
            "thresholds": {k: float(v) for k, v in alert_threshold.items()},
            "callback": alert_callback,
            "higher_is_better": higher_is_better,
        }

    def check_performance_alert(self, model_name: str, metrics: Dict[str, float]) -> bool:
        """Check ``metrics`` against the registered thresholds.

        Args:
            model_name: Monitored model.
            metrics: Observed metric values.

        Returns:
            ``True`` when at least one threshold is breached.

        Raises:
            KeyError: If no monitor was set up for ``model_name``.
        """
        if model_name not in self.monitors_:
            raise KeyError(f"No performance monitoring configured for '{model_name}'")
        monitor = self.monitors_[model_name]
        breaches = {}
        for metric, threshold in monitor["thresholds"].items():
            if metric not in metrics:
                continue
            value = float(metrics[metric])
            breached = value < threshold if monitor["higher_is_better"] else value > threshold
            if breached:
                breaches[metric] = {"value": value, "threshold": threshold}
        if not breaches:
            return False
        alert = {"model_name": model_name, "breaches": breaches, "timestamp": _utc_now()}
        self.alerts_.append(alert)
        self.logger.warning("Performance alert for %s: %s", model_name, breaches)
        if monitor["callback"] is not None:
            monitor["callback"](alert)
        return True

    # -- export -------------------------------------------------------- #
    def to_dataframe(self) -> pd.DataFrame:
        """Return all experiments as a flat DataFrame."""
        rows = []
        for experiment in self.experiments_.values():
            row = {"id": experiment["id"], "name": experiment["name"], "status": experiment["status"]}
            row.update({f"param_{k}": v for k, v in experiment["parameters"].items()})
            row.update({f"metric_{k}": v for k, v in experiment["metrics"].items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def export(self) -> Dict[str, Any]:
        """Return a deep copy of all tracked data."""
        return copy.deepcopy(
            {
                "experiments": self.experiments_,
                "comparisons": self.comparisons_,
                "performance_history": self.performance_history_,
                "alerts": self.alerts_,
            }
        )


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
def _version_key(version: str) -> Tuple[Any, ...]:
    """Sort key that orders semantic versions numerically."""
    parts: List[Any] = []
    for token in str(version).replace("-", ".").split("."):
        parts.append((0, int(token)) if token.isdigit() else (1, token))
    return tuple(parts)


class ModelRegistry(LoggerMixin):
    """Versioned model registry with deployment and performance tracking.

    Models are kept in memory (deep-copied on registration).  When
    ``storage_dir`` is given, every registered model is also persisted with
    :mod:`joblib` so the registry survives process restarts.

    Args:
        storage_dir: Optional directory for persisted models.
    """

    def __init__(self, storage_dir: Optional[Union[str, Path]] = None):
        self.storage_dir = Path(storage_dir) if storage_dir is not None else None
        if self.storage_dir is not None:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.models_: Dict[str, Dict[str, Any]] = {}
        self.deployments_: Dict[str, Dict[str, Any]] = {}
        self.performance_: Dict[str, List[Dict[str, Any]]] = {}

    def register_model(
        self,
        model: BaseEstimator,
        name: str,
        version: str = "1.0.0",
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> str:
        """Register a model version and return its identifier.

        Args:
            model: (Fitted) estimator.
            name: Model name shared across versions.
            version: Version string (semantic versions sort correctly).
            description: Free-text description.
            metadata: Arbitrary metadata.
            tags: Tags used by :meth:`search_models`.

        Returns:
            The model id.

        Raises:
            ValueError: If ``name``/``version`` is already registered.
        """
        for record in self.models_.values():
            if record["name"] == name and record["version"] == version:
                raise ValueError(f"Model '{name}' version '{version}' is already registered")
        model_id = uuid.uuid4().hex
        record = {
            "id": model_id,
            "name": name,
            "version": version,
            "description": description,
            "metadata": dict(metadata or {}),
            "tags": list(tags or []),
            "estimator_type": type(model).__name__,
            "hyperparameters": model.get_params(deep=False) if hasattr(model, "get_params") else {},
            "registered_at": _utc_now(),
            "status": "registered",
            "model": copy.deepcopy(model),
            "path": None,
        }
        if self.storage_dir is not None and HAS_JOBLIB:
            path = self.storage_dir / f"{model_id}.joblib"
            joblib.dump(model, path)
            record["path"] = str(path)
        self.models_[model_id] = record
        self.performance_[model_id] = []
        self.logger.info("Registered model '%s' v%s as %s", name, version, model_id)
        return model_id

    def _record(self, model_id: str) -> Dict[str, Any]:
        try:
            return self.models_[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown model id '{model_id}'") from exc

    def get_model(self, model_id: str) -> BaseEstimator:
        """Return the registered estimator (loading from disk if needed)."""
        record = self._record(model_id)
        if record["model"] is None and record["path"] and HAS_JOBLIB:
            record["model"] = joblib.load(record["path"])
        return record["model"]

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Return the model's metadata record without the estimator."""
        record = self._record(model_id)
        return {k: copy.deepcopy(v) for k, v in record.items() if k != "model"}

    def get_model_versions(self, name: str) -> List[Dict[str, Any]]:
        """Return all versions of ``name`` sorted from oldest to newest."""
        versions = [self.get_model_info(mid) for mid, rec in self.models_.items() if rec["name"] == name]
        return sorted(versions, key=lambda rec: _version_key(rec["version"]))

    def get_latest_model(self, name: str) -> BaseEstimator:
        """Return the estimator of the highest version of ``name``."""
        versions = self.get_model_versions(name)
        if not versions:
            raise KeyError(f"No model registered under name '{name}'")
        return self.get_model(versions[-1]["id"])

    def list_models(self) -> List[Dict[str, Any]]:
        """Return metadata for every registered model."""
        return [self.get_model_info(mid) for mid in self.models_]

    def search_models(
        self,
        tags: Optional[Sequence[str]] = None,
        name_pattern: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find models by tags (all must match), glob name pattern, metadata or status.

        Args:
            tags: Tags every result must carry.
            name_pattern: ``fnmatch`` pattern (e.g. ``"classification*"``).
            metadata_filter: Key/value pairs that must match exactly.
            status: Required status (``"registered"``, ``"deployed"``, ...).

        Returns:
            Matching metadata records.
        """
        wanted = set(tags or [])
        results = []
        for record in self.models_.values():
            if wanted and not wanted.issubset(record["tags"]):
                continue
            if name_pattern is not None and not fnmatch.fnmatch(record["name"], name_pattern):
                continue
            if metadata_filter and any(record["metadata"].get(k) != v for k, v in metadata_filter.items()):
                continue
            if status is not None and record["status"] != status:
                continue
            results.append(self.get_model_info(record["id"]))
        return results

    # -- deployments --------------------------------------------------- #
    def deploy_model(
        self,
        model_id: str,
        environment: str,
        endpoint: Optional[str] = None,
        deployment_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record a deployment of ``model_id`` and return the deployment id."""
        record = self._record(model_id)
        deployment_id = uuid.uuid4().hex
        self.deployments_[deployment_id] = {
            "id": deployment_id,
            "model_id": model_id,
            "model_name": record["name"],
            "version": record["version"],
            "environment": environment,
            "endpoint": endpoint,
            "config": dict(deployment_config or {}),
            "status": "deployed",
            "deployed_at": _utc_now(),
            "undeployed_at": None,
        }
        record["status"] = "deployed"
        self.logger.info("Deployed %s v%s to %s", record["name"], record["version"], environment)
        return deployment_id

    def get_deployment_info(self, deployment_id: str) -> Dict[str, Any]:
        """Return a copy of the deployment record."""
        try:
            return copy.deepcopy(self.deployments_[deployment_id])
        except KeyError as exc:
            raise KeyError(f"Unknown deployment id '{deployment_id}'") from exc

    def undeploy_model(self, deployment_id: str) -> Dict[str, Any]:
        """Mark a deployment as retired."""
        deployment = self.deployments_[deployment_id]
        deployment["status"] = "undeployed"
        deployment["undeployed_at"] = _utc_now()
        model_id = deployment["model_id"]
        still_live = any(
            d["model_id"] == model_id and d["status"] == "deployed" for d in self.deployments_.values()
        )
        if not still_live:
            self.models_[model_id]["status"] = "registered"
        return copy.deepcopy(deployment)

    def list_deployments(
        self, environment: Optional[str] = None, active_only: bool = False
    ) -> List[Dict[str, Any]]:
        """List deployments, optionally filtered by environment / status."""
        return [
            copy.deepcopy(d)
            for d in self.deployments_.values()
            if (environment is None or d["environment"] == environment)
            and (not active_only or d["status"] == "deployed")
        ]

    # -- performance --------------------------------------------------- #
    def log_model_performance(
        self,
        model_id: str,
        metrics: Dict[str, float],
        dataset: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """Append a performance record for ``model_id``."""
        self._record(model_id)
        self.performance_[model_id].append(
            {
                "metrics": {k: float(v) for k, v in metrics.items()},
                "dataset": dataset,
                "timestamp": timestamp or _utc_now(),
            }
        )

    def get_performance_history(self, model_id: str) -> List[Dict[str, Any]]:
        """Return performance records for ``model_id`` in insertion order."""
        self._record(model_id)
        return copy.deepcopy(self.performance_[model_id])

    def delete_model(self, model_id: str) -> None:
        """Remove a model (and its persisted file) from the registry."""
        record = self.models_.pop(model_id)
        self.performance_.pop(model_id, None)
        if record["path"]:
            Path(record["path"]).unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Bayesian optimisation
# --------------------------------------------------------------------------- #
class BayesianOptimizer(LoggerMixin):
    """Sequential model-based hyperparameter optimisation.

    Two backends are available:

    * ``"optuna"`` -- Tree-structured Parzen Estimator via Optuna (used by
      default when Optuna is installed).
    * ``"gp"`` -- a Gaussian-process surrogate built on
      :class:`~sklearn.gaussian_process.GaussianProcessRegressor` with
      expected-improvement, probability-of-improvement or UCB acquisition.
      Available without any optional dependency.

    ``"random"`` performs plain random search and is mostly useful as a
    baseline.  Search spaces are dictionaries whose values are ``(low, high)``
    tuples (int or float ranges), lists (categorical choices) or explicit
    ``{"type": ..., ...}`` dictionaries.

    Args:
        scoring: Default scoring specification.
        random_state: Seed for sampling.
        n_jobs: Parallelism for cross-validation.
        backend: ``"auto"``, ``"optuna"``, ``"gp"`` or ``"random"``.
        n_initial_points: Random evaluations before the surrogate is used.
        n_candidates: Random candidates scored by the acquisition function
            (GP backend).
        exploration_weight: ``xi`` for EI / PI acquisition.
        ucb_kappa: Exploration constant for the UCB acquisition.
    """

    _BACKENDS = ("auto", "optuna", "gp", "random")

    def __init__(
        self,
        scoring: ScoringLike = "accuracy",
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        backend: str = "auto",
        n_initial_points: int = 5,
        n_candidates: int = 1000,
        exploration_weight: float = 0.01,
        ucb_kappa: float = 2.0,
    ):
        if backend not in self._BACKENDS:
            raise ValueError(f"backend must be one of {self._BACKENDS}, got '{backend}'")
        self.scoring = scoring
        self.random_state = _resolve_random_state(random_state)
        self.n_jobs = n_jobs
        self.backend = backend
        self.n_initial_points = n_initial_points
        self.n_candidates = n_candidates
        self.exploration_weight = exploration_weight
        self.ucb_kappa = ucb_kappa
        self.results_: Dict[str, Any] = {}

    # -- search-space handling ------------------------------------------ #
    def resolve_backend(self, backend: Optional[str] = None) -> str:
        """Resolve ``"auto"`` to a concrete backend."""
        backend = backend or self.backend
        if backend == "auto":
            return "optuna" if HAS_OPTUNA else "gp"
        if backend == "optuna" and not HAS_OPTUNA:
            raise ImportError("backend='optuna' requires the optuna package (pip install optuna)")
        return backend

    @staticmethod
    def parse_search_space(search_space: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Normalise a search space into typed parameter descriptions.

        Args:
            search_space: Mapping parameter -> ``(low, high)``, list of
                choices or ``{"type": "int"|"float"|"categorical", ...}``.

        Returns:
            Mapping parameter -> ``{"type", "low", "high", "log"}`` or
            ``{"type": "categorical", "choices"}``.

        Raises:
            ValueError: If a specification cannot be interpreted.
        """
        parsed: Dict[str, Dict[str, Any]] = {}
        for name, spec in search_space.items():
            if isinstance(spec, dict):
                kind = spec.get("type", "float")
                if kind == "categorical":
                    parsed[name] = {"type": "categorical", "choices": list(spec["choices"])}
                elif kind in ("int", "float", "loguniform", "log"):
                    log = kind in ("loguniform", "log") or bool(spec.get("log", False))
                    parsed[name] = {
                        "type": "int" if kind == "int" else "float",
                        "low": spec["low"],
                        "high": spec["high"],
                        "log": log,
                    }
                else:
                    raise ValueError(f"Unknown parameter type '{kind}' for '{name}'")
            elif isinstance(spec, tuple) and len(spec) in (2, 3) and all(_is_number(v) for v in spec[:2]):
                low, high = spec[0], spec[1]
                log = len(spec) == 3 and str(spec[2]).lower() in ("log", "log-uniform", "loguniform")
                if _is_integer(low) and _is_integer(high):
                    parsed[name] = {"type": "int", "low": int(low), "high": int(high), "log": log}
                else:
                    parsed[name] = {"type": "float", "low": float(low), "high": float(high), "log": log}
                if parsed[name]["low"] > parsed[name]["high"]:
                    raise ValueError(f"Invalid range for '{name}': low > high")
            elif isinstance(spec, (list, tuple, set, np.ndarray)):
                parsed[name] = {"type": "categorical", "choices": list(spec)}
            else:
                raise ValueError(f"Cannot interpret search-space entry for '{name}': {spec!r}")
        return parsed

    @staticmethod
    def _sample_random(space: Dict[str, Dict[str, Any]], rng: np.random.Generator) -> ParamDict:
        params: ParamDict = {}
        for name, spec in space.items():
            if spec["type"] == "categorical":
                params[name] = spec["choices"][int(rng.integers(len(spec["choices"])))]
            elif spec["type"] == "int":
                if spec["log"]:
                    value = round(np.exp(rng.uniform(np.log(spec["low"]), np.log(spec["high"]))))
                else:
                    value = int(rng.integers(spec["low"], spec["high"] + 1))
                params[name] = int(np.clip(value, spec["low"], spec["high"]))
            else:
                if spec["log"]:
                    params[name] = float(np.exp(rng.uniform(np.log(spec["low"]), np.log(spec["high"]))))
                else:
                    params[name] = float(rng.uniform(spec["low"], spec["high"]))
        return params

    @staticmethod
    def _encode(space: Dict[str, Dict[str, Any]], params: ParamDict) -> np.ndarray:
        """Map parameters to the unit hypercube."""
        encoded = []
        for name, spec in space.items():
            value = params[name]
            if spec["type"] == "categorical":
                choices = spec["choices"]
                index = choices.index(value) if value in choices else 0
                encoded.append(index / max(len(choices) - 1, 1))
            elif spec["log"]:
                span = np.log(spec["high"]) - np.log(spec["low"])
                encoded.append((np.log(max(value, 1e-12)) - np.log(spec["low"])) / span if span > 0 else 0.0)
            else:
                span = spec["high"] - spec["low"]
                encoded.append((value - spec["low"]) / span if span > 0 else 0.0)
        return np.asarray(encoded, dtype=float)

    @staticmethod
    def _decode(space: Dict[str, Dict[str, Any]], point: np.ndarray) -> ParamDict:
        """Map a unit-hypercube point back to parameters."""
        params: ParamDict = {}
        for (name, spec), coordinate in zip(space.items(), np.clip(point, 0.0, 1.0)):
            if spec["type"] == "categorical":
                choices = spec["choices"]
                params[name] = choices[round(coordinate * (len(choices) - 1))]
            else:
                if spec["log"]:
                    raw = float(
                        np.exp(
                            np.log(spec["low"]) + coordinate * (np.log(spec["high"]) - np.log(spec["low"]))
                        )
                    )
                else:
                    raw = float(spec["low"] + coordinate * (spec["high"] - spec["low"]))
                if spec["type"] == "int":
                    params[name] = int(np.clip(round(raw), spec["low"], spec["high"]))
                else:
                    params[name] = float(np.clip(raw, spec["low"], spec["high"]))
        return params

    # -- objective ------------------------------------------------------- #
    def _make_objective(
        self,
        model: BaseEstimator,
        X: ArrayLike,
        y: ArrayLike,
        cv: Any,
        scoring: ScoringLike,
    ) -> Callable[[ParamDict], float]:
        splitter = _make_cv(cv, model, y, self.random_state)
        scorer = _resolve_scorer(scoring)

        def objective(params: ParamDict) -> float:
            estimator = clone(model).set_params(**params)
            try:
                scores = cross_val_score(
                    estimator, X, y, cv=splitter, scoring=scorer, n_jobs=self.n_jobs, error_score=np.nan
                )
            except Exception as exc:
                self.logger.warning("Configuration %s failed: %s", params, exc)
                return float("nan")
            return float(np.nanmean(scores)) if np.isfinite(scores).any() else float("nan")

        return objective

    # -- surrogate ------------------------------------------------------- #
    def _propose_gp(
        self,
        X_obs: List[np.ndarray],
        y_obs: List[float],
        space: Dict[str, Dict[str, Any]],
        acquisition: str,
        rng: np.random.Generator,
        seen: List[ParamDict],
    ) -> Tuple[ParamDict, float]:
        X_arr = np.asarray(X_obs, dtype=float)
        y_arr = np.asarray(y_obs, dtype=float)
        n_dims = X_arr.shape[1]
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.full(n_dims, 0.5), length_scale_bounds=(1e-2, 1e2), nu=2.5
        ) + WhiteKernel(1e-4, (1e-8, 1e-1))
        gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=int(rng.integers(2**31 - 1))
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            gp.fit(X_arr, y_arr)
        n_local = max(1, self.n_candidates // 4)
        best_x = X_arr[int(np.argmax(y_arr))]
        candidates = np.vstack(
            [
                rng.uniform(size=(self.n_candidates - n_local, n_dims)),
                np.clip(best_x + rng.normal(0.0, 0.1, size=(n_local, n_dims)), 0.0, 1.0),
            ]
        )
        mu, sigma = gp.predict(candidates, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        y_best = y_arr.max()
        xi = self.exploration_weight
        if acquisition == "expected_improvement":
            improvement = mu - y_best - xi
            z = improvement / sigma
            values = improvement * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
        elif acquisition == "probability_of_improvement":
            values = stats.norm.cdf((mu - y_best - xi) / sigma)
        elif acquisition == "upper_confidence_bound":
            values = mu + self.ucb_kappa * sigma
        else:  # pragma: no cover - validated earlier
            raise ValueError(f"Unknown acquisition function '{acquisition}'")
        for index in np.argsort(-values):
            params = self._decode(space, candidates[index])
            if params not in seen:
                return params, float(values[index])
        return self._sample_random(space, rng), float("nan")

    def _run_gp(
        self,
        objective: Callable[[ParamDict], float],
        space: Dict[str, Dict[str, Any]],
        n_iterations: int,
        acquisition: str,
        history: List[Dict[str, Any]],
        early_stopping_rounds: Optional[int],
        early_stopping_threshold: Optional[float],
        rng: np.random.Generator,
        use_surrogate: bool = True,
    ) -> Tuple[List[Dict[str, Any]], bool, str]:
        X_obs = [self._encode(space, h["params"]) for h in history if np.isfinite(h["score"])]
        y_obs = [float(h["score"]) for h in history if np.isfinite(h["score"])]
        seen = [dict(h["params"]) for h in history]
        best = max(y_obs) if y_obs else -np.inf
        n_initial = max(0, self.n_initial_points - len(y_obs))
        new_history: List[Dict[str, Any]] = []
        stale = 0
        early_stopped = False
        reason = "completed"
        for iteration in range(n_iterations):
            if use_surrogate and iteration >= n_initial and len(y_obs) >= 2:
                params, acq_value = self._propose_gp(X_obs, y_obs, space, acquisition, rng, seen)
                source = acquisition
            else:
                params, acq_value = self._sample_random(space, rng), float("nan")
                source = "random"
            score = objective(params)
            record = {
                "iteration": iteration,
                "params": params,
                "score": score,
                "source": source,
                "acquisition_value": acq_value,
            }
            new_history.append(record)
            seen.append(dict(params))
            if np.isfinite(score):
                X_obs.append(self._encode(space, params))
                y_obs.append(score)
            self.logger.debug("Iteration %d (%s): %.4f %s", iteration, source, score, params)
            if np.isfinite(score) and score > best + 1e-12:
                best, stale = score, 0
            else:
                stale += 1
            if early_stopping_threshold is not None and best >= early_stopping_threshold:
                early_stopped, reason = True, "threshold_reached"
                break
            if early_stopping_rounds is not None and stale >= early_stopping_rounds:
                early_stopped, reason = True, "no_improvement"
                break
        return new_history, early_stopped, reason

    # -- optuna backend -------------------------------------------------- #
    @staticmethod
    def _optuna_distributions(space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        distributions = {}
        for name, spec in space.items():
            if spec["type"] == "categorical":
                distributions[name] = optuna.distributions.CategoricalDistribution(spec["choices"])
            elif spec["type"] == "int":
                distributions[name] = optuna.distributions.IntDistribution(
                    spec["low"], spec["high"], log=spec["log"]
                )
            else:
                distributions[name] = optuna.distributions.FloatDistribution(
                    spec["low"], spec["high"], log=spec["log"]
                )
        return distributions

    @staticmethod
    def _optuna_suggest(trial: Any, space: Dict[str, Dict[str, Any]]) -> ParamDict:
        params: ParamDict = {}
        for name, spec in space.items():
            if spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            elif spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"], log=spec["log"])
            else:
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec["log"])
        return params

    def _seed_optuna_study(
        self, study: Any, space: Dict[str, Dict[str, Any]], history: List[Dict[str, Any]]
    ) -> int:
        distributions = self._optuna_distributions(space)
        added = 0
        for record in history:
            if not np.isfinite(record["score"]):
                continue
            params = {name: _to_python(record["params"][name]) for name in space if name in record["params"]}
            if len(params) != len(space):
                continue
            try:
                study.add_trial(
                    optuna.trial.create_trial(
                        params=params, distributions=distributions, value=float(record["score"])
                    )
                )
                added += 1
            except (ValueError, TypeError) as exc:
                self.logger.warning("Could not warm-start optuna with %s: %s", params, exc)
        return added

    def _run_optuna(
        self,
        objective: Callable[[ParamDict], float],
        space: Dict[str, Dict[str, Any]],
        n_iterations: int,
        history: List[Dict[str, Any]],
        early_stopping_rounds: Optional[int],
        early_stopping_threshold: Optional[float],
        seed: int,
    ) -> Tuple[List[Dict[str, Any]], bool, str]:
        sampler = optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=max(1, min(self.n_initial_points, n_iterations))
        )
        study = optuna.create_study(direction="maximize", sampler=sampler)
        self._seed_optuna_study(study, space, history)
        new_history: List[Dict[str, Any]] = []
        finite = [h["score"] for h in history if np.isfinite(h["score"])]
        state = {
            "best": max(finite) if finite else -np.inf,
            "stale": 0,
            "early_stopped": False,
            "reason": "completed",
        }

        def _objective(trial: Any) -> float:
            params = self._optuna_suggest(trial, space)
            score = objective(params)
            new_history.append(
                {"iteration": len(new_history), "params": params, "score": score, "source": "tpe"}
            )
            if not np.isfinite(score):
                raise optuna.TrialPruned()
            return score

        def _callback(study_: Any, trial: Any) -> None:
            if (
                trial.state == optuna.trial.TrialState.COMPLETE
                and trial.value is not None
                and trial.value > state["best"] + 1e-12
            ):
                state["best"], state["stale"] = trial.value, 0
            else:
                state["stale"] += 1
            if early_stopping_threshold is not None and state["best"] >= early_stopping_threshold:
                state["early_stopped"], state["reason"] = True, "threshold_reached"
                study_.stop()
            elif early_stopping_rounds is not None and state["stale"] >= early_stopping_rounds:
                state["early_stopped"], state["reason"] = True, "no_improvement"
                study_.stop()

        study.optimize(_objective, n_trials=n_iterations, callbacks=[_callback], show_progress_bar=False)
        return new_history, state["early_stopped"], state["reason"]

    # -- public API ------------------------------------------------------ #
    def optimize(
        self,
        model: BaseEstimator,
        search_space: Dict[str, Any],
        X: ArrayLike,
        y: ArrayLike,
        n_iterations: int = 20,
        cv: Any = 3,
        scoring: ScoringLike = None,
        acquisition_function: str = "expected_improvement",
        early_stopping_rounds: Optional[int] = None,
        early_stopping_threshold: Optional[float] = None,
        warm_start: bool = False,
        previous_results: Optional[Sequence[Dict[str, Any]]] = None,
        backend: Optional[str] = None,
    ) -> Tuple[ParamDict, Dict[str, Any]]:
        """Maximise the cross-validated score over ``search_space``.

        Args:
            model: Estimator template.
            search_space: See :meth:`parse_search_space`.
            X: Feature matrix.
            y: Target vector.
            n_iterations: Number of new configurations to evaluate.
            cv: Folds or splitter.
            scoring: Scoring override.
            acquisition_function: ``"expected_improvement"``,
                ``"probability_of_improvement"`` or ``"upper_confidence_bound"``
                (GP backend; Optuna uses TPE regardless).
            early_stopping_rounds: Stop after this many non-improving rounds.
            early_stopping_threshold: Stop once the best score reaches this.
            warm_start: Seed the search with ``previous_results``.
            previous_results: Records with ``params`` and ``score`` (e.g. the
                ``optimization_history`` of an earlier run).
            backend: Backend override.

        Returns:
            ``(best_params, results)``; ``results`` holds ``best_score``,
            ``best_estimator``, ``optimization_history`` (previous records are
            included when warm-starting, flagged with ``warm_start=True``),
            ``early_stopped``, ``stopping_reason`` and ``backend``.

        Raises:
            ValueError: For unknown acquisition functions or empty spaces.
            RuntimeError: If no configuration could be evaluated.
        """
        acquisition = _ACQUISITION_ALIASES.get(acquisition_function.lower())
        if acquisition is None:
            raise ValueError(f"Unknown acquisition function '{acquisition_function}'")
        space = self.parse_search_space(search_space)
        if not space:
            raise ValueError("search_space must not be empty")
        backend = self.resolve_backend(backend)
        scoring = self.scoring if scoring is None else scoring
        objective = self._make_objective(model, X, y, cv, scoring)

        history: List[Dict[str, Any]] = []
        if warm_start and previous_results:
            for record in previous_results:
                params = {name: record["params"][name] for name in space if name in record.get("params", {})}
                if len(params) != len(space) or "score" not in record:
                    continue
                score = float(record["score"]) if record["score"] is not None else float("nan")
                history.append(
                    {
                        "iteration": -1,
                        "params": params,
                        "score": score,
                        "source": "warm_start",
                        "warm_start": True,
                    }
                )
            self.logger.info("Warm-starting with %d previous evaluations", len(history))

        start = time.perf_counter()
        rng = np.random.default_rng(self.random_state)
        if backend == "optuna":
            new_history, early_stopped, reason = self._run_optuna(
                objective,
                space,
                n_iterations,
                history,
                early_stopping_rounds,
                early_stopping_threshold,
                self.random_state,
            )
        else:
            new_history, early_stopped, reason = self._run_gp(
                objective,
                space,
                n_iterations,
                acquisition,
                history,
                early_stopping_rounds,
                early_stopping_threshold,
                rng,
                use_surrogate=(backend == "gp"),
            )
        full_history = history + new_history
        finite = [h for h in full_history if np.isfinite(h["score"])]
        if not finite:
            raise RuntimeError("Every evaluated configuration failed; check the search space")
        best = max(finite, key=lambda h: h["score"])
        best_params = {name: _to_python(value) for name, value in best["params"].items()}
        best_estimator = clone(model).set_params(**best_params).fit(X, y)
        results = {
            "best_params": best_params,
            "best_score": float(best["score"]),
            "best_estimator": best_estimator,
            "optimization_history": full_history,
            "n_iterations": len(new_history),
            "n_warm_start": len(history),
            "early_stopped": early_stopped,
            "stopping_reason": reason,
            "backend": backend,
            "acquisition_function": acquisition,
            "execution_time": time.perf_counter() - start,
            "scoring": _scoring_name(scoring),
            "convergence": [
                float(v)
                for v in np.maximum.accumulate(
                    [h["score"] if np.isfinite(h["score"]) else -np.inf for h in new_history]
                )
            ]
            if new_history
            else [],
        }
        self.results_ = results
        self.logger.info(
            "Bayesian optimisation (%s): best %s = %.4f", backend, results["scoring"], results["best_score"]
        )
        return best_params, results

    def multi_objective_optimize(
        self,
        model: BaseEstimator,
        search_space: Dict[str, Any],
        X: ArrayLike,
        y: ArrayLike,
        objectives: Sequence[str] = ("accuracy", "model_simplicity"),
        n_iterations: int = 20,
        cv: Any = 3,
        backend: Optional[str] = None,
        reference_point: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Multi-objective optimisation returning the Pareto front.

        With Optuna a multi-objective TPE study is used; the scikit-learn
        backend uses random scalarisation (Tchebycheff) with the GP surrogate.

        Args:
            model: Estimator template.
            search_space: See :meth:`parse_search_space`.
            X: Feature matrix.
            y: Target vector.
            objectives: Scorer names and/or special objectives (see
                :func:`_evaluate_objectives`).
            n_iterations: Number of configurations to evaluate.
            cv: Folds or splitter.
            backend: Backend override.
            reference_point: Optional hypervolume reference in raw objective
                units; defaults to the worst observed value per objective.

        Returns:
            ``(pareto_solutions, results)`` where each solution has ``params``
            and ``objectives`` and ``results`` holds ``pareto_front``,
            ``all_trials``, ``hypervolume`` (raw units),
            ``hypervolume_normalized`` (objectives min-max scaled to [0, 1]),
            ``trade_off_analysis`` and ``directions``.
        """
        objectives = list(objectives)
        space = self.parse_search_space(search_space)
        backend = self.resolve_backend(backend)
        directions = np.array([_objective_direction(o) for o in objectives])
        trials: List[Dict[str, Any]] = []

        def evaluate(params: ParamDict) -> Dict[str, float]:
            values = _evaluate_objectives(model, params, X, y, objectives, cv, self.random_state, self.n_jobs)
            trials.append({"iteration": len(trials), "params": dict(params), "objectives": values})
            return values

        start = time.perf_counter()
        if backend == "optuna":
            sampler = optuna.samplers.TPESampler(
                seed=self.random_state, n_startup_trials=max(1, min(self.n_initial_points, n_iterations))
            )
            study = optuna.create_study(
                directions=["maximize" if d > 0 else "minimize" for d in directions], sampler=sampler
            )

            def _objective(trial: Any) -> Tuple[float, ...]:
                values = evaluate(self._optuna_suggest(trial, space))
                return tuple(float(values[o]) for o in objectives)

            study.optimize(_objective, n_trials=n_iterations, show_progress_bar=False)
        else:
            rng = np.random.default_rng(self.random_state)
            X_obs: List[np.ndarray] = []
            seen: List[ParamDict] = []
            for iteration in range(n_iterations):
                use_gp = backend == "gp" and iteration >= self.n_initial_points and len(trials) >= 2
                if use_gp:
                    matrix = np.array([[t["objectives"][o] for o in objectives] for t in trials]) * directions
                    low, high = matrix.min(axis=0), matrix.max(axis=0)
                    span = np.where(high - low > 0, high - low, 1.0)
                    normalised = (matrix - low) / span
                    weights = rng.dirichlet(np.ones(len(objectives)))
                    scalar = np.min(weights * normalised + 1e-9, axis=1) + 0.05 * (weights * normalised).sum(
                        axis=1
                    )
                    params, _ = self._propose_gp(
                        X_obs, list(scalar), space, "expected_improvement", rng, seen
                    )
                else:
                    params = self._sample_random(space, rng)
                evaluate(params)
                X_obs.append(self._encode(space, params))
                seen.append(dict(params))

        points = np.array([[t["objectives"][o] for o in objectives] for t in trials]) * directions
        mask = _pareto_mask(points)
        pareto = [t for t, keep in zip(trials, mask) if keep]
        if reference_point is not None:
            reference = np.array([reference_point[o] for o in objectives]) * directions
        else:
            reference = points.min(axis=0)
        hypervolume = _hypervolume(points[mask], reference, random_state=self.random_state)
        low, high = points.min(axis=0), points.max(axis=0)
        span = np.where(high - low > 0, high - low, 1.0)
        normalised_hv = _hypervolume(
            (points[mask] - low) / span, np.zeros(len(objectives)), random_state=self.random_state
        )
        results = {
            "pareto_front": pareto,
            "pareto_solutions": pareto,
            "all_trials": trials,
            "hypervolume": float(hypervolume),
            "hypervolume_normalized": float(normalised_hv),
            "reference_point": {o: float(r * d) for o, r, d in zip(objectives, reference, directions)},
            "objectives": objectives,
            "directions": {o: "minimize" if d < 0 else "maximize" for o, d in zip(objectives, directions)},
            "trade_off_analysis": _trade_off_analysis(trials, pareto, objectives),
            "n_iterations": len(trials),
            "n_pareto_optimal": len(pareto),
            "backend": backend,
            "execution_time": time.perf_counter() - start,
        }
        self.results_ = results
        return pareto, results


# --------------------------------------------------------------------------- #
# Grid search pipeline
# --------------------------------------------------------------------------- #
class GridSearchPipeline(LoggerMixin):
    """Exhaustive, randomised, adaptive, parallel and constrained grid search.

    Every search returns ``(best_estimator, results)`` where ``results``
    contains ``best_params``, ``best_score``, ``cv_results`` (one record per
    evaluated candidate) and ``execution_time``.

    Args:
        scoring: Default scoring specification.
        random_state: Seed for sampling and fold shuffling.
        n_jobs: Default parallelism.
        verbose: Verbosity passed to scikit-learn searches.
    """

    def __init__(
        self,
        scoring: ScoringLike = "accuracy",
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        self.scoring = scoring
        self.random_state = _resolve_random_state(random_state)
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.results_: Dict[str, Any] = {}

    def _scoring(self, scoring: ScoringLike) -> Any:
        return _resolve_scorer(self.scoring if scoring is None else scoring)

    def _run(
        self, search: Any, X: ArrayLike, y: ArrayLike, scoring: ScoringLike
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        start = time.perf_counter()
        search.fit(X, y)
        results = {
            "best_params": dict(search.best_params_),
            "best_score": float(search.best_score_),
            "best_index": int(search.best_index_),
            "cv_results": _cv_results_to_records(search.cv_results_),
            "n_candidates": len(search.cv_results_["params"]),
            "execution_time": time.perf_counter() - start,
            "scoring": _scoring_name(self.scoring if scoring is None else scoring),
            "refit_time": float(getattr(search, "refit_time_", 0.0)),
        }
        self.results_ = results
        return search.best_estimator_, results

    def exhaustive_search(
        self,
        model: BaseEstimator,
        param_grid: Union[Dict[str, Sequence[Any]], List[Dict[str, Sequence[Any]]]],
        X: ArrayLike,
        y: ArrayLike,
        cv: Any = 3,
        scoring: ScoringLike = None,
        n_jobs: Optional[int] = None,
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Evaluate every combination in ``param_grid``.

        Args:
            model: Estimator template.
            param_grid: Grid (dict or list of dicts).
            X: Feature matrix.
            y: Target vector.
            cv: Folds or splitter.
            scoring: Scoring override.
            n_jobs: Parallelism override.

        Returns:
            ``(best_estimator, results)``.
        """
        search = GridSearchCV(
            clone(model),
            param_grid,
            cv=_make_cv(cv, model, y, self.random_state),
            scoring=self._scoring(scoring),
            n_jobs=self.n_jobs if n_jobs is None else n_jobs,
            verbose=self.verbose,
        )
        return self._run(search, X, y, scoring)

    def randomized_search(
        self,
        model: BaseEstimator,
        param_distributions: Dict[str, Any],
        X: ArrayLike,
        y: ArrayLike,
        n_iter: int = 20,
        cv: Any = 3,
        scoring: ScoringLike = None,
        n_jobs: Optional[int] = None,
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Sample ``n_iter`` configurations from ``param_distributions``.

        Args:
            model: Estimator template.
            param_distributions: Lists or scipy distributions per parameter.
            X: Feature matrix.
            y: Target vector.
            n_iter: Number of sampled configurations.
            cv: Folds or splitter.
            scoring: Scoring override.
            n_jobs: Parallelism override.

        Returns:
            ``(best_estimator, results)``.
        """
        search = RandomizedSearchCV(
            clone(model),
            param_distributions,
            n_iter=n_iter,
            cv=_make_cv(cv, model, y, self.random_state),
            scoring=self._scoring(scoring),
            n_jobs=self.n_jobs if n_jobs is None else n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return self._run(search, X, y, scoring)

    @staticmethod
    def _refine_grid(
        grid: Dict[str, Sequence[Any]],
        best_params: ParamDict,
        n_points: int,
        shrink_factor: float,
    ) -> Dict[str, List[Any]]:
        """Zoom the grid in around ``best_params`` for numeric parameters."""
        refined: Dict[str, List[Any]] = {}
        for name, values in grid.items():
            best = best_params.get(name)
            numeric = sorted({v for v in values if _is_number(v)})
            if best is None or not _is_number(best) or len(numeric) < 2:
                refined[name] = [best]
                continue
            position = numeric.index(best) if best in numeric else int(np.searchsorted(numeric, best))
            below = numeric[position - 1] if position > 0 else None
            above = numeric[position + 1] if position + 1 < len(numeric) else None
            span_low = (best - below) if below is not None else (above - best)
            span_high = (above - best) if above is not None else (best - below)
            low = best - span_low * shrink_factor
            high = best + span_high * shrink_factor
            if numeric[0] > 0:
                low = max(low, best * shrink_factor, numeric[0] * shrink_factor)
            candidates = list(np.linspace(low, high, n_points)) + [best]
            if all(_is_integer(v) for v in numeric):
                lower_bound = 1 if numeric[0] >= 1 else numeric[0]
                candidates = [int(max(lower_bound, round(v))) for v in candidates]
            else:
                candidates = [float(v) for v in candidates]
            refined[name] = sorted(set(candidates))
        return refined

    def adaptive_search(
        self,
        model: BaseEstimator,
        initial_param_grid: Dict[str, Sequence[Any]],
        X: ArrayLike,
        y: ArrayLike,
        refinement_rounds: int = 2,
        cv: Any = 3,
        scoring: ScoringLike = None,
        n_points: int = 3,
        shrink_factor: float = 0.5,
        min_improvement: float = 1e-4,
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Grid search that repeatedly zooms in around the best configuration.

        Numeric parameters are refined to ``n_points`` values inside a window
        of ``shrink_factor`` times the distance to the neighbouring grid
        values; non-numeric parameters are frozen at their best value.
        Refinement stops early when a round yields no improvement.

        Args:
            model: Estimator template.
            initial_param_grid: Starting grid.
            X: Feature matrix.
            y: Target vector.
            refinement_rounds: Maximum number of refinement rounds.
            cv: Folds or splitter.
            scoring: Scoring override.
            n_points: Grid points per numeric parameter per round.
            shrink_factor: Window shrink factor per round.
            min_improvement: Minimum score gain to continue refining.

        Returns:
            ``(best_estimator, results)`` with ``refinement_history``.
        """
        current_grid: Dict[str, List[Any]] = {k: list(v) for k, v in initial_param_grid.items()}
        best_model, initial = self.exhaustive_search(model, current_grid, X, y, cv=cv, scoring=scoring)
        best_score, best_params = initial["best_score"], initial["best_params"]
        history: List[Dict[str, Any]] = []
        total_candidates = initial["n_candidates"]
        for round_index in range(refinement_rounds):
            refined = self._refine_grid(current_grid, best_params, n_points, shrink_factor)
            if all(len(v) == 1 for v in refined.values()):
                self.logger.info("Adaptive search: grid cannot be refined further")
                break
            candidate_model, round_results = self.exhaustive_search(
                model, refined, X, y, cv=cv, scoring=scoring
            )
            improvement = round_results["best_score"] - best_score
            improved = improvement > min_improvement
            history.append(
                {
                    "round": round_index + 1,
                    "param_grid": refined,
                    "best_params": round_results["best_params"],
                    "best_score": round_results["best_score"],
                    "improvement": float(improvement),
                    "improved": improved,
                    "n_candidates": round_results["n_candidates"],
                }
            )
            total_candidates += round_results["n_candidates"]
            if improved:
                best_model, best_score, best_params = (
                    candidate_model,
                    round_results["best_score"],
                    round_results["best_params"],
                )
            current_grid = refined
            if not improved:
                self.logger.info("Adaptive search: no improvement in round %d, stopping", round_index + 1)
                break
        results = {
            "best_params": best_params,
            "best_score": best_score,
            "initial_results": initial,
            "refinement_history": history,
            "n_refinement_rounds": len(history),
            "total_candidates": total_candidates,
            "scoring": initial["scoring"],
        }
        self.results_ = results
        return best_model, results

    def parallel_search(
        self,
        model: BaseEstimator,
        param_grid: Union[Dict[str, Sequence[Any]], List[Dict[str, Sequence[Any]]]],
        X: ArrayLike,
        y: ArrayLike,
        cv: Any = 3,
        n_jobs: int = -1,
        scoring: ScoringLike = None,
        pre_dispatch: Union[int, str] = "2*n_jobs",
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Exhaustive search with explicit joblib parallelism.

        Args:
            model: Estimator template.
            param_grid: Grid (dict or list of dicts).
            X: Feature matrix.
            y: Target vector.
            cv: Folds or splitter.
            n_jobs: Number of parallel workers.
            scoring: Scoring override.
            pre_dispatch: joblib pre-dispatch setting.

        Returns:
            ``(best_estimator, results)`` including ``n_jobs``.
        """
        search = GridSearchCV(
            clone(model),
            param_grid,
            cv=_make_cv(cv, model, y, self.random_state),
            scoring=self._scoring(scoring),
            n_jobs=n_jobs,
            pre_dispatch=pre_dispatch,
            verbose=self.verbose,
        )
        best_model, results = self._run(search, X, y, scoring)
        results["n_jobs"] = n_jobs
        return best_model, results

    def constrained_search(
        self,
        model: BaseEstimator,
        param_grid: Dict[str, Sequence[Any]],
        constraint_function: Callable[[ParamDict], bool],
        X: ArrayLike,
        y: ArrayLike,
        cv: Any = 3,
        scoring: ScoringLike = None,
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Grid search restricted to combinations satisfying a predicate.

        Args:
            model: Estimator template.
            param_grid: Grid.
            constraint_function: ``params -> bool``; only truthy combinations
                are evaluated.
            X: Feature matrix.
            y: Target vector.
            cv: Folds or splitter.
            scoring: Scoring override.

        Returns:
            ``(best_estimator, results)`` including ``feasible_combinations``,
            ``infeasible_combinations`` and ``total_combinations``.

        Raises:
            ValueError: If no combination satisfies the constraint.
        """
        combinations = list(ParameterGrid(param_grid))
        feasible = [params for params in combinations if constraint_function(params)]
        if not feasible:
            raise ValueError("No parameter combination satisfies the constraint")
        self.logger.info(
            "Constrained search: %d of %d combinations feasible", len(feasible), len(combinations)
        )
        search = GridSearchCV(
            clone(model),
            [{k: [v] for k, v in params.items()} for params in feasible],
            cv=_make_cv(cv, model, y, self.random_state),
            scoring=self._scoring(scoring),
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        best_model, results = self._run(search, X, y, scoring)
        results.update(
            {
                "feasible_combinations": len(feasible),
                "infeasible_combinations": len(combinations) - len(feasible),
                "total_combinations": len(combinations),
            }
        )
        return best_model, results


# --------------------------------------------------------------------------- #
# Legacy analysers (kept for backwards compatibility)
# --------------------------------------------------------------------------- #
class AdvancedModelSelector(LoggerMixin):
    """Model selection with per-model hyperparameter optimisation.

    Args:
        cv_strategy: ``"stratified"``, ``"kfold"`` or ``"timeseries"``.
        cv_folds: Number of folds.
        scoring: Scoring specification.
        random_state: Seed.
        n_jobs: Parallelism.
    """

    def __init__(
        self,
        cv_strategy: str = "stratified",
        cv_folds: int = 5,
        scoring: ScoringLike = "accuracy",
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = -1,
    ):
        self.cv_strategy = cv_strategy
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = _resolve_random_state(random_state)
        self.n_jobs = n_jobs
        self.cv_splitter_ = self._get_cv_splitter()
        self.results_: Dict[str, Any] = {}

    def _get_cv_splitter(self) -> Any:
        if self.cv_strategy == "stratified":
            return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        if self.cv_strategy == "kfold":
            return KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        if self.cv_strategy == "timeseries":
            return TimeSeriesSplit(n_splits=self.cv_folds)
        raise ValueError(f"Unknown CV strategy: {self.cv_strategy}")

    def select_best_model(
        self,
        models: Dict[str, BaseEstimator],
        X: ArrayLike,
        y: ArrayLike,
        param_grids: Optional[Dict[str, Dict[str, Sequence[Any]]]] = None,
    ) -> Tuple[str, BaseEstimator, Dict[str, Any]]:
        """Select the best model, tuning those that have a parameter grid.

        Args:
            models: Mapping name -> estimator.
            X: Feature matrix.
            y: Target vector.
            param_grids: Optional per-model parameter grids.

        Returns:
            ``(best_model_name, best_estimator, results)``.
        """
        param_grids = param_grids or {}
        model_scores: Dict[str, float] = {}
        model_details: Dict[str, Dict[str, Any]] = {}
        for name, model in models.items():
            try:
                grid = param_grids.get(name, {})
                if grid:
                    estimator, best_params, best_score = self._optimize_hyperparameters(
                        model, X, y, grid, "grid_search"
                    )
                    model_details[name] = {
                        "model": estimator,
                        "best_params": best_params,
                        "best_score": best_score,
                    }
                else:
                    scores = cross_val_score(
                        model,
                        X,
                        y,
                        cv=self.cv_splitter_,
                        scoring=_resolve_scorer(self.scoring),
                        n_jobs=self.n_jobs,
                    )
                    best_score = float(np.mean(scores))
                    model_details[name] = {
                        "model": model,
                        "best_params": {},
                        "best_score": best_score,
                        "cv_scores": scores,
                    }
                model_scores[name] = best_score
                self.logger.info("%s: CV score = %.4f", name, best_score)
            except Exception as exc:
                self.logger.error("Error evaluating %s: %s", name, exc)
                model_scores[name] = -np.inf
                model_details[name] = {"error": str(exc)}
        best_name = max(model_scores, key=model_scores.get)
        self.results_ = {
            "model_scores": model_scores,
            "model_details": model_details,
            "best_model": best_name,
        }
        return best_name, model_details[best_name].get("model"), self.results_

    def optimize_hyperparameters(
        self,
        model: BaseEstimator,
        X: ArrayLike,
        y: ArrayLike,
        param_grid: Dict[str, Any],
        method: str = "grid_search",
        n_trials: int = 100,
    ) -> Tuple[BaseEstimator, ParamDict, float]:
        """Optimise a single model with grid, random or Bayesian search.

        Args:
            model: Estimator template.
            X: Feature matrix.
            y: Target vector.
            param_grid: Grid, distributions or Bayesian search space.
            method: ``"grid_search"``, ``"random_search"``, ``"optuna"`` or
                ``"bayesian"``.
            n_trials: Trials for random / Bayesian search.

        Returns:
            ``(fitted_estimator, best_params, best_score)``.
        """
        return self._optimize_hyperparameters(model, X, y, param_grid, method, n_trials)

    def _optimize_hyperparameters(
        self,
        model: BaseEstimator,
        X: ArrayLike,
        y: ArrayLike,
        param_grid: Dict[str, Any],
        method: str = "grid_search",
        n_trials: int = 100,
    ) -> Tuple[BaseEstimator, ParamDict, float]:
        scoring = _resolve_scorer(self.scoring)
        if method == "grid_search":
            search: Any = GridSearchCV(
                model, param_grid, cv=self.cv_splitter_, scoring=scoring, n_jobs=self.n_jobs
            )
        elif method == "random_search":
            search = RandomizedSearchCV(
                model,
                param_grid,
                cv=self.cv_splitter_,
                scoring=scoring,
                n_jobs=self.n_jobs,
                n_iter=n_trials,
                random_state=self.random_state,
            )
        elif method in ("optuna", "bayesian"):
            optimizer = BayesianOptimizer(
                scoring=self.scoring,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                backend="optuna" if method == "optuna" else "auto",
            )
            best_params, results = optimizer.optimize(
                model, param_grid, X, y, n_iterations=n_trials, cv=self.cv_splitter_
            )
            return results["best_estimator"], best_params, results["best_score"]
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        search.fit(X, y)
        return search.best_estimator_, dict(search.best_params_), float(search.best_score_)

    def get_model_comparison_report(self) -> pd.DataFrame:
        """Return the last selection as a DataFrame sorted by score."""
        if not self.results_:
            return pd.DataFrame()
        rows = []
        for name, details in self.results_["model_details"].items():
            if "error" in details:
                continue
            row = {
                "model": name,
                "best_score": details["best_score"],
                "is_best": name == self.results_["best_model"],
            }
            for index, (param, value) in enumerate(list(details.get("best_params", {}).items())[:3]):
                row[f"param_{index + 1}"] = f"{param}={value}"
            rows.append(row)
        frame = pd.DataFrame(rows)
        return frame.sort_values("best_score", ascending=False) if not frame.empty else frame


class MultiObjectiveSelector(LoggerMixin):
    """Select models that are Pareto-optimal across several metrics.

    Args:
        metrics: Scorer names.
        weights: Weights for the aggregated score (equal by default).
        cv_folds: Number of folds.
        random_state: Seed.
    """

    def __init__(
        self,
        metrics: Sequence[str],
        weights: Optional[Sequence[float]] = None,
        cv_folds: int = 5,
        random_state: Optional[int] = None,
    ):
        self.metrics = list(metrics)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.metrics)
        if len(self.weights) != len(self.metrics):
            raise ValueError("Number of weights must match number of metrics")
        self.cv_folds = cv_folds
        self.random_state = _resolve_random_state(random_state)
        self.cv_splitter_ = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

    def select_pareto_optimal_models(
        self,
        models: Dict[str, BaseEstimator],
        X: ArrayLike,
        y: ArrayLike,
    ) -> Dict[str, Any]:
        """Evaluate every model on every metric and find the Pareto set.

        Args:
            models: Mapping name -> estimator.
            X: Feature matrix.
            y: Target vector.

        Returns:
            Dictionary with ``model_scores``, ``weighted_scores``,
            ``pareto_optimal`` and ``best_overall``.
        """
        model_scores: Dict[str, Dict[str, float]] = {}
        for name, model in models.items():
            cv_results = cross_validate(
                model, X, y, cv=self.cv_splitter_, scoring={m: m for m in self.metrics}, error_score=np.nan
            )
            model_scores[name] = {
                m: float(np.nan_to_num(np.mean(cv_results[f"test_{m}"]), nan=-np.inf)) for m in self.metrics
            }
        weighted = {
            name: float(sum(scores[m] * w for m, w in zip(self.metrics, self.weights)))
            for name, scores in model_scores.items()
        }
        names = list(model_scores)
        points = np.array([[model_scores[n][m] for m in self.metrics] for n in names])
        pareto = [n for n, keep in zip(names, _pareto_mask(points)) if keep]
        return {
            "model_scores": model_scores,
            "weighted_scores": weighted,
            "pareto_optimal": pareto,
            "best_overall": max(weighted, key=weighted.get),
            "metrics": self.metrics,
            "weights": self.weights,
        }


class NestedCrossValidation(LoggerMixin):
    """Nested cross-validation over several models.

    Args:
        outer_cv: Outer folds.
        inner_cv: Inner folds.
        random_state: Seed.
    """

    def __init__(self, outer_cv: int = 5, inner_cv: int = 3, random_state: Optional[int] = None):
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.random_state = _resolve_random_state(random_state)

    def evaluate_models(
        self,
        models: Dict[str, BaseEstimator],
        X: ArrayLike,
        y: ArrayLike,
        param_grids: Optional[Dict[str, Dict[str, Sequence[Any]]]] = None,
        scoring: ScoringLike = "accuracy",
    ) -> Dict[str, Any]:
        """Run nested CV for every model.

        Args:
            models: Mapping name -> estimator.
            X: Feature matrix.
            y: Target vector.
            param_grids: Optional per-model grids for the inner search.
            scoring: Scoring specification.

        Returns:
            Dictionary with per-model ``results`` and the ``best_model``.
        """
        param_grids = param_grids or {}
        first = next(iter(models.values()))
        outer = _make_cv(self.outer_cv, first, y, self.random_state)
        inner = _make_cv(self.inner_cv, first, y, self.random_state + 1)
        scorer_spec = _resolve_scorer(scoring)
        nested_scores: Dict[str, List[float]] = {name: [] for name in models}
        selected: Dict[str, List[ParamDict]] = {name: [] for name in models}
        for fold, (train_idx, test_idx) in enumerate(outer.split(X, y)):
            self.logger.info("Outer fold %d/%d", fold + 1, self.outer_cv)
            X_train, X_test = _take(X, train_idx), _take(X, test_idx)
            y_train, y_test = _take(y, train_idx), _take(y, test_idx)
            for name, model in models.items():
                grid = param_grids.get(name, {})
                if grid:
                    search = GridSearchCV(clone(model), grid, cv=inner, scoring=scorer_spec)
                    search.fit(X_train, y_train)
                    fitted, params = search.best_estimator_, dict(search.best_params_)
                else:
                    fitted, params = clone(model).fit(X_train, y_train), {}
                scorer = check_scoring(fitted, scoring=scorer_spec)
                nested_scores[name].append(float(scorer(fitted, X_test, y_test)))
                selected[name].append(params)
        results = {
            name: {
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "scores": scores,
                "selected_params": selected[name],
            }
            for name, scores in nested_scores.items()
        }
        best = max(results, key=lambda name: results[name]["mean_score"])
        return {
            "results": results,
            "best_model": best,
            "outer_cv_folds": self.outer_cv,
            "inner_cv_folds": self.inner_cv,
        }


class LearningCurveAnalyzer(LoggerMixin):
    """Learning-curve based bias/variance diagnostics.

    Args:
        cv_folds: Number of folds.
        random_state: Seed.
        n_jobs: Parallelism.
    """

    def __init__(self, cv_folds: int = 5, random_state: Optional[int] = None, n_jobs: Optional[int] = None):
        self.cv_folds = cv_folds
        self.random_state = _resolve_random_state(random_state)
        self.n_jobs = n_jobs

    def analyze_learning_curve(
        self,
        model: BaseEstimator,
        X: ArrayLike,
        y: ArrayLike,
        train_sizes: Optional[np.ndarray] = None,
        scoring: ScoringLike = "accuracy",
    ) -> Dict[str, Any]:
        """Compute a learning curve and derive bias/variance labels.

        Args:
            model: Estimator template.
            X: Feature matrix.
            y: Target vector.
            train_sizes: Relative or absolute training sizes.
            scoring: Scoring specification.

        Returns:
            Dictionary with curve statistics, ``bias_level``,
            ``variance_level`` and ``recommendations``.
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        sizes, train_scores, val_scores = learning_curve(
            model,
            X,
            y,
            train_sizes=train_sizes,
            cv=_make_cv(self.cv_folds, model, y, self.random_state),
            scoring=_resolve_scorer(scoring),
            n_jobs=self.n_jobs,
            shuffle=True,
            random_state=self.random_state,
        )
        train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
        val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)
        final_train, final_val = float(train_mean[-1]), float(val_mean[-1])
        gap = final_train - final_val
        bias_level = "high" if final_val < 0.7 else "medium" if final_val < 0.85 else "low"
        variance_level = "high" if gap > 0.1 else "medium" if gap > 0.05 else "low"
        recommendations = []
        if bias_level == "high":
            recommendations.append("Consider using a more complex model or adding features")
        if variance_level == "high":
            recommendations.append("Consider regularization, more data, or simpler model")
        if not recommendations:
            recommendations.append("Model appears well-balanced")
        return {
            "train_sizes": sizes.tolist(),
            "train_scores_mean": train_mean.tolist(),
            "train_scores_std": train_std.tolist(),
            "val_scores_mean": val_mean.tolist(),
            "val_scores_std": val_std.tolist(),
            "final_train_score": final_train,
            "final_val_score": final_val,
            "overfitting_gap": float(gap),
            "bias_level": bias_level,
            "variance_level": variance_level,
            "recommendations": recommendations,
            "scoring_metric": _scoring_name(scoring),
        }

    def compare_models_learning_curves(
        self,
        models: Dict[str, BaseEstimator],
        X: ArrayLike,
        y: ArrayLike,
        scoring: ScoringLike = "accuracy",
    ) -> Dict[str, Dict[str, Any]]:
        """Analyse the learning curve of every model.

        Args:
            models: Mapping name -> estimator.
            X: Feature matrix.
            y: Target vector.
            scoring: Scoring specification.

        Returns:
            Mapping name -> learning-curve analysis (or ``{"error": ...}``).
        """
        results: Dict[str, Dict[str, Any]] = {}
        for name, model in models.items():
            try:
                results[name] = self.analyze_learning_curve(model, X, y, scoring=scoring)
            except Exception as exc:
                self.logger.error("Error analysing %s: %s", name, exc)
                results[name] = {"error": str(exc)}
        return results


class ValidationCurveAnalyzer(LoggerMixin):
    """Validation-curve based hyperparameter sensitivity analysis.

    Args:
        cv_folds: Number of folds.
        random_state: Seed.
        n_jobs: Parallelism.
    """

    def __init__(self, cv_folds: int = 5, random_state: Optional[int] = None, n_jobs: Optional[int] = None):
        self.cv_folds = cv_folds
        self.random_state = _resolve_random_state(random_state)
        self.n_jobs = n_jobs

    def analyze_validation_curve(
        self,
        model: BaseEstimator,
        X: ArrayLike,
        y: ArrayLike,
        param_name: str,
        param_range: Sequence[Any],
        scoring: ScoringLike = "accuracy",
    ) -> Dict[str, Any]:
        """Compute a validation curve for one hyperparameter.

        Args:
            model: Estimator template.
            X: Feature matrix.
            y: Target vector.
            param_name: Parameter to vary.
            param_range: Values to try.
            scoring: Scoring specification.

        Returns:
            Dictionary with curve statistics, ``best_param_value`` and a
            ``sensitivity`` label.
        """
        train_scores, val_scores = validation_curve(
            model,
            X,
            y,
            param_name=param_name,
            param_range=param_range,
            cv=_make_cv(self.cv_folds, model, y, self.random_state),
            scoring=_resolve_scorer(scoring),
            n_jobs=self.n_jobs,
        )
        train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
        val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)
        best_idx = int(np.argmax(val_mean))
        score_range = float(val_mean.max() - val_mean.min())
        sensitivity = "high" if score_range > 0.1 else "medium" if score_range > 0.05 else "low"
        return {
            "param_name": param_name,
            "param_range": list(param_range),
            "train_scores_mean": train_mean.tolist(),
            "train_scores_std": train_std.tolist(),
            "val_scores_mean": val_mean.tolist(),
            "val_scores_std": val_std.tolist(),
            "best_param_value": param_range[best_idx],
            "best_score": float(val_mean[best_idx]),
            "sensitivity": sensitivity,
            "score_range": score_range,
            "scoring_metric": _scoring_name(scoring),
        }


class AutoMLSelector(LoggerMixin):
    """Time-budgeted automatic model selection with quick tuning.

    Args:
        time_budget: Budget in seconds.
        random_state: Seed.
        n_jobs: Parallelism.
    """

    def __init__(
        self, time_budget: int = 300, random_state: Optional[int] = None, n_jobs: Optional[int] = None
    ):
        self.time_budget = time_budget
        self.random_state = _resolve_random_state(random_state)
        self.n_jobs = n_jobs
        self.start_time: Optional[float] = None
        self.results_: Dict[str, Any] = {}

    def auto_select(self, X: ArrayLike, y: ArrayLike, task_type: str = "auto") -> Dict[str, Any]:
        """Evaluate candidate models within the budget and tune the best ones.

        Args:
            X: Feature matrix.
            y: Target vector.
            task_type: ``"auto"``, ``"classification"`` or ``"regression"``.

        Returns:
            Dictionary with ``best_model``, ``best_score``,
            ``evaluated_models`` and timing information.
        """
        self.start_time = time.perf_counter()
        if task_type == "auto":
            task_type = AutoModelSelector(random_state=self.random_state).infer_task_type(y)
        models = self._get_candidate_models(task_type)
        best_model: Optional[str] = None
        best_score = -np.inf
        evaluated: Dict[str, Dict[str, Any]] = {}
        for name in self._prioritize_models(models, np.asarray(X).shape):
            if self._time_remaining() < 10:
                break
            model = models[name]
            try:
                started = time.perf_counter()
                quick_score = self._quick_evaluate(model, X, y, task_type)
                evaluated[name] = {
                    "quick_score": quick_score,
                    "eval_time": time.perf_counter() - started,
                    "model": model,
                }
                if quick_score > best_score:
                    best_score, best_model = quick_score, name
                self.logger.info("%s: %.4f (%.1fs)", name, quick_score, evaluated[name]["eval_time"])
            except Exception as exc:
                self.logger.error("Error evaluating %s: %s", name, exc)
        if self._time_remaining() > 30 and evaluated:
            top = sorted(evaluated.items(), key=lambda item: item[1]["quick_score"], reverse=True)[:3]
            for name, info in top:
                if self._time_remaining() < 20:
                    break
                try:
                    optimized = self._optimize_model(info["model"], X, y, task_type)
                    evaluated[name]["optimized_score"] = optimized
                    if optimized > best_score:
                        best_score, best_model = optimized, name
                except Exception as exc:
                    self.logger.error("Error optimizing %s: %s", name, exc)
        self.results_ = {
            "best_model": best_model,
            "best_score": best_score,
            "task_type": task_type,
            "evaluated_models": evaluated,
            "total_time": time.perf_counter() - self.start_time,
            "time_budget": self.time_budget,
        }
        return self.results_

    def _get_candidate_models(self, task_type: str) -> Dict[str, BaseEstimator]:
        rs = self.random_state
        if task_type == "classification":
            return {
                "logistic": LogisticRegression(random_state=rs, max_iter=1000),
                "random_forest": RandomForestClassifier(random_state=rs, n_jobs=self.n_jobs),
                "gradient_boosting": GradientBoostingClassifier(random_state=rs),
                "svm": SVC(random_state=rs),
                "naive_bayes": GaussianNB(),
            }
        return {
            "linear": LinearRegression(),
            "ridge": Ridge(random_state=rs),
            "random_forest": RandomForestRegressor(random_state=rs, n_jobs=self.n_jobs),
            "gradient_boosting": GradientBoostingRegressor(random_state=rs),
            "svr": SVR(),
        }

    @staticmethod
    def _prioritize_models(models: Dict[str, BaseEstimator], data_shape: Tuple[int, ...]) -> List[str]:
        n_samples = data_shape[0]
        priority = ["naive_bayes", "logistic", "linear", "ridge", "random_forest", "gradient_boosting"]
        if n_samples <= 5000:
            priority.extend(["svm", "svr"])
        order = [name for name in priority if name in models]
        order.extend(name for name in models if name not in order)
        return order

    def _quick_evaluate(self, model: BaseEstimator, X: ArrayLike, y: ArrayLike, task_type: str) -> float:
        scoring = "accuracy" if task_type == "classification" else "r2"
        return float(
            np.mean(
                cross_val_score(
                    model,
                    X,
                    y,
                    cv=_make_cv(3, model, y, self.random_state),
                    scoring=scoring,
                    n_jobs=self.n_jobs,
                )
            )
        )

    def _optimize_model(self, model: BaseEstimator, X: ArrayLike, y: ArrayLike, task_type: str) -> float:
        grids = {
            "RandomForestClassifier": {"n_estimators": [50, 100], "max_depth": [5, 10, None]},
            "RandomForestRegressor": {"n_estimators": [50, 100], "max_depth": [5, 10, None]},
            "Ridge": {"alpha": [0.1, 1.0, 10.0]},
            "SVC": {"C": [0.1, 1.0, 10.0]},
            "SVR": {"C": [0.1, 1.0, 10.0]},
        }
        grid = grids.get(type(model).__name__)
        if not grid:
            return self._quick_evaluate(model, X, y, task_type)
        scoring = "accuracy" if task_type == "classification" else "r2"
        search = GridSearchCV(
            model, grid, cv=_make_cv(3, model, y, self.random_state), scoring=scoring, n_jobs=self.n_jobs
        )
        search.fit(X, y)
        return float(search.best_score_)

    def _time_remaining(self) -> float:
        if self.start_time is None:
            return float(self.time_budget)
        return max(0.0, self.time_budget - (time.perf_counter() - self.start_time))


__all__ = [
    "HAS_OPTUNA",
    "AdvancedModelSelector",
    "AutoMLSelector",
    "AutoModelSelector",
    "BayesianOptimizer",
    "CrossValidationPipeline",
    "DynamicEnsembleSelector",
    "GridSearchPipeline",
    "HyperparameterOptimizer",
    "LearningCurveAnalyzer",
    "ModelComparator",
    "ModelEnsemblePipeline",
    "ModelRegistry",
    "ModelSelectionPipeline",
    "MultiObjectiveSelector",
    "NestedCrossValidation",
    "PerformanceTracker",
    "ValidationCurveAnalyzer",
    "WeightedEnsembleClassifier",
    "WeightedEnsembleRegressor",
]
