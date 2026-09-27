"""Classification model wrappers with a uniform, sklearn-compatible interface.

Every wrapper in this module derives from :class:`ClassificationModel`, which
is itself a proper scikit-learn estimator (``BaseEstimator`` +
``ClassifierMixin``). A wrapper stores its constructor arguments unchanged,
builds the underlying scikit-learn estimator lazily, and delegates
``predict``/``predict_proba``/``decision_function`` to it. Because the
wrapper honours ``get_params``/``set_params``, it can be cloned, placed in a
``Pipeline``, or passed straight to ``GridSearchCV`` and ``cross_validate``.

On top of the scikit-learn API the wrappers add a small research workflow:
``train`` (alias of ``fit``), ``evaluate`` (a dictionary of standard
classification metrics), ``cross_validate``, ``tune_hyperparameters``,
``get_feature_importance``, ``save_model`` / ``load_model`` and
``get_hyperparameter_grid`` (a sensible default search space per algorithm).

Some scikit-learn constructor arguments have been deprecated across releases
(``LogisticRegression(penalty=...)`` and ``SVC(probability=...)`` in 1.8/1.9).
The wrappers keep those arguments for API stability and translate them to the
supported equivalents of the installed scikit-learn version.

Gradient-boosting libraries are optional: ``XGBoostClassifierModel`` and
``LightGBMClassifierModel`` are always importable, but constructing their
underlying estimator raises ``ImportError`` unless :data:`HAS_XGBOOST` /
:data:`HAS_LIGHTGBM` is ``True``.
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union

import joblib
import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier as _SkRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_validate as _sk_cross_validate
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from sklearn_mastery.config.logging_config import LoggerMixin, get_logger

try:  # optional dependency
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover - exercised only without xgboost installed
    XGBClassifier = None  # type: ignore[assignment,misc]
    HAS_XGBOOST = False

try:  # optional dependency
    from lightgbm import LGBMClassifier

    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover - exercised only without lightgbm installed
    LGBMClassifier = None  # type: ignore[assignment,misc]
    HAS_LIGHTGBM = False

logger = get_logger(__name__)

ParamGrid = Dict[str, Sequence[Any]]


# --------------------------------------------------------------------------- helpers
def _param_default(estimator_cls: Type[Any], name: str) -> Any:
    """Return the default of constructor parameter ``name`` (``inspect.Parameter.empty`` if absent)."""
    params = inspect.signature(estimator_cls.__init__).parameters
    return params[name].default if name in params else inspect.Parameter.empty


def _is_deprecated(estimator_cls: Type[Any], name: str) -> bool:
    """True when scikit-learn marks constructor parameter ``name`` with the ``"deprecated"`` sentinel."""
    return _param_default(estimator_cls, name) == "deprecated"


def _delegate_has(attr: str) -> Callable[[Any], bool]:
    """Build an ``available_if`` predicate that checks the underlying estimator for ``attr``.

    The predicate looks at the fitted estimator when available, otherwise at a
    freshly built template. It never raises so that ``hasattr`` stays usable.
    """

    def check(self: ClassificationModel) -> bool:
        try:
            return hasattr(self.model, attr)
        except NotImplementedError:
            return False

    return check


# ----------------------------------------------------------------------- base class
class ClassificationModel(ClassifierMixin, BaseEstimator, LoggerMixin):
    """Base class for all classification wrappers.

    Subclasses implement :meth:`_build_estimator`, which returns an *unfitted*
    scikit-learn classifier configured from the wrapper's constructor
    parameters. Everything else (fitting, prediction, evaluation, tuning,
    persistence) is provided here.

    Constructor parameters of a subclass must be stored verbatim on ``self``
    under the same name (scikit-learn convention) so that ``get_params`` /
    ``set_params`` and ``clone`` work.

    Attributes:
        model_: The fitted underlying scikit-learn estimator.
        classes_: Class labels seen during :meth:`fit`.
        n_features_in_: Number of features seen during :meth:`fit`.
        feature_names_in_: Feature names when ``X`` is a DataFrame.

    Raises:
        NotImplementedError: When ``train``/``predict`` are called on the
            abstract base class itself.
    """

    #: Set to ``True`` for estimators that require integer labels ``0..n_classes-1``.
    _encode_labels: ClassVar[bool] = False

    # ----------------------------------------------------------- abstract hooks
    def _build_estimator(self) -> Any:
        """Return an unfitted scikit-learn classifier configured from ``self`` params.

        Raises:
            NotImplementedError: Always, on the abstract base class.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement `_build_estimator()`.")

    def get_hyperparameter_grid(self) -> ParamGrid:
        """Return a default hyperparameter search space for :meth:`tune_hyperparameters`.

        Returns:
            Mapping of wrapper parameter names to candidate values. Empty for
            the base class.
        """
        return {}

    # ------------------------------------------------------------- properties
    @property
    def model(self) -> Any:
        """The underlying scikit-learn estimator.

        Returns the fitted estimator after :meth:`fit`; before that a fresh,
        unfitted estimator built from the current parameters (mutating it has
        no effect on subsequent fits - use :meth:`set_params` instead).
        """
        fitted = getattr(self, "model_", None)
        return fitted if fitted is not None else self._build_estimator()

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called successfully."""
        return getattr(self, "model_", None) is not None

    def __sklearn_is_fitted__(self) -> bool:
        return self.is_fitted

    @property
    def feature_importances_(self) -> np.ndarray:
        """Impurity-based feature importances of the fitted estimator, when available."""
        return np.asarray(self._fitted().feature_importances_)

    @property
    def coef_(self) -> np.ndarray:
        """Coefficients of the fitted linear estimator, when available."""
        return np.asarray(self._fitted().coef_)

    @property
    def intercept_(self) -> np.ndarray:
        """Intercept of the fitted linear estimator, when available."""
        return np.asarray(self._fitted().intercept_)

    # --------------------------------------------------------------- internals
    def _require_concrete(self) -> None:
        if type(self)._build_estimator is ClassificationModel._build_estimator:
            raise NotImplementedError(
                "ClassificationModel is abstract; use a concrete subclass such as LogisticRegressionModel."
            )

    def _fitted(self) -> Any:
        """Return the fitted underlying estimator or raise."""
        self._require_concrete()
        check_is_fitted(self)
        return self.model_

    def _encode(self, y: ArrayLike) -> np.ndarray:
        if not self._encode_labels:
            return np.asarray(y)
        self.label_encoder_ = LabelEncoder().fit(y)
        return self.label_encoder_.transform(y)

    def _decode(self, y_pred: np.ndarray) -> np.ndarray:
        if self._encode_labels and getattr(self, "label_encoder_", None) is not None:
            return self.label_encoder_.inverse_transform(np.asarray(y_pred).astype(int))
        return np.asarray(y_pred)

    # --------------------------------------------------------------- training
    def fit(self, X: ArrayLike, y: ArrayLike, **fit_params: Any) -> ClassificationModel:
        """Fit the underlying estimator.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target labels of shape ``(n_samples,)``.
            **fit_params: Forwarded to the underlying estimator's ``fit``.

        Returns:
            The fitted wrapper (``self``).

        Raises:
            NotImplementedError: On the abstract base class.
        """
        self._require_concrete()
        estimator = self._build_estimator()
        y_fit = self._encode(y)
        self.logger.debug("Fitting %s on X%s", type(estimator).__name__, getattr(X, "shape", ""))
        start = time.perf_counter()
        estimator.fit(X, y_fit, **fit_params)
        self.training_time_ = time.perf_counter() - start
        self.model_ = estimator
        self.classes_ = (
            self.label_encoder_.classes_ if self._encode_labels else np.asarray(estimator.classes_)
        )
        self.n_features_in_ = int(getattr(estimator, "n_features_in_", np.shape(X)[1]))
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def train(self, X: ArrayLike, y: ArrayLike, **fit_params: Any) -> ClassificationModel:
        """Alias of :meth:`fit` kept for the project's tutorial API.

        Args:
            X: Feature matrix.
            y: Target labels.
            **fit_params: Forwarded to :meth:`fit`.

        Returns:
            The fitted wrapper (``self``).
        """
        return self.fit(X, y, **fit_params)

    # ------------------------------------------------------------- prediction
    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predicted labels of shape ``(n_samples,)``.

        Raises:
            NotImplementedError: On the abstract base class.
            NotFittedError: When called before :meth:`fit`.
        """
        return self._decode(self._fitted().predict(X))

    @available_if(_delegate_has("predict_proba"))
    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Predict class probabilities (only when the underlying estimator supports it).

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Probabilities of shape ``(n_samples, n_classes)``, columns ordered as ``classes_``.
        """
        return np.asarray(self._fitted().predict_proba(X))

    @available_if(_delegate_has("predict_log_proba"))
    def predict_log_proba(self, X: ArrayLike) -> np.ndarray:
        """Predict log class probabilities (only when the underlying estimator supports it)."""
        return np.asarray(self._fitted().predict_log_proba(X))

    @available_if(_delegate_has("decision_function"))
    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Evaluate the decision function (only when the underlying estimator supports it)."""
        return np.asarray(self._fitted().decision_function(X))

    # ------------------------------------------------------------- evaluation
    def evaluate(self, X: ArrayLike, y: ArrayLike, average: Optional[str] = None) -> Dict[str, Any]:
        """Compute standard classification metrics on ``(X, y)``.

        Args:
            X: Feature matrix.
            y: True labels.
            average: Averaging strategy for precision/recall/F1. ``None`` selects
                ``"binary"`` for two classes and ``"macro"`` otherwise.

        Returns:
            Dictionary with ``accuracy``, ``balanced_accuracy``, ``precision``,
            ``recall``, ``f1``, ``confusion_matrix`` (ndarray),
            ``classification_report`` (nested dict) and, when probabilities
            are available, ``roc_auc``.
        """
        y_true = np.asarray(y)
        y_pred = self.predict(X)
        n_classes = len(self.classes_)
        if average is None:
            average = "binary" if n_classes == 2 else "macro"
        pos_label: Any = self.classes_[-1] if average == "binary" else 1

        metrics: Dict[str, Any] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0)
            ),
            "f1": float(f1_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=self.classes_),
            "classification_report": classification_report(
                y_true, y_pred, labels=self.classes_, output_dict=True, zero_division=0
            ),
        }
        if hasattr(self, "predict_proba"):
            try:
                proba = self.predict_proba(X)
                if n_classes == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
                else:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=self.classes_)
                    )
            except ValueError as exc:  # e.g. a class missing from y
                self.logger.debug("ROC AUC unavailable: %s", exc)
        self.logger.info(
            "%s evaluation: accuracy=%.4f f1=%.4f", type(self).__name__, metrics["accuracy"], metrics["f1"]
        )
        return metrics

    def cross_validate(
        self,
        X: ArrayLike,
        y: ArrayLike,
        cv: Any = 5,
        scoring: Union[str, Sequence[str], None] = None,
        n_jobs: Optional[int] = None,
        return_train_score: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Cross-validate a clone of this wrapper.

        Args:
            X: Feature matrix.
            y: Target labels.
            cv: Number of folds or a cross-validation splitter.
            scoring: Scorer name(s); defaults to the estimator's ``score`` (accuracy).
            n_jobs: Parallel jobs for the folds.
            return_train_score: Whether to include training-fold scores.

        Returns:
            Dictionary of per-fold arrays as produced by
            :func:`sklearn.model_selection.cross_validate` (``test_score``,
            ``train_score``, ``fit_time``, ``score_time``).
        """
        self._require_concrete()
        results = _sk_cross_validate(
            clone(self), X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, return_train_score=return_train_score
        )
        self.logger.info(
            "%s CV (%s folds): mean test score=%.4f",
            type(self).__name__,
            len(results["fit_time"]),
            float(np.mean(results.get("test_score", np.nan))),
        )
        return results

    def tune_hyperparameters(
        self,
        X: ArrayLike,
        y: ArrayLike,
        param_grid: Optional[ParamGrid] = None,
        cv: Any = 5,
        scoring: Optional[str] = None,
        n_iter: Optional[int] = None,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """Search hyperparameters with cross-validation and adopt the best configuration.

        The search runs on a clone; afterwards ``self`` is updated with the
        best parameters and holds the refitted best estimator.

        Args:
            X: Feature matrix.
            y: Target labels.
            param_grid: Wrapper parameter names to candidate values. Defaults
                to :meth:`get_hyperparameter_grid`.
            cv: Number of folds or a splitter.
            scoring: Scorer name; defaults to accuracy.
            n_iter: When given, use randomized search with this many draws
                instead of an exhaustive grid.
            n_jobs: Parallel jobs.
            random_state: Seed for randomized search.

        Returns:
            ``(best_params, best_score)``.

        Raises:
            ValueError: If no search space is available.
        """
        self._require_concrete()
        grid = dict(param_grid) if param_grid is not None else self.get_hyperparameter_grid()
        if not grid:
            raise ValueError(f"No parameter grid supplied and {type(self).__name__} defines no default grid.")

        search: Union[GridSearchCV, RandomizedSearchCV]
        if n_iter is None:
            search = GridSearchCV(clone(self), grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
        else:
            search = RandomizedSearchCV(
                clone(self),
                grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        search.fit(X, y)

        self.set_params(**search.best_params_)
        best = search.best_estimator_
        for attr in ("model_", "classes_", "n_features_in_", "feature_names_in_", "label_encoder_"):
            if hasattr(best, attr):
                setattr(self, attr, getattr(best, attr))
        self.best_params_ = dict(search.best_params_)
        self.best_score_ = float(search.best_score_)
        self.cv_results_ = search.cv_results_
        self.logger.info(
            "%s tuning: best score=%.4f params=%s", type(self).__name__, self.best_score_, self.best_params_
        )
        return self.best_params_, self.best_score_

    # ----------------------------------------------------------- introspection
    def get_feature_importance(self, normalize: bool = True) -> np.ndarray:
        """Return per-feature importance scores.

        Uses ``feature_importances_`` when the estimator exposes it, otherwise
        the mean absolute value of ``coef_`` across classes.

        Args:
            normalize: Scale the scores to sum to one (when their sum is positive).

        Returns:
            Array of shape ``(n_features,)`` with non-negative scores.

        Raises:
            AttributeError: If the estimator exposes neither attribute.
        """
        estimator = self._fitted()
        if hasattr(estimator, "feature_importances_"):
            scores = np.asarray(estimator.feature_importances_, dtype=float)
        elif hasattr(estimator, "coef_"):
            scores = np.abs(np.atleast_2d(np.asarray(estimator.coef_, dtype=float))).mean(axis=0)
        else:
            raise AttributeError(
                f"{type(estimator).__name__} exposes neither feature_importances_ nor coef_."
            )
        total = scores.sum()
        if normalize and total > 0:
            scores = scores / total
        return scores

    def get_top_features(
        self, k: int = 10, feature_names: Optional[Sequence[str]] = None
    ) -> List[Tuple[Union[int, str], float]]:
        """Return the ``k`` most important features as ``(feature, score)`` pairs.

        Args:
            k: Number of features to return.
            feature_names: Names to report instead of indices; defaults to
                ``feature_names_in_`` when fitted on a DataFrame.

        Returns:
            List of ``(name_or_index, score)`` sorted by descending score.
        """
        scores = self.get_feature_importance()
        names: Sequence[Union[int, str]]
        if feature_names is not None:
            names = list(feature_names)
        elif getattr(self, "feature_names_in_", None) is not None:
            names = list(self.feature_names_in_)
        else:
            names = list(range(len(scores)))
        order = np.argsort(scores)[::-1][:k]
        return [(names[i], float(scores[i])) for i in order]

    # ------------------------------------------------------------- persistence
    def save_model(self, filepath: Union[str, Path]) -> Path:
        """Persist the fitted wrapper with joblib.

        Args:
            filepath: Destination path; parent directories are created.

        Returns:
            The path written.
        """
        self._fitted()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        self.logger.info("Saved %s to %s", type(self).__name__, path)
        return path

    def load_model(self, filepath: Union[str, Path]) -> ClassificationModel:
        """Load a wrapper saved with :meth:`save_model` into ``self``.

        Args:
            filepath: Path of the saved model.

        Returns:
            ``self`` with parameters and fitted state replaced.

        Raises:
            TypeError: If the file holds a different wrapper class.
        """
        loaded = self.load(filepath)
        if not isinstance(loaded, type(self)):
            raise TypeError(f"{filepath} holds a {type(loaded).__name__}, not a {type(self).__name__}.")
        self.__dict__.update(loaded.__dict__)
        return self

    @staticmethod
    def load(filepath: Union[str, Path]) -> ClassificationModel:
        """Load and return a wrapper saved with :meth:`save_model`.

        Args:
            filepath: Path of the saved model.

        Returns:
            The deserialized wrapper.

        Raises:
            TypeError: If the file does not hold a :class:`ClassificationModel`.
        """
        loaded = joblib.load(Path(filepath))
        if not isinstance(loaded, ClassificationModel):
            raise TypeError(
                f"{filepath} does not contain a ClassificationModel (found {type(loaded).__name__})."
            )
        return loaded


class _StagedMixin:
    """Staged (per-boosting-iteration) predictions for additive ensembles."""

    _fitted: Callable[[], Any]

    @available_if(_delegate_has("staged_predict"))
    def staged_predict(self, X: ArrayLike) -> Iterator[np.ndarray]:
        """Yield predictions after each boosting stage."""
        for pred in self._fitted().staged_predict(X):
            yield np.asarray(pred)

    @available_if(_delegate_has("staged_predict_proba"))
    def staged_predict_proba(self, X: ArrayLike) -> Iterator[np.ndarray]:
        """Yield class probabilities after each boosting stage."""
        for proba in self._fitted().staged_predict_proba(X):
            yield np.asarray(proba)

    @available_if(_delegate_has("staged_decision_function"))
    def staged_decision_function(self, X: ArrayLike) -> Iterator[np.ndarray]:
        """Yield decision-function values after each boosting stage."""
        for value in self._fitted().staged_decision_function(X):
            yield np.asarray(value)


# ------------------------------------------------------------------ linear
class LogisticRegressionModel(ClassificationModel):
    """Logistic regression wrapper.

    ``penalty`` is kept for API stability. On scikit-learn releases where
    ``LogisticRegression(penalty=...)`` is deprecated it is translated to the
    equivalent ``l1_ratio`` (``"l2"`` -> 0, ``"l1"`` -> 1, ``"elasticnet"`` ->
    the given ``l1_ratio``, ``None`` -> ``C=inf``).

    Args:
        penalty: Regularization type: ``"l1"``, ``"l2"``, ``"elasticnet"`` or ``None``.
        C: Inverse regularization strength.
        l1_ratio: Elastic-net mixing parameter (used when ``penalty="elasticnet"``).
        solver: Optimization algorithm.
        max_iter: Maximum number of solver iterations.
        tol: Stopping tolerance.
        fit_intercept: Whether to fit an intercept term.
        intercept_scaling: Intercept scaling for ``liblinear``.
        class_weight: Class weights or ``"balanced"``.
        dual: Dual formulation (``liblinear`` with L2 only).
        warm_start: Reuse the previous solution as initialization.
        n_jobs: Parallel jobs (solver dependent).
        verbose: Verbosity level.
        random_state: Seed for stochastic solvers.
    """

    def __init__(
        self,
        penalty: Optional[str] = "l2",
        C: float = 1.0,
        l1_ratio: Optional[float] = None,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        tol: float = 1e-4,
        fit_intercept: bool = True,
        intercept_scaling: float = 1.0,
        class_weight: Union[Dict[Any, float], str, None] = None,
        dual: bool = False,
        warm_start: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: Optional[int] = None,
    ) -> None:
        self.penalty = penalty
        self.C = C
        self.l1_ratio = l1_ratio
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.dual = dual
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def _build_estimator(self) -> LogisticRegression:
        params: Dict[str, Any] = dict(
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            class_weight=self.class_weight,
            dual=self.dual,
            warm_start=self.warm_start,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
        )
        if _is_deprecated(LogisticRegression, "penalty"):
            if self.penalty is None:
                params["C"] = np.inf
                params["l1_ratio"] = 0.0
            elif self.penalty == "l2":
                params["l1_ratio"] = 0.0
            elif self.penalty == "l1":
                params["l1_ratio"] = 1.0
            elif self.penalty == "elasticnet":
                if self.l1_ratio is None:
                    raise ValueError("penalty='elasticnet' requires l1_ratio in [0, 1].")
                params["l1_ratio"] = self.l1_ratio
            else:
                raise ValueError(f"Unknown penalty {self.penalty!r}.")
        else:  # pragma: no cover - older scikit-learn
            params["penalty"] = self.penalty
            params["l1_ratio"] = self.l1_ratio
        return LogisticRegression(**params)

    def get_hyperparameter_grid(self) -> ParamGrid:
        return {"C": [0.01, 0.1, 1.0, 10.0, 100.0], "penalty": ["l2"]}

    def get_coefficients(self) -> np.ndarray:
        """Return the fitted coefficients of shape ``(n_classes_or_1, n_features)``."""
        return self.coef_

    def get_intercept(self) -> np.ndarray:
        """Return the fitted intercept(s)."""
        return self.intercept_


# ------------------------------------------------------------------- trees
class DecisionTreeClassifierModel(ClassificationModel):
    """Decision tree wrapper (``sklearn.tree.DecisionTreeClassifier``).

    Args:
        criterion: Split quality function.
        splitter: Split strategy.
        max_depth: Maximum tree depth.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required at a leaf.
        min_weight_fraction_leaf: Minimum weighted fraction at a leaf.
        max_features: Number of features considered per split.
        max_leaf_nodes: Maximum number of leaves.
        min_impurity_decrease: Minimum impurity decrease for a split.
        class_weight: Class weights or ``"balanced"``.
        ccp_alpha: Cost-complexity pruning parameter.
        random_state: Seed controlling feature permutation.
    """

    def __init__(
        self,
        criterion: str = "gini",
        splitter: str = "best",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[int, float, str, None] = None,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        class_weight: Union[Dict[Any, float], str, None] = None,
        ccp_alpha: float = 0.0,
        random_state: Optional[int] = None,
    ) -> None:
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state

    def _build_estimator(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(**self.get_params(deep=False))

    def get_hyperparameter_grid(self) -> ParamGrid:
        return {
            "max_depth": [None, 3, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"],
        }

    def get_depth(self) -> int:
        """Return the depth of the fitted tree."""
        return int(self._fitted().get_depth())

    def get_n_leaves(self) -> int:
        """Return the number of leaves of the fitted tree."""
        return int(self._fitted().get_n_leaves())


class _ForestModel(ClassificationModel):
    """Shared implementation for random-forest style ensembles."""

    _estimator_cls: ClassVar[Type[Any]]

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[int, float, str, None] = "sqrt",
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        class_weight: Union[Dict[Any, float], str, None] = None,
        ccp_alpha: float = 0.0,
        max_samples: Union[int, float, None] = None,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        warm_start: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.warm_start = warm_start
        self.random_state = random_state

    def _build_estimator(self) -> Any:
        return self._estimator_cls(**self.get_params(deep=False))

    def get_hyperparameter_grid(self) -> ParamGrid:
        return {
            "n_estimators": [100, 200, 500],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"],
        }

    def get_oob_score(self) -> float:
        """Return the out-of-bag accuracy.

        Raises:
            AttributeError: If the model was fitted with ``oob_score=False``.
        """
        estimator = self._fitted()
        if not getattr(estimator, "oob_score", False):
            raise AttributeError("Out-of-bag score requires oob_score=True at construction.")
        return float(estimator.oob_score_)


class RandomForestClassifierModel(_ForestModel):
    """Random forest wrapper (``sklearn.ensemble.RandomForestClassifier``).

    Args:
        n_estimators: Number of trees.
        criterion: Split quality function.
        max_depth: Maximum tree depth.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required at a leaf.
        min_weight_fraction_leaf: Minimum weighted fraction at a leaf.
        max_features: Number of features considered per split.
        max_leaf_nodes: Maximum number of leaves per tree.
        min_impurity_decrease: Minimum impurity decrease for a split.
        bootstrap: Draw bootstrap samples per tree.
        oob_score: Estimate generalisation accuracy on out-of-bag samples.
        class_weight: Class weights, ``"balanced"`` or ``"balanced_subsample"``.
        ccp_alpha: Cost-complexity pruning parameter.
        max_samples: Bootstrap sample size.
        n_jobs: Parallel jobs.
        verbose: Verbosity level.
        warm_start: Add trees to the existing forest on refit.
        random_state: Seed for bootstrapping and feature sampling.
    """

    _estimator_cls = _SkRandomForestClassifier


class ExtraTreesClassifierModel(_ForestModel):
    """Extremely randomized trees wrapper (``sklearn.ensemble.ExtraTreesClassifier``).

    Accepts the same arguments as :class:`RandomForestClassifierModel`;
    ``bootstrap`` defaults to ``False`` as in scikit-learn.
    """

    _estimator_cls = ExtraTreesClassifier

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[int, float, str, None] = "sqrt",
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = False,
        oob_score: bool = False,
        class_weight: Union[Dict[Any, float], str, None] = None,
        ccp_alpha: float = 0.0,
        max_samples: Union[int, float, None] = None,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        warm_start: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            n_jobs=n_jobs,
            verbose=verbose,
            warm_start=warm_start,
            random_state=random_state,
        )


# ---------------------------------------------------------------- boosting
class GradientBoostingClassifierModel(_StagedMixin, ClassificationModel):
    """Gradient boosting wrapper (``sklearn.ensemble.GradientBoostingClassifier``).

    Args:
        n_estimators: Number of boosting stages.
        learning_rate: Shrinkage applied to each tree.
        max_depth: Depth of the individual regression trees.
        subsample: Fraction of samples used per stage.
        loss: Loss function to optimise.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required at a leaf.
        min_weight_fraction_leaf: Minimum weighted fraction at a leaf.
        max_features: Number of features considered per split.
        max_leaf_nodes: Maximum number of leaves per tree.
        min_impurity_decrease: Minimum impurity decrease for a split.
        init: Initial estimator.
        validation_fraction: Hold-out fraction for early stopping.
        n_iter_no_change: Patience for early stopping (``None`` disables it).
        tol: Early-stopping tolerance.
        ccp_alpha: Cost-complexity pruning parameter.
        warm_start: Add stages to the existing ensemble on refit.
        verbose: Verbosity level.
        random_state: Seed for subsampling and tree building.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: Optional[int] = 3,
        subsample: float = 1.0,
        loss: str = "log_loss",
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[int, float, str, None] = None,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        init: Any = None,
        validation_fraction: float = 0.1,
        n_iter_no_change: Optional[int] = None,
        tol: float = 1e-4,
        ccp_alpha: float = 0.0,
        warm_start: bool = False,
        verbose: int = 0,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.loss = loss
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.init = init
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.ccp_alpha = ccp_alpha
        self.warm_start = warm_start
        self.verbose = verbose
        self.random_state = random_state

    def _build_estimator(self) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(**self.get_params(deep=False))

    def get_hyperparameter_grid(self) -> ParamGrid:
        return {
            "n_estimators": [100, 200, 500],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 5],
            "subsample": [0.8, 1.0],
        }


class AdaBoostClassifierModel(_StagedMixin, ClassificationModel):
    """AdaBoost wrapper (``sklearn.ensemble.AdaBoostClassifier``).

    Args:
        estimator: Base learner; scikit-learn defaults to a depth-1 tree.
        n_estimators: Maximum number of boosting rounds.
        learning_rate: Weight applied to each classifier per round.
        random_state: Seed passed to the base learners.
    """

    def __init__(
        self,
        estimator: Any = None,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _build_estimator(self) -> AdaBoostClassifier:
        params = self.get_params(deep=False)
        if params["estimator"] is not None:
            params["estimator"] = clone(params["estimator"])
        return AdaBoostClassifier(**params)

    def get_hyperparameter_grid(self) -> ParamGrid:
        return {"n_estimators": [50, 100, 200], "learning_rate": [0.1, 0.5, 1.0]}


class XGBoostClassifierModel(ClassificationModel):
    """XGBoost wrapper (``xgboost.XGBClassifier``); requires the optional ``xgboost`` package.

    Labels are internally encoded to ``0..n_classes-1`` and decoded on
    prediction, so arbitrary label values are accepted.

    Args:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate (``eta``).
        subsample: Row subsampling ratio per round.
        colsample_bytree: Column subsampling ratio per tree.
        min_child_weight: Minimum sum of instance weights in a child.
        gamma: Minimum loss reduction required to split.
        reg_alpha: L1 regularization on weights.
        reg_lambda: L2 regularization on weights.
        scale_pos_weight: Positive-class weight for imbalanced binary tasks.
        objective: XGBoost objective; ``None`` lets XGBoost infer it.
        eval_metric: Evaluation metric; ``None`` uses XGBoost's default.
        tree_method: Tree construction algorithm.
        n_jobs: Parallel threads.
        verbosity: XGBoost verbosity level.
        random_state: Seed.

    Raises:
        ImportError: On estimator construction when ``xgboost`` is missing.
    """

    _encode_labels = True

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        scale_pos_weight: float = 1.0,
        objective: Optional[str] = None,
        eval_metric: Optional[str] = None,
        tree_method: Optional[str] = None,
        n_jobs: Optional[int] = None,
        verbosity: int = 0,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.objective = objective
        self.eval_metric = eval_metric
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.random_state = random_state

    def _build_estimator(self) -> Any:
        if not HAS_XGBOOST:
            raise ImportError(
                "XGBoostClassifierModel requires xgboost: pip install 'sklearn-mastery[boosting]'."
            )
        params = {k: v for k, v in self.get_params(deep=False).items() if v is not None}
        return XGBClassifier(**params)

    def get_hyperparameter_grid(self) -> ParamGrid:
        return {
            "n_estimators": [100, 200, 500],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }


class LightGBMClassifierModel(ClassificationModel):
    """LightGBM wrapper (``lightgbm.LGBMClassifier``); requires the optional ``lightgbm`` package.

    Args:
        n_estimators: Number of boosting rounds.
        num_leaves: Maximum leaves per tree.
        max_depth: Maximum tree depth (``-1`` for unlimited).
        learning_rate: Boosting learning rate.
        boosting_type: ``"gbdt"``, ``"dart"`` or ``"rf"``.
        subsample: Row subsampling ratio.
        subsample_freq: Frequency of row subsampling (``0`` disables it).
        colsample_bytree: Column subsampling ratio per tree.
        min_child_samples: Minimum samples per leaf.
        min_child_weight: Minimum hessian sum per leaf.
        min_split_gain: Minimum gain to split.
        reg_alpha: L1 regularization.
        reg_lambda: L2 regularization.
        class_weight: Class weights or ``"balanced"``.
        objective: LightGBM objective; ``None`` infers it.
        importance_type: ``"split"`` or ``"gain"``.
        n_jobs: Parallel threads.
        verbose: LightGBM verbosity (``-1`` silences it).
        random_state: Seed.

    Raises:
        ImportError: On estimator construction when ``lightgbm`` is missing.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        boosting_type: str = "gbdt",
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        min_child_samples: int = 20,
        min_child_weight: float = 1e-3,
        min_split_gain: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        class_weight: Union[Dict[Any, float], str, None] = None,
        objective: Optional[str] = None,
        importance_type: str = "split",
        n_jobs: Optional[int] = None,
        verbose: int = -1,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.boosting_type = boosting_type
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.min_child_samples = min_child_samples
        self.min_child_weight = min_child_weight
        self.min_split_gain = min_split_gain
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.class_weight = class_weight
        self.objective = objective
        self.importance_type = importance_type
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def _build_estimator(self) -> Any:
        if not HAS_LIGHTGBM:
            raise ImportError(
                "LightGBMClassifierModel requires lightgbm: pip install 'sklearn-mastery[boosting]'."
            )
        return LGBMClassifier(**self.get_params(deep=False))

    def get_hyperparameter_grid(self) -> ParamGrid:
        return {
            "n_estimators": [100, 200, 500],
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.01, 0.05, 0.1],
            "min_child_samples": [10, 20, 50],
        }


# -------------------------------------------------------------------- kernel
class SVMClassifierModel(ClassificationModel):
    """Support vector machine wrapper (``sklearn.svm.SVC``).

    ``probability=True`` enables :meth:`predict_proba`. On scikit-learn
    releases where ``SVC(probability=...)`` is deprecated the wrapper fits
    ``CalibratedClassifierCV(SVC(...), method="sigmoid", ensemble=False)``,
    the documented replacement (Platt scaling with 5-fold CV).

    Args:
        C: Regularization parameter.
        kernel: Kernel type.
        degree: Degree of the polynomial kernel.
        gamma: Kernel coefficient.
        coef0: Independent term for ``poly``/``sigmoid`` kernels.
        shrinking: Use the shrinking heuristic.
        probability: Enable probability estimates.
        tol: Stopping tolerance.
        cache_size: Kernel cache size in MB.
        class_weight: Class weights or ``"balanced"``.
        max_iter: Hard iteration limit (``-1`` for none).
        decision_function_shape: ``"ovr"`` or ``"ovo"``.
        break_ties: Break ties by decision-function confidence.
        verbose: Enable libsvm verbose output.
        random_state: Seed used for probability calibration.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: Union[str, float] = "scale",
        coef0: float = 0.0,
        shrinking: bool = True,
        probability: bool = False,
        tol: float = 1e-3,
        cache_size: float = 200,
        class_weight: Union[Dict[Any, float], str, None] = None,
        max_iter: int = -1,
        decision_function_shape: str = "ovr",
        break_ties: bool = False,
        verbose: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.verbose = verbose
        self.random_state = random_state

    def _build_estimator(self) -> Any:
        params = self.get_params(deep=False)
        probability = params.pop("probability")
        if not _is_deprecated(SVC, "probability"):  # pragma: no cover - older scikit-learn
            return SVC(probability=probability, **params)
        svc = SVC(**params)
        if not probability:
            return svc
        return CalibratedClassifierCV(svc, method="sigmoid", cv=5, ensemble=False)

    @property
    def svm_(self) -> SVC:
        """The fitted ``SVC`` (unwrapped from probability calibration if applicable)."""
        estimator = self._fitted()
        if isinstance(estimator, CalibratedClassifierCV):
            return estimator.calibrated_classifiers_[0].estimator
        return estimator

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Evaluate the SVM decision function (signed distance to the hyperplane)."""
        return np.asarray(self.svm_.decision_function(X))

    def get_support_vectors(self) -> np.ndarray:
        """Return the support vectors of shape ``(n_support_vectors, n_features)``."""
        return np.asarray(self.svm_.support_vectors_)

    def get_hyperparameter_grid(self) -> ParamGrid:
        return {"C": [0.1, 1.0, 10.0, 100.0], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]}


# --------------------------------------------------------------- neighbours
class KNNClassifierModel(ClassificationModel):
    """k-nearest neighbours wrapper (``sklearn.neighbors.KNeighborsClassifier``).

    Args:
        n_neighbors: Number of neighbours.
        weights: ``"uniform"``, ``"distance"`` or a callable.
        algorithm: Neighbour search algorithm.
        leaf_size: Leaf size for tree-based searches.
        p: Minkowski power parameter.
        metric: Distance metric.
        metric_params: Extra keyword arguments for the metric.
        n_jobs: Parallel jobs for neighbour search.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: Union[str, Callable[..., Any]] = "uniform",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: Union[str, Callable[..., Any]] = "minkowski",
        metric_params: Optional[Dict[str, Any]] = None,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def _build_estimator(self) -> KNeighborsClassifier:
        return KNeighborsClassifier(**self.get_params(deep=False))

    def get_hyperparameter_grid(self) -> ParamGrid:
        return {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"], "p": [1, 2]}


# ------------------------------------------------------------- naive bayes
class NaiveBayesModel(ClassificationModel):
    """Naive Bayes wrapper supporting the Gaussian, multinomial, Bernoulli and complement variants.

    Args:
        nb_type: ``"gaussian"``, ``"multinomial"``, ``"bernoulli"`` or ``"complement"``.
        priors: Class prior probabilities (Gaussian).
        var_smoothing: Variance smoothing fraction (Gaussian).
        alpha: Additive smoothing (multinomial/Bernoulli/complement).
        fit_prior: Learn class priors (multinomial/Bernoulli/complement).
        class_prior: Fixed class priors (multinomial/Bernoulli/complement).

    Raises:
        ValueError: On estimator construction for an unknown ``nb_type``.
    """

    _VARIANTS: ClassVar[Dict[str, Type[Any]]] = {
        "gaussian": GaussianNB,
        "multinomial": MultinomialNB,
        "bernoulli": BernoulliNB,
        "complement": ComplementNB,
    }

    def __init__(
        self,
        nb_type: str = "gaussian",
        priors: Optional[ArrayLike] = None,
        var_smoothing: float = 1e-9,
        alpha: float = 1.0,
        fit_prior: bool = True,
        class_prior: Optional[ArrayLike] = None,
    ) -> None:
        self.nb_type = nb_type
        self.priors = priors
        self.var_smoothing = var_smoothing
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def _build_estimator(self) -> Any:
        if self.nb_type not in self._VARIANTS:
            raise ValueError(f"nb_type must be one of {sorted(self._VARIANTS)}, got {self.nb_type!r}.")
        if self.nb_type == "gaussian":
            return GaussianNB(priors=self.priors, var_smoothing=self.var_smoothing)
        return self._VARIANTS[self.nb_type](
            alpha=self.alpha, fit_prior=self.fit_prior, class_prior=self.class_prior
        )

    def get_hyperparameter_grid(self) -> ParamGrid:
        if self.nb_type == "gaussian":
            return {"var_smoothing": np.logspace(-11, -5, 7).tolist()}
        return {"alpha": [0.01, 0.1, 0.5, 1.0, 2.0]}


# ---------------------------------------------------------- neural network
class NeuralNetworkClassifierModel(ClassificationModel):
    """Multi-layer perceptron wrapper (``sklearn.neural_network.MLPClassifier``).

    Args:
        hidden_layer_sizes: Neurons per hidden layer.
        activation: Hidden-layer activation function.
        solver: Weight optimisation algorithm.
        alpha: L2 regularization strength.
        batch_size: Minibatch size for stochastic solvers.
        learning_rate: Learning-rate schedule (``sgd`` only).
        learning_rate_init: Initial learning rate.
        power_t: Exponent for inverse scaling (``sgd`` only).
        max_iter: Maximum number of epochs.
        shuffle: Shuffle samples each iteration.
        tol: Optimisation tolerance.
        momentum: Momentum for ``sgd``.
        nesterovs_momentum: Use Nesterov momentum.
        early_stopping: Stop when validation score stops improving.
        validation_fraction: Hold-out fraction for early stopping.
        beta_1: Adam first-moment decay.
        beta_2: Adam second-moment decay.
        epsilon: Adam numerical stability constant.
        n_iter_no_change: Patience for early stopping.
        max_fun: Maximum function calls (``lbfgs`` only).
        warm_start: Reuse the previous solution as initialization.
        verbose: Print progress.
        random_state: Seed for weight initialization and shuffling.
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (100,),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 1e-4,
        batch_size: Union[int, str] = "auto",
        learning_rate: str = "constant",
        learning_rate_init: float = 1e-3,
        power_t: float = 0.5,
        max_iter: int = 200,
        shuffle: bool = True,
        tol: float = 1e-4,
        momentum: float = 0.9,
        nesterovs_momentum: bool = True,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        n_iter_no_change: int = 10,
        max_fun: int = 15000,
        warm_start: bool = False,
        verbose: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.tol = tol
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun
        self.warm_start = warm_start
        self.verbose = verbose
        self.random_state = random_state

    def _build_estimator(self) -> MLPClassifier:
        return MLPClassifier(**self.get_params(deep=False))

    def get_hyperparameter_grid(self) -> ParamGrid:
        return {
            "hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "alpha": [1e-4, 1e-3, 1e-2],
            "learning_rate_init": [1e-3, 1e-2],
        }

    @property
    def loss_curve_(self) -> List[float]:
        """Training loss per epoch (stochastic solvers only)."""
        return list(self._fitted().loss_curve_)


# ------------------------------------------------------- arbitrary estimator
class AdvancedClassifier(ClassificationModel):
    """Wrap any scikit-learn-compatible classifier in the project interface.

    Useful for estimators without a dedicated wrapper (e.g. a ``Pipeline`` or
    a third-party classifier) while keeping ``evaluate``, ``cross_validate``,
    ``tune_hyperparameters`` and persistence. Nested parameters are reachable
    through ``estimator__<name>`` in ``set_params`` and search grids.

    Args:
        estimator: An unfitted classifier; defaults to ``LogisticRegression``.
    """

    def __init__(self, estimator: Any = None) -> None:
        self.estimator = estimator

    def _build_estimator(self) -> Any:
        return clone(self.estimator) if self.estimator is not None else LogisticRegression(max_iter=1000)

    def get_hyperparameter_grid(self) -> ParamGrid:
        inner = self.estimator
        if isinstance(inner, ClassificationModel):
            return {f"estimator__{k}": v for k, v in inner.get_hyperparameter_grid().items()}
        return {}


# ----------------------------------------------------------------- factory
class ClassificationModels(LoggerMixin):
    """Factory and registry for classification wrappers.

    Example:
        >>> models = ClassificationModels()
        >>> rf = models.get_random_forest(n_estimators=200, random_state=42)
        >>> rf.fit(X, y).score(X, y)  # doctest: +SKIP
    """

    _REGISTRY: ClassVar[Dict[str, Type[ClassificationModel]]] = {
        "logistic_regression": LogisticRegressionModel,
        "decision_tree": DecisionTreeClassifierModel,
        "random_forest": RandomForestClassifierModel,
        "extra_trees": ExtraTreesClassifierModel,
        "gradient_boosting": GradientBoostingClassifierModel,
        "ada_boost": AdaBoostClassifierModel,
        "xgboost": XGBoostClassifierModel,
        "lightgbm": LightGBMClassifierModel,
        "svm": SVMClassifierModel,
        "knn": KNNClassifierModel,
        "naive_bayes": NaiveBayesModel,
        "neural_network": NeuralNetworkClassifierModel,
    }

    @classmethod
    def available_models(cls) -> List[str]:
        """Return the registered model names (optional back-ends included only when installed)."""
        names = []
        for name in cls._REGISTRY:
            if name == "xgboost" and not HAS_XGBOOST:
                continue
            if name == "lightgbm" and not HAS_LIGHTGBM:
                continue
            names.append(name)
        return names

    @classmethod
    def register_custom_model(cls, name: str, model_cls: Type[ClassificationModel]) -> None:
        """Register a wrapper class under ``name`` for :meth:`get_model`.

        Args:
            name: Registry key.
            model_cls: A :class:`ClassificationModel` subclass.

        Raises:
            TypeError: If ``model_cls`` is not a :class:`ClassificationModel` subclass.
        """
        if not (inspect.isclass(model_cls) and issubclass(model_cls, ClassificationModel)):
            raise TypeError("Custom models must subclass ClassificationModel.")
        cls._REGISTRY[name] = model_cls

    def _instantiate(self, model_cls: Type[ClassificationModel], **params: Any) -> ClassificationModel:
        """Build ``model_cls`` keeping only the constructor arguments it accepts.

        Mirrors :class:`RegressionModels`: unsupported keyword arguments (for
        example ``class_weight`` on a model without it) are dropped with a
        warning instead of raising ``TypeError``.
        """
        accepted = set(model_cls().get_params())
        dropped = sorted(set(params) - accepted)
        if dropped:
            self.logger.warning("%s ignores unsupported argument(s): %s", model_cls.__name__, dropped)
        kept = {k: v for k, v in params.items() if k in accepted}
        self.logger.debug("Creating %s with %s", model_cls.__name__, kept)
        return model_cls(**kept)

    def get_model(self, name: str, **params: Any) -> ClassificationModel:
        """Instantiate a registered wrapper by name.

        Args:
            name: Registry key (see :meth:`available_models`).
            **params: Constructor arguments; unsupported ones are dropped with
                a warning.

        Returns:
            A new, unfitted wrapper.

        Raises:
            KeyError: For an unknown name.
        """
        try:
            model_cls = self._REGISTRY[name]
        except KeyError as exc:
            raise KeyError(f"Unknown model {name!r}. Available: {sorted(self._REGISTRY)}") from exc
        return self._instantiate(model_cls, **params)

    def train_model(
        self, X: ArrayLike, y: ArrayLike, algorithm: str = "random_forest", **params: Any
    ) -> Tuple[ClassificationModel, float]:
        """Build and fit a wrapper in one call (parity with :meth:`RegressionModels.train_model`).

        Args:
            X: Training features.
            y: Training labels.
            algorithm: Registry key of the wrapper.
            **params: Constructor arguments for the wrapper.

        Returns:
            ``(fitted_model, training_time_seconds)``.
        """
        model = self.get_model(algorithm, **params)
        model.train(X, y)
        return model, float(model.training_time_)

    def get_logistic_regression(self, **params: Any) -> LogisticRegressionModel:
        """Create a :class:`LogisticRegressionModel`."""
        return LogisticRegressionModel(**params)

    def get_decision_tree(self, **params: Any) -> DecisionTreeClassifierModel:
        """Create a :class:`DecisionTreeClassifierModel`."""
        return DecisionTreeClassifierModel(**params)

    def get_random_forest(self, **params: Any) -> RandomForestClassifierModel:
        """Create a :class:`RandomForestClassifierModel`."""
        return RandomForestClassifierModel(**params)

    def get_extra_trees(self, **params: Any) -> ExtraTreesClassifierModel:
        """Create an :class:`ExtraTreesClassifierModel`."""
        return ExtraTreesClassifierModel(**params)

    def get_gradient_boosting(self, **params: Any) -> GradientBoostingClassifierModel:
        """Create a :class:`GradientBoostingClassifierModel`."""
        return GradientBoostingClassifierModel(**params)

    def get_ada_boost(self, **params: Any) -> AdaBoostClassifierModel:
        """Create an :class:`AdaBoostClassifierModel`."""
        return AdaBoostClassifierModel(**params)

    def get_xgboost(self, **params: Any) -> XGBoostClassifierModel:
        """Create an :class:`XGBoostClassifierModel` (requires ``xgboost``)."""
        if not HAS_XGBOOST:
            raise ImportError("xgboost is not installed: pip install 'sklearn-mastery[boosting]'.")
        return XGBoostClassifierModel(**params)

    def get_lightgbm(self, **params: Any) -> LightGBMClassifierModel:
        """Create a :class:`LightGBMClassifierModel` (requires ``lightgbm``)."""
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm is not installed: pip install 'sklearn-mastery[boosting]'.")
        return LightGBMClassifierModel(**params)

    def get_svm(self, **params: Any) -> SVMClassifierModel:
        """Create an :class:`SVMClassifierModel`."""
        return SVMClassifierModel(**params)

    def get_knn(self, **params: Any) -> KNNClassifierModel:
        """Create a :class:`KNNClassifierModel`."""
        return KNNClassifierModel(**params)

    def get_naive_bayes(self, **params: Any) -> NaiveBayesModel:
        """Create a :class:`NaiveBayesModel`."""
        return NaiveBayesModel(**params)

    def get_neural_network(self, **params: Any) -> NeuralNetworkClassifierModel:
        """Create a :class:`NeuralNetworkClassifierModel`."""
        return NeuralNetworkClassifierModel(**params)

    def get_all_models(self, random_state: Optional[int] = None) -> Dict[str, ClassificationModel]:
        """Instantiate every available wrapper with default settings.

        Args:
            random_state: Seed applied to every wrapper that accepts one.

        Returns:
            Mapping of registry name to unfitted wrapper.
        """
        models: Dict[str, ClassificationModel] = {}
        for name in self.available_models():
            model = self._REGISTRY[name]()
            if random_state is not None and "random_state" in model.get_params(deep=False):
                model.set_params(random_state=random_state)
            models[name] = model
        return models


# Short aliases used by scripts and the verification script.
RandomForestClassifier = RandomForestClassifierModel
SVMClassifier = SVMClassifierModel

__all__ = [
    "HAS_LIGHTGBM",
    "HAS_XGBOOST",
    "AdaBoostClassifierModel",
    "AdvancedClassifier",
    "ClassificationModel",
    "ClassificationModels",
    "DecisionTreeClassifierModel",
    "ExtraTreesClassifierModel",
    "GradientBoostingClassifierModel",
    "KNNClassifierModel",
    "LightGBMClassifierModel",
    "LogisticRegressionModel",
    "NaiveBayesModel",
    "NeuralNetworkClassifierModel",
    "RandomForestClassifier",
    "RandomForestClassifierModel",
    "SVMClassifier",
    "SVMClassifierModel",
    "XGBoostClassifierModel",
]
