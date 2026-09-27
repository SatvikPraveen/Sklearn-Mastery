"""Regression model wrappers with a uniform, scikit-learn compatible API.

Every wrapper in this module subclasses :class:`RegressionModel`, which is a
``BaseEstimator`` + ``RegressorMixin`` shell around an underlying scikit-learn
regressor.  Wrappers expose their hyperparameters as constructor arguments so
that :func:`sklearn.base.clone`, ``GridSearchCV`` and ``Pipeline`` all work,
while adding a small convenience layer (``train``, ``evaluate``,
``cross_validate``, ``tune_hyperparameters``, ``save_model``/``load_model``)
and pass-through access to ``coef_`` / ``feature_importances_``.

Gradient-boosting wrappers for XGBoost and LightGBM are available when the
optional dependencies are installed (see ``HAS_XGBOOST`` / ``HAS_LIGHTGBM``);
the module imports cleanly without them.

Example:
    >>> from sklearn.datasets import make_regression
    >>> from sklearn_mastery.models.supervised.regression import RidgeRegressionModel
    >>> X, y = make_regression(n_samples=100, n_features=5, random_state=0)
    >>> model = RidgeRegressionModel(alpha=1.0).train(X, y)
    >>> metrics = model.evaluate(X, y)
    >>> sorted(metrics)[:3]
    ['explained_variance', 'mae', 'max_error']
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union

import joblib
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate as _sk_cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from sklearn_mastery.config.logging_config import LoggerMixin, get_logger
from sklearn_mastery.config.settings import ModelDefaults, settings

try:  # optional dependency
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover - exercised only without xgboost installed
    XGBRegressor = None  # type: ignore[assignment,misc]
    HAS_XGBOOST = False

try:  # optional dependency
    from lightgbm import LGBMRegressor

    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover - exercised only without lightgbm installed
    LGBMRegressor = None  # type: ignore[assignment,misc]
    HAS_LIGHTGBM = False

logger = get_logger(__name__)

ParamGrid = Dict[str, Sequence[Any]]
PathLike = Union[str, Path]


# --------------------------------------------------------------------------- #
# Base wrapper
# --------------------------------------------------------------------------- #
class RegressionModel(RegressorMixin, BaseEstimator, LoggerMixin):
    """Base class for all regression wrappers.

    The class is a thin, clone-compatible shell around a scikit-learn
    regressor.  Concrete subclasses only need to implement
    :meth:`_build_estimator`, which turns the wrapper's constructor
    parameters into an unfitted scikit-learn estimator.  Everything else
    (training, prediction, evaluation, cross-validation, tuning and
    persistence) is provided here.

    ``RegressionModel`` itself is not usable for training: :meth:`train` and
    :meth:`predict` raise ``NotImplementedError`` until a subclass provides
    :meth:`_build_estimator`.

    Attributes:
        model_: The fitted underlying scikit-learn estimator (set by
            :meth:`train`).
        n_features_in_: Number of features seen during :meth:`train`.
        feature_names_in_: Feature names seen during :meth:`train` when ``X``
            carried them (e.g. a pandas ``DataFrame``).
    """

    #: Default hyperparameter search space; overridden per subclass.
    _default_param_grid: ParamGrid = {}

    # ----------------------------------------------------------------- hooks
    def _build_estimator(self) -> BaseEstimator:
        """Create an unfitted scikit-learn estimator from ``self``'s params.

        Returns:
            An unfitted scikit-learn regressor.

        Raises:
            NotImplementedError: Always, on the abstract base class.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is abstract; subclasses must implement `_build_estimator`."
        )

    def _is_concrete(self) -> bool:
        """Return ``True`` when a subclass has provided :meth:`_build_estimator`."""
        return type(self)._build_estimator is not RegressionModel._build_estimator

    def __sklearn_is_fitted__(self) -> bool:
        """Report fitted state to :func:`sklearn.utils.validation.check_is_fitted`."""
        return "model_" in self.__dict__

    def __sklearn_tags__(self):  # type: ignore[no-untyped-def]
        """Inherit input/target/regressor tags from the underlying estimator.

        This lets scikit-learn know, for example, that tree-based wrappers
        accept sparse input and missing values exactly as their estimator does.
        """
        tags = super().__sklearn_tags__()
        if not self._is_concrete():
            return tags
        try:
            estimator_tags = self._build_estimator().__sklearn_tags__()
        except Exception as exc:  # optional backend missing, or estimator lacks tags
            self.logger.debug("Tag delegation unavailable for %s: %s", type(self).__name__, exc)
            return tags
        tags.input_tags = estimator_tags.input_tags
        tags.target_tags = estimator_tags.target_tags
        tags.regressor_tags = estimator_tags.regressor_tags
        return tags

    def __getattr__(self, name: str) -> Any:
        """Forward fitted attributes (``*_``) to the underlying estimator.

        Anything not found on the wrapper whose name ends with an underscore
        (``n_iter_``, ``estimators_``, ``support_vectors_``, ...) is looked
        up on ``model_``. Only invoked when normal attribute lookup fails.

        Raises:
            AttributeError: If unfitted, or the estimator lacks the attribute.
        """
        if name.startswith("__") or not name.endswith("_") or name == "model_":
            raise AttributeError(f"{type(self).__name__} has no attribute '{name}'.")
        return self._fitted_attr(name)

    # ------------------------------------------------------------ properties
    @property
    def model(self) -> BaseEstimator:
        """The underlying scikit-learn estimator.

        Returns the fitted estimator after :meth:`train`; before that, a fresh
        unfitted estimator built from the current hyperparameters.

        Raises:
            NotImplementedError: If called on the abstract base class.
        """
        fitted = getattr(self, "model_", None)
        if fitted is not None:
            return fitted
        return self._build_estimator()

    @property
    def coef_(self) -> NDArray[np.float64]:
        """Coefficients of the underlying linear estimator.

        Raises:
            AttributeError: If the model is unfitted or the estimator has no
                ``coef_`` (e.g. tree ensembles, non-linear SVR kernels).
        """
        return self._fitted_attr("coef_")

    @property
    def intercept_(self) -> Union[float, NDArray[np.float64]]:
        """Intercept of the underlying linear estimator.

        Raises:
            AttributeError: If unfitted or the estimator has no ``intercept_``.
        """
        return self._fitted_attr("intercept_")

    @property
    def feature_importances_(self) -> NDArray[np.float64]:
        """Impurity-based feature importances of the underlying estimator.

        Raises:
            AttributeError: If unfitted or the estimator has no
                ``feature_importances_``.
        """
        return self._fitted_attr("feature_importances_")

    def _fitted_attr(self, name: str) -> Any:
        """Fetch ``name`` from the fitted estimator, mapping absence to ``AttributeError``."""
        fitted = self.__dict__.get("model_")
        if fitted is None:
            raise AttributeError(f"{type(self).__name__} has no attribute '{name}' before `train`/`fit`.")
        try:
            return getattr(fitted, name)
        except AttributeError as exc:
            raise AttributeError(
                f"Underlying estimator {type(fitted).__name__} exposes no '{name}'."
            ) from exc

    # ---------------------------------------------------------------- fitting
    def train(self, X: ArrayLike, y: ArrayLike, **fit_params: Any) -> RegressionModel:
        """Fit the underlying estimator.

        Args:
            X: Training features of shape ``(n_samples, n_features)``.
            y: Target values of shape ``(n_samples,)``.
            **fit_params: Extra keyword arguments forwarded to the underlying
                estimator's ``fit`` (e.g. ``sample_weight``).

        Returns:
            ``self`` (fitted).

        Raises:
            NotImplementedError: If called on the abstract base class.
        """
        if not self._is_concrete():
            raise NotImplementedError(f"{type(self).__name__}.train must be implemented by a subclass.")

        estimator = self._build_estimator()
        start = time.perf_counter()
        estimator.fit(X, y, **fit_params)
        self.training_time_ = time.perf_counter() - start
        self.model_ = estimator

        n_features_in = getattr(estimator, "n_features_in_", None)
        if n_features_in is not None:
            self.n_features_in_ = int(n_features_in)
        feature_names_in = getattr(estimator, "feature_names_in_", None)
        if feature_names_in is not None:
            self.feature_names_in_ = np.asarray(feature_names_in, dtype=object)

        self.logger.debug(
            "Fitted %s (n_features=%s) in %.3fs",
            type(estimator).__name__,
            getattr(self, "n_features_in_", "?"),
            self.training_time_,
        )
        return self

    def fit(self, X: ArrayLike, y: ArrayLike, **fit_params: Any) -> RegressionModel:
        """scikit-learn alias for :meth:`train`.

        Args:
            X: Training features.
            y: Target values.
            **fit_params: Forwarded to the underlying estimator's ``fit``.

        Returns:
            ``self`` (fitted).
        """
        return self.train(X, y, **fit_params)

    # ------------------------------------------------------------- inference
    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict targets for ``X``.

        Args:
            X: Features of shape ``(n_samples, n_features)``.

        Returns:
            Predictions of shape ``(n_samples,)`` as ``float64``.

        Raises:
            NotImplementedError: If called on the abstract base class.
            sklearn.exceptions.NotFittedError: If :meth:`train` was not called.
        """
        if not self._is_concrete():
            raise NotImplementedError(f"{type(self).__name__}.predict must be implemented by a subclass.")
        check_is_fitted(self)
        return np.asarray(self.model_.predict(X), dtype=np.float64)

    # ------------------------------------------------------------ evaluation
    def evaluate(self, X: ArrayLike, y: ArrayLike) -> Dict[str, float]:
        """Compute standard regression metrics on ``(X, y)``.

        Args:
            X: Features.
            y: True targets.

        Returns:
            Dictionary with ``mse``, ``rmse`` (``sqrt(mse)``), ``mae``, ``r2``,
            ``explained_variance`` and ``max_error``.
        """
        y_true = np.asarray(y, dtype=np.float64)
        y_pred = self.predict(X)
        mse = float(mean_squared_error(y_true, y_pred))
        metrics = {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "explained_variance": float(explained_variance_score(y_true, y_pred)),
        }
        if y_true.ndim == 1:
            metrics["max_error"] = float(max_error(y_true, y_pred))
        self.logger.debug("Evaluation for %s: %s", type(self).__name__, metrics)
        return metrics

    def cross_validate(
        self,
        X: ArrayLike,
        y: ArrayLike,
        cv: Any = 5,
        scoring: Union[str, Sequence[str], None] = None,
        n_jobs: Optional[int] = None,
        return_estimator: bool = False,
    ) -> Dict[str, NDArray[Any]]:
        """Run scikit-learn cross-validation on a clone of this wrapper.

        Args:
            X: Features.
            y: Targets.
            cv: Number of folds or a cross-validation splitter.
            scoring: Scoring name(s); ``None`` uses the estimator's ``score``
                (R²).
            n_jobs: Parallel jobs for scikit-learn.
            return_estimator: Include the fitted clones in the result.

        Returns:
            The :func:`sklearn.model_selection.cross_validate` result dict,
            always including ``test_score``/``train_score`` (or per-scorer
            variants) and ``fit_time``/``score_time``.
        """
        return _sk_cross_validate(
            clone(self),
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            return_train_score=True,
            return_estimator=return_estimator,
        )

    # ---------------------------------------------------------------- tuning
    def get_hyperparameter_grid(self) -> ParamGrid:
        """Return a sensible default hyperparameter grid for this wrapper.

        Returns:
            Mapping of wrapper parameter names to candidate values (a copy).
        """
        return {k: list(v) for k, v in self._default_param_grid.items()}

    def tune_hyperparameters(
        self,
        X: ArrayLike,
        y: ArrayLike,
        param_grid: Optional[ParamGrid] = None,
        cv: Any = 5,
        scoring: Optional[str] = None,
        n_jobs: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """Grid-search the wrapper's hyperparameters and adopt the best ones.

        After the search, ``self`` is updated in place: its parameters are
        set to the best combination and ``model_`` is the refitted estimator.

        Args:
            X: Features.
            y: Targets.
            param_grid: Parameter grid keyed by wrapper constructor arguments.
                ``None`` uses :meth:`get_hyperparameter_grid`.
            cv: Number of folds or a splitter.
            scoring: Scoring name; ``None`` uses R².
            n_jobs: Parallel jobs for the search.

        Returns:
            ``(best_params, best_score)``.

        Raises:
            ValueError: If no grid is supplied and the wrapper has no default.
        """
        grid = param_grid if param_grid is not None else self.get_hyperparameter_grid()
        if not grid:
            raise ValueError(f"No hyperparameter grid available for {type(self).__name__}.")

        unknown = set(grid) - set(self.get_params())
        if unknown:
            raise ValueError(f"Unknown parameter(s) for {type(self).__name__}: {sorted(unknown)}")

        search = GridSearchCV(clone(self), grid, cv=cv, scoring=scoring, n_jobs=n_jobs, refit=True)
        search.fit(X, y)

        best_params: Dict[str, Any] = dict(search.best_params_)
        best_score = float(search.best_score_)
        self.set_params(**best_params)
        best_wrapper: RegressionModel = search.best_estimator_
        self.model_ = best_wrapper.model_
        for attr in ("n_features_in_", "feature_names_in_", "training_time_"):
            if hasattr(best_wrapper, attr):
                setattr(self, attr, getattr(best_wrapper, attr))
        self.cv_results_ = search.cv_results_

        self.logger.info("Best params for %s: %s (score=%.4f)", type(self).__name__, best_params, best_score)
        return best_params, best_score

    # ---------------------------------------------------------- introspection
    def get_coefficients(self) -> NDArray[np.float64]:
        """Return the fitted linear coefficients.

        Returns:
            Array of shape ``(n_features,)`` (or ``(n_targets, n_features)``).

        Raises:
            AttributeError: If unfitted or the estimator has no ``coef_``.
        """
        return np.asarray(self.coef_, dtype=np.float64)

    def get_intercept(self) -> Union[float, NDArray[np.float64]]:
        """Return the fitted intercept.

        Returns:
            A ``float`` for single-output models, otherwise an array.

        Raises:
            AttributeError: If unfitted or the estimator has no ``intercept_``.
        """
        intercept = np.asarray(self.intercept_, dtype=np.float64)
        return float(intercept) if intercept.ndim == 0 or intercept.size == 1 else intercept

    def get_feature_importance(self) -> NDArray[np.float64]:
        """Return per-feature importances.

        Uses ``feature_importances_`` when the estimator provides it, and
        falls back to ``|coef_|`` for linear models.

        Returns:
            Non-negative array of shape ``(n_features,)``.

        Raises:
            AttributeError: If unfitted or neither attribute is available.
        """
        try:
            return np.asarray(self.feature_importances_, dtype=np.float64)
        except AttributeError:
            coef = np.asarray(self.coef_, dtype=np.float64)
            return np.abs(coef if coef.ndim == 1 else coef.mean(axis=0))

    # ----------------------------------------------------------- persistence
    def save_model(self, filepath: PathLike) -> Path:
        """Persist the wrapper (including its fitted estimator) with joblib.

        Args:
            filepath: Destination path; parent directories are created.

        Returns:
            The resolved path written.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        self.logger.info("Saved %s to %s", type(self).__name__, path)
        return path

    def load_model(self, filepath: PathLike) -> RegressionModel:
        """Load a wrapper saved with :meth:`save_model` into ``self``.

        Args:
            filepath: Path produced by :meth:`save_model`.

        Returns:
            ``self`` with parameters and fitted state replaced.

        Raises:
            TypeError: If the file does not contain a :class:`RegressionModel`
                of a compatible class.
        """
        loaded = joblib.load(Path(filepath))
        if not isinstance(loaded, RegressionModel):
            raise TypeError(f"{filepath} does not contain a RegressionModel (got {type(loaded).__name__}).")
        if not isinstance(loaded, type(self)):
            raise TypeError(
                f"Cannot load {type(loaded).__name__} into {type(self).__name__}; "
                f"instantiate {type(loaded).__name__} instead."
            )
        self.__dict__.clear()
        self.__dict__.update(loaded.__dict__)
        self.logger.info("Loaded %s from %s", type(self).__name__, filepath)
        return self


# --------------------------------------------------------------------------- #
# Linear models
# --------------------------------------------------------------------------- #
class LinearRegressionModel(RegressionModel):
    """Ordinary least squares wrapper around :class:`sklearn.linear_model.LinearRegression`.

    Args:
        fit_intercept: Whether to estimate an intercept term.
        copy_X: Copy ``X`` before fitting.
        n_jobs: Parallel jobs for multi-target problems.
        positive: Constrain coefficients to be non-negative.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        copy_X: bool = True,
        n_jobs: Optional[int] = None,
        positive: bool = False,
    ) -> None:
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive

    def _build_estimator(self) -> LinearRegression:
        return LinearRegression(
            fit_intercept=self.fit_intercept,
            copy_X=self.copy_X,
            n_jobs=self.n_jobs,
            positive=self.positive,
        )


class RidgeRegressionModel(RegressionModel):
    """L2-regularised linear regression (:class:`sklearn.linear_model.Ridge`).

    Args:
        alpha: Regularisation strength; larger values shrink coefficients more.
        fit_intercept: Whether to estimate an intercept term.
        solver: Ridge solver (``"auto"``, ``"svd"``, ``"cholesky"``, ...).
        max_iter: Maximum iterations for iterative solvers.
        tol: Solver tolerance.
        random_state: Seed used by stochastic solvers (``"sag"``/``"saga"``).
    """

    _default_param_grid: ParamGrid = ModelDefaults.REGRESSION_MODELS["ridge"]

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        solver: str = "auto",
        max_iter: Optional[int] = None,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ) -> None:
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _build_estimator(self) -> Ridge:
        return Ridge(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            solver=self.solver,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )


class LassoRegressionModel(RegressionModel):
    """L1-regularised linear regression (:class:`sklearn.linear_model.Lasso`).

    Lasso drives some coefficients exactly to zero, performing embedded
    feature selection.

    Args:
        alpha: Regularisation strength.
        fit_intercept: Whether to estimate an intercept term.
        max_iter: Maximum coordinate-descent iterations.
        tol: Optimisation tolerance.
        selection: ``"cyclic"`` or ``"random"`` coordinate updates.
        random_state: Seed for ``selection="random"``.
    """

    _default_param_grid: ParamGrid = ModelDefaults.REGRESSION_MODELS["lasso"]

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        selection: str = "cyclic",
        random_state: Optional[int] = None,
    ) -> None:
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.selection = selection
        self.random_state = random_state

    def _build_estimator(self) -> Lasso:
        return Lasso(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            selection=self.selection,
            random_state=self.random_state,
        )


class ElasticNetModel(RegressionModel):
    """Combined L1/L2 regularisation (:class:`sklearn.linear_model.ElasticNet`).

    Args:
        alpha: Overall regularisation strength.
        l1_ratio: Mix between L1 (``1.0``) and L2 (``0.0``) penalties.
        fit_intercept: Whether to estimate an intercept term.
        max_iter: Maximum coordinate-descent iterations.
        tol: Optimisation tolerance.
        selection: ``"cyclic"`` or ``"random"`` coordinate updates.
        random_state: Seed for ``selection="random"``.
    """

    _default_param_grid: ParamGrid = ModelDefaults.REGRESSION_MODELS["elastic_net"]

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        selection: str = "cyclic",
        random_state: Optional[int] = None,
    ) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.selection = selection
        self.random_state = random_state

    def _build_estimator(self) -> ElasticNet:
        return ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            selection=self.selection,
            random_state=self.random_state,
        )


# --------------------------------------------------------------------------- #
# Tree-based models
# --------------------------------------------------------------------------- #
class DecisionTreeRegressorModel(RegressionModel):
    """Single regression tree (:class:`sklearn.tree.DecisionTreeRegressor`).

    Args:
        criterion: Split quality measure (``"squared_error"``, ``"friedman_mse"``,
            ``"absolute_error"``, ``"poisson"``).
        max_depth: Maximum tree depth; ``None`` grows until pure leaves.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required at a leaf.
        max_features: Number/fraction of features considered per split.
        random_state: Seed controlling feature permutation at each split.
    """

    _default_param_grid: ParamGrid = {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    def __init__(
        self,
        criterion: str = "squared_error",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        max_features: Union[int, float, str, None] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def _build_estimator(self) -> DecisionTreeRegressor:
        return DecisionTreeRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
        )


class RandomForestRegressorModel(RegressionModel):
    """Bagged tree ensemble (:class:`sklearn.ensemble.RandomForestRegressor`).

    Args:
        n_estimators: Number of trees.
        criterion: Split quality measure.
        max_depth: Maximum depth per tree.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required at a leaf.
        max_features: Number/fraction of features considered per split.
        bootstrap: Draw bootstrap samples per tree.
        n_jobs: Parallel jobs for fitting/prediction.
        random_state: Seed for bootstrapping and feature sampling.
    """

    _default_param_grid: ParamGrid = ModelDefaults.REGRESSION_MODELS["random_forest"]

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "squared_error",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        max_features: Union[int, float, str, None] = 1.0,
        bootstrap: bool = True,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _build_estimator(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )


class ExtraTreesRegressorModel(RegressionModel):
    """Extremely randomised trees (:class:`sklearn.ensemble.ExtraTreesRegressor`).

    Args:
        n_estimators: Number of trees.
        criterion: Split quality measure.
        max_depth: Maximum depth per tree.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required at a leaf.
        max_features: Number/fraction of features considered per split.
        bootstrap: Draw bootstrap samples per tree (off by default).
        n_jobs: Parallel jobs for fitting/prediction.
        random_state: Seed for split thresholds and feature sampling.
    """

    _default_param_grid: ParamGrid = ModelDefaults.REGRESSION_MODELS["random_forest"]

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "squared_error",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        max_features: Union[int, float, str, None] = 1.0,
        bootstrap: bool = False,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _build_estimator(self) -> ExtraTreesRegressor:
        return ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )


class GradientBoostingRegressorModel(RegressionModel):
    """Boosted trees (:class:`sklearn.ensemble.GradientBoostingRegressor`).

    Args:
        loss: Loss to optimise (``"squared_error"``, ``"absolute_error"``,
            ``"huber"``, ``"quantile"``).
        learning_rate: Shrinkage applied to each tree's contribution.
        n_estimators: Number of boosting stages.
        subsample: Fraction of samples used per stage.
        max_depth: Maximum depth per tree.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required at a leaf.
        max_features: Number/fraction of features considered per split.
        random_state: Seed for subsampling and feature permutation.
    """

    _default_param_grid: ParamGrid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
    }

    def __init__(
        self,
        loss: str = "squared_error",
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        max_depth: Optional[int] = 3,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        max_features: Union[int, float, str, None] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def _build_estimator(self) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(
            loss=self.loss,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
        )

    def staged_predict(self, X: ArrayLike) -> Iterator[NDArray[np.float64]]:
        """Yield predictions after each boosting stage.

        Args:
            X: Features of shape ``(n_samples, n_features)``.

        Yields:
            Predictions of shape ``(n_samples,)`` for stages ``1..n_estimators``.

        Raises:
            sklearn.exceptions.NotFittedError: If :meth:`train` was not called.
        """
        check_is_fitted(self)
        for stage in self.model_.staged_predict(X):
            yield np.asarray(stage, dtype=np.float64)


class AdaBoostRegressorModel(RegressionModel):
    """AdaBoost.R2 ensemble (:class:`sklearn.ensemble.AdaBoostRegressor`).

    Args:
        estimator: Base learner; ``None`` uses a depth-3 decision tree.
        n_estimators: Maximum number of boosting rounds.
        learning_rate: Weight applied to each regressor per round.
        loss: Loss used to update sample weights (``"linear"``, ``"square"``,
            ``"exponential"``).
        random_state: Seed for the bootstrap sampling of each round.
    """

    _default_param_grid: ParamGrid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1.0],
        "loss": ["linear", "square", "exponential"],
    }

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        loss: str = "linear",
        random_state: Optional[int] = None,
    ) -> None:
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state

    def _build_estimator(self) -> AdaBoostRegressor:
        return AdaBoostRegressor(
            estimator=clone(self.estimator) if self.estimator is not None else None,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            loss=self.loss,
            random_state=self.random_state,
        )


# --------------------------------------------------------------------------- #
# Kernel and neural models
# --------------------------------------------------------------------------- #
class SVMRegressorModel(RegressionModel):
    """Epsilon-SVR (:class:`sklearn.svm.SVR`).

    Args:
        kernel: Kernel type (``"linear"``, ``"poly"``, ``"rbf"``, ``"sigmoid"``).
        C: Regularisation parameter (inverse strength).
        epsilon: Width of the epsilon-insensitive tube.
        gamma: Kernel coefficient for ``"rbf"``/``"poly"``/``"sigmoid"``.
        degree: Polynomial degree (``"poly"`` only).
        coef0: Independent kernel term (``"poly"``/``"sigmoid"``).
        tol: Stopping tolerance.
        max_iter: Hard iteration limit; ``-1`` for none.
        random_state: Accepted for API uniformity; SVR is deterministic and
            does not use it.
    """

    _default_param_grid: ParamGrid = ModelDefaults.REGRESSION_MODELS["svr"]

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.1,
        gamma: Union[str, float] = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        tol: float = 1e-3,
        max_iter: int = -1,
        random_state: Optional[int] = None,
    ) -> None:
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

    def _build_estimator(self) -> SVR:
        return SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            tol=self.tol,
            max_iter=self.max_iter,
        )


class NeuralNetworkRegressorModel(RegressionModel):
    """Multi-layer perceptron (:class:`sklearn.neural_network.MLPRegressor`).

    Args:
        hidden_layer_sizes: Units per hidden layer.
        activation: Hidden-layer activation.
        solver: Weight optimiser (``"adam"``, ``"sgd"``, ``"lbfgs"``).
        alpha: L2 penalty strength.
        learning_rate: Learning-rate schedule (``"sgd"`` only).
        learning_rate_init: Initial learning rate.
        max_iter: Maximum epochs / iterations.
        early_stopping: Hold out validation data to stop early.
        random_state: Seed for weight initialisation and shuffling.
    """

    _default_param_grid: ParamGrid = {
        "hidden_layer_sizes": [(50,), (100,), (100, 50)],
        "alpha": [1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-3, 1e-2],
    }

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (100,),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 1e-4,
        learning_rate: str = "constant",
        learning_rate_init: float = 1e-3,
        max_iter: int = 200,
        early_stopping: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.random_state = random_state

    def _build_estimator(self) -> MLPRegressor:
        return MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            random_state=self.random_state,
        )


# --------------------------------------------------------------------------- #
# Optional gradient-boosting libraries
# --------------------------------------------------------------------------- #
class XGBoostRegressorModel(RegressionModel):
    """XGBoost regressor wrapper (requires the optional ``xgboost`` package).

    Args:
        n_estimators: Number of boosting rounds.
        learning_rate: Step-size shrinkage.
        max_depth: Maximum tree depth.
        subsample: Row subsampling ratio per round.
        colsample_bytree: Column subsampling ratio per tree.
        reg_alpha: L1 regularisation on leaf weights.
        reg_lambda: L2 regularisation on leaf weights.
        n_jobs: Parallel threads.
        random_state: Seed for subsampling.

    Raises:
        ImportError: On :meth:`train` when ``xgboost`` is not installed.
    """

    _default_param_grid: ParamGrid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.03, 0.1, 0.3],
        "max_depth": [3, 6, 9],
    }

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _build_estimator(self) -> BaseEstimator:
        if not HAS_XGBOOST:
            raise ImportError("XGBoostRegressorModel requires `xgboost`; install with `pip install xgboost`.")
        return XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbosity=0,
        )


class LightGBMRegressorModel(RegressionModel):
    """LightGBM regressor wrapper (requires the optional ``lightgbm`` package).

    Args:
        n_estimators: Number of boosting rounds.
        learning_rate: Step-size shrinkage.
        num_leaves: Maximum leaves per tree.
        max_depth: Maximum tree depth; ``-1`` for unlimited.
        subsample: Row subsampling ratio per round.
        colsample_bytree: Column subsampling ratio per tree.
        reg_alpha: L1 regularisation.
        reg_lambda: L2 regularisation.
        n_jobs: Parallel threads.
        random_state: Seed for subsampling.

    Raises:
        ImportError: On :meth:`train` when ``lightgbm`` is not installed.
    """

    _default_param_grid: ParamGrid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.03, 0.1, 0.3],
        "num_leaves": [15, 31, 63],
    }

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        max_depth: int = -1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _build_estimator(self) -> BaseEstimator:
        if not HAS_LIGHTGBM:
            raise ImportError(
                "LightGBMRegressorModel requires `lightgbm`; install with `pip install lightgbm`."
            )
        return LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=-1,
        )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
REGRESSION_MODEL_REGISTRY: Dict[str, Type[RegressionModel]] = {
    "linear_regression": LinearRegressionModel,
    "ridge": RidgeRegressionModel,
    "lasso": LassoRegressionModel,
    "elastic_net": ElasticNetModel,
    "decision_tree": DecisionTreeRegressorModel,
    "random_forest": RandomForestRegressorModel,
    "extra_trees": ExtraTreesRegressorModel,
    "gradient_boosting": GradientBoostingRegressorModel,
    "adaboost": AdaBoostRegressorModel,
    "svr": SVMRegressorModel,
    "neural_network": NeuralNetworkRegressorModel,
    "xgboost": XGBoostRegressorModel,
    "lightgbm": LightGBMRegressorModel,
}
"""Canonical algorithm name -> wrapper class."""

_MODEL_ALIASES: Dict[str, str] = {
    "linear": "linear_regression",
    "ols": "linear_regression",
    "elasticnet": "elastic_net",
    "rf": "random_forest",
    "et": "extra_trees",
    "gb": "gradient_boosting",
    "gbr": "gradient_boosting",
    "ada": "adaboost",
    "svm": "svr",
    "svm_regression": "svr",
    "mlp": "neural_network",
    "nn": "neural_network",
    "xgb": "xgboost",
    "lgbm": "lightgbm",
}


class RegressionModels(LoggerMixin):
    """Factory for building and training regression wrappers by name.

    Args:
        random_state: Seed injected into every wrapper that accepts one unless
            the caller passes ``random_state`` explicitly. Defaults to
            ``settings.RANDOM_SEED``.
    """

    def __init__(self, random_state: Optional[int] = None) -> None:
        self.random_state = settings.RANDOM_SEED if random_state is None else random_state

    # ----------------------------------------------------------- utilities
    @staticmethod
    def available_models(include_optional: bool = True) -> List[str]:
        """List canonical model names.

        Args:
            include_optional: Include ``xgboost``/``lightgbm`` only when their
                packages are importable.

        Returns:
            Sorted list of names accepted by :meth:`get_model`.
        """
        names = []
        for name in REGRESSION_MODEL_REGISTRY:
            if name == "xgboost" and not (include_optional and HAS_XGBOOST):
                continue
            if name == "lightgbm" and not (include_optional and HAS_LIGHTGBM):
                continue
            names.append(name)
        return sorted(names)

    @staticmethod
    def _resolve(name: str) -> Type[RegressionModel]:
        key = name.strip().lower().replace("-", "_").replace(" ", "_")
        key = _MODEL_ALIASES.get(key, key)
        try:
            return REGRESSION_MODEL_REGISTRY[key]
        except KeyError as exc:
            raise ValueError(
                f"Unknown regression algorithm '{name}'. Available: {sorted(REGRESSION_MODEL_REGISTRY)}"
            ) from exc

    def _instantiate(self, cls: Type[RegressionModel], **kwargs: Any) -> RegressionModel:
        accepted = set(cls().get_params())
        dropped = sorted(set(kwargs) - accepted)
        if dropped:
            self.logger.warning("%s ignores unsupported argument(s): %s", cls.__name__, dropped)
        params = {k: v for k, v in kwargs.items() if k in accepted}
        if "random_state" in accepted and "random_state" not in params:
            params["random_state"] = self.random_state
        return cls(**params)

    def get_model(self, name: str, **kwargs: Any) -> RegressionModel:
        """Build a wrapper by algorithm name.

        Args:
            name: Canonical name (see :meth:`available_models`) or alias such
                as ``"rf"``, ``"svm"``, ``"nn"``.
            **kwargs: Constructor arguments; unsupported ones are dropped with
                a warning.

        Returns:
            An unfitted wrapper.

        Raises:
            ValueError: If ``name`` is unknown.
            ImportError: If an optional backend is requested but not installed.
        """
        cls = self._resolve(name)
        if cls is XGBoostRegressorModel and not HAS_XGBOOST:
            raise ImportError("xgboost is not installed.")
        if cls is LightGBMRegressorModel and not HAS_LIGHTGBM:
            raise ImportError("lightgbm is not installed.")
        return self._instantiate(cls, **kwargs)

    def train_model(
        self,
        X: ArrayLike,
        y: ArrayLike,
        algorithm: str = "random_forest",
        **kwargs: Any,
    ) -> Tuple[RegressionModel, float]:
        """Build and fit a wrapper in one call.

        Args:
            X: Training features.
            y: Targets.
            algorithm: Algorithm name or alias.
            **kwargs: Constructor arguments for the wrapper.

        Returns:
            ``(fitted_model, training_time_seconds)``.
        """
        model = self.get_model(algorithm, **kwargs)
        model.train(X, y)
        return model, float(model.training_time_)

    # --------------------------------------------------------------- getters
    def get_linear_regression(self, **kwargs: Any) -> LinearRegressionModel:
        """Return a :class:`LinearRegressionModel`."""
        return self._instantiate(LinearRegressionModel, **kwargs)  # type: ignore[return-value]

    def get_ridge_regression(self, **kwargs: Any) -> RidgeRegressionModel:
        """Return a :class:`RidgeRegressionModel`."""
        return self._instantiate(RidgeRegressionModel, **kwargs)  # type: ignore[return-value]

    def get_lasso_regression(self, **kwargs: Any) -> LassoRegressionModel:
        """Return a :class:`LassoRegressionModel`."""
        return self._instantiate(LassoRegressionModel, **kwargs)  # type: ignore[return-value]

    def get_elastic_net(self, **kwargs: Any) -> ElasticNetModel:
        """Return an :class:`ElasticNetModel`."""
        return self._instantiate(ElasticNetModel, **kwargs)  # type: ignore[return-value]

    def get_decision_tree_regression(self, **kwargs: Any) -> DecisionTreeRegressorModel:
        """Return a :class:`DecisionTreeRegressorModel`."""
        return self._instantiate(DecisionTreeRegressorModel, **kwargs)  # type: ignore[return-value]

    def get_random_forest_regression(self, **kwargs: Any) -> RandomForestRegressorModel:
        """Return a :class:`RandomForestRegressorModel`."""
        return self._instantiate(RandomForestRegressorModel, **kwargs)  # type: ignore[return-value]

    def get_extra_trees_regression(self, **kwargs: Any) -> ExtraTreesRegressorModel:
        """Return an :class:`ExtraTreesRegressorModel`."""
        return self._instantiate(ExtraTreesRegressorModel, **kwargs)  # type: ignore[return-value]

    def get_gradient_boosting_regression(self, **kwargs: Any) -> GradientBoostingRegressorModel:
        """Return a :class:`GradientBoostingRegressorModel`."""
        return self._instantiate(GradientBoostingRegressorModel, **kwargs)  # type: ignore[return-value]

    def get_adaboost_regression(self, **kwargs: Any) -> AdaBoostRegressorModel:
        """Return an :class:`AdaBoostRegressorModel`."""
        return self._instantiate(AdaBoostRegressorModel, **kwargs)  # type: ignore[return-value]

    def get_svr(self, **kwargs: Any) -> SVMRegressorModel:
        """Return an :class:`SVMRegressorModel`."""
        return self._instantiate(SVMRegressorModel, **kwargs)  # type: ignore[return-value]

    get_svm_regression = get_svr

    def get_neural_network_regression(self, **kwargs: Any) -> NeuralNetworkRegressorModel:
        """Return a :class:`NeuralNetworkRegressorModel`."""
        return self._instantiate(NeuralNetworkRegressorModel, **kwargs)  # type: ignore[return-value]

    def get_xgboost_regression(self, **kwargs: Any) -> XGBoostRegressorModel:
        """Return an :class:`XGBoostRegressorModel`.

        Raises:
            ImportError: If ``xgboost`` is not installed.
        """
        return self.get_model("xgboost", **kwargs)  # type: ignore[return-value]

    def get_lightgbm_regression(self, **kwargs: Any) -> LightGBMRegressorModel:
        """Return a :class:`LightGBMRegressorModel`.

        Raises:
            ImportError: If ``lightgbm`` is not installed.
        """
        return self.get_model("lightgbm", **kwargs)  # type: ignore[return-value]


__all__ = [
    "HAS_LIGHTGBM",
    "HAS_XGBOOST",
    "REGRESSION_MODEL_REGISTRY",
    "AdaBoostRegressorModel",
    "DecisionTreeRegressorModel",
    "ElasticNetModel",
    "ExtraTreesRegressorModel",
    "GradientBoostingRegressorModel",
    "LassoRegressionModel",
    "LightGBMRegressorModel",
    "LinearRegressionModel",
    "NeuralNetworkRegressorModel",
    "RandomForestRegressorModel",
    "RegressionModel",
    "RegressionModels",
    "RidgeRegressionModel",
    "SVMRegressorModel",
    "XGBoostRegressorModel",
]
