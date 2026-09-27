"""Cross-validation, learning-curve and validation-curve utilities.

Wraps :mod:`sklearn.model_selection` with deterministic fold construction
(shuffled splitters seeded by ``random_state``), consistent result
dictionaries and small diagnostic helpers (overfitting / convergence
detection, optimal hyper-parameter selection).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    StratifiedKFold,
    cross_validate,
    learning_curve,
    validation_curve,
)

from sklearn_mastery.config.logging_config import LoggerMixin
from sklearn_mastery.config.settings import settings
from sklearn_mastery.evaluation.utils import ensure_numpy_array, to_scalar

CVLike = Union[int, BaseCrossValidator, Any]


def default_scoring(model: BaseEstimator) -> Optional[str]:
    """Return a sensible default scorer name for ``model``.

    Args:
        model: Any estimator.

    Returns:
        ``"accuracy"`` for classifiers, ``"r2"`` for regressors, otherwise ``None``
        (which makes scikit-learn fall back to ``model.score``).
    """
    if is_classifier(model):
        return "accuracy"
    if is_regressor(model):
        return "r2"
    return None


def make_cv(
    cv: CVLike,
    model: Optional[BaseEstimator],
    y: Optional[np.ndarray],
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: Optional[bool] = None,
) -> CVLike:
    """Build a deterministic cross-validation splitter.

    Args:
        cv: Number of folds, or an existing splitter/iterable (returned unchanged).
        model: Estimator; used to decide on stratification when ``stratify`` is ``None``.
        y: Targets; used to check stratification feasibility.
        random_state: Seed for shuffling.
        shuffle: Whether to shuffle samples before splitting.
        stratify: ``True`` forces :class:`StratifiedKFold`, ``False`` forces
            :class:`KFold`; ``None`` stratifies for classifiers whenever every
            class has at least ``cv`` members.

    Returns:
        A splitter object.
    """
    if not isinstance(cv, (int, np.integer)):
        return cv
    n_splits = int(cv)
    if n_splits < 2:
        raise ValueError("cv must be at least 2")
    if stratify is None:
        stratify = False
        if model is not None and is_classifier(model) and y is not None:
            _, counts = np.unique(ensure_numpy_array(y), return_counts=True)
            stratify = bool(counts.min() >= n_splits)
    kwargs: Dict[str, Any] = {"n_splits": n_splits, "shuffle": shuffle}
    if shuffle:
        kwargs["random_state"] = random_state
    return StratifiedKFold(**kwargs) if stratify else KFold(**kwargs)


class CrossValidator(LoggerMixin):
    """Deterministic *k*-fold cross-validation with train/test scores and timings.

    Args:
        cv: Number of folds or a scikit-learn splitter.
        scoring: Scorer name, list of names, dict or callable (scikit-learn
            conventions). ``None`` uses the estimator's ``score`` method.
        random_state: Seed used when shuffling folds.
        stratify: ``True``/``False`` to force (non-)stratified folds; ``None``
            stratifies classifiers when feasible.
        shuffle: Shuffle samples before splitting.
        n_jobs: Parallel jobs passed to scikit-learn (``None`` = sequential).
        return_train_score: Include training-fold scores.
        error_score: Value assigned to folds whose fit fails (``np.nan`` keeps
            the run going and logs a warning; ``"raise"`` propagates the error).
    """

    def __init__(
        self,
        cv: CVLike = 5,
        scoring: Union[str, Sequence[str], Dict[str, Any], None] = None,
        random_state: Optional[int] = None,
        stratify: Optional[bool] = None,
        shuffle: bool = True,
        n_jobs: Optional[int] = None,
        return_train_score: bool = True,
        error_score: Union[float, str] = np.nan,
    ):
        self.cv = cv
        self.scoring = scoring
        self.random_state = settings.RANDOM_SEED if random_state is None else random_state
        self.stratify = stratify
        self.shuffle = shuffle
        self.n_jobs = n_jobs
        self.return_train_score = return_train_score
        self.error_score = error_score

    def build_cv(self, model: Optional[BaseEstimator] = None, y: Optional[np.ndarray] = None) -> CVLike:
        """Return the concrete splitter used for ``model``/``y`` (see :func:`make_cv`)."""
        return make_cv(self.cv, model, y, self.random_state, self.shuffle, self.stratify)

    def cross_validate(
        self,
        model: BaseEstimator,
        X: Any,
        y: Any,
        groups: Optional[np.ndarray] = None,
        fit_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """Run cross-validation and return per-fold arrays.

        Args:
            model: Unfitted estimator (cloned for each fold).
            X: Feature matrix.
            y: Targets.
            groups: Optional group labels for group-aware splitters.
            fit_params: Extra keyword arguments forwarded to ``fit``.

        Returns:
            Dict with ``fit_time``, ``score_time`` and, for a single scorer,
            ``test_score``/``train_score``; for multiple scorers
            ``test_<name>``/``train_<name>`` for each name.
        """
        y_arr = ensure_numpy_array(y) if y is not None else None
        splitter = self.build_cv(model, y_arr)
        scoring = self.scoring
        if isinstance(scoring, (list, tuple)):
            scoring = list(scoring)
        results = cross_validate(
            model,
            X,
            y,
            groups=groups,
            scoring=scoring,
            cv=splitter,
            n_jobs=self.n_jobs,
            return_train_score=self.return_train_score,
            error_score=self.error_score,
            params=fit_params,
        )
        results = {k: np.asarray(v) for k, v in results.items()}
        n_failed = (
            int(np.isnan(results.get("test_score", np.zeros(0))).sum()) if "test_score" in results else 0
        )
        if n_failed:
            self.logger.warning(
                "%d of %d folds failed to fit; scores recorded as NaN", n_failed, len(results["test_score"])
            )
        self.logger.info("Cross-validated %s on %d folds", type(model).__name__, len(results["fit_time"]))
        return results

    @staticmethod
    def get_cv_statistics(scores: Dict[str, np.ndarray], key: str = "test_score") -> Dict[str, float]:
        """Summarise one score array from :meth:`cross_validate`.

        Args:
            scores: Result of :meth:`cross_validate`.
            key: Which entry to summarise (e.g. ``"test_score"`` or ``"test_f1"``).

        Returns:
            Dict with ``mean``, ``std`` (population, ``ddof=0``), ``sem``
            (``ddof=1``), ``median``, ``min``, ``max``, ``n_folds`` and ``n_failed``
            (NaN folds are ignored in the statistics).

        Raises:
            KeyError: If ``key`` is not present in ``scores``.
        """
        if key not in scores:
            raise KeyError(f"{key!r} not in cross-validation results; available: {sorted(scores)}")
        arr = np.asarray(scores[key], dtype=float)
        valid = arr[np.isfinite(arr)]
        n_failed = int(len(arr) - len(valid))
        if len(valid) == 0:
            nan = float("nan")
            return {
                "mean": nan,
                "std": nan,
                "sem": nan,
                "median": nan,
                "min": nan,
                "max": nan,
                "n_folds": len(arr),
                "n_failed": n_failed,
            }
        return {
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid)),
            "sem": float(np.std(valid, ddof=1) / np.sqrt(len(valid))) if len(valid) > 1 else 0.0,
            "median": float(np.median(valid)),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "n_folds": len(arr),
            "n_failed": n_failed,
        }

    def summarize(self, scores: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Statistics for every ``test_*``/``train_*`` entry of a result dict."""
        return {k: self.get_cv_statistics(scores, k) for k in scores if k.startswith(("test_", "train_"))}


class LearningCurveAnalyzer(LoggerMixin):
    """Learning curves (score vs. training-set size) and their diagnosis.

    Args:
        scoring: Scorer name; ``None`` picks accuracy/R² by estimator type.
        n_jobs: Parallel jobs for scikit-learn.
        random_state: Seed for fold construction and subsampling.
        shuffle: Shuffle before selecting training subsets.
    """

    def __init__(
        self,
        scoring: Optional[str] = None,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
        shuffle: bool = True,
    ):
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = settings.RANDOM_SEED if random_state is None else random_state
        self.shuffle = shuffle

    def generate_learning_curve(
        self,
        model: BaseEstimator,
        X: Any,
        y: Any,
        train_sizes: Optional[Sequence[float]] = None,
        cv: CVLike = 5,
        scoring: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute training and validation scores for increasing training sizes.

        Args:
            model: Unfitted estimator.
            X: Feature matrix.
            y: Targets.
            train_sizes: Fractions (``0 < f <= 1``) or absolute sizes; default
                ``np.linspace(0.1, 1.0, 5)``.
            cv: Number of folds or splitter.
            scoring: Scorer name (overrides the instance default).

        Returns:
            Dict with ``train_sizes`` (absolute, shape ``(n_sizes,)``),
            ``train_scores`` and ``validation_scores`` (shape ``(n_sizes, n_folds)``),
            their ``*_mean``/``*_std`` summaries, ``scoring`` and ``cv``.
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 5)
        scoring = scoring or self.scoring or default_scoring(model)
        y_arr = ensure_numpy_array(y)
        splitter = make_cv(cv, model, y_arr, self.random_state, self.shuffle)
        sizes_abs, train_scores, val_scores = learning_curve(
            model,
            X,
            y,
            train_sizes=np.asarray(train_sizes),
            cv=splitter,
            scoring=scoring,
            n_jobs=self.n_jobs,
            shuffle=self.shuffle,
            random_state=self.random_state if self.shuffle else None,
        )
        self.logger.info("Learning curve for %s over %d training sizes", type(model).__name__, len(sizes_abs))
        return {
            "train_sizes": np.asarray(sizes_abs),
            "train_scores": np.asarray(train_scores),
            "validation_scores": np.asarray(val_scores),
            "train_scores_mean": np.mean(train_scores, axis=1),
            "train_scores_std": np.std(train_scores, axis=1),
            "validation_scores_mean": np.mean(val_scores, axis=1),
            "validation_scores_std": np.std(val_scores, axis=1),
            "scoring": scoring,
            "cv": cv,
        }

    def analyze_learning_curve(
        self,
        results: Dict[str, Any],
        gap_threshold: float = 0.05,
        convergence_tol: float = 0.01,
    ) -> Dict[str, Any]:
        """Diagnose overfitting and convergence from learning-curve results.

        * ``overfitting_detected``: train-validation gap at the largest size
          exceeds ``gap_threshold``.
        * ``convergence_detected``: validation score changed by less than
          ``convergence_tol`` between the two largest training sizes.

        Args:
            results: Output of :meth:`generate_learning_curve`.
            gap_threshold: Maximum acceptable train/validation gap.
            convergence_tol: Maximum validation-score change deemed "flat".

        Returns:
            Dict with the two boolean flags, ``final_gap``, ``max_gap``,
            ``final_train_score``, ``final_validation_score``,
            ``validation_improvement`` (last minus first) and ``recommendations``.
        """
        train_mean = np.asarray(results["train_scores_mean"], dtype=float)
        val_mean = np.asarray(results["validation_scores_mean"], dtype=float)
        gaps = train_mean - val_mean
        final_gap = float(gaps[-1])
        overfitting = bool(final_gap > gap_threshold)
        converged = bool(len(val_mean) >= 2 and abs(val_mean[-1] - val_mean[-2]) < convergence_tol)
        improvement = float(val_mean[-1] - val_mean[0])

        recommendations: List[str] = []
        if overfitting:
            recommendations.append(
                f"Train/validation gap of {final_gap:.3f} exceeds {gap_threshold}: add regularisation, "
                "reduce model capacity or collect more data."
            )
        if not converged:
            recommendations.append(
                "Validation score is still improving with more data: additional training samples "
                "are likely to help."
            )
        elif not overfitting:
            recommendations.append(
                "Validation score has plateaued with a small gap: performance is capacity-limited; "
                "consider richer features or a more expressive model."
            )
        if not recommendations:
            recommendations.append("Learning curve looks healthy.")
        return {
            "overfitting_detected": overfitting,
            "convergence_detected": converged,
            "final_gap": final_gap,
            "max_gap": float(np.max(gaps)),
            "final_train_score": float(train_mean[-1]),
            "final_validation_score": float(val_mean[-1]),
            "validation_improvement": improvement,
            "gap_threshold": gap_threshold,
            "convergence_tol": convergence_tol,
            "recommendations": recommendations,
        }

    @staticmethod
    def prepare_learning_curve_plot(results: Dict[str, Any], title: Optional[str] = None) -> Dict[str, Any]:
        """Plot-ready summary (means, stds, labels) of learning-curve results."""
        scoring = results.get("scoring") or "score"
        return {
            "train_sizes": np.asarray(results["train_sizes"]),
            "train_scores_mean": np.asarray(results["train_scores_mean"]),
            "train_scores_std": np.asarray(results["train_scores_std"]),
            "validation_scores_mean": np.asarray(results["validation_scores_mean"]),
            "validation_scores_std": np.asarray(results["validation_scores_std"]),
            "title": title or "Learning Curve",
            "xlabel": "Training examples",
            "ylabel": str(scoring).replace("_", " ").title(),
        }


class ValidationCurveAnalyzer(LoggerMixin):
    """Validation curves (score vs. one hyper-parameter) and optimum selection.

    Args:
        scoring: Scorer name; ``None`` picks accuracy/R² by estimator type.
        n_jobs: Parallel jobs for scikit-learn.
        random_state: Seed for fold construction.
    """

    def __init__(
        self,
        scoring: Optional[str] = None,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = settings.RANDOM_SEED if random_state is None else random_state

    def generate_validation_curve(
        self,
        model: BaseEstimator,
        X: Any,
        y: Any,
        param_name: str,
        param_range: Sequence[Any],
        cv: CVLike = 5,
        scoring: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute train/validation scores for each value of ``param_name``.

        Args:
            model: Unfitted estimator exposing ``param_name`` via ``set_params``.
            X: Feature matrix.
            y: Targets.
            param_name: Hyper-parameter to vary.
            param_range: Candidate values.
            cv: Number of folds or splitter.
            scoring: Scorer name (overrides the instance default).

        Returns:
            Dict with ``param_name``, ``param_range`` (list, original types),
            ``train_scores``/``validation_scores`` (``(n_params, n_folds)``), their
            ``*_mean``/``*_std`` summaries and ``scoring``.
        """
        param_range = list(param_range)
        if not param_range:
            raise ValueError("param_range must contain at least one value")
        scoring = scoring or self.scoring or default_scoring(model)
        y_arr = ensure_numpy_array(y)
        splitter = make_cv(cv, model, y_arr, self.random_state)
        train_scores, val_scores = validation_curve(
            model,
            X,
            y,
            param_name=param_name,
            param_range=param_range,
            cv=splitter,
            scoring=scoring,
            n_jobs=self.n_jobs,
        )
        self.logger.info(
            "Validation curve for %s over %d values of %s", type(model).__name__, len(param_range), param_name
        )
        return {
            "param_name": param_name,
            "param_range": param_range,
            "train_scores": np.asarray(train_scores),
            "validation_scores": np.asarray(val_scores),
            "train_scores_mean": np.mean(train_scores, axis=1),
            "train_scores_std": np.std(train_scores, axis=1),
            "validation_scores_mean": np.mean(val_scores, axis=1),
            "validation_scores_std": np.std(val_scores, axis=1),
            "scoring": scoring,
        }

    @staticmethod
    def find_optimal_parameter(results: Dict[str, Any]) -> Any:
        """Return the parameter value with the highest mean validation score.

        Ties resolve to the first (typically simplest) value in ``param_range``.

        Args:
            results: Output of :meth:`generate_validation_curve`.

        Returns:
            The optimal value, as a Python scalar when it came from numpy.
        """
        val_mean = np.asarray(results["validation_scores_mean"], dtype=float)
        best_idx = int(np.nanargmax(val_mean))
        return to_scalar(results["param_range"][best_idx])

    @staticmethod
    def prepare_validation_curve_plot(results: Dict[str, Any], title: Optional[str] = None) -> Dict[str, Any]:
        """Plot-ready summary of validation-curve results."""
        return {
            "param_name": results["param_name"],
            "param_range": list(results["param_range"]),
            "train_scores_mean": np.asarray(results["train_scores_mean"]),
            "train_scores_std": np.asarray(results["train_scores_std"]),
            "validation_scores_mean": np.asarray(results["validation_scores_mean"]),
            "validation_scores_std": np.asarray(results["validation_scores_std"]),
            "title": title or f"Validation Curve ({results['param_name']})",
            "xlabel": results["param_name"],
            "ylabel": str(results.get("scoring") or "score").replace("_", " ").title(),
        }

    # ------------------------------------------------------------------ #
    # Legacy API (kept for backward compatibility)
    # ------------------------------------------------------------------ #
    def compute_validation_curve(
        self,
        model: BaseEstimator,
        X: Any,
        y: Any,
        param_name: str,
        param_range: Sequence[Any],
        cv_folds: Optional[int] = None,
        scoring: Optional[str] = None,
        n_jobs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Legacy wrapper returning list-based summaries and the best parameter."""
        if n_jobs is not None:
            self.n_jobs = n_jobs
        res = self.generate_validation_curve(
            model, X, y, param_name, param_range, cv=cv_folds or settings.DEFAULT_CV_FOLDS, scoring=scoring
        )
        best_idx = int(np.nanargmax(res["validation_scores_mean"]))
        return {
            "param_name": param_name,
            "param_range": res["param_range"],
            "train_scores_mean": res["train_scores_mean"].tolist(),
            "train_scores_std": res["train_scores_std"].tolist(),
            "val_scores_mean": res["validation_scores_mean"].tolist(),
            "val_scores_std": res["validation_scores_std"].tolist(),
            "scoring_metric": res["scoring"],
            "best_param_idx": best_idx,
            "best_param_value": to_scalar(res["param_range"][best_idx]),
            "best_score": float(res["validation_scores_mean"][best_idx]),
            "best_score_std": float(res["validation_scores_std"][best_idx]),
        }

    def compute_learning_curve(
        self,
        model: BaseEstimator,
        X: Any,
        y: Any,
        train_sizes: Optional[Sequence[float]] = None,
        cv_folds: Optional[int] = None,
        scoring: Optional[str] = None,
        n_jobs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Legacy wrapper around :class:`LearningCurveAnalyzer` with list outputs."""
        analyzer = LearningCurveAnalyzer(scoring=scoring, n_jobs=n_jobs, random_state=self.random_state)
        res = analyzer.generate_learning_curve(
            model,
            X,
            y,
            train_sizes=train_sizes if train_sizes is not None else np.linspace(0.1, 1.0, 10),
            cv=cv_folds or settings.DEFAULT_CV_FOLDS,
        )
        return {
            "train_sizes": res["train_sizes"].tolist(),
            "train_scores_mean": res["train_scores_mean"].tolist(),
            "train_scores_std": res["train_scores_std"].tolist(),
            "val_scores_mean": res["validation_scores_mean"].tolist(),
            "val_scores_std": res["validation_scores_std"].tolist(),
            "scoring_metric": res["scoring"],
        }

    @staticmethod
    def analyze_overfitting(
        learning_curve_results: Dict[str, Any], threshold: float = 0.05
    ) -> Dict[str, Any]:
        """Legacy overfitting summary from :meth:`compute_learning_curve` output."""
        train_scores = np.asarray(learning_curve_results["train_scores_mean"], dtype=float)
        val_scores = np.asarray(
            learning_curve_results.get(
                "val_scores_mean", learning_curve_results.get("validation_scores_mean")
            ),
            dtype=float,
        )
        gap = train_scores - val_scores
        over_idx = np.flatnonzero(gap > threshold)
        return {
            "is_overfitting": bool(gap[-1] > threshold),
            "final_gap": float(gap[-1]),
            "max_gap": float(np.max(gap)),
            "overfitting_threshold": threshold,
            "overfitting_start_idx": int(over_idx[0]) if len(over_idx) else None,
            "gap_trend": "increasing" if gap[-1] > gap[0] else "decreasing",
        }


__all__ = [
    "CrossValidator",
    "LearningCurveAnalyzer",
    "ValidationCurveAnalyzer",
    "default_scoring",
    "make_cv",
]
