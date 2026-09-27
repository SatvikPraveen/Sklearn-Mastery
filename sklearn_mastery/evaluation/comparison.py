"""Side-by-side comparison and ranking of several models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

from sklearn_mastery.config.logging_config import LoggerMixin
from sklearn_mastery.config.settings import settings
from sklearn_mastery.evaluation.cross_validation import CVLike, default_scoring, make_cv
from sklearn_mastery.evaluation.metrics import CustomMetric, ModelEvaluator
from sklearn_mastery.evaluation.statistical_tests import StatisticalTester
from sklearn_mastery.evaluation.utils import ensure_numpy_array, is_higher_better


class PerformanceComparator(LoggerMixin):
    """Evaluate, rank and statistically compare a dictionary of models.

    Args:
        task_type: ``"classification"`` or ``"regression"``.
        random_state: Seed for the shared cross-validation folds.
        alpha: Significance level for the statistical tests.
        n_jobs: Parallel jobs for cross-validation.
        custom_metrics: Extra ``name -> f(y_true, y_pred)`` metrics.
    """

    def __init__(
        self,
        task_type: str = "classification",
        random_state: Optional[int] = None,
        alpha: float = 0.05,
        n_jobs: Optional[int] = None,
        custom_metrics: Optional[Dict[str, CustomMetric]] = None,
    ):
        self.task_type = task_type
        self.random_state = settings.RANDOM_SEED if random_state is None else random_state
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.evaluator = ModelEvaluator(task_type=task_type, custom_metrics=custom_metrics)
        self.tester = StatisticalTester(alpha=alpha, random_state=self.random_state)

    def compare_models(
        self,
        models: Dict[str, BaseEstimator],
        X: Any,
        y: Any,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Hold-out metrics for every *fitted* model.

        Args:
            models: Mapping ``name -> fitted estimator``.
            X: Feature matrix.
            y: Targets.
            sample_weight: Optional per-sample weights.

        Returns:
            Mapping ``name -> metrics`` (see :meth:`ModelEvaluator.evaluate`).
        """
        if not models:
            raise ValueError("models must contain at least one estimator")
        comparison = {
            name: self.evaluator.evaluate(model, X, y, sample_weight=sample_weight)
            for name, model in models.items()
        }
        self.logger.info("Compared %d models on %d samples", len(models), len(ensure_numpy_array(X)))
        return comparison

    def rank_models(
        self,
        models: Dict[str, BaseEstimator],
        X: Any,
        y: Any,
        metric: str = "accuracy",
        sample_weight: Optional[np.ndarray] = None,
    ) -> List[Tuple[str, float]]:
        """Rank fitted models by one metric (best first).

        Error-type metrics (``mse``, ``rmse``, ``mae``, ...) are ranked ascending,
        everything else descending.

        Args:
            models: Mapping ``name -> fitted estimator``.
            X: Feature matrix.
            y: Targets.
            metric: Key of :meth:`compare_models` output to rank by.
            sample_weight: Optional per-sample weights.

        Returns:
            List of ``(name, score)`` tuples.

        Raises:
            KeyError: If ``metric`` is not produced for some model.
        """
        comparison = self.compare_models(models, X, y, sample_weight=sample_weight)
        missing = [name for name, m in comparison.items() if metric not in m]
        if missing:
            raise KeyError(f"Metric {metric!r} not available for models: {missing}")
        ranking = [(name, float(m[metric])) for name, m in comparison.items()]
        ranking.sort(key=lambda item: item[1], reverse=is_higher_better(metric))
        return ranking

    def statistical_comparison(
        self,
        models: Dict[str, BaseEstimator],
        X: Any,
        y: Any,
        cv: CVLike = 5,
        scoring: Optional[str] = None,
        test: str = "auto",
    ) -> Dict[str, Any]:
        """Cross-validate every model on identical folds and test for differences.

        Two models are compared with a paired *t*-test (or Wilcoxon when
        ``test="wilcoxon"``); three or more with the Friedman test followed by
        Nemenyi and Holm-corrected pairwise tests. See
        :meth:`StatisticalTester.compare_multiple_models`.

        Args:
            models: Mapping ``name -> unfitted estimator`` (cloned per fold).
            X: Feature matrix.
            y: Targets.
            cv: Number of folds or a splitter; the same splits are used for all models.
            scoring: Scorer name; defaults to accuracy/R² by estimator type.
            test: ``"auto"``, ``"t"`` or ``"wilcoxon"``.

        Returns:
            Dict with one entry per model (``{scores, mean, std, rank}``) plus
            ``scoring``, ``cv``, ``best_model``, ``p_value``, ``is_significant`` and
            ``significance_test`` (the full test result).
        """
        if len(models) < 2:
            raise ValueError("At least two models are required for a statistical comparison")
        y_arr = ensure_numpy_array(y)
        first = next(iter(models.values()))
        scoring = scoring or default_scoring(first)
        splitter = make_cv(cv, first, y_arr, self.random_state)

        scores: Dict[str, np.ndarray] = {}
        for name, model in models.items():
            scores[name] = np.asarray(
                cross_val_score(
                    model, X, y, cv=splitter, scoring=scoring, n_jobs=self.n_jobs, error_score=np.nan
                )
            )
        test_result = self.tester.compare_multiple_models(scores, test=test)

        means = {name: float(np.nanmean(s)) for name, s in scores.items()}
        order = sorted(means, key=means.get, reverse=is_higher_better(scoring or "score"))
        result: Dict[str, Any] = {
            name: {
                "scores": scores[name],
                "mean": means[name],
                "std": float(np.nanstd(scores[name])),
                "rank": order.index(name) + 1,
            }
            for name in models
        }
        result.update(
            {
                "scoring": scoring,
                "cv": cv,
                "best_model": order[0],
                "p_value": test_result["p_value"],
                "is_significant": test_result["is_significant"],
                "significance_test": test_result,
            }
        )
        self.logger.info(
            "Statistical comparison of %d models (%s): p=%.4f",
            len(models),
            test_result["method"],
            test_result["p_value"],
        )
        return result

    @staticmethod
    def to_dataframe(comparison: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Convert :meth:`compare_models` output into a models x metrics DataFrame."""
        return pd.DataFrame.from_dict(comparison, orient="index")


__all__ = ["PerformanceComparator"]
