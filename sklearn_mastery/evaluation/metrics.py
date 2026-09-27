"""Metric computation and model evaluation.

* :class:`MetricsCalculator` computes classification / regression / clustering
  metrics from predictions, with explicit input validation and a documented
  averaging policy for multi-class problems.
* :class:`ModelEvaluator` evaluates *fitted* estimators on held-out data
  (``evaluate`` / ``detailed_report``) and offers the original comprehensive
  train/test workflow (``evaluate_model``) used by the CLI.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    davies_bouldin_score,
    explained_variance_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    normalized_mutual_info_score,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score, learning_curve

from sklearn_mastery.config.logging_config import LoggerMixin
from sklearn_mastery.config.settings import settings
from sklearn_mastery.evaluation.utils import (
    compute_metric_stability,
    ensure_numpy_array,
    format_metric_value,
    safe_metric_computation,
    to_builtin,
    validate_evaluation_inputs,
    validate_targets,
)

TASK_TYPES = ("classification", "regression", "clustering")
CustomMetric = Callable[[np.ndarray, np.ndarray], float]


def resolve_average(y_true: np.ndarray, y_pred: np.ndarray, average: Optional[str]) -> Optional[str]:
    """Resolve the averaging strategy for precision/recall/F1.

    ``"auto"`` selects ``"binary"`` when exactly two labels occur in
    ``y_true`` and ``y_pred`` together and ``1`` is one of them (so scikit-learn's
    default ``pos_label=1`` applies), otherwise ``"macro"``. Any other value
    is passed through unchanged.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        average: ``"auto"``, ``None`` or a scikit-learn averaging name.

    Returns:
        The averaging string to hand to scikit-learn.
    """
    if average != "auto":
        return average
    labels = np.unique(np.concatenate([np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()]))
    if len(labels) == 2 and any(lbl == 1 for lbl in labels):
        return "binary"
    return "macro"


class MetricsCalculator(LoggerMixin):
    """Compute evaluation metrics from ground truth and predictions.

    Args:
        task_type: ``"classification"``, ``"regression"`` or ``"clustering"``.
        average: Averaging for multi-class precision/recall/F1: ``"auto"``
            (binary when applicable, else macro), ``"macro"``, ``"micro"``,
            ``"weighted"`` or ``"binary"``.
        zero_division: Value returned by precision/recall/F1 when undefined.

    Raises:
        ValueError: If ``task_type`` is unknown.
    """

    def __init__(self, task_type: str = "classification", average: str = "auto", zero_division: float = 0):
        if task_type not in TASK_TYPES:
            raise ValueError(f"Unknown task type: {task_type!r}. Must be one of {list(TASK_TYPES)}")
        self.task_type = task_type
        self.average = average
        self.zero_division = zero_division

    # ------------------------------------------------------------------ #
    # Aggregate
    # ------------------------------------------------------------------ #
    def calculate_all_metrics(
        self,
        y_true: Any,
        y_pred: Any,
        sample_weight: Optional[np.ndarray] = None,
        y_proba: Optional[np.ndarray] = None,
        labels: Optional[Sequence[Any]] = None,
    ) -> Dict[str, float]:
        """Compute the full metric set for the configured task.

        Classification: ``accuracy``, ``balanced_accuracy``, ``precision``,
        ``recall``, ``f1``, ``matthews_corrcoef``, ``cohen_kappa`` and, when
        ``y_proba`` is given, ``roc_auc``. Regression: ``mse``, ``rmse``, ``mae``,
        ``r2``, ``explained_variance``, ``median_absolute_error`` and ``mape``
        (``NaN`` when ``y_true`` contains zeros).

        Args:
            y_true: Ground truth (cluster labels for clustering).
            y_pred: Predictions (feature matrix ``X`` for clustering).
            sample_weight: Optional per-sample weights.
            y_proba: Class probabilities or decision scores for ``roc_auc``.
            labels: Class labels aligned with the columns of ``y_proba``.

        Returns:
            Mapping of metric name to Python float.
        """
        if self.task_type == "clustering":
            return self.clustering_metrics(y_pred, y_true)
        y_true, y_pred = validate_targets(y_true, y_pred)
        if self.task_type == "regression":
            mse = self.mse(y_true, y_pred, sample_weight)
            metrics = {
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "mae": self.mae(y_true, y_pred, sample_weight),
                "r2": self.r2(y_true, y_pred, sample_weight),
                "explained_variance": float(
                    explained_variance_score(y_true, y_pred, sample_weight=sample_weight)
                ),
                "median_absolute_error": float(
                    median_absolute_error(y_true, y_pred, sample_weight=sample_weight)
                ),
                "mape": self.mape(y_true, y_pred, sample_weight),
            }
            return metrics

        avg = resolve_average(y_true, y_pred, self.average)
        metrics = {
            "accuracy": self.accuracy(y_true, y_pred, sample_weight),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight)),
            "precision": self.precision(y_true, y_pred, average=avg, sample_weight=sample_weight),
            "recall": self.recall(y_true, y_pred, average=avg, sample_weight=sample_weight),
            "f1": self.f1_score(y_true, y_pred, average=avg, sample_weight=sample_weight),
            "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight)),
            "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, sample_weight=sample_weight)),
        }
        if y_proba is not None:
            auc = self.roc_auc(y_true, y_proba, labels=labels, sample_weight=sample_weight)
            if auc is not None:
                metrics["roc_auc"] = auc
        return metrics

    # ------------------------------------------------------------------ #
    # Classification
    # ------------------------------------------------------------------ #
    def accuracy(self, y_true: Any, y_pred: Any, sample_weight: Optional[np.ndarray] = None) -> float:
        """Fraction of correctly classified samples.

        Raises:
            ValueError: For empty or length-mismatched inputs.
        """
        y_true, y_pred = validate_targets(y_true, y_pred)
        return float(accuracy_score(y_true, y_pred, sample_weight=sample_weight))

    def precision(
        self,
        y_true: Any,
        y_pred: Any,
        average: Optional[str] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """Precision ``TP / (TP + FP)`` with the resolved averaging strategy."""
        y_true, y_pred = validate_targets(y_true, y_pred)
        avg = resolve_average(y_true, y_pred, average or self.average)
        return float(
            precision_score(
                y_true, y_pred, average=avg, zero_division=self.zero_division, sample_weight=sample_weight
            )
        )

    def recall(
        self,
        y_true: Any,
        y_pred: Any,
        average: Optional[str] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """Recall ``TP / (TP + FN)`` with the resolved averaging strategy."""
        y_true, y_pred = validate_targets(y_true, y_pred)
        avg = resolve_average(y_true, y_pred, average or self.average)
        return float(
            recall_score(
                y_true, y_pred, average=avg, zero_division=self.zero_division, sample_weight=sample_weight
            )
        )

    def f1_score(
        self,
        y_true: Any,
        y_pred: Any,
        average: Optional[str] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """Harmonic mean of precision and recall with the resolved averaging strategy."""
        y_true, y_pred = validate_targets(y_true, y_pred)
        avg = resolve_average(y_true, y_pred, average or self.average)
        return float(
            f1_score(
                y_true, y_pred, average=avg, zero_division=self.zero_division, sample_weight=sample_weight
            )
        )

    def roc_auc(
        self,
        y_true: Any,
        y_score: Any,
        labels: Optional[Sequence[Any]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """Area under the ROC curve; one-vs-rest macro average for multi-class.

        Args:
            y_true: Labels.
            y_score: 1-D positive-class scores, or an ``(n_samples, n_classes)``
                probability matrix (the positive column is used for binary problems).
            labels: Class labels matching the columns of ``y_score``.
            sample_weight: Optional per-sample weights.

        Returns:
            The AUC, or ``None`` when it is undefined (e.g. a single class in ``y_true``).
        """
        y_true = ensure_numpy_array(y_true)
        y_score = ensure_numpy_array(y_score)
        n_classes = len(np.unique(y_true))
        try:
            if y_score.ndim == 2 and y_score.shape[1] == 2 and n_classes <= 2:
                y_score = y_score[:, 1]
            if y_score.ndim == 1:
                return float(roc_auc_score(y_true, y_score, sample_weight=sample_weight))
            return float(
                roc_auc_score(
                    y_true,
                    y_score,
                    multi_class="ovr",
                    average="macro",
                    labels=list(labels) if labels is not None else None,
                    sample_weight=sample_weight,
                )
            )
        except ValueError as exc:
            self.logger.warning("ROC AUC undefined: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Regression
    # ------------------------------------------------------------------ #
    def mse(self, y_true: Any, y_pred: Any, sample_weight: Optional[np.ndarray] = None) -> float:
        """Mean squared error."""
        y_true, y_pred = validate_targets(y_true, y_pred)
        return float(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))

    def rmse(self, y_true: Any, y_pred: Any, sample_weight: Optional[np.ndarray] = None) -> float:
        """Root mean squared error (``sqrt`` of :meth:`mse`)."""
        return float(np.sqrt(self.mse(y_true, y_pred, sample_weight)))

    def mae(self, y_true: Any, y_pred: Any, sample_weight: Optional[np.ndarray] = None) -> float:
        """Mean absolute error."""
        y_true, y_pred = validate_targets(y_true, y_pred)
        return float(mean_absolute_error(y_true, y_pred, sample_weight=sample_weight))

    def r2(self, y_true: Any, y_pred: Any, sample_weight: Optional[np.ndarray] = None) -> float:
        """Coefficient of determination."""
        y_true, y_pred = validate_targets(y_true, y_pred)
        return float(r2_score(y_true, y_pred, sample_weight=sample_weight))

    def mape(self, y_true: Any, y_pred: Any, sample_weight: Optional[np.ndarray] = None) -> float:
        """Mean absolute percentage error; ``NaN`` when ``y_true`` contains zeros."""
        y_true, y_pred = validate_targets(y_true, y_pred)
        if np.any(y_true == 0):
            return float("nan")
        return float(mean_absolute_percentage_error(y_true, y_pred, sample_weight=sample_weight))

    # ------------------------------------------------------------------ #
    # Clustering
    # ------------------------------------------------------------------ #
    def clustering_metrics(self, X: Any, labels: Any, y_true: Optional[Any] = None) -> Dict[str, float]:
        """Internal (and, if ``y_true`` is given, external) clustering metrics.

        Args:
            X: Feature matrix used for clustering.
            labels: Cluster assignments.
            y_true: Optional ground-truth labels.

        Returns:
            Dict with ``silhouette_score``, ``calinski_harabasz_score``,
            ``davies_bouldin_score``, ``n_clusters`` and, with ``y_true``,
            ``adjusted_rand_score`` and ``normalized_mutual_info``.
        """
        X = ensure_numpy_array(X)
        labels = ensure_numpy_array(labels)
        metrics = {
            "silhouette_score": safe_metric_computation(silhouette_score, X, labels),
            "calinski_harabasz_score": safe_metric_computation(calinski_harabasz_score, X, labels),
            "davies_bouldin_score": safe_metric_computation(davies_bouldin_score, X, labels),
            "n_clusters": len(np.unique(labels[labels != -1])),
        }
        if y_true is not None:
            metrics["adjusted_rand_score"] = safe_metric_computation(adjusted_rand_score, y_true, labels)
            metrics["normalized_mutual_info"] = safe_metric_computation(
                normalized_mutual_info_score, y_true, labels
            )
        return metrics


class ModelEvaluator(LoggerMixin):
    """Evaluate fitted estimators on held-out data.

    Args:
        task_type: ``"classification"``, ``"regression"`` or ``"clustering"``.
        custom_metrics: Mapping ``name -> f(y_true, y_pred)`` added to every
            evaluation.
        average: Averaging strategy forwarded to :class:`MetricsCalculator`.

    Raises:
        ValueError: If ``task_type`` is unknown.
    """

    def __init__(
        self,
        task_type: str = "classification",
        custom_metrics: Optional[Dict[str, CustomMetric]] = None,
        average: str = "auto",
    ):
        if task_type not in TASK_TYPES:
            raise ValueError(f"Unknown task type: {task_type!r}. Must be one of {list(TASK_TYPES)}")
        self.task_type = task_type
        self.custom_metrics: Dict[str, CustomMetric] = dict(custom_metrics or {})
        self.calculator = MetricsCalculator(task_type=task_type, average=average)
        self.results: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        model: BaseEstimator,
        X: Any,
        y: Any,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Score a fitted model on ``(X, y)``.

        For classification the ROC AUC is computed from ``predict_proba`` (or
        ``decision_function`` for binary problems) and omitted when neither is
        available. Custom metrics receive the hard predictions.

        Args:
            model: Fitted estimator (``predict``; optionally ``predict_proba``).
            X: Feature matrix.
            y: Targets (ignored for clustering unless given, then used for external metrics).
            sample_weight: Optional per-sample weights for the built-in metrics.

        Returns:
            Flat mapping of metric name to float (see
            :meth:`MetricsCalculator.calculate_all_metrics` for the keys).

        Raises:
            AttributeError: If ``model`` lacks the required prediction method.
            ValueError: If ``X``/``y`` are inconsistent or empty.
        """
        X = ensure_numpy_array(X)
        if self.task_type == "clustering":
            labels = model.labels_ if hasattr(model, "labels_") else model.fit_predict(X)
            metrics = self.calculator.clustering_metrics(X, labels, y)
            return self._add_custom_metrics(metrics, ensure_numpy_array(labels), ensure_numpy_array(labels))

        y = ensure_numpy_array(y)
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length: {len(X)} vs {len(y)}")
        if len(X) == 0:
            raise ValueError("Empty dataset provided")

        y_pred = ensure_numpy_array(model.predict(X))
        if self.task_type == "classification":
            y_score = self._predict_scores(model, X)
            metrics = self.calculator.calculate_all_metrics(
                y,
                y_pred,
                sample_weight=sample_weight,
                y_proba=y_score,
                labels=getattr(model, "classes_", None),
            )
        else:
            metrics = self.calculator.calculate_all_metrics(y, y_pred, sample_weight=sample_weight)
        return self._add_custom_metrics(metrics, y, y_pred)

    def detailed_report(
        self,
        model: BaseEstimator,
        X: Any,
        y: Any,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Metrics plus task-specific diagnostics.

        Classification adds the ``confusion_matrix`` (``np.ndarray``), the
        scikit-learn ``classification_report`` dict and the ``labels``; regression
        adds a ``residuals`` summary. All nested values are JSON-serialisable
        except the confusion-matrix array.

        Args:
            model: Fitted estimator.
            X: Feature matrix.
            y: Targets.
            sample_weight: Optional per-sample weights.

        Returns:
            Dict with ``task_type``, ``model_name``, ``n_samples``, ``metrics`` and
            the extras described above.
        """
        metrics = self.evaluate(model, X, y, sample_weight=sample_weight)
        X = ensure_numpy_array(X)
        report: Dict[str, Any] = {
            "task_type": self.task_type,
            "model_name": type(model).__name__,
            "n_samples": len(X),
            "metrics": to_builtin(metrics),
        }
        if self.task_type == "clustering":
            return report
        y = ensure_numpy_array(y)
        y_pred = ensure_numpy_array(model.predict(X))
        if self.task_type == "classification":
            labels = np.unique(np.concatenate([y, y_pred]))
            report["labels"] = to_builtin(labels)
            report["confusion_matrix"] = confusion_matrix(
                y, y_pred, labels=labels, sample_weight=sample_weight
            )
            report["classification_report"] = to_builtin(
                classification_report(y, y_pred, labels=labels, output_dict=True, zero_division=0)
            )
        else:
            residuals = y.astype(float) - y_pred.astype(float)
            report["residuals"] = {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "min": float(np.min(residuals)),
                "max": float(np.max(residuals)),
                "q25": float(np.percentile(residuals, 25)),
                "q75": float(np.percentile(residuals, 75)),
            }
        return report

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _add_custom_metrics(
        self, metrics: Dict[str, float], y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        for name, fn in self.custom_metrics.items():
            metrics[name] = float(fn(y_true, y_pred))
        return metrics

    @staticmethod
    def _predict_scores(model: BaseEstimator, X: np.ndarray) -> Optional[np.ndarray]:
        """Class probabilities, else binary decision scores, else ``None``."""
        if hasattr(model, "predict_proba"):
            try:
                return ensure_numpy_array(model.predict_proba(X))
            except (AttributeError, ValueError, NotImplementedError):
                pass
        if hasattr(model, "decision_function"):
            try:
                scores = ensure_numpy_array(model.decision_function(X))
                return scores if scores.ndim == 1 else None
            except (AttributeError, ValueError, NotImplementedError):
                pass
        return None

    # ------------------------------------------------------------------ #
    # Legacy comprehensive workflow (used by the CLI)
    # ------------------------------------------------------------------ #
    def evaluate_model(
        self,
        model: BaseEstimator,
        X_train: Any,
        X_test: Any,
        y_train: Any,
        y_test: Any,
        model_name: str = "Model",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Comprehensive train/test evaluation with cross-validation and learning curves.

        Args:
            model: Fitted estimator.
            X_train: Training features.
            X_test: Test features.
            y_train: Training targets.
            y_test: Test targets.
            model_name: Label used in logs and the stored ``results``.
            **kwargs: ``include_calibration``, ``include_fairness`` (with
                ``sensitive_attrs``), ``include_feature_importance`` (with ``feature_names``).

        Returns:
            Nested results dict keyed by ``test_*``/``train_*`` metrics, plus
            ``cross_validation`` and ``learning_curves`` for supervised tasks.
        """
        self.logger.info("Evaluating %s for %s", model_name, self.task_type)
        validate_evaluation_inputs(X_test, y_test, self.task_type)

        results: Dict[str, Any] = {
            "model_name": model_name,
            "task_type": self.task_type,
            "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        }
        if self.task_type == "classification":
            results.update(self._evaluate_classification(model, X_train, X_test, y_train, y_test))
        elif self.task_type == "regression":
            results.update(self._evaluate_regression(model, X_train, X_test, y_train, y_test))
        else:
            results.update(self._evaluate_clustering(model, X_train, y_train))

        if self.task_type != "clustering":
            results["cross_validation"] = self._cross_validation_analysis(model, X_train, y_train)
            results["learning_curves"] = self._compute_learning_curves(model, X_train, y_train)

        if kwargs.get("include_calibration") and hasattr(model, "predict_proba"):
            results["calibration"] = self.evaluate_calibration(y_test, model.predict_proba(X_test))
        if kwargs.get("include_fairness") and "sensitive_attrs" in kwargs:
            results["fairness"] = self.evaluate_fairness(
                y_test, model.predict(X_test), kwargs["sensitive_attrs"]
            )
        if kwargs.get("include_feature_importance"):
            results["feature_importance"] = self.analyze_feature_importance(
                model, kwargs.get("feature_names")
            )

        self.results[model_name] = results
        return results

    def _evaluate_classification(
        self, model: BaseEstimator, X_train: Any, X_test: Any, y_train: Any, y_test: Any
    ) -> Dict[str, Any]:
        y_train, y_test = ensure_numpy_array(y_train), ensure_numpy_array(y_test)
        y_pred = ensure_numpy_array(model.predict(X_test))
        y_train_pred = ensure_numpy_array(model.predict(X_train))
        proba_test = self._predict_scores(model, X_test)
        proba_train = self._predict_scores(model, X_train)
        classes = getattr(model, "classes_", None)

        test_metrics = self.calculator.calculate_all_metrics(
            y_test, y_pred, y_proba=proba_test, labels=classes
        )
        train_metrics = self.calculator.calculate_all_metrics(
            y_train, y_train_pred, y_proba=proba_train, labels=classes
        )
        results: Dict[str, Any] = {}
        for key in ("accuracy", "precision", "recall", "f1"):
            results[f"test_{key}"] = test_metrics[key]
            results[f"train_{key}"] = train_metrics[key]
        results["test_roc_auc"] = test_metrics.get("roc_auc")
        results["train_roc_auc"] = train_metrics.get("roc_auc")

        results["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        results["classification_report"] = to_builtin(
            classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        )
        if (
            proba_test is not None
            and proba_test.ndim == 2
            and proba_test.shape[1] == 2
            and len(np.unique(y_test)) == 2
        ):
            fpr, tpr, thr = roc_curve(y_test, proba_test[:, 1])
            results["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist()}
            prec, rec, pr_thr = precision_recall_curve(y_test, proba_test[:, 1])
            results["pr_curve"] = {
                "precision": prec.tolist(),
                "recall": rec.tolist(),
                "thresholds": pr_thr.tolist(),
            }
        results["overfitting_score"] = results["train_accuracy"] - results["test_accuracy"]
        return results

    def _evaluate_regression(
        self, model: BaseEstimator, X_train: Any, X_test: Any, y_train: Any, y_test: Any
    ) -> Dict[str, Any]:
        y_train, y_test = ensure_numpy_array(y_train), ensure_numpy_array(y_test)
        y_pred = ensure_numpy_array(model.predict(X_test))
        y_train_pred = ensure_numpy_array(model.predict(X_train))
        test_metrics = self.calculator.calculate_all_metrics(y_test, y_pred)
        train_metrics = self.calculator.calculate_all_metrics(y_train, y_train_pred)
        results: Dict[str, Any] = {}
        for key in ("mse", "rmse", "mae", "r2", "mape"):
            results[f"test_{key}"] = None if np.isnan(test_metrics[key]) else test_metrics[key]
            results[f"train_{key}"] = None if np.isnan(train_metrics[key]) else train_metrics[key]
        residuals = y_test.astype(float) - y_pred.astype(float)
        results["residuals"] = {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
            "q25": float(np.percentile(residuals, 25)),
            "q75": float(np.percentile(residuals, 75)),
        }
        if len(y_test) > 1:
            corr = np.corrcoef(y_test.astype(float), y_pred.astype(float))[0, 1]
            results["pred_actual_corr"] = None if np.isnan(corr) else float(corr)
        else:
            results["pred_actual_corr"] = None
        results["overfitting_score"] = results["train_r2"] - results["test_r2"]
        return results

    def _evaluate_clustering(
        self, model: BaseEstimator, X: Any, y_true: Optional[Any] = None
    ) -> Dict[str, Any]:
        X = ensure_numpy_array(X)
        labels = ensure_numpy_array(model.labels_ if hasattr(model, "labels_") else model.fit_predict(X))
        results: Dict[str, Any] = self.calculator.clustering_metrics(X, labels, y_true)
        unique_labels, counts = np.unique(labels, return_counts=True)
        results["cluster_sizes"] = dict(zip(unique_labels.tolist(), counts.tolist()))
        if -1 in labels:
            results["n_noise_points"] = int(np.sum(labels == -1))
            results["noise_ratio"] = float(results["n_noise_points"] / len(labels))
        return results

    def _cross_validation_analysis(
        self, model: BaseEstimator, X: Any, y: Any, cv_folds: Optional[int] = None
    ) -> Dict[str, Any]:
        cv_folds = cv_folds or settings.DEFAULT_CV_FOLDS
        if self.task_type == "classification":
            scoring_metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
            scoring_metrics.append("roc_auc" if len(np.unique(ensure_numpy_array(y))) == 2 else "roc_auc_ovr")
        else:
            scoring_metrics = ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]

        cv_results: Dict[str, Any] = {}
        for metric in scoring_metrics:
            try:
                scores = cross_val_score(
                    model, X, y, cv=cv_folds, scoring=metric, n_jobs=settings.DEFAULT_N_JOBS
                )
            except (ValueError, TypeError) as exc:
                self.logger.warning("Could not compute CV score for %s: %s", metric, exc)
                cv_results[metric] = None
                continue
            stability = compute_metric_stability(scores)
            cv_results[metric] = {
                "mean": stability["mean"],
                "std": stability["std"],
                "scores": scores.tolist(),
                "stability": stability,
            }
        return cv_results

    def _compute_learning_curves(
        self, model: BaseEstimator, X: Any, y: Any, train_sizes: Optional[np.ndarray] = None
    ) -> Optional[Dict[str, Any]]:
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        scoring = "accuracy" if self.task_type == "classification" else "r2"
        try:
            sizes, train_scores, val_scores = learning_curve(
                model,
                X,
                y,
                train_sizes=train_sizes,
                cv=settings.DEFAULT_CV_FOLDS,
                scoring=scoring,
                n_jobs=settings.DEFAULT_N_JOBS,
                shuffle=True,
                random_state=settings.RANDOM_SEED,
            )
        except (ValueError, TypeError) as exc:
            self.logger.warning("Could not compute learning curves: %s", exc)
            return None
        return {
            "train_sizes": sizes.tolist(),
            "train_scores_mean": np.mean(train_scores, axis=1).tolist(),
            "train_scores_std": np.std(train_scores, axis=1).tolist(),
            "val_scores_mean": np.mean(val_scores, axis=1).tolist(),
            "val_scores_std": np.std(val_scores, axis=1).tolist(),
            "scoring_metric": scoring,
        }

    def evaluate_calibration(self, y_true: Any, y_pred_proba: Any, n_bins: int = 10) -> Dict[str, Any]:
        """Brier score, expected calibration error and calibration curve (binary only).

        Delegates to :class:`sklearn_mastery.evaluation.analyzers.CalibrationAnalyzer`.
        """
        from sklearn_mastery.evaluation.analyzers import CalibrationAnalyzer

        y_true = ensure_numpy_array(y_true)
        y_pred_proba = ensure_numpy_array(y_pred_proba)
        if len(np.unique(y_true)) != 2 or (y_pred_proba.ndim == 2 and y_pred_proba.shape[1] != 2):
            self.logger.warning("Calibration evaluation requires binary classification")
            return {}
        analyzer = CalibrationAnalyzer(n_bins=n_bins)
        metrics = analyzer.calculate_calibration_metrics(y_true, y_pred_proba)
        curve = analyzer.generate_calibration_curve(y_true, y_pred_proba)
        return {
            "brier_score": metrics["brier_score"],
            "expected_calibration_error": metrics["ece"],
            "maximum_calibration_error": metrics["mce"],
            "calibration_curve": {
                "fraction_of_positives": curve["fraction_of_positives"].tolist(),
                "mean_predicted_value": curve["mean_predicted_value"].tolist(),
            },
        }

    def analyze_feature_importance(
        self, model: BaseEstimator, feature_names: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Summarise ``feature_importances_`` when the model exposes them."""
        if not hasattr(model, "feature_importances_"):
            return None
        importances = np.asarray(model.feature_importances_, dtype=float)
        names = feature_names or [f"feature_{i}" for i in range(len(importances))]
        df = pd.DataFrame({"feature": names, "importance": importances}).sort_values(
            "importance", ascending=False
        )
        return {
            "feature_importance_df": df.to_dict("records"),
            "summary_stats": {
                "mean_importance": float(np.mean(importances)),
                "std_importance": float(np.std(importances)),
                "max_importance": float(np.max(importances)),
                "min_importance": float(np.min(importances)),
                "n_zero_importance": int(np.sum(importances == 0)),
            },
            "top_features": df.head(10).to_dict("records"),
        }

    def evaluate_fairness(self, y_true: Any, y_pred: Any, sensitive_attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Per-group accuracy and demographic-parity difference for each sensitive attribute."""
        y_true, y_pred = validate_targets(y_true, y_pred)
        fairness: Dict[str, Any] = {}
        for attr_name, values in sensitive_attrs.items():
            values = ensure_numpy_array(values)
            if len(values) != len(y_true):
                self.logger.warning("Sensitive attribute %s length mismatch", attr_name)
                continue
            group_metrics: Dict[str, float] = {}
            positive_rates: Dict[Any, float] = {}
            for group in np.unique(values):
                mask = values == group
                group_metrics[f"accuracy_group_{group}"] = float(accuracy_score(y_true[mask], y_pred[mask]))
                positive_rates[group] = float(np.mean(y_pred[mask] == 1))
            if len(positive_rates) > 1:
                group_metrics["demographic_parity_diff"] = max(positive_rates.values()) - min(
                    positive_rates.values()
                )
            fairness[attr_name] = group_metrics
        return fairness

    def evaluate_time_series(
        self, y_true: Any, y_pred: Any, seasonal_period: Optional[int] = None
    ) -> Dict[str, float]:
        """Regression metrics plus MASE relative to a seasonal-naive forecast (Hyndman & Koehler, 2006)."""
        y_true, y_pred = validate_targets(y_true, y_pred)
        calc = MetricsCalculator("regression")
        results: Dict[str, float] = calc.calculate_all_metrics(y_true, y_pred)
        if seasonal_period and 0 < seasonal_period < len(y_true):
            naive_errors = np.abs(y_true[seasonal_period:] - y_true[:-seasonal_period])
            mae_naive = float(np.mean(naive_errors))
            results["mase"] = results["mae"] / mae_naive if mae_naive > 0 else float("inf")
        return results

    def compare_models(self, results_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """Tabulate several :meth:`evaluate_model` results, sorted by the primary metric."""
        primary = {
            "classification": "test_accuracy",
            "regression": "test_r2",
            "clustering": "silhouette_score",
        }
        columns = {
            "classification": [
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_roc_auc",
                "overfitting_score",
            ],
            "regression": ["test_r2", "test_rmse", "test_mae", "test_mape", "overfitting_score"],
            "clustering": [
                "silhouette_score",
                "calinski_harabasz_score",
                "davies_bouldin_score",
                "n_clusters",
                "adjusted_rand_score",
            ],
        }[self.task_type]
        rows = []
        for result in results_list:
            row: Dict[str, Any] = {"model_name": result.get("model_name")}
            row.update({c: result.get(c) for c in columns})
            for metric, values in (result.get("cross_validation") or {}).items():
                if values is not None:
                    row[f"cv_{metric}_mean"] = values["mean"]
                    row[f"cv_{metric}_std"] = values["std"]
            rows.append(row)
        df = pd.DataFrame(rows)
        key = primary[self.task_type]
        if key in df.columns:
            df = df.sort_values(key, ascending=(key == "davies_bouldin_score"))
        return df

    def generate_evaluation_summary(self, result: Dict[str, Any]) -> str:
        """Human-readable multi-line summary of an :meth:`evaluate_model` result."""
        fmt = format_metric_value
        task_type = result["task_type"]
        lines = [f"=== {result['model_name']} Evaluation Summary ===", f"Task Type: {task_type.title()}", ""]
        if task_type == "classification":
            lines += [
                f"Test Accuracy: {fmt(result.get('test_accuracy'), 'accuracy')}",
                f"Test Precision: {fmt(result.get('test_precision'), 'precision')}",
                f"Test Recall: {fmt(result.get('test_recall'), 'recall')}",
                f"Test F1-Score: {fmt(result.get('test_f1'), 'f1')}",
            ]
            if result.get("test_roc_auc") is not None:
                lines.append(f"Test ROC AUC: {fmt(result['test_roc_auc'], 'roc_auc')}")
            lines += [
                "",
                f"Training Accuracy: {fmt(result.get('train_accuracy'), 'accuracy')}",
                f"Overfitting Score: {fmt(result.get('overfitting_score'), 'overfitting')}",
            ]
        elif task_type == "regression":
            lines += [
                f"Test R²: {fmt(result.get('test_r2'), 'r2')}",
                f"Test RMSE: {fmt(result.get('test_rmse'), 'rmse')}",
                f"Test MAE: {fmt(result.get('test_mae'), 'mae')}",
            ]
            if result.get("test_mape") is not None:
                lines.append(f"Test MAPE: {fmt(result['test_mape'], 'mape')}")
            lines += [
                "",
                f"Training R²: {fmt(result.get('train_r2'), 'r2')}",
                f"Overfitting Score: {fmt(result.get('overfitting_score'), 'overfitting')}",
            ]
        else:
            lines += [
                f"Number of Clusters: {result.get('n_clusters', 'N/A')}",
                f"Silhouette Score: {fmt(result.get('silhouette_score'), 'silhouette')}",
                f"Calinski-Harabasz Score: {fmt(result.get('calinski_harabasz_score'), 'ch')}",
                f"Davies-Bouldin Score: {fmt(result.get('davies_bouldin_score'), 'db')}",
            ]
            if result.get("adjusted_rand_score") is not None:
                lines.append(f"Adjusted Rand Score: {fmt(result['adjusted_rand_score'], 'ari')}")

        cv_results = result.get("cross_validation") or {}
        if cv_results:
            lines += ["", "Cross-Validation Results:"]
            for metric, values in cv_results.items():
                if values is not None:
                    lines.append(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
        if result.get("calibration"):
            cal = result["calibration"]
            lines += [
                "",
                "Calibration Analysis:",
                f"  Brier Score: {fmt(cal.get('brier_score'), 'brier')}",
                f"  Expected Calibration Error: {fmt(cal.get('expected_calibration_error'), 'ece')}",
            ]
        if result.get("fairness"):
            lines += ["", "Fairness Analysis:"]
            for attr_name, fm in result["fairness"].items():
                if "demographic_parity_diff" in fm:
                    lines.append(
                        f"  {attr_name} Demographic Parity Diff: {fm['demographic_parity_diff']:.4f}"
                    )
        return "\n".join(lines)


__all__ = ["TASK_TYPES", "MetricsCalculator", "ModelEvaluator", "resolve_average"]
