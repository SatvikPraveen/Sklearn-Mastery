"""Diagnostic analyzers for fitted models and their predictions.

Each analyzer is a small, stateless helper that turns predictions into
structured, plot-ready dictionaries:

* :class:`ResidualAnalyzer` - regression residual diagnostics (normality,
  heteroscedasticity, autocorrelation).
* :class:`FeatureImportanceAnalyzer` - impurity, coefficient and permutation
  importances with ranking helpers.
* :class:`ConfusionMatrixAnalyzer` - confusion matrices and per-class metrics.
* :class:`ROCAnalyzer` / :class:`PrecisionRecallAnalyzer` - threshold curves
  and optimal operating points.
* :class:`CalibrationAnalyzer` - reliability diagrams, Brier score, ECE/MCE.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from sklearn_mastery.config.logging_config import LoggerMixin
from sklearn_mastery.config.settings import settings
from sklearn_mastery.evaluation.utils import (
    ensure_numpy_array,
    to_builtin,
    validate_binary_probabilities,
    validate_targets,
)


# --------------------------------------------------------------------------- #
# Residuals
# --------------------------------------------------------------------------- #
class ResidualAnalyzer(LoggerMixin):
    """Residual diagnostics for regression predictions.

    Args:
        alpha: Significance level for the normality and heteroscedasticity tests.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def analyze_residuals(self, y_true: Any, y_pred: Any) -> Dict[str, Any]:
        """Residual statistics, hypothesis tests and plot-ready data.

        Args:
            y_true: Observed values.
            y_pred: Fitted/predicted values.

        Returns:
            Dict with ``residuals``, ``mean_residual``, ``std_residual``,
            ``median_residual``, ``max_abs_residual``, ``skewness``, ``kurtosis``
            (excess), ``durbin_watson`` (Durbin & Watson, 1950; ~2 means no lag-1
            autocorrelation), ``normality``, ``heteroscedasticity`` and
            ``residual_plots`` (residuals-vs-fitted, Q-Q, histogram, scale-location).
        """
        y_true, y_pred = validate_targets(y_true, y_pred)
        residuals = y_true.astype(float) - y_pred.astype(float)
        std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
        standardized = residuals / std if std > 0 else np.zeros_like(residuals)
        (theo_q, sample_q), _ = stats.probplot(residuals, dist="norm")
        counts, bin_edges = np.histogram(residuals, bins="auto")
        dw = (
            float(np.sum(np.diff(residuals) ** 2) / np.sum(residuals**2))
            if np.any(residuals != 0)
            else float("nan")
        )
        return {
            "residuals": residuals,
            "mean_residual": float(np.mean(residuals)),
            "std_residual": std,
            "median_residual": float(np.median(residuals)),
            "max_abs_residual": float(np.max(np.abs(residuals))),
            "skewness": float(stats.skew(residuals)) if len(residuals) > 2 else float("nan"),
            "kurtosis": float(stats.kurtosis(residuals)) if len(residuals) > 3 else float("nan"),
            "durbin_watson": dw,
            "normality": self.test_normality(y_true, y_pred),
            "heteroscedasticity": self.test_heteroscedasticity(y_true, y_pred),
            "residual_plots": {
                "residuals_vs_fitted": {"fitted": y_pred.astype(float), "residuals": residuals},
                "qq_plot": {
                    "theoretical_quantiles": np.asarray(theo_q),
                    "sample_quantiles": np.asarray(sample_q),
                },
                "histogram": {"counts": counts, "bin_edges": bin_edges},
                "scale_location": {
                    "fitted": y_pred.astype(float),
                    "sqrt_abs_standardized_residuals": np.sqrt(np.abs(standardized)),
                },
            },
        }

    def test_normality(self, y_true: Any, y_pred: Any) -> Dict[str, Any]:
        """Test whether the residuals are normally distributed.

        Uses the Shapiro-Wilk test for ``n <= 5000`` (Shapiro & Wilk, 1965;
        Royston, 1995) and D'Agostino-Pearson's K² test otherwise
        (D'Agostino, Belanger & D'Agostino, 1990).

        Args:
            y_true: Observed values.
            y_pred: Predicted values.

        Returns:
            Dict with ``test``, ``statistic``, ``p_value``, ``is_normal``
            (``p >= alpha``) and ``alpha``.

        Raises:
            ValueError: If fewer than three residuals are available.
        """
        y_true, y_pred = validate_targets(y_true, y_pred)
        residuals = y_true.astype(float) - y_pred.astype(float)
        n = len(residuals)
        if n < 3:
            raise ValueError("At least three residuals are required for a normality test")
        if np.allclose(residuals, residuals[0]):
            return {
                "test": "shapiro_wilk",
                "statistic": float("nan"),
                "p_value": 1.0,
                "is_normal": True,
                "alpha": self.alpha,
            }
        if n <= 5000:
            statistic, p_value = stats.shapiro(residuals)
            test = "shapiro_wilk"
        else:
            statistic, p_value = stats.normaltest(residuals)
            test = "dagostino_pearson"
        return {
            "test": test,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "is_normal": bool(p_value >= self.alpha),
            "alpha": self.alpha,
        }

    def test_heteroscedasticity(self, y_true: Any, y_pred: Any) -> Dict[str, Any]:
        """Breusch-Pagan test for heteroscedasticity (Koenker's studentized form).

        Regresses the squared residuals on the fitted values; the Lagrange
        multiplier statistic ``LM = n * R²`` follows a chi-square distribution
        with one degree of freedom under homoscedasticity.

        Args:
            y_true: Observed values.
            y_pred: Fitted values (the explanatory variable of the auxiliary regression).

        Returns:
            Dict with ``test``, ``statistic``, ``p_value``, ``has_heteroscedasticity``
            (``p < alpha``) and ``alpha``.

        References:
            Breusch, T. S. & Pagan, A. R. (1979). A simple test for heteroscedasticity
            and random coefficient variation. *Econometrica*, 47(5), 1287-1294.
            Koenker, R. (1981). A note on studentizing a test for heteroscedasticity.
            *Journal of Econometrics*, 17(1), 107-112.
        """
        y_true, y_pred = validate_targets(y_true, y_pred)
        fitted = y_pred.astype(float)
        residuals = y_true.astype(float) - fitted
        n = len(residuals)
        if n < 3:
            raise ValueError("At least three observations are required")
        sq = residuals**2
        if np.allclose(fitted, fitted[0]) or np.allclose(sq, sq[0]):
            r2 = 0.0
        else:
            r2 = float(stats.linregress(fitted, sq).rvalue ** 2)
        lm = n * r2
        p_value = float(stats.chi2.sf(lm, df=1))
        return {
            "test": "breusch_pagan",
            "statistic": float(lm),
            "p_value": p_value,
            "has_heteroscedasticity": bool(p_value < self.alpha),
            "alpha": self.alpha,
        }


# --------------------------------------------------------------------------- #
# Feature importance
# --------------------------------------------------------------------------- #
class FeatureImportanceAnalyzer(LoggerMixin):
    """Extract and rank feature importances from fitted models.

    Args:
        random_state: Seed for permutation importance.
        n_repeats: Permutation repeats per feature.
        scoring: Scorer for permutation importance (``None`` = estimator score).
        n_jobs: Parallel jobs for permutation importance.
    """

    def __init__(
        self,
        random_state: Optional[int] = None,
        n_repeats: int = 5,
        scoring: Optional[str] = None,
        n_jobs: Optional[int] = None,
    ):
        self.random_state = settings.RANDOM_SEED if random_state is None else random_state
        self.n_repeats = n_repeats
        self.scoring = scoring
        self.n_jobs = n_jobs

    @staticmethod
    def _final_estimator(model: BaseEstimator) -> BaseEstimator:
        return model.steps[-1][1] if isinstance(model, Pipeline) else model

    def get_feature_importance(
        self,
        model: BaseEstimator,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        method: str = "auto",
    ) -> np.ndarray:
        """Return one importance value per feature.

        Methods:

        * ``"tree_based"``: ``feature_importances_`` (mean impurity decrease).
        * ``"coefficients"``: ``|coef_|`` (averaged over outputs/classes).
        * ``"permutation"``: mean score drop when a feature is shuffled
          (Breiman, 2001; Fisher, Rudin & Dominici, 2019); requires ``X`` and ``y``.
        * ``"auto"``: the first of the above that applies.

        Args:
            model: Fitted estimator (a :class:`~sklearn.pipeline.Pipeline` is unwrapped
                for the attribute-based methods).
            X: Feature matrix (permutation only).
            y: Targets (permutation only).
            method: Importance method.

        Returns:
            Array of shape ``(n_features,)``.

        Raises:
            ValueError: If the method is unknown or not applicable to ``model``.
        """
        est = self._final_estimator(model)
        if method == "auto":
            if hasattr(est, "feature_importances_"):
                method = "tree_based"
            elif hasattr(est, "coef_"):
                method = "coefficients"
            elif X is not None and y is not None:
                method = "permutation"
            else:
                raise ValueError(
                    "Model exposes neither feature_importances_ nor coef_; pass X and y for permutation"
                )
        if method == "tree_based":
            if not hasattr(est, "feature_importances_"):
                raise ValueError(f"{type(est).__name__} has no feature_importances_")
            return np.asarray(est.feature_importances_, dtype=float)
        if method == "coefficients":
            if not hasattr(est, "coef_"):
                raise ValueError(f"{type(est).__name__} has no coef_")
            coef = np.abs(np.asarray(est.coef_, dtype=float))
            return coef.mean(axis=0) if coef.ndim > 1 else coef
        if method == "permutation":
            if X is None or y is None:
                raise ValueError("Permutation importance requires X and y")
            result = permutation_importance(
                model,
                X,
                y,
                scoring=self.scoring,
                n_repeats=self.n_repeats,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            self.permutation_result_ = result
            return np.asarray(result.importances_mean, dtype=float)
        raise ValueError(f"Unknown importance method: {method!r}")

    def rank_features(
        self,
        model: BaseEstimator,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        method: str = "auto",
        feature_names: Optional[Sequence[str]] = None,
    ) -> List[Tuple[Any, float]]:
        """Features sorted by decreasing importance.

        Args:
            model: Fitted estimator.
            X: Feature matrix (permutation only).
            y: Targets (permutation only).
            method: See :meth:`get_feature_importance`.
            feature_names: Labels; defaults to column indices.

        Returns:
            List of ``(feature, importance)`` tuples, most important first.
        """
        importance = self.get_feature_importance(model, X, y, method)
        names: List[Any] = list(feature_names) if feature_names is not None else list(range(len(importance)))
        if len(names) != len(importance):
            raise ValueError("feature_names length does not match the number of features")
        order = np.argsort(-importance, kind="stable")
        return [(names[i], float(importance[i])) for i in order]

    def get_top_k_features(
        self,
        model: BaseEstimator,
        k: int = 5,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        method: str = "auto",
        feature_names: Optional[Sequence[str]] = None,
    ) -> List[Tuple[Any, float]]:
        """The ``k`` most important features (see :meth:`rank_features`)."""
        if k < 1:
            raise ValueError("k must be at least 1")
        return self.rank_features(model, X, y, method, feature_names)[:k]


# --------------------------------------------------------------------------- #
# Confusion matrix
# --------------------------------------------------------------------------- #
class ConfusionMatrixAnalyzer(LoggerMixin):
    """Confusion-matrix construction and per-class metrics.

    Args:
        labels: Fixed label order; defaults to the sorted union of true and predicted labels.
    """

    def __init__(self, labels: Optional[Sequence[Any]] = None):
        self.labels = list(labels) if labels is not None else None

    def _labels(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        if self.labels is not None:
            return np.asarray(self.labels)
        return np.unique(np.concatenate([y_true, y_pred]))

    def analyze_confusion_matrix(self, y_true: Any, y_pred: Any) -> Dict[str, Any]:
        """Confusion matrix with row-normalised version and per-class metrics.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Dict with ``confusion_matrix`` (rows = true, columns = predicted),
            ``normalized_cm`` (rows sum to 1; empty rows are 0), ``labels``,
            ``class_metrics``, ``accuracy``, ``n_samples`` and ``most_confused_pair``
            (``(true, predicted, count)`` of the largest off-diagonal cell, or ``None``).
        """
        y_true, y_pred = validate_targets(y_true, y_pred)
        labels = self._labels(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        row_sums = cm.sum(axis=1, keepdims=True)
        normalized = np.divide(cm, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums > 0)
        off_diag = cm.astype(float).copy()
        np.fill_diagonal(off_diag, -1)
        i, j = np.unravel_index(int(np.argmax(off_diag)), cm.shape)
        most_confused = (
            (to_builtin(labels[i]), to_builtin(labels[j]), int(cm[i, j])) if cm[i, j] > 0 else None
        )
        return {
            "confusion_matrix": cm,
            "normalized_cm": normalized,
            "labels": to_builtin(labels),
            "class_metrics": self.get_class_metrics(y_true, y_pred),
            "accuracy": float(np.trace(cm) / cm.sum()) if cm.sum() else float("nan"),
            "n_samples": int(cm.sum()),
            "most_confused_pair": most_confused,
        }

    def get_class_metrics(self, y_true: Any, y_pred: Any) -> Dict[Any, Dict[str, float]]:
        """Per-class precision, recall, F1, support, specificity and FPR.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Mapping ``label -> {precision, recall, f1_score, support, specificity, false_positive_rate}``.
        """
        y_true, y_pred = validate_targets(y_true, y_pred)
        labels = self._labels(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        total = cm.sum()
        metrics: Dict[Any, Dict[str, float]] = {}
        for idx, label in enumerate(labels):
            tp = cm[idx, idx]
            fp = cm[:, idx].sum() - tp
            fn = cm[idx, :].sum() - tp
            tn = total - tp - fp - fn
            metrics[to_builtin(label)] = {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1_score": float(f1[idx]),
                "support": int(support[idx]),
                "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            }
        return metrics

    def prepare_confusion_matrix_plot(
        self, y_true: Any, y_pred: Any, normalize: bool = False, title: Optional[str] = None
    ) -> Dict[str, Any]:
        """Plot-ready matrix, labels and axis titles."""
        analysis = self.analyze_confusion_matrix(y_true, y_pred)
        return {
            "matrix": analysis["normalized_cm"] if normalize else analysis["confusion_matrix"],
            "labels": analysis["labels"],
            "title": title or ("Normalized Confusion Matrix" if normalize else "Confusion Matrix"),
            "xlabel": "Predicted label",
            "ylabel": "True label",
            "normalized": normalize,
        }


# --------------------------------------------------------------------------- #
# ROC
# --------------------------------------------------------------------------- #
class ROCAnalyzer(LoggerMixin):
    """Receiver-operating-characteristic analysis for binary and multi-class scores."""

    def generate_roc_curve(
        self, y_true: Any, y_proba: Any, pos_label: Optional[Any] = None
    ) -> Dict[str, Any]:
        """ROC curve and AUC for binary scores.

        Args:
            y_true: Binary labels.
            y_proba: Positive-class scores (or an ``(n, 2)`` probability matrix).
            pos_label: Label treated as positive (scikit-learn default when ``None``).

        Returns:
            Dict with ``fpr``, ``tpr``, ``thresholds`` (first entry is ``inf``,
            following scikit-learn) and ``auc``.

        Raises:
            ValueError: For inconsistent shapes or non-binary labels.
        """
        y_true, y_proba = validate_binary_probabilities(y_true, y_proba)
        fpr, tpr, thresholds = roc_curve(y_true, y_proba, pos_label=pos_label)
        return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": float(auc(fpr, tpr))}

    def find_optimal_threshold(self, y_true: Any, y_proba: Any, method: str = "youden") -> float:
        """Operating point maximising Youden's J (``tpr - fpr``) or closest to ``(0, 1)``.

        Args:
            y_true: Binary labels.
            y_proba: Positive-class scores.
            method: ``"youden"`` (Youden, 1950) or ``"closest"`` (minimum Euclidean
                distance to the top-left corner).

        Returns:
            The score threshold (a finite value of ``y_proba``).
        """
        roc = self.generate_roc_curve(y_true, y_proba)
        finite = np.isfinite(roc["thresholds"])
        fpr, tpr, thr = roc["fpr"][finite], roc["tpr"][finite], roc["thresholds"][finite]
        if method == "youden":
            idx = int(np.argmax(tpr - fpr))
        elif method == "closest":
            idx = int(np.argmin(np.sqrt(fpr**2 + (1 - tpr) ** 2)))
        else:
            raise ValueError(f"Unknown method: {method!r}")
        return float(thr[idx])

    def prepare_roc_plot(self, y_true: Any, y_proba: Any, title: Optional[str] = None) -> Dict[str, Any]:
        """Plot-ready ROC data with axis labels and the chance diagonal."""
        roc = self.generate_roc_curve(y_true, y_proba)
        roc.update(
            {
                "title": title or f"ROC Curve (AUC = {roc['auc']:.3f})",
                "xlabel": "False positive rate",
                "ylabel": "True positive rate",
                "diagonal": np.array([0.0, 1.0]),
            }
        )
        return roc

    def analyze_multiclass_roc(
        self, y_true: Any, y_proba: Any, labels: Optional[Sequence[Any]] = None
    ) -> Dict[str, Any]:
        """One-vs-rest ROC curves per class plus macro/micro AUC.

        Args:
            y_true: Labels.
            y_proba: ``(n_samples, n_classes)`` probability matrix.
            labels: Class order of the columns (defaults to sorted unique labels).

        Returns:
            Dict ``"class_<label>" -> {fpr, tpr, thresholds, auc}`` plus
            ``"macro_average"`` and ``"micro_average"`` entries holding ``auc``.
        """
        y_true = ensure_numpy_array(y_true)
        y_proba = ensure_numpy_array(y_proba)
        if y_proba.ndim != 2:
            raise ValueError("y_proba must be a 2-D probability matrix")
        labels_arr = np.asarray(labels) if labels is not None else np.unique(y_true)
        if len(labels_arr) != y_proba.shape[1]:
            raise ValueError("Number of labels must match the number of probability columns")
        result: Dict[str, Any] = {}
        aucs = []
        for j, label in enumerate(labels_arr):
            binary = (y_true == label).astype(int)
            if binary.min() == binary.max():
                continue
            curve = self.generate_roc_curve(binary, y_proba[:, j])
            result[f"class_{to_builtin(label)}"] = curve
            aucs.append(curve["auc"])
        onehot = (y_true[:, None] == labels_arr[None, :]).astype(int)
        result["macro_average"] = {"auc": float(np.mean(aucs)) if aucs else float("nan")}
        result["micro_average"] = {"auc": float(roc_auc_score(onehot.ravel(), y_proba.ravel()))}
        return result


# --------------------------------------------------------------------------- #
# Precision-recall
# --------------------------------------------------------------------------- #
class PrecisionRecallAnalyzer(LoggerMixin):
    """Precision-recall analysis for binary and multi-class scores."""

    def generate_precision_recall_curve(
        self, y_true: Any, y_proba: Any, pos_label: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Precision-recall curve and average precision.

        Args:
            y_true: Binary labels.
            y_proba: Positive-class scores.
            pos_label: Positive label (scikit-learn default when ``None``).

        Returns:
            Dict with ``precision``, ``recall`` (length ``n + 1``), ``thresholds``
            (length ``n``), ``average_precision`` and ``baseline`` (positive rate).
        """
        y_true, y_proba = validate_binary_probabilities(y_true, y_proba)
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba, pos_label=pos_label)
        ap = float(
            average_precision_score(y_true, y_proba, pos_label=pos_label if pos_label is not None else 1)
        )
        positive = pos_label if pos_label is not None else 1
        return {
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds,
            "average_precision": ap,
            "baseline": float(np.mean(y_true == positive)),
        }

    def find_f1_optimal_threshold(self, y_true: Any, y_proba: Any) -> float:
        """Threshold maximising F1 along the precision-recall curve.

        Args:
            y_true: Binary labels.
            y_proba: Positive-class scores.

        Returns:
            The score threshold with the highest F1.
        """
        pr = self.generate_precision_recall_curve(y_true, y_proba)
        p, r = pr["precision"][:-1], pr["recall"][:-1]
        denom = p + r
        f1 = np.divide(2 * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
        return float(pr["thresholds"][int(np.argmax(f1))])

    def analyze_multiclass_precision_recall(
        self, y_true: Any, y_proba: Any, labels: Optional[Sequence[Any]] = None
    ) -> Dict[str, Any]:
        """One-vs-rest precision-recall curves per class with macro/micro averages.

        Args:
            y_true: Labels.
            y_proba: ``(n_samples, n_classes)`` probability matrix.
            labels: Class order of the columns (defaults to sorted unique labels).

        Returns:
            Dict ``"class_<label>" -> {precision, recall, thresholds, average_precision}``
            plus ``"macro_average"`` and ``"micro_average"`` holding ``average_precision``.
        """
        y_true = ensure_numpy_array(y_true)
        y_proba = ensure_numpy_array(y_proba)
        if y_proba.ndim != 2:
            raise ValueError("y_proba must be a 2-D probability matrix")
        labels_arr = np.asarray(labels) if labels is not None else np.unique(y_true)
        if len(labels_arr) != y_proba.shape[1]:
            raise ValueError("Number of labels must match the number of probability columns")
        result: Dict[str, Any] = {}
        aps = []
        for j, label in enumerate(labels_arr):
            binary = (y_true == label).astype(int)
            if binary.max() == 0:
                continue
            curve = self.generate_precision_recall_curve(binary, y_proba[:, j])
            result[f"class_{to_builtin(label)}"] = curve
            aps.append(curve["average_precision"])
        onehot = (y_true[:, None] == labels_arr[None, :]).astype(int)
        result["macro_average"] = {"average_precision": float(np.mean(aps)) if aps else float("nan")}
        result["micro_average"] = {
            "average_precision": float(average_precision_score(onehot.ravel(), y_proba.ravel()))
        }
        return result

    def prepare_precision_recall_plot(
        self, y_true: Any, y_proba: Any, title: Optional[str] = None
    ) -> Dict[str, Any]:
        """Plot-ready precision-recall data with axis labels."""
        pr = self.generate_precision_recall_curve(y_true, y_proba)
        pr.update(
            {
                "title": title or f"Precision-Recall Curve (AP = {pr['average_precision']:.3f})",
                "xlabel": "Recall",
                "ylabel": "Precision",
            }
        )
        return pr


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
class CalibrationAnalyzer(LoggerMixin):
    """Probability-calibration diagnostics for binary classifiers.

    Args:
        n_bins: Number of probability bins.
        strategy: ``"uniform"`` (equal-width) or ``"quantile"`` bins for the
            scikit-learn calibration curve; ECE/MCE always use uniform bins.
    """

    def __init__(self, n_bins: int = 10, strategy: str = "uniform"):
        if n_bins < 1:
            raise ValueError("n_bins must be positive")
        self.n_bins = n_bins
        self.strategy = strategy

    @staticmethod
    def _bin_index(y_proba: np.ndarray, n_bins: int) -> np.ndarray:
        """Uniform bin ``[i/n, (i+1)/n)`` for each probability; ``1.0`` falls in the last bin."""
        return np.clip((y_proba * n_bins).astype(int), 0, n_bins - 1)

    def _bin_stats(self, y_true: np.ndarray, y_proba: np.ndarray, n_bins: int) -> Dict[str, np.ndarray]:
        idx = self._bin_index(y_proba, n_bins)
        counts = np.bincount(idx, minlength=n_bins)
        sum_true = np.bincount(idx, weights=y_true, minlength=n_bins)
        sum_conf = np.bincount(idx, weights=y_proba, minlength=n_bins)
        nonempty = counts > 0
        accuracy = np.divide(sum_true, counts, out=np.zeros(n_bins), where=nonempty)
        confidence = np.divide(sum_conf, counts, out=np.zeros(n_bins), where=nonempty)
        return {"counts": counts, "accuracy": accuracy, "confidence": confidence, "nonempty": nonempty}

    def generate_calibration_curve(
        self, y_true: Any, y_proba: Any, n_bins: Optional[int] = None, strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """Reliability curve points and Brier score.

        Args:
            y_true: Binary labels (0/1).
            y_proba: Positive-class probabilities in ``[0, 1]``.
            n_bins: Overrides the instance default.
            strategy: Overrides the instance default.

        Returns:
            Dict with ``mean_predicted_value``, ``fraction_of_positives`` (non-empty
            bins only), ``brier_score``, ``n_bins`` and ``strategy``.
        """
        y_true, y_proba = self._validate(y_true, y_proba)
        n_bins = n_bins or self.n_bins
        strategy = strategy or self.strategy
        frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy=strategy)
        return {
            "mean_predicted_value": mean_pred,
            "fraction_of_positives": frac_pos,
            "brier_score": float(brier_score_loss(y_true, y_proba)),
            "n_bins": n_bins,
            "strategy": strategy,
        }

    def generate_reliability_diagram(
        self, y_true: Any, y_proba: Any, n_bins: Optional[int] = None
    ) -> Dict[str, Any]:
        """Uniform-bin reliability-diagram data.

        Args:
            y_true: Binary labels (0/1).
            y_proba: Positive-class probabilities.
            n_bins: Overrides the instance default.

        Returns:
            Dict with ``bin_boundaries`` (``n_bins + 1``), ``bin_lowers``/``bin_uppers``
            (``n_bins``) and, for non-empty bins only, ``bin_centers``, ``y``
            (observed positive fraction), ``confidence`` (mean predicted probability),
            ``counts`` and ``gap`` (``confidence - y``); plus ``ece``.
        """
        y_true, y_proba = self._validate(y_true, y_proba)
        n_bins = n_bins or self.n_bins
        boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        stats_ = self._bin_stats(y_true, y_proba, n_bins)
        keep = stats_["nonempty"]
        centers = (boundaries[:-1] + boundaries[1:]) / 2
        weights = stats_["counts"] / len(y_true)
        ece = float(np.sum(weights * np.abs(stats_["confidence"] - stats_["accuracy"])))
        return {
            "bin_boundaries": boundaries,
            "bin_lowers": boundaries[:-1],
            "bin_uppers": boundaries[1:],
            "bin_centers": centers[keep],
            "y": stats_["accuracy"][keep],
            "confidence": stats_["confidence"][keep],
            "counts": stats_["counts"][keep],
            "gap": (stats_["confidence"] - stats_["accuracy"])[keep],
            "ece": ece,
            "n_bins": n_bins,
        }

    def calculate_calibration_metrics(
        self, y_true: Any, y_proba: Any, n_bins: Optional[int] = None
    ) -> Dict[str, float]:
        """Brier score, expected/maximum calibration error and log loss.

        ECE is the count-weighted mean absolute gap between confidence and
        accuracy over uniform bins; MCE is the largest gap (Naeini, Cooper &
        Hauskrecht, 2015; Guo et al., 2017).

        Args:
            y_true: Binary labels (0/1).
            y_proba: Positive-class probabilities.
            n_bins: Overrides the instance default.

        Returns:
            Dict with ``brier_score`` (Brier, 1950), ``ece``, ``mce``, ``log_loss`` and ``n_bins``.

        References:
            Naeini, M. P., Cooper, G. F. & Hauskrecht, M. (2015). Obtaining well calibrated
            probabilities using Bayesian binning. *AAAI*, 2901-2907.
            Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). On calibration of
            modern neural networks. *ICML*, 1321-1330.
        """
        y_true, y_proba = self._validate(y_true, y_proba)
        n_bins = n_bins or self.n_bins
        stats_ = self._bin_stats(y_true, y_proba, n_bins)
        gaps = np.abs(stats_["confidence"] - stats_["accuracy"])[stats_["nonempty"]]
        weights = (stats_["counts"] / len(y_true))[stats_["nonempty"]]
        return {
            "brier_score": float(brier_score_loss(y_true, y_proba)),
            "ece": float(np.sum(weights * gaps)),
            "mce": float(np.max(gaps)) if len(gaps) else 0.0,
            "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
            "n_bins": n_bins,
        }

    def compare_model_calibration(
        self, models: Dict[str, BaseEstimator], X: Any, y: Any, n_bins: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Calibration metrics and curves for several fitted binary classifiers.

        Models without ``predict_proba`` are skipped with a warning.

        Args:
            models: Mapping ``name -> fitted classifier``.
            X: Feature matrix.
            y: Binary labels.
            n_bins: Overrides the instance default.

        Returns:
            Mapping ``name -> {brier_score, ece, mce, log_loss, calibration_curve}``.
        """
        comparison: Dict[str, Dict[str, Any]] = {}
        for name, model in models.items():
            if not hasattr(model, "predict_proba"):
                self.logger.warning("Skipping %s: no predict_proba", name)
                continue
            proba = ensure_numpy_array(model.predict_proba(X))
            metrics = self.calculate_calibration_metrics(y, proba, n_bins)
            curve = self.generate_calibration_curve(y, proba, n_bins)
            metrics["calibration_curve"] = {
                "mean_predicted_value": curve["mean_predicted_value"],
                "fraction_of_positives": curve["fraction_of_positives"],
            }
            comparison[name] = metrics
        return comparison

    @staticmethod
    def _validate(y_true: Any, y_proba: Any) -> Tuple[np.ndarray, np.ndarray]:
        y_true, y_proba = validate_binary_probabilities(y_true, y_proba)
        if np.any((y_proba < 0) | (y_proba > 1)):
            raise ValueError("y_proba must contain probabilities in [0, 1]")
        classes = np.unique(y_true)
        if not np.all(np.isin(classes, [0, 1])):
            raise ValueError("y_true must be encoded as 0/1 for calibration analysis")
        return y_true.astype(float), y_proba.astype(float)


__all__ = [
    "CalibrationAnalyzer",
    "ConfusionMatrixAnalyzer",
    "FeatureImportanceAnalyzer",
    "PrecisionRecallAnalyzer",
    "ROCAnalyzer",
    "ResidualAnalyzer",
]
