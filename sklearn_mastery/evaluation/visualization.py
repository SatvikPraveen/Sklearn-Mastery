"""Static (matplotlib) visualization toolkit for data, model and performance analysis.

The module exposes focused visualizer classes plus :class:`ModelVisualizationSuite`,
a convenience facade kept for backward compatibility:

* :class:`DataVisualizer` - dataset inspection (class balance, distributions,
  correlations, missing values).
* :class:`ModelVisualizer` - fitted-model views (decision boundaries,
  regression fits, importances, residuals).
* :class:`PerformanceVisualizer` - evaluation curves (confusion matrix, ROC,
  precision-recall, learning/validation/calibration curves).
* :class:`FeatureVisualizer` - feature ranking and per-class distributions.
* :class:`ComparisonVisualizer` - multi-model comparisons (bars, radar,
  heatmap, score distributions).
* :class:`InteractiveVisualizer` - re-exported from
  :mod:`sklearn_mastery.evaluation.interactive` (Plotly with matplotlib fallback).

Every plotting method draws on a caller-supplied ``ax`` when given, otherwise
creates its own figure. Methods return the ``Axes`` (single-panel) or the
``Figure`` (multi-panel) and never call ``plt.show()``, so they work headless.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from sklearn_mastery.config.logging_config import LoggerMixin
from sklearn_mastery.config.settings import settings
from sklearn_mastery.evaluation._viz_common import (
    CATEGORICAL_PALETTE,
    DIVERGING_CMAP,
    HAS_IPYWIDGETS,
    HAS_PLOTLY,
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
    REFERENCE_COLOR,
    SEQUENTIAL_CMAP,
    as_dataframe,
    default_feature_names,
    finalize_figure,
    is_discrete_target,
    resolve_axes,
    select_features,
    series_colors,
    style_axes,
    subplot_grid,
    to_1d_array,
    to_2d_array,
    trapezoid,
)
from sklearn_mastery.evaluation.interactive import InteractiveVisualizer

__all__ = [
    "HAS_IPYWIDGETS",
    "HAS_PLOTLY",
    "ComparisonVisualizer",
    "DataVisualizer",
    "FeatureVisualizer",
    "InteractiveVisualizer",
    "ModelVisualizationSuite",
    "ModelVisualizer",
    "PerformanceVisualizer",
]


class _BaseVisualizer(LoggerMixin):
    """Common state for the static visualizers.

    Args:
        figure_size: Default size for figures created by the visualizer.
        palette: Categorical colour sequence (fixed order, never re-cycled).
    """

    def __init__(
        self, figure_size: Optional[Tuple[float, float]] = None, palette: Optional[Sequence[str]] = None
    ) -> None:
        self.figure_size: Tuple[float, float] = tuple(figure_size or settings.FIGURE_SIZE)  # type: ignore[assignment]
        self.palette: List[str] = list(palette or CATEGORICAL_PALETTE)

    def _colors(self, n: int) -> List[str]:
        return series_colors(n, self.palette)

    def _axes(
        self, ax: Optional[Axes], figsize: Optional[Tuple[float, float]] = None, **kw: Any
    ) -> Tuple[Figure, Axes, bool]:
        return resolve_axes(ax, figsize or self.figure_size, **kw)


# =============================================================================== data
class DataVisualizer(_BaseVisualizer):
    """Plots for inspecting a dataset before modelling."""

    def plot_class_distribution(
        self,
        y: Any,
        ax: Optional[Axes] = None,
        class_names: Optional[Sequence[str]] = None,
        normalize: bool = False,
        title: str = "Class distribution",
        save_path: Optional[str] = None,
    ) -> Axes:
        """Bar chart of class frequencies.

        Args:
            y: Class labels.
            ax: Axes to draw on; a new figure is created when ``None``.
            class_names: Display names in the order of ``np.unique(y)``.
            normalize: Show proportions instead of counts.
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The axes containing the bars.

        Raises:
            ValueError: If ``y`` is empty.
        """
        y = to_1d_array(y, "y")
        if y.size == 0:
            raise ValueError("y is empty; nothing to plot")
        classes, counts = np.unique(y, return_counts=True)
        values = counts / counts.sum() if normalize else counts
        labels = [str(c) for c in (class_names if class_names is not None else classes)]
        if len(labels) != len(classes):
            raise ValueError(f"Expected {len(classes)} class names, got {len(labels)}")
        fig, ax, created = self._axes(ax)
        bars = ax.bar(range(len(classes)), values, color=self._colors(len(classes)), width=0.7)
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.1%}" if normalize else f"{int(value)}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 2),
                textcoords="offset points",
            )
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(labels)
        style_axes(ax, title, "Class", "Proportion" if normalize else "Count")
        finalize_figure(fig, created, save_path)
        return ax

    def plot_feature_distributions(
        self,
        X: Any,
        feature_names: Optional[Sequence[str]] = None,
        bins: int = 30,
        max_cols: int = 3,
        kde: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
    ) -> Figure:
        """Histogram grid with exactly one panel per feature.

        Passing fewer ``feature_names`` than columns plots only the leading
        ``len(feature_names)`` columns.

        Args:
            X: Feature matrix.
            feature_names: Column names (defaults to ``feature_i``).
            bins: Histogram bins.
            max_cols: Maximum panels per row.
            kde: Overlay a Gaussian kernel-density estimate.
            figsize: Explicit figure size.
            save_path: Optional output path.

        Returns:
            Figure whose ``axes`` has one entry per feature.
        """
        X, names = select_features(X, feature_names)
        fig, axes = subplot_grid(len(names), max_cols=max_cols, figsize=figsize)
        color = self.palette[0]
        for ax, name, col in zip(axes, names, X.T):
            col = col[np.isfinite(col)]
            ax.hist(col, bins=bins, color=color, alpha=0.85, density=kde)
            if kde and col.size > 1 and np.std(col) > 0:
                grid = np.linspace(col.min(), col.max(), 200)
                ax.plot(grid, stats.gaussian_kde(col)(grid), color=REFERENCE_COLOR, linewidth=1.5)
            style_axes(ax, name, None, "Density" if kde else "Count")
        finalize_figure(fig, True, save_path)
        return fig

    def plot_correlation_matrix(
        self,
        X: Any,
        feature_names: Optional[Sequence[str]] = None,
        ax: Optional[Axes] = None,
        method: str = "pearson",
        annot: Optional[bool] = None,
        title: str = "Feature correlation",
        save_path: Optional[str] = None,
    ) -> Axes:
        """Heatmap of pairwise feature correlations on a diverging scale.

        Args:
            X: Feature matrix or DataFrame.
            feature_names: Column names.
            ax: Axes to draw on.
            method: ``"pearson"``, ``"spearman"`` or ``"kendall"``.
            annot: Write coefficients in cells (default: only for <= 12 features).
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The axes containing the heatmap.
        """
        names = default_feature_names(to_2d_array(X).shape[1], feature_names, X)
        df = pd.DataFrame(to_2d_array(X), columns=names)
        corr = df.corr(method=method)
        if annot is None:
            annot = len(names) <= 12
        fig, ax, created = self._axes(ax)
        sns.heatmap(
            corr,
            vmin=-1,
            vmax=1,
            center=0,
            cmap=DIVERGING_CMAP,
            annot=annot,
            fmt=".2f",
            square=True,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"shrink": 0.8, "label": f"{method} r"},
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        finalize_figure(fig, created, save_path)
        return ax

    def plot_scatter_matrix(
        self,
        X: Any,
        y: Any = None,
        feature_names: Optional[Sequence[str]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        alpha: float = 0.6,
        bins: int = 20,
        save_path: Optional[str] = None,
    ) -> Figure:
        """Pairwise scatter matrix (``n_features x n_features`` panels).

        Diagonal panels hold histograms; off-diagonal panels scatter one
        feature against another, coloured by class when ``y`` is discrete.

        Args:
            X: Feature matrix.
            y: Optional target used for colouring.
            feature_names: Column names.
            figsize: Explicit figure size.
            alpha: Marker transparency.
            bins: Histogram bins on the diagonal.
            save_path: Optional output path.

        Returns:
            Figure with exactly ``n_features ** 2`` axes.
        """
        X = to_2d_array(X)
        n = X.shape[1]
        names = default_feature_names(n, feature_names)
        yv = None if y is None else to_1d_array(y, "y")
        groups: List[Tuple[str, np.ndarray, str]]
        if yv is not None and is_discrete_target(yv):
            classes = np.unique(yv)
            groups = [(f"class {c}", yv == c, col) for c, col in zip(classes, self._colors(len(classes)))]
        else:
            groups = [("samples", np.ones(X.shape[0], dtype=bool), self.palette[0])]
        fig, axes = plt.subplots(n, n, figsize=figsize or (2.6 * n, 2.6 * n), squeeze=False)
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                for label, mask, color in groups:
                    if i == j:
                        ax.hist(X[mask, i], bins=bins, color=color, alpha=0.6, label=label)
                    else:
                        ax.scatter(X[mask, j], X[mask, i], s=10, color=color, alpha=alpha, label=label)
                if i == n - 1:
                    ax.set_xlabel(names[j])
                else:
                    ax.set_xticklabels([])
                if j == 0:
                    ax.set_ylabel(names[i])
                else:
                    ax.set_yticklabels([])
                ax.tick_params(labelsize=7)
        if len(groups) > 1:
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper right", fontsize=8)
        finalize_figure(fig, True, save_path)
        return fig

    def plot_target_distribution(
        self,
        y: Any,
        ax: Optional[Axes] = None,
        bins: int = 30,
        kde: bool = True,
        title: str = "Target distribution",
        target_name: str = "target",
        save_path: Optional[str] = None,
    ) -> Axes:
        """Histogram (with optional KDE) of a continuous target.

        Args:
            y: Target values.
            ax: Axes to draw on.
            bins: Histogram bins.
            kde: Overlay a kernel-density estimate.
            title: Plot title.
            target_name: X axis label.
            save_path: Optional output path.

        Returns:
            The axes containing the histogram.

        Raises:
            ValueError: If ``y`` is empty.
        """
        y = to_1d_array(y, "y").astype(float)
        y = y[np.isfinite(y)]
        if y.size == 0:
            raise ValueError("y is empty; nothing to plot")
        fig, ax, created = self._axes(ax)
        ax.hist(y, bins=bins, color=self.palette[0], alpha=0.85, density=kde)
        if kde and y.size > 1 and np.std(y) > 0:
            grid = np.linspace(y.min(), y.max(), 200)
            ax.plot(grid, stats.gaussian_kde(y)(grid), color=REFERENCE_COLOR, linewidth=1.5, label="KDE")
            ax.legend()
        ax.axvline(np.mean(y), color=REFERENCE_COLOR, linestyle=":", linewidth=1)
        style_axes(ax, title, target_name, "Density" if kde else "Count")
        finalize_figure(fig, created, save_path)
        return ax

    def plot_feature_target_relationship(
        self,
        x: Any,
        y: Any,
        feature_name: str = "feature",
        target_name: str = "target",
        ax: Optional[Axes] = None,
        alpha: float = 0.6,
        fit_line: bool = True,
        save_path: Optional[str] = None,
    ) -> Axes:
        """Scatter a single feature against the target.

        Discrete targets are colour-coded by class; continuous targets get an
        optional least-squares fit line.

        Args:
            x: Feature values.
            y: Target values.
            feature_name: X axis label.
            target_name: Y axis label.
            ax: Axes to draw on.
            alpha: Marker transparency.
            fit_line: Draw a linear fit for continuous targets.
            save_path: Optional output path.

        Returns:
            The axes containing the scatter.

        Raises:
            ValueError: If ``x`` and ``y`` have different lengths.
        """
        x = to_1d_array(x, "x")
        y = to_1d_array(y, "y")
        if x.shape != y.shape:
            raise ValueError("x and y must have the same length")
        fig, ax, created = self._axes(ax)
        if is_discrete_target(y):
            classes = np.unique(y)
            rng = np.random.default_rng(settings.RANDOM_SEED)
            for cls, color in zip(classes, self._colors(len(classes))):
                mask = y == cls
                jitter = rng.uniform(-0.15, 0.15, mask.sum())
                ax.scatter(
                    x[mask],
                    np.full(mask.sum(), float(np.where(classes == cls)[0][0])) + jitter,
                    s=18,
                    color=color,
                    alpha=alpha,
                    label=f"class {cls}",
                )
            ax.set_yticks(range(len(classes)))
            ax.set_yticklabels([str(c) for c in classes])
            ax.legend(fontsize=8)
        else:
            ax.scatter(x, y, s=18, color=self.palette[0], alpha=alpha)
            if fit_line and x.size > 1 and np.std(x) > 0:
                slope, intercept, r_value, _, _ = stats.linregress(x.astype(float), y.astype(float))
                grid = np.linspace(x.min(), x.max(), 100)
                ax.plot(
                    grid,
                    slope * grid + intercept,
                    color=REFERENCE_COLOR,
                    linewidth=1.5,
                    label=f"fit (r = {r_value:.2f})",
                )
                ax.legend(fontsize=8)
        style_axes(ax, f"{feature_name} vs {target_name}", feature_name, target_name)
        finalize_figure(fig, created, save_path)
        return ax

    def plot_missing_values_heatmap(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        ax: Optional[Axes] = None,
        feature_names: Optional[Sequence[str]] = None,
        title: str = "Missing values",
        save_path: Optional[str] = None,
    ) -> Axes:
        """Binary heatmap of missing entries (rows x columns).

        Column labels carry the percentage missing.

        Args:
            data: DataFrame or array.
            ax: Axes to draw on.
            feature_names: Column names for array input.
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The axes containing the heatmap.
        """
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            arr = to_2d_array(data, "data")
            df = pd.DataFrame(arr, columns=default_feature_names(arr.shape[1], feature_names))
        mask = df.isna()
        pct = mask.mean(axis=0) * 100
        fig, ax, created = self._axes(ax)
        sns.heatmap(
            mask, cmap=ListedColormap(["#f0f0ee", self.palette[1]]), cbar=False, ax=ax, yticklabels=False
        )
        ax.set_xticklabels([f"{c}\n{p:.0f}%" for c, p in zip(df.columns, pct)], rotation=0, fontsize=8)
        ax.set_title(f"{title} ({mask.values.mean():.1%} of cells)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        finalize_figure(fig, created, save_path)
        return ax


# ============================================================================== models
class ModelVisualizer(_BaseVisualizer):
    """Plots describing a fitted estimator."""

    def plot_decision_boundary(
        self,
        model: Any,
        X: Any,
        y: Any,
        ax: Optional[Axes] = None,
        feature_names: Optional[Sequence[str]] = None,
        resolution: int = 200,
        alpha: float = 0.3,
        title: str = "Decision boundary",
        save_path: Optional[str] = None,
    ) -> Axes:
        """Filled-contour decision regions of a 2-D classifier with the data overlaid.

        Args:
            model: Fitted classifier exposing ``predict``.
            X: Feature matrix with exactly two columns.
            y: Class labels.
            ax: Axes to draw on.
            feature_names: Axis labels.
            resolution: Grid points per axis.
            alpha: Transparency of the decision regions.
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The axes containing the plot.

        Raises:
            ValueError: If ``X`` does not have exactly two features.
            AttributeError: If ``model`` has no ``predict`` method.
        """
        X = to_2d_array(X)
        y = to_1d_array(y, "y")
        if X.shape[1] != 2:
            raise ValueError(f"Decision boundary plots require exactly 2 features, got {X.shape[1]}")
        if not hasattr(model, "predict"):
            raise AttributeError(f"{type(model).__name__} has no 'predict' method")
        names = default_feature_names(2, feature_names)
        classes = np.unique(y)
        colors = self._colors(len(classes))
        pad_x = 0.1 * np.ptp(X[:, 0]) or 1.0
        pad_y = 0.1 * np.ptp(X[:, 1]) or 1.0
        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min() - pad_x, X[:, 0].max() + pad_x, resolution),
            np.linspace(X[:, 1].min() - pad_y, X[:, 1].max() + pad_y, resolution),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        pred = np.asarray(model.predict(grid))
        class_index = {c: i for i, c in enumerate(classes)}
        Z = np.array([class_index.get(p, -1) for p in pred], dtype=float).reshape(xx.shape)
        fig, ax, created = self._axes(ax)
        levels = np.arange(-0.5, len(classes) + 0.5, 1.0)
        ax.contourf(xx, yy, Z, levels=levels, cmap=ListedColormap(colors), alpha=alpha)
        ax.contour(xx, yy, Z, levels=levels[1:-1], colors=REFERENCE_COLOR, linewidths=0.8)
        for cls, color in zip(classes, colors):
            mask = y == cls
            ax.scatter(
                X[mask, 0],
                X[mask, 1],
                s=22,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                label=f"class {cls}",
            )
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.legend(fontsize=8)
        style_axes(ax, title, names[0], names[1])
        ax.grid(False)
        finalize_figure(fig, created, save_path)
        return ax

    def plot_regression_line(
        self,
        model: Any,
        X: Any,
        y: Any,
        ax: Optional[Axes] = None,
        feature_index: int = 0,
        feature_name: Optional[str] = None,
        target_name: str = "target",
        title: str = "Regression fit",
        save_path: Optional[str] = None,
    ) -> Axes:
        """Scatter of the data with the model's fitted curve along one feature.

        For multi-feature inputs the remaining features are held at their
        mean while ``feature_index`` sweeps its observed range.

        Args:
            model: Fitted regressor exposing ``predict``.
            X: Feature matrix.
            y: Target values.
            ax: Axes to draw on.
            feature_index: Column swept along the x axis.
            feature_name: X axis label.
            target_name: Y axis label.
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The axes containing the plot.

        Raises:
            AttributeError: If ``model`` has no ``predict`` method.
            IndexError: If ``feature_index`` is out of range.
        """
        X = to_2d_array(X)
        y = to_1d_array(y, "y")
        if not hasattr(model, "predict"):
            raise AttributeError(f"{type(model).__name__} has no 'predict' method")
        if not 0 <= feature_index < X.shape[1]:
            raise IndexError(f"feature_index {feature_index} out of range for {X.shape[1]} features")
        name = feature_name or default_feature_names(X.shape[1])[feature_index]
        grid = np.tile(X.mean(axis=0), (200, 1))
        grid[:, feature_index] = np.linspace(X[:, feature_index].min(), X[:, feature_index].max(), 200)
        fitted = np.asarray(model.predict(grid))
        fig, ax, created = self._axes(ax)
        ax.scatter(X[:, feature_index], y, s=18, color=self.palette[0], alpha=0.6, label="observed")
        ax.plot(grid[:, feature_index], fitted, color=self.palette[1], linewidth=2, label="model")
        ax.legend(fontsize=8)
        style_axes(ax, title, name, target_name)
        finalize_figure(fig, created, save_path)
        return ax

    def plot_feature_importance(
        self,
        importances: Any,
        feature_names: Optional[Sequence[str]] = None,
        ax: Optional[Axes] = None,
        std: Any = None,
        top_n: Optional[int] = None,
        title: str = "Feature importance",
        save_path: Optional[str] = None,
    ) -> Axes:
        """Horizontal bar chart of importances, largest at the top.

        Args:
            importances: Importance per feature.
            feature_names: Feature labels.
            ax: Axes to draw on.
            std: Optional per-feature standard deviation drawn as error bars.
            top_n: Keep only the ``top_n`` largest.
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The axes containing the bars.

        Raises:
            ValueError: If ``importances`` is empty or lengths mismatch.
        """
        imp = to_1d_array(importances, "importances").astype(float)
        if imp.size == 0:
            raise ValueError("importances is empty")
        names = default_feature_names(imp.size, feature_names)
        err = None if std is None else to_1d_array(std, "std").astype(float)
        if err is not None and err.shape != imp.shape:
            raise ValueError("std must have the same length as importances")
        order = np.argsort(imp)[::-1]
        if top_n is not None:
            order = order[:top_n]
        order = order[::-1]
        fig, ax, created = self._axes(ax)
        ax.barh(
            [names[i] for i in order],
            imp[order],
            color=self.palette[0],
            xerr=None if err is None else err[order],
            ecolor=REFERENCE_COLOR,
            capsize=3,
        )
        style_axes(ax, title, "Importance", None)
        finalize_figure(fig, created, save_path)
        return ax

    def plot_residuals(
        self,
        y_pred: Any,
        residuals: Any,
        ax: Optional[Axes] = None,
        title: str = "Residuals vs fitted",
        save_path: Optional[str] = None,
    ) -> Axes:
        """Residuals against fitted values with a zero reference and a smoothed trend.

        Args:
            y_pred: Fitted values.
            residuals: ``y_true - y_pred``.
            ax: Axes to draw on.
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The axes containing the scatter.

        Raises:
            ValueError: If lengths mismatch.
        """
        y_pred = to_1d_array(y_pred, "y_pred").astype(float)
        residuals = to_1d_array(residuals, "residuals").astype(float)
        if y_pred.shape != residuals.shape:
            raise ValueError("y_pred and residuals must have the same length")
        fig, ax, created = self._axes(ax)
        ax.scatter(y_pred, residuals, s=18, color=self.palette[0], alpha=0.6)
        ax.axhline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=1)
        if y_pred.size >= 10:
            order = np.argsort(y_pred)
            window = max(5, y_pred.size // 10)
            trend = pd.Series(residuals[order]).rolling(window, center=True, min_periods=1).mean().to_numpy()
            ax.plot(y_pred[order], trend, color=self.palette[1], linewidth=1.5, label="rolling mean")
            ax.legend(fontsize=8)
        style_axes(ax, title, "Fitted value", "Residual")
        finalize_figure(fig, created, save_path)
        return ax

    def plot_prediction_vs_actual(
        self,
        y_true: Any,
        y_pred: Any,
        ax: Optional[Axes] = None,
        title: str = "Predicted vs actual",
        save_path: Optional[str] = None,
    ) -> Axes:
        """Scatter of predictions against truth with the identity line and R².

        Args:
            y_true: True targets.
            y_pred: Predicted targets.
            ax: Axes to draw on.
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The axes containing the plot.

        Raises:
            ValueError: If lengths mismatch.
        """
        y_true = to_1d_array(y_true, "y_true").astype(float)
        y_pred = to_1d_array(y_pred, "y_pred").astype(float)
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same length")
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        fig, ax, created = self._axes(ax)
        ax.scatter(y_true, y_pred, s=18, color=self.palette[0], alpha=0.6)
        ax.plot([lo, hi], [lo, hi], color=REFERENCE_COLOR, linestyle="--", linewidth=1, label="identity")
        ax.text(0.03, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes, va="top", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
        style_axes(ax, title, "Actual", "Predicted")
        finalize_figure(fig, created, save_path)
        return ax

    def plot_residual_diagnostics(
        self,
        y_true: Any,
        y_pred: Any,
        figsize: Optional[Tuple[float, float]] = None,
        title: str = "Residual analysis",
        save_path: Optional[str] = None,
    ) -> Figure:
        """Four-panel residual diagnostics (vs fitted, Q-Q, histogram, pred-vs-actual).

        Args:
            y_true: True targets.
            y_pred: Predicted targets.
            figsize: Figure size.
            title: Figure title.
            save_path: Optional output path.

        Returns:
            The figure with four axes.
        """
        y_true = to_1d_array(y_true, "y_true").astype(float)
        y_pred = to_1d_array(y_pred, "y_pred").astype(float)
        residuals = y_true - y_pred
        fig, axes = plt.subplots(2, 2, figsize=figsize or (12, 9))
        self.plot_residuals(y_pred, residuals, ax=axes[0, 0])
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].get_lines()[0].set(color=self.palette[0], markersize=4)
        axes[0, 1].get_lines()[1].set(color=REFERENCE_COLOR)
        style_axes(axes[0, 1], "Normal Q-Q")
        axes[1, 0].hist(residuals, bins=30, color=self.palette[0], alpha=0.85)
        style_axes(axes[1, 0], "Residual distribution", "Residual", "Count")
        self.plot_prediction_vs_actual(y_true, y_pred, ax=axes[1, 1])
        fig.suptitle(title, fontsize=14, fontweight="bold")
        finalize_figure(fig, True, save_path)
        return fig


# ========================================================================= performance
class PerformanceVisualizer(_BaseVisualizer):
    """Evaluation curves and matrices for fitted models."""

    def plot_confusion_matrix(
        self,
        cm: Any,
        ax: Optional[Axes] = None,
        class_names: Optional[Sequence[str]] = None,
        normalize: bool = False,
        title: str = "Confusion matrix",
        save_path: Optional[str] = None,
    ) -> Axes:
        """Annotated confusion-matrix heatmap.

        Args:
            cm: Square confusion matrix (e.g. from ``sklearn.metrics.confusion_matrix``).
            ax: Axes to draw on.
            class_names: Class labels.
            normalize: Show row-normalised rates.
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The axes containing the heatmap.

        Raises:
            ValueError: If ``cm`` is not square.
        """
        cm = np.asarray(cm, dtype=float)
        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
            raise ValueError(f"cm must be square, got shape {cm.shape}")
        k = cm.shape[0]
        labels = [str(c) for c in (class_names if class_names is not None else range(k))]
        if len(labels) != k:
            raise ValueError(f"Expected {k} class names, got {len(labels)}")
        if normalize:
            cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-12, None)
        fig, ax, created = self._axes(ax)
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else ".0f",
            cmap=SEQUENTIAL_CMAP,
            xticklabels=labels,
            yticklabels=labels,
            square=True,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        finalize_figure(fig, created, save_path)
        return ax

    def plot_roc_curve(
        self,
        fpr: Any,
        tpr: Any,
        roc_auc: Optional[float] = None,
        ax: Optional[Axes] = None,
        label: str = "model",
        title: str = "ROC curve",
        save_path: Optional[str] = None,
    ) -> Axes:
        """ROC curve with the chance diagonal.

        Args:
            fpr: False-positive rates.
            tpr: True-positive rates.
            roc_auc: Area under the curve (computed when ``None``).
            ax: Axes to draw on (curves from several calls can share it).
            label: Legend label.
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The axes containing the curve.

        Raises:
            ValueError: If ``fpr`` and ``tpr`` differ in length.
        """
        fpr = to_1d_array(fpr, "fpr").astype(float)
        tpr = to_1d_array(tpr, "tpr").astype(float)
        if fpr.shape != tpr.shape:
            raise ValueError("fpr and tpr must have the same length")
        if roc_auc is None:
            roc_auc = trapezoid(tpr, fpr)
        fig, ax, created = self._axes(ax)
        n_existing = sum(1 for line in ax.get_lines() if line.get_label() != "chance")
        ax.plot(
            fpr,
            tpr,
            color=self._colors(n_existing + 1)[-1],
            linewidth=2,
            label=f"{label} (AUC = {roc_auc:.3f})",
        )
        if not any(line.get_label() == "chance" for line in ax.get_lines()):
            ax.plot([0, 1], [0, 1], color=REFERENCE_COLOR, linestyle=":", linewidth=1, label="chance")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right", fontsize=8)
        style_axes(ax, title, "False positive rate", "True positive rate")
        finalize_figure(fig, created, save_path)
        return ax

    def plot_precision_recall_curve(
        self,
        precision: Any,
        recall: Any,
        average_precision: Optional[float] = None,
        ax: Optional[Axes] = None,
        label: str = "model",
        baseline: Optional[float] = None,
        title: str = "Precision-recall curve",
        save_path: Optional[str] = None,
    ) -> Axes:
        """Precision-recall step curve.

        Args:
            precision: Precision values.
            recall: Recall values.
            average_precision: AP score shown in the legend.
            ax: Axes to draw on.
            label: Legend label.
            baseline: Positive-class prevalence drawn as a dotted reference.
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The axes containing the curve.

        Raises:
            ValueError: If ``precision`` and ``recall`` differ in length.
        """
        precision = to_1d_array(precision, "precision").astype(float)
        recall = to_1d_array(recall, "recall").astype(float)
        if precision.shape != recall.shape:
            raise ValueError("precision and recall must have the same length")
        fig, ax, created = self._axes(ax)
        n_existing = len(ax.get_lines())
        legend = label if average_precision is None else f"{label} (AP = {average_precision:.3f})"
        ax.step(
            recall, precision, where="post", color=self._colors(n_existing + 1)[-1], linewidth=2, label=legend
        )
        if baseline is not None:
            ax.axhline(
                baseline,
                color=REFERENCE_COLOR,
                linestyle=":",
                linewidth=1,
                label=f"prevalence = {baseline:.2f}",
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower left", fontsize=8)
        style_axes(ax, title, "Recall", "Precision")
        finalize_figure(fig, created, save_path)
        return ax

    @staticmethod
    def _mean_std(data: Dict[str, Any], *keys: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return ``(mean, std)`` from raw fold scores or precomputed keys."""
        for key in keys:
            if key in data:
                scores = np.asarray(data[key], dtype=float)
                if scores.ndim == 2:
                    return scores.mean(axis=1), scores.std(axis=1)
                std_key = key.replace("_scores", "_scores_std").replace("_mean", "_std")
                std = data.get(std_key) if key.endswith("_mean") else data.get(f"{key}_std")
                return scores, None if std is None else np.asarray(std, dtype=float)
        raise KeyError(f"None of {keys} found in learning-curve data")

    def plot_learning_curves(
        self,
        learning_data: Dict[str, Any],
        ax: Optional[Axes] = None,
        score_name: Optional[str] = None,
        title: str = "Learning curves",
        save_path: Optional[str] = None,
    ) -> Axes:
        """Training and validation score against training-set size.

        Accepts raw fold scores (``train_scores``/``validation_scores`` shaped
        ``(n_sizes, n_folds)``) or precomputed ``*_scores_mean``/``*_scores_std``.

        Args:
            learning_data: Dictionary with ``train_sizes`` and score arrays.
            ax: Axes to draw on.
            score_name: Y axis label (falls back to ``scoring_metric`` key).
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The axes containing both curves.

        Raises:
            KeyError: If required keys are absent.
        """
        sizes = to_1d_array(learning_data["train_sizes"], "train_sizes")
        tr_mean, tr_std = self._mean_std(learning_data, "train_scores", "train_scores_mean")
        va_mean, va_std = self._mean_std(
            learning_data,
            "validation_scores",
            "val_scores",
            "test_scores",
            "val_scores_mean",
            "validation_scores_mean",
        )
        c_train, c_val = self._colors(2)
        fig, ax, created = self._axes(ax)
        for mean, std, color, label in (
            (tr_mean, tr_std, c_train, "training"),
            (va_mean, va_std, c_val, "validation"),
        ):
            ax.plot(sizes, mean, marker="o", color=color, linewidth=2, label=label)
            if std is not None:
                ax.fill_between(sizes, mean - std, mean + std, color=color, alpha=0.15)
        ax.legend(loc="best", fontsize=8)
        style_axes(ax, title, "Training set size", score_name or learning_data.get("scoring_metric", "Score"))
        finalize_figure(fig, created, save_path)
        return ax

    def plot_validation_curve(
        self,
        param_range: Any,
        train_scores: Any,
        val_scores: Any,
        param_name: str = "parameter",
        ax: Optional[Axes] = None,
        log_scale: Optional[bool] = None,
        score_name: str = "Score",
        title: str = "Validation curve",
        save_path: Optional[str] = None,
    ) -> Axes:
        """Training and validation score against a hyper-parameter.

        Non-numeric parameter values (e.g. ``None`` or strings) are placed on
        a categorical axis.

        Args:
            param_range: Parameter values.
            train_scores: Fold scores ``(n_params, n_folds)`` or means.
            val_scores: Fold scores ``(n_params, n_folds)`` or means.
            param_name: X axis label.
            ax: Axes to draw on.
            log_scale: Force/disable a log x axis (auto: >100x span).
            score_name: Y axis label.
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The axes containing both curves.

        Raises:
            ValueError: If the score arrays do not match ``param_range``.
        """
        values = list(param_range)
        tr = np.asarray(train_scores, dtype=float)
        va = np.asarray(val_scores, dtype=float)
        if tr.shape[0] != len(values) or va.shape[0] != len(values):
            raise ValueError("train_scores and val_scores must have one row per parameter value")
        tr_mean, tr_std = (tr.mean(axis=1), tr.std(axis=1)) if tr.ndim == 2 else (tr, None)
        va_mean, va_std = (va.mean(axis=1), va.std(axis=1)) if va.ndim == 2 else (va, None)
        try:
            x = np.asarray(values, dtype=float)
            numeric = np.all(np.isfinite(x))
        except (TypeError, ValueError):
            numeric = False
        if not numeric:
            x = np.arange(len(values), dtype=float)
        c_train, c_val = self._colors(2)
        fig, ax, created = self._axes(ax)
        for mean, std, color, label in (
            (tr_mean, tr_std, c_train, "training"),
            (va_mean, va_std, c_val, "validation"),
        ):
            ax.plot(x, mean, marker="o", color=color, linewidth=2, label=label)
            if std is not None:
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
        best = int(np.argmax(va_mean))
        ax.axvline(x[best], color=REFERENCE_COLOR, linestyle=":", linewidth=1, label=f"best = {values[best]}")
        if not numeric:
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in values])
        elif log_scale or (log_scale is None and x.min() > 0 and x.max() / x.min() > 100):
            ax.set_xscale("log")
        ax.legend(loc="best", fontsize=8)
        style_axes(ax, title, param_name, score_name)
        finalize_figure(fig, created, save_path)
        return ax

    def plot_calibration_curve(
        self,
        fraction_of_positives: Any,
        mean_predicted_value: Any,
        ax: Optional[Axes] = None,
        label: str = "model",
        title: str = "Calibration curve",
        save_path: Optional[str] = None,
    ) -> Axes:
        """Reliability diagram with the perfectly-calibrated diagonal.

        Args:
            fraction_of_positives: Observed positive rate per bin.
            mean_predicted_value: Mean predicted probability per bin.
            ax: Axes to draw on.
            label: Legend label.
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The axes containing the curve.

        Raises:
            ValueError: If the inputs differ in length.
        """
        frac = to_1d_array(fraction_of_positives, "fraction_of_positives").astype(float)
        mean = to_1d_array(mean_predicted_value, "mean_predicted_value").astype(float)
        if frac.shape != mean.shape:
            raise ValueError("fraction_of_positives and mean_predicted_value must have the same length")
        fig, ax, created = self._axes(ax)
        n_existing = sum(1 for line in ax.get_lines() if line.get_label() != "perfectly calibrated")
        ax.plot(mean, frac, marker="s", color=self._colors(n_existing + 1)[-1], linewidth=2, label=label)
        if not any(line.get_label() == "perfectly calibrated" for line in ax.get_lines()):
            ax.plot(
                [0, 1],
                [0, 1],
                color=REFERENCE_COLOR,
                linestyle=":",
                linewidth=1,
                label="perfectly calibrated",
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=8)
        style_axes(ax, title, "Mean predicted probability", "Fraction of positives")
        finalize_figure(fig, created, save_path)
        return ax


# ============================================================================ features
class FeatureVisualizer(_BaseVisualizer):
    """Feature-level plots: rankings, target correlations, per-class distributions."""

    def plot_feature_importance_ranking(
        self,
        importance_scores: Any,
        feature_names: Optional[Sequence[str]] = None,
        ax: Optional[Axes] = None,
        top_n: Optional[int] = None,
        title: str = "Feature importance ranking",
        save_path: Optional[str] = None,
    ) -> Axes:
        """Ranked horizontal bars of importance scores.

        Args:
            importance_scores: One score per feature.
            feature_names: Feature labels.
            ax: Axes to draw on.
            top_n: Keep only the ``top_n`` features.
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The axes containing the bars.
        """
        return ModelVisualizer(self.figure_size, self.palette).plot_feature_importance(
            importance_scores,
            feature_names=feature_names,
            ax=ax,
            top_n=top_n,
            title=title,
            save_path=save_path,
        )

    def plot_feature_correlation_with_target(
        self,
        correlations: Any,
        feature_names: Optional[Sequence[str]] = None,
        ax: Optional[Axes] = None,
        sort: bool = True,
        title: str = "Feature-target correlation",
        save_path: Optional[str] = None,
    ) -> Axes:
        """Signed bar chart of each feature's correlation with the target.

        Positive and negative correlations use the diverging colour pair.

        Args:
            correlations: Correlation coefficient per feature.
            feature_names: Feature labels.
            ax: Axes to draw on.
            sort: Order bars by absolute correlation.
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The axes containing the bars.

        Raises:
            ValueError: If ``correlations`` is empty.
        """
        corr = to_1d_array(correlations, "correlations").astype(float)
        if corr.size == 0:
            raise ValueError("correlations is empty")
        names = default_feature_names(corr.size, feature_names)
        order = np.argsort(np.abs(corr)) if sort else np.arange(corr.size)
        fig, ax, created = self._axes(ax)
        ax.barh(
            [names[i] for i in order],
            corr[order],
            color=[POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in corr[order]],
        )
        ax.axvline(0, color=REFERENCE_COLOR, linewidth=1)
        ax.set_xlim(-1, 1)
        style_axes(ax, title, "Correlation with target", None)
        finalize_figure(fig, created, save_path)
        return ax

    def plot_pairwise_relationships(
        self,
        X: Any,
        feature_names: Optional[Sequence[str]] = None,
        y: Any = None,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
    ) -> Figure:
        """Scatter matrix of pairwise feature relationships.

        Args:
            X: Feature matrix.
            feature_names: Feature labels.
            y: Optional target for colouring.
            figsize: Figure size.
            save_path: Optional output path.

        Returns:
            Figure with ``n_features ** 2`` axes.
        """
        return DataVisualizer(self.figure_size, self.palette).plot_scatter_matrix(
            X, y=y, feature_names=feature_names, figsize=figsize, save_path=save_path
        )

    def plot_feature_distributions_by_class(
        self,
        X: Any,
        y: Any,
        feature_names: Optional[Sequence[str]] = None,
        bins: int = 20,
        max_cols: int = 3,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
    ) -> Figure:
        """Overlaid per-class histograms, one panel per feature.

        When ``feature_names`` is shorter than ``X.shape[1]`` only the first
        ``len(feature_names)`` columns are shown.

        Args:
            X: Feature matrix.
            y: Class labels.
            feature_names: Names (and count) of features to plot.
            bins: Histogram bins.
            max_cols: Maximum panels per row.
            figsize: Figure size.
            save_path: Optional output path.

        Returns:
            Figure with one axes per plotted feature.

        Raises:
            ValueError: If more names than features are given.
        """
        X, names = select_features(X, feature_names)
        y = to_1d_array(y, "y")
        classes = np.unique(y)
        colors = self._colors(len(classes))
        fig, axes = subplot_grid(len(names), max_cols=max_cols, figsize=figsize)
        for j, (ax, name) in enumerate(zip(axes, names)):
            edges = np.histogram_bin_edges(X[:, j], bins=bins)
            for cls, color in zip(classes, colors):
                ax.hist(
                    X[y == cls, j], bins=edges, color=color, alpha=0.5, density=True, label=f"class {cls}"
                )
            style_axes(ax, name, None, "Density")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=8)
        finalize_figure(fig, True, save_path)
        return fig


# ========================================================================== comparison
class ComparisonVisualizer(_BaseVisualizer):
    """Plots comparing several models on one or more metrics."""

    def plot_model_comparison_bar(
        self,
        comparison_data: Union[Dict[str, Dict[str, float]], pd.DataFrame],
        metric: str,
        ax: Optional[Axes] = None,
        sort: bool = True,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Axes:
        """Bar chart of one metric across models with value labels.

        Args:
            comparison_data: ``{model: {metric: value}}`` or a DataFrame indexed by model.
            metric: Column to plot.
            ax: Axes to draw on.
            sort: Order models by the metric (descending).
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The axes containing the bars.

        Raises:
            KeyError: If ``metric`` is not present.
        """
        df = as_dataframe(comparison_data)
        if metric not in df.columns:
            raise KeyError(f"metric '{metric}' not found; available: {list(df.columns)}")
        series = df[metric].astype(float)
        if sort:
            series = series.sort_values(ascending=False)
        fig, ax, created = self._axes(ax)
        bars = ax.bar(
            [str(i) for i in series.index], series.to_numpy(), color=self._colors(len(series)), width=0.7
        )
        for bar, value in zip(bars, series):
            ax.annotate(
                f"{value:.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 2),
                textcoords="offset points",
            )
        ax.tick_params(axis="x", rotation=20)
        style_axes(ax, title or f"{metric} by model", "Model", metric)
        finalize_figure(fig, created, save_path)
        return ax

    def plot_model_comparison_radar(
        self,
        comparison_data: Union[Dict[str, Dict[str, float]], pd.DataFrame],
        ax: Optional[Axes] = None,
        metrics: Optional[Sequence[str]] = None,
        title: str = "Model comparison",
        save_path: Optional[str] = None,
    ) -> Axes:
        """Radar (spider) chart with one closed polygon per model.

        Args:
            comparison_data: ``{model: {metric: value}}`` or DataFrame.
            ax: Polar axes to draw on; created when ``None``.
            metrics: Metrics (spokes) to include; defaults to all columns.
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The polar axes containing the polygons.

        Raises:
            ValueError: If ``ax`` is not a polar axes or fewer than 3 metrics remain.
        """
        df = as_dataframe(comparison_data)
        cols = list(metrics) if metrics is not None else list(df.columns)
        if len(cols) < 3:
            raise ValueError("A radar chart needs at least 3 metrics")
        if ax is not None and ax.name != "polar":
            raise ValueError("ax must be created with projection='polar'")
        fig, ax, created = self._axes(ax, projection="polar")
        angles = np.linspace(0, 2 * np.pi, len(cols), endpoint=False)
        closed = np.concatenate([angles, angles[:1]])
        for (model, row), color in zip(df[cols].iterrows(), self._colors(len(df))):
            values = row.to_numpy(dtype=float)
            values = np.concatenate([values, values[:1]])
            ax.plot(closed, values, color=color, linewidth=2, marker="o", markersize=4, label=str(model))
            ax.fill(closed, values, color=color, alpha=0.1)
        ax.set_xticks(angles)
        ax.set_xticklabels(cols)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=18)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        finalize_figure(fig, created, save_path)
        return ax

    def plot_performance_heatmap(
        self,
        comparison_data: Union[Dict[str, Dict[str, float]], pd.DataFrame],
        ax: Optional[Axes] = None,
        metrics: Optional[Sequence[str]] = None,
        annot: bool = True,
        title: str = "Performance heatmap",
        save_path: Optional[str] = None,
    ) -> Axes:
        """Models x metrics heatmap on a sequential scale.

        Args:
            comparison_data: ``{model: {metric: value}}`` or DataFrame.
            ax: Axes to draw on.
            metrics: Columns to include.
            annot: Annotate cells with values.
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The axes containing the heatmap.
        """
        df = as_dataframe(comparison_data)
        if metrics is not None:
            df = df[list(metrics)]
        fig, ax, created = self._axes(ax)
        sns.heatmap(
            df.astype(float),
            annot=annot,
            fmt=".3f",
            cmap=SEQUENTIAL_CMAP,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Model")
        finalize_figure(fig, created, save_path)
        return ax

    def plot_metric_distribution(
        self,
        cv_scores: Dict[str, Any],
        ax: Optional[Axes] = None,
        kind: str = "box",
        metric_name: str = "Score",
        title: str = "Cross-validation score distribution",
        save_path: Optional[str] = None,
    ) -> Axes:
        """Box or violin plot of per-fold scores for each model.

        Args:
            cv_scores: ``{model: fold_scores}``.
            ax: Axes to draw on.
            kind: ``"box"`` or ``"violin"``.
            metric_name: Y axis label.
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The axes containing the plot.

        Raises:
            ValueError: If ``cv_scores`` is empty or ``kind`` is unknown.
        """
        if not cv_scores:
            raise ValueError("cv_scores is empty")
        if kind not in ("box", "violin"):
            raise ValueError("kind must be 'box' or 'violin'")
        names = list(cv_scores)
        data = [to_1d_array(cv_scores[n], n).astype(float) for n in names]
        colors = self._colors(len(names))
        fig, ax, created = self._axes(ax)
        positions = range(1, len(names) + 1)
        if kind == "box":
            parts = ax.boxplot(
                data,
                positions=list(positions),
                patch_artist=True,
                showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": REFERENCE_COLOR},
            )
            bodies = parts["boxes"]
        else:
            parts = ax.violinplot(data, positions=list(positions), showmeans=True, showmedians=False)
            bodies = parts["bodies"]
        for body, color in zip(bodies, colors):
            body.set_facecolor(color)
            body.set_alpha(0.7)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(names, rotation=20)
        style_axes(ax, title, "Model", metric_name)
        finalize_figure(fig, created, save_path)
        return ax


# =============================================================================== suite
class ModelVisualizationSuite(LoggerMixin):
    """Facade combining the specialised visualizers behind one object.

    Kept for backward compatibility with earlier releases; new code can use
    the focused classes directly. All ``plot_*`` methods return a
    :class:`matplotlib.figure.Figure`; the ``create_*`` methods return Plotly
    figures when Plotly is installed and matplotlib figures otherwise.

    Args:
        style: Matplotlib style name applied to figures created by the suite.
        figure_size: Default figure size.
        palette: Categorical palette.
    """

    def __init__(
        self,
        style: Optional[str] = None,
        figure_size: Optional[Tuple[float, float]] = None,
        palette: Optional[Sequence[str]] = None,
    ) -> None:
        self.style = style or settings.STYLE
        self.figure_size: Tuple[float, float] = tuple(figure_size or settings.FIGURE_SIZE)  # type: ignore[assignment]
        self.colors: List[str] = list(palette or CATEGORICAL_PALETTE)
        if self.style not in plt.style.available and self.style != "default":
            self.logger.warning("Matplotlib style '%s' not available; using 'default'", self.style)
            self.style = "default"
        self.data = DataVisualizer(self.figure_size, self.colors)
        self.model = ModelVisualizer(self.figure_size, self.colors)
        self.performance = PerformanceVisualizer(self.figure_size, self.colors)
        self.features = FeatureVisualizer(self.figure_size, self.colors)
        self.comparison = ComparisonVisualizer(self.figure_size, self.colors)
        self.interactive = InteractiveVisualizer(palette=self.colors)

    def _new_axes(self, figsize: Optional[Tuple[float, float]] = None) -> Tuple[Figure, Axes]:
        with plt.style.context(self.style):
            fig, ax = plt.subplots(figsize=figsize or self.figure_size)
        return fig, ax

    @staticmethod
    def _save(fig: Figure, save_path: Optional[str]) -> Figure:
        return finalize_figure(fig, True, save_path)

    def plot_confusion_matrix(
        self,
        y_true: Any,
        y_pred: Any,
        class_names: Optional[Sequence[str]] = None,
        normalize: bool = True,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None,
    ) -> Figure:
        """Confusion matrix computed from labels and predictions.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            class_names: Class labels.
            normalize: Show row-normalised rates.
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The figure.
        """
        cm = confusion_matrix(to_1d_array(y_true, "y_true"), to_1d_array(y_pred, "y_pred"))
        fig, ax = self._new_axes()
        self.performance.plot_confusion_matrix(
            cm, ax=ax, class_names=class_names, normalize=normalize, title=title
        )
        return self._save(fig, save_path)

    def plot_roc_curves(
        self,
        roc_data: Dict[str, Dict[str, Any]],
        title: str = "ROC Curves Comparison",
        save_path: Optional[str] = None,
    ) -> Figure:
        """Overlay ROC curves for several models.

        Args:
            roc_data: ``{model: {"fpr": ..., "tpr": ..., "auc": optional}}``.
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The figure.
        """
        fig, ax = self._new_axes()
        for model_name, data in roc_data.items():
            self.performance.plot_roc_curve(
                data["fpr"], data["tpr"], data.get("auc"), ax=ax, label=model_name, title=title
            )
        return self._save(fig, save_path)

    def plot_precision_recall_curves(
        self,
        pr_data: Dict[str, Dict[str, Any]],
        title: str = "Precision-Recall Curves",
        save_path: Optional[str] = None,
    ) -> Figure:
        """Overlay precision-recall curves for several models.

        Args:
            pr_data: ``{model: {"precision": ..., "recall": ..., "average_precision": optional}}``.
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The figure.
        """
        fig, ax = self._new_axes()
        for model_name, data in pr_data.items():
            ap = data.get("average_precision")
            if ap is None:
                ap = trapezoid(data["precision"], data["recall"])
            self.performance.plot_precision_recall_curve(
                data["precision"], data["recall"], ap, ax=ax, label=model_name, title=title
            )
        return self._save(fig, save_path)

    def plot_learning_curves(
        self, learning_data: Dict[str, Any], title: str = "Learning Curves", save_path: Optional[str] = None
    ) -> Figure:
        """Learning curves from raw fold scores or precomputed means/stds.

        Args:
            learning_data: See :meth:`PerformanceVisualizer.plot_learning_curves`.
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The figure.
        """
        fig, ax = self._new_axes()
        self.performance.plot_learning_curves(learning_data, ax=ax, title=title)
        return self._save(fig, save_path)

    def plot_validation_curve(
        self,
        validation_data: Dict[str, Any],
        title: str = "Validation Curve",
        save_path: Optional[str] = None,
    ) -> Figure:
        """Validation curve from a result dictionary.

        Args:
            validation_data: Dictionary with ``param_range``, ``param_name`` and
                ``train_scores``/``val_scores`` (raw) or ``*_scores_mean``.
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The figure.
        """
        train = validation_data.get("train_scores", validation_data.get("train_scores_mean"))
        val = validation_data.get(
            "val_scores", validation_data.get("val_scores_mean", validation_data.get("validation_scores"))
        )
        fig, ax = self._new_axes()
        self.performance.plot_validation_curve(
            validation_data["param_range"],
            train,
            val,
            param_name=validation_data.get("param_name", "parameter"),
            ax=ax,
            score_name=validation_data.get("scoring_metric", "Score"),
            title=title,
        )
        return self._save(fig, save_path)

    def plot_feature_importance(
        self,
        feature_names: Sequence[str],
        importance_scores: Any,
        title: str = "Feature Importance",
        max_features: int = 20,
        save_path: Optional[str] = None,
    ) -> Figure:
        """Top-``max_features`` importance bars.

        Args:
            feature_names: Feature labels.
            importance_scores: Importance per feature.
            title: Plot title.
            max_features: Number of features shown.
            save_path: Optional output path.

        Returns:
            The figure.
        """
        n = min(max_features, len(feature_names))
        fig, ax = self._new_axes((self.figure_size[0], max(4.0, 0.4 * n + 1.5)))
        self.model.plot_feature_importance(
            importance_scores, feature_names=feature_names, ax=ax, top_n=max_features, title=title
        )
        return self._save(fig, save_path)

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric_column: str,
        title: str = "Model Performance Comparison",
        name_column: str = "model_name",
        save_path: Optional[str] = None,
    ) -> Figure:
        """Bar chart of one metric across the models in ``comparison_df``.

        Args:
            comparison_df: DataFrame with a model-name column and metric columns.
            metric_column: Metric to plot.
            title: Plot title.
            name_column: Column holding model names (falls back to the index).
            save_path: Optional output path.

        Returns:
            The figure.
        """
        df = comparison_df.set_index(name_column) if name_column in comparison_df.columns else comparison_df
        fig, ax = self._new_axes()
        self.comparison.plot_model_comparison_bar(df, metric_column, ax=ax, title=title)
        return self._save(fig, save_path)

    def plot_residuals(
        self, y_true: Any, y_pred: Any, title: str = "Residual Analysis", save_path: Optional[str] = None
    ) -> Figure:
        """Four-panel residual diagnostics.

        Args:
            y_true: True targets.
            y_pred: Predicted targets.
            title: Figure title.
            save_path: Optional output path.

        Returns:
            The figure.
        """
        with plt.style.context(self.style):
            return self.model.plot_residual_diagnostics(y_true, y_pred, title=title, save_path=save_path)

    def plot_clustering_results(
        self,
        X: Any,
        labels: Any,
        centers: Optional[Any] = None,
        title: str = "Clustering Results",
        save_path: Optional[str] = None,
    ) -> Figure:
        """Scatter of cluster assignments, PCA-projected to 2-D when needed.

        Args:
            X: Feature matrix.
            labels: Cluster labels (``-1`` marks noise).
            centers: Optional cluster centres in the original space.
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The figure.
        """
        X = to_2d_array(X)
        labels = to_1d_array(labels, "labels")
        centers_arr = None if centers is None else to_2d_array(centers, "centers")
        if X.shape[1] > 2:
            scaler = StandardScaler().fit(X)
            pca = PCA(n_components=2, random_state=settings.RANDOM_SEED).fit(scaler.transform(X))
            X_viz = pca.transform(scaler.transform(X))
            centers_viz = None if centers_arr is None else pca.transform(scaler.transform(centers_arr))
            axis_labels = ("PC 1", "PC 2")
        else:
            X_viz, centers_viz, axis_labels = X, centers_arr, ("Feature 1", "Feature 2")
        uniques = [u for u in np.unique(labels) if u != -1]
        fig, ax = self._new_axes()
        for cluster, color in zip(uniques, series_colors(len(uniques), self.colors)):
            mask = labels == cluster
            ax.scatter(
                X_viz[mask, 0], X_viz[mask, 1], s=30, color=color, alpha=0.75, label=f"cluster {cluster}"
            )
        if np.any(labels == -1):
            ax.scatter(
                X_viz[labels == -1, 0],
                X_viz[labels == -1, 1],
                s=20,
                color=REFERENCE_COLOR,
                marker="x",
                alpha=0.5,
                label="noise",
            )
        if centers_viz is not None:
            ax.scatter(
                centers_viz[:, 0],
                centers_viz[:, 1],
                s=180,
                color="black",
                marker="X",
                edgecolor="white",
                linewidth=1.5,
                label="centres",
                zorder=4,
            )
        ax.legend(fontsize=8)
        style_axes(ax, title, *axis_labels)
        return self._save(fig, save_path)

    def plot_decision_boundary(
        self,
        model: Any,
        X: Any,
        y: Any,
        feature_names: Optional[Sequence[str]] = None,
        title: str = "Decision Boundary",
        save_path: Optional[str] = None,
    ) -> Figure:
        """Decision regions of a 2-D classifier.

        Args:
            model: Fitted classifier.
            X: Two-column feature matrix.
            y: Labels.
            feature_names: Axis labels.
            title: Plot title.
            save_path: Optional output path.

        Returns:
            The figure.

        Raises:
            ValueError: If ``X`` does not have exactly two columns.
        """
        fig, ax = self._new_axes()
        try:
            self.model.plot_decision_boundary(model, X, y, ax=ax, feature_names=feature_names, title=title)
        except Exception:
            plt.close(fig)
            raise
        return self._save(fig, save_path)

    def plot_calibration_curve(
        self,
        y_true: Any,
        y_prob: Any,
        n_bins: int = 10,
        title: str = "Calibration Plot",
        save_path: Optional[str] = None,
    ) -> Figure:
        """Reliability diagram plus a histogram of predicted probabilities.

        Args:
            y_true: Binary labels.
            y_prob: Positive-class probabilities.
            n_bins: Calibration bins.
            title: Figure title.
            save_path: Optional output path.

        Returns:
            The figure with two axes.
        """
        from sklearn.calibration import calibration_curve

        y_true = to_1d_array(y_true, "y_true")
        y_prob = to_1d_array(y_prob, "y_prob").astype(float)
        frac, mean = calibration_curve(y_true, y_prob, n_bins=n_bins)
        with plt.style.context(self.style):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figure_size[0], self.figure_size[1] / 1.6))
        self.performance.plot_calibration_curve(frac, mean, ax=ax1)
        ax2.hist(y_prob, bins=n_bins, range=(0, 1), color=self.colors[0], alpha=0.85)
        style_axes(ax2, "Predicted probabilities", "Predicted probability", "Count")
        fig.suptitle(title, fontsize=14, fontweight="bold")
        return self._save(fig, save_path)

    def plot_feature_importance_waterfall(
        self,
        base_value: float,
        feature_contributions: Any,
        feature_names: Sequence[str],
        prediction: Optional[float] = None,
        title: str = "Feature Contribution Analysis",
        max_features: int = 15,
        save_path: Optional[str] = None,
    ) -> Figure:
        """SHAP-style waterfall from a base value to the prediction.

        Args:
            base_value: Expected value before any feature contribution.
            feature_contributions: Signed contribution per feature.
            feature_names: Feature labels.
            prediction: Final prediction (defaults to base + sum of contributions).
            title: Plot title.
            max_features: Largest contributions shown individually; the rest are pooled.
            save_path: Optional output path.

        Returns:
            The figure.
        """
        contrib = to_1d_array(feature_contributions, "feature_contributions").astype(float)
        names = default_feature_names(contrib.size, feature_names)
        order = np.argsort(np.abs(contrib))[::-1]
        shown = order[:max_features]
        rest = order[max_features:]
        labels = [names[i] for i in shown]
        values = list(contrib[shown])
        if rest.size:
            labels.append(f"other ({rest.size})")
            values.append(float(contrib[rest].sum()))
        prediction = base_value + float(contrib.sum()) if prediction is None else prediction
        fig, ax = self._new_axes()
        cumulative = base_value
        for i, (label, value) in enumerate(zip(labels, values)):
            ax.bar(
                i, value, bottom=cumulative, color=POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR, width=0.7
            )
            ax.text(
                i,
                cumulative + value,
                f"{value:+.3f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )
            cumulative += value
        ax.axhline(
            base_value, color=REFERENCE_COLOR, linestyle=":", linewidth=1, label=f"base = {base_value:.3f}"
        )
        ax.axhline(
            prediction,
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=1,
            label=f"prediction = {prediction:.3f}",
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend(fontsize=8)
        style_axes(ax, title, None, "Output value")
        return self._save(fig, save_path)

    def plot_fairness_metrics(
        self,
        fairness_results: Dict[str, float],
        sensitive_attribute: str,
        title: str = "Model Fairness Analysis",
        threshold: float = 0.1,
        save_path: Optional[str] = None,
    ) -> Figure:
        """Group-wise accuracy and aggregate fairness gaps.

        Args:
            fairness_results: Mapping with ``accuracy_group_<g>`` entries and
                optional gap metrics such as ``demographic_parity_diff``.
            sensitive_attribute: Name of the protected attribute.
            title: Figure title.
            threshold: Acceptable absolute gap drawn as a reference line.
            save_path: Optional output path.

        Returns:
            The figure with two axes.
        """
        with plt.style.context(self.style):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figure_size[0], self.figure_size[1] / 1.6))
        groups = {
            k.replace("accuracy_group_", ""): v
            for k, v in fairness_results.items()
            if k.startswith("accuracy_group_")
        }
        if groups:
            bars = ax1.bar(
                list(groups), list(groups.values()), color=series_colors(len(groups), self.colors), width=0.7
            )
            for bar, value in zip(bars, groups.values()):
                ax1.annotate(
                    f"{value:.3f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    xytext=(0, 2),
                    textcoords="offset points",
                )
            ax1.set_ylim(0, 1.05)
        style_axes(ax1, "Accuracy by group", f"{sensitive_attribute} group", "Accuracy")
        gaps = {
            k: v
            for k, v in fairness_results.items()
            if k.endswith(("_diff", "_gap", "_ratio")) and v is not None
        }
        if gaps:
            labels = [k.replace("_", " ") for k in gaps]
            ax2.bar(
                labels,
                list(gaps.values()),
                color=[NEGATIVE_COLOR if abs(v) > threshold else POSITIVE_COLOR for v in gaps.values()],
                width=0.6,
            )
            ax2.axhline(
                threshold,
                color=REFERENCE_COLOR,
                linestyle="--",
                linewidth=1,
                label=f"threshold = {threshold}",
            )
            ax2.tick_params(axis="x", rotation=20)
            ax2.legend(fontsize=8)
        style_axes(ax2, "Fairness gaps", None, "Gap")
        fig.suptitle(title, fontsize=14, fontweight="bold")
        return self._save(fig, save_path)

    def create_interactive_scatter_plot(
        self,
        X: Any,
        y: Any,
        feature_names: Optional[Sequence[str]] = None,
        target_name: str = "Target",
        title: str = "Interactive Data Exploration",
    ) -> Any:
        """Interactive scatter of the first two features coloured by target.

        Args:
            X: Feature matrix with at least two columns.
            y: Target values.
            feature_names: Axis labels.
            target_name: Legend title.
            title: Plot title.

        Returns:
            Plotly figure (or matplotlib figure without Plotly).

        Raises:
            ValueError: If ``X`` has fewer than two features.
        """
        X = to_2d_array(X)
        if X.shape[1] < 2:
            raise ValueError("Need at least 2 features for a scatter plot")
        names = default_feature_names(X.shape[1], feature_names)
        return self.interactive.create_interactive_scatter(
            X[:, 0], X[:, 1], y, x_label=names[0], y_label=names[1], color_label=target_name, title=title
        )

    def create_model_performance_dashboard(
        self, results: Dict[str, Any], save_path: Optional[str] = None
    ) -> Any:
        """Task-aware performance dashboard from an evaluation result dictionary.

        Args:
            results: Output of ``ModelEvaluator`` containing ``task_type`` and metrics.
            save_path: Optional HTML output path.

        Returns:
            Plotly figure (or matplotlib figure without Plotly).
        """
        task_type = results.get("task_type", "classification")
        if task_type == "classification":
            fig = self._create_classification_dashboard(results)
        elif task_type == "regression":
            fig = self._create_regression_dashboard(results)
        else:
            fig = self._create_clustering_dashboard(results)
        if save_path:
            self.interactive.save_html(fig, save_path)
        return fig

    def _bar_panels(self, panels: List[Tuple[str, Dict[str, float]]], title: str) -> Any:
        """Render ``(panel_title, {label: value})`` bar panels with either backend."""
        panels = [(name, data) for name, data in panels if data]
        if not panels:
            panels = [("No metrics", {"n/a": 0.0})]
        if self.interactive.backend == "plotly":
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(rows=1, cols=len(panels), subplot_titles=[p[0] for p in panels])
            for col, (_, data) in enumerate(panels, start=1):
                fig.add_trace(
                    go.Bar(
                        x=list(data),
                        y=list(data.values()),
                        marker={"color": self.colors[0]},
                        showlegend=False,
                    ),
                    row=1,
                    col=col,
                )
            fig.update_layout(title=title, height=450)
            return fig
        fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4), squeeze=False)
        for ax, (name, data) in zip(axes[0], panels):
            ax.bar(list(data), list(data.values()), color=self.colors[0], width=0.7)
            ax.tick_params(axis="x", rotation=20)
            style_axes(ax, name)
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig

    @staticmethod
    def _cv_means(results: Dict[str, Any], limit: int = 4) -> Dict[str, float]:
        cv = results.get("cross_validation") or {}
        out: Dict[str, float] = {}
        for metric, stats_ in list(cv.items())[:limit]:
            value = stats_.get("mean") if isinstance(stats_, dict) else stats_
            if value is not None:
                out[metric.replace("neg_", "").replace("_", " ")] = float(value)
        return out

    def _create_classification_dashboard(self, results: Dict[str, Any]) -> Any:
        metrics = {
            m.replace("test_", ""): float(results[m])
            for m in ("test_accuracy", "test_precision", "test_recall", "test_f1")
            if results.get(m) is not None
        }
        panels: List[Tuple[str, Dict[str, float]]] = [
            ("Test metrics", metrics),
            ("Cross-validation", self._cv_means(results)),
        ]
        model_name = results.get("model_name", "Model")
        fig = self._bar_panels(panels, f"Classification performance - {model_name}")
        cm = results.get("confusion_matrix")
        roc = results.get("roc_curve")
        if self.interactive.backend == "plotly" and (cm is not None or roc is not None):
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            dash = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=("Confusion matrix", "ROC curve", "Test metrics", "Cross-validation"),
                specs=[[{"type": "heatmap"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
            )
            if cm is not None:
                dash.add_trace(
                    go.Heatmap(z=np.asarray(cm), colorscale=SEQUENTIAL_CMAP, showscale=False), row=1, col=1
                )
            if roc is not None:
                dash.add_trace(
                    go.Scatter(
                        x=roc["fpr"], y=roc["tpr"], mode="lines", name="ROC", line={"color": self.colors[0]}
                    ),
                    row=1,
                    col=2,
                )
                dash.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="chance",
                        line={"color": REFERENCE_COLOR, "dash": "dot"},
                    ),
                    row=1,
                    col=2,
                )
            for trace, (row, col) in zip(fig.data, ((2, 1), (2, 2))):
                dash.add_trace(trace, row=row, col=col)
            dash.update_layout(
                title=f"Classification performance - {model_name}", height=800, showlegend=False
            )
            return dash
        return fig

    def _create_regression_dashboard(self, results: Dict[str, Any]) -> Any:
        metrics = {
            m.replace("test_", "").upper(): float(results[m])
            for m in ("test_r2", "test_rmse", "test_mae")
            if results.get(m) is not None
        }
        residuals = results.get("residuals") or {}
        res_stats = {
            k: float(residuals[k]) for k in ("mean", "std", "min", "max") if residuals.get(k) is not None
        }
        panels = [
            ("Test metrics", metrics),
            ("Cross-validation", self._cv_means(results, 3)),
            ("Residual statistics", res_stats),
        ]
        return self._bar_panels(panels, f"Regression performance - {results.get('model_name', 'Model')}")

    def _create_clustering_dashboard(self, results: Dict[str, Any]) -> Any:
        metrics = {
            m.replace("_score", "").replace("_", " "): float(results[m])
            for m in ("silhouette_score", "calinski_harabasz_score", "davies_bouldin_score")
            if results.get(m) is not None
        }
        sizes = {f"cluster {c}": float(s) for c, s in (results.get("cluster_sizes") or {}).items()}
        info = {
            k.replace("_", " "): float(results[k])
            for k in ("n_clusters", "n_noise_points")
            if results.get(k) is not None
        }
        return self._bar_panels(
            [("Clustering metrics", metrics), ("Cluster sizes", sizes), ("Cluster info", info)],
            f"Clustering performance - {results.get('model_name', 'Model')}",
        )
