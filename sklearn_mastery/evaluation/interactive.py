"""Interactive (Plotly-based) visualizations with matplotlib fallbacks.

:class:`InteractiveVisualizer` produces ``plotly.graph_objects.Figure``
objects when :mod:`plotly` is importable and degrades to static matplotlib
figures otherwise, so notebooks, reports and headless test runs share one
API. No method displays anything; callers decide whether to ``show`` or
export (see :meth:`InteractiveVisualizer.export_to_html`).
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

from sklearn_mastery.config.logging_config import LoggerMixin
from sklearn_mastery.evaluation._viz_common import (
    CATEGORICAL_PALETTE,
    HAS_IPYWIDGETS,
    HAS_PLOTLY,
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
    REFERENCE_COLOR,
    SEQUENTIAL_CMAP,
    default_feature_names,
    is_discrete_target,
    plotly_available,
    series_colors,
    subplot_grid,
    to_1d_array,
    to_2d_array,
    trapezoid,
)

__all__ = ["HAS_IPYWIDGETS", "HAS_PLOTLY", "InteractiveVisualizer"]


#: Responsive layout presets: width/height in pixels plus typography/margins.
RESPONSIVE_LAYOUTS: Dict[str, Dict[str, Any]] = {
    "mobile": {"width": 360, "height": 480, "font_size": 10, "margin": 30, "legend_orientation": "h"},
    "tablet": {"width": 768, "height": 560, "font_size": 12, "margin": 40, "legend_orientation": "h"},
    "desktop": {"width": 1100, "height": 650, "font_size": 13, "margin": 50, "legend_orientation": "v"},
    "presentation": {"width": 1600, "height": 900, "font_size": 18, "margin": 70, "legend_orientation": "v"},
}

#: Default theme; keys mirror those accepted by :meth:`InteractiveVisualizer.apply_custom_theme`.
DEFAULT_THEME: Dict[str, Any] = {
    "background_color": "#ffffff",
    "paper_color": "#ffffff",
    "grid_color": "#e6e6e3",
    "text_color": "#0b0b0b",
    "font_family": "Helvetica, Arial, sans-serif",
    "font_size": 12,
}


def _sort_key(value: Any) -> Tuple[int, Any]:
    """Sort helper placing numeric values before strings (``None`` last)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return (2, "")
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


class InteractiveVisualizer(LoggerMixin):
    """Interactive plotting toolkit backed by Plotly with matplotlib fallbacks.

    Args:
        theme: Optional theme overrides (see :meth:`apply_custom_theme`).
        palette: Categorical colour sequence; defaults to the package palette.
        layout: Responsive preset name (``"mobile"``, ``"tablet"``,
            ``"desktop"`` or ``"presentation"``) controlling default size.
        prefer_plotly: When ``False`` matplotlib is always used.

    Attributes:
        theme: Active theme configuration.
        palette: Active categorical palette.
    """

    def __init__(
        self,
        theme: Optional[Dict[str, Any]] = None,
        palette: Optional[Sequence[str]] = None,
        layout: str = "desktop",
        prefer_plotly: bool = True,
    ) -> None:
        self.theme: Dict[str, Any] = dict(DEFAULT_THEME)
        if theme:
            self.theme.update(theme)
        self.palette: List[str] = list(palette or CATEGORICAL_PALETTE)
        self.layout_config: Dict[str, Any] = self.get_responsive_config(layout)
        self.prefer_plotly = prefer_plotly

    # ------------------------------------------------------------------ helpers
    @property
    def backend(self) -> str:
        """Backend used for the next figure: ``"plotly"`` or ``"matplotlib"``."""
        return "plotly" if self.prefer_plotly and plotly_available() else "matplotlib"

    def _use_plotly(self) -> bool:
        if self.backend == "plotly":
            return True
        self.logger.info("Plotly unavailable; falling back to matplotlib")
        return False

    def _colors(self, n: int) -> List[str]:
        return series_colors(n, self.palette)

    def _layout(self, title: Optional[str] = None, **overrides: Any) -> Dict[str, Any]:
        """Build a Plotly layout dict from the theme and responsive preset."""
        cfg = self.layout_config
        margin = cfg["margin"]
        layout: Dict[str, Any] = {
            "title": {"text": title} if title else None,
            "width": cfg["width"],
            "height": cfg["height"],
            "plot_bgcolor": self.theme["background_color"],
            "paper_bgcolor": self.theme.get("paper_color", self.theme["background_color"]),
            "font": {
                "family": self.theme["font_family"],
                "color": self.theme["text_color"],
                "size": self.theme.get("font_size", cfg["font_size"]),
            },
            "margin": {"l": margin, "r": margin, "t": margin + 30, "b": margin},
            "legend": {"orientation": cfg["legend_orientation"]},
        }
        layout.update(overrides)
        return {k: v for k, v in layout.items() if v is not None}

    def _axis_style(self, title: Optional[str] = None) -> Dict[str, Any]:
        style: Dict[str, Any] = {
            "gridcolor": self.theme["grid_color"],
            "zerolinecolor": self.theme["grid_color"],
        }
        if title is not None:
            style["title"] = {"text": title}
        return style

    def _style_mpl_axes(
        self,
        ax: plt.Axes,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
    ) -> None:
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold")
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    def _mpl_figure(self, nrows: int = 1, ncols: int = 1, **kwargs: Any) -> Tuple[Figure, Any]:
        cfg = self.layout_config
        size = (cfg["width"] / 100.0, cfg["height"] / 100.0)
        fig, axes = plt.subplots(nrows, ncols, figsize=kwargs.pop("figsize", size), **kwargs)
        return fig, axes

    # ------------------------------------------------------------ configuration
    def apply_custom_theme(self, theme_config: Dict[str, Any]) -> InteractiveVisualizer:
        """Merge ``theme_config`` into the active theme and return ``self``.

        Recognised keys: ``background_color``, ``paper_color``, ``grid_color``,
        ``text_color``, ``font_family`` and ``font_size``. Unknown keys are
        stored and ignored by the built-in renderers.

        Args:
            theme_config: Mapping of theme keys to values.

        Returns:
            The visualizer itself, to allow chaining.

        Raises:
            TypeError: If ``theme_config`` is not a mapping.
        """
        if not isinstance(theme_config, dict):
            raise TypeError("theme_config must be a dict")
        self.theme.update(theme_config)
        return self

    def get_responsive_config(self, layout: str = "desktop") -> Dict[str, Any]:
        """Return the size/typography preset for a layout name.

        Args:
            layout: One of ``"mobile"``, ``"tablet"``, ``"desktop"``,
                ``"presentation"``.

        Returns:
            Dictionary with ``width``, ``height``, ``font_size``, ``margin``
            and ``legend_orientation`` entries.

        Raises:
            ValueError: If ``layout`` is unknown.
        """
        try:
            return dict(RESPONSIVE_LAYOUTS[layout])
        except KeyError as exc:
            raise ValueError(f"Unknown layout '{layout}'. Choose from {sorted(RESPONSIVE_LAYOUTS)}") from exc

    def set_layout(self, layout: str) -> InteractiveVisualizer:
        """Switch the responsive preset used for new figures.

        Args:
            layout: Preset name (see :meth:`get_responsive_config`).

        Returns:
            The visualizer itself.
        """
        self.layout_config = self.get_responsive_config(layout)
        return self

    # --------------------------------------------------------------- scatter
    def create_interactive_scatter(
        self,
        x: Any,
        y: Any,
        color: Any = None,
        x_label: str = "x",
        y_label: str = "y",
        color_label: str = "color",
        color_scheme: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
        hover_text: Optional[Sequence[str]] = None,
        marker_size: int = 8,
    ) -> Any:
        """Create a 2-D scatter plot optionally coloured by a third variable.

        Args:
            x: X coordinates.
            y: Y coordinates.
            color: Optional categorical labels or continuous values per point.
            x_label: X axis title.
            y_label: Y axis title.
            color_label: Legend / colourbar title.
            color_scheme: Custom colour list (categorical) or overrides the palette.
            title: Figure title.
            hover_text: Optional per-point hover strings.
            marker_size: Marker diameter in pixels.

        Returns:
            ``plotly.graph_objects.Figure`` or a matplotlib ``Figure``.

        Raises:
            ValueError: If ``x``, ``y`` and ``color`` lengths differ.
        """
        x = to_1d_array(x, "x")
        y = to_1d_array(y, "y")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same length")
        c = None if color is None else to_1d_array(color, "color")
        if c is not None and c.shape[0] != x.shape[0]:
            raise ValueError("color must have the same length as x")
        discrete = c is not None and is_discrete_target(c)
        palette = list(color_scheme) if color_scheme else self.palette

        if self._use_plotly():
            import plotly.graph_objects as go

            fig = go.Figure()
            if c is None:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="markers",
                        marker={"size": marker_size, "color": palette[0]},
                        text=hover_text,
                        name="points",
                    )
                )
            elif discrete:
                classes = np.unique(c)
                colors = series_colors(len(classes), palette)
                for cls, col in zip(classes, colors):
                    mask = c == cls
                    fig.add_trace(
                        go.Scatter(
                            x=x[mask],
                            y=y[mask],
                            mode="markers",
                            name=f"{color_label}={cls}",
                            marker={"size": marker_size, "color": col},
                            text=None if hover_text is None else np.asarray(hover_text)[mask],
                        )
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="markers",
                        name=color_label,
                        text=hover_text,
                        marker={
                            "size": marker_size,
                            "color": c,
                            "colorscale": SEQUENTIAL_CMAP,
                            "colorbar": {"title": color_label},
                        },
                    )
                )
            fig.update_layout(
                **self._layout(title, xaxis=self._axis_style(x_label), yaxis=self._axis_style(y_label))
            )
            return fig

        fig, ax = self._mpl_figure()
        if c is None:
            ax.scatter(x, y, s=marker_size**2, color=palette[0], alpha=0.8)
        elif discrete:
            classes = np.unique(c)
            for cls, col in zip(classes, series_colors(len(classes), palette)):
                mask = c == cls
                ax.scatter(
                    x[mask], y[mask], s=marker_size**2, color=col, alpha=0.8, label=f"{color_label}={cls}"
                )
            ax.legend()
        else:
            sc = ax.scatter(x, y, s=marker_size**2, c=c, cmap=SEQUENTIAL_CMAP, alpha=0.8)
            fig.colorbar(sc, ax=ax, label=color_label)
        self._style_mpl_axes(ax, title, x_label, y_label)
        return fig

    def create_3d_scatter(
        self,
        x: Any,
        y: Any,
        z: Any,
        color: Any = None,
        x_label: str = "x",
        y_label: str = "y",
        z_label: str = "z",
        color_label: str = "color",
        color_scheme: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
        marker_size: int = 4,
    ) -> Any:
        """Create a 3-D scatter plot.

        Args:
            x: X coordinates.
            y: Y coordinates.
            z: Z coordinates.
            color: Optional labels/values per point.
            x_label: X axis title.
            y_label: Y axis title.
            z_label: Z axis title.
            color_label: Legend title.
            color_scheme: Custom categorical colours.
            title: Figure title.
            marker_size: Marker size.

        Returns:
            Plotly figure or matplotlib figure with a 3-D axes.

        Raises:
            ValueError: If coordinate arrays differ in length.
        """
        x, y, z = (to_1d_array(v, n) for v, n in ((x, "x"), (y, "y"), (z, "z")))
        if not (x.shape[0] == y.shape[0] == z.shape[0]):
            raise ValueError("x, y and z must have the same length")
        c = None if color is None else to_1d_array(color, "color")
        discrete = c is not None and is_discrete_target(c)
        palette = list(color_scheme) if color_scheme else self.palette

        if self._use_plotly():
            import plotly.graph_objects as go

            fig = go.Figure()
            if c is None:
                fig.add_trace(
                    go.Scatter3d(
                        x=x, y=y, z=z, mode="markers", marker={"size": marker_size, "color": palette[0]}
                    )
                )
            elif discrete:
                classes = np.unique(c)
                for cls, col in zip(classes, series_colors(len(classes), palette)):
                    mask = c == cls
                    fig.add_trace(
                        go.Scatter3d(
                            x=x[mask],
                            y=y[mask],
                            z=z[mask],
                            mode="markers",
                            name=f"{color_label}={cls}",
                            marker={"size": marker_size, "color": col},
                        )
                    )
            else:
                fig.add_trace(
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode="markers",
                        marker={
                            "size": marker_size,
                            "color": c,
                            "colorscale": SEQUENTIAL_CMAP,
                            "colorbar": {"title": color_label},
                        },
                    )
                )
            fig.update_layout(
                **self._layout(
                    title,
                    scene={
                        "xaxis": {"title": x_label},
                        "yaxis": {"title": y_label},
                        "zaxis": {"title": z_label},
                    },
                )
            )
            return fig

        fig = plt.figure(figsize=(self.layout_config["width"] / 100.0, self.layout_config["height"] / 100.0))
        ax = fig.add_subplot(111, projection="3d")
        if c is None:
            ax.scatter(x, y, z, color=palette[0], s=marker_size**2)
        elif discrete:
            classes = np.unique(c)
            for cls, col in zip(classes, series_colors(len(classes), palette)):
                mask = c == cls
                ax.scatter(
                    x[mask], y[mask], z[mask], color=col, s=marker_size**2, label=f"{color_label}={cls}"
                )
            ax.legend()
        else:
            sc = ax.scatter(x, y, z, c=c, cmap=SEQUENTIAL_CMAP, s=marker_size**2)
            fig.colorbar(sc, ax=ax, label=color_label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        if title:
            ax.set_title(title)
        return fig

    # --------------------------------------------------------- data explorer
    def create_feature_explorer(
        self,
        X: Any,
        y: Any = None,
        feature_names: Optional[Sequence[str]] = None,
        bins: int = 30,
        title: str = "Feature explorer",
    ) -> Any:
        """Build a histogram explorer with a dropdown selecting the feature.

        When ``y`` is a discrete target, each feature's histogram is split by
        class. With Plotly the dropdown toggles trace visibility; the
        matplotlib fallback renders one panel per feature.

        Args:
            X: Feature matrix.
            y: Optional target.
            feature_names: Names for the columns of ``X``.
            bins: Number of histogram bins.
            title: Figure title.

        Returns:
            Plotly figure with ``updatemenus`` or a matplotlib figure.
        """
        X = to_2d_array(X)
        names = default_feature_names(X.shape[1], feature_names)
        yv = None if y is None else to_1d_array(y, "y")
        classes = np.unique(yv) if yv is not None and is_discrete_target(yv) else None
        colors = self._colors(1 if classes is None else len(classes))

        if self._use_plotly():
            import plotly.graph_objects as go

            fig = go.Figure()
            traces_per_feature = 1 if classes is None else len(classes)
            for j, name in enumerate(names):
                if classes is None:
                    fig.add_trace(
                        go.Histogram(
                            x=X[:, j], nbinsx=bins, name=name, marker={"color": colors[0]}, visible=j == 0
                        )
                    )
                else:
                    for k, cls in enumerate(classes):
                        fig.add_trace(
                            go.Histogram(
                                x=X[yv == cls, j],
                                nbinsx=bins,
                                name=f"class {cls}",
                                marker={"color": colors[k]},
                                opacity=0.65,
                                visible=j == 0,
                            )
                        )
            buttons = []
            n_traces = len(names) * traces_per_feature
            for j, name in enumerate(names):
                visible = [False] * n_traces
                for k in range(traces_per_feature):
                    visible[j * traces_per_feature + k] = True
                buttons.append(
                    {
                        "label": name,
                        "method": "update",
                        "args": [{"visible": visible}, {"xaxis.title.text": name}],
                    }
                )
            fig.update_layout(
                **self._layout(
                    title,
                    barmode="overlay",
                    xaxis=self._axis_style(names[0]),
                    yaxis=self._axis_style("count"),
                    updatemenus=[
                        {"buttons": buttons, "direction": "down", "x": 0.0, "y": 1.15, "showactive": True}
                    ],
                )
            )
            return fig

        fig, axes = subplot_grid(len(names))
        for ax, name, j in zip(axes, names, range(len(names))):
            if classes is None:
                ax.hist(X[:, j], bins=bins, color=colors[0], alpha=0.85)
            else:
                for k, cls in enumerate(classes):
                    ax.hist(X[yv == cls, j], bins=bins, color=colors[k], alpha=0.55, label=f"class {cls}")
            self._style_mpl_axes(ax, name)
        if classes is not None:
            axes[0].legend(fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        return fig

    # ----------------------------------------------------- model comparison
    def create_model_comparison_dashboard(
        self,
        models_data: Dict[str, Dict[str, Any]],
        metrics: Optional[Sequence[str]] = None,
        title: str = "Model comparison",
    ) -> Any:
        """Create one panel per metric showing the score distribution per model.

        Args:
            models_data: ``{model_name: {metric_name: scores}}`` where
                ``scores`` is a scalar or a sequence of cross-validation scores.
            metrics: Subset/order of metrics to plot; defaults to all found.
            title: Figure title.

        Returns:
            Plotly subplot figure (box plots) or matplotlib figure.

        Raises:
            ValueError: If ``models_data`` is empty.
        """
        if not models_data:
            raise ValueError("models_data is empty")
        model_names = list(models_data)
        if metrics is None:
            metrics = list(dict.fromkeys(m for d in models_data.values() for m in d))
        metrics = list(metrics)
        colors = self._colors(len(model_names))

        def _scores(model: str, metric: str) -> np.ndarray:
            return np.atleast_1d(np.asarray(models_data[model].get(metric, []), dtype=float))

        if self._use_plotly():
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=metrics, shared_yaxes=False)
            for col, metric in enumerate(metrics, start=1):
                for model, color in zip(model_names, colors):
                    s = _scores(model, metric)
                    if s.size == 0:
                        continue
                    trace = go.Box(
                        y=s,
                        name=model,
                        marker={"color": color},
                        legendgroup=model,
                        showlegend=col == 1,
                        boxmean=True,
                    )
                    fig.add_trace(trace, row=1, col=col)
            fig.update_layout(**self._layout(title, boxmode="group"))
            fig.update_yaxes(**self._axis_style())
            fig.update_xaxes(**self._axis_style())
            return fig

        fig, axes = self._mpl_figure(1, len(metrics), squeeze=False)
        for ax, metric in zip(axes[0], metrics):
            data = [_scores(m, metric) for m in model_names]
            box = ax.boxplot(data, patch_artist=True, showmeans=True)
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_xticks(range(1, len(model_names) + 1))
            ax.set_xticklabels(model_names, rotation=30, ha="right")
            self._style_mpl_axes(ax, metric)
        fig.suptitle(title)
        fig.tight_layout()
        return fig

    def create_interactive_learning_curves(
        self,
        train_sizes: Any,
        models_curves: Dict[str, Dict[str, Any]],
        title: str = "Learning curves",
        score_label: str = "score",
    ) -> Any:
        """Plot training/validation learning curves for several models.

        Args:
            train_sizes: Training-set sizes (x axis).
            models_curves: ``{model: {"train_scores_mean", "train_scores_std",
                "val_scores_mean", "val_scores_std"}}``; ``*_std`` keys are optional.
            title: Figure title.
            score_label: Y axis title.

        Returns:
            Plotly figure or matplotlib figure.

        Raises:
            ValueError: If a curve's length differs from ``train_sizes``.
        """
        sizes = to_1d_array(train_sizes, "train_sizes")
        colors = self._colors(len(models_curves))
        curves: List[Tuple[str, str, np.ndarray, Optional[np.ndarray]]] = []
        for (model, data), color in zip(models_curves.items(), colors):
            for split in ("train", "val"):
                mean_key = f"{split}_scores_mean"
                if mean_key not in data:
                    raise ValueError(f"'{model}' is missing '{mean_key}'")
                mean = to_1d_array(data[mean_key], mean_key)
                if mean.shape[0] != sizes.shape[0]:
                    raise ValueError(
                        f"'{model}' {mean_key} length {mean.shape[0]} != train_sizes length {sizes.shape[0]}"
                    )
                std = data.get(f"{split}_scores_std")
                curves.append((model, split, mean, None if std is None else to_1d_array(std)))
        model_color = dict(zip(models_curves, colors))

        if self._use_plotly():
            import plotly.graph_objects as go

            fig = go.Figure()
            for model, split, mean, std in curves:
                color = model_color[model]
                dash = "solid" if split == "train" else "dash"
                label = f"{model} ({'train' if split == 'train' else 'validation'})"
                if std is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=np.concatenate([sizes, sizes[::-1]]),
                            y=np.concatenate([mean + std, (mean - std)[::-1]]),
                            fill="toself",
                            fillcolor=color,
                            opacity=0.12,
                            line={"width": 0},
                            hoverinfo="skip",
                            showlegend=False,
                            legendgroup=model,
                        )
                    )
                fig.add_trace(
                    go.Scatter(
                        x=sizes,
                        y=mean,
                        mode="lines+markers",
                        name=label,
                        legendgroup=model,
                        line={"color": color, "dash": dash, "width": 2},
                    )
                )
            fig.update_layout(
                **self._layout(
                    title, xaxis=self._axis_style("training set size"), yaxis=self._axis_style(score_label)
                )
            )
            return fig

        fig, ax = self._mpl_figure()
        for model, split, mean, std in curves:
            color = model_color[model]
            ls = "-" if split == "train" else "--"
            ax.plot(
                sizes,
                mean,
                ls,
                color=color,
                marker="o",
                linewidth=2,
                label=f"{model} ({'train' if split == 'train' else 'validation'})",
            )
            if std is not None:
                ax.fill_between(sizes, mean - std, mean + std, color=color, alpha=0.12)
        ax.legend()
        self._style_mpl_axes(ax, title, "training set size", score_label)
        fig.tight_layout()
        return fig

    def create_interactive_feature_importance(
        self,
        importance_data: Union[Dict[str, Any], Any],
        feature_names: Optional[Sequence[str]] = None,
        top_n: Optional[int] = None,
        title: str = "Feature importance",
    ) -> Any:
        """Plot (grouped) horizontal bars of feature importance per model.

        Args:
            importance_data: ``{model: importances}`` or a single importance array.
            feature_names: Feature labels.
            top_n: Keep only the ``top_n`` features by mean importance.
            title: Figure title.

        Returns:
            Plotly figure or matplotlib figure.

        Raises:
            ValueError: If importance vectors have inconsistent lengths.
        """
        if not isinstance(importance_data, dict):
            importance_data = {"model": importance_data}
        matrix = {m: to_1d_array(v, m).astype(float) for m, v in importance_data.items()}
        lengths = {v.shape[0] for v in matrix.values()}
        if len(lengths) != 1:
            raise ValueError("All importance vectors must have the same length")
        n_features = lengths.pop()
        names = default_feature_names(n_features, feature_names)
        mean_imp = np.mean(np.vstack(list(matrix.values())), axis=0)
        order = np.argsort(mean_imp)[::-1]
        if top_n is not None:
            order = order[:top_n]
        order = order[::-1]  # ascending so that the best feature sits on top
        labels = [names[i] for i in order]
        colors = self._colors(len(matrix))

        if self._use_plotly():
            import plotly.graph_objects as go

            fig = go.Figure()
            for (model, imp), color in zip(matrix.items(), colors):
                fig.add_trace(
                    go.Bar(y=labels, x=imp[order], orientation="h", name=model, marker={"color": color})
                )
            fig.update_layout(
                **self._layout(
                    title,
                    barmode="group",
                    xaxis=self._axis_style("importance"),
                    yaxis=self._axis_style("feature"),
                )
            )
            return fig

        fig, ax = self._mpl_figure()
        n_models = len(matrix)
        height = 0.8 / n_models
        pos = np.arange(len(labels))
        for k, ((model, imp), color) in enumerate(zip(matrix.items(), colors)):
            ax.barh(
                pos + (k - (n_models - 1) / 2) * height, imp[order], height=height, color=color, label=model
            )
        ax.set_yticks(pos)
        ax.set_yticklabels(labels)
        if n_models > 1:
            ax.legend()
        self._style_mpl_axes(ax, title, "importance", "feature")
        fig.tight_layout()
        return fig

    def create_interactive_confusion_matrix(
        self,
        cm_data: Union[Dict[str, Any], Any],
        class_names: Optional[Sequence[str]] = None,
        normalize: bool = False,
        title: str = "Confusion matrices",
    ) -> Any:
        """Draw one annotated heatmap per confusion matrix.

        Args:
            cm_data: ``{model: confusion_matrix}`` or a single square matrix.
            class_names: Labels for the classes.
            normalize: Row-normalise each matrix to rates.
            title: Figure title.

        Returns:
            Plotly subplot figure or matplotlib figure.

        Raises:
            ValueError: If a matrix is not square or shapes differ.
        """
        if not isinstance(cm_data, dict):
            cm_data = {"model": cm_data}
        matrices: Dict[str, np.ndarray] = {}
        for name, cm in cm_data.items():
            arr = np.asarray(cm, dtype=float)
            if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                raise ValueError(f"Confusion matrix for '{name}' must be square, got {arr.shape}")
            if normalize:
                arr = arr / np.clip(arr.sum(axis=1, keepdims=True), 1e-12, None)
            matrices[name] = arr
        n_classes = {m.shape[0] for m in matrices.values()}
        if len(n_classes) != 1:
            raise ValueError("All confusion matrices must have the same number of classes")
        k = n_classes.pop()
        labels = [str(c) for c in (class_names if class_names is not None else range(k))]
        if len(labels) != k:
            raise ValueError(f"Expected {k} class names, got {len(labels)}")
        fmt = ".2f" if normalize else ".0f"

        if self._use_plotly():
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=1, cols=len(matrices), subplot_titles=list(matrices), horizontal_spacing=0.08
            )
            for col, (name, cm) in enumerate(matrices.items(), start=1):
                fig.add_trace(
                    go.Heatmap(
                        z=cm,
                        x=labels,
                        y=labels,
                        colorscale=SEQUENTIAL_CMAP,
                        showscale=col == len(matrices),
                        text=[[format(v, fmt) for v in row] for row in cm],
                        texttemplate="%{text}",
                        hovertemplate="true=%{y}<br>pred=%{x}<br>value=%{z}<extra></extra>",
                    ),
                    row=1,
                    col=col,
                )
                fig.update_xaxes(title_text="predicted", row=1, col=col)
                fig.update_yaxes(title_text="true", autorange="reversed", row=1, col=col)
            fig.update_layout(**self._layout(title))
            return fig

        fig, axes = self._mpl_figure(1, len(matrices), squeeze=False)
        for ax, (name, cm) in zip(axes[0], matrices.items()):
            im = ax.imshow(cm, cmap=SEQUENTIAL_CMAP)
            ax.set_xticks(range(k))
            ax.set_yticks(range(k))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)
            thresh = cm.max() / 2.0 if cm.size else 0
            for i in range(k):
                for j in range(k):
                    ax.text(
                        j,
                        i,
                        format(cm[i, j], fmt),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=9,
                    )
            ax.set_title(name)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
        fig.suptitle(title)
        return fig

    def create_interactive_roc_curves(
        self, roc_data: Dict[str, Dict[str, Any]], title: str = "ROC curves"
    ) -> Any:
        """Overlay ROC curves for several models with a chance diagonal.

        Args:
            roc_data: ``{model: {"fpr": ..., "tpr": ..., "auc": float}}``;
                ``auc`` is computed with the trapezoid rule when missing.
            title: Figure title.

        Returns:
            Plotly figure or matplotlib figure.

        Raises:
            ValueError: If ``roc_data`` is empty or ``fpr``/``tpr`` lengths differ.
        """
        if not roc_data:
            raise ValueError("roc_data is empty")
        colors = self._colors(len(roc_data))
        series = []
        for (model, data), color in zip(roc_data.items(), colors):
            fpr = to_1d_array(data["fpr"], "fpr")
            tpr = to_1d_array(data["tpr"], "tpr")
            if fpr.shape != tpr.shape:
                raise ValueError(f"fpr/tpr length mismatch for '{model}'")
            score = data.get("auc")
            if score is None:
                score = float(trapezoid(tpr, fpr))
            series.append((f"{model} (AUC = {score:.3f})", fpr, tpr, color))

        if self._use_plotly():
            import plotly.graph_objects as go

            fig = go.Figure()
            for label, fpr, tpr, color in series:
                fig.add_trace(
                    go.Scatter(x=fpr, y=tpr, mode="lines", name=label, line={"color": color, "width": 2})
                )
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="chance",
                    line={"color": REFERENCE_COLOR, "dash": "dot", "width": 1},
                )
            )
            fig.update_layout(
                **self._layout(
                    title,
                    xaxis=self._axis_style("false positive rate") | {"range": [0, 1]},
                    yaxis=self._axis_style("true positive rate") | {"range": [0, 1.02]},
                )
            )
            return fig

        fig, ax = self._mpl_figure()
        for label, fpr, tpr, color in series:
            ax.plot(fpr, tpr, color=color, linewidth=2, label=label)
        ax.plot([0, 1], [0, 1], linestyle=":", color=REFERENCE_COLOR, linewidth=1, label="chance")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right")
        self._style_mpl_axes(ax, title, "false positive rate", "true positive rate")
        fig.tight_layout()
        return fig

    def create_interactive_hyperparameter_heatmap(
        self,
        results: Union[Sequence[Dict[str, Any]], pd.DataFrame],
        x_param: str,
        y_param: str,
        score_key: str = "score",
        aggregate: str = "mean",
        title: Optional[str] = None,
    ) -> Any:
        """Pivot tuning results into a ``y_param x x_param`` score heatmap.

        Args:
            results: Records with at least ``x_param``, ``y_param`` and ``score_key``.
            x_param: Parameter placed on the x axis.
            y_param: Parameter placed on the y axis.
            score_key: Name of the score field.
            aggregate: Pandas aggregation applied to duplicate cells.
            title: Figure title.

        Returns:
            Plotly heatmap or matplotlib figure.

        Raises:
            ValueError: If required keys are missing.
        """
        df = results.copy() if isinstance(results, pd.DataFrame) else pd.DataFrame(list(results))
        for key in (x_param, y_param, score_key):
            if key not in df.columns:
                raise ValueError(f"results is missing '{key}'")
        x_vals = sorted(df[x_param].unique().tolist(), key=_sort_key)
        y_vals = sorted(df[y_param].unique().tolist(), key=_sort_key)
        pivot = df.pivot_table(index=y_param, columns=x_param, values=score_key, aggfunc=aggregate)
        pivot = pivot.reindex(index=y_vals, columns=x_vals)
        z = pivot.to_numpy(dtype=float)
        x_labels = [str(v) for v in x_vals]
        y_labels = [str(v) for v in y_vals]
        title = title or f"{score_key} by {x_param} and {y_param}"

        if self._use_plotly():
            import plotly.graph_objects as go

            fig = go.Figure(
                go.Heatmap(
                    z=z,
                    x=x_labels,
                    y=y_labels,
                    colorscale=SEQUENTIAL_CMAP,
                    text=[["" if np.isnan(v) else f"{v:.3f}" for v in row] for row in z],
                    texttemplate="%{text}",
                    colorbar={"title": score_key},
                    hovertemplate=f"{x_param}=%{{x}}<br>{y_param}=%{{y}}<br>{score_key}=%{{z:.4f}}<extra></extra>",
                )
            )
            fig.update_layout(
                **self._layout(
                    title,
                    xaxis={"title": x_param, "type": "category"},
                    yaxis={"title": y_param, "type": "category"},
                )
            )
            return fig

        fig, ax = self._mpl_figure()
        im = ax.imshow(z, cmap=SEQUENTIAL_CMAP, aspect="auto")
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
        finite = z[np.isfinite(z)]
        thresh = (finite.max() + finite.min()) / 2 if finite.size else 0
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                if np.isfinite(z[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{z[i, j]:.3f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if z[i, j] > thresh else "black",
                    )
        fig.colorbar(im, ax=ax, label=score_key)
        ax.set_title(title)
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        fig.tight_layout()
        return fig

    # ---------------------------------------------------------- animations
    def create_animated_learning_curves(
        self,
        epochs: Any,
        train_scores: Any,
        val_scores: Any,
        title: str = "Training progress",
        train_label: str = "train",
        val_label: str = "validation",
        score_label: str = "score",
    ) -> Any:
        """Animate epoch-wise training and validation scores.

        With Plotly the figure carries one frame per epoch and play/slider
        controls; matplotlib renders the final curves statically.

        Args:
            epochs: Epoch indices.
            train_scores: Training score per epoch.
            val_scores: Validation score per epoch.
            title: Figure title.
            train_label: Legend label for the training series.
            val_label: Legend label for the validation series.
            score_label: Y axis title.

        Returns:
            Plotly figure or matplotlib figure.

        Raises:
            ValueError: If the sequences have different lengths.
        """
        ep = to_1d_array(epochs, "epochs")
        tr = to_1d_array(train_scores, "train_scores").astype(float)
        va = to_1d_array(val_scores, "val_scores").astype(float)
        if not (ep.shape == tr.shape == va.shape):
            raise ValueError("epochs, train_scores and val_scores must have the same length")
        c_train, c_val = self._colors(2)
        y_min, y_max = float(np.nanmin([tr.min(), va.min()])), float(np.nanmax([tr.max(), va.max()]))
        pad = 0.05 * (y_max - y_min or 1.0)

        if self._use_plotly():
            import plotly.graph_objects as go

            def traces(i: int) -> List[Any]:
                return [
                    go.Scatter(
                        x=ep[: i + 1],
                        y=tr[: i + 1],
                        mode="lines",
                        name=train_label,
                        line={"color": c_train, "width": 2},
                    ),
                    go.Scatter(
                        x=ep[: i + 1],
                        y=va[: i + 1],
                        mode="lines",
                        name=val_label,
                        line={"color": c_val, "width": 2, "dash": "dash"},
                    ),
                ]

            frames = [go.Frame(data=traces(i), name=str(ep[i])) for i in range(len(ep))]
            fig = go.Figure(data=traces(len(ep) - 1), frames=frames)
            steps = [
                {
                    "method": "animate",
                    "label": str(ep[i]),
                    "args": [[str(ep[i])], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
                }
                for i in range(len(ep))
            ]
            fig.update_layout(
                **self._layout(
                    title,
                    xaxis=self._axis_style("epoch") | {"range": [float(ep.min()), float(ep.max())]},
                    yaxis=self._axis_style(score_label) | {"range": [y_min - pad, y_max + pad]},
                    updatemenus=[
                        {
                            "type": "buttons",
                            "showactive": False,
                            "x": 0.0,
                            "y": 1.15,
                            "buttons": [
                                {
                                    "label": "Play",
                                    "method": "animate",
                                    "args": [
                                        None,
                                        {"frame": {"duration": 60, "redraw": False}, "fromcurrent": True},
                                    ],
                                },
                                {
                                    "label": "Pause",
                                    "method": "animate",
                                    "args": [
                                        [None],
                                        {"mode": "immediate", "frame": {"duration": 0, "redraw": False}},
                                    ],
                                },
                            ],
                        }
                    ],
                    sliders=[{"steps": steps, "currentvalue": {"prefix": "epoch: "}}],
                )
            )
            return fig

        self.logger.warning("Animation requires plotly; rendering static learning curves")
        fig, ax = self._mpl_figure()
        ax.plot(ep, tr, color=c_train, linewidth=2, label=train_label)
        ax.plot(ep, va, color=c_val, linewidth=2, linestyle="--", label=val_label)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.legend()
        self._style_mpl_axes(ax, title, "epoch", score_label)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------- networks
    def create_correlation_network(
        self,
        corr_matrix: Any,
        feature_names: Optional[Sequence[str]] = None,
        threshold: float = 0.5,
        title: str = "Correlation network",
    ) -> Any:
        """Draw features as nodes on a circle, linking pairs with ``|r| >= threshold``.

        Edge width scales with ``|r|``; positive and negative correlations use
        the diverging pair of colours.

        Args:
            corr_matrix: Square correlation matrix (array or DataFrame).
            feature_names: Node labels (taken from DataFrame columns if omitted).
            threshold: Minimum absolute correlation to draw an edge.
            title: Figure title.

        Returns:
            Plotly figure or matplotlib figure.

        Raises:
            ValueError: If the matrix is not square or the threshold is invalid.
        """
        if isinstance(corr_matrix, pd.DataFrame) and feature_names is None:
            feature_names = [str(c) for c in corr_matrix.columns]
        corr = np.asarray(corr_matrix, dtype=float)
        if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
            raise ValueError(f"corr_matrix must be square, got {corr.shape}")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must lie in [0, 1]")
        n = corr.shape[0]
        names = default_feature_names(n, feature_names)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs, ys = np.cos(angles), np.sin(angles)
        edges = [
            (i, j, corr[i, j]) for i in range(n) for j in range(i + 1, n) if abs(corr[i, j]) >= threshold
        ]
        degree = np.zeros(n)
        for i, j, _ in edges:
            degree[i] += 1
            degree[j] += 1

        if self._use_plotly():
            import plotly.graph_objects as go

            fig = go.Figure()
            for i, j, r in edges:
                fig.add_trace(
                    go.Scatter(
                        x=[xs[i], xs[j]],
                        y=[ys[i], ys[j]],
                        mode="lines",
                        line={"width": 1 + 6 * abs(r), "color": POSITIVE_COLOR if r > 0 else NEGATIVE_COLOR},
                        opacity=0.7,
                        hoverinfo="text",
                        text=f"{names[i]} - {names[j]}: r = {r:.2f}",
                        showlegend=False,
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers+text",
                    text=names,
                    textposition="top center",
                    marker={
                        "size": 14 + 4 * degree,
                        "color": self.palette[0],
                        "line": {"color": "white", "width": 2},
                    },
                    hovertemplate="%{text}<br>links: %{marker.size}<extra></extra>",
                    name="features",
                )
            )
            axis = {"visible": False, "range": [-1.4, 1.4]}
            fig.update_layout(
                **self._layout(title, xaxis=axis, yaxis=axis | {"scaleanchor": "x"}, showlegend=False)
            )
            return fig

        fig, ax = self._mpl_figure()
        for i, j, r in edges:
            ax.plot(
                [xs[i], xs[j]],
                [ys[i], ys[j]],
                color=POSITIVE_COLOR if r > 0 else NEGATIVE_COLOR,
                linewidth=1 + 4 * abs(r),
                alpha=0.7,
            )
        ax.scatter(
            xs, ys, s=150 + 40 * degree, color=self.palette[0], edgecolor="white", linewidth=2, zorder=3
        )
        for name, x, y in zip(names, xs, ys):
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.axis("off")
        ax.set_title(title)
        return fig

    def create_interactive_decision_tree(
        self,
        model: Any,
        feature_names: Optional[Sequence[str]] = None,
        class_names: Optional[Sequence[str]] = None,
        title: str = "Decision tree",
    ) -> Any:
        """Render a fitted decision tree as a node-link diagram with hover details.

        Args:
            model: Fitted ``DecisionTreeClassifier``/``DecisionTreeRegressor``.
            feature_names: Feature labels used in split descriptions.
            class_names: Class labels for leaf descriptions (classifiers).
            title: Figure title.

        Returns:
            Plotly figure or a matplotlib figure from :func:`sklearn.tree.plot_tree`.

        Raises:
            AttributeError: If ``model`` has no fitted ``tree_`` attribute.
        """
        if not hasattr(model, "tree_"):
            raise AttributeError("model must be a fitted sklearn decision tree (missing 'tree_')")
        tree = model.tree_
        names = default_feature_names(tree.n_features, feature_names)
        is_clf = hasattr(model, "classes_")
        if is_clf:
            classes = [str(c) for c in (class_names if class_names is not None else model.classes_)]

        # Layout: leaves get consecutive x positions, internal nodes sit above their children's mean.
        x_pos = np.zeros(tree.node_count)
        depth = np.zeros(tree.node_count, dtype=int)
        counter = [0]

        def place(node: int, d: int) -> float:
            depth[node] = d
            left, right = tree.children_left[node], tree.children_right[node]
            if left == -1:
                x_pos[node] = counter[0]
                counter[0] += 1
            else:
                x_pos[node] = (place(left, d + 1) + place(right, d + 1)) / 2.0
            return x_pos[node]

        place(0, 0)
        y_pos = -depth.astype(float)

        def describe(node: int) -> str:
            n = int(tree.n_node_samples[node])
            impurity = float(tree.impurity[node])
            if tree.children_left[node] == -1:
                head = "leaf"
            else:
                head = f"{names[tree.feature[node]]} <= {tree.threshold[node]:.3f}"
            if is_clf:
                value = tree.value[node][0]
                pred = classes[int(np.argmax(value))]
                detail = f"class = {pred}"
            else:
                detail = f"value = {tree.value[node][0][0]:.3f}"
            return f"{head}<br>samples = {n}<br>impurity = {impurity:.3f}<br>{detail}"

        labels = [describe(i) for i in range(tree.node_count)]
        is_leaf = tree.children_left == -1

        if self._use_plotly():
            import plotly.graph_objects as go

            edge_x: List[Optional[float]] = []
            edge_y: List[Optional[float]] = []
            for node in range(tree.node_count):
                for child in (tree.children_left[node], tree.children_right[node]):
                    if child != -1:
                        edge_x += [x_pos[node], x_pos[child], None]
                        edge_y += [y_pos[node], y_pos[child], None]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line={"color": REFERENCE_COLOR, "width": 1},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_pos,
                    y=y_pos,
                    mode="markers+text",
                    text=[lbl.split("<br>")[0] for lbl in labels],
                    textposition="bottom center",
                    hovertext=labels,
                    hoverinfo="text",
                    marker={
                        "size": 16,
                        "color": [self.palette[2] if leaf else self.palette[0] for leaf in is_leaf],
                        "line": {"color": "white", "width": 1},
                    },
                    showlegend=False,
                )
            )
            fig.update_layout(**self._layout(title, xaxis={"visible": False}, yaxis={"visible": False}))
            return fig

        from sklearn.tree import plot_tree

        fig, ax = self._mpl_figure()
        plot_tree(
            model,
            feature_names=names,
            class_names=classes if is_clf else None,
            filled=True,
            ax=ax,
            fontsize=8,
        )
        ax.set_title(title)
        return fig

    def create_monitoring_dashboard(
        self,
        performance_data: Union[Dict[str, Any], pd.DataFrame],
        timestamp_key: str = "timestamp",
        metrics: Optional[Sequence[str]] = None,
        rolling_window: Optional[int] = None,
        title: str = "Model monitoring",
    ) -> Any:
        """Stack one time-series panel per monitored metric.

        Args:
            performance_data: Mapping/DataFrame with a timestamp column and
                one column per metric.
            timestamp_key: Name of the timestamp column.
            metrics: Metrics to display (default: all non-timestamp columns).
            rolling_window: Optional window for an overlaid rolling mean.
            title: Figure title.

        Returns:
            Plotly subplot figure or matplotlib figure.

        Raises:
            ValueError: If no metric columns are available.
        """
        df = (
            performance_data.copy()
            if isinstance(performance_data, pd.DataFrame)
            else pd.DataFrame(performance_data)
        )
        if timestamp_key in df.columns:
            t = df[timestamp_key]
        else:
            t = pd.Series(np.arange(len(df)), name=timestamp_key)
        metrics = list(metrics) if metrics is not None else [c for c in df.columns if c != timestamp_key]
        if not metrics:
            raise ValueError("No metric columns to plot")
        colors = self._colors(len(metrics))

        if self._use_plotly():
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=len(metrics), cols=1, shared_xaxes=True, subplot_titles=metrics, vertical_spacing=0.08
            )
            for row, (metric, color) in enumerate(zip(metrics, colors), start=1):
                fig.add_trace(
                    go.Scatter(
                        x=t, y=df[metric], mode="lines", name=metric, line={"color": color, "width": 1.5}
                    ),
                    row=row,
                    col=1,
                )
                if rolling_window:
                    fig.add_trace(
                        go.Scatter(
                            x=t,
                            y=df[metric].rolling(rolling_window, min_periods=1).mean(),
                            mode="lines",
                            name=f"{metric} (rolling)",
                            line={"color": color, "width": 2.5, "dash": "dot"},
                        ),
                        row=row,
                        col=1,
                    )
            layout = self._layout(title)
            layout["height"] = max(layout["height"], 220 * len(metrics))
            fig.update_layout(**layout)
            fig.update_xaxes(**self._axis_style())
            fig.update_yaxes(**self._axis_style())
            return fig

        fig, axes = self._mpl_figure(len(metrics), 1, sharex=True, squeeze=False)
        for ax, metric, color in zip(axes[:, 0], metrics, colors):
            ax.plot(t, df[metric], color=color, linewidth=1.2, label=metric)
            if rolling_window:
                ax.plot(
                    t,
                    df[metric].rolling(rolling_window, min_periods=1).mean(),
                    color=color,
                    linewidth=2,
                    linestyle=":",
                    label=f"{metric} (rolling)",
                )
            self._style_mpl_axes(ax, metric)
        axes[-1, 0].set_xlabel(timestamp_key)
        fig.suptitle(title)
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig

    # --------------------------------------------------------------- reports
    def create_comprehensive_report(
        self,
        models_results: Dict[str, Dict[str, Any]],
        X_test: Any,
        y_test: Any,
        feature_names: Optional[Sequence[str]] = None,
        class_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Assemble a dictionary of figures summarising several fitted models.

        Args:
            models_results: ``{name: {"model": estimator, "predictions": y_pred,
                "probabilities": y_proba | None, "test_score": float}}``.
                Missing predictions are computed from the estimator.
            X_test: Held-out features.
            y_test: Held-out targets.
            feature_names: Feature labels.
            class_names: Class labels.

        Returns:
            Dictionary with ``summary`` (DataFrame) and figure entries
            ``model_comparison``, ``confusion_matrices``, ``roc_curves``
            (when probabilities exist), ``feature_importance`` (when any model
            exposes importances) and ``feature_explorer``.

        Raises:
            ValueError: If ``models_results`` is empty.
        """
        if not models_results:
            raise ValueError("models_results is empty")
        X = to_2d_array(X_test, "X_test")
        y = to_1d_array(y_test, "y_test")
        names = default_feature_names(X.shape[1], feature_names, X_test)
        classes = np.unique(y)
        report: Dict[str, Any] = {}

        summary_rows, cms, rocs, importances = {}, {}, {}, {}
        for model_name, res in models_results.items():
            model = res.get("model")
            y_pred = res.get("predictions")
            if y_pred is None and model is not None:
                y_pred = model.predict(X)
            y_pred = to_1d_array(y_pred, "predictions")
            score = res.get("test_score")
            if score is None and model is not None:
                score = float(model.score(X, y))
            summary_rows[model_name] = {"test_score": score, "accuracy": float(np.mean(y_pred == y))}
            cms[model_name] = confusion_matrix(y, y_pred, labels=classes)
            proba = res.get("probabilities")
            if proba is None and model is not None and hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
            if proba is not None:
                proba = np.asarray(proba)
                if len(classes) == 2:
                    pos = proba[:, 1] if proba.ndim == 2 else proba
                    fpr, tpr, _ = roc_curve(y, pos, pos_label=classes[1])
                else:
                    y_bin = label_binarize(y, classes=classes)
                    fpr, tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
                rocs[model_name] = {"fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr))}
            imp = self._extract_importance(model)
            if imp is not None and imp.shape[0] == X.shape[1]:
                importances[model_name] = imp

        summary = pd.DataFrame.from_dict(summary_rows, orient="index")
        summary.index.name = "model"
        report["summary"] = summary
        report["model_comparison"] = self.create_model_comparison_dashboard(
            {m: {k: v for k, v in row.items() if v is not None} for m, row in summary_rows.items()},
            title="Test performance",
        )
        report["confusion_matrices"] = self.create_interactive_confusion_matrix(
            cms, class_names=class_names or [str(c) for c in classes]
        )
        if rocs:
            report["roc_curves"] = self.create_interactive_roc_curves(
                rocs, title="ROC curves" + ("" if len(classes) == 2 else " (micro-average)")
            )
        if importances:
            report["feature_importance"] = self.create_interactive_feature_importance(importances, names)
        report["feature_explorer"] = self.create_feature_explorer(X, y, names)
        return report

    @staticmethod
    def _extract_importance(model: Any) -> Optional[np.ndarray]:
        """Return a per-feature importance vector for tree or linear models."""
        if model is None:
            return None
        if hasattr(model, "feature_importances_"):
            return np.asarray(model.feature_importances_, dtype=float)
        if hasattr(model, "coef_"):
            coef = np.asarray(model.coef_, dtype=float)
            return np.abs(coef).mean(axis=0) if coef.ndim == 2 else np.abs(coef)
        return None

    def create_model_explainer(
        self,
        model: Any,
        X: Any,
        y: Any,
        feature_names: Optional[Sequence[str]] = None,
        class_names: Optional[Sequence[str]] = None,
        n_repeats: int = 5,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build global-explanation figures for a fitted model.

        Importance comes from ``feature_importances_``/``coef_`` when present
        and from :func:`sklearn.inspection.permutation_importance` otherwise.

        Args:
            model: Fitted estimator.
            X: Evaluation features.
            y: Evaluation targets.
            feature_names: Feature labels.
            class_names: Class labels (classification).
            n_repeats: Permutation repeats when importances are computed.
            random_state: Seed for permutation importance.

        Returns:
            Dictionary with ``importance`` (DataFrame), ``importance_source``,
            ``feature_importance`` (figure), ``performance`` (confusion matrix
            or prediction-vs-actual figure) and ``top_features`` (scatter of
            the two most important features coloured by prediction).

        Raises:
            AttributeError: If ``model`` cannot ``predict``.
        """
        if not hasattr(model, "predict"):
            raise AttributeError("model must implement predict()")
        Xa = to_2d_array(X)
        ya = to_1d_array(y, "y")
        names = default_feature_names(Xa.shape[1], feature_names, X)
        imp = self._extract_importance(model)
        source = "model"
        if imp is None or imp.shape[0] != Xa.shape[1]:
            from sklearn.inspection import permutation_importance

            imp = permutation_importance(
                model, Xa, ya, n_repeats=n_repeats, random_state=random_state
            ).importances_mean
            source = "permutation"
        importance = (
            pd.DataFrame({"feature": names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        y_pred = to_1d_array(model.predict(Xa), "predictions")
        is_clf = hasattr(model, "classes_") or is_discrete_target(ya)

        result: Dict[str, Any] = {
            "importance": importance,
            "importance_source": source,
            "feature_importance": self.create_interactive_feature_importance(
                {type(model).__name__: imp}, names, title="Feature importance"
            ),
        }
        if is_clf:
            classes = np.unique(np.concatenate([ya, y_pred]))
            cm = confusion_matrix(ya, y_pred, labels=classes)
            result["performance"] = self.create_interactive_confusion_matrix(
                {type(model).__name__: cm}, class_names=class_names or [str(c) for c in classes]
            )
        else:
            result["performance"] = self.create_interactive_scatter(
                ya, y_pred, x_label="actual", y_label="predicted", title="Prediction vs actual"
            )
        top = importance["feature"].head(2).tolist()
        idx = [names.index(t) for t in top]
        if len(idx) == 2:
            result["top_features"] = self.create_interactive_scatter(
                Xa[:, idx[0]],
                Xa[:, idx[1]],
                y_pred,
                x_label=top[0],
                y_label=top[1],
                color_label="prediction",
                title="Top features vs prediction",
            )
        return result

    def create_feature_selection_tool(
        self,
        X: Any,
        y: Any,
        feature_names: Optional[Sequence[str]] = None,
        method: str = "auto",
        title: str = "Feature selection",
    ) -> Any:
        """Rank features by a univariate score with a slider selecting the top-k.

        Args:
            X: Feature matrix.
            y: Target vector.
            feature_names: Feature labels.
            method: ``"f_test"``, ``"mutual_info"`` or ``"auto"`` (F-test).
            title: Figure title.

        Returns:
            Plotly bar chart with a top-k slider, or a matplotlib bar chart.

        Raises:
            ValueError: If ``method`` is unknown.
        """
        from sklearn.feature_selection import (
            f_classif,
            f_regression,
            mutual_info_classif,
            mutual_info_regression,
        )

        Xa = to_2d_array(X)
        ya = to_1d_array(y, "y")
        names = default_feature_names(Xa.shape[1], feature_names, X)
        is_clf = is_discrete_target(ya)
        if method in ("auto", "f_test"):
            scores = (f_classif if is_clf else f_regression)(Xa, ya)[0]
            score_label = "F statistic"
        elif method == "mutual_info":
            scores = (mutual_info_classif if is_clf else mutual_info_regression)(Xa, ya, random_state=0)
            score_label = "mutual information"
        else:
            raise ValueError("method must be 'auto', 'f_test' or 'mutual_info'")
        scores = np.nan_to_num(np.asarray(scores, dtype=float))
        order = np.argsort(scores)[::-1]
        ranked_names = [names[i] for i in order]
        ranked_scores = scores[order]
        n = len(names)
        selected, unselected = self.palette[0], self.theme["grid_color"]

        if self._use_plotly():
            import plotly.graph_objects as go

            fig = go.Figure(
                go.Bar(x=ranked_names, y=ranked_scores, marker={"color": [selected] * n}, name=score_label)
            )
            steps = [
                {
                    "method": "restyle",
                    "label": str(k),
                    "args": [{"marker.color": [[selected] * k + [unselected] * (n - k)]}],
                }
                for k in range(1, n + 1)
            ]
            fig.update_layout(
                **self._layout(
                    title,
                    xaxis=self._axis_style("feature (ranked)"),
                    yaxis=self._axis_style(score_label),
                    sliders=[{"active": n - 1, "steps": steps, "currentvalue": {"prefix": "top-k = "}}],
                )
            )
            return fig

        fig, ax = self._mpl_figure()
        ax.bar(ranked_names, ranked_scores, color=selected)
        ax.tick_params(axis="x", rotation=45)
        self._style_mpl_axes(ax, title, "feature (ranked)", score_label)
        fig.tight_layout()
        return fig

    # ---------------------------------------------------------------- tables
    def create_interactive_table(
        self,
        data: pd.DataFrame,
        sortable: bool = True,
        filterable: bool = True,
        highlight_best: Optional[Sequence[str]] = None,
        higher_is_better: bool = True,
        sort_by: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Any:
        """Render a DataFrame as a table highlighting the best value per column.

        Args:
            data: Table contents.
            sortable: Sort rows by ``sort_by`` (or the first highlighted column).
            filterable: Recorded in the figure metadata for front-end consumers.
            highlight_best: Columns whose best cell is highlighted.
            higher_is_better: Direction used for "best".
            sort_by: Column to sort by when ``sortable``.
            title: Figure title.

        Returns:
            Plotly ``Table`` figure, or a matplotlib figure holding the table when Plotly is unavailable.

        Raises:
            ValueError: If a highlighted column is missing.
        """
        df = data.copy()
        highlight = list(highlight_best or [])
        for col in highlight:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not in data")
        sort_col = sort_by or (highlight[0] if highlight else None)
        if sortable and sort_col is not None:
            df = df.sort_values(sort_col, ascending=not higher_is_better).reset_index(drop=True)
        best_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        for col in highlight:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().any():
                best_mask.loc[values.idxmax() if higher_is_better else values.idxmin(), col] = True

        if self._use_plotly():
            import plotly.graph_objects as go

            fill = [["#dbe9fb" if best_mask.at[i, c] else "#ffffff" for i in df.index] for c in df.columns]
            fmt_cells = [[f"{v:.4g}" if isinstance(v, float) else v for v in df[c]] for c in df.columns]
            fig = go.Figure(
                go.Table(
                    header={
                        "values": [f"<b>{c}</b>" for c in df.columns],
                        "fill_color": "#f0f0ee",
                        "align": "left",
                    },
                    cells={"values": fmt_cells, "fill_color": fill, "align": "left"},
                )
            )
            layout = self._layout(title)
            layout["meta"] = {"sortable": sortable, "filterable": filterable, "highlight_best": highlight}
            fig.update_layout(**layout)
            return fig

        fig, ax = self._mpl_figure(figsize=(max(6.0, 1.6 * len(df.columns)), 0.5 * len(df) + 1.2))
        ax.axis("off")
        cell_text = [
            [f"{v:.4g}" if isinstance(v, float) else str(v) for v in row]
            for row in df.itertuples(index=False)
        ]
        cell_colours = [
            ["#dbe9fb" if best_mask.iat[i, j] else "#ffffff" for j in range(len(df.columns))]
            for i in range(len(df))
        ]
        table = ax.table(
            cellText=cell_text,
            colLabels=list(df.columns),
            cellColours=cell_colours,
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold")
        return fig

    # --------------------------------------------------------------- widgets
    def create_model_comparison_widgets(self, models_data: Optional[Dict[str, Dict[str, Any]]] = None) -> Any:
        """Build an ``ipywidgets`` panel with a metric selector driving a comparison plot.

        Args:
            models_data: ``{model: {metric: scores}}``; when ``None`` a minimal
                placeholder dataset is used so the interface still renders.

        Returns:
            ``ipywidgets.VBox`` containing a dropdown and an output area.

        Raises:
            ImportError: If ``ipywidgets`` is not installed.
        """
        if not HAS_IPYWIDGETS:
            raise ImportError("ipywidgets is required for widget interfaces (pip install ipywidgets)")
        import ipywidgets as widgets

        models_data = models_data or {"model": {"score": [0.0]}}
        metrics = list(dict.fromkeys(m for d in models_data.values() for m in d))
        dropdown = widgets.Dropdown(options=metrics, value=metrics[0], description="metric")
        output = widgets.Output()

        def render(metric: str) -> None:
            output.clear_output(wait=True)
            with output:
                fig = self.create_model_comparison_dashboard(
                    models_data, metrics=[metric], title=f"{metric} by model"
                )
                if hasattr(fig, "show"):
                    fig.show()

        dropdown.observe(lambda change: render(change["new"]), names="value")
        render(metrics[0])
        return widgets.VBox([dropdown, output])

    # ---------------------------------------------------------------- export
    def export_to_html(
        self, fig: Any, include_plotlyjs: Union[bool, str] = True, full_html: bool = True
    ) -> str:
        """Serialise a figure (or report dict) to an HTML string.

        Plotly figures use ``Figure.to_html``; matplotlib figures are embedded
        as base64 PNG images; dictionaries (reports) are concatenated with a
        heading per entry, DataFrames via ``to_html``.

        Args:
            fig: Plotly figure, matplotlib figure, DataFrame or a dict of these.
            include_plotlyjs: Passed to Plotly (``True``, ``False`` or ``"cdn"``).
            full_html: Wrap the output in a complete ``<html>`` document.

        Returns:
            HTML string.

        Raises:
            TypeError: If ``fig`` is of an unsupported type.
        """
        body = self._to_html_fragment(fig, include_plotlyjs)
        if not full_html:
            return body
        return f"<html><head><meta charset='utf-8'></head><body>{body}</body></html>"

    def _to_html_fragment(self, obj: Any, include_plotlyjs: Union[bool, str]) -> str:
        if isinstance(obj, dict):
            return "".join(
                f"<h2>{key}</h2>{self._to_html_fragment(val, include_plotlyjs)}" for key, val in obj.items()
            )
        if isinstance(obj, Figure):
            buf = io.BytesIO()
            obj.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"<div><img src='data:image/png;base64,{encoded}' alt='figure'/></div>"
        if isinstance(obj, pd.DataFrame):
            return f"<div>{obj.to_html()}</div>"
        if hasattr(obj, "to_html"):
            html = (
                obj.to_html(include_plotlyjs=include_plotlyjs, full_html=False)
                if hasattr(obj, "to_plotly_json")
                else obj.to_html()
            )
            return f"<div>{html}</div>"
        raise TypeError(f"Cannot export object of type {type(obj).__name__} to HTML")

    def save_html(self, fig: Any, path: str, include_plotlyjs: Union[bool, str] = True) -> str:
        """Write :meth:`export_to_html` output to ``path`` and return the path.

        Args:
            fig: Object accepted by :meth:`export_to_html`.
            path: Destination file.
            include_plotlyjs: Passed to Plotly.

        Returns:
            The path written.
        """
        html = self.export_to_html(fig, include_plotlyjs=include_plotlyjs, full_html=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        self.logger.info("Saved interactive HTML to %s", path)
        return path
