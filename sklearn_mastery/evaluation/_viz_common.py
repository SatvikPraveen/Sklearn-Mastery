"""Shared helpers for the visualization modules.

This private module centralises the optional-dependency flags, the colour
palette and small array/axes utilities used by both the static
(:mod:`sklearn_mastery.evaluation.visualization`) and interactive
(:mod:`sklearn_mastery.evaluation.interactive`) toolkits so that neither has
to import the other.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sklearn_mastery.config.settings import settings

try:  # pragma: no cover - exercised implicitly by the import guard
    import plotly  # noqa: F401

    HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    HAS_PLOTLY = False

try:  # pragma: no cover
    import ipywidgets  # noqa: F401

    HAS_IPYWIDGETS = True
except ImportError:  # pragma: no cover
    HAS_IPYWIDGETS = False


# --------------------------------------------------------------------------- palette
#: Fixed-order categorical palette (eight slots, colour-vision-deficiency safe on
#: adjacent pairs). Series are always assigned in this order, never re-cycled.
CATEGORICAL_PALETTE: Tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
#: Single-hue colormap for magnitudes.
SEQUENTIAL_CMAP = "Blues"
#: Two-hue colormap with a neutral midpoint for signed quantities.
DIVERGING_CMAP = "RdBu_r"
#: Neutral colour for reference lines (diagonals, zero lines, baselines).
REFERENCE_COLOR = "#52514e"
#: Colours for the "positive" / "negative" poles of signed bar charts.
POSITIVE_COLOR = "#2a78d6"
NEGATIVE_COLOR = "#eb6834"


def series_colors(n: int, palette: Optional[Sequence[str]] = None) -> List[str]:
    """Return ``n`` colours in fixed palette order.

    For more series than the palette holds, colours are sampled evenly from
    the ``viridis`` colormap instead of re-cycling hues, so that adjacent
    series stay distinguishable.

    Args:
        n: Number of colours required.
        palette: Optional palette overriding :data:`CATEGORICAL_PALETTE`.

    Returns:
        List of hex colour strings of length ``n``.
    """
    palette = list(palette or CATEGORICAL_PALETTE)
    if n <= len(palette):
        return palette[:n]
    cmap = plt.get_cmap("viridis", n)
    return [plt.matplotlib.colors.to_hex(cmap(i)) for i in range(n)]


def plotly_available() -> bool:
    """Check at call time whether :mod:`plotly` can be imported.

    Unlike the module-level :data:`HAS_PLOTLY` flag this re-executes the
    import machinery, so tests that monkeypatch ``builtins.__import__`` (or
    environments where plotly is uninstalled mid-session) are honoured.

    Returns:
        ``True`` when ``plotly.graph_objects`` imports successfully.
    """
    try:
        import plotly.graph_objects  # noqa: F401
    except ImportError:
        return False
    return True


# --------------------------------------------------------------------------- arrays
def trapezoid(y: Any, x: Any) -> float:
    """Trapezoid-rule integral compatible with numpy 1.x and 2.x.

    Args:
        y: Sample values.
        x: Sample positions.

    Returns:
        The integral of ``y`` over ``x``.
    """
    func = getattr(np, "trapezoid", None) or np.trapz  # noqa: NPY201
    return float(func(np.asarray(y, dtype=float), np.asarray(x, dtype=float)))


def to_1d_array(values: Any, name: str = "values") -> np.ndarray:
    """Coerce ``values`` to a flat numpy array.

    Args:
        values: Array-like input (list, Series, ndarray).
        name: Name used in error messages.

    Returns:
        One-dimensional numpy array.

    Raises:
        ValueError: If ``values`` cannot be flattened to one dimension.
    """
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        if arr.shape[1:] != (1,) * (arr.ndim - 1):
            raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
        arr = arr.reshape(-1)
    return arr


def to_2d_array(X: Any, name: str = "X") -> np.ndarray:
    """Coerce ``X`` to a two-dimensional numpy array.

    Args:
        X: Array-like or DataFrame.
        name: Name used in error messages.

    Returns:
        Two-dimensional numpy array with shape ``(n_samples, n_features)``.

    Raises:
        ValueError: If ``X`` is empty or has more than two dimensions.
    """
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    return arr


def default_feature_names(
    n_features: int,
    feature_names: Optional[Sequence[str]] = None,
    X: Any = None,
) -> List[str]:
    """Return feature names, generating ``feature_i`` defaults when absent.

    Args:
        n_features: Expected number of features.
        feature_names: User supplied names, or ``None``.
        X: Optional DataFrame whose columns provide names.

    Returns:
        List of ``n_features`` names.

    Raises:
        ValueError: If ``feature_names`` has the wrong length.
    """
    if feature_names is None and isinstance(X, pd.DataFrame):
        feature_names = [str(c) for c in X.columns]
    if feature_names is None:
        return [f"feature_{i}" for i in range(n_features)]
    names = [str(n) for n in feature_names]
    if len(names) != n_features:
        raise ValueError(f"Expected {n_features} feature names, got {len(names)}")
    return names


def select_features(
    X: Any, feature_names: Optional[Sequence[str]] = None, name: str = "X"
) -> Tuple[np.ndarray, List[str]]:
    """Return ``X`` restricted to the named features together with their names.

    When ``feature_names`` is shorter than the number of columns only the
    first ``len(feature_names)`` columns are kept, which lets callers plot a
    subset without slicing ``X`` themselves.

    Args:
        X: Feature matrix or DataFrame.
        feature_names: Names of the leading columns to keep (all when ``None``).
        name: Name used in error messages.

    Returns:
        The (possibly sliced) array and the matching feature names.

    Raises:
        ValueError: If more names than columns are given.
    """
    if feature_names is None and isinstance(X, pd.DataFrame):
        feature_names = [str(c) for c in X.columns]
    arr = to_2d_array(X, name)
    if feature_names is None:
        return arr, default_feature_names(arr.shape[1])
    names = [str(n) for n in feature_names]
    if len(names) > arr.shape[1]:
        raise ValueError(f"{len(names)} feature names given for {arr.shape[1]} features in {name}")
    return arr[:, : len(names)], names


def is_discrete_target(y: np.ndarray, max_classes: int = 20) -> bool:
    """Heuristically decide whether ``y`` is categorical.

    Args:
        y: Target array.
        max_classes: Maximum number of unique values considered categorical.

    Returns:
        ``True`` for integer/object/bool targets with few unique values.
    """
    if y.dtype.kind in "OUSb":
        return True
    uniques = np.unique(y)
    if len(uniques) > max_classes:
        return False
    return bool(np.all(np.mod(uniques.astype(float), 1) == 0))


# --------------------------------------------------------------------------- figures
def resolve_axes(
    ax: Optional[Axes],
    figsize: Optional[Tuple[float, float]] = None,
    **subplot_kw: Any,
) -> Tuple[Figure, Axes, bool]:
    """Return ``(fig, ax, created)`` creating a new figure when ``ax`` is ``None``.

    Args:
        ax: Existing axes or ``None``.
        figsize: Figure size for a newly created figure.
        **subplot_kw: Extra keyword arguments (e.g. ``projection="polar"``).

    Returns:
        The figure, the axes to draw on and a flag telling whether the figure
        was created by this call (callers only ``tight_layout`` their own).
    """
    if ax is not None:
        return ax.figure, ax, False
    fig, ax = plt.subplots(figsize=figsize or settings.FIGURE_SIZE, subplot_kw=subplot_kw or None)
    return fig, ax, True


def subplot_grid(
    n_plots: int,
    max_cols: int = 3,
    panel_size: Tuple[float, float] = (4.0, 3.2),
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, List[Axes]]:
    """Create a figure with exactly ``n_plots`` axes laid out in a grid.

    Surplus grid cells are removed so ``len(fig.axes) == n_plots``.

    Args:
        n_plots: Number of panels required (must be positive).
        max_cols: Maximum number of columns.
        panel_size: Width/height of one panel in inches.
        figsize: Explicit figure size overriding ``panel_size``.

    Returns:
        The figure and a flat list of its axes.

    Raises:
        ValueError: If ``n_plots`` is not positive.
    """
    if n_plots <= 0:
        raise ValueError("n_plots must be positive")
    n_cols = min(n_plots, max_cols)
    n_rows = math.ceil(n_plots / n_cols)
    size = figsize or (panel_size[0] * n_cols, panel_size[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=size, squeeze=False)
    flat = list(axes.ravel())
    for extra in flat[n_plots:]:
        fig.delaxes(extra)
    return fig, flat[:n_plots]


def finalize_figure(fig: Figure, created: bool, save_path: Optional[str] = None) -> Figure:
    """Apply layout and optionally save a figure without displaying it.

    Args:
        fig: Figure to finalize.
        created: Whether the calling function owns the figure. Layout is only
            applied to figures we created, so user-provided axes are untouched.
        save_path: Optional path to write the figure to.

    Returns:
        The same figure.
    """
    if created:
        try:
            fig.tight_layout()
        except (ValueError, RuntimeError):  # pragma: no cover - layout edge cases
            pass
    if save_path:
        fig.savefig(save_path, dpi=settings.DPI, bbox_inches="tight")
    return fig


def style_axes(
    ax: Axes, title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None
) -> None:
    """Apply recessive grid/spines and optional labels to an axes.

    Args:
        ax: Axes to style.
        title: Optional title.
        xlabel: Optional x-axis label.
        ylabel: Optional y-axis label.
    """
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ax.name != "polar":
        ax.grid(True, alpha=0.25, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)


def as_dataframe(data: Union[pd.DataFrame, dict], index_name: str = "model") -> pd.DataFrame:
    """Convert a ``{row: {column: value}}`` mapping or DataFrame into a DataFrame.

    Args:
        data: Nested mapping or DataFrame.
        index_name: Name of the index when converting from a mapping.

    Returns:
        DataFrame with one row per outer key.
    """
    if isinstance(data, pd.DataFrame):
        return data.copy()
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = index_name
    return df
