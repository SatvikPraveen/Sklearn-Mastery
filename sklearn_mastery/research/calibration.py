"""Probability calibration diagnostics.

* :func:`expected_calibration_error` / :func:`maximum_calibration_error`
  (Naeini et al., 2015; Guo et al., 2017).
* :func:`brier_score_decomposition` — Murphy (1973) decomposition into
  reliability, resolution and uncertainty.
* :func:`reliability_diagram` — binned reliability plot with sample-count
  histogram.

All functions accept binary targets with positive-class probabilities, or
multiclass targets with a probability matrix (top-label calibration).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "BrierDecomposition",
    "CalibrationBins",
    "brier_score_decomposition",
    "compute_calibration_bins",
    "expected_calibration_error",
    "maximum_calibration_error",
    "reliability_diagram",
]


def _top_label(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce inputs to (correct_indicator, confidence) arrays."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim == 1:
        conf = y_prob
        outcome = (y_true == 1).astype(float) if set(np.unique(y_true)) <= {0, 1} else None
        if outcome is None:
            raise ValueError("1-D probabilities require binary targets encoded as 0/1")
        return outcome, conf
    if y_prob.ndim != 2:
        raise ValueError("y_prob must be 1-D or 2-D")
    if y_prob.shape[0] != y_true.shape[0]:
        raise ValueError("y_true and y_prob have different lengths")
    if y_prob.shape[1] == 2 and set(np.unique(y_true)) <= {0, 1}:
        return (y_true == 1).astype(float), y_prob[:, 1]
    pred = y_prob.argmax(axis=1)
    conf = y_prob.max(axis=1)
    classes = np.unique(y_true)
    labels = classes[pred] if len(classes) == y_prob.shape[1] else pred
    return (labels == y_true).astype(float), conf


@dataclass
class CalibrationBins:
    """Per-bin calibration statistics."""

    edges: np.ndarray
    counts: np.ndarray
    mean_confidence: np.ndarray
    mean_accuracy: np.ndarray

    @property
    def gaps(self) -> np.ndarray:
        return np.abs(self.mean_accuracy - self.mean_confidence)


def compute_calibration_bins(y_true, y_prob, n_bins: int = 10, strategy: str = "uniform") -> CalibrationBins:
    """Bin predictions by confidence and compute per-bin accuracy/confidence.

    Args:
        y_true: Targets.
        y_prob: Probabilities (1-D positive-class or 2-D matrix).
        n_bins: Number of bins.
        strategy: ``'uniform'`` (equal-width) or ``'quantile'`` (equal-mass).
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    outcome, conf = _top_label(y_true, y_prob)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(conf, np.linspace(0, 1, n_bins + 1)))
        if edges.size < 2:
            edges = np.array([0.0, 1.0])
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, len(edges) - 2)
    n_eff = len(edges) - 1
    counts = np.bincount(idx, minlength=n_eff).astype(float)
    sum_conf = np.bincount(idx, weights=conf, minlength=n_eff)
    sum_acc = np.bincount(idx, weights=outcome, minlength=n_eff)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.where(counts > 0, sum_conf / counts, np.nan)
        mean_acc = np.where(counts > 0, sum_acc / counts, np.nan)
    return CalibrationBins(edges=edges, counts=counts, mean_confidence=mean_conf, mean_accuracy=mean_acc)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10, strategy: str = "uniform") -> float:
    """Expected calibration error: count-weighted mean |accuracy − confidence|."""
    bins = compute_calibration_bins(y_true, y_prob, n_bins, strategy)
    mask = bins.counts > 0
    weights = bins.counts[mask] / bins.counts.sum()
    return float(np.sum(weights * bins.gaps[mask]))


def maximum_calibration_error(y_true, y_prob, n_bins: int = 10, strategy: str = "uniform") -> float:
    """Maximum calibration error: worst-bin |accuracy − confidence|."""
    bins = compute_calibration_bins(y_true, y_prob, n_bins, strategy)
    mask = bins.counts > 0
    return float(np.max(bins.gaps[mask])) if mask.any() else 0.0


@dataclass
class BrierDecomposition:
    """Murphy decomposition: ``brier = reliability - resolution + uncertainty``."""

    brier: float
    reliability: float
    resolution: float
    uncertainty: float


def brier_score_decomposition(y_true, y_prob, n_bins: int = 10) -> BrierDecomposition:
    """Decompose the (binary) Brier score into reliability, resolution and uncertainty.

    Lower reliability and higher resolution are better; uncertainty depends
    only on the class prior. Uses uniform bins in probability space.
    """
    outcome, conf = _top_label(y_true, y_prob)
    n = conf.size
    if n == 0:
        raise ValueError("empty input")
    bins = compute_calibration_bins(outcome, conf, n_bins, "uniform")
    base_rate = float(outcome.mean())
    mask = bins.counts > 0
    reliability = float(
        np.sum(bins.counts[mask] * (bins.mean_confidence[mask] - bins.mean_accuracy[mask]) ** 2) / n
    )
    resolution = float(np.sum(bins.counts[mask] * (bins.mean_accuracy[mask] - base_rate) ** 2) / n)
    uncertainty = base_rate * (1 - base_rate)
    brier = float(np.mean((conf - outcome) ** 2))
    return BrierDecomposition(
        brier=brier, reliability=reliability, resolution=resolution, uncertainty=uncertainty
    )


def reliability_diagram(
    y_true,
    y_prob,
    n_bins: int = 10,
    strategy: str = "uniform",
    ax=None,
    label: Optional[str] = None,
    show_histogram: bool = True,
):
    """Plot a reliability diagram (accuracy vs. confidence per bin).

    Returns:
        The matplotlib axes holding the reliability curve.
    """
    import matplotlib.pyplot as plt

    bins = compute_calibration_bins(y_true, y_prob, n_bins, strategy)
    ece = expected_calibration_error(y_true, y_prob, n_bins, strategy)
    if ax is None:
        if show_histogram:
            _, (ax, ax_hist) = plt.subplots(
                2, 1, figsize=(5, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
            )
        else:
            _, ax = plt.subplots(figsize=(5, 5))
            ax_hist = None
    else:
        ax_hist = None
    mask = bins.counts > 0
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1, label="perfect")
    ax.plot(
        bins.mean_confidence[mask],
        bins.mean_accuracy[mask],
        marker="o",
        label=f"{label or 'model'} (ECE={ece:.3f})",
    )
    ax.set_ylabel("empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    if ax_hist is not None:
        centres = (bins.edges[:-1] + bins.edges[1:]) / 2
        ax_hist.bar(centres, bins.counts, width=np.diff(bins.edges), edgecolor="white")
        ax_hist.set_xlabel("confidence")
        ax_hist.set_ylabel("count")
    else:
        ax.set_xlabel("confidence")
    return ax
