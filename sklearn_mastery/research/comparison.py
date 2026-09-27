"""Rigorous statistical comparison of learning algorithms.

Implements the procedures recommended for comparing classifiers across
multiple datasets and for comparing two algorithms on a single dataset with
cross-validated estimates:

* :func:`friedman_test` — Friedman rank test with the Iman–Davenport F
  correction (Demšar, 2006, §3.2.2).
* :func:`nemenyi_critical_difference` / :func:`nemenyi_posthoc` — Nemenyi
  post-hoc test based on the Studentised range distribution.
* :func:`wilcoxon_holm` — pairwise Wilcoxon signed-rank tests with Holm
  step-down correction (Demšar, 2006; Benavoli et al., 2016 recommend this
  over the Nemenyi test when a control is not designated).
* :func:`corrected_resampled_ttest` — Nadeau & Bengio (2003) variance
  correction for the dependence between overlapping cross-validation folds.
* :func:`bayesian_correlated_ttest` — Bayesian counterpart with a region of
  practical equivalence (Corani & Benavoli, 2015; Benavoli et al., 2017).
* :func:`plot_critical_difference_diagram` — Demšar-style CD diagram.

References
----------
Demšar, J. (2006). Statistical comparisons of classifiers over multiple data
sets. *JMLR*, 7, 1–30.
Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error.
*Machine Learning*, 52, 239–281.
Corani, G., & Benavoli, A. (2015). A Bayesian approach for comparing
cross-validated algorithms on multiple data sets. *Machine Learning*, 100.
Benavoli, A., Corani, G., Demšar, J., & Zaffalon, M. (2017). Time for a change:
a tutorial for comparing multiple classifiers through Bayesian analysis.
*JMLR*, 18(77), 1–36.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "BayesianComparison",
    "FriedmanResult",
    "bayesian_correlated_ttest",
    "corrected_resampled_ttest",
    "friedman_test",
    "holm_correction",
    "nemenyi_critical_difference",
    "nemenyi_posthoc",
    "plot_critical_difference_diagram",
    "wilcoxon_holm",
]

ScoreMatrix = Union[pd.DataFrame, np.ndarray]


def _as_frame(scores: ScoreMatrix) -> pd.DataFrame:
    if isinstance(scores, pd.DataFrame):
        df = scores.copy()
    else:
        arr = np.asarray(scores, dtype=float)
        if arr.ndim != 2:
            raise ValueError("scores must be a 2-D (datasets x estimators) array")
        df = pd.DataFrame(arr, columns=[f"est_{j}" for j in range(arr.shape[1])])
    if df.isna().any().any():
        raise ValueError("scores contain NaN; every estimator must be evaluated on every dataset")
    if df.shape[1] < 2:
        raise ValueError("at least two estimators are required")
    return df.astype(float)


# --------------------------------------------------------------------------- #
# Friedman + Nemenyi
# --------------------------------------------------------------------------- #
@dataclass
class FriedmanResult:
    """Outcome of :func:`friedman_test`.

    Attributes:
        statistic: Friedman chi-square statistic.
        p_value: p-value of the chi-square approximation.
        iman_davenport_statistic: F-distributed corrected statistic.
        iman_davenport_p_value: p-value of the F approximation (preferred).
        average_ranks: Mean rank of each estimator (1 = best).
        n_datasets: Number of datasets (blocks).
        n_estimators: Number of estimators (treatments).
    """

    statistic: float
    p_value: float
    iman_davenport_statistic: float
    iman_davenport_p_value: float
    average_ranks: pd.Series
    n_datasets: int
    n_estimators: int

    def reject(self, alpha: float = 0.05) -> bool:
        """Whether the null of equal performance is rejected at ``alpha``."""
        return bool(self.iman_davenport_p_value < alpha)


def friedman_test(scores: ScoreMatrix, higher_is_better: bool = True) -> FriedmanResult:
    """Friedman test over a datasets x estimators score matrix.

    Args:
        scores: Rows are datasets, columns are estimators.
        higher_is_better: Rank direction.

    Returns:
        :class:`FriedmanResult` with both the chi-square and the Iman–Davenport
        statistics. The latter is less conservative and is what
        :meth:`FriedmanResult.reject` uses.

    Raises:
        ValueError: If fewer than two estimators or two datasets are given.
    """
    df = _as_frame(scores)
    n, k = df.shape
    if n < 2:
        raise ValueError("at least two datasets are required for the Friedman test")
    ranks = df.rank(axis=1, ascending=not higher_is_better, method="average")
    avg = ranks.mean(axis=0)
    chi2 = 12.0 * n / (k * (k + 1)) * (float((avg**2).sum()) - k * (k + 1) ** 2 / 4.0)
    p_chi2 = float(stats.chi2.sf(chi2, k - 1))
    denom = n * (k - 1) - chi2
    if denom <= 0:  # perfect agreement across datasets: F is unbounded
        f_stat, p_f = float("inf"), 0.0
    else:
        f_stat = (n - 1) * chi2 / denom
        p_f = float(stats.f.sf(f_stat, k - 1, (k - 1) * (n - 1)))
    return FriedmanResult(
        statistic=float(chi2),
        p_value=p_chi2,
        iman_davenport_statistic=float(f_stat),
        iman_davenport_p_value=p_f,
        average_ranks=avg.rename("average_rank"),
        n_datasets=n,
        n_estimators=k,
    )


def nemenyi_critical_difference(n_estimators: int, n_datasets: int, alpha: float = 0.05) -> float:
    """Critical difference of average ranks for the Nemenyi test.

    ``CD = q_alpha * sqrt(k (k + 1) / (6 N))`` where ``q_alpha`` is the
    Studentised range quantile divided by ``sqrt(2)``.
    """
    if n_estimators < 2 or n_datasets < 1:
        raise ValueError("need at least two estimators and one dataset")
    q_alpha = stats.studentized_range.ppf(1 - alpha, n_estimators, np.inf) / np.sqrt(2)
    return float(q_alpha * np.sqrt(n_estimators * (n_estimators + 1) / (6.0 * n_datasets)))


def nemenyi_posthoc(scores: ScoreMatrix, higher_is_better: bool = True) -> pd.DataFrame:
    """Pairwise Nemenyi p-values (symmetric estimator x estimator matrix)."""
    df = _as_frame(scores)
    n, k = df.shape
    avg = df.rank(axis=1, ascending=not higher_is_better, method="average").mean(axis=0)
    se = np.sqrt(k * (k + 1) / (6.0 * n))
    names = list(df.columns)
    p = pd.DataFrame(np.ones((k, k)), index=names, columns=names)
    for a, b in combinations(names, 2):
        q = abs(avg[a] - avg[b]) / se * np.sqrt(2)
        pv = float(stats.studentized_range.sf(q, k, np.inf))
        p.loc[a, b] = p.loc[b, a] = min(1.0, pv)
    return p


# --------------------------------------------------------------------------- #
# Wilcoxon + Holm
# --------------------------------------------------------------------------- #
def holm_correction(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def wilcoxon_holm(
    scores: ScoreMatrix,
    alpha: float = 0.05,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """All pairwise Wilcoxon signed-rank tests with Holm correction.

    Args:
        scores: datasets x estimators matrix (or paired fold scores).
        alpha: Family-wise significance level.
        higher_is_better: Determines the sign of ``winner``.

    Returns:
        DataFrame with one row per pair: ``estimator_a, estimator_b, statistic,
        p_value, p_adjusted, significant, winner, median_difference``.
    """
    df = _as_frame(scores)
    rows: List[Dict[str, object]] = []
    for a, b in combinations(df.columns, 2):
        diff = df[a].to_numpy() - df[b].to_numpy()
        if np.allclose(diff, 0):
            stat, p = 0.0, 1.0
        else:
            stat, p = stats.wilcoxon(df[a], df[b], zero_method="wilcox", alternative="two-sided")
        med = float(np.median(diff))
        if higher_is_better:
            winner = a if med > 0 else b if med < 0 else None
        else:
            winner = a if med < 0 else b if med > 0 else None
        rows.append(
            {
                "estimator_a": a,
                "estimator_b": b,
                "statistic": float(stat),
                "p_value": float(p),
                "median_difference": med,
                "winner": winner,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_adjusted"] = holm_correction(out["p_value"].tolist())
    out["significant"] = out["p_adjusted"] < alpha
    return out[
        [
            "estimator_a",
            "estimator_b",
            "statistic",
            "p_value",
            "p_adjusted",
            "significant",
            "winner",
            "median_difference",
        ]
    ]


# --------------------------------------------------------------------------- #
# Two-algorithm, single-dataset tests on cross-validated scores
# --------------------------------------------------------------------------- #
def _cv_geometry(n_train: Optional[int], n_test: Optional[int], n_splits: Optional[int]) -> float:
    """Return the ``n_test / n_train`` ratio used by the Nadeau–Bengio correction."""
    if n_train is not None and n_test is not None:
        if n_train <= 0 or n_test <= 0:
            raise ValueError("n_train and n_test must be positive")
        return n_test / n_train
    if n_splits is not None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        return 1.0 / (n_splits - 1)
    raise ValueError("provide either (n_train, n_test) or n_splits")


def corrected_resampled_ttest(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    n_splits: Optional[int] = None,
) -> Tuple[float, float]:
    """Nadeau–Bengio corrected paired t-test for (repeated) k-fold scores.

    The naive paired t-test over cross-validation folds underestimates the
    variance because training sets overlap. The correction inflates the
    variance by ``(1/n + n_test/n_train)``.

    Args:
        scores_a: Fold scores of algorithm A (aligned with ``scores_b``).
        scores_b: Fold scores of algorithm B.
        n_train: Training-set size per fold.
        n_test: Test-set size per fold.
        n_splits: Alternative to sizes for plain k-fold (``n_test/n_train = 1/(k-1)``).

    Returns:
        ``(t_statistic, two_sided_p_value)``.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("scores_a and scores_b must be 1-D and aligned")
    n = a.size
    if n < 2:
        raise ValueError("need at least two paired scores")
    ratio = _cv_geometry(n_train, n_test, n_splits)
    d = a - b
    mean = d.mean()
    var = d.var(ddof=1)
    if var == 0:
        return (0.0 if mean == 0 else float(np.inf) * np.sign(mean), 0.0 if mean != 0 else 1.0)
    se = np.sqrt((1.0 / n + ratio) * var)
    t = mean / se
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return float(t), float(p)


@dataclass
class BayesianComparison:
    """Posterior summary of :func:`bayesian_correlated_ttest`.

    Attributes:
        p_left: Probability that A is worse than B by more than the ROPE.
        p_rope: Probability that A and B are practically equivalent.
        p_right: Probability that A is better than B by more than the ROPE.
        mean_difference: Posterior mean of ``A - B``.
        rope: Half-width of the region of practical equivalence.
    """

    p_left: float
    p_rope: float
    p_right: float
    mean_difference: float
    rope: float
    df: float
    scale: float

    def decision(self, threshold: float = 0.95) -> str:
        """Return ``'A > B'``, ``'A < B'``, ``'equivalent'`` or ``'undecided'``."""
        if self.p_right >= threshold:
            return "A > B"
        if self.p_left >= threshold:
            return "A < B"
        if self.p_rope >= threshold:
            return "equivalent"
        return "undecided"


def bayesian_correlated_ttest(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    rope: float = 0.01,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    n_splits: Optional[int] = None,
) -> BayesianComparison:
    """Bayesian correlated t-test (Corani & Benavoli, 2015).

    Places a Student-t posterior on the mean difference with the Nadeau–Bengio
    correlation correction and integrates it over three regions: below
    ``-rope``, inside ``[-rope, rope]`` (practical equivalence) and above ``rope``.

    Args:
        scores_a: Fold scores of algorithm A.
        scores_b: Fold scores of algorithm B (aligned).
        rope: Half-width of the region of practical equivalence, on the metric
            scale (e.g. ``0.01`` for one accuracy point).
        n_train, n_test, n_splits: Cross-validation geometry as in
            :func:`corrected_resampled_ttest`.

    Returns:
        :class:`BayesianComparison` with the three posterior probabilities.
    """
    if rope < 0:
        raise ValueError("rope must be non-negative")
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        raise ValueError("scores must be aligned 1-D arrays with at least two entries")
    n = a.size
    ratio = _cv_geometry(n_train, n_test, n_splits)
    d = a - b
    mean = float(d.mean())
    var = float(d.var(ddof=1))
    scale = float(np.sqrt((1.0 / n + ratio) * var))
    dof = n - 1
    if scale == 0.0:
        left, right = float(mean < -rope), float(mean > rope)
        return BayesianComparison(left, 1.0 - left - right, right, mean, rope, dof, scale)
    dist = stats.t(df=dof, loc=mean, scale=scale)
    p_left = float(dist.cdf(-rope))
    p_right = float(dist.sf(rope))
    p_rope = max(0.0, 1.0 - p_left - p_right)
    return BayesianComparison(p_left, p_rope, p_right, mean, rope, dof, scale)


# --------------------------------------------------------------------------- #
# Critical-difference diagram
# --------------------------------------------------------------------------- #
def plot_critical_difference_diagram(
    average_ranks: Union[pd.Series, Dict[str, float]],
    critical_difference: float,
    ax=None,
    title: Optional[str] = None,
    highlight_best: bool = True,
):
    """Draw a Demšar critical-difference diagram.

    Estimators are placed on a rank axis; groups whose average ranks differ by
    less than ``critical_difference`` are connected by a horizontal bar.

    Args:
        average_ranks: Mean rank per estimator (lower is better).
        critical_difference: Output of :func:`nemenyi_critical_difference`.
        ax: Matplotlib axes to draw on (created when ``None``).
        title: Optional title.
        highlight_best: Bold the best-ranked estimator label.

    Returns:
        The matplotlib :class:`~matplotlib.axes.Axes`.
    """
    import matplotlib.pyplot as plt

    ranks = pd.Series(average_ranks, dtype=float).sort_values()
    k = len(ranks)
    if k < 2:
        raise ValueError("need at least two estimators")
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 1.2 * k), 0.6 * k + 2))

    lo, hi = 1.0, float(k)
    ax.set_xlim(hi + 0.4, lo - 0.4)  # best (rank 1) on the right, as in Demšar
    ax.set_ylim(-0.4 * k - 1.2, 1.4)
    ax.spines[["left", "right", "bottom"]].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.xaxis.set_ticks_position("top")
    ax.set_xticks(np.arange(1, k + 1))

    # CD bar
    ax.plot([lo, lo + critical_difference], [1.1, 1.1], color="black", lw=2)
    ax.text(
        lo + critical_difference / 2,
        1.2,
        f"CD = {critical_difference:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    # Estimator stems and labels (alternate sides for legibility)
    names = list(ranks.index)
    half = (k + 1) // 2
    for i, name in enumerate(names):
        r = ranks[name]
        if i < half:  # right side, best first
            y_label = -0.4 * (i + 1)
            x_label = lo - 0.35
            ha = "right"
        else:
            y_label = -0.4 * (k - i)
            x_label = hi + 0.35
            ha = "left"
        ax.plot([r, r], [0.9, y_label], color="black", lw=1)
        ax.plot([r, x_label], [y_label, y_label], color="black", lw=1)
        weight = "bold" if (highlight_best and i == 0) else "normal"
        ax.text(x_label, y_label, f"{name} ({r:.2f})", ha=ha, va="center", fontsize=9, fontweight=weight)

    # Cliques: maximal sets of estimators within CD
    cliques: List[Tuple[int, int]] = []
    sorted_r = ranks.to_numpy()
    i = 0
    while i < k:
        j = i
        while j + 1 < k and sorted_r[j + 1] - sorted_r[i] <= critical_difference:
            j += 1
        if j > i and not any(s <= i and e >= j for s, e in cliques):
            cliques.append((i, j))
        i += 1
    for level, (s, e) in enumerate(cliques):
        y = 0.75 - 0.15 * level
        ax.plot([sorted_r[s] - 0.05, sorted_r[e] + 0.05], [y, y], color="black", lw=3, solid_capstyle="round")

    if title:
        ax.set_title(title, pad=25)
    return ax
