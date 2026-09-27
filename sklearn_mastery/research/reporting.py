"""Publication-ready tables from benchmark results.

Produces Markdown and LaTeX tables of ``mean ± std`` per (dataset, estimator)
with the best estimator per dataset emphasised, plus an average-rank row, in
the layout used by most empirical ML papers.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

__all__ = ["format_mean_std_table", "results_to_latex", "results_to_markdown"]


def _mean_std_frames(results: pd.DataFrame, metric: str):
    df = results[results["metric"] == metric]
    if df.empty:
        raise KeyError(f"metric {metric!r} not present in results")
    mean = df.pivot_table(index="dataset", columns="estimator", values="value", aggfunc="mean")
    std = df.pivot_table(index="dataset", columns="estimator", values="value", aggfunc="std").fillna(0.0)
    return mean, std


def format_mean_std_table(
    results: pd.DataFrame,
    metric: str,
    higher_is_better: bool = True,
    precision: int = 3,
    bold_best: bool = True,
    fmt: str = "markdown",
    include_rank: bool = True,
) -> pd.DataFrame:
    """Build a string table of ``mean ± std`` cells.

    Args:
        results: Long-form results from :class:`BenchmarkResult`.
        metric: Metric to tabulate.
        higher_is_better: Determines which cell is emphasised.
        precision: Decimal places.
        bold_best: Emphasise the best estimator per dataset.
        fmt: ``'markdown'`` (``**x**``) or ``'latex'`` (``\\textbf{x}``).
        include_rank: Append an ``avg. rank`` row.

    Returns:
        DataFrame of strings, datasets as rows, estimators as columns.
    """
    mean, std = _mean_std_frames(results, metric)
    pm = "±" if fmt == "markdown" else r"$\pm$"
    table = pd.DataFrame(index=mean.index, columns=mean.columns, dtype=object)
    for ds in mean.index:
        row = mean.loc[ds]
        best = row.idxmax() if higher_is_better else row.idxmin()
        for est in mean.columns:
            cell = f"{row[est]:.{precision}f} {pm} {std.loc[ds, est]:.{precision}f}"
            if bold_best and est == best:
                cell = f"**{cell}**" if fmt == "markdown" else rf"\textbf{{{cell}}}"
            table.loc[ds, est] = cell
    if include_rank:
        ranks = mean.rank(axis=1, ascending=not higher_is_better, method="average").mean(axis=0)
        table.loc["avg. rank"] = [f"{r:.2f}" for r in ranks]
    return table


def results_to_markdown(results: pd.DataFrame, metric: str, caption: Optional[str] = None, **kwargs) -> str:
    """Render a Markdown table (GitHub-flavoured)."""
    table = format_mean_std_table(results, metric, fmt="markdown", **kwargs)
    cols = list(table.columns)
    lines = ["| dataset | " + " | ".join(cols) + " |", "|---|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, row in table.iterrows():
        lines.append(f"| {idx} | " + " | ".join(str(v) for v in row.tolist()) + " |")
    body = "\n".join(lines)
    return f"{body}\n\n*{caption}*\n" if caption else body + "\n"


def results_to_latex(
    results: pd.DataFrame,
    metric: str,
    caption: Optional[str] = None,
    label: Optional[str] = None,
    **kwargs,
) -> str:
    """Render a LaTeX ``table`` environment (requires ``booktabs``)."""
    table = format_mean_std_table(results, metric, fmt="latex", **kwargs)
    cols = list(table.columns)
    spec = "l" + "c" * len(cols)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        "dataset & " + " & ".join(_latex_escape(c) for c in cols) + r" \\",
        r"\midrule",
    ]
    for idx, row in table.iterrows():
        if idx == "avg. rank":
            lines.append(r"\midrule")
        lines.append(_latex_escape(str(idx)) + " & " + " & ".join(str(v) for v in row.tolist()) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
