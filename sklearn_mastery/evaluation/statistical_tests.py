"""Statistical significance testing for model comparison.

Implements the tests recommended in the machine-learning evaluation
literature for comparing classifiers/regressors:

* paired *t*-test and the variance-corrected resampled *t*-test
  (Nadeau & Bengio, 2003) for scores from repeated cross-validation;
* Wilcoxon signed-rank test (Wilcoxon, 1945; Demšar, 2006);
* McNemar's test on paired predictions (McNemar, 1947; Dietterich, 1998);
* Friedman test with Iman-Davenport correction and the Nemenyi post-hoc
  critical difference for many models over many datasets (Demšar, 2006);
* bootstrap confidence intervals and the paired bootstrap test
  (Efron & Tibshirani, 1993; Koehn, 2004; Berg-Kirkpatrick et al., 2012).

Every method returns a plain dictionary with the statistic, p-value,
significance flag at ``alpha`` and a human-readable interpretation.

``ValidationCurveAnalyzer`` historically lived in this module; it is
re-exported from :mod:`sklearn_mastery.evaluation.cross_validation` for
backward compatibility.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from scipy import stats

from sklearn_mastery.config.logging_config import LoggerMixin
from sklearn_mastery.evaluation.cross_validation import ValidationCurveAnalyzer
from sklearn_mastery.evaluation.utils import (
    bootstrap_confidence_interval,
    compute_effect_size,
    ensure_numpy_array,
    holm_bonferroni,
    interpret_effect_size,
    make_rng,
    validate_targets,
)

#: Critical values ``q_alpha`` of the Nemenyi test (studentized range
#: statistic divided by sqrt(2)) for ``k = 2..10`` models, from Demšar
#: (2006), Table 5(a).
NEMENYI_Q_ALPHA: Dict[float, Dict[int, float]] = {
    0.05: {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164},
    0.10: {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920},
}


def _paired_scores(scores1: Any, scores2: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate two paired score vectors and return them as float arrays."""
    s1 = ensure_numpy_array(scores1).astype(float).ravel()
    s2 = ensure_numpy_array(scores2).astype(float).ravel()
    if len(s1) != len(s2):
        raise ValueError("Score arrays must have the same length")
    if len(s1) < 2:
        raise ValueError("At least two paired observations are required")
    if not (np.all(np.isfinite(s1)) and np.all(np.isfinite(s2))):
        raise ValueError("Scores contain NaN or infinite values")
    return s1, s2


class StatisticalTester(LoggerMixin):
    """Statistical significance tests for comparing model performance.

    Args:
        alpha: Significance level used for all ``is_significant`` flags.
        random_state: Seed for the bootstrap-based procedures.

    Raises:
        ValueError: If ``alpha`` is not in ``(0, 1)``.
    """

    def __init__(self, alpha: float = 0.05, random_state: Optional[int] = None):
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        self.alpha = alpha
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    # Two-model tests on paired scores
    # ------------------------------------------------------------------ #
    def paired_t_test(
        self,
        scores1: Sequence[float],
        scores2: Sequence[float],
        model1_name: str = "Model 1",
        model2_name: str = "Model 2",
    ) -> Dict[str, Any]:
        """Paired Student's *t*-test on per-fold (or per-dataset) scores.

        Tests ``H0: mean(scores1 - scores2) == 0`` assuming the paired
        differences are approximately normal. When the scores come from
        *k*-fold cross-validation the folds share training data, which
        inflates the Type-I error (Dietterich, 1998); prefer
        :meth:`corrected_resampled_t_test` in that setting.

        Args:
            scores1: Scores of the first model, one per fold/dataset.
            scores2: Scores of the second model, paired with ``scores1``.
            model1_name: Label for reporting.
            model2_name: Label for reporting.

        Returns:
            Dict with ``t_statistic``, ``p_value`` (two-sided), ``df``,
            ``mean_difference``, ``cohens_d`` (pooled), ``cohens_dz`` (paired),
            ``effect_size_label``, ``is_significant``, ``alpha`` and ``interpretation``.

        Raises:
            ValueError: If the inputs are not paired or contain fewer than two observations.

        References:
            Student (1908). The probable error of a mean. *Biometrika*, 6(1), 1-25.
            Dietterich, T. G. (1998). Approximate statistical tests for comparing
            supervised classification learning algorithms. *Neural Computation*, 10(7), 1895-1923.
        """
        s1, s2 = _paired_scores(scores1, scores2)
        diff = s1 - s2
        n = len(diff)
        if np.allclose(np.std(diff, ddof=1), 0.0):
            # Identical differences on every fold: no variance to estimate. A zero
            # shift is "no evidence"; a constant non-zero shift is overwhelming evidence.
            t_stat, p_value = self._degenerate_t(float(np.mean(diff)))
        else:
            res = stats.ttest_rel(s1, s2)
            t_stat, p_value = float(res.statistic), float(res.pvalue)

        cohens_d = compute_effect_size(s1, s2, method="cohens_d")
        cohens_dz = compute_effect_size(s1, s2, method="cohens_dz")
        is_significant = bool(p_value < self.alpha)
        result = {
            "test": "paired_t_test",
            "model1_name": model1_name,
            "model2_name": model2_name,
            "model1_mean": float(np.mean(s1)),
            "model2_mean": float(np.mean(s2)),
            "mean_difference": float(np.mean(diff)),
            "t_statistic": t_stat,
            "df": n - 1,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "cohens_dz": cohens_dz,
            "effect_size_label": interpret_effect_size(cohens_d),
            "is_significant": is_significant,
            "alpha": self.alpha,
        }
        result["interpretation"] = self._interpret_difference(
            p_value, float(np.mean(diff)), cohens_d, model1_name, model2_name
        )
        self.logger.info("Paired t-test %s vs %s: t=%.3f, p=%.4f", model1_name, model2_name, t_stat, p_value)
        return result

    def corrected_resampled_t_test(
        self,
        scores1: Sequence[float],
        scores2: Sequence[float],
        n_train: int,
        n_test: int,
        model1_name: str = "Model 1",
        model2_name: str = "Model 2",
    ) -> Dict[str, Any]:
        """Nadeau-Bengio variance-corrected resampled *t*-test.

        Cross-validation folds overlap, so the plain paired *t*-test underestimates
        the variance of the mean difference. The corrected statistic scales the
        variance by ``1/k + n_test/n_train``:

        ``t = mean(d) / sqrt((1/k + n_test/n_train) * var(d))``, with ``k - 1`` df.

        Args:
            scores1: Per-fold scores of the first model.
            scores2: Per-fold scores of the second model (paired).
            n_train: Training-set size of each fold.
            n_test: Test-set size of each fold.
            model1_name: Label for reporting.
            model2_name: Label for reporting.

        Returns:
            Dict with the corrected ``t_statistic``, ``p_value``, ``df``,
            ``mean_difference``, ``is_significant`` and ``interpretation``.

        References:
            Nadeau, C. & Bengio, Y. (2003). Inference for the generalization
            error. *Machine Learning*, 52, 239-281.
            Bouckaert, R. R. & Frank, E. (2004). Evaluating the replicability of
            significance tests for comparing learning algorithms. *PAKDD*, 3-12.
        """
        s1, s2 = _paired_scores(scores1, scores2)
        if n_train <= 0 or n_test <= 0:
            raise ValueError("n_train and n_test must be positive")
        diff = s1 - s2
        k = len(diff)
        var = float(np.var(diff, ddof=1))
        mean_diff = float(np.mean(diff))
        if var == 0.0:
            t_stat, p_value = self._degenerate_t(mean_diff)
        else:
            t_stat = mean_diff / np.sqrt((1.0 / k + n_test / n_train) * var)
            p_value = float(2 * stats.t.sf(abs(t_stat), df=k - 1))
        cohens_d = compute_effect_size(s1, s2)
        result = {
            "test": "corrected_resampled_t_test",
            "model1_name": model1_name,
            "model2_name": model2_name,
            "mean_difference": mean_diff,
            "t_statistic": float(t_stat),
            "df": k - 1,
            "p_value": p_value,
            "variance_correction": 1.0 / k + n_test / n_train,
            "cohens_d": cohens_d,
            "is_significant": bool(p_value < self.alpha),
            "alpha": self.alpha,
            "interpretation": self._interpret_difference(
                p_value, mean_diff, cohens_d, model1_name, model2_name
            ),
        }
        self.logger.info(
            "Corrected t-test %s vs %s: t=%.3f, p=%.4f", model1_name, model2_name, t_stat, p_value
        )
        return result

    def wilcoxon_test(
        self,
        scores1: Sequence[float],
        scores2: Sequence[float],
        model1_name: str = "Model 1",
        model2_name: str = "Model 2",
    ) -> Dict[str, Any]:
        """Wilcoxon signed-rank test on paired scores.

        Non-parametric alternative to the paired *t*-test: ranks the absolute
        differences and compares the rank sums of positive and negative
        differences. Zero differences are discarded (Wilcoxon's method).
        Demšar (2006) recommends it for comparing two classifiers across
        multiple datasets.

        Args:
            scores1: Scores of the first model.
            scores2: Paired scores of the second model.
            model1_name: Label for reporting.
            model2_name: Label for reporting.

        Returns:
            Dict with the ``statistic`` (smaller rank sum ``W``), ``p_value``,
            ``n_nonzero`` differences, the rank-biserial correlation ``effect_size``
            (Kerby, 2014), ``is_significant`` and ``interpretation``.

        References:
            Wilcoxon, F. (1945). Individual comparisons by ranking methods.
            *Biometrics Bulletin*, 1(6), 80-83.
            Demšar, J. (2006). Statistical comparisons of classifiers over
            multiple data sets. *JMLR*, 7, 1-30.
        """
        s1, s2 = _paired_scores(scores1, scores2)
        diff = s1 - s2
        nonzero = diff[diff != 0]
        if len(nonzero) == 0:
            statistic, p_value, r_rb = 0.0, 1.0, 0.0
        else:
            res = stats.wilcoxon(s1, s2, zero_method="wilcox", alternative="two-sided")
            statistic, p_value = float(res.statistic), float(res.pvalue)
            ranks = stats.rankdata(np.abs(nonzero))
            w_pos = float(np.sum(ranks[nonzero > 0]))
            w_neg = float(np.sum(ranks[nonzero < 0]))
            r_rb = (w_pos - w_neg) / (w_pos + w_neg)
        result = {
            "test": "wilcoxon_signed_rank",
            "model1_name": model1_name,
            "model2_name": model2_name,
            "median_difference": float(np.median(diff)),
            "mean_difference": float(np.mean(diff)),
            "statistic": statistic,
            "p_value": p_value,
            "n_nonzero": len(nonzero),
            "effect_size": float(r_rb),
            "is_significant": bool(p_value < self.alpha),
            "alpha": self.alpha,
            "interpretation": self._interpret_difference(
                p_value, float(np.median(diff)), r_rb, model1_name, model2_name
            ),
        }
        self.logger.info("Wilcoxon %s vs %s: W=%.3f, p=%.4f", model1_name, model2_name, statistic, p_value)
        return result

    # ------------------------------------------------------------------ #
    # McNemar's test on paired predictions
    # ------------------------------------------------------------------ #
    def mcnemar_test(
        self,
        y_true: Sequence[Any],
        y_pred1: Sequence[Any],
        y_pred2: Sequence[Any],
        model1_name: str = "Model 1",
        model2_name: str = "Model 2",
        exact: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """McNemar's test for two classifiers evaluated on the same samples.

        Builds the 2x2 table of correct/incorrect decisions

        ==============  =================  =================
        \\               model 2 correct    model 2 wrong
        ==============  =================  =================
        model 1 correct  ``a``              ``b``
        model 1 wrong    ``c``              ``d``
        ==============  =================  =================

        and tests ``H0: b == c`` (equal disagreement counts). Uses the
        continuity-corrected chi-square statistic ``(|b - c| - 1)^2 / (b + c)``
        with one degree of freedom (Edwards, 1948) when ``b + c >= 25``, and the
        exact two-sided binomial test otherwise (Dietterich, 1998).

        Args:
            y_true: Ground-truth labels.
            y_pred1: Predictions of the first classifier.
            y_pred2: Predictions of the second classifier.
            model1_name: Label for reporting.
            model2_name: Label for reporting.
            exact: Force the exact (``True``) or chi-square (``False``) variant;
                ``None`` selects automatically.

        Returns:
            Dict with the ``contingency_table``, ``statistic`` (chi-square or
            ``None`` for the exact test), ``p_value``, ``method``, per-model accuracies,
            ``is_significant`` and ``interpretation``.

        References:
            McNemar, Q. (1947). Note on the sampling error of the difference between
            correlated proportions or percentages. *Psychometrika*, 12(2), 153-157.
            Dietterich, T. G. (1998). *Neural Computation*, 10(7), 1895-1923.
        """
        y_true, y_pred1 = validate_targets(y_true, y_pred1)
        y_true, y_pred2 = validate_targets(y_true, y_pred2)
        correct1 = y_pred1 == y_true
        correct2 = y_pred2 == y_true
        a = int(np.sum(correct1 & correct2))
        b = int(np.sum(correct1 & ~correct2))
        c = int(np.sum(~correct1 & correct2))
        d = int(np.sum(~correct1 & ~correct2))
        table = np.array([[a, b], [c, d]])

        n_discordant = b + c
        use_exact = (n_discordant < 25) if exact is None else bool(exact)
        if n_discordant == 0:
            statistic, p_value, method = (None if use_exact else 0.0), 1.0, ("exact" if use_exact else "chi2")
        elif use_exact:
            statistic = None
            p_value = float(min(1.0, 2.0 * stats.binom.cdf(min(b, c), n_discordant, 0.5)))
            method = "exact"
        else:
            statistic = float((abs(b - c) - 1) ** 2 / n_discordant)
            p_value = float(stats.chi2.sf(statistic, df=1))
            method = "chi2"

        is_significant = bool(p_value < self.alpha)
        if not is_significant:
            interpretation = (
                f"No significant difference in error rates between {model1_name} and "
                f"{model2_name} (p={p_value:.4f})"
            )
        else:
            better = model1_name if b > c else model2_name
            interpretation = f"{better} makes significantly fewer errors (b={b}, c={c}, p={p_value:.4f})"
        result = {
            "test": "mcnemar",
            "method": method,
            "model1_name": model1_name,
            "model2_name": model2_name,
            "contingency_table": table,
            "n_discordant": n_discordant,
            "model1_accuracy": float(np.mean(correct1)),
            "model2_accuracy": float(np.mean(correct2)),
            "statistic": statistic,
            "chi2_statistic": statistic,
            "p_value": p_value,
            "is_significant": is_significant,
            "alpha": self.alpha,
            "interpretation": interpretation,
        }
        self.logger.info("McNemar %s vs %s (%s): p=%.4f", model1_name, model2_name, method, p_value)
        return result

    # ------------------------------------------------------------------ #
    # Multiple-model tests
    # ------------------------------------------------------------------ #
    def friedman_test(
        self,
        scores: Union[np.ndarray, Sequence[Sequence[float]]],
        model_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Friedman test for ``k >= 3`` models evaluated on ``N`` datasets/folds.

        Ranks the models within each row (rank 1 = best score) and tests whether
        the average ranks differ. Reports both the classical chi-square statistic
        and the less conservative Iman-Davenport *F* statistic

        ``F_F = (N - 1) chi2_F / (N (k - 1) - chi2_F)`` ~ ``F(k - 1, (k - 1)(N - 1))``.

        Args:
            scores: Array of shape ``(n_datasets, n_models)``.
            model_names: Column labels; defaults to ``model_0..``.

        Returns:
            Dict with ``chi2_statistic``, ``p_value`` (chi-square), ``f_statistic``,
            ``f_p_value`` (Iman-Davenport), ``average_ranks``, ``ranking``,
            ``n_datasets``, ``n_models``, ``is_significant`` and ``interpretation``.

        Raises:
            ValueError: If fewer than three models or two datasets are given.

        References:
            Friedman, M. (1937). The use of ranks to avoid the assumption of normality
            implicit in the analysis of variance. *JASA*, 32(200), 675-701.
            Iman, R. L. & Davenport, J. M. (1980). Approximations of the critical
            region of the Friedman statistic. *Comm. in Statistics*, 9(6), 571-595.
            Demšar, J. (2006). *JMLR*, 7, 1-30.
        """
        mat = np.asarray(scores, dtype=float)
        if mat.ndim != 2:
            raise ValueError("scores must be a 2-D array of shape (n_datasets, n_models)")
        n, k = mat.shape
        if k < 3:
            raise ValueError("Friedman test requires at least three models; use a paired test for two")
        if n < 2:
            raise ValueError("Friedman test requires at least two datasets/folds")
        names = list(model_names) if model_names is not None else [f"model_{i}" for i in range(k)]
        if len(names) != k:
            raise ValueError("model_names length must match the number of columns")

        ranks = np.apply_along_axis(lambda row: stats.rankdata(-row), 1, mat)
        avg_ranks = ranks.mean(axis=0)
        chi2_stat, p_value = stats.friedmanchisquare(*[mat[:, j] for j in range(k)])
        chi2_stat, p_value = float(chi2_stat), float(p_value)
        denom = n * (k - 1) - chi2_stat
        if denom <= 0:
            f_stat, f_p = float("inf"), 0.0
        else:
            f_stat = (n - 1) * chi2_stat / denom
            f_p = float(stats.f.sf(f_stat, k - 1, (k - 1) * (n - 1)))

        order = np.argsort(avg_ranks)
        ranking = [(names[i], float(avg_ranks[i])) for i in order]
        is_significant = bool(p_value < self.alpha)
        interpretation = (
            f"Average ranks differ significantly (chi2={chi2_stat:.3f}, p={p_value:.4f}); "
            f"best ranked: {ranking[0][0]}"
            if is_significant
            else f"No significant difference among the {k} models (chi2={chi2_stat:.3f}, p={p_value:.4f})"
        )
        self.logger.info("Friedman test over %d models x %d datasets: p=%.4f", k, n, p_value)
        return {
            "test": "friedman",
            "n_datasets": n,
            "n_models": k,
            "model_names": names,
            "chi2_statistic": chi2_stat,
            "p_value": p_value,
            "f_statistic": float(f_stat),
            "f_p_value": f_p,
            "average_ranks": dict(zip(names, avg_ranks.astype(float).tolist())),
            "ranking": ranking,
            "is_significant": is_significant,
            "alpha": self.alpha,
            "interpretation": interpretation,
        }

    def nemenyi_post_hoc(
        self,
        scores: Union[np.ndarray, Sequence[Sequence[float]]],
        model_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Nemenyi post-hoc test after a significant Friedman test.

        Two models differ significantly when their average ranks differ by at
        least the critical difference ``CD = q_alpha * sqrt(k (k + 1) / (6 N))``.
        Also reports the normal-approximation *z* scores of each rank difference
        and their Holm-adjusted p-values (Demšar, 2006, §3.2.2).

        Args:
            scores: Array of shape ``(n_datasets, n_models)``.
            model_names: Column labels.

        Returns:
            Dict with ``critical_difference``, ``q_alpha``, ``average_ranks`` and a
            ``pairwise`` mapping ``"A vs B" -> {rank_difference, z, p_value,
            p_value_holm, significant}``.

        Raises:
            ValueError: If ``alpha`` is not 0.05/0.10 or ``k`` exceeds the tabulated range.

        References:
            Nemenyi, P. (1963). *Distribution-free multiple comparisons*. PhD thesis, Princeton.
            Demšar, J. (2006). *JMLR*, 7, 1-30 (Table 5).
        """
        mat = np.asarray(scores, dtype=float)
        n, k = mat.shape
        names = list(model_names) if model_names is not None else [f"model_{i}" for i in range(k)]
        table = NEMENYI_Q_ALPHA.get(round(self.alpha, 2))
        if table is None or k not in table:
            raise ValueError(
                "Nemenyi critical values are tabulated for alpha in {0.05, 0.10} and 2 <= k <= 10"
            )
        q_alpha = table[k]
        ranks = np.apply_along_axis(lambda row: stats.rankdata(-row), 1, mat)
        avg_ranks = ranks.mean(axis=0)
        se = np.sqrt(k * (k + 1) / (6.0 * n))
        cd = q_alpha * se

        pairs = list(itertools.combinations(range(k), 2))
        z_scores = [abs(avg_ranks[i] - avg_ranks[j]) / se for i, j in pairs]
        raw_p = [float(2 * stats.norm.sf(z)) for z in z_scores]
        holm_p = holm_bonferroni(raw_p)
        pairwise: Dict[str, Dict[str, Any]] = {}
        for (i, j), z, p, ph in zip(pairs, z_scores, raw_p, holm_p):
            rd = float(abs(avg_ranks[i] - avg_ranks[j]))
            pairwise[f"{names[i]} vs {names[j]}"] = {
                "rank_difference": rd,
                "z": float(z),
                "p_value": p,
                "p_value_holm": float(ph),
                "significant": bool(rd > cd),
            }
        return {
            "test": "nemenyi",
            "q_alpha": q_alpha,
            "critical_difference": float(cd),
            "average_ranks": dict(zip(names, avg_ranks.astype(float).tolist())),
            "pairwise": pairwise,
            "alpha": self.alpha,
        }

    def compare_multiple_models(
        self,
        scores: Dict[str, Sequence[float]],
        test: str = "auto",
    ) -> Dict[str, Any]:
        """Compare two or more models from paired score vectors.

        * Two models: paired *t*-test (``test="t"``/``"auto"``) or Wilcoxon (``"wilcoxon"``).
        * Three or more: Friedman omnibus test; if significant, the Nemenyi
          post-hoc test and Holm-adjusted pairwise tests are added.

        Args:
            scores: Mapping ``model_name -> per-fold scores`` (equal lengths).
            test: ``"auto"``, ``"t"`` or ``"wilcoxon"`` for the pairwise procedure.

        Returns:
            Dict with ``method``, ``p_value``, ``is_significant``, ``best_model``,
            ``omnibus`` (full result of the primary test), ``pairwise`` results
            with ``p_value_holm`` and ``post_hoc`` (Nemenyi, ``k >= 3`` only).

        Raises:
            ValueError: For fewer than two models or unequal score lengths.
        """
        names = list(scores)
        if len(names) < 2:
            raise ValueError("At least two models are required for a comparison")
        arrays = [ensure_numpy_array(scores[n]).astype(float).ravel() for n in names]
        if len({len(a) for a in arrays}) != 1:
            raise ValueError("All score vectors must have the same length (paired design)")
        mat = np.column_stack(arrays)
        means = mat.mean(axis=0)
        best_model = names[int(np.argmax(means))]

        pair_fn: Callable[..., Dict[str, Any]] = (
            self.wilcoxon_test if test == "wilcoxon" else self.paired_t_test
        )
        pairs = list(itertools.combinations(range(len(names)), 2))
        pairwise = {
            f"{names[i]} vs {names[j]}": pair_fn(mat[:, i], mat[:, j], names[i], names[j]) for i, j in pairs
        }
        holm = holm_bonferroni([r["p_value"] for r in pairwise.values()])
        for r, ph in zip(pairwise.values(), holm):
            r["p_value_holm"] = float(ph)
            r["is_significant_holm"] = bool(ph < self.alpha)

        if len(names) == 2:
            omnibus = next(iter(pairwise.values()))
            return {
                "method": omnibus["test"],
                "model_names": names,
                "means": dict(zip(names, means.astype(float).tolist())),
                "best_model": best_model,
                "p_value": omnibus["p_value"],
                "is_significant": omnibus["is_significant"],
                "omnibus": omnibus,
                "pairwise": pairwise,
                "post_hoc": None,
                "alpha": self.alpha,
            }

        omnibus = self.friedman_test(mat, names)
        post_hoc = None
        if omnibus["is_significant"]:
            try:
                post_hoc = self.nemenyi_post_hoc(mat, names)
            except ValueError as exc:  # alpha or k outside the tabulated range
                self.logger.warning("Nemenyi post-hoc unavailable: %s", exc)
        return {
            "method": "friedman",
            "model_names": names,
            "means": dict(zip(names, means.astype(float).tolist())),
            "best_model": best_model,
            "p_value": omnibus["p_value"],
            "is_significant": omnibus["is_significant"],
            "omnibus": omnibus,
            "pairwise": pairwise,
            "post_hoc": post_hoc,
            "alpha": self.alpha,
        }

    # ------------------------------------------------------------------ #
    # Bootstrap
    # ------------------------------------------------------------------ #
    def bootstrap_confidence_interval(
        self,
        data: Sequence[float],
        statistic: Callable[[np.ndarray], float] = np.mean,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        method: str = "percentile",
    ) -> Dict[str, Any]:
        """Bootstrap confidence interval for a statistic of ``data``.

        Thin wrapper around
        :func:`sklearn_mastery.evaluation.utils.bootstrap_confidence_interval`
        seeded with this tester's ``random_state``. See that function for the
        available methods (percentile, basic, BCa) and references.

        Args:
            data: One-dimensional sample (e.g. per-fold scores).
            statistic: Function mapping a sample to a scalar.
            n_bootstrap: Number of resamples.
            confidence_level: Coverage probability.
            method: ``"percentile"``, ``"basic"`` or ``"bca"``.

        Returns:
            Dict with ``statistic``, ``ci_lower``, ``ci_upper`` and metadata.
        """
        return bootstrap_confidence_interval(
            data,
            statistic=statistic,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            method=method,
            random_state=self.random_state,
        )

    def bootstrap_test(
        self,
        metric_func: Callable[[np.ndarray, np.ndarray], float],
        y_true: Sequence[Any],
        y_pred1: Sequence[Any],
        y_pred2: Sequence[Any],
        n_bootstrap: int = 1000,
        model1_name: str = "Model 1",
        model2_name: str = "Model 2",
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """Paired bootstrap test for the difference of a metric between two models.

        Resamples test instances with replacement, recomputes
        ``delta* = metric(y, pred1) - metric(y, pred2)`` on each resample and
        estimates the two-sided achieved significance level as the fraction of
        resamples with ``|delta* - delta| >= |delta|`` (the bootstrap distribution
        shifted to the null). The percentile interval of ``delta*`` is also returned.

        Args:
            metric_func: Metric ``f(y_true, y_pred) -> float`` (e.g. ``accuracy_score``).
            y_true: Ground truth.
            y_pred1: Predictions of the first model.
            y_pred2: Predictions of the second model.
            n_bootstrap: Number of resamples.
            model1_name: Label for reporting.
            model2_name: Label for reporting.
            confidence_level: Coverage of the interval on the difference.

        Returns:
            Dict with ``observed_difference``, ``mean_difference`` (bootstrap mean),
            ``std_difference``, ``ci_lower``, ``ci_upper``, ``p_value``,
            ``is_significant``, ``n_bootstrap`` and ``interpretation``.

        References:
            Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*, ch. 16.
            Koehn, P. (2004). Statistical significance tests for machine translation
            evaluation. *EMNLP*, 388-395.
            Berg-Kirkpatrick, T., Burkett, D. & Klein, D. (2012). An empirical investigation
            of statistical significance in NLP. *EMNLP*, 995-1005.
        """
        y_true, y_pred1 = validate_targets(y_true, y_pred1)
        y_true, y_pred2 = validate_targets(y_true, y_pred2)
        rng = make_rng(self.random_state)
        n = len(y_true)
        observed = float(metric_func(y_true, y_pred1) - metric_func(y_true, y_pred2))
        diffs = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            diffs[b] = metric_func(y_true[idx], y_pred1[idx]) - metric_func(y_true[idx], y_pred2[idx])
        centered = diffs - observed
        p_value = float(np.mean(np.abs(centered) >= abs(observed)))
        alpha = 1.0 - confidence_level
        lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        is_significant = bool(p_value < self.alpha)
        result = {
            "test": "paired_bootstrap",
            "model1_name": model1_name,
            "model2_name": model2_name,
            "observed_difference": observed,
            "mean_difference": float(np.mean(diffs)),
            "std_difference": float(np.std(diffs, ddof=1)) if n_bootstrap > 1 else float("nan"),
            "ci_lower": float(lo),
            "ci_upper": float(hi),
            "confidence_level": confidence_level,
            "p_value": p_value,
            "is_significant": is_significant,
            "alpha": self.alpha,
            "n_bootstrap": n_bootstrap,
            "interpretation": self._interpret_difference(p_value, observed, None, model1_name, model2_name),
        }
        self.logger.info(
            "Bootstrap test %s vs %s: delta=%.4f, p=%.4f", model1_name, model2_name, observed, p_value
        )
        return result

    def compute_confidence_intervals(
        self, scores: Sequence[float], confidence: float = 0.95
    ) -> Dict[str, float]:
        """Student-*t* confidence interval for the mean of fold scores.

        ``mean +/- t_{1-alpha/2, n-1} * s / sqrt(n)``; appropriate when the
        per-fold scores are approximately normal.

        Args:
            scores: Fold scores.
            confidence: Coverage probability.

        Returns:
            Dict with ``mean``, ``std_error``, ``lower_bound``, ``upper_bound``,
            ``confidence_level`` and ``margin_error``.
        """
        arr = ensure_numpy_array(scores).astype(float).ravel()
        if len(arr) < 2:
            raise ValueError("At least two scores are required for a confidence interval")
        mean_score = float(np.mean(arr))
        std_error = float(stats.sem(arr))
        t_value = float(stats.t.ppf((1 + confidence) / 2, len(arr) - 1))
        margin = t_value * std_error
        return {
            "mean": mean_score,
            "std_error": std_error,
            "lower_bound": mean_score - margin,
            "upper_bound": mean_score + margin,
            "confidence_level": confidence,
            "margin_error": margin,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _degenerate_t(mean_diff: float) -> tuple[float, float]:
        """(t, p) when the paired differences have zero variance."""
        if np.isclose(mean_diff, 0.0):
            return 0.0, 1.0
        return float(np.copysign(np.inf, mean_diff)), 0.0

    def _interpret_difference(
        self,
        p_value: float,
        difference: float,
        effect_size: Optional[float],
        model1_name: str,
        model2_name: str,
    ) -> str:
        """Build a one-line interpretation of a two-model comparison."""
        if p_value >= self.alpha:
            return f"No significant difference between {model1_name} and {model2_name} (p={p_value:.4f})"
        better, worse = (model1_name, model2_name) if difference > 0 else (model2_name, model1_name)
        if effect_size is None:
            return f"{better} significantly outperforms {worse} (p={p_value:.4f})"
        label = interpret_effect_size(effect_size)
        return (
            f"{better} significantly outperforms {worse} with a {label} effect size "
            f"(p={p_value:.4f}, effect={abs(effect_size):.3f})"
        )


__all__: List[str] = ["NEMENYI_Q_ALPHA", "StatisticalTester", "ValidationCurveAnalyzer"]
