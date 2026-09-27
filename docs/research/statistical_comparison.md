# Statistical comparison

`sklearn_mastery.research.comparison` implements the procedures recommended
for comparing learning algorithms. Which one to use depends on **how many
algorithms** you compare and on **how many datasets** the scores come from.

## Which test?

| Situation | Procedure | Function |
|---|---|---|
| $k \ge 2$ algorithms on $N \ge 2$ datasets: do they differ at all? | Friedman test with the Iman-Davenport $F$ correction | [`friedman_test`][sklearn_mastery.research.comparison.friedman_test] |
| ...which pairs differ (all-vs-all, no control) | Nemenyi post-hoc test and critical-difference diagram | [`nemenyi_critical_difference`][sklearn_mastery.research.comparison.nemenyi_critical_difference], [`nemenyi_posthoc`][sklearn_mastery.research.comparison.nemenyi_posthoc], [`plot_critical_difference_diagram`][sklearn_mastery.research.comparison.plot_critical_difference_diagram] |
| ...which pairs differ, with more power than Nemenyi | Pairwise Wilcoxon signed-rank tests with Holm step-down correction | [`wilcoxon_holm`][sklearn_mastery.research.comparison.wilcoxon_holm], [`holm_correction`][sklearn_mastery.research.comparison.holm_correction] |
| 2 algorithms on **one** dataset, scores from (repeated) $k$-fold CV | Nadeau-Bengio corrected resampled $t$-test | [`corrected_resampled_ttest`][sklearn_mastery.research.comparison.corrected_resampled_ttest] |
| Same, but you want $P(\text{better})$, $P(\text{equivalent})$, $P(\text{worse})$ | Bayesian correlated $t$-test with a region of practical equivalence | [`bayesian_correlated_ttest`][sklearn_mastery.research.comparison.bayesian_correlated_ttest] |

Two things to avoid, both discussed by Demšar (2006):

- **A paired $t$-test across datasets.** Scores from different datasets are
  not commensurable and are rarely normally distributed; use ranks.
- **A plain $t$-test across cross-validation folds.** Training sets overlap,
  so the fold scores are positively correlated and the naive variance is
  badly underestimated. Use the corrected tests.

All multi-dataset functions accept a `datasets x estimators` matrix, which is
exactly what
[`BenchmarkResult.score_matrix`][sklearn_mastery.research.benchmark.BenchmarkResult.score_matrix]
returns. Pass `higher_is_better=False` for error metrics.

## Friedman test with the Iman-Davenport correction

Rank the $k$ estimators on each of the $N$ datasets (rank 1 = best, ties get
average ranks) and let $R_j$ be the mean rank of estimator $j$. Under the null
hypothesis that all estimators are equivalent, the Friedman statistic

$$
\chi_F^2 = \frac{12N}{k(k+1)} \left[ \sum_{j=1}^{k} R_j^2 - \frac{k(k+1)^2}{4} \right]
$$

follows a $\chi^2$ distribution with $k-1$ degrees of freedom. Iman and
Davenport (1980) showed that this is conservative and proposed

$$
F_F = \frac{(N-1)\,\chi_F^2}{N(k-1) - \chi_F^2} \sim F\big(k-1,\;(k-1)(N-1)\big),
$$

which is what [`FriedmanResult.reject`][sklearn_mastery.research.comparison.FriedmanResult.reject]
uses.

```python
import numpy as np
import pandas as pd
from sklearn_mastery.research import friedman_test

scores = pd.DataFrame(
    {"a": [0.90, 0.85, 0.88, 0.92, 0.80],
     "b": [0.88, 0.80, 0.85, 0.90, 0.78],
     "c": [0.80, 0.75, 0.82, 0.85, 0.70]},
    index=["d1", "d2", "d3", "d4", "d5"],
)
fr = friedman_test(scores)
print(fr.average_ranks)                 # a=1.0, b=2.0, c=3.0
print(fr.statistic, fr.p_value)         # chi-square
print(fr.iman_davenport_statistic, fr.iman_davenport_p_value)
print(fr.reject(alpha=0.05))
```

When every dataset ranks the estimators identically the denominator of $F_F$
is zero; the function then reports $F_F = \infty$ and $p = 0$.

## Nemenyi post-hoc test and critical difference

If the Friedman test rejects, two estimators differ significantly when their
average ranks differ by at least the critical difference

$$
\mathrm{CD} = q_\alpha \sqrt{\frac{k(k+1)}{6N}},
$$

where $q_\alpha$ is the Studentised range statistic for $k$ groups and
infinite degrees of freedom divided by $\sqrt{2}$. The implementation
computes $q_\alpha$ from `scipy.stats.studentized_range` instead of a lookup
table, so any $k$ and $\alpha$ are supported.

```python
from sklearn_mastery.research import nemenyi_critical_difference, nemenyi_posthoc

cd = nemenyi_critical_difference(fr.n_estimators, fr.n_datasets, alpha=0.05)
p_matrix = nemenyi_posthoc(scores)      # symmetric estimator x estimator p-values
print(round(cd, 3))
print(p_matrix.round(3))
```

### Critical-difference diagram

A CD diagram (Demšar, 2006, Fig. 1) places every estimator on a rank axis and
connects groups whose average ranks are within one CD of each other:

```python
import matplotlib
matplotlib.use("Agg")
from sklearn_mastery.research import plot_critical_difference_diagram

ax = plot_critical_difference_diagram(fr.average_ranks, cd, title="accuracy, alpha = 0.05")
ax.figure.savefig("cd_diagram.png", dpi=150, bbox_inches="tight")
```

Rank 1 is drawn on the right, the best estimator's label is bold, and the
thick bars are the cliques of estimators that are *not* significantly
different. The CLI command `sklearn-mastery compare RESULTS_DIR --cd-diagram
cd.png` produces the same figure from a saved benchmark.

## Pairwise Wilcoxon signed-rank tests with Holm correction

The Nemenyi test has low power because it controls the error over all
$k(k-1)/2$ comparisons at once. When no algorithm is designated as the
control, Demšar (2006) and Benavoli et al. (2016) recommend pairwise Wilcoxon
signed-rank tests with a step-down correction instead.

For each pair, the Wilcoxon test ranks the absolute differences
$d_i = s_i^{(a)} - s_i^{(b)}$ across datasets and compares the sum of ranks of
positive and negative differences. The $m$ resulting p-values are then
adjusted with Holm's procedure: sort them $p_{(1)} \le \dots \le p_{(m)}$ and
set

$$
\tilde p_{(i)} = \min\Big(1,\; \max_{j \le i} \big[(m - j + 1)\, p_{(j)}\big]\Big),
$$

which controls the family-wise error rate at $\alpha$ while being uniformly
more powerful than Bonferroni.

```python
from sklearn_mastery.research import holm_correction, wilcoxon_holm

pairs = wilcoxon_holm(scores, alpha=0.05)
print(pairs[["estimator_a", "estimator_b", "p_value", "p_adjusted", "significant", "winner"]])

print(holm_correction([0.01, 0.04, 0.03]))   # adjusted p-values in the original order
```

`winner` is the estimator with the larger median difference (or `None` for
ties) and `significant` compares the *adjusted* p-value with `alpha`. With
fewer than about ten datasets the Wilcoxon test cannot reach $p < 0.05$ after
correction; collect more datasets rather than lowering `alpha`.

## Corrected resampled $t$-test (Nadeau & Bengio)

For two algorithms on one dataset with $n$ paired fold scores
$d_i = s_i^{(a)} - s_i^{(b)}$, the naive paired $t$-statistic uses
$\hat\sigma^2 / n$ as the variance of the mean difference. Nadeau and Bengio
(2003) showed that, because training sets overlap, the variance should be
inflated by the ratio of test to training set size:

$$
t = \frac{\bar d}{\sqrt{\left(\frac{1}{n} + \frac{n_{\text{test}}}{n_{\text{train}}}\right)\hat\sigma^2}}
\qquad \text{with } t \sim t_{n-1} \text{ under } H_0 .
$$

For plain $k$-fold cross-validation $n_{\text{test}} / n_{\text{train}} = 1/(k-1)$,
which you can pass as `n_splits=k`; otherwise give `n_train` and `n_test`
explicitly. With $r$ repeats of $k$-fold, $n = rk$ and the ratio stays
$1/(k-1)$.

```python
from sklearn_mastery.research import corrected_resampled_ttest

rng = np.random.default_rng(0)
a = 0.90 + 0.02 * rng.standard_normal(10)     # 2 repeats x 5 folds
b = 0.88 + 0.02 * rng.standard_normal(10)
t, p = corrected_resampled_ttest(a, b, n_splits=5)
print(round(t, 3), round(p, 4))
```

Taken from a benchmark, where `paired_scores` returns fold scores aligned
across estimators because every estimator saw the same splits:

```python
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn_mastery.research import BenchmarkSuite

result = BenchmarkSuite(
    estimators={"rf": RandomForestClassifier(n_estimators=50, random_state=0),
                "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))},
    datasets={"wine": load_wine(return_X_y=True)},
    scoring="accuracy", n_splits=5, n_repeats=2, random_state=0,
).run()
paired = result.paired_scores("accuracy", "wine")          # 10 rows (repeat, fold) x 2 estimators
t, p = corrected_resampled_ttest(paired["rf"], paired["logreg"], n_splits=5)
print(round(t, 3), round(p, 4))
```

## Bayesian correlated $t$-test

A $p$-value answers "how surprising is this difference if the algorithms were
identical?", which is rarely the question of interest. Corani and Benavoli
(2015) place the same correlated-variance model in a Bayesian setting: the
posterior of the mean difference $\mu$ is a Student distribution

$$
\mu \sim \mathrm{St}\!\left(n-1,\; \bar d,\; \left(\tfrac{1}{n} + \tfrac{n_{\text{test}}}{n_{\text{train}}}\right)\hat\sigma^2\right),
$$

and one reports the probability mass in three regions defined by a **region
of practical equivalence** (ROPE) of half-width $r$ on the metric scale:

$$
P(\text{left}) = P(\mu < -r), \qquad
P(\text{rope}) = P(-r \le \mu \le r), \qquad
P(\text{right}) = P(\mu > r).
$$

```python
from sklearn_mastery.research import bayesian_correlated_ttest

post = bayesian_correlated_ttest(a, b, rope=0.01, n_splits=5)   # 0.01 = one accuracy point
print(round(post.p_left, 3), round(post.p_rope, 3), round(post.p_right, 3))
print(post.mean_difference, post.decision())    # 'A > B', 'A < B', 'equivalent' or 'undecided'
```

`decision(threshold=0.95)` returns a verdict only when one of the three
probabilities exceeds the threshold; otherwise the comparison is `'undecided'`,
which is often the honest answer for ten folds. Benavoli et al. (2017)
recommend $r = 0.01$ for accuracy.

## Legacy interface

[`StatisticalTester`][sklearn_mastery.evaluation.statistical_tests.StatisticalTester]
in the evaluation package offers dictionary-returning versions of the paired
*t*, Wilcoxon, McNemar, corrected resampled *t*, Friedman and Nemenyi tests plus
bootstrap confidence intervals. It predates the research layer and is kept for
the examples; new code should use the functions on this page.

## References

- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data
  sets. *Journal of Machine Learning Research*, 7, 1-30.
- Iman, R. L., & Davenport, J. M. (1980). Approximations of the critical region
  of the Friedman statistic. *Communications in Statistics*, 9(6), 571-595.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure.
  *Scandinavian Journal of Statistics*, 6(2), 65-70.
- Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error.
  *Machine Learning*, 52, 239-281.
- Corani, G., & Benavoli, A. (2015). A Bayesian approach for comparing
  cross-validated algorithms on multiple data sets. *Machine Learning*, 100,
  285-304.
- Benavoli, A., Corani, G., & Mangili, F. (2016). Should we really use post-hoc
  tests based on mean-ranks? *Journal of Machine Learning Research*, 17(5), 1-10.
- Benavoli, A., Corani, G., Demšar, J., & Zaffalon, M. (2017). Time for a
  change: a tutorial for comparing multiple classifiers through Bayesian
  analysis. *Journal of Machine Learning Research*, 18(77), 1-36.
