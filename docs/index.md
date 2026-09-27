# sklearn-mastery

**A research-grade toolkit for reproducible machine-learning experiments with scikit-learn.**

`sklearn-mastery` turns the usual "train a few models and eyeball the numbers"
workflow into a defensible experimental protocol. The package provides:

- **A benchmark harness** ([`BenchmarkSuite`][sklearn_mastery.research.benchmark.BenchmarkSuite])
  that evaluates many estimators on many datasets under *identical* repeated
  stratified splits (so paired tests are valid), with optional nested
  hyperparameter tuning, tidy long-form results and a hashed provenance
  manifest for every run.
- **The statistical procedures the literature recommends** for comparing
  learning algorithms: Friedman with the Iman-Davenport correction, Nemenyi
  post-hoc tests and critical-difference diagrams (Demšar, 2006), pairwise
  Wilcoxon with Holm correction, the Nadeau-Bengio corrected resampled
  *t*-test, and the Bayesian correlated *t*-test with a region of practical
  equivalence (Benavoli et al., 2017).
- **Diagnostics beyond accuracy**: calibration (ECE, MCE, Murphy's Brier
  decomposition, reliability diagrams) and bootstrap bias-variance
  decomposition (Domingos, 2000).
- **Publication-ready tables** in Markdown and LaTeX (mean ± std, best per
  dataset emphasised, average-rank row).
- **A consistent, sklearn-compatible model layer** (classification,
  regression, clustering, dimensionality reduction, ensembles with diversity
  analysis), **composable preprocessing pipelines**, **deterministic synthetic
  data generators** and **data validation / drift detection**.

## Installation

```bash
git clone https://github.com/SatvikPraveen/Sklearn-Mastery.git
cd Sklearn-Mastery
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + test/lint tooling
pip install -e ".[all]"          # every optional extra (boosting, tuning, ...)
```

Requires Python 3.9+ and scikit-learn 1.3+. Heavy optional dependencies
(`xgboost`, `lightgbm`, `optuna`, `shap`, `imbalanced-learn`, `plotly`,
`umap-learn`) are imported lazily; every module works without them. See
[Getting started](getting_started.md) for the extras table.

## Quick start: a defensible model comparison in 20 lines

```python
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn_mastery.research import (
    BenchmarkSuite, friedman_test, nemenyi_critical_difference,
    plot_critical_difference_diagram, results_to_markdown, wilcoxon_holm,
)

suite = BenchmarkSuite(
    estimators={
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "svm": make_pipeline(StandardScaler(), SVC()),
        "rf": RandomForestClassifier(n_estimators=100, random_state=0),
    },
    datasets={name: loader(return_X_y=True)
              for name, loader in [("iris", load_iris), ("wine", load_wine),
                                   ("cancer", load_breast_cancer)]},
    scoring=["accuracy", "f1_macro"], n_splits=5, n_repeats=2, random_state=42,
)
result = suite.run()                                  # tidy DataFrame + provenance manifest
print(results_to_markdown(result.results, "accuracy"))

scores = result.score_matrix("accuracy")              # datasets x estimators
fr = friedman_test(scores)                            # H0: all estimators perform the same
cd = nemenyi_critical_difference(fr.n_estimators, fr.n_datasets, alpha=0.05)
print(fr.average_ranks, fr.iman_davenport_p_value, cd)
print(wilcoxon_holm(scores))                          # pairwise, family-wise error controlled
plot_critical_difference_diagram(fr.average_ranks, cd).figure.savefig("cd.png")
result.save("runs/baseline")                          # results.csv + manifest.json + best_params.json
```

Comparing two algorithms on a **single** dataset? Cross-validation folds are
not independent, so use the corrected tests:

```python
from sklearn_mastery.research import bayesian_correlated_ttest, corrected_resampled_ttest

paired = result.paired_scores("accuracy", "wine")     # (repeat, fold) x estimator
t, p = corrected_resampled_ttest(paired["rf"], paired["svm"], n_splits=5)
post = bayesian_correlated_ttest(paired["rf"], paired["svm"], rope=0.01, n_splits=5)
print(post.p_left, post.p_rope, post.p_right, post.decision())
```

The same workflow is available from the shell (see the [CLI guide](guide/cli.md)):

```bash
sklearn-mastery benchmark --dataset iris --dataset wine --dataset breast_cancer -o runs/demo
sklearn-mastery compare runs/demo --cd-diagram cd.png
```

## Where to go next

| I want to... | Read |
|---|---|
| Install the package and run my first benchmark | [Getting started](getting_started.md) |
| Compare many estimators on many datasets | [Benchmarking](research/benchmarking.md) |
| Pick the right significance test and understand the formulas | [Statistical comparison](research/statistical_comparison.md) |
| Check whether probabilities are trustworthy, or where the error comes from | [Calibration and bias-variance](research/calibration_bias_variance.md) |
| Generate, preprocess and validate data | [Data](guide/data.md) |
| Use the model wrappers, factories, clustering and dimensionality reduction | [Models](guide/models.md), [Ensembles](guide/ensembles.md) |
| Build pipelines and tune hyperparameters | [Pipelines](guide/pipelines.md) |
| Compute metrics, run tests and draw plots | [Evaluation](guide/evaluation.md) |
| Understand how trees, forests and boosting work under the hood | [From scratch](from_scratch.md) |
| Look up a signature | [API reference](api/research.md) |

## Package layout

```
sklearn_mastery/
├── research/          Benchmarking, statistical comparison, calibration, bias-variance,
│                      reporting, reproducibility (the core of the toolkit)
├── data/              SyntheticDataGenerator, DataPreprocessor, encoders, DataValidator,
│                      SchemaValidator, drift detection
├── models/
│   ├── supervised/    Classification and regression wrappers with uniform fit/evaluate/tune API
│   ├── unsupervised/  Clustering (with optimal-k / eps selection) and dimensionality reduction
│   └── ensemble/      Voting, bagging, boosting, stacking, blending + diversity measures
├── pipelines/         Custom transformers, feature unions, pipeline factory, model selection
├── evaluation/        Metrics, statistical tests, cross-validation, analyzers, visualization
├── from_scratch/      Educational NumPy implementations of trees, forests and boosting
├── config/            Pydantic settings (env-overridable) and library-style logging
└── cli.py             `sklearn-mastery` command-line interface
```

Everything is importable from `sklearn_mastery.<subpackage>`; importing the
package has no side effects (no logging configuration, no directory creation).
Output paths derive from `settings.PROJECT_ROOT`, which defaults to the current
working directory and can be overridden with the `SKLEARN_MASTERY_ROOT`
environment variable.
