---
hide:
  - toc
---

<div class="skm-hero" markdown>

# sklearn-mastery

**Reproducible machine-learning experiments with scikit-learn, from the
statistics that make a comparison defensible down to the algorithms
themselves, re-implemented from first principles.**

[Get started](getting_started.md){ .md-button .md-button--primary }
[Benchmark & compare](research/benchmarking.md){ .md-button }
[GitHub](https://github.com/SatvikPraveen/Sklearn-Mastery){ .md-button }

</div>

<div class="skm-grid" markdown>

<div class="skm-card" markdown>
### Benchmark harness
Many estimators × many datasets under *identical* repeated stratified splits,
optional nested tuning, tidy results and a hashed provenance manifest.
[BenchmarkSuite →](research/benchmarking.md)
</div>

<div class="skm-card" markdown>
### Rigorous comparison
Friedman + Iman-Davenport, Nemenyi critical-difference diagrams, Wilcoxon-Holm,
Nadeau-Bengio corrected and Bayesian correlated *t*-tests with a ROPE.
[Statistical tests →](research/statistical_comparison.md)
</div>

<div class="skm-card" markdown>
### Beyond accuracy
Calibration (ECE, MCE, Brier decomposition, reliability diagrams) and
bootstrap bias-variance decomposition.
[Diagnostics →](research/calibration_bias_variance.md)
</div>

<div class="skm-card" markdown>
### Algorithms from scratch
CART, bagging, random forests, AdaBoost, gradient boosting and an XGBoost-style
booster in plain NumPy, derived in the docstrings and tested against the libraries.
[From scratch →](from_scratch.md)
</div>

<div class="skm-card" markdown>
### Model & pipeline layer
Uniform sklearn-compatible wrappers, ensembles with diversity analysis,
composable transformers, feature unions, factories and Bayesian optimisation.
[Models →](guide/models.md) · [Pipelines →](guide/pipelines.md)
</div>

<div class="skm-card" markdown>
### Data & evaluation
Deterministic synthetic generators, preprocessing, validation and drift
detection; metrics, analyzers and headless visualisation.
[Data →](guide/data.md) · [Evaluation →](guide/evaluation.md)
</div>

</div>

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
