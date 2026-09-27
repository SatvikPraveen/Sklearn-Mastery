# sklearn-mastery

**A research-grade toolkit for reproducible machine-learning experiments with scikit-learn.**

[![CI](https://github.com/SatvikPraveen/Sklearn-Mastery/actions/workflows/ci.yml/badge.svg)](https://github.com/SatvikPraveen/Sklearn-Mastery/actions/workflows/ci.yml)
[![Docs](https://github.com/SatvikPraveen/Sklearn-Mastery/actions/workflows/docs.yml/badge.svg)](https://satvikpraveen.github.io/Sklearn-Mastery/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn 1.3+](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Cite](https://img.shields.io/badge/cite-CITATION.cff-lightgrey.svg)](CITATION.cff)

`sklearn-mastery` turns the usual "train a few models and eyeball the numbers"
workflow into a defensible experimental protocol. It provides:

- **A benchmark harness** that evaluates many estimators on many datasets under
  *identical* repeated stratified splits (so paired tests are valid), with
  optional nested hyperparameter tuning, tidy long-form results, and a hashed
  provenance manifest for every run.
- **The statistical procedures the literature actually recommends** for
  comparing learning algorithms: Friedman with the Iman–Davenport correction,
  Nemenyi post-hoc and critical-difference diagrams (Demšar, 2006), pairwise
  Wilcoxon with Holm correction, the Nadeau–Bengio corrected resampled *t*-test,
  and the Bayesian correlated *t*-test with a region of practical equivalence
  (Benavoli et al., 2017).
- **Diagnostics beyond accuracy**: calibration (ECE, MCE, Murphy's Brier
  decomposition, reliability diagrams) and bootstrap bias–variance decomposition
  (Domingos, 2000).
- **Publication-ready tables** in Markdown and LaTeX (mean ± std, best per
  dataset emphasised, average-rank row).
- **From-scratch, NumPy-only implementations of the tree ensembles** that most
  scikit-learn tutorials only ever call: CART decision trees, bagging, random
  forests, AdaBoost (SAMME / R2), gradient boosting and an XGBoost-style
  second-order booster, each derived in its docstring, tested against its
  library counterpart, and fully sklearn-compatible.
- **A consistent, sklearn-compatible model layer** (classification, regression,
  clustering, dimensionality reduction, ensembles with diversity analysis),
  **composable preprocessing pipelines**, **deterministic synthetic data
  generators**, and **data validation / drift detection**, all covered by an
  extensive test suite.

**Documentation:** https://satvikpraveen.github.io/Sklearn-Mastery/

---

## Installation

```bash
git clone https://github.com/SatvikPraveen/Sklearn-Mastery.git
cd Sklearn-Mastery
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + test/lint tooling
# optional extras: boosting (xgboost, lightgbm), tuning (optuna), interpret (shap, lime),
#                  imbalanced, tracking (mlflow), viz (plotly), notebooks, docs, all
pip install -e ".[all]"
```

Requires Python 3.9+ and scikit-learn 1.3+. Heavy optional dependencies are
imported lazily; every module works without them.

---

## Quick start: a defensible model comparison in 20 lines

```python
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
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
        "rf": RandomForestClassifier(n_estimators=300, random_state=0),
    },
    datasets={name: loader(return_X_y=True)
              for name, loader in [("iris", load_iris), ("wine", load_wine),
                                   ("cancer", load_breast_cancer), ("digits", load_digits)]},
    scoring=["accuracy", "f1_macro"], n_splits=5, n_repeats=3, random_state=42, n_jobs=-1,
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

The same workflow is available from the shell:

```bash
sklearn-mastery benchmark --dataset iris --dataset wine --dataset breast_cancer -o runs/demo
sklearn-mastery compare runs/demo --cd-diagram cd.png
sklearn-mastery generate-data --dataset-type classification --complexity high -o data.csv
sklearn-mastery train data.csv --algorithm gradient_boosting -o runs/gb
sklearn-mastery info
```

---

## The algorithms, from first principles

Most repositories show `RandomForestClassifier().fit(X, y)`. This one also
shows *what that call does*. `sklearn_mastery.from_scratch` re-implements the
classic tree ensembles in plain NumPy, with the mathematics derived in each
module docstring and every estimator exposing the same interface as its
library counterpart, so you can drop either into a pipeline, a grid search or
the benchmark suite above.

| Algorithm | Scratch class | What it implements |
|---|---|---|
| Decision tree | `DecisionTreeClassifierScratch`, `DecisionTreeRegressorScratch` | CART exact greedy splits (Gini, entropy, squared error, Friedman MSE), sample weights, MDI importances, `export_text`, weakest-link cost-complexity pruning |
| Bagging | `BaggingClassifierScratch`, `BaggingRegressorScratch` | Bootstrap aggregation, feature bagging, soft/hard voting, out-of-bag estimates |
| Random forest | `RandomForestClassifierScratch`, `RandomForestRegressorScratch` | Per-split feature subsampling (the decorrelation trick), OOB score, MDI and permutation importance |
| AdaBoost | `AdaBoostClassifierScratch`, `AdaBoostRegressorScratch` | SAMME with multiclass weight update and early stopping; AdaBoost.R2 with weighted-median prediction |
| Gradient boosting | `GradientBoostingClassifierScratch`, `GradientBoostingRegressorScratch` | Friedman's algorithm with squared/absolute/Huber loss, binomial and multinomial deviance, per-leaf Newton updates, stochastic subsampling |
| XGBoost | `XGBoostClassifierScratch`, `XGBoostRegressorScratch` | Second-order objective with the gain formula, λ/γ/min_child_weight regularisation, sparsity-aware missing-value direction, column subsampling, early stopping, gain/weight/cover importances |

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn_mastery.from_scratch import (
    DecisionTreeClassifierScratch, RandomForestClassifierScratch, XGBoostClassifierScratch,
)

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)

tree = DecisionTreeClassifierScratch(max_depth=3, ccp_alpha=0.01).fit(X_tr, y_tr)
print(tree.export_text())                       # readable if/else rules

rf = RandomForestClassifierScratch(n_estimators=200, oob_score=True, random_state=0).fit(X_tr, y_tr)
print(rf.oob_score_, rf.score(X_te, y_te))      # OOB estimate vs. hold-out

xgb = XGBoostClassifierScratch(n_estimators=300, max_depth=3, learning_rate=0.1,
                               reg_lambda=1.0, gamma=0.1, early_stopping_rounds=20)
xgb.fit(X_tr, y_tr, eval_set=(X_te, y_te))
print(xgb.best_iteration_, xgb.get_feature_importance("gain")[:5])
```

With identical hyperparameters the scratch XGBoost reproduces
`xgboost.XGBClassifier(tree_method="exact")` margins to within 2e-6 on the
breast-cancer data, and the scratch pruned tree yields the same leaf count as
scikit-learn for every `ccp_alpha` tested. Run
`python examples/from_scratch_showcase.py` for a side-by-side comparison
table and figure, or read the derivations in the
[documentation](https://satvikpraveen.github.io/Sklearn-Mastery/from_scratch/).

---

## Package layout

```
sklearn_mastery/
├── research/          Benchmarking, statistical comparison, calibration, bias-variance,
│                      reporting, reproducibility (the core of the toolkit)
├── from_scratch/      NumPy implementations of CART, bagging, random forest, AdaBoost,
│                      gradient boosting and XGBoost, derived and tested against the libraries
├── data/              SyntheticDataGenerator, DataPreprocessor, encoders, DataValidator,
│                      SchemaValidator, drift detection
├── models/
│   ├── supervised/    Classification and regression wrappers with uniform fit/evaluate/tune API
│   ├── unsupervised/  Clustering (with optimal-k / eps selection) and dimensionality reduction
│   └── ensemble/      Voting, bagging, boosting, stacking, blending + diversity measures
├── pipelines/         Custom transformers, feature unions, pipeline factory, model selection
├── evaluation/        Metrics, statistical tests, cross-validation, analyzers, visualization
├── config/            Pydantic settings (env-overridable) and library-style logging
└── cli.py             `sklearn-mastery` command-line interface
```

Everything is importable from `sklearn_mastery.<subpackage>`; importing the
package has no side effects (no logging configuration, no directory creation).
Output paths derive from `settings.PROJECT_ROOT`, which defaults to the current
working directory and can be overridden with `SKLEARN_MASTERY_ROOT`.

---

## Methodology notes

| Question | Recommended procedure | Function |
|---|---|---|
| Do *k* algorithms differ across *N* datasets? | Friedman test with Iman–Davenport *F* correction | `friedman_test` |
| Which pairs differ (no designated control)? | Wilcoxon signed-rank + Holm step-down | `wilcoxon_holm` |
| Visual summary of ranks | Nemenyi critical-difference diagram | `nemenyi_critical_difference`, `plot_critical_difference_diagram` |
| Two algorithms, one dataset, CV folds | Nadeau–Bengio corrected resampled *t*-test | `corrected_resampled_ttest` |
| Same, but with practical-equivalence reasoning | Bayesian correlated *t*-test with ROPE | `bayesian_correlated_ttest` |
| Are probabilities trustworthy? | ECE / MCE, Brier decomposition, reliability diagram | `expected_calibration_error`, `brier_score_decomposition`, `reliability_diagram` |
| Where does the error come from? | Bootstrap bias–variance decomposition | `bias_variance_decomposition` |
| Tuned performance without selection bias | Nested CV (`param_grids=` in `BenchmarkSuite`) | `BenchmarkSuite` |

References: Demšar (2006) *JMLR* 7; Nadeau & Bengio (2003) *Machine Learning*
52; Corani & Benavoli (2015) *Machine Learning* 100; Benavoli et al. (2017)
*JMLR* 18; Cawley & Talbot (2010) *JMLR* 11; Domingos (2000) *ICML*; Guo et al.
(2017) *ICML*. Full citations are in the module docstrings and `CITATION.cff`.

---

## Development

```bash
make install-dev     # editable install + pre-commit hooks
make test            # pytest with coverage (parallel)
make lint            # ruff check + format check
make type-check      # mypy on typed subpackages
make check           # everything CI runs
make docs            # build the documentation site locally
pytest -m slow       # notebooks and example scripts (several minutes)
```

Continuous integration runs linting, type checks, the test matrix (Python
3.10–3.12 on Linux, 3.12 on macOS) and a distribution build on every push and
pull request. See [CONTRIBUTING.md](CONTRIBUTING.md) and
[CHANGELOG.md](CHANGELOG.md).

## Notebooks and examples

`notebooks/` walks through data generation, preprocessing, supervised and
unsupervised learning, ensembles, model selection and advanced techniques.
`examples/real_world_scenarios/` contains domain scripts (finance, healthcare,
manufacturing, marketing, technology) built on the model and evaluation layers,
and `examples/from_scratch_showcase.py` compares the from-scratch ensembles
with their library counterparts. Notebooks are stored without outputs and are
executed end-to-end in the slow test tier (`pytest -m slow`).

## Citing

If this toolkit contributes to your research, please cite it using the metadata
in [`CITATION.cff`](CITATION.cff) (GitHub renders a "Cite this repository"
button from it).

## License

MIT © Satvik Praveen
