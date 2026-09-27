# Benchmarking

`sklearn_mastery.research.benchmark` evaluates a set of estimators on a set of
datasets under one fixed resampling protocol and returns tidy, long-form
results with provenance. Its design follows three principles:

1. **Identical splits for every estimator.** Splits are generated once per
   (dataset, repeat) and reused for all estimators, so per-fold scores are
   *paired* and the tests in [Statistical comparison](statistical_comparison.md)
   are valid.
2. **Tidy output.** One row per `(dataset, estimator, repeat, fold, metric)`,
   ready for pandas, seaborn or the reporting helpers.
3. **Provenance.** Every run carries a
   [`RunManifest`][sklearn_mastery.research.reproducibility.RunManifest] with
   the seed, a hash of the configuration and an environment snapshot.

## `BenchmarkSuite`

```python
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn_mastery.research import BenchmarkSuite

suite = BenchmarkSuite(
    estimators={
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "rf": RandomForestClassifier(n_estimators=100, random_state=0),
    },
    datasets={
        "iris": load_iris(return_X_y=True),
        "wine": load_wine(return_X_y=True),
        "cancer": lambda: load_breast_cancer(return_X_y=True),   # lazy loaders are fine
    },
    scoring=["accuracy", "f1_macro"],
    n_splits=5,
    n_repeats=2,
    random_state=42,
    n_jobs=1,
    name="tutorial",
)
result = suite.run()
```

Constructor arguments:

| Argument | Meaning |
|---|---|
| `estimators` | `{name: unfitted estimator}`. Anything with `fit`/`predict` works, including pipelines. If the estimator has a `random_state` parameter it is set from the master seed for every fold. |
| `datasets` | `{name: (X, y)}` or `{name: callable}` returning `(X, y)`. Callables are loaded lazily and cached. |
| `scoring` | A scorer name, a list of names, or `{name: scorer}` with callables. Defaults to the estimator's own `score` (accuracy or R²). |
| `n_splits`, `n_repeats` | Outer protocol: repeated stratified *k*-fold for classifiers, repeated *k*-fold for regressors. Each repeat draws a fresh shuffle seed from the master seed. |
| `cv` | An explicit scikit-learn cross-validator that overrides `n_splits`/`n_repeats`. |
| `param_grids`, `inner_cv`, `tuning_scoring` | Nested tuning, see below. |
| `random_state` | Master seed. Splits and estimator seeds derive from it, so two runs with the same seed produce identical results. |
| `n_jobs` | Parallelism over `(dataset, repeat, fold, estimator)` tasks (joblib). |
| `name` | Recorded in the manifest. |

## `BenchmarkResult`

`run()` returns a [`BenchmarkResult`][sklearn_mastery.research.benchmark.BenchmarkResult]
with three fields: `results` (DataFrame), `manifest` (`RunManifest`) and
`best_params` (only populated by nested tuning).

```python
result.results.columns.tolist()
# ['dataset', 'estimator', 'repeat', 'fold', 'metric', 'value', 'fit_time', 'score_time']
```

Views that derive from the long-form frame:

```python
result.summary("accuracy")            # datasets x estimators, mean over folds
result.summary()                      # (dataset, estimator, metric) -> mean, std, count
result.score_matrix("accuracy")       # datasets x estimators; the input to friedman_test
result.score_matrix("accuracy", aggregate="median")
result.rank_table("accuracy")         # per-dataset ranks (1 = best) + an "average_rank" row
result.paired_scores("accuracy", "wine")   # (repeat, fold) x estimators, aligned across estimators
```

`score_matrix` feeds the multi-dataset tests (Friedman, Nemenyi, Wilcoxon-Holm);
`paired_scores` feeds the single-dataset tests (corrected resampled *t*-test,
Bayesian correlated *t*-test).

## Nested tuning

Reporting the cross-validated score of a *tuned* estimator is optimistically
biased unless tuning happens inside every outer fold (Cawley & Talbot, 2010).
`BenchmarkSuite` does this when you pass `param_grids`: for each outer fold a
`GridSearchCV` with `inner_cv` folds selects the parameters on the training
part only, refits, and the refitted estimator is scored on the outer test
fold.

```python
from sklearn.svm import SVC

suite = BenchmarkSuite(
    estimators={
        "svm": make_pipeline(StandardScaler(), SVC()),
        "rf": RandomForestClassifier(random_state=0),
    },
    datasets={"iris": load_iris(return_X_y=True), "wine": load_wine(return_X_y=True)},
    scoring=["accuracy"],
    param_grids={
        "svm": {"svc__C": [0.1, 1, 10], "svc__gamma": ["scale", 0.01]},
        "rf": {"n_estimators": [50, 100]},
    },
    inner_cv=3,                 # folds of the inner search
    tuning_scoring="accuracy",  # defaults to the first scorer
    n_splits=3, n_repeats=1, random_state=0,
)
tuned = suite.run()
tuned.best_params["wine"]["svm"]   # list with the best parameters of every outer fold
```

Estimators without a grid are simply fitted. Grids use scikit-learn's
`step__param` convention for pipelines. Passing a grid for an unknown
estimator raises `KeyError` at construction time.

!!! tip "How many folds and repeats?"
    Five outer folds with two or three repeats is a reasonable default for
    datasets of a few hundred to a few thousand rows. More repeats reduce the
    variance of the *estimate* but do not make folds independent, which is why
    the single-dataset tests apply a variance correction.

## Saving and loading

```python
path = result.save("runs/tutorial")
# runs/tutorial/results.csv, manifest.json, best_params.json

from sklearn_mastery.research import BenchmarkResult
loaded = BenchmarkResult.load("runs/tutorial")
assert loaded.manifest.hash == result.manifest.hash
```

The manifest records the estimator classes and constructor parameters, the
dataset names, the protocol, the seed and the environment (Python, platform,
versions of the key packages and the git revision). `manifest.hash` is a
SHA-256 prefix of the configuration and seed (the environment is excluded), so
two runs with identical settings share a hash and can be de-duplicated.

```python
print(result.manifest.name, result.manifest.seed)
print(result.manifest.environment["packages"]["scikit-learn"])
```

## Reporting tables

`sklearn_mastery.research.reporting` renders the `mean ± std` tables used in
most empirical papers, with the best estimator per dataset emphasised and an
average-rank row appended.

```python
from sklearn_mastery.research import format_mean_std_table, results_to_latex, results_to_markdown

print(results_to_markdown(result.results, "accuracy", caption="Accuracy, mean ± std over folds"))
print(results_to_latex(result.results, "f1_macro", caption="Macro F1", label="tab:f1"))

table = format_mean_std_table(result.results, "accuracy", precision=2, include_rank=False)
table.to_csv("accuracy_table.csv")
```

`format_mean_std_table` returns a DataFrame of strings; `results_to_markdown`
produces a GitHub-flavoured table and `results_to_latex` a `table` environment
that requires the `booktabs` package. Pass `higher_is_better=False` for error
metrics so the *lowest* mean is emphasised and ranked first.

Example output:

```text
| dataset | knn | logreg | rf |
|---|---|---|---|
| cancer | 0.963 ± 0.014 | **0.976 ± 0.011** | 0.958 ± 0.019 |
| iris | 0.957 ± 0.041 | **0.963 ± 0.040** | 0.950 ± 0.041 |
| wine | 0.958 ± 0.030 | **0.986 ± 0.022** | 0.972 ± 0.028 |
| avg. rank | 2.67 | 1.00 | 2.33 |
```

## Reproducibility helpers

The `reproducibility` module is used by the suite but is also useful on its own:

```python
from sklearn_mastery.research import RunManifest, capture_environment, config_hash, set_global_seed

rng = set_global_seed(0)          # seeds random, numpy (and torch if installed); returns a Generator
env = capture_environment()       # dict: timestamp, python, platform, packages, git_revision
h = config_hash({"model": "svm", "C": 1.0})          # stable 12-character SHA-256 prefix

manifest = RunManifest(name="ablation-1", config={"model": "svm", "C": 1.0}, seed=0)
manifest.save("runs/ablation-1.json")
same = RunManifest.load("runs/ablation-1.json")
assert same.hash == manifest.hash
```

`to_jsonable` converts numpy scalars and arrays, paths, sets and estimators
(as `{"__estimator__": class name, "params": ...}`) into JSON-safe values and
is what makes manifests serialisable.

## Regression benchmarks

Nothing is classification-specific. With regressors the suite uses repeated
(unstratified) *k*-fold and the default scorer is R²:

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

reg = BenchmarkSuite(
    estimators={"ridge": make_pipeline(StandardScaler(), Ridge()),
                "gbr": GradientBoostingRegressor(random_state=0)},
    datasets={"diabetes": load_diabetes(return_X_y=True)},
    scoring=["r2", "neg_mean_absolute_error"],
    n_splits=5, random_state=0,
).run()
print(reg.summary("r2"))
```

## References

- Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection
  and subsequent selection bias in performance evaluation. *JMLR*, 11, 2079-2107.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data
  sets. *JMLR*, 7, 1-30.
