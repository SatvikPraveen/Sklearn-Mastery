# Command-line interface

Installing the package registers the `sklearn-mastery` console script
(entry point `sklearn_mastery.cli:main`, built with click). It exposes the
research workflow without writing Python. `python -m sklearn_mastery.cli` is
equivalent.

```text
Usage: sklearn-mastery [OPTIONS] COMMAND [ARGS]...

  sklearn-mastery: reproducible scikit-learn experiments.

Options:
  --version            Show the version and exit.
  -v, --verbose        Enable debug logging.
  --log-file FILE      Also log to this file.
  --help               Show this message and exit.

Commands:
  benchmark         Benchmark the default estimator zoo on built-in and/or CSV datasets.
  compare           Statistically compare estimators from a saved benchmark directory.
  generate-data     Generate a synthetic dataset and write it to CSV.
  info              Print the captured execution environment as JSON.
  launch-notebooks  Start JupyterLab in the notebooks directory.
  train             Train a named estimator on a CSV file and report hold-out and CV scores.
```

Global options come *before* the command name. `--verbose` switches the
console handler to `DEBUG`; `--log-file` adds a file handler. Logging is set
up with `setup_logging(rich_console=True)`, so output is colourised when
`rich` is installed.

## `generate-data`

Write a synthetic dataset to CSV using
[`SyntheticDataGenerator`][sklearn_mastery.data.generators.SyntheticDataGenerator].

| Option | Default | Meaning |
|---|---|---|
| `--dataset-type [classification\|regression\|clustering]` | `classification` | Kind of problem. |
| `--n-samples INTEGER` | `1000` | Number of rows. |
| `--n-features INTEGER` | `20` | Number of feature columns. |
| `--complexity [linear\|medium\|high]` | `medium` | Classification: decision-boundary complexity (`classification_complexity_spectrum`). Regression: `linear` uses `linear_regression_data`, anything else `regression_with_collinearity`. Ignored for clustering. |
| `--seed INTEGER` | `settings.RANDOM_SEED` (42) | Random seed. |
| `-o, --output FILE` | `generated_data.csv` | Output path; parent directories are created. |

Columns are named `feature_0 ... feature_{n-1}` plus `target` (absent for
clustering, which uses `clustering_blobs_with_noise`).

```bash
sklearn-mastery generate-data --dataset-type classification --complexity high \
    --n-samples 2000 --n-features 15 --seed 7 -o data/synthetic.csv
```

## `train`

Train one estimator from the model factories on a CSV file, print hold-out
and cross-validation scores as JSON, and optionally persist everything.

```text
Usage: sklearn-mastery train [OPTIONS] DATA_FILE
```

| Option | Default | Meaning |
|---|---|---|
| `DATA_FILE` | required | CSV with features and a target column. |
| `--target TEXT` | `target` | Name of the target column. |
| `--algorithm TEXT` | `random_forest` | Name understood by `ClassificationModels.get_model` or `RegressionModels.get_model` (see below). |
| `--task-type [classification\|regression]` | `classification` | Selects the factory and whether the split is stratified. |
| `--test-size FLOAT` | `0.2` | Hold-out fraction. |
| `--cv-folds INTEGER` | `5` | Folds for `cross_val_score` on the training part. |
| `-o, --output-dir DIRECTORY` | none | When given, writes `<algorithm>.joblib`, `<algorithm>_metrics.json` and `<algorithm>_manifest.json`. |

Classification algorithms: `logistic_regression`, `decision_tree`,
`random_forest`, `extra_trees`, `gradient_boosting`, `ada_boost`, `svm`,
`knn`, `naive_bayes`, `neural_network`, plus `xgboost` / `lightgbm` when
installed. Regression algorithms: `linear_regression`, `ridge`, `lasso`,
`elastic_net`, `decision_tree`, `random_forest`, `extra_trees`,
`gradient_boosting`, `adaboost`, `svr`, `neural_network`, plus `xgboost` /
`lightgbm`.

```bash
sklearn-mastery train data/synthetic.csv --algorithm gradient_boosting --cv-folds 5 -o runs/gb
```

The JSON printed contains `algorithm`, `task_type`, `holdout_score`,
`cv_mean`, `cv_std`, `cv_scores`, `n_train` and `n_test`. The split and the
model seed use `settings.RANDOM_SEED`.

## `benchmark`

Run [`BenchmarkSuite`][sklearn_mastery.research.benchmark.BenchmarkSuite]
with a default estimator zoo on built-in and/or CSV datasets, print a
Markdown `mean ± std` table, run the Friedman test when there are at least two
datasets, and optionally save the run.

| Option | Default | Meaning |
|---|---|---|
| `--dataset [breast_cancer\|diabetes\|digits\|iris\|wine]` | `iris`, `wine`, `breast_cancer` (classification) or `diabetes` (regression) | Built-in scikit-learn datasets; repeatable. |
| `--csv FILE` | none | CSV dataset(s); repeatable. The file stem is the dataset name. |
| `--target TEXT` | `target` | Target column for CSV datasets. |
| `--task-type [classification\|regression]` | `classification` | Must match the built-in datasets chosen (`diabetes` is regression). |
| `--scoring TEXT` | `accuracy`, `f1_macro` (classification) or `r2` (regression) | Scorer name(s); repeatable. The first is the primary metric. |
| `--n-splits INTEGER` | `5` | Outer folds. |
| `--n-repeats INTEGER` | `3` | Repeats of the outer CV. |
| `--n-jobs INTEGER` | `1` | Parallel jobs. |
| `--seed INTEGER` | `settings.RANDOM_SEED` | Master seed. |
| `--quick` | off | Use 3 folds and 1 repeat (smoke test). |
| `-o, --output-dir DIRECTORY` | none | Save `results.csv`, `manifest.json` and `best_params.json` via `BenchmarkResult.save`. |

The default zoo is `logreg`, `svm_rbf`, `knn` (each behind a
`StandardScaler`), `random_forest` and `gradient_boosting` for
classification, and `ridge`, `svr`, `knn`, `random_forest`,
`gradient_boosting` for regression.

```bash
sklearn-mastery benchmark --dataset iris --dataset wine --dataset breast_cancer \
    --scoring accuracy --scoring f1_macro --n-splits 5 --n-repeats 3 -o runs/demo
sklearn-mastery benchmark --task-type regression --dataset diabetes --csv data/house.csv --quick
```

## `compare`

Load a saved benchmark directory and run the statistical comparison.

```text
Usage: sklearn-mastery compare [OPTIONS] RESULTS_DIR
```

| Option | Default | Meaning |
|---|---|---|
| `RESULTS_DIR` | required | Directory written by `benchmark -o` (or `BenchmarkResult.save`). |
| `--metric TEXT` | first metric in the results | Metric to compare. |
| `--alpha FLOAT` | `0.05` | Significance level. |
| `--rope FLOAT` | `0.01` | ROPE half-width for the Bayesian test (single-dataset case). |
| `--cd-diagram FILE` | none | Save a critical-difference diagram as PNG. |

With **two or more datasets** the command prints the score matrix, the
average ranks, the Friedman chi-square and Iman-Davenport statistics, the
Nemenyi critical difference and the table of pairwise Wilcoxon tests with
Holm correction. With **one dataset** it runs the Bayesian correlated
*t*-test of every estimator against the best one and prints
$P(\text{left})$, $P(\text{rope})$, $P(\text{right})$ and the decision.

```bash
sklearn-mastery compare runs/demo --metric accuracy --alpha 0.05 --cd-diagram runs/demo/cd.png
```

## `info`

Print the output of
[`capture_environment`][sklearn_mastery.research.reproducibility.capture_environment]
as JSON: UTC timestamp, Python version and implementation, platform, CPU
count, versions of numpy / scipy / pandas / scikit-learn / joblib /
matplotlib / xgboost / lightgbm / optuna / imbalanced-learn (null when not
installed) and the current git revision. Paste it into bug reports and
experiment logs.

```bash
sklearn-mastery info
```

## `launch-notebooks`

Start JupyterLab (`python -m jupyterlab --no-browser`) in the tutorial
notebook directory. Requires `pip install jupyterlab`.

| Option | Default | Meaning |
|---|---|---|
| `--port INTEGER` | `8888` | Port to serve on. |
| `--notebook-dir TEXT` | `notebooks` | Directory to open; must exist. |

## End-to-end example

```bash
sklearn-mastery generate-data --dataset-type classification --complexity high -o data.csv
sklearn-mastery train data.csv --algorithm gradient_boosting -o runs/gb
sklearn-mastery benchmark --csv data.csv --dataset wine --quick -o runs/csv-vs-wine
sklearn-mastery compare runs/csv-vs-wine --cd-diagram cd.png
sklearn-mastery -v --log-file run.log info
```
