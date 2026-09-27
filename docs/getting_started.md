# Getting started

## Requirements

- Python 3.9 or newer (CI covers 3.10 to 3.12 on Linux and 3.12 on macOS).
- scikit-learn 1.3+, NumPy, pandas, SciPy, joblib, matplotlib, seaborn,
  pydantic 2, click and tqdm. All of these are installed automatically.

## Installation

The package is not yet published on PyPI; install it from the repository:

```bash
git clone https://github.com/SatvikPraveen/Sklearn-Mastery.git
cd Sklearn-Mastery
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Optional extras

Optional back-ends are imported lazily. Every module imports and works without
them; the corresponding wrappers raise an informative `ImportError` only when
you actually ask for a missing back-end. Each subpackage exposes `HAS_*` flags
(for example `sklearn_mastery.models.ensemble.HAS_XGBOOST`) that you can check
at runtime.

| Extra | Installs | Enables |
|---|---|---|
| `boosting` | `xgboost`, `lightgbm` | `XGBoostClassifierModel`, `LightGBMRegressorModel`, `BoostingEnsemble(algorithm="xgboost")`, ... |
| `tuning` | `optuna` | TPE back-end of `BayesianOptimizer` (a Gaussian-process back-end is always available) |
| `imbalanced` | `imbalanced-learn` | SMOTE and friends in `ImbalancedDataHandler` and imbalanced pipelines |
| `interpret` | `shap`, `lime` | interpretability helpers in the examples |
| `tracking` | `mlflow` | experiment tracking in the examples |
| `viz` | `plotly` | interactive plots in `ModelVisualizationSuite` |
| `notebooks` | `jupyter`, `ipywidgets`, `umap-learn` | the tutorial notebooks and the UMAP wrapper |
| `dev` | `pytest`, `ruff`, `mypy`, `pre-commit`, ... | running the test suite and linters |
| `docs` | `mkdocs-material`, `mkdocstrings`, ... | building this site |
| `all` | everything above except `dev` and `docs` | |

```bash
pip install -e ".[boosting,tuning]"   # pick what you need
pip install -e ".[all]"               # or everything
```

## Verify the installation

```bash
sklearn-mastery --version
sklearn-mastery info          # prints Python, platform, package versions and git revision
```

`sklearn-mastery info` uses
[`capture_environment`][sklearn_mastery.research.reproducibility.capture_environment],
the same function that stamps every benchmark manifest.

## Your first experiment

The smallest meaningful experiment compares two or more estimators on one or
more datasets under a shared resampling protocol.

```python
from sklearn.datasets import load_iris, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from sklearn_mastery.research import BenchmarkSuite

suite = BenchmarkSuite(
    estimators={
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "tree": DecisionTreeClassifier(random_state=0),
    },
    datasets={"iris": load_iris(return_X_y=True), "wine": load_wine(return_X_y=True)},
    scoring=["accuracy", "f1_macro"],
    n_splits=5,
    n_repeats=2,
    random_state=0,
)
result = suite.run()

print(result.results.head())              # one row per (dataset, estimator, repeat, fold, metric)
print(result.summary("accuracy"))          # mean accuracy, datasets x estimators
print(result.rank_table("accuracy"))       # per-dataset ranks + average rank
print(result.manifest.hash)                # content hash of the configuration + seed
```

Everything downstream (statistical tests, tables, CD diagrams) consumes the
`result` object. Continue with the [Benchmarking guide](research/benchmarking.md).

## Working with the library layers directly

The research layer works with *any* scikit-learn estimator. The remaining
subpackages are independent building blocks that you can use on their own:

```python
from sklearn_mastery.data import SyntheticDataGenerator, DataValidator
from sklearn_mastery.models.supervised import ClassificationModels
from sklearn_mastery.evaluation import MetricsCalculator

gen = SyntheticDataGenerator(random_state=0)
X, y = gen.classification_dataset(n_samples=300, n_features=10, n_classes=3)

report = DataValidator().validate_dataset(X, y, task_type="classification")
print(report["validation_status"], report["total_issues"])

model = ClassificationModels().get_model("random_forest", n_estimators=50, random_state=0)
model.fit(X[:200], y[:200])
print(MetricsCalculator().calculate_all_metrics(y[200:], model.predict(X[200:])))
```

## Logging and configuration

Importing `sklearn_mastery` never configures logging. The package logger
carries a `NullHandler`; call `setup_logging` in applications and notebooks:

```python
from sklearn_mastery import setup_logging, settings

setup_logging(log_level="INFO")          # console handler (uses rich when installed)
print(settings.RANDOM_SEED, settings.RESULTS_DIR)
```

`settings` is a pydantic-settings object. Every field can be overridden with
an environment variable prefixed `SKLEARN_MASTERY_` (for example
`SKLEARN_MASTERY_RANDOM_SEED=7`) or a `.env` file in the working directory.
`settings.PROJECT_ROOT` defaults to the current working directory; result
directories are only created when you call `settings.ensure_directories()`.

## Running the tests

```bash
pip install -e ".[dev]"
pytest -n auto -q            # full suite in parallel
ruff check sklearn_mastery tests && ruff format --check sklearn_mastery tests
mypy sklearn_mastery/research sklearn_mastery/config
```

See [Contributing](development/contributing.md) for the full developer workflow.
