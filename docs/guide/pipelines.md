# Pipelines

`sklearn_mastery.pipelines` has four modules:

| Module | What it provides |
|---|---|
| `custom_transformers` | DataFrame-aware transformers (scaling, imputation, encoding, outliers, dates, text, interactions, binning, feature selection, validation, debugging) |
| `feature_union` | Parallel feature branches (`AdvancedFeatureUnion`, weighted, conditional and dynamic unions), a column transformer, a declarative `FeaturePipelineBuilder`, `PipelineComposer`, `FeatureStacker` and a registry of transformer placeholders |
| `pipeline_factory` | `PipelineFactory`, `PipelineConfig`, `CustomPipelineBuilder`, task-specific factories, `AutoMLPipelineBuilder`, `PipelineOptimizer` and the registries of algorithm and preprocessing names |
| `model_selection` | Cross-validated model selection, grid / random / Bayesian / multi-objective search, nested CV, learning and validation curves, model comparison, weighted and dynamic ensembles, a model registry and an experiment tracker |

Everything is importable from `sklearn_mastery.pipelines` directly. All
transformers work on NumPy arrays and pandas DataFrames, implement
`get_feature_names_out`, and can be cloned.

## Custom transformers

```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn_mastery.pipelines import (
    CustomScaler, FeatureSelector, MissingValueHandler, OutlierRemover, PipelineDebugger,
)

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 6)), columns=[f"f{i}" for i in range(6)])
X.iloc[::15, 2] = np.nan
y = (X["f0"] + 0.5 * X["f1"] + rng.normal(scale=0.3, size=200) > 0).astype(int).to_numpy()

pipe = Pipeline([
    ("impute", MissingValueHandler(strategy="median", add_indicator=True)),
    ("scale", CustomScaler(method="robust")),
    ("select", FeatureSelector(method="univariate", k=4)),
    ("debug", PipelineDebugger(step_name="after_selection")),
    ("model", LogisticRegression()),
]).fit(X, y)
print(pipe.score(X, y))
print(pipe[:-1].get_feature_names_out())
print(pipe.named_steps["select"].get_selected_features())

remover = OutlierRemover(method="iqr", threshold=1.5).fit(X)
print(remover.get_outlier_mask(X).sum(), remover.transform(X).shape)
```

!!! warning "`OutlierRemover` changes the number of rows"
    `OutlierRemover.transform` drops rows, so it cannot be the middle of a
    supervised `Pipeline` (the target would no longer align). Use it before
    splitting, or use `get_outlier_mask` to filter `X` and `y` together. For
    in-pipeline handling prefer `NumericTransformer(handle_outliers=True)`
    which clips instead of dropping.

Other transformers in the module:

| Transformer | Purpose |
|---|---|
| `NumericTransformer` | impute, clip outliers, log/sqrt/Box-Cox transforms and scale numeric columns in place |
| `CategoricalEncoder`, `TargetEncoder`, `DomainSpecificEncoder` | one-hot / ordinal / target / hashing encodings with unknown-category handling; `TargetEncoder(cv_folds=5)` uses out-of-fold encodings in `fit_transform` |
| `AdvancedImputer` | KNN imputation for numerics and mode imputation for categoricals |
| `DateTimeTransformer`, `TimeSeriesFeatureCreator` | calendar features (optionally cyclical), lags and rolling statistics |
| `TextTransformer`, `TextFeatureExtractor` | TF-IDF / count vectorisation of text columns, optional length features |
| `PolynomialFeatureCreator`, `FeatureInteractionCreator` | polynomial and pairwise interaction terms with readable names |
| `BinningTransformer` | uniform / quantile / k-means binning with optional one-hot output |
| `FeatureScaler` | per-feature scaler chosen from the feature's distribution |
| `DataValidator` | schema check that passes data through unchanged |

`sklearn_mastery.preprocessing` re-exports these transformers under a shorter
import path.

## Feature unions

[`AdvancedFeatureUnion`][sklearn_mastery.pipelines.feature_union.AdvancedFeatureUnion]
runs branches in parallel on the same input and stacks their outputs; each
branch is an estimator, a `Pipeline`, or a **placeholder string** resolved
through the transformer registry (`registered_transformers()` lists them, e.g.
`"standard_scaler"`, `"onehot_encoder"`, `"pca"`, `"univariate"`).

```python
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

from sklearn_mastery.pipelines import (
    AdvancedFeatureUnion, ConditionalFeatureUnion, FeaturePipelineBuilder,
    PipelineComposer, WeightedFeatureUnion, registered_transformers, resolve_transformer,
)

union = AdvancedFeatureUnion([
    ("pca", PCA(n_components=2)),
    ("best", SelectKBest(k=3)),
    ("scaled", "standard_scaler"),          # placeholder resolved from the registry
]).fit(X.fillna(0), y)
print(union.transform(X.fillna(0)).shape)   # 2 + 3 + 6 columns
print(union.get_feature_names_out()[:4])

weighted = WeightedFeatureUnion([("pca", PCA(2)), ("raw", "standard_scaler")], weights={"pca": 2.0})
print(weighted.fit_transform(X.fillna(0), y).shape)

conditional = ConditionalFeatureUnion(
    [("poly", "polynomial_features"), ("raw", "standard_scaler")],
    conditions={"poly": lambda X: X.shape[1] <= 3},      # condition(X) -> bool; branch fitted only when True
)
print(conditional.fit_transform(X.fillna(0), y).shape)

print(resolve_transformer("robust_scaler"))
print(len(registered_transformers()))
```

`DynamicFeatureUnion` chooses its branches with a strategy callable at fit
time, `ColumnTransformer` (the package's own, DataFrame-friendly variant)
applies different transformers to column groups, and `ParallelFeatureProcessor`
runs column-group processors on a thread pool.

### Declarative builders

```python
builder = FeaturePipelineBuilder(random_state=0)
auto = builder.build_auto_pipeline(X.fillna(0), y)       # type-aware preprocessing from the dtypes
print([name for name, _ in auto.steps])

composed = PipelineComposer().compose([
    ("impute", MissingValueHandler()),
    ("union", AdvancedFeatureUnion([("pca", PCA(2)), ("raw", "standard_scaler")])),
    ("model", LogisticRegression()),
])
print(composed.fit(X, y).score(X, y))
```

`FeaturePipelineBuilder` also offers `add_preprocessing_module`,
`add_feature_engineering_module`, `add_selection_module`,
`build_custom_pipeline(spec)` from a dictionary, and `optimize_pipeline` for a
small random or grid search over the pipeline's parameters.

## Pipeline factory

[`PipelineFactory`][sklearn_mastery.pipelines.pipeline_factory.PipelineFactory]
builds complete `Pipeline` objects from an algorithm name and a preprocessing
level, from a configuration, or from a data profile.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score

from sklearn_mastery.pipelines import (
    PipelineConfig, PipelineFactory, available_classifiers, available_preprocessing_steps, infer_task_type,
)

Xc, yc = load_breast_cancer(return_X_y=True)
print(available_classifiers())
print(available_preprocessing_steps()[:6])
print(infer_task_type(yc))

factory = PipelineFactory(random_state=0)
pipe = factory.create_classification_pipeline(algorithm="random_forest", preprocessing_level="standard",
                                              model_params={"n_estimators": 50})
print([name for name, _ in pipe.steps])
print(cross_val_score(pipe, Xc, yc, cv=3).mean())

adaptive = factory.auto_create_pipeline(Xc, yc, task_type="auto", complexity_level="medium")
print([name for name, _ in adaptive.steps])

config = PipelineConfig.quick_start_classification()          # imputer + scaler + random forest template
config = config.set_option("model.n_estimators", 50)
print(config.task_type, config.get_model_config()["type"])
from_config = factory.create_pipeline(config)
print(from_config.steps[-1][0], type(from_config.steps[-1][1]).__name__)

tuned = factory.create_pipeline_with_auto_tuning("logistic_regression", "classification",
                                                  param_grid={"model__C": [0.1, 1.0]}, cv=3)
print(type(tuned).__name__)                                   # GridSearchCV around the pipeline
```

`preprocessing_level` is `"minimal"`, `"standard"` or `"advanced"`, and
`preprocessing_steps` accepts explicit registry names. Other entry points:
`create_regression_pipeline`, `create_production_pipeline`,
`create_custom_pipeline`, `create_preprocessing_pipeline`. The task-specific
[`ClassificationPipelineFactory`][sklearn_mastery.pipelines.pipeline_factory.ClassificationPipelineFactory]
and [`RegressionPipelineFactory`][sklearn_mastery.pipelines.pipeline_factory.RegressionPipelineFactory]
add scenario helpers such as `create_imbalanced_pipeline`,
`create_text_classification_pipeline`, `create_polynomial_regression_pipeline`
and `create_ensemble_pipeline`.

### Fluent builder

```python
from sklearn_mastery.pipelines import CustomPipelineBuilder

custom = (
    CustomPipelineBuilder(task_type="classification", random_state=0)
    .add_preprocessing_step("standard_scaler")
    .add_feature_selection_step("univariate", k=10)
    .add_feature_engineering_step("pca", n_components=5)
    .add_model_step("gradient_boosting", n_estimators=50)
    .build()
)
print([name for name, _ in custom.steps])
print(cross_val_score(custom, Xc, yc, cv=3).mean())
```

[`AutoMLPipelineBuilder`][sklearn_mastery.pipelines.pipeline_factory.AutoMLPipelineBuilder]
profiles the data, recommends preprocessing, feature engineering and models,
and builds the recommended pipeline; `PipelineOptimizer` searches over
scalers, feature-selection sizes, algorithms and hyperparameters.

## Model selection

### Cross-validated selection

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn_mastery.pipelines import ModelSelectionPipeline, NestedCrossValidation

models = {
    "logreg": LogisticRegression(max_iter=2000),
    "rf": RandomForestClassifier(n_estimators=50, random_state=0),
    "svm": SVC(),
}
selector = ModelSelectionPipeline(cv=3, scoring=["accuracy", "f1"], random_state=0)
best_name, info = selector.select_best_model(models, Xc, yc, preprocessing=["standard_scaler"])
print(best_name, info["mean_score"][best_name], info["std_score"][best_name])
print(selector.summary())

nested = NestedCrossValidation(outer_cv=3, inner_cv=2, random_state=0).evaluate_models(
    {"rf": RandomForestClassifier(random_state=0), "svm": SVC()}, Xc, yc,
    param_grids={"rf": {"n_estimators": [20, 50]}, "svm": {"C": [0.5, 1.0]}},
)
print(nested["best_model"], sorted(nested["results"]))
```

`CrossValidationPipeline` collects stratified, time-series, group and nested
variants that return rich dictionaries; `GridSearchPipeline` adds adaptive
(zoom-in), constrained and explicitly parallel grid searches;
`AdvancedModelSelector`, `AutoModelSelector` and `AutoMLSelector` recommend
and select models under a time budget; `MultiObjectiveSelector` returns the
Pareto-optimal set over several metrics.

!!! tip "Which selection tool for a paper?"
    For a *reported* comparison use
    [`BenchmarkSuite`][sklearn_mastery.research.benchmark.BenchmarkSuite] with
    `param_grids` and the tests in [Statistical comparison](../research/statistical_comparison.md).
    The classes here are convenient for interactive exploration and for the
    example scripts.

### Hyperparameter optimisation

[`HyperparameterOptimizer`][sklearn_mastery.pipelines.model_selection.HyperparameterOptimizer]
wraps grid, random, Bayesian and multi-objective search behind one object;
[`BayesianOptimizer`][sklearn_mastery.pipelines.model_selection.BayesianOptimizer]
is the sequential model-based engine behind it.

```python
from sklearn_mastery.pipelines import BayesianOptimizer, HyperparameterOptimizer

search_space = {
    "n_estimators": (20, 100),            # int range
    "max_depth": (2, 8),                  # int range
    "max_features": ["sqrt", "log2"],     # categorical
    "min_samples_leaf": {"type": "int", "low": 1, "high": 5},
}
bo = BayesianOptimizer(scoring="accuracy", random_state=0, backend="gp", n_initial_points=3)
best_params, info = bo.optimize(RandomForestClassifier(random_state=0), search_space, Xc, yc,
                                n_iterations=6, cv=3, acquisition_function="expected_improvement")
print(best_params, round(info["best_score"], 3), info["backend"], len(info["optimization_history"]))

hpo = HyperparameterOptimizer(scoring="accuracy", random_state=0)
grid_best, grid_info = hpo.grid_search(SVC(), {"C": [0.1, 1.0, 10.0]}, Xc, yc, cv=3)
print(grid_best, round(grid_info["best_score"], 3))
```

Search-space values are `(low, high)` tuples (int or float ranges), lists of
choices, or explicit `{"type": "int" | "float" | "categorical", ...}`
dictionaries (`"log": True` samples a float range log-uniformly). The
`"gp"` back-end (Gaussian-process surrogate with expected improvement,
probability of improvement or UCB acquisition) needs no optional dependency;
`backend="auto"` prefers Optuna's TPE when `optuna` is installed. `optimize`
supports early stopping (`early_stopping_rounds`, `early_stopping_threshold`)
and warm starts from a previous `optimization_history`.
`multi_objective_optimize` returns a Pareto front over several objectives.

### Diagnostics, comparison and tracking

- `LearningCurveAnalyzer.analyze_learning_curve(model, X, y)` returns the
  curve plus bias/variance labels; `ValidationCurveAnalyzer` does the same for
  one hyperparameter.
- `ModelComparator` combines statistical comparison, runtime profiling,
  robustness to noise / outliers and calibration in one report.
- `WeightedEnsembleClassifier` / `WeightedEnsembleRegressor` and
  `DynamicEnsembleSelector` (best-local-accuracy selection over pre-fitted
  classifiers) are built by `ModelEnsemblePipeline`, whose `optimize_ensemble`
  learns member weights on out-of-fold predictions.
- `ModelRegistry` versions estimators on disk with deployment records and a
  performance history; `PerformanceTracker` is an in-memory experiment and
  drift tracker.

```python
from sklearn_mastery.pipelines import ModelRegistry, PerformanceTracker

registry = ModelRegistry(storage_dir="runs/registry")
model_id = registry.register_model(pipe.fit(Xc, yc), name="rf-cancer", version="1.0.0", tags=["demo"])
registry.log_model_performance(model_id, {"accuracy": 0.96}, dataset="cancer")
print(registry.get_model_info(model_id)["name"], registry.get_performance_history(model_id)[0]["metrics"])

tracker = PerformanceTracker()
exp = tracker.start_experiment("baseline", parameters={"model": "rf"})
tracker.log_metrics(exp, {"accuracy": 0.96, "f1": 0.95})
tracker.end_experiment(exp)
print(tracker.to_dataframe()[["name", "status"]])
```
