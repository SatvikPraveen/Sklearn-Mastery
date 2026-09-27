# Data

`sklearn_mastery.data` contains three independent tools:

- **Generators** (`sklearn_mastery.data.generators`): deterministic synthetic
  datasets for every modelling challenge covered by the toolkit.
- **Preprocessors** (`sklearn_mastery.data.preprocessors`): an end-to-end
  `DataPreprocessor`, a cardinality-aware `CategoricalEncoder`, a
  `NumericalTransformer` and an `ImbalancedDataHandler`.
- **Validators** (`sklearn_mastery.data.validators`): data-quality reports,
  schema validation and drift detection.

## Synthetic data generators

[`SyntheticDataGenerator`][sklearn_mastery.data.generators.SyntheticDataGenerator]
is a seeded factory. Every method accepts an optional `random_state` that
overrides the instance seed, and calling the same method twice with the same
seed returns identical arrays. `DataGenerator` is an alias;
`ClassificationDataGenerator`, `RegressionDataGenerator` and
`ClusteringDataGenerator` are named views of the same class.

```python
from sklearn_mastery.data import SyntheticDataGenerator

gen = SyntheticDataGenerator(random_state=42)

# Classification
X, y = gen.classification_dataset(n_samples=500, n_features=20, n_informative=8, n_classes=3)
X, y = gen.imbalanced_classification_data(n_samples=500, imbalance_ratio=0.1)
X, y = gen.classification_with_noise(n_samples=500, n_features=20, noise_features=5, flip_y=0.05)
X, y = gen.classification_complexity_spectrum("high", n_samples=500)     # 'linear' | 'medium' | 'high'
X, y = gen.multilabel_classification(n_samples=200, n_classes=5)
X, y, true_importance = gen.feature_selection_showcase_data(n_samples=500, n_features=50)

# Regression
X, y = gen.linear_regression_data(n_samples=500, n_features=10, noise_level=0.1)
X, y, groups = gen.regression_with_collinearity(n_samples=500, n_features=10)
X, y, coef = gen.high_dimensional_regression(n_samples=100, n_features=300, n_informative=10)
X, y = gen.nonlinear_regression(n_samples=500, nonlinearity_type="polynomial")
X, y, outlier_mask = gen.regression_with_outliers(n_samples=500, outlier_fraction=0.1)

# Clustering and shapes
X, labels = gen.clustering_dataset(n_samples=500, n_clusters=4)
X = gen.clustering_blobs_with_noise(n_samples=500, n_clusters=4, outlier_fraction=0.1)
X, labels = gen.make_spirals(n_samples=400, n_spirals=2)
X, labels = gen.complex_shapes(n_samples=400, shape_type="moons")

# Streams, sequences and mixed tables
X, y, drift_points = gen.concept_drift_classification(n_samples=1000, n_drift_points=2)
frame, y = gen.mixed_data_types(n_samples=300, n_numerical=5, n_categorical=3, n_ordinal=2)   # DataFrame, target
X, y = gen.anomaly_detection_data(n_samples=500, contamination=0.05)
```

`COMPLEXITY_LEVELS` is the tuple `("linear", "medium", "high")` accepted by
`classification_complexity_spectrum` and by the CLI. `generate_dataset_suite`
returns one representative dataset per challenge as a dictionary, which is a
convenient way to smoke-test a pipeline across problem types.

!!! note "Return shapes differ by generator"
    Most methods return `(X, y)`, but the collinearity, high-dimensional,
    outlier, feature-selection and concept-drift generators return a third
    element describing the ground truth (groups, coefficients, masks, drift
    points). `mixed_data_types` returns a pandas DataFrame with categorical
    dtypes plus the target. Check the
    [API reference](../api/data.md) for each signature.

## Preprocessing

### `DataPreprocessor`

[`DataPreprocessor`][sklearn_mastery.data.preprocessors.DataPreprocessor] is a
scikit-learn transformer that runs a fixed, individually switchable sequence
of steps and returns a dense array: imputation, categorical encoding,
interaction features, outlier detection, scaling, feature selection and PCA.

```python
import numpy as np
from sklearn_mastery.data import DataPreprocessor, SyntheticDataGenerator

frame, y = SyntheticDataGenerator(random_state=0).mixed_data_types(n_samples=300, n_categorical=2)
frame.iloc[::20, 0] = np.nan                     # inject missing values

prep = DataPreprocessor(
    handle_missing=True, encode_categoricals=True, scale_features=True,
    remove_outliers=True, feature_selection=True, feature_selection_k=8,
    random_state=0,
)
X_t = prep.fit_transform(frame, y)
print(X_t.shape)
print(prep.get_applied_steps())                  # e.g. ['imputer', 'encoder', 'outlier_detector', 'scaler', 'feature_selector']
print(prep.get_feature_names_out()[:5])
print(prep.get_preprocessing_summary()["n_features_out"])
```

Outlier handling never drops rows: rows flagged by `IsolationForest` at fit
time are excluded from *fitting* the downstream steps (see `outlier_mask_`),
but `transform` always preserves the number of rows, which keeps the
transformer usable inside a `Pipeline`. `DataPreprocessor.analyze_data(X)`
returns skewness, missing-value and dtype statistics without fitting anything.

### `CategoricalEncoder` and `NumericalTransformer`

```python
from sklearn_mastery.data import CategoricalEncoder, NumericalTransformer

enc = CategoricalEncoder(strategy="auto", max_cardinality=50, onehot_max_categories=10)
encoded = enc.fit_transform(frame, y)            # DataFrame; numeric columns pass through
print(enc.get_feature_names_out()[:5])

num = NumericalTransformer(apply_log=True, create_polynomials=True, polynomial_degree=2)
numeric_only = frame.select_dtypes("number")
expanded = num.fit_transform(numeric_only)
print(expanded.shape)
```

`strategy="auto"` one-hot encodes low-cardinality columns, ordinal-encodes
medium-cardinality ones and target-encodes (with smoothing) high-cardinality
ones when `y` is available; the other strategies (`"onehot"`, `"ordinal"`,
`"target"`, `"binary"`, `"frequency"`) apply one method everywhere. Unknown
categories at transform time are handled according to `handle_unknown`.

### `ImbalancedDataHandler`

```python
from sklearn_mastery.data import ImbalancedDataHandler

X, y = SyntheticDataGenerator(random_state=0).imbalanced_classification_data(n_samples=400, imbalance_ratio=0.1)
print(ImbalancedDataHandler.class_distribution(y))

handler = ImbalancedDataHandler(method="random_over", random_state=0)   # or "random_under"
X_res, y_res = handler.handle_imbalance(X, y)
print(ImbalancedDataHandler.class_distribution(y_res))
```

`"random_over"` (aliases `"oversample"`, `"oversampling"`) and `"random_under"`
(`"undersample"`, `"undersampling"`) work without extra dependencies;
`"smote"` (the default) requires `imbalanced-learn`
(`pip install -e ".[imbalanced]"`). `sampling_strategy` accepts the usual
`"auto"`, a float ratio or a `{class: count}` mapping.

## Validation

### Data-quality reports

[`DataValidator`][sklearn_mastery.data.validators.DataValidator] runs a battery
of checks (missing values, duplicates, constant and low-variance features,
outliers, high-cardinality categoricals, multicollinearity, feature scale,
class balance, target distribution, small datasets) and collects
[`ValidationIssue`][sklearn_mastery.data.validators.ValidationIssue] records
with a severity (`info`, `warning`, `error`, `critical`).

```python
import pandas as pd
from sklearn_mastery.data import DataValidator

X = pd.DataFrame({
    "age": [25, 32, 47, 51, 62, 23, 44, 38, 29, 55] * 5,
    "income": [40, 52, 88, 91, 120, 30, 75, 66, 45, 99] * 5,
    "constant": [1] * 50,
})
y = pd.Series([0, 0, 1, 1, 1, 0, 1, 1, 0, 1] * 5)

validator = DataValidator(strict_mode=False, correlation_threshold=0.9)
report = validator.validate_dataset(X, y, task_type="classification")
print(report["validation_status"], report["is_valid"])   # 'PASSED', 'PASSED_WITH_WARNINGS' or 'FAILED'
print(report["severity_counts"])
for issue in report["detailed_issues"]:
    print(issue["severity"], issue["category"], issue["message"])
print(report["recommendations"])
print(validator.format_summary())
```

`validate(X, y)` returns just the boolean; `validate_classification_data` and
`validate_regression_data` are task-specific entry points with switches per
check; `validate_pipeline_input` checks a feature matrix against the schema
seen at training time. `add_custom_rule(name, rule)` registers your own
callable `(X, y) -> list of issues`. `DataValidator.infer_task_type(y)` is the
heuristic the CLI and the pipeline factory rely on.

### Schema validation

[`SchemaValidator`][sklearn_mastery.data.validators.SchemaValidator] checks a
DataFrame against a declarative schema with `required_columns`,
`column_types`, `value_ranges`, `allowed_values` and `non_nullable` keys.

```python
from sklearn_mastery.data import SchemaValidator

schema = {
    "required_columns": ["age", "income"],
    "column_types": {"age": "numeric", "income": "numeric"},
    "value_ranges": {"age": (0, 120), "income": (0, None)},
    "non_nullable": ["age"],
}
sv = SchemaValidator(schema)
print(sv.validate(X))                          # True
print(sv.validate(X.assign(age=-1)))           # False
print(sv.get_validation_errors())

inferred = SchemaValidator.infer_schema(X)     # derive a schema from training data
print(sorted(inferred))
```

`validate_or_raise` raises `ValueError` with all messages joined, which is the
right behaviour at the entrance of a production pipeline.

### Drift detection

`DataValidator.detect_data_drift` compares a reference frame (training data)
with a current frame (production data) column by column. Numeric columns use
a two-sample Kolmogorov-Smirnov test (`method="ks_test"`, drift when the
p-value is below `threshold`) or the Population Stability Index
(`method="psi"`, drift when PSI exceeds `threshold`; 0.1 is moderate, 0.25
severe). Categorical columns use a chi-square test or a frequency PSI.

$$
\mathrm{PSI} = \sum_{b=1}^{B} (q_b - p_b)\,\ln\frac{q_b}{p_b},
$$

where $p_b$ and $q_b$ are the reference and current proportions in bin $b$.

```python
rng = np.random.default_rng(0)
reference = pd.DataFrame({"x1": rng.normal(0, 1, 500), "x2": rng.normal(5, 2, 500)})
current = pd.DataFrame({"x1": rng.normal(0.8, 1, 500), "x2": rng.normal(5, 2, 500)})

drift = DataValidator().detect_data_drift(reference, current, method="ks_test", threshold=0.05)
print(drift["drift_detected"], drift["drifted_features"])
print(drift["feature_results"]["x1"])

psi = DataValidator().detect_data_drift(reference, current, method="psi", threshold=0.1)
print(psi["n_drifted_features"])
```

The `concept_drift_classification` generator produces a stream with known
drift points, which is useful for testing monitoring code end to end.
