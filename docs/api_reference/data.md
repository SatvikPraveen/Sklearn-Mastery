# File: docs/api_reference/data.md

# Location: docs/api_reference/data.md

# Data Processing API Reference

Complete API documentation for data generation, preprocessing, and validation modules.

## Data Generators

### `src.data.generators.DataGenerator`

Core class for generating synthetic datasets for testing and development.

```python
from src.data.generators import DataGenerator

generator = DataGenerator(random_state=42)
```

#### Methods

##### `generate_classification_data()`

Generate synthetic classification datasets.

```python
X, y = generator.generate_classification_data(
    n_samples=1000,
    n_features=20,
    n_classes=3,
    n_informative=15,
    n_redundant=2,
    n_clusters_per_class=1,
    flip_y=0.01,
    class_sep=1.0,
    hypercube=True,
    shift=0.0,
    scale=1.0,
    shuffle=True,
    random_state=None
)
```

**Parameters:**

- `n_samples` (int): Number of samples to generate
- `n_features` (int): Total number of features
- `n_classes` (int): Number of classes/labels
- `n_informative` (int): Number of informative features
- `n_redundant` (int): Number of redundant features
- `n_clusters_per_class` (int): Number of clusters per class
- `flip_y` (float): Fraction of samples with randomly flipped labels
- `class_sep` (float): Factor multiplying hypercube size
- `hypercube` (bool): If True, clusters are hypercubes
- `shift` (float): Shift features by specified value
- `scale` (float): Multiply features by specified value
- `shuffle` (bool): Shuffle samples and features
- `random_state` (int): Random seed for reproducibility

**Returns:**

- `X` (ndarray): Feature matrix of shape (n_samples, n_features)
- `y` (ndarray): Target vector of shape (n_samples,)

##### `generate_regression_data()`

Generate synthetic regression datasets.

```python
X, y = generator.generate_regression_data(
    n_samples=1000,
    n_features=10,
    n_informative=10,
    n_targets=1,
    bias=0.0,
    effective_rank=None,
    tail_strength=0.5,
    noise=0.0,
    shuffle=True,
    coef=False,
    random_state=None
)
```

**Parameters:**

- `n_samples` (int): Number of samples
- `n_features` (int): Number of features
- `n_informative` (int): Number of informative features
- `n_targets` (int): Number of regression targets
- `bias` (float): Bias term in underlying linear model
- `effective_rank` (int): Approximate effective rank of input matrix
- `tail_strength` (float): Strength of fat tail in singular values
- `noise` (float): Standard deviation of Gaussian noise
- `shuffle` (bool): Shuffle samples and features
- `coef` (bool): If True, return coefficients
- `random_state` (int): Random seed

**Returns:**

- `X` (ndarray): Feature matrix
- `y` (ndarray): Target vector(s)
- `coef` (ndarray, optional): True coefficients if `coef=True`

##### `generate_clustering_data()`

Generate synthetic clustering datasets.

```python
X, y = generator.generate_clustering_data(
    n_samples=1000,
    n_features=2,
    n_clusters=3,
    n_centers=None,
    cluster_std=1.0,
    center_box=(-10, 10),
    shuffle=True,
    random_state=None,
    return_centers=False
)
```

**Parameters:**

- `n_samples` (int): Number of samples
- `n_features` (int): Number of features
- `n_clusters` (int): Number of clusters
- `n_centers` (int): Number of centers (defaults to n_clusters)
- `cluster_std` (float): Standard deviation of clusters
- `center_box` (tuple): Bounding box for cluster centers
- `shuffle` (bool): Shuffle the samples
- `random_state` (int): Random seed
- `return_centers` (bool): If True, return cluster centers

**Returns:**

- `X` (ndarray): Feature matrix
- `y` (ndarray): Cluster labels
- `centers` (ndarray, optional): Cluster centers if `return_centers=True`

##### `generate_time_series_data()`

Generate synthetic time series data.

```python
ts_data = generator.generate_time_series_data(
    n_samples=1000,
    n_features=5,
    trend='linear',
    seasonality=12,
    noise_level=0.1,
    anomaly_rate=0.05,
    random_state=None
)
```

**Parameters:**

- `n_samples` (int): Number of time steps
- `n_features` (int): Number of time series features
- `trend` (str): Type of trend ('linear', 'exponential', 'none')
- `seasonality` (int): Seasonal period (0 for no seasonality)
- `noise_level` (float): Standard deviation of noise
- `anomaly_rate` (float): Fraction of anomalous points
- `random_state` (int): Random seed

**Returns:**

- `ts_data` (dict): Dictionary containing:
  - `X`: Time series features
  - `y`: Target values (if applicable)
  - `timestamps`: Time indices
  - `anomalies`: Boolean mask for anomalies

## Data Preprocessors

### `src.data.preprocessors.DataPreprocessor`

Comprehensive data preprocessing pipeline.

```python
from src.data.preprocessors import DataPreprocessor

preprocessor = DataPreprocessor()
```

#### Methods

##### `fit_transform()`

Fit preprocessing pipeline and transform data.

```python
X_transformed = preprocessor.fit_transform(
    X,
    y=None,
    steps=['standard_scaler', 'feature_selection'],
    feature_selection_method='univariate',
    feature_selection_k=10,
    handle_missing='impute',
    handle_outliers='clip',
    encode_categorical='onehot'
)
```

**Parameters:**

- `X` (array-like): Input features
- `y` (array-like, optional): Target values
- `steps` (list): Preprocessing steps to apply
- `feature_selection_method` (str): Method for feature selection
- `feature_selection_k` (int): Number of features to select
- `handle_missing` (str): Strategy for missing values
- `handle_outliers` (str): Strategy for outliers
- `encode_categorical` (str): Categorical encoding method

**Returns:**

- `X_transformed` (ndarray): Preprocessed feature matrix

##### `transform()`

Transform new data using fitted preprocessor.

```python
X_new_transformed = preprocessor.transform(X_new)
```

##### `inverse_transform()`

Reverse preprocessing transformations.

```python
X_original = preprocessor.inverse_transform(X_transformed)
```

##### `get_feature_names()`

Get feature names after preprocessing.

```python
feature_names = preprocessor.get_feature_names()
```

#### Available Preprocessing Steps

**Scaling:**

- `standard_scaler`: Zero mean, unit variance
- `min_max_scaler`: Scale to [0, 1] range
- `robust_scaler`: Use median and IQR
- `quantile_uniform`: Transform to uniform distribution
- `quantile_normal`: Transform to normal distribution

**Feature Selection:**

- `univariate`: Statistical tests (f_classif, f_regression)
- `recursive`: Recursive feature elimination
- `lasso`: L1 regularization
- `tree_based`: Tree feature importance
- `variance_threshold`: Remove low-variance features

**Missing Value Handling:**

- `drop`: Remove samples with missing values
- `impute`: Fill with mean/median/mode
- `knn_impute`: K-nearest neighbors imputation
- `iterative_impute`: Iterative imputation

**Outlier Handling:**

- `clip`: Clip to percentile bounds
- `remove`: Remove outlier samples
- `isolation_forest`: Isolation Forest detection
- `local_outlier_factor`: LOF detection

## Data Validators

### `src.data.validators.DataValidator`

Validate data quality and integrity.

```python
from src.data.validators import DataValidator

validator = DataValidator()
```

#### Methods

##### `validate_classification_data()`

Validate classification dataset.

```python
validation_report = validator.validate_classification_data(
    X, y,
    check_missing=True,
    check_duplicates=True,
    check_outliers=True,
    check_class_balance=True,
    check_feature_correlation=True
)
```

**Returns:**

- `validation_report` (dict): Comprehensive validation results

##### `validate_regression_data()`

Validate regression dataset.

```python
validation_report = validator.validate_regression_data(
    X, y,
    check_missing=True,
    check_duplicates=True,
    check_outliers=True,
    check_feature_correlation=True,
    check_target_distribution=True
)
```

##### `detect_data_drift()`

Detect distribution drift between datasets.

```python
drift_report = validator.detect_data_drift(
    X_reference, X_current,
    method='ks_test',
    threshold=0.05
)
```

**Parameters:**

- `X_reference` (array-like): Reference dataset
- `X_current` (array-like): Current dataset
- `method` (str): Drift detection method
- `threshold` (float): Significance threshold

##### `validate_pipeline_input()`

Validate input for ML pipeline.

```python
is_valid, issues = validator.validate_pipeline_input(
    X, y,
    expected_features=None,
    expected_dtypes=None,
    allow_missing=False
)
```

## Specialized Data Generators

### `TextDataGenerator`

Generate text classification datasets.

```python
from src.data.generators import TextDataGenerator

text_gen = TextDataGenerator()
texts, labels = text_gen.generate_text_classification_data(
    n_samples=1000,
    n_classes=3,
    vocab_size=5000,
    avg_length=100,
    language='english'
)
```

### `ImageDataGenerator`

Generate synthetic image datasets.

```python
from src.data.generators import ImageDataGenerator

image_gen = ImageDataGenerator()
images, labels = image_gen.generate_image_classification_data(
    n_samples=1000,
    image_shape=(32, 32, 3),
    n_classes=10,
    complexity='medium'
)
```

### `GraphDataGenerator`

Generate graph/network datasets.

```python
from src.data.generators import GraphDataGenerator

graph_gen = GraphDataGenerator()
graphs, labels = graph_gen.generate_graph_classification_data(
    n_graphs=500,
    avg_nodes=20,
    edge_probability=0.1,
    n_classes=2
)
```

## Utility Functions

### `load_sample_datasets()`

Load built-in sample datasets.

```python
from src.data.loaders import load_sample_datasets

# Available datasets
datasets = load_sample_datasets()
X_iris, y_iris = datasets['iris']
X_wine, y_wine = datasets['wine']
X_breast_cancer, y_breast_cancer = datasets['breast_cancer']
```

### `create_train_test_split()`

Enhanced train-test splitting with stratification and validation sets.

```python
from src.data.utils import create_train_test_split

splits = create_train_test_split(
    X, y,
    test_size=0.2,
    val_size=0.1,
    stratify=True,
    random_state=42
)

X_train, X_val, X_test = splits['X_train'], splits['X_val'], splits['X_test']
y_train, y_val, y_test = splits['y_train'], splits['y_val'], splits['y_test']
```

### `generate_cross_validation_folds()`

Create custom cross-validation folds.

```python
from src.data.utils import generate_cross_validation_folds

cv_folds = generate_cross_validation_folds(
    X, y,
    cv_type='stratified_kfold',
    n_splits=5,
    shuffle=True,
    random_state=42
)

for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
    X_fold_train, X_fold_val = X[train_idx], X[val_idx]
    y_fold_train, y_fold_val = y[train_idx], y[val_idx]
```

## Configuration

### Default Settings

```python
# Data generation defaults
DEFAULT_GENERATOR_CONFIG = {
    'classification': {
        'n_samples': 1000,
        'n_features': 20,
        'n_classes': 2,
        'class_sep': 1.0,
        'flip_y': 0.01
    },
    'regression': {
        'n_samples': 1000,
        'n_features': 10,
        'noise': 0.1,
        'bias': 0.0
    },
    'clustering': {
        'n_samples': 1000,
        'n_features': 2,
        'n_clusters': 3,
        'cluster_std': 1.0
    }
}

# Preprocessing defaults
DEFAULT_PREPROCESSING_CONFIG = {
    'scaling': 'standard_scaler',
    'feature_selection': None,
    'missing_strategy': 'impute',
    'outlier_strategy': 'clip',
    'categorical_encoding': 'onehot'
}
```

## Examples

### Complete Data Pipeline

```python
from src.data.generators import DataGenerator
from src.data.preprocessors import DataPreprocessor
from src.data.validators import DataValidator

# Generate data
generator = DataGenerator(random_state=42)
X_raw, y = generator.generate_classification_data(
    n_samples=2000,
    n_features=25,
    n_classes=3,
    n_informative=20,
    noise=0.1
)

# Validate data quality
validator = DataValidator()
validation_report = validator.validate_classification_data(X_raw, y)
print(f"Data quality score: {validation_report['overall_score']:.2f}")

# Preprocess data
preprocessor = DataPreprocessor()
X_processed = preprocessor.fit_transform(
    X_raw,
    steps=['standard_scaler', 'feature_selection'],
    feature_selection_method='univariate',
    feature_selection_k=15
)

# Create train/test splits
from src.data.utils import create_train_test_split
splits = create_train_test_split(
    X_processed, y,
    test_size=0.2,
    val_size=0.1,
    stratify=True
)

print(f"Training set: {splits['X_train'].shape}")
print(f"Validation set: {splits['X_val'].shape}")
print(f"Test set: {splits['X_test'].shape}")
```

### Custom Data Generation

```python
# Create custom dataset with specific characteristics
def create_imbalanced_dataset():
    generator = DataGenerator()

    # Generate majority class
    X_maj, y_maj = generator.generate_classification_data(
        n_samples=800,
        n_features=10,
        n_classes=1,
        random_state=42
    )
    y_maj = np.zeros(len(y_maj))  # Class 0

    # Generate minority class
    X_min, y_min = generator.generate_classification_data(
        n_samples=200,
        n_features=10,
        n_classes=1,
        random_state=123
    )
    y_min = np.ones(len(y_min))  # Class 1

    # Combine datasets
    X = np.vstack([X_maj, X_min])
    y = np.hstack([y_maj, y_min])

    return X, y

X_imbalanced, y_imbalanced = create_imbalanced_dataset()
print(f"Class distribution: {np.bincount(y_imbalanced.astype(int))}")
```

## See Also

- [Getting Started Tutorial](../tutorials/getting_started.md)
- [Data Preprocessing Guide](../tutorials/data_preprocessing.md)
- [Model Training](../api_reference/models.md)
- [Evaluation Metrics](../api_reference/evaluation.md)
