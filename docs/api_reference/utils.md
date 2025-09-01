# Utils API Reference

**File Location:** `docs/api_reference/utils.md`

Utility functions and helper classes for data preprocessing, visualization, and common machine learning tasks.

## Data Utilities

### `data_utils` Module

#### `load_dataset(name, **kwargs)`

Load built-in datasets with preprocessing options.

**Parameters:**

- `name` (str): Dataset name ('iris', 'wine', 'breast_cancer', 'digits', 'boston')
- `return_X_y` (bool, default=False): Return (data, target) tuple instead of Bunch object
- `as_frame` (bool, default=False): Return data as pandas DataFrame

**Returns:**

- Dataset object or (X, y) tuple

**Example:**

```python
from myml.utils.data_utils import load_dataset

# Load as sklearn Bunch
dataset = load_dataset('iris')
X, y = dataset.data, dataset.target

# Load as tuple
X, y = load_dataset('iris', return_X_y=True)

# Load as DataFrame
data = load_dataset('iris', as_frame=True)
```

#### `split_data(X, y, test_size=0.2, val_size=0.1, **kwargs)`

Split data into train/validation/test sets.

**Parameters:**

- `X` (array-like): Features
- `y` (array-like): Target variable
- `test_size` (float, default=0.2): Test set proportion
- `val_size` (float, default=0.1): Validation set proportion
- `random_state` (int, default=None): Random seed
- `stratify` (bool, default=True): Stratified split for classification

**Returns:**

- tuple: (X_train, X_val, X_test, y_train, y_val, y_test)

**Example:**

```python
from myml.utils.data_utils import split_data

X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    X, y, test_size=0.3, val_size=0.15, random_state=42
)
```

#### `generate_synthetic_data(n_samples=1000, n_features=20, **kwargs)`

Generate synthetic datasets for testing.

**Parameters:**

- `n_samples` (int): Number of samples
- `n_features` (int): Number of features
- `n_classes` (int, default=2): Number of classes (classification)
- `n_clusters` (int, default=3): Number of clusters (clustering)
- `task_type` (str): 'classification', 'regression', or 'clustering'
- `noise` (float, default=0.1): Noise level
- `random_state` (int, default=None): Random seed

**Returns:**

- tuple: (X, y) for supervised tasks, X for clustering

**Example:**

```python
from myml.utils.data_utils import generate_synthetic_data

# Classification dataset
X, y = generate_synthetic_data(
    n_samples=1000, n_features=10, n_classes=3,
    task_type='classification', noise=0.1, random_state=42
)

# Regression dataset
X, y = generate_synthetic_data(
    n_samples=500, n_features=5, task_type='regression',
    noise=0.2, random_state=42
)

# Clustering dataset
X = generate_synthetic_data(
    n_samples=800, n_features=2, n_clusters=4,
    task_type='clustering', random_state=42
)
```

### `preprocessing_utils` Module

#### `scale_features(X, method='standard', **kwargs)`

Scale features using various methods.

**Parameters:**

- `X` (array-like): Input features
- `method` (str): 'standard', 'minmax', 'robust', 'quantile'
- `feature_range` (tuple, default=(0,1)): Range for MinMax scaling
- `quantile_range` (tuple, default=(25.0, 75.0)): Quantile range for robust scaling

**Returns:**

- array: Scaled features
- scaler: Fitted scaler object

**Example:**

```python
from myml.utils.preprocessing_utils import scale_features

# Standard scaling
X_scaled, scaler = scale_features(X_train, method='standard')
X_test_scaled = scaler.transform(X_test)

# MinMax scaling to [0, 1]
X_scaled, scaler = scale_features(X_train, method='minmax', feature_range=(0, 1))

# Robust scaling (less sensitive to outliers)
X_scaled, scaler = scale_features(X_train, method='robust')
```

#### `encode_categorical(X, method='onehot', **kwargs)`

Encode categorical variables.

**Parameters:**

- `X` (array-like or DataFrame): Categorical data
- `method` (str): 'onehot', 'label', 'target', 'binary'
- `drop` (str, default='first'): Strategy for dropping columns in one-hot
- `handle_unknown` (str, default='error'): How to handle unknown categories

**Returns:**

- array: Encoded features
- encoder: Fitted encoder object

**Example:**

```python
from myml.utils.preprocessing_utils import encode_categorical
import pandas as pd

# One-hot encoding
df = pd.DataFrame({'color': ['red', 'blue', 'green', 'red']})
X_encoded, encoder = encode_categorical(df, method='onehot')

# Label encoding
X_encoded, encoder = encode_categorical(df['color'], method='label')
```

#### `handle_missing_values(X, strategy='mean', **kwargs)`

Handle missing values in datasets.

**Parameters:**

- `X` (array-like): Input features with missing values
- `strategy` (str): 'mean', 'median', 'most_frequent', 'constant'
- `fill_value` (scalar): Value for 'constant' strategy
- `missing_values` (scalar, default=np.nan): Placeholder for missing values

**Returns:**

- array: Imputed features
- imputer: Fitted imputer object

**Example:**

```python
from myml.utils.preprocessing_utils import handle_missing_values
import numpy as np

# Create data with missing values
X_missing = np.array([[1, 2, np.nan], [np.nan, 3, 4], [5, np.nan, 6]])

# Mean imputation
X_imputed, imputer = handle_missing_values(X_missing, strategy='mean')

# Median imputation
X_imputed, imputer = handle_missing_values(X_missing, strategy='median')

# Constant fill
X_imputed, imputer = handle_missing_values(
    X_missing, strategy='constant', fill_value=0
)
```

## Visualization Utilities

### `plot_utils` Module

#### `plot_confusion_matrix(y_true, y_pred, classes=None, **kwargs)`

Plot confusion matrix with customization options.

**Parameters:**

- `y_true` (array-like): True labels
- `y_pred` (array-like): Predicted labels
- `classes` (list, optional): Class names
- `normalize` (str, default=None): 'true', 'pred', 'all', or None
- `title` (str, default='Confusion Matrix'): Plot title
- `cmap` (str, default='Blues'): Colormap

**Returns:**

- matplotlib figure and axes objects

**Example:**

```python
from myml.utils.plot_utils import plot_confusion_matrix

fig, ax = plot_confusion_matrix(
    y_test, y_pred,
    classes=['Class A', 'Class B', 'Class C'],
    normalize='true',
    title='Normalized Confusion Matrix'
)
```

#### `plot_learning_curves(estimator, X, y, **kwargs)`

Plot training and validation learning curves.

**Parameters:**

- `estimator`: ML model
- `X` (array-like): Features
- `y` (array-like): Target variable
- `cv` (int, default=5): Cross-validation folds
- `train_sizes` (array-like, default=None): Training set sizes
- `scoring` (str, default=None): Scoring metric

**Returns:**

- matplotlib figure and axes objects

**Example:**

```python
from myml.utils.plot_utils import plot_learning_curves
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
fig, ax = plot_learning_curves(
    rf, X_train, y_train, cv=5, scoring='accuracy'
)
```

#### `plot_feature_importance(model, feature_names=None, **kwargs)`

Plot feature importance for tree-based models.

**Parameters:**

- `model`: Fitted model with feature*importances* attribute
- `feature_names` (list, optional): Names of features
- `top_k` (int, default=None): Show only top k features
- `title` (str, default='Feature Importance'): Plot title

**Returns:**

- matplotlib figure and axes objects

**Example:**

```python
from myml.utils.plot_utils import plot_feature_importance

# Train model
rf.fit(X_train, y_train)

# Plot importance
fig, ax = plot_feature_importance(
    rf, feature_names=feature_names, top_k=10
)
```

#### `plot_roc_curves(models_dict, X_test, y_test, **kwargs)`

Plot ROC curves for multiple models.

**Parameters:**

- `models_dict` (dict): Dictionary of {name: fitted_model}
- `X_test` (array-like): Test features
- `y_test` (array-like): Test labels
- `title` (str, default='ROC Curves'): Plot title

**Returns:**

- matplotlib figure and axes objects

**Example:**

```python
from myml.utils.plot_utils import plot_roc_curves

models = {
    'Random Forest': rf_model,
    'SVM': svm_model,
    'Logistic Regression': lr_model
}

fig, ax = plot_roc_curves(models, X_test, y_test)
```

#### `plot_decision_boundary(model, X, y, **kwargs)`

Plot 2D decision boundary for classification models.

**Parameters:**

- `model`: Fitted classification model
- `X` (array-like): 2D features
- `y` (array-like): Target labels
- `resolution` (float, default=0.01): Mesh resolution
- `alpha` (float, default=0.8): Transparency for decision regions

**Returns:**

- matplotlib figure and axes objects

**Example:**

```python
from myml.utils.plot_utils import plot_decision_boundary

# For 2D data only
if X_train.shape[1] == 2:
    fig, ax = plot_decision_boundary(model, X_train, y_train)
```

## Model Utilities

### `model_utils` Module

#### `cross_validate_models(models_dict, X, y, **kwargs)`

Cross-validate multiple models and compare performance.

**Parameters:**

- `models_dict` (dict): Dictionary of {name: model}
- `X` (array-like): Features
- `y` (array-like): Target variable
- `cv` (int, default=5): Number of CV folds
- `scoring` (str or list, default='accuracy'): Scoring metric(s)
- `return_train_score` (bool, default=True): Include training scores

**Returns:**

- DataFrame: Cross-validation results

**Example:**

```python
from myml.utils.model_utils import cross_validate_models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42)
}

results = cross_validate_models(
    models, X_train, y_train, cv=5,
    scoring=['accuracy', 'precision', 'recall', 'f1']
)
print(results)
```

#### `hyperparameter_search(model, param_grid, X, y, **kwargs)`

Perform hyperparameter search with cross-validation.

**Parameters:**

- `model`: ML model
- `param_grid` (dict): Parameter grid for search
- `X` (array-like): Features
- `y` (array-like): Target variable
- `search_type` (str, default='grid'): 'grid' or 'random'
- `cv` (int, default=5): Cross-validation folds
- `scoring` (str, default='accuracy'): Scoring metric
- `n_jobs` (int, default=-1): Parallel jobs

**Returns:**

- Best model and search results

**Example:**

```python
from myml.utils.model_utils import hyperparameter_search
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

best_model, search_results = hyperparameter_search(
    rf, param_grid, X_train, y_train,
    search_type='grid', cv=5, scoring='f1_weighted'
)

print(f"Best parameters: {best_model.get_params()}")
print(f"Best CV score: {search_results.best_score_:.3f}")
```

#### `evaluate_model_performance(model, X_test, y_test, **kwargs)`

Comprehensive model evaluation with multiple metrics.

**Parameters:**

- `model`: Fitted model
- `X_test` (array-like): Test features
- `y_test` (array-like): Test labels
- `task_type` (str): 'classification' or 'regression'
- `average` (str, default='weighted'): Averaging for multiclass metrics
- `plot_results` (bool, default=True): Generate evaluation plots

**Returns:**

- Dictionary of evaluation metrics

**Example:**

```python
from myml.utils.model_utils import evaluate_model_performance

# For classification
metrics = evaluate_model_performance(
    best_model, X_test, y_test,
    task_type='classification', plot_results=True
)

print(f"Test Accuracy: {metrics['accuracy']:.3f}")
print(f"Test F1-Score: {metrics['f1_score']:.3f}")
```

### `ensemble_utils` Module

#### `create_voting_ensemble(base_models, **kwargs)`

Create voting ensemble from base models.

**Parameters:**

- `base_models` (list): List of (name, model) tuples
- `voting` (str, default='hard'): 'hard' or 'soft' voting
- `weights` (list, optional): Model weights for voting

**Returns:**

- Fitted voting classifier

**Example:**

```python
from myml.utils.ensemble_utils import create_voting_ensemble

base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('lr', LogisticRegression(random_state=42))
]

voting_ensemble = create_voting_ensemble(
    base_models, voting='soft', weights=[2, 1, 1]
)
voting_ensemble.fit(X_train, y_train)
```

#### `create_stacking_ensemble(base_models, meta_model, **kwargs)`

Create stacking ensemble with meta-learner.

**Parameters:**

- `base_models` (list): List of (name, model) tuples
- `meta_model`: Meta-learner model
- `cv` (int, default=5): Cross-validation for stacking
- `use_features_in_secondary` (bool, default=False): Include original features

**Returns:**

- Fitted stacking classifier

**Example:**

```python
from myml.utils.ensemble_utils import create_stacking_ensemble
from sklearn.linear_model import LogisticRegression

meta_learner = LogisticRegression(random_state=42)
stacking_ensemble = create_stacking_ensemble(
    base_models, meta_learner, cv=5
)
stacking_ensemble.fit(X_train, y_train)
```

## File I/O Utilities

### `io_utils` Module

#### `save_model(model, filepath, **kwargs)`

Save trained model to disk.

**Parameters:**

- `model`: Trained model object
- `filepath` (str): Path to save model
- `format` (str, default='pickle'): 'pickle' or 'joblib'
- `compress` (bool, default=True): Compress saved file

**Example:**

```python
from myml.utils.io_utils import save_model

# Save model
save_model(best_model, 'models/best_rf_model.pkl')

# Save with compression
save_model(best_model, 'models/best_rf_model.joblib',
          format='joblib', compress=True)
```

#### `load_model(filepath, **kwargs)`

Load saved model from disk.

**Parameters:**

- `filepath` (str): Path to saved model
- `format` (str, default='auto'): 'pickle', 'joblib', or 'auto'

**Returns:**

- Loaded model object

**Example:**

```python
from myml.utils.io_utils import load_model

# Load model
loaded_model = load_model('models/best_rf_model.pkl')

# Make predictions
predictions = loaded_model.predict(X_new)
```

#### `save_results(results, filepath, **kwargs)`

Save experiment results to various formats.

**Parameters:**

- `results` (dict or DataFrame): Results to save
- `filepath` (str): Output file path
- `format` (str, default='json'): 'json', 'csv', 'excel'
- `indent` (int, default=2): JSON indentation

**Example:**

```python
from myml.utils.io_utils import save_results

# Save results as JSON
save_results(cv_results, 'results/experiment_results.json')

# Save as CSV
save_results(metrics_df, 'results/model_metrics.csv')
```

## Performance Utilities

### `performance_utils` Module

#### `profile_memory_usage(func, *args, **kwargs)`

Profile memory usage of a function.

**Parameters:**

- `func`: Function to profile
- `*args, **kwargs`: Function arguments

**Returns:**

- Result of function and memory usage statistics

**Example:**

```python
from myml.utils.performance_utils import profile_memory_usage

def train_large_model():
    model = RandomForestClassifier(n_estimators=1000)
    model.fit(X_train, y_train)
    return model

result, memory_stats = profile_memory_usage(train_large_model)
print(f"Peak memory usage: {memory_stats['peak_memory']:.2f} MB")
```

#### `benchmark_algorithms(algorithms, X, y, **kwargs)`

Benchmark multiple algorithms on the same dataset.

**Parameters:**

- `algorithms` (dict): Dictionary of {name: model}
- `X` (array-like): Features
- `y` (array-like): Target variable
- `n_runs` (int, default=5): Number of timing runs
- `test_size` (float, default=0.2): Test set proportion

**Returns:**

- DataFrame with performance comparisons

**Example:**

```python
from myml.utils.performance_utils import benchmark_algorithms

algorithms = {
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC(),
    'Logistic Regression': LogisticRegression()
}

benchmark_results = benchmark_algorithms(
    algorithms, X, y, n_runs=3, test_size=0.3
)
print(benchmark_results)
```

## Configuration Utilities

### `config_utils` Module

#### `set_global_config(**kwargs)`

Set global configuration parameters.

**Parameters:**

- `random_state` (int): Global random seed
- `n_jobs` (int): Default number of parallel jobs
- `verbose` (bool): Enable verbose output

**Example:**

```python
from myml.utils.config_utils import set_global_config

# Set global configuration
set_global_config(random_state=42, n_jobs=-1, verbose=True)
```

#### `get_config_info()`

Get information about current configuration.

**Returns:**

- Dictionary with configuration details

**Example:**

```python
from myml.utils.config_utils import get_config_info

config_info = get_config_info()
print(f"Random state: {config_info['random_state']}")
print(f"Number of jobs: {config_info['n_jobs']}")
```

These utility functions provide essential functionality for data preprocessing, model evaluation, visualization, and performance analysis in machine learning workflows.
