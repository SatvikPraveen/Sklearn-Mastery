# File: docs/troubleshooting.md

# Location: docs/troubleshooting.md

# Troubleshooting Guide

Common issues, solutions, and debugging strategies for the ML Pipeline Framework.

## Installation Issues

### Package Installation Failures

**Problem**: `pip install` fails with dependency conflicts.

**Solution**:

```bash
# Create clean virtual environment
python -m venv ml_pipeline_env
source ml_pipeline_env/bin/activate  # On Windows: ml_pipeline_env\Scripts\activate

# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install minimal requirements first
pip install -r requirements-minimal.txt

# Then install full requirements
pip install -r requirements.txt
```

**Alternative**: Use conda environment

```bash
conda create -n ml_pipeline python=3.9
conda activate ml_pipeline
conda install --file requirements.txt
```

### ImportError: Cannot import module

**Problem**: `ImportError: No module named 'src'`

**Solution**: Install package in development mode

```bash
pip install -e .
```

**Or add to Python path**:

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
```

### Version Compatibility Issues

**Problem**: Sklearn version conflicts with framework.

**Solution**: Check compatibility matrix

```python
import sklearn
print(f"Scikit-learn version: {sklearn.__version__}")

# Required versions
REQUIRED_VERSIONS = {
    'scikit-learn': '>=1.0.0',
    'pandas': '>=1.3.0',
    'numpy': '>=1.20.0'
}
```

## Data Issues

### Memory Errors with Large Datasets

**Problem**: `MemoryError` when loading large datasets.

**Solutions**:

1. **Chunked processing**:

```python
from src.data.utils import process_in_chunks

def process_large_dataset(filepath, chunk_size=10000):
    results = []
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        processed_chunk = preprocess_chunk(chunk)
        results.append(processed_chunk)
    return pd.concat(results, ignore_index=True)
```

2. **Data type optimization**:

```python
def optimize_datatypes(df):
    # Convert float64 to float32
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')

    # Convert int64 to smaller ints where possible
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        if df[col].min() >= -128 and df[col].max() <= 127:
            df[col] = df[col].astype('int8')
        elif df[col].min() >= -32768 and df[col].max() <= 32767:
            df[col] = df[col].astype('int16')
        elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
            df[col] = df[col].astype('int32')

    return df
```

3. **Use sparse matrices**:

```python
from scipy.sparse import csr_matrix

# Convert to sparse format for high-dimensional data
X_sparse = csr_matrix(X)
```

### Data Quality Issues

**Problem**: Poor model performance due to data quality.

**Diagnostic checklist**:

```python
from src.data.validators import DataValidator

validator = DataValidator()

# Comprehensive data validation
validation_report = validator.validate_classification_data(X, y)

print("Data Quality Report:")
print(f"Missing values: {validation_report['missing_values_percent']:.2f}%")
print(f"Duplicate rows: {validation_report['duplicate_rows']}")
print(f"Class imbalance ratio: {validation_report['class_imbalance_ratio']:.2f}")
print(f"High correlation features: {len(validation_report['high_correlation_pairs'])}")
```

**Solutions**:

- Handle missing values appropriately
- Remove or merge duplicate entries
- Address class imbalance with resampling
- Remove highly correlated features

### Feature Engineering Problems

**Problem**: Features not improving model performance.

**Debug feature engineering**:

```python
# Check feature importance
from src.evaluation.utils import analyze_feature_importance

importance_analysis = analyze_feature_importance(
    model, X_train, y_train,
    feature_names=feature_names
)

# Identify low-importance features
low_importance = importance_analysis['features'][
    importance_analysis['importance'] < 0.001
]
print(f"Low importance features: {low_importance}")

# Check feature distributions
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, feature in enumerate(X.columns[:4]):
    ax = axes[i//2, i%2]
    X[feature].hist(bins=50, ax=ax)
    ax.set_title(f'{feature} Distribution')
```

## Model Training Issues

### Model Not Converging

**Problem**: Model fails to converge or takes too long.

**Solutions by algorithm**:

**Logistic Regression**:

```python
# Increase max_iter and try different solvers
model = LogisticRegression(
    max_iter=2000,
    solver='saga',  # Better for large datasets
    random_state=42
)
```

**SVM**:

```python
# Scale features and adjust C parameter
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = SVC(C=1.0, gamma='scale', max_iter=1000)
```

**Gradient Boosting**:

```python
# Reduce learning rate and increase n_estimators
model = GradientBoostingClassifier(
    learning_rate=0.01,  # Slower learning
    n_estimators=1000,
    early_stopping_rounds=10,
    validation_fraction=0.1
)
```

### Overfitting Issues

**Problem**: High training accuracy, poor test performance.

**Detection**:

```python
from src.evaluation.visualization import plot_learning_curve

# Plot learning curves to detect overfitting
plot_learning_curve(
    model, X_train, y_train,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10)
)
```

**Solutions**:

1. **Regularization**:

```python
# Add L1/L2 regularization
model = Ridge(alpha=1.0)  # L2 regularization
model = Lasso(alpha=0.1)  # L1 regularization
```

2. **Early stopping**:

```python
# For gradient boosting
model = GradientBoostingClassifier(
    n_estimators=1000,
    validation_fraction=0.1,
    n_iter_no_change=10,
    tol=1e-4
)
```

3. **Cross-validation**:

```python
# Use cross-validation for hyperparameter tuning
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X_train, y_train, cv=10)
print(f"CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

### Underfitting Issues

**Problem**: Both training and test performance are poor.

**Solutions**:

1. **Increase model complexity**:

```python
# More complex models
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,  # Remove depth limit
    min_samples_split=2
)
```

2. **Feature engineering**:

```python
# Add polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

3. **Ensemble methods**:

```python
from src.models.ensemble.methods import EnsembleMethods
ensemble = EnsembleMethods()
voting_clf = ensemble.get_voting_classifier([
    ('rf', RandomForestClassifier()),
    ('gb', GradientBoostingClassifier()),
    ('svm', SVC(probability=True))
])
```

## Performance Issues

### Slow Training Times

**Problem**: Model training takes too long.

**Optimization strategies**:

1. **Parallel processing**:

```python
# Use all available cores
model = RandomForestClassifier(n_jobs=-1)

# Parallel cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)
```

2. **Feature selection**:

```python
from src.pipelines.custom_transformers import FeatureSelector

# Remove irrelevant features first
selector = FeatureSelector(method='univariate', k=100)
X_selected = selector.fit_transform(X, y)
```

3. **Sample size reduction**:

```python
# Use stratified sampling for large datasets
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, train_size=0.1, random_state=42)
train_index, _ = next(sss.split(X, y))
X_sample = X[train_index]
y_sample = y[train_index]
```

### Memory Usage Optimization

**Problem**: High memory consumption during training.

**Solutions**:

```python
# Use online/incremental learning
from sklearn.linear_model import SGDClassifier

# Incremental learning
model = SGDClassifier()
for batch_X, batch_y in get_data_batches():
    model.partial_fit(batch_X, batch_y, classes=np.unique(y))

# Use memory mapping for large arrays
import numpy as np
X_memmap = np.memmap('large_dataset.dat', dtype='float32', mode='r', shape=(n_samples, n_features))
```

## Pipeline Issues

### Pipeline Serialization Problems

**Problem**: Cannot save/load trained pipelines.

**Solutions**:

```python
# Use joblib instead of pickle
import joblib

# Save pipeline
joblib.dump(pipeline, 'trained_pipeline.joblib', compress=3)

# Load pipeline
loaded_pipeline = joblib.load('trained_pipeline.joblib')

# For complex custom transformers, use dill
import dill
with open('pipeline.dill', 'wb') as f:
    dill.dump(pipeline, f)
```

### Custom Transformer Issues

**Problem**: Custom transformers not working in pipelines.

**Solution**: Ensure proper sklearn interface:

```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, param1=None):
        self.param1 = param1

    def fit(self, X, y=None):
        # Fit logic here
        return self

    def transform(self, X):
        # Transform logic here
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        # Return output feature names
        return output_features
```

## Debugging Strategies

### Model Performance Debugging

**Step-by-step debugging**:

```python
def debug_model_performance(model, X_train, y_train, X_test, y_test):
    """Comprehensive model debugging."""

    print("=== Model Performance Debug ===")

    # 1. Check data shapes and types
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Data types: {X_train.dtypes.value_counts()}")

    # 2. Check for data leakage
    train_mean = X_train.mean()
    test_mean = X_test.mean()
    print(f"Feature mean difference: {abs(train_mean - test_mean).max():.4f}")

    # 3. Basic model performance
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Training score: {train_score:.4f}")
    print(f"Test score: {test_score:.4f}")
    print(f"Overfitting gap: {train_score - test_score:.4f}")

    # 4. Check class distribution
    if hasattr(y_train, 'value_counts'):
        print(f"Class distribution: {y_train.value_counts()}")

    # 5. Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        top_features = np.argsort(model.feature_importances_)[-10:]
        print(f"Top 10 features: {top_features}")

    return {
        'train_score': train_score,
        'test_score': test_score,
        'overfitting_gap': train_score - test_score
    }
```

### Cross-Validation Debugging

**Problem**: Inconsistent cross-validation results.

**Debug CV issues**:

```python
from sklearn.model_selection import cross_validate
import numpy as np

def debug_cross_validation(model, X, y, cv=5):
    """Debug cross-validation inconsistencies."""

    cv_results = cross_validate(
        model, X, y, cv=cv,
        scoring=['accuracy', 'f1_weighted'],
        return_train_score=True,
        return_estimator=True
    )

    train_scores = cv_results['train_accuracy']
    test_scores = cv_results['test_accuracy']

    print(f"Train scores: {train_scores}")
    print(f"Test scores: {test_scores}")
    print(f"Train mean ± std: {train_scores.mean():.3f} ± {train_scores.std():.3f}")
    print(f"Test mean ± std: {test_scores.mean():.3f} ± {test_scores.std():.3f}")

    # Check for high variance
    if test_scores.std() > 0.1:
        print("WARNING: High variance in CV scores - check data stratification")

    return cv_results
```

## Error Messages and Solutions

### Common Error Messages

#### `ValueError: Input contains NaN`

**Cause**: Missing values in dataset.

**Solution**:

```python
# Check for missing values
print(f"Missing values: {X.isnull().sum().sum()}")

# Handle missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
```

#### `ValueError: could not convert string to float`

**Cause**: Categorical variables not encoded.

**Solution**:

```python
# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
categorical_columns = X.select_dtypes(include=['object']).columns

for col in categorical_columns:
    X[col] = le.fit_transform(X[col].astype(str))
```

#### `ValueError: Input contains infinity or a value too large`

**Cause**: Infinite or very large values in features.

**Solution**:

```python
# Check for infinite values
print(f"Infinite values: {np.isinf(X).sum().sum()}")

# Replace infinite values
X = np.where(np.isinf(X), np.nan, X)
# Then handle NaN values as above
```

#### `MemoryError`

**Solutions**:

```python
# 1. Reduce precision
X = X.astype('float32')

# 2. Use sparse matrices
from scipy.sparse import csr_matrix
X_sparse = csr_matrix(X)

# 3. Process in batches
batch_size = 1000
for i in range(0, len(X), batch_size):
    batch = X[i:i+batch_size]
    # Process batch
```

#### `NotFittedError: This model has not been fitted yet`

**Cause**: Trying to predict before training model.

**Solution**:

```python
# Always fit before predict
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Logging and Monitoring

### Enable Debug Logging

```python
import logging
from src.config.logging_config import setup_logging

# Enable debug logging
setup_logging(level=logging.DEBUG)

# Add custom logging to your code
logger = logging.getLogger(__name__)
logger.debug("Debug information")
logger.info("Training started")
logger.warning("Performance issue detected")
logger.error("Training failed")
```

### Performance Monitoring

```python
import time
import psutil
import numpy as np

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.start_memory = None

    def start(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024

    def stop(self):
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024

        print(f"Execution time: {end_time - self.start_time:.2f}s")
        print(f"Memory usage: {end_memory - self.start_memory:.2f}MB")

        return {
            'time': end_time - self.start_time,
            'memory': end_memory - self.start_memory
        }

# Usage
monitor = PerformanceMonitor()
monitor.start()
# ... your code ...
stats = monitor.stop()
```

## Testing and Validation

### Unit Testing Models

```python
import unittest
from src.models.supervised.classification import ClassificationModels

class TestClassificationModels(unittest.TestCase):
    def setUp(self):
        self.models = ClassificationModels()

    def test_random_forest_creation(self):
        rf = self.models.get_random_forest()
        self.assertIsNotNone(rf)

    def test_model_fit_predict(self):
        from src.data.generators import DataGenerator

        generator = DataGenerator()
        X, y = generator.generate_classification_data(n_samples=100)

        rf = self.models.get_random_forest()
        rf.fit(X, y)
        predictions = rf.predict(X)

        self.assertEqual(len(predictions), len(y))

if __name__ == '__main__':
    unittest.main()
```

### Integration Testing

```python
def test_complete_pipeline():
    """Test entire pipeline from data to predictions."""

    # Generate data
    from src.data.generators import DataGenerator
    generator = DataGenerator()
    X, y = generator.generate_classification_data(n_samples=1000)

    # Create pipeline
    from src.pipelines.pipeline_factory import PipelineFactory
    factory = PipelineFactory()
    pipeline = factory.create_classification_pipeline('random_forest')

    # Train and predict
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    # Validate results
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, predictions)

    assert accuracy > 0.7, f"Pipeline accuracy too low: {accuracy}"
    print(f"Integration test passed - Accuracy: {accuracy:.3f}")

test_complete_pipeline()
```

## Environment Issues

### Version Conflicts

**Check versions**:

```python
import pkg_resources

def check_package_versions():
    """Check installed package versions."""
    required_packages = [
        'scikit-learn', 'pandas', 'numpy', 'matplotlib',
        'seaborn', 'joblib', 'scipy'
    ]

    for package in required_packages:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"{package}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"{package}: NOT INSTALLED")

check_package_versions()
```

### GPU/CUDA Issues

**Check GPU availability**:

```python
# For XGBoost GPU support
try:
    import xgboost as xgb
    print(f"XGBoost version: {xgb.__version__}")
    print(f"XGBoost GPU support: {xgb.get_config()['use_gpu']}")
except Exception as e:
    print(f"XGBoost GPU issue: {e}")

# System info
import platform
print(f"Platform: {platform.platform()}")
print(f"Python: {platform.python_version()}")
```

## Quick Fixes Checklist

### Before Running Code

- [ ] Check data for missing values, inf, NaN
- [ ] Verify feature and target shapes match
- [ ] Ensure categorical variables are encoded
- [ ] Scale features if using distance-based algorithms
- [ ] Check for data leakage between train/test

### Model Training

- [ ] Set random_state for reproducibility
- [ ] Use appropriate cross-validation strategy
- [ ] Monitor for overfitting/underfitting
- [ ] Start with simple models before complex ones
- [ ] Check class balance for classification

### Performance Issues

- [ ] Use n_jobs=-1 for parallel processing
- [ ] Consider feature selection for high-dimensional data
- [ ] Use incremental learning for large datasets
- [ ] Profile memory usage for optimization
- [ ] Consider data type optimization (float32 vs float64)

### Debugging Steps

1. **Data**: Validate input data quality and format
2. **Model**: Test with simple model first
3. **Pipeline**: Check each step individually
4. **Cross-validation**: Verify CV strategy matches problem
5. **Metrics**: Ensure appropriate evaluation metrics
6. **Logs**: Enable detailed logging for diagnosis

## Getting Help

### Framework-Specific Issues

- Check the [GitHub Issues](https://github.com/your-org/ml-pipeline-framework/issues)
- Review [API documentation](api_reference/index.md)
- Run diagnostic scripts in `examples/diagnostics/`

### General ML Issues

- Sklearn documentation: https://scikit-learn.org/stable/
- Stack Overflow with tags: `scikit-learn`, `machine-learning`
- ML forums and communities

### Performance Optimization

- Profile code with `cProfile` or `py-spy`
- Use memory profilers like `memory_profiler`
- Consider cloud-based solutions for large datasets

## Reporting Bugs

When reporting issues, include:

1. **Environment**: Python version, OS, package versions
2. **Data**: Dataset size, feature types, target distribution
3. **Code**: Minimal reproducible example
4. **Error**: Complete error traceback
5. **Expected vs Actual**: What you expected vs what happened

```python
# Diagnostic script to include with bug reports
def create_bug_report():
    """Generate system and environment info for bug reports."""
    import sys, platform, pkg_resources

    report = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'packages': {pkg.project_name: pkg.version
                    for pkg in pkg_resources.working_set}
    }

    print("=== Bug Report Info ===")
    for key, value in report.items():
        print(f"{key}: {value}")

    return report

create_bug_report()
```

## See Also

- [Getting Started Guide](tutorials/getting_started.md)
- [API Reference](api_reference/index.md)
- [FAQ](FAQ.md)
- [Contributing Guidelines](CONTRIBUTING.md)
