# File: docs/algorithm_guides/regression.md

# Location: docs/algorithm_guides/regression.md

# Regression Algorithms

Comprehensive guide to regression algorithms available in the ML Pipeline Framework.

## Overview

Regression algorithms predict continuous numerical values. Our framework provides optimized implementations with automatic hyperparameter tuning and performance evaluation.

## Available Algorithms

### Linear Regression

**Best for**: Linear relationships, interpretability, baseline models

```python
from src.models.supervised.regression import RegressionModels

models = RegressionModels()
linear_reg = models.get_linear_regression()

# Custom configuration
linear_reg = models.get_linear_regression(
    fit_intercept=True,
    normalize=True,
    copy_X=True
)
```

**When to use**:

- Linear relationships between features and target
- Need for interpretable coefficients
- Baseline model for comparison
- Small to medium datasets

**Hyperparameters**:

- `fit_intercept`: Whether to calculate intercept (default: True)
- `normalize`: Whether to normalize features (default: False)
- `positive`: Whether to force positive coefficients (default: False)

### Random Forest Regression

**Best for**: Non-linear relationships, feature importance, robust predictions

```python
rf_reg = models.get_random_forest_regression()

# Optimized for performance
rf_reg = models.get_random_forest_regression(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1
)
```

**When to use**:

- Non-linear relationships
- Mixed data types (numerical/categorical)
- Need feature importance scores
- Medium to large datasets

**Key hyperparameters**:

- `n_estimators`: Number of trees (50-500)
- `max_depth`: Maximum tree depth (3-20)
- `min_samples_split`: Minimum samples to split (2-20)
- `max_features`: Features per split ('sqrt', 'log2', None)

### Gradient Boosting Regression

**Best for**: High accuracy, complex patterns, structured data

```python
gb_reg = models.get_gradient_boosting_regression()

# High performance configuration
gb_reg = models.get_gradient_boosting_regression(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    validation_fraction=0.1
)
```

**When to use**:

- Maximum predictive accuracy needed
- Complex non-linear patterns
- Structured/tabular data
- Sufficient computational resources

**Tuning strategy**:

1. Start with `learning_rate=0.1`, `n_estimators=100`
2. Increase `n_estimators` until overfitting
3. Reduce `learning_rate` and increase `n_estimators`
4. Tune `max_depth` (3-8) and `subsample` (0.5-1.0)

### Support Vector Regression (SVR)

**Best for**: Non-linear patterns, robust to outliers, high-dimensional data

```python
svr = models.get_svr()

# For non-linear patterns
svr_rbf = models.get_svr(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    epsilon=0.1
)
```

**When to use**:

- High-dimensional feature spaces
- Robust predictions needed
- Non-linear relationships
- Medium-sized datasets

**Kernel selection**:

- `linear`: Linear relationships, high dimensions
- `rbf`: Non-linear, general purpose (default)
- `poly`: Polynomial relationships
- `sigmoid`: Neural network-like

### Ridge Regression

**Best for**: Multicollinearity, regularization, stable predictions

```python
ridge = models.get_ridge_regression()

# With cross-validation for alpha
ridge = models.get_ridge_regression(
    alpha=1.0,
    fit_intercept=True,
    normalize=True,
    solver='auto'
)
```

**When to use**:

- Multicollinearity in features
- Need regularization
- Prevent overfitting
- More features than samples

**Alpha selection**:

- Small alpha (0.1-1.0): Less regularization
- Large alpha (10-100): More regularization
- Use RidgeCV for automatic selection

### Lasso Regression

**Best for**: Feature selection, sparse solutions, interpretability

```python
lasso = models.get_lasso_regression()

# For feature selection
lasso = models.get_lasso_regression(
    alpha=0.1,
    fit_intercept=True,
    normalize=True,
    max_iter=2000
)
```

**When to use**:

- Automatic feature selection needed
- Sparse solutions preferred
- Many irrelevant features
- Interpretable model required

**Benefits**:

- Automatic feature selection (coefficients → 0)
- Sparse model interpretation
- Handles irrelevant features
- Prevents overfitting

## Performance Comparison

| Algorithm         | Speed      | Accuracy   | Interpretability | Scalability |
| ----------------- | ---------- | ---------- | ---------------- | ----------- |
| Linear            | ⭐⭐⭐⭐⭐ | ⭐⭐       | ⭐⭐⭐⭐⭐       | ⭐⭐⭐⭐⭐  |
| Random Forest     | ⭐⭐⭐     | ⭐⭐⭐⭐   | ⭐⭐⭐           | ⭐⭐⭐⭐    |
| Gradient Boosting | ⭐⭐       | ⭐⭐⭐⭐⭐ | ⭐⭐             | ⭐⭐⭐      |
| SVR               | ⭐⭐       | ⭐⭐⭐⭐   | ⭐               | ⭐⭐        |
| Ridge             | ⭐⭐⭐⭐   | ⭐⭐⭐     | ⭐⭐⭐⭐         | ⭐⭐⭐⭐⭐  |
| Lasso             | ⭐⭐⭐⭐   | ⭐⭐⭐     | ⭐⭐⭐⭐⭐       | ⭐⭐⭐⭐    |

## Usage Examples

### Basic Regression Pipeline

```python
from src.data.generators import DataGenerator
from src.models.supervised.regression import RegressionModels
from src.evaluation.metrics import ModelEvaluator

# Generate sample data
generator = DataGenerator()
X, y = generator.generate_regression_data(
    n_samples=1000,
    n_features=10,
    noise=0.1,
    random_state=42
)

# Train multiple models
models = RegressionModels()
evaluator = ModelEvaluator()

algorithms = [
    ('Linear', models.get_linear_regression()),
    ('Random Forest', models.get_random_forest_regression()),
    ('Gradient Boosting', models.get_gradient_boosting_regression())
]

results = {}
for name, model in algorithms:
    scores = evaluator.cross_validate_model(model, X, y, cv=5)
    results[name] = {
        'mean_score': scores['test_score'].mean(),
        'std_score': scores['test_score'].std()
    }

# Print results
for name, metrics in results.items():
    print(f"{name}: {metrics['mean_score']:.3f} ± {metrics['std_score']:.3f}")
```

### Advanced Pipeline with Preprocessing

```python
from src.pipelines.pipeline_factory import PipelineFactory
from src.data.preprocessors import DataPreprocessor

# Create preprocessing pipeline
preprocessor = DataPreprocessor()
pipeline_factory = PipelineFactory()

# Build complete pipeline
pipeline = pipeline_factory.create_regression_pipeline(
    model_type='random_forest',
    preprocessing_steps=['standard_scaler', 'feature_selection'],
    model_params={
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 5
    }
)

# Train and evaluate
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

# Get feature importance
if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
    importances = pipeline.named_steps['model'].feature_importances_
    feature_names = preprocessor.get_feature_names()

    for name, importance in zip(feature_names, importances):
        print(f"{name}: {importance:.3f}")
```

## Hyperparameter Tuning

### Grid Search Example

```python
from src.pipelines.model_selection import ModelSelector

selector = ModelSelector()

# Define parameter grid
param_grid = {
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }
}

# Perform grid search
best_model = selector.grid_search_cv(
    X_train, y_train,
    model_type='random_forest',
    param_grid=param_grid['random_forest'],
    cv=5,
    scoring='neg_mean_squared_error'
)

print("Best parameters:", best_model.best_params_)
print("Best score:", best_model.best_score_)
```

### Bayesian Optimization

```python
from src.pipelines.model_selection import BayesianOptimizer

optimizer = BayesianOptimizer()

# Define search space
search_space = {
    'n_estimators': (50, 500),
    'max_depth': (3, 20),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 10)
}

# Optimize hyperparameters
best_params = optimizer.optimize_regression(
    X_train, y_train,
    model_type='random_forest',
    search_space=search_space,
    n_calls=50,
    cv=5
)

print("Optimized parameters:", best_params)
```

## Evaluation Metrics

### Regression Metrics Available

```python
from src.evaluation.metrics import RegressionMetrics

metrics = RegressionMetrics()

# Calculate all metrics
results = metrics.calculate_all_metrics(y_true, y_pred)
print(results)

# Individual metrics
mse = metrics.mean_squared_error(y_true, y_pred)
mae = metrics.mean_absolute_error(y_true, y_pred)
r2 = metrics.r2_score(y_true, y_pred)
rmse = metrics.root_mean_squared_error(y_true, y_pred)
```

**Available metrics**:

- **MSE**: Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **RMSE**: Root Mean Squared Error (lower is better)
- **R²**: R-squared/Coefficient of determination (higher is better)
- **MAPE**: Mean Absolute Percentage Error
- **Explained Variance**: Explained variance score

## Best Practices

### Algorithm Selection Guide

1. **Start Simple**: Begin with Linear Regression for baseline
2. **Consider Data Size**:
   - Small data (< 1K): Linear, Ridge, Lasso
   - Medium data (1K-100K): Random Forest, SVR
   - Large data (> 100K): Gradient Boosting, Linear
3. **Feature Relationships**:
   - Linear: Linear Regression, Ridge, Lasso
   - Non-linear: Random Forest, Gradient Boosting, SVR
4. **Interpretability Needs**:
   - High: Linear, Lasso, Ridge
   - Medium: Random Forest
   - Low: Gradient Boosting, SVR

### Performance Optimization

```python
# For large datasets
model = models.get_random_forest_regression(
    n_estimators=100,  # Reduce for speed
    max_depth=10,      # Limit depth
    n_jobs=-1,         # Use all cores
    random_state=42    # Reproducibility
)

# For high accuracy
model = models.get_gradient_boosting_regression(
    n_estimators=300,
    learning_rate=0.05,  # Lower learning rate
    max_depth=6,
    subsample=0.8,       # Prevent overfitting
    early_stopping_rounds=10
)
```

### Common Issues and Solutions

**Overfitting**:

- Use cross-validation
- Reduce model complexity
- Add regularization (Ridge/Lasso)
- Increase training data

**Poor Performance**:

- Check data quality and preprocessing
- Try different algorithms
- Tune hyperparameters
- Feature engineering

**Slow Training**:

- Reduce dataset size for tuning
- Use simpler models for exploration
- Optimize hyperparameters systematically
- Use parallel processing (`n_jobs=-1`)

## See Also

- [Model Selection Guide](../tutorials/model_selection.md)
- [Data Preprocessing](../tutorials/data_preprocessing.md)
- [API Reference: Models](../api_reference/models.md)
- [Evaluation Metrics](../api_reference/evaluation.md)
