# File: docs/api_reference/pipelines.md

# Location: docs/api_reference/pipelines.md

# Pipeline API Reference

Complete API documentation for ML pipeline creation, custom transformers, and model selection modules.

## Pipeline Factory

### `src.pipelines.pipeline_factory.PipelineFactory`

Factory class for creating complete ML pipelines with preprocessing and model training.

```python
from src.pipelines.pipeline_factory import PipelineFactory

factory = PipelineFactory()
```

#### Methods

##### `create_classification_pipeline()`

Create complete classification pipeline.

```python
pipeline = factory.create_classification_pipeline(
    model_type='random_forest',
    preprocessing_steps=['standard_scaler', 'feature_selection'],
    model_params=None,
    preprocessing_params=None,
    feature_selection_params=None,
    pipeline_name=None
)
```

**Parameters:**

- `model_type` (str): Type of classifier ('random_forest', 'gradient_boosting', 'svm', 'logistic_regression', 'naive_bayes')
- `preprocessing_steps` (list): List of preprocessing steps
- `model_params` (dict): Parameters for the model
- `preprocessing_params` (dict): Parameters for preprocessing steps
- `feature_selection_params` (dict): Feature selection parameters
- `pipeline_name` (str): Name for the pipeline

**Returns:**

- `pipeline` (sklearn.Pipeline): Complete ML pipeline

##### `create_regression_pipeline()`

Create complete regression pipeline.

```python
pipeline = factory.create_regression_pipeline(
    model_type='random_forest',
    preprocessing_steps=['standard_scaler'],
    model_params=None,
    preprocessing_params=None,
    pipeline_name=None
)
```

**Available model types:**

- `linear_regression`: Linear regression
- `ridge`: Ridge regression
- `lasso`: Lasso regression
- `random_forest`: Random forest regressor
- `gradient_boosting`: Gradient boosting regressor
- `svr`: Support vector regression

##### `create_clustering_pipeline()`

Create clustering pipeline with preprocessing.

```python
pipeline = factory.create_clustering_pipeline(
    model_type='kmeans',
    preprocessing_steps=['standard_scaler', 'pca'],
    model_params=None,
    preprocessing_params=None,
    pipeline_name=None
)
```

**Available model types:**

- `kmeans`: K-Means clustering
- `dbscan`: DBSCAN clustering
- `hierarchical`: Agglomerative clustering
- `gaussian_mixture`: Gaussian Mixture Model
- `spectral`: Spectral clustering

##### `create_ensemble_pipeline()`

Create ensemble learning pipeline.

```python
pipeline = factory.create_ensemble_pipeline(
    ensemble_type='voting',
    base_models=['random_forest', 'gradient_boosting', 'svm'],
    preprocessing_steps=['standard_scaler'],
    ensemble_params=None,
    base_model_params=None,
    pipeline_name=None
)
```

**Ensemble types:**

- `voting`: Voting classifier/regressor
- `bagging`: Bagging ensemble
- `boosting`: AdaBoost ensemble
- `stacking`: Stacking ensemble

##### `create_custom_pipeline()`

Create pipeline with custom components.

```python
pipeline = factory.create_custom_pipeline(
    steps=[
        ('preprocessor', custom_preprocessor),
        ('feature_selector', custom_selector),
        ('model', custom_model)
    ],
    pipeline_name='custom_pipeline'
)
```

#### Available Preprocessing Steps

**Scaling:**

- `standard_scaler`: StandardScaler
- `min_max_scaler`: MinMaxScaler
- `robust_scaler`: RobustScaler
- `quantile_transformer`: QuantileTransformer
- `power_transformer`: PowerTransformer

**Feature Selection:**

- `feature_selection`: SelectKBest with configurable score function
- `variance_threshold`: VarianceThreshold
- `recursive_elimination`: RFE
- `lasso_selection`: SelectFromModel with Lasso
- `tree_selection`: SelectFromModel with tree-based feature importance

**Dimensionality Reduction:**

- `pca`: Principal Component Analysis
- `kernel_pca`: Kernel PCA
- `ica`: Independent Component Analysis
- `lda`: Linear Discriminant Analysis
- `tsne`: t-SNE (for visualization)

**Data Processing:**

- `polynomial_features`: Polynomial feature generation
- `interaction_features`: Feature interactions
- `missing_indicator`: Missing value indicators

## Custom Transformers

### `src.pipelines.custom_transformers.DataFrameSelector`

Select specific columns from pandas DataFrame.

```python
from src.pipelines.custom_transformers import DataFrameSelector

selector = DataFrameSelector(attribute_names=['feature1', 'feature2'])
X_selected = selector.fit_transform(df)
```

### `src.pipelines.custom_transformers.CategoricalEncoder`

Enhanced categorical encoding with multiple strategies.

```python
from src.pipelines.custom_transformers import CategoricalEncoder

encoder = CategoricalEncoder(
    encoding_type='onehot',  # 'onehot', 'ordinal', 'target', 'binary'
    handle_unknown='ignore',
    drop_first=True
)
X_encoded = encoder.fit_transform(X_categorical, y)
```

**Parameters:**

- `encoding_type` (str): Type of encoding
- `handle_unknown` (str): How to handle unknown categories
- `drop_first` (bool): Drop first category to avoid multicollinearity
- `min_frequency` (int): Minimum frequency for category inclusion

### `src.pipelines.custom_transformers.OutlierRemover`

Remove outliers using various detection methods.

```python
from src.pipelines.custom_transformers import OutlierRemover

outlier_remover = OutlierRemover(
    method='isolation_forest',  # 'isolation_forest', 'local_outlier_factor', 'one_class_svm'
    contamination=0.1,
    random_state=42
)
X_clean = outlier_remover.fit_transform(X)
```

### `src.pipelines.custom_transformers.FeatureGenerator`

Generate new features through various transformations.

```python
from src.pipelines.custom_transformers import FeatureGenerator

feature_gen = FeatureGenerator(
    operations=['log', 'sqrt', 'square', 'interactions'],
    interaction_degree=2,
    include_bias=False
)
X_enhanced = feature_gen.fit_transform(X)
```

**Available operations:**

- `log`: Logarithmic transformation
- `sqrt`: Square root transformation
- `square`: Square transformation
- `reciprocal`: Reciprocal transformation
- `interactions`: Polynomial interactions
- `binning`: Equal-width binning
- `clustering_features`: Cluster-based features

### `src.pipelines.custom_transformers.TimeSeriesTransformer`

Transform time series data for ML algorithms.

```python
from src.pipelines.custom_transformers import TimeSeriesTransformer

ts_transformer = TimeSeriesTransformer(
    lag_features=[1, 2, 3, 7],
    rolling_features=['mean', 'std'],
    rolling_windows=[7, 30],
    seasonal_features=True,
    trend_features=True
)
X_ts = ts_transformer.fit_transform(time_series_data)
```

### `src.pipelines.custom_transformers.TextPreprocessor`

Preprocess text data for ML pipelines.

```python
from src.pipelines.custom_transformers import TextPreprocessor

text_prep = TextPreprocessor(
    lowercase=True,
    remove_punctuation=True,
    remove_stopwords=True,
    stemming=True,
    min_word_length=2,
    max_features=5000
)
X_text = text_prep.fit_transform(text_data)
```

## Model Selection

### `src.pipelines.model_selection.ModelSelector`

Automated model selection and hyperparameter tuning.

```python
from src.pipelines.model_selection import ModelSelector

selector = ModelSelector(
    task_type='classification',  # 'classification', 'regression', 'clustering'
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    random_state=42
)
```

#### Methods

##### `grid_search_cv()`

Perform grid search cross-validation.

```python
best_model = selector.grid_search_cv(
    X_train, y_train,
    model_type='random_forest',
    param_grid={
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    },
    cv=5,
    scoring='accuracy',
    refit=True,
    verbose=1
)
```

##### `random_search_cv()`

Perform randomized search cross-validation.

```python
best_model = selector.random_search_cv(
    X_train, y_train,
    model_type='gradient_boosting',
    param_distributions={
        'n_estimators': randint(50, 500),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10)
    },
    n_iter=50,
    cv=5,
    scoring='f1_weighted'
)
```

##### `bayesian_optimization()`

Use Bayesian optimization for hyperparameter tuning.

```python
best_model = selector.bayesian_optimization(
    X_train, y_train,
    model_type='svm',
    search_space={
        'C': (0.1, 100.0, 'log-uniform'),
        'gamma': (0.001, 1.0, 'log-uniform'),
        'kernel': ['rbf', 'poly', 'sigmoid']
    },
    n_calls=100,
    acq_func='EI',  # Expected Improvement
    cv=5
)
```

##### `compare_models()`

Compare multiple models with cross-validation.

```python
comparison_results = selector.compare_models(
    X_train, y_train,
    models=['random_forest', 'gradient_boosting', 'svm', 'logistic_regression'],
    cv=5,
    scoring=['accuracy', 'f1_weighted', 'roc_auc'],
    include_feature_importance=True,
    plot_results=True
)
```

**Returns:**

- `comparison_results` (dict): Detailed comparison metrics and plots

##### `auto_select_best_model()`

Automatically select the best model from multiple algorithms.

```python
best_pipeline = selector.auto_select_best_model(
    X_train, y_train,
    candidate_models=None,  # Use default set
    preprocessing_options=['standard_scaler', 'robust_scaler'],
    feature_selection_options=[None, 'univariate', 'recursive'],
    max_evaluation_time=3600,  # 1 hour budget
    early_stopping=True
)
```

### `src.pipelines.model_selection.HyperparameterOptimizer`

Advanced hyperparameter optimization strategies.

```python
from src.pipelines.model_selection import HyperparameterOptimizer

optimizer = HyperparameterOptimizer()
```

#### Methods

##### `successive_halving()`

Use successive halving for efficient hyperparameter search.

```python
best_params = optimizer.successive_halving(
    model, param_grid,
    X_train, y_train,
    factor=3,
    resource='n_samples',
    max_resources='auto',
    aggressive_elimination=False,
    cv=3
)
```

##### `hyperband()`

Implement Hyperband algorithm for hyperparameter optimization.

```python
best_params = optimizer.hyperband(
    model, param_distributions,
    X_train, y_train,
    max_budget=100,
    eta=3,
    cv=3,
    random_state=42
)
```

##### `optuna_optimization()`

Use Optuna for advanced hyperparameter optimization.

```python
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
    }
    model = GradientBoostingClassifier(**params)
    return cross_val_score(model, X_train, y_train, cv=5).mean()

best_params = optimizer.optuna_optimization(
    objective,
    n_trials=100,
    timeout=3600,
    sampler='TPE',  # Tree-structured Parzen Estimator
    pruner='MedianPruner'
)
```

## Pipeline Utilities

### `src.pipelines.utils.PipelineUtils`

Utility functions for pipeline operations.

```python
from src.pipelines.utils import PipelineUtils

utils = PipelineUtils()
```

#### Methods

##### `save_pipeline()`

Save trained pipeline to disk.

```python
utils.save_pipeline(
    pipeline,
    filepath='trained_pipeline.pkl',
    include_metadata=True
)
```

##### `load_pipeline()`

Load saved pipeline from disk.

```python
loaded_pipeline = utils.load_pipeline('trained_pipeline.pkl')
```

##### `inspect_pipeline()`

Inspect pipeline components and parameters.

```python
inspection_report = utils.inspect_pipeline(
    pipeline,
    include_feature_names=True,
    include_parameters=True,
    include_performance=True
)
```

##### `validate_pipeline()`

Validate pipeline structure and compatibility.

```python
validation_result = utils.validate_pipeline(
    pipeline,
    X_sample, y_sample,
    check_fit=True,
    check_transform=True,
    check_predict=True
)
```

##### `optimize_pipeline_memory()`

Optimize pipeline for memory usage.

```python
optimized_pipeline = utils.optimize_pipeline_memory(
    pipeline,
    memory_limit_gb=4.0,
    enable_caching=True,
    temp_folder='/tmp/sklearn_cache'
)
```

## Advanced Pipeline Patterns

### Multi-Target Pipeline

```python
from src.pipelines.multi_target import MultiTargetPipeline

multi_pipeline = MultiTargetPipeline(
    estimator=RandomForestRegressor(),
    n_jobs=-1
)

# For multiple continuous targets
multi_pipeline.fit(X_train, y_multi_train)
predictions = multi_pipeline.predict(X_test)
```

### Streaming Pipeline

```python
from src.pipelines.streaming import StreamingPipeline

streaming_pipeline = StreamingPipeline(
    base_pipeline=pipeline,
    batch_size=1000,
    update_frequency='daily',
    drift_detection=True
)

# Process streaming data
for batch_X, batch_y in data_stream:
    predictions = streaming_pipeline.predict(batch_X)
    streaming_pipeline.partial_fit(batch_X, batch_y)
```

### Federated Pipeline

```python
from src.pipelines.federated import FederatedPipeline

fed_pipeline = FederatedPipeline(
    base_pipeline=pipeline,
    aggregation_method='federated_averaging',
    privacy_budget=1.0,
    secure_aggregation=True
)

# Train across distributed clients
for client_data in client_datasets:
    fed_pipeline.fit_client(client_data['X'], client_data['y'])

final_model = fed_pipeline.aggregate_models()
```

## Examples

### Complete Classification Pipeline

```python
from src.pipelines.pipeline_factory import PipelineFactory
from src.pipelines.model_selection import ModelSelector
from src.evaluation.metrics import ModelEvaluator

# Create pipeline factory
factory = PipelineFactory()

# Create classification pipeline
pipeline = factory.create_classification_pipeline(
    model_type='random_forest',
    preprocessing_steps=[
        'standard_scaler',
        'feature_selection',
        'polynomial_features'
    ],
    model_params={
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42
    },
    preprocessing_params={
        'feature_selection__k': 15,
        'polynomial_features__degree': 2,
        'polynomial_features__include_bias': False
    }
)

# Train pipeline
pipeline.fit(X_train, y_train)

# Evaluate pipeline
evaluator = ModelEvaluator()
metrics = evaluator.evaluate_classification_model(
    pipeline, X_test, y_test,
    include_plots=True
)

print(f"Pipeline Accuracy: {metrics['accuracy']:.4f}")
print(f"Pipeline F1-Score: {metrics['f1_weighted']:.4f}")
```

### Automated Model Selection

```python
from src.pipelines.model_selection import ModelSelector

# Initialize selector
selector = ModelSelector(task_type='classification')

# Compare multiple models
comparison = selector.compare_models(
    X_train, y_train,
    models=['random_forest', 'gradient_boosting', 'svm', 'logistic_regression'],
    cv=5,
    scoring=['accuracy', 'f1_weighted', 'precision_weighted'],
    plot_results=True
)

# Select best model automatically
best_pipeline = selector.auto_select_best_model(
    X_train, y_train,
    max_evaluation_time=1800,  # 30 minutes
    early_stopping=True
)

print(f"Best model: {best_pipeline.named_steps['model'].__class__.__name__}")
print(f"Best CV score: {best_pipeline.score(X_test, y_test):.4f}")
```

### Custom Transformer Pipeline

```python
from src.pipelines.custom_transformers import *
from sklearn.pipeline import Pipeline

# Create custom pipeline with multiple transformers
custom_pipeline = Pipeline([
    ('outlier_removal', OutlierRemover(method='isolation_forest')),
    ('feature_generation', FeatureGenerator(operations=['log', 'sqrt', 'interactions'])),
    ('categorical_encoding', CategoricalEncoder(encoding_type='target')),
    ('scaling', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif, k=20)),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# Train and evaluate
custom_pipeline.fit(X_train, y_train)
custom_score = custom_pipeline.score(X_test, y_test)
print(f"Custom pipeline accuracy: {custom_score:.4f}")
```

## Configuration

### Default Pipeline Settings

```python
# Pipeline factory defaults
DEFAULT_PIPELINE_CONFIG = {
    'preprocessing': {
        'scaling': 'standard_scaler',
        'feature_selection': None,
        'missing_value_strategy': 'impute'
    },
    'model_selection': {
        'cv_folds': 5,
        'scoring': 'accuracy',
        'n_jobs': -1,
        'refit': True
    },
    'optimization': {
        'max_evals': 100,
        'timeout': 3600,
        'early_stopping': True
    }
}
```

## See Also

- [Model Training](models.md)
- [Data Processing](data.md)
- [Evaluation Metrics](evaluation.md)
- [Model Selection Tutorial](../tutorials/model_selection.md)
