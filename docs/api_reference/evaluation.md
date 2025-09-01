# File: docs/api_reference/evaluation.md

# Location: docs/api_reference/evaluation.md

# Evaluation API Reference

Complete API documentation for model evaluation, metrics, statistical tests, and visualization modules.

## Model Evaluator

### `src.evaluation.metrics.ModelEvaluator`

Core class for comprehensive model evaluation across different ML tasks.

```python
from src.evaluation.metrics import ModelEvaluator

evaluator = ModelEvaluator()
```

#### Methods

##### `cross_validate_model()`

Perform cross-validation with comprehensive metrics.

```python
cv_results = evaluator.cross_validate_model(
    model,
    X, y,
    cv=5,
    scoring='accuracy',
    return_train_score=True,
    return_estimator=False,
    n_jobs=-1,
    verbose=0
)
```

**Parameters:**

- `model`: Sklearn-compatible model
- `X` (array-like): Feature matrix
- `y` (array-like): Target vector
- `cv` (int/object): Cross-validation strategy
- `scoring` (str/callable): Scoring metric
- `return_train_score` (bool): Include training scores
- `return_estimator` (bool): Return fitted estimators
- `n_jobs` (int): Number of parallel jobs
- `verbose` (int): Verbosity level

**Returns:**

- `cv_results` (dict): Cross-validation scores and metadata

##### `evaluate_classification_model()`

Comprehensive classification model evaluation.

```python
metrics = evaluator.evaluate_classification_model(
    model, X_test, y_test,
    class_names=None,
    average='weighted',
    include_plots=True,
    plot_dir=None
)
```

**Parameters:**

- `model`: Trained classification model
- `X_test` (array-like): Test features
- `y_test` (array-like): True test labels
- `class_names` (list): Class label names
- `average` (str): Averaging strategy for multiclass
- `include_plots` (bool): Generate evaluation plots
- `plot_dir` (str): Directory to save plots

**Returns:**

- `metrics` (dict): Comprehensive classification metrics

##### `evaluate_regression_model()`

Comprehensive regression model evaluation.

```python
metrics = evaluator.evaluate_regression_model(
    model, X_test, y_test,
    include_plots=True,
    plot_dir=None
)
```

**Returns:**

- `metrics` (dict): Comprehensive regression metrics

##### `evaluate_clustering_model()`

Evaluate clustering model performance.

```python
metrics = evaluator.evaluate_clustering_model(
    model, X, y_true=None,
    include_plots=True,
    plot_dir=None
)
```

**Parameters:**

- `model`: Fitted clustering model
- `X` (array-like): Feature matrix
- `y_true` (array-like, optional): True cluster labels
- `include_plots` (bool): Generate clustering plots
- `plot_dir` (str): Directory to save plots

**Returns:**

- `metrics` (dict): Clustering evaluation metrics

## Classification Metrics

### `src.evaluation.metrics.ClassificationMetrics`

Specialized classification metrics calculator.

```python
from src.evaluation.metrics import ClassificationMetrics

clf_metrics = ClassificationMetrics()
```

#### Methods

##### `accuracy_score()`

Calculate accuracy score.

```python
accuracy = clf_metrics.accuracy_score(y_true, y_pred, normalize=True)
```

##### `precision_recall_fscore()`

Calculate precision, recall, and F1-score.

```python
precision, recall, f1, support = clf_metrics.precision_recall_fscore(
    y_true, y_pred,
    labels=None,
    pos_label=1,
    average='weighted',
    warn_for=('precision', 'recall', 'f-score'),
    beta=1.0,
    sample_weight=None,
    zero_division='warn'
)
```

##### `roc_auc_score()`

Calculate ROC AUC score for binary and multiclass problems.

```python
# Binary classification
auc_binary = clf_metrics.roc_auc_score(y_true, y_scores)

# Multiclass classification
auc_multiclass = clf_metrics.roc_auc_score(
    y_true, y_scores,
    multi_class='ovr',  # or 'ovo'
    average='weighted'
)
```

##### `confusion_matrix()`

Generate confusion matrix with analysis.

```python
cm_results = clf_metrics.confusion_matrix(
    y_true, y_pred,
    labels=None,
    sample_weight=None,
    normalize=None,  # None, 'true', 'pred', 'all'
    include_analysis=True
)
```

**Returns:**

- `cm_results` (dict): Contains confusion matrix and derived metrics

##### `classification_report()`

Generate comprehensive classification report.

```python
report = clf_metrics.classification_report(
    y_true, y_pred,
    labels=None,
    target_names=None,
    sample_weight=None,
    digits=2,
    output_dict=True,
    zero_division='warn'
)
```

## Regression Metrics

### `src.evaluation.metrics.RegressionMetrics`

Specialized regression metrics calculator.

```python
from src.evaluation.metrics import RegressionMetrics

reg_metrics = RegressionMetrics()
```

#### Methods

##### `mean_squared_error()`

Calculate MSE with optional multioutput handling.

```python
mse = reg_metrics.mean_squared_error(
    y_true, y_pred,
    sample_weight=None,
    multioutput='uniform_average'
)
```

##### `mean_absolute_error()`

Calculate MAE.

```python
mae = reg_metrics.mean_absolute_error(
    y_true, y_pred,
    sample_weight=None,
    multioutput='uniform_average'
)
```

##### `r2_score()`

Calculate coefficient of determination (R²).

```python
r2 = reg_metrics.r2_score(
    y_true, y_pred,
    sample_weight=None,
    multioutput='uniform_average'
)
```

##### `explained_variance_score()`

Calculate explained variance score.

```python
evs = reg_metrics.explained_variance_score(
    y_true, y_pred,
    sample_weight=None,
    multioutput='uniform_average'
)
```

##### `mean_absolute_percentage_error()`

Calculate MAPE.

```python
mape = reg_metrics.mean_absolute_percentage_error(
    y_true, y_pred,
    sample_weight=None,
    multioutput='uniform_average'
)
```

##### `regression_metrics_summary()`

Calculate all regression metrics at once.

```python
all_metrics = reg_metrics.regression_metrics_summary(
    y_true, y_pred,
    sample_weight=None
)
```

**Returns:**

- `all_metrics` (dict): Complete set of regression metrics

## Clustering Metrics

### `src.evaluation.metrics.ClusteringMetrics`

Specialized clustering evaluation metrics.

```python
from src.evaluation.metrics import ClusteringMetrics

cluster_metrics = ClusteringMetrics()
```

#### Methods

##### `silhouette_score()`

Calculate silhouette coefficient.

```python
silhouette = cluster_metrics.silhouette_score(
    X, labels,
    metric='euclidean',
    sample_size=None,
    random_state=None
)
```

##### `adjusted_rand_score()`

Calculate adjusted rand index.

```python
ari = cluster_metrics.adjusted_rand_score(labels_true, labels_pred)
```

##### `normalized_mutual_info_score()`

Calculate normalized mutual information.

```python
nmi = cluster_metrics.normalized_mutual_info_score(
    labels_true, labels_pred,
    average_method='arithmetic'
)
```

##### `calinski_harabasz_score()`

Calculate Calinski-Harabasz index.

```python
ch_score = cluster_metrics.calinski_harabasz_score(X, labels)
```

##### `davies_bouldin_score()`

Calculate Davies-Bouldin index.

```python
db_score = cluster_metrics.davies_bouldin_score(X, labels)
```

##### `clustering_metrics_summary()`

Calculate comprehensive clustering metrics.

```python
all_metrics = cluster_metrics.clustering_metrics_summary(
    X, labels_pred,
    labels_true=None,
    include_internal=True,
    include_external=True
)
```

## Statistical Tests

### `src.evaluation.statistical_tests.StatisticalTester`

Perform statistical significance tests for model comparison.

```python
from src.evaluation.statistical_tests import StatisticalTester

tester = StatisticalTester()
```

#### Methods

##### `paired_ttest()`

Perform paired t-test for model comparison.

```python
t_stat, p_value, is_significant = tester.paired_ttest(
    scores_a, scores_b,
    alpha=0.05,
    alternative='two-sided'
)
```

##### `mcnemar_test()`

Perform McNemar's test for classifier comparison.

```python
chi2_stat, p_value, is_significant = tester.mcnemar_test(
    y_true, y_pred_a, y_pred_b,
    alpha=0.05,
    exact=True,
    correction=True
)
```

##### `wilcoxon_signed_rank()`

Non-parametric test for paired samples.

```python
w_stat, p_value, is_significant = tester.wilcoxon_signed_rank(
    scores_a, scores_b,
    alpha=0.05,
    alternative='two-sided',
    zero_method='wilcox'
)
```

##### `friedman_test()`

Test for differences across multiple algorithms.

```python
f_stat, p_value, is_significant = tester.friedman_test(
    *score_arrays,
    alpha=0.05
)
```

##### `nemenyi_posthoc()`

Post-hoc analysis after Friedman test.

```python
rankings, critical_difference = tester.nemenyi_posthoc(
    score_matrix,
    alpha=0.05
)
```

##### `bootstrap_confidence_interval()`

Calculate bootstrap confidence intervals.

```python
ci_lower, ci_upper = tester.bootstrap_confidence_interval(
    scores,
    confidence_level=0.95,
    n_bootstrap=1000,
    statistic=np.mean,
    random_state=None
)
```

## Visualization

### `src.evaluation.visualization.EvaluationVisualizer`

Create evaluation plots and visualizations.

```python
from src.evaluation.visualization import EvaluationVisualizer

visualizer = EvaluationVisualizer(figsize=(10, 8), style='seaborn')
```

#### Methods

##### `plot_confusion_matrix()`

Plot confusion matrix with customization.

```python
fig, ax = visualizer.plot_confusion_matrix(
    y_true, y_pred,
    class_names=None,
    normalize=False,
    cmap='Blues',
    title='Confusion Matrix'
)
```

##### `plot_roc_curve()`

Plot ROC curve for binary/multiclass classification.

```python
fig, ax = visualizer.plot_roc_curve(
    y_true, y_scores,
    pos_label=None,
    sample_weight=None,
    drop_intermediate=True,
    response_method='auto',
    name=None,
    ax=None
)
```

##### `plot_precision_recall_curve()`

Plot precision-recall curve.

```python
fig, ax = visualizer.plot_precision_recall_curve(
    y_true, y_scores,
    pos_label=None,
    sample_weight=None,
    response_method='auto',
    name=None,
    ax=None
)
```

##### `plot_learning_curve()`

Plot learning curves to analyze bias/variance.

```python
fig, ax = visualizer.plot_learning_curve(
    model, X, y,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy',
    shuffle=False,
    random_state=None
)
```

##### `plot_validation_curve()`

Plot validation curves for hyperparameter tuning.

```python
fig, ax = visualizer.plot_validation_curve(
    model, X, y,
    param_name,
    param_range,
    cv=None,
    scoring='accuracy',
    n_jobs=None,
    ax=None
)
```

##### `plot_feature_importance()`

Plot feature importance from tree-based models.

```python
fig, ax = visualizer.plot_feature_importance(
    model,
    feature_names=None,
    max_features=20,
    title='Feature Importance'
)
```

##### `plot_regression_residuals()`

Plot residuals for regression analysis.

```python
fig, ax = visualizer.plot_regression_residuals(
    y_true, y_pred,
    title='Residual Plot'
)
```

##### `plot_clustering_results()`

Visualize clustering results.

```python
fig, ax = visualizer.plot_clustering_results(
    X, labels,
    centers=None,
    title='Clustering Results',
    max_dimensions=2
)
```

## Utilities

### `src.evaluation.utils.EvaluationUtils`

Utility functions for evaluation tasks.

```python
from src.evaluation.utils import EvaluationUtils

utils = EvaluationUtils()
```

#### Methods

##### `stratified_sample()`

Create stratified sample for evaluation.

```python
X_sample, y_sample, indices = utils.stratified_sample(
    X, y,
    sample_size=1000,
    random_state=42
)
```

##### `bootstrap_sample()`

Generate bootstrap samples.

```python
X_boot, y_boot, indices = utils.bootstrap_sample(
    X, y,
    n_samples=None,
    random_state=42
)
```

##### `cross_validation_iterator()`

Create custom cross-validation iterators.

```python
cv_iterator = utils.cross_validation_iterator(
    X, y,
    cv_type='stratified_kfold',
    n_splits=5,
    test_size=0.2,
    random_state=42
)

for train_idx, test_idx in cv_iterator:
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

##### `calculate_model_complexity()`

Estimate model complexity metrics.

```python
complexity_metrics = utils.calculate_model_complexity(
    model,
    include_parameters=True,
    include_memory=True,
    include_inference_time=True
)
```

##### `generate_evaluation_report()`

Generate comprehensive evaluation report.

```python
report = utils.generate_evaluation_report(
    model,
    X_test, y_test,
    task_type='classification',
    include_plots=True,
    output_format='html',
    save_path='evaluation_report.html'
)
```

## Advanced Evaluation

### Model Calibration

```python
from src.evaluation.calibration import CalibrationEvaluator

cal_evaluator = CalibrationEvaluator()

# Evaluate calibration
calibration_metrics = cal_evaluator.evaluate_calibration(
    y_true, y_proba,
    n_bins=10,
    strategy='uniform'
)

# Plot calibration curve
cal_evaluator.plot_calibration_curve(
    y_true, y_proba,
    n_bins=10,
    name='Model'
)

# Calibrate probabilities
calibrated_proba = cal_evaluator.calibrate_probabilities(
    y_proba,
    y_true,
    method='isotonic'  # or 'sigmoid'
)
```

### Fairness Evaluation

```python
from src.evaluation.fairness import FairnessEvaluator

fairness_eval = FairnessEvaluator()

# Evaluate demographic parity
fairness_metrics = fairness_eval.evaluate_demographic_parity(
    y_true, y_pred,
    sensitive_features,
    privileged_groups=[1],
    positive_label=1
)

# Evaluate equalized odds
eq_odds_metrics = fairness_eval.evaluate_equalized_odds(
    y_true, y_pred,
    sensitive_features,
    privileged_groups=[1],
    positive_label=1
)
```

### Interpretability Analysis

```python
from src.evaluation.interpretability import InterpretabilityAnalyzer

interp_analyzer = InterpretabilityAnalyzer()

# SHAP values analysis
shap_values = interp_analyzer.calculate_shap_values(
    model, X_test,
    feature_names=feature_names
)

# Permutation importance
perm_importance = interp_analyzer.permutation_importance(
    model, X_test, y_test,
    scoring='accuracy',
    n_repeats=10,
    random_state=42
)

# Partial dependence plots
interp_analyzer.plot_partial_dependence(
    model, X_test,
    features=[0, 1, (0, 1)],
    feature_names=feature_names
)
```

## Examples

### Complete Model Evaluation

```python
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.visualization import EvaluationVisualizer
from src.evaluation.statistical_tests import StatisticalTester

# Initialize evaluators
evaluator = ModelEvaluator()
visualizer = EvaluationVisualizer()
tester = StatisticalTester()

# Evaluate classification model
metrics = evaluator.evaluate_classification_model(
    model, X_test, y_test,
    class_names=['Class A', 'Class B', 'Class C'],
    include_plots=True
)

print("Classification Metrics:")
for metric_name, value in metrics.items():
    if isinstance(value, (int, float)):
        print(f"  {metric_name}: {value:.4f}")

# Cross-validation comparison
cv_scores_rf = evaluator.cross_validate_model(
    RandomForestClassifier(), X_train, y_train, cv=5
)['test_score']

cv_scores_gb = evaluator.cross_validate_model(
    GradientBoostingClassifier(), X_train, y_train, cv=5
)['test_score']

# Statistical comparison
t_stat, p_value, significant = tester.paired_ttest(
    cv_scores_rf, cv_scores_gb
)

print(f"\nModel Comparison:")
print(f"Random Forest: {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")
print(f"Gradient Boosting: {cv_scores_gb.mean():.4f} ± {cv_scores_gb.std():.4f}")
print(f"Statistical significance: p-value = {p_value:.4f}")
```

## Configuration

### Default Settings

```python
# Evaluation defaults
DEFAULT_EVALUATION_CONFIG = {
    'cross_validation': {
        'cv': 5,
        'scoring': 'accuracy',
        'n_jobs': -1
    },
    'visualization': {
        'figsize': (10, 8),
        'style': 'seaborn',
        'dpi': 150
    },
    'statistical_tests': {
        'alpha': 0.05,
        'bootstrap_samples': 1000
    }
}
```

## See Also

- [Model Training](models.md)
- [Data Processing](data.md)
- [Pipeline Factory](pipelines.md)
- [Statistical Analysis Guide](../tutorials/statistical_analysis.md)
