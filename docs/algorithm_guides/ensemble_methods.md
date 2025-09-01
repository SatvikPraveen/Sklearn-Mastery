# File: docs/algorithm_guides/ensemble_methods.md

# Location: docs/algorithm_guides/ensemble_methods.md

# Ensemble Methods

Comprehensive guide to ensemble learning algorithms that combine multiple models for superior performance.

## Overview

Ensemble methods combine predictions from multiple models to create more robust and accurate predictions than individual models. Our framework provides optimized implementations of voting, bagging, boosting, and stacking techniques.

## Core Ensemble Strategies

### Voting Classifiers

**Best for**: Combining diverse models, reducing variance, stable predictions

```python
from src.models.ensemble.methods import EnsembleMethods

ensemble = EnsembleMethods()
voting_clf = ensemble.get_voting_classifier()

# Hard voting (majority vote)
hard_voting = ensemble.get_voting_classifier(
    base_estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('svm', SVC(kernel='rbf', probability=True)),
        ('nb', GaussianNB())
    ],
    voting='hard'
)

# Soft voting (probability averaging)
soft_voting = ensemble.get_voting_classifier(
    base_estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('gb', GradientBoostingClassifier(n_estimators=100)),
        ('lr', LogisticRegression())
    ],
    voting='soft'
)
```

**When to use**:

- Have diverse, well-performing base models
- Want to reduce overfitting
- Base models have similar performance
- Need interpretable ensemble decisions

**Voting types**:

- **Hard voting**: Majority class vote (faster)
- **Soft voting**: Average probabilities (usually better)

### Bagging (Bootstrap Aggregating)

**Best for**: Reducing variance, parallel training, high-variance models

```python
# Random Forest (specialized bagging)
rf_ensemble = ensemble.get_random_forest()

# General bagging with any base estimator
bagging_clf = ensemble.get_bagging_classifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    n_jobs=-1
)

# Extra Trees (Extremely Randomized Trees)
extra_trees = ensemble.get_extra_trees_classifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    n_jobs=-1
)
```

**Key benefits**:

- Reduces overfitting of high-variance models
- Can be parallelized efficiently
- Out-of-bag error estimation
- Works well with decision trees

**Parameters**:

- `n_estimators`: Number of base models (50-500)
- `max_samples`: Fraction of samples per model (0.5-1.0)
- `max_features`: Fraction of features per model (0.5-1.0)
- `bootstrap`: Whether to use bootstrap sampling

### Boosting Methods

**Best for**: Reducing bias, sequential improvement, weak learners

#### AdaBoost

```python
ada_boost = ensemble.get_ada_boost_classifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=1.0,
    algorithm='SAMME.R'
)
```

#### Gradient Boosting

```python
gradient_boost = ensemble.get_gradient_boosting_classifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)
```

#### XGBoost Integration

```python
xgb_ensemble = ensemble.get_xgboost_classifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0
)
```

**Boosting characteristics**:

- Sequential training (cannot parallelize base models)
- Focuses on previously misclassified examples
- Can overfit with too many iterations
- Generally higher accuracy than bagging

### Stacking (Stacked Generalization)

**Best for**: Maximum performance, heterogeneous models, complex patterns

```python
# Two-level stacking
stacking_clf = ensemble.get_stacking_classifier(
    level_0_estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('gb', GradientBoostingClassifier(n_estimators=100)),
        ('svm', SVC(kernel='rbf', probability=True)),
        ('nb', GaussianNB())
    ],
    level_1_estimator=LogisticRegression(),
    cv=5,
    stack_method='predict_proba',
    n_jobs=-1
)

# Multi-level stacking
multi_level = ensemble.get_multi_level_stacking(
    level_0_estimators=[
        ('rf', RandomForestClassifier()),
        ('et', ExtraTreesClassifier()),
        ('gb', GradientBoostingClassifier())
    ],
    level_1_estimators=[
        ('lr', LogisticRegression()),
        ('svm', SVC(probability=True))
    ],
    level_2_estimator=XGBClassifier(),
    cv_folds=[5, 3],  # Different CV for each level
    stack_methods=['predict_proba', 'predict_proba']
)
```

**Stacking levels**:

- **Level 0**: Base models trained on original data
- **Level 1**: Meta-model trained on base model predictions
- **Level 2+**: Additional meta-levels for complex patterns

## Advanced Ensemble Techniques

### Dynamic Ensemble Selection

**Best for**: Heterogeneous data regions, adaptive selection

```python
# Dynamic classifier selection
des = ensemble.get_dynamic_ensemble_selection(
    pool_classifiers=[
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        SVC(probability=True),
        KNeighborsClassifier()
    ],
    selection_method='OLA',  # Overall Local Accuracy
    k_neighbors=7,
    DFP=True  # Dynamic Frienemy Pruning
)

# Dynamic ensemble selection
des_ensemble = ensemble.get_dynamic_ensemble_selection(
    pool_classifiers=base_classifiers,
    selection_method='LCA',  # Local Class Accuracy
    combination_method='weighted_average'
)
```

### Bayesian Model Averaging

**Best for**: Uncertainty quantification, probabilistic predictions

```python
# Bayesian ensemble
bayesian_ensemble = ensemble.get_bayesian_ensemble(
    base_models=[
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier()),
        ('lr', LogisticRegression())
    ],
    prior_weights='uniform',  # or 'performance_based'
    posterior_sampling=True,
    n_samples=1000
)

# Get predictions with uncertainty
predictions, uncertainties = bayesian_ensemble.predict_with_uncertainty(X_test)
```

### Multi-Armed Bandit Ensembles

**Best for**: Online learning, adaptive model selection

```python
# Thompson sampling ensemble
thompson_ensemble = ensemble.get_thompson_sampling_ensemble(
    base_models=base_models,
    alpha_prior=1.0,
    beta_prior=1.0,
    exploration_rate=0.1
)

# UCB ensemble
ucb_ensemble = ensemble.get_ucb_ensemble(
    base_models=base_models,
    confidence_bound=1.96,
    exploration_factor=0.5
)
```

## Regression Ensembles

### Voting Regressor

```python
voting_reg = ensemble.get_voting_regressor(
    estimators=[
        ('rf', RandomForestRegressor(n_estimators=100)),
        ('gb', GradientBoostingRegressor(n_estimators=100)),
        ('svr', SVR(kernel='rbf'))
    ],
    weights=[2, 2, 1]  # Weight more accurate models higher
)
```

### Stacking Regressor

```python
stacking_reg = ensemble.get_stacking_regressor(
    level_0_estimators=[
        ('rf', RandomForestRegressor()),
        ('gb', GradientBoostingRegressor()),
        ('lr', LinearRegression()),
        ('svr', SVR())
    ],
    level_1_estimator=Ridge(alpha=1.0),
    cv=5
)
```

## Usage Examples

### Complete Ensemble Pipeline

```python
from src.data.generators import DataGenerator
from src.models.ensemble.methods import EnsembleMethods
from src.evaluation.metrics import ModelEvaluator
from src.pipelines.model_selection import EnsembleSelector

# Generate data
generator = DataGenerator()
X, y = generator.generate_classification_data(
    n_samples=5000,
    n_features=20,
    n_classes=3,
    n_informative=15,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize ensemble methods
ensemble = EnsembleMethods()
evaluator = ModelEvaluator()

# Create base models with diverse characteristics
base_models = {
    'rf': RandomForestClassifier(n_estimators=100, random_state=42),
    'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'svm': SVC(kernel='rbf', probability=True, random_state=42),
    'nb': GaussianNB(),
    'lr': LogisticRegression(random_state=42)
}

# Test different ensemble strategies
ensemble_strategies = {
    'Voting (Hard)': ensemble.get_voting_classifier(
        list(base_models.items()), voting='hard'
    ),
    'Voting (Soft)': ensemble.get_voting_classifier(
        list(base_models.items()), voting='soft'
    ),
    'Bagging': ensemble.get_bagging_classifier(
        DecisionTreeClassifier(), n_estimators=100
    ),
    'AdaBoost': ensemble.get_ada_boost_classifier(n_estimators=100),
    'Gradient Boosting': ensemble.get_gradient_boosting_classifier(
        n_estimators=100
    ),
    'Stacking': ensemble.get_stacking_classifier(
        list(base_models.items()),
        final_estimator=LogisticRegression()
    )
}

# Evaluate all ensemble methods
results = {}
for name, model in ensemble_strategies.items():
    # Cross-validation
    cv_scores = evaluator.cross_validate_model(
        model, X_train, y_train, cv=5
    )

    # Test set performance
    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)
    test_probs = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    # Detailed metrics
    metrics = evaluator.calculate_classification_metrics(
        y_test, test_predictions, test_probs
    )

    results[name] = {
        'cv_mean': cv_scores['test_score'].mean(),
        'cv_std': cv_scores['test_score'].std(),
        'test_accuracy': metrics['accuracy'],
        'test_f1': metrics['f1_weighted'],
        'test_auc': metrics['roc_auc_ovr'] if test_probs is not None else None
    }

# Display results
print("Ensemble Method Comparison:")
print("-" * 80)
for name, metrics in results.items():
    print(f"{name:20s} | "
          f"CV: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f} | "
          f"Test Acc: {metrics['test_accuracy']:.3f} | "
          f"F1: {metrics['test_f1']:.3f}")
```

### Hyperparameter Optimization for Ensembles

```python
from src.pipelines.model_selection import EnsembleOptimizer

optimizer = EnsembleOptimizer()

# Optimize stacking ensemble
stacking_space = {
    'level_0_rf__n_estimators': (50, 200),
    'level_0_rf__max_depth': (5, 20),
    'level_0_gb__n_estimators': (50, 200),
    'level_0_gb__learning_rate': (0.01, 0.3),
    'final_estimator__C': (0.1, 10.0)
}

best_stacking = optimizer.optimize_stacking(
    X_train, y_train,
    base_estimators=[
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier())
    ],
    final_estimator=LogisticRegression(),
    param_space=stacking_space,
    n_calls=50,
    cv=5
)

print("Best stacking parameters:", best_stacking.best_params_)
print("Best CV score:", best_stacking.best_score_)
```

### Ensemble Model Selection

```python
# Automatic ensemble selection
selector = EnsembleSelector()

# Find best ensemble strategy
best_ensemble = selector.select_best_ensemble(
    X_train, y_train,
    ensemble_types=['voting', 'bagging', 'boosting', 'stacking'],
    base_model_types=['tree', 'linear', 'svm', 'nb'],
    selection_metric='f1_weighted',
    cv=5,
    n_jobs=-1
)

print("Best ensemble type:", best_ensemble['type'])
print("Best configuration:", best_ensemble['config'])
print("CV Performance:", best_ensemble['cv_score'])

# Get the trained model
final_model = best_ensemble['model']
final_predictions = final_model.predict(X_test)
```

## Performance Analysis

### Diversity Analysis

```python
from src.evaluation.ensemble import DiversityAnalyzer

analyzer = DiversityAnalyzer()

# Measure base model diversity
diversity_metrics = analyzer.calculate_diversity(
    base_models, X_test, y_test
)

print("Diversity Metrics:")
print(f"Q-Statistics: {diversity_metrics['q_statistics']:.3f}")
print(f"Correlation Coefficient: {diversity_metrics['correlation']:.3f}")
print(f"Disagreement: {diversity_metrics['disagreement']:.3f}")
print(f"Double Fault: {diversity_metrics['double_fault']:.3f}")

# Visualize diversity
analyzer.plot_diversity_matrix(base_models, X_test, y_test)
```

### Error Analysis

```python
# Analyze ensemble errors
error_analyzer = ensemble.get_error_analyzer()

# Individual model errors
individual_errors = error_analyzer.analyze_individual_errors(
    base_models, X_test, y_test
)

# Ensemble error breakdown
ensemble_errors = error_analyzer.analyze_ensemble_errors(
    ensemble_model, X_test, y_test,
    include_base_predictions=True
)

# Error correlation analysis
error_correlations = error_analyzer.compute_error_correlations(
    base_models, X_test, y_test
)

print("Error Analysis Results:")
for model_name, error_rate in individual_errors.items():
    print(f"{model_name}: {error_rate:.3f}")

print(f"Ensemble error: {ensemble_errors['ensemble_error']:.3f}")
print(f"Error reduction: {ensemble_errors['error_reduction']:.3f}")
```

## Best Practices

### Model Selection for Ensembles

**Diversity principles**:

1. **Algorithm diversity**: Different learning algorithms
2. **Parameter diversity**: Same algorithm, different parameters
3. **Data diversity**: Different training subsets
4. **Feature diversity**: Different feature subsets

**Base model selection**:

```python
# Good ensemble: diverse, complementary models
good_ensemble = [
    RandomForestClassifier(),      # Tree-based, bagging
    GradientBoostingClassifier(),  # Tree-based, boosting
    SVC(kernel='rbf'),            # Kernel method
    LogisticRegression(),         # Linear method
    GaussianNB()                  # Probabilistic method
]

# Poor ensemble: similar models
poor_ensemble = [
    RandomForestClassifier(n_estimators=100),
    RandomForestClassifier(n_estimators=200),
    ExtraTreesClassifier(n_estimators=100),
    # All tree-based with similar characteristics
]
```

### Ensemble Size Optimization

```python
def optimize_ensemble_size(base_models, X_train, y_train, X_val, y_val):
    """Find optimal number of models in ensemble."""
    performance_curve = []

    for n_models in range(1, len(base_models) + 1):
        # Create ensemble with n_models
        current_ensemble = base_models[:n_models]
        voting_clf = VotingClassifier(
            [(f'model_{i}', model) for i, model in enumerate(current_ensemble)],
            voting='soft'
        )

        # Train and evaluate
        voting_clf.fit(X_train, y_train)
        score = voting_clf.score(X_val, y_val)
        performance_curve.append(score)

    # Find optimal size (considering performance vs complexity)
    optimal_size = np.argmax(performance_curve) + 1
    return optimal_size, performance_curve

optimal_n, curve = optimize_ensemble_size(
    good_ensemble, X_train, y_train, X_val, y_val
)
print(f"Optimal ensemble size: {optimal_n}")
```

### Computational Efficiency

```python
# Efficient ensemble training
def create_efficient_ensemble(X_train, y_train, time_budget=300):
    """Create ensemble within time budget."""
    start_time = time.time()
    trained_models = []

    # Quick models first
    quick_models = [
        ('nb', GaussianNB()),
        ('lr', LogisticRegression()),
        ('dt', DecisionTreeClassifier(max_depth=10))
    ]

    # Slower but accurate models
    slow_models = [
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('gb', GradientBoostingClassifier(n_estimators=100)),
        ('svm', SVC(probability=True))
    ]

    all_models = quick_models + slow_models

    for name, model in all_models:
        if time.time() - start_time > time_budget:
            break

        model.fit(X_train, y_train)
        trained_models.append((name, model))

        print(f"Trained {name} in {time.time() - start_time:.1f}s")

    # Create final ensemble
    return VotingClassifier(trained_models, voting='soft')

efficient_ensemble = create_efficient_ensemble(X_train, y_train, time_budget=180)
```

## Troubleshooting

### Common Issues

**Poor ensemble performance**:

- Base models too similar (low diversity)
- Weak base models dominating strong ones
- Overfitting in stacking

**Solutions**:

```python
# Increase diversity
diverse_models = [
    ('linear', make_pipeline(StandardScaler(), LogisticRegression())),
    ('tree', RandomForestClassifier()),
    ('kernel', SVC(probability=True)),
    ('naive', GaussianNB())
]

# Weight models by performance
model_weights = []
for name, model in base_models:
    cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()
    model_weights.append(cv_score)

weighted_voting = VotingClassifier(
    base_models,
    voting='soft',
    weights=model_weights
)

# Prevent stacking overfitting
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=Ridge(alpha=10.0),  # Regularized meta-model
    cv=10,  # More CV folds
    stack_method='predict_proba'
)
```

## See Also

- [Classification Algorithms](classification.md)
- [Regression Algorithms](regression.md)
- [Model Selection](../tutorials/model_selection.md)
- [API Reference: Ensemble](../api_reference/models.md#ensemble)
