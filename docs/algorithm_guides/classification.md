# Classification Algorithms Guide

This guide provides comprehensive coverage of classification algorithms implemented in the Scikit-Learn Mastery Project. Each algorithm includes theoretical background, implementation details, use cases, and practical examples.

## Table of Contents

1. [Overview](#overview)
2. [Logistic Regression](#logistic-regression)
3. [Random Forest](#random-forest)
4. [Support Vector Machines](#support-vector-machines)
5. [Gradient Boosting](#gradient-boosting)
6. [Neural Networks](#neural-networks)
7. [Naive Bayes](#naive-bayes)
8. [K-Nearest Neighbors](#k-nearest-neighbors)
9. [Decision Trees](#decision-trees)
10. [Extra Trees](#extra-trees)
11. [AdaBoost](#adaboost)
12. [Algorithm Comparison](#algorithm-comparison)
13. [Best Practices](#best-practices)

## Overview

Classification is a supervised learning task where the goal is to predict discrete class labels for input samples. This project implements 10+ classification algorithms, each with custom wrappers providing consistent interfaces and additional functionality.

### Common Interface

All classification models inherit from the base `ClassificationModel` class and provide:

```python
class ClassificationModel:
    def train(self, X, y)           # Train the model
    def predict(self, X)            # Make predictions
    def predict_proba(self, X)      # Probability estimates
    def evaluate(self, X, y)        # Comprehensive evaluation
    def cross_validate(self, X, y)  # Cross-validation
    def save_model(self, path)      # Model persistence
    def load_model(self, path)      # Model loading
```

## Logistic Regression

### Theory

Logistic regression uses the logistic function to model the probability of class membership. Despite its name, it's a classification algorithm that models the log-odds of the positive class.

**Mathematical Foundation:**

- **Sigmoid Function**: `p(y=1|x) = 1 / (1 + e^(-wx - b))`
- **Cost Function**: Cross-entropy loss
- **Optimization**: Gradient descent or L-BFGS

### Implementation

```python
from src.models.supervised.classification import LogisticRegressionModel

# Basic usage
model = LogisticRegressionModel(
    penalty='l2',           # Regularization type
    C=1.0,                 # Regularization strength (inverse)
    solver='lbfgs',        # Optimization algorithm
    max_iter=1000,         # Maximum iterations
    random_state=42
)

# Train model
model.train(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Get model coefficients
coefficients = model.get_coefficients()
intercept = model.get_intercept()
```

### When to Use

**✅ Good for:**

- Linear separable data
- Need for interpretable results
- Probability estimates required
- Large datasets (efficient)
- Baseline model

**❌ Avoid when:**

- Complex non-linear relationships
- Feature interactions are crucial
- Very small datasets

### Hyperparameters

| Parameter  | Description             | Typical Range                | Default |
| ---------- | ----------------------- | ---------------------------- | ------- |
| `C`        | Regularization strength | 0.001 - 100                  | 1.0     |
| `penalty`  | Regularization type     | 'l1', 'l2', 'elasticnet'     | 'l2'    |
| `solver`   | Optimization algorithm  | 'lbfgs', 'saga', 'liblinear' | 'lbfgs' |
| `max_iter` | Maximum iterations      | 100 - 10000                  | 1000    |

### Example: Multi-class Classification

```python
from src.data.generators import ClassificationDataGenerator
from src.utils.evaluation import ModelEvaluator

# Generate multi-class data
generator = ClassificationDataGenerator()
X, y = generator.generate_multiclass_classification(
    n_samples=1000,
    n_features=10,
    n_classes=4,
    n_informative=8
)

# Train logistic regression
model = LogisticRegressionModel(
    multi_class='ovr',  # One-vs-Rest strategy
    random_state=42
)
model.train(X, y)

# Evaluate
evaluator = ModelEvaluator(task_type='classification')
metrics = evaluator.evaluate(model, X, y)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1-Score: {metrics['f1']:.3f}")
```

## Random Forest

### Theory

Random Forest is an ensemble method that combines multiple decision trees using bootstrap sampling and random feature selection. It reduces overfitting and improves generalization.

**Key Concepts:**

- **Bootstrap Aggregating (Bagging)**: Train on random subsets
- **Random Feature Selection**: Consider subset of features at each split
- **Majority Voting**: Combine predictions from all trees

### Implementation

```python
from src.models.supervised.classification import RandomForestClassifierModel

# Configuration
model = RandomForestClassifierModel(
    n_estimators=100,         # Number of trees
    max_depth=None,          # Maximum tree depth
    min_samples_split=2,     # Minimum samples to split
    min_samples_leaf=1,      # Minimum samples in leaf
    max_features='sqrt',     # Features to consider for splits
    bootstrap=True,          # Bootstrap sampling
    random_state=42
)

# Advanced features
model.train(X_train, y_train)

# Feature importance
importance = model.get_feature_importance()
top_features = model.get_top_features(k=10)

# Out-of-bag score
oob_score = model.get_oob_score() if model.oob_score else None
```

### Feature Importance Analysis

```python
import matplotlib.pyplot as plt
import numpy as np

# Train model with OOB scoring
model = RandomForestClassifierModel(
    n_estimators=100,
    oob_score=True,
    random_state=42
)
model.train(X, y)

# Get feature importance
importance = model.get_feature_importance()
feature_names = [f'Feature_{i}' for i in range(len(importance))]

# Plot importance
plt.figure(figsize=(10, 6))
indices = np.argsort(importance)[::-1]
plt.bar(range(len(importance)), importance[indices])
plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.show()

print(f"OOB Score: {model.get_oob_score():.3f}")
```

### When to Use

**✅ Good for:**

- Mixed data types
- Feature importance needed
- Robust to outliers
- Non-linear relationships
- Default choice for many problems

**❌ Consider alternatives when:**

- Memory constraints (large forests)
- Need interpretability
- Real-time predictions required

## Support Vector Machines

### Theory

SVMs find the optimal hyperplane that separates classes with the maximum margin. They can handle non-linear data using kernel functions.

**Key Concepts:**

- **Support Vectors**: Data points closest to decision boundary
- **Margin**: Distance between support vectors and hyperplane
- **Kernel Trick**: Map data to higher dimensions

### Implementation

```python
from src.models.supervised.classification import SVMClassifierModel

# Linear SVM
linear_svm = SVMClassifierModel(
    kernel='linear',
    C=1.0,
    random_state=42
)

# RBF (Gaussian) Kernel SVM
rbf_svm = SVMClassifierModel(
    kernel='rbf',
    C=1.0,
    gamma='scale',      # Kernel coefficient
    random_state=42
)

# Polynomial Kernel SVM
poly_svm = SVMClassifierModel(
    kernel='poly',
    degree=3,           # Polynomial degree
    coef0=0.0,         # Independent term
    C=1.0,
    random_state=42
)

# For probability estimates
prob_svm = SVMClassifierModel(
    kernel='rbf',
    probability=True,   # Enable probability estimates
    random_state=42
)
```

### Kernel Selection Guide

```python
from sklearn.model_selection import validation_curve
import numpy as np

# Compare different kernels
kernels = ['linear', 'rbf', 'poly']
results = {}

for kernel in kernels:
    model = SVMClassifierModel(kernel=kernel, random_state=42)

    # Validation curve for C parameter
    param_range = np.logspace(-3, 2, 6)
    train_scores, val_scores = validation_curve(
        model.model, X, y,
        param_name='C',
        param_range=param_range,
        cv=5, scoring='accuracy'
    )

    results[kernel] = {
        'train_mean': np.mean(train_scores, axis=1),
        'val_mean': np.mean(val_scores, axis=1),
        'best_score': np.max(np.mean(val_scores, axis=1))
    }

# Print best kernel
best_kernel = max(results.keys(), key=lambda k: results[k]['best_score'])
print(f"Best kernel: {best_kernel} (Score: {results[best_kernel]['best_score']:.3f})")
```

### When to Use

**✅ Good for:**

- High-dimensional data
- Clear margin between classes
- Memory efficient (only stores support vectors)
- Effective with limited samples

**❌ Avoid when:**

- Very large datasets (>100k samples)
- Noisy data with overlapping classes
- Probability estimates crucial (without probability=True)

## Gradient Boosting

### Theory

Gradient Boosting builds models sequentially, with each new model correcting errors from previous models. It optimizes a differentiable loss function using gradient descent.

**Algorithm Steps:**

1. Initialize with constant prediction
2. For each iteration:
   - Calculate residuals (errors)
   - Train weak learner on residuals
   - Add to ensemble with learning rate

### Implementation

```python
from src.models.supervised.classification import GradientBoostingClassifierModel

# Standard configuration
model = GradientBoostingClassifierModel(
    n_estimators=100,        # Number of boosting stages
    learning_rate=0.1,       # Shrinkage parameter
    max_depth=3,            # Individual tree depth
    min_samples_split=2,     # Minimum samples to split
    min_samples_leaf=1,      # Minimum samples in leaf
    subsample=1.0,          # Fraction of samples for training
    random_state=42
)

# Training with early stopping
model.train(X_train, y_train)

# Staged predictions (useful for early stopping)
staged_predictions = list(model.staged_predict(X_test))
staged_probabilities = list(model.staged_predict_proba(X_test))

# Feature importance
importance = model.get_feature_importance()
```

### Early Stopping Example

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = GradientBoostingClassifierModel(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)
model.train(X_train, y_train)

# Calculate validation loss for each stage
val_losses = []
for pred_proba in model.staged_predict_proba(X_val):
    val_losses.append(log_loss(y_val, pred_proba))

# Find optimal number of estimators
optimal_n_estimators = np.argmin(val_losses) + 1

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.axvline(x=optimal_n_estimators, color='red', linestyle='--',
           label=f'Optimal n_estimators: {optimal_n_estimators}')
plt.xlabel('Number of Estimators')
plt.ylabel('Log Loss')
plt.title('Gradient Boosting - Early Stopping')
plt.legend()
plt.show()

print(f"Optimal number of estimators: {optimal_n_estimators}")
```

### When to Use

**✅ Good for:**

- Structured/tabular data
- High predictive performance needed
- Can handle missing values
- Feature interactions important

**❌ Consider alternatives when:**

- Training time is critical
- Simple, interpretable model needed
- Very noisy data

## Neural Networks

### Theory

Multi-layer Perceptrons (MLPs) use multiple layers of neurons with non-linear activation functions to learn complex patterns. They use backpropagation for training.

### Implementation

```python
from src.models.supervised.classification import NeuralNetworkClassifierModel

# Basic MLP
model = NeuralNetworkClassifierModel(
    hidden_layer_sizes=(100, 50),  # Two hidden layers
    activation='relu',              # Activation function
    solver='adam',                 # Optimization algorithm
    alpha=0.0001,                  # L2 regularization
    learning_rate='constant',       # Learning rate schedule
    learning_rate_init=0.001,      # Initial learning rate
    max_iter=200,                  # Maximum iterations
    random_state=42
)

# Advanced configuration
advanced_model = NeuralNetworkClassifierModel(
    hidden_layer_sizes=(200, 100, 50),
    activation='tanh',
    solver='lbfgs',               # For small datasets
    alpha=0.01,                   # Stronger regularization
    early_stopping=True,          # Early stopping
    validation_fraction=0.1,      # Validation set size
    n_iter_no_change=10,         # Patience for early stopping
    random_state=42
)
```

### Architecture Selection

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'hidden_layer_sizes': [
        (50,), (100,), (200,),
        (100, 50), (200, 100),
        (100, 50, 25)
    ],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01]
}

# Grid search
model = NeuralNetworkClassifierModel()
grid_search = GridSearchCV(
    model.model, param_grid,
    cv=5, scoring='accuracy',
    n_jobs=-1, verbose=1
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
```

### When to Use

**✅ Good for:**

- Large datasets
- Complex non-linear relationships
- Image/text data (with appropriate preprocessing)
- Feature interactions important

**❌ Avoid when:**

- Small datasets (<1000 samples)
- Interpretability required
- Limited computational resources

## Algorithm Comparison

### Performance Comparison Framework

```python
from src.utils.evaluation import PerformanceComparator
from sklearn.model_selection import cross_val_score

# Define models to compare
models = {
    'Logistic Regression': LogisticRegressionModel(random_state=42),
    'Random Forest': RandomForestClassifierModel(n_estimators=100, random_state=42),
    'SVM': SVMClassifierModel(kernel='rbf', random_state=42),
    'Gradient Boosting': GradientBoostingClassifierModel(n_estimators=100, random_state=42),
    'Neural Network': NeuralNetworkClassifierModel(hidden_layer_sizes=(100,), random_state=42)
}

# Compare models
comparator = PerformanceComparator()
comparison_results = {}

for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model.model, X, y, cv=5, scoring='accuracy')

    # Train on full dataset and evaluate
    model.train(X_train, y_train)
    predictions = model.predict(X_test)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    comparison_results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy': accuracy_score(y_test, predictions),
        'test_precision': precision_score(y_test, predictions, average='weighted'),
        'test_recall': recall_score(y_test, predictions, average='weighted'),
        'test_f1': f1_score(y_test, predictions, average='weighted')
    }

# Display results
import pandas as pd
results_df = pd.DataFrame(comparison_results).T
print(results_df.round(3))
```

### Dataset Size Recommendations

| Dataset Size     | Recommended Algorithms                            | Notes                      |
| ---------------- | ------------------------------------------------- | -------------------------- |
| < 1,000          | Logistic Regression, SVM, Naive Bayes             | Simple models work well    |
| 1,000 - 10,000   | Random Forest, SVM, Gradient Boosting             | Tree-based methods excel   |
| 10,000 - 100,000 | Random Forest, Gradient Boosting, Neural Networks | Ensemble methods preferred |
| > 100,000        | Gradient Boosting, Neural Networks, Linear models | Scalable algorithms        |

## Best Practices

### 1. Data Preprocessing

```python
from sklearn.pipeline import Pipeline
from src.pipelines.custom_transformers import *

# Comprehensive preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('imputer', MissingValueImputer(strategy='mean')),
    ('outlier_removal', OutlierRemover(method='iqr')),
    ('encoding', CategoryEncoder(method='onehot')),
    ('scaling', ScalerTransformer(method='standard')),
    ('feature_selection', FeatureSelector(method='univariate', k=20))
])

# Apply preprocessing
X_preprocessed = preprocessing_pipeline.fit_transform(X, y)
```

### 2. Model Selection Strategy

```python
def select_best_model(X, y, models_dict, cv=5):
    """
    Systematic model selection with cross-validation.
    """
    results = {}

    for name, model in models_dict.items():
        # Time the training
        import time
        start_time = time.time()

        # Cross-validation
        cv_scores = cross_val_score(model.model, X, y, cv=cv, scoring='accuracy')

        training_time = time.time() - start_time

        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': training_time
        }

    # Select best model
    best_model = max(results.keys(), key=lambda k: results[k]['cv_mean'])

    return best_model, results

# Usage
best_model_name, all_results = select_best_model(X, y, models)
print(f"Best model: {best_model_name}")
```

### 3. Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Random Forest hyperparameter tuning
rf_param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None] + list(range(5, 20)),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

rf_model = RandomForestClassifierModel(random_state=42)
random_search = RandomizedSearchCV(
    rf_model.model, rf_param_dist,
    n_iter=100, cv=5, scoring='accuracy',
    random_state=42, n_jobs=-1
)

random_search.fit(X_train, y_train)
print("Best parameters:", random_search.best_params_)
```

### 4. Handling Imbalanced Data

```python
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from collections import Counter

# Check class distribution
print("Class distribution:", Counter(y))

# Method 1: Class weights
class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y), y=y
)
class_weight_dict = dict(zip(np.unique(y), class_weights))

model_with_weights = RandomForestClassifierModel(
    class_weight=class_weight_dict,
    random_state=42
)

# Method 2: SMOTE oversampling
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
print("Balanced distribution:", Counter(y_balanced))

# Train on balanced data
model_balanced = RandomForestClassifierModel(random_state=42)
model_balanced.train(X_balanced, y_balanced)
```

### 5. Model Interpretation

```python
# Feature importance analysis
def analyze_feature_importance(model, feature_names, top_k=10):
    """
    Analyze and visualize feature importance.
    """
    if hasattr(model, 'get_feature_importance'):
        importance = model.get_feature_importance()

        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        # Plot top features
        plt.figure(figsize=(10, 6))
        top_features = importance_df.head(top_k)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_k} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

        return importance_df
    else:
        print("Model doesn't support feature importance")
        return None

# Usage
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
importance_df = analyze_feature_importance(model, feature_names)
```

## Conclusion

This guide provides a comprehensive overview of classification algorithms in the Scikit-Learn Mastery Project. Each algorithm has its strengths and ideal use cases:

- **Start with Logistic Regression** for baseline and interpretability
- **Use Random Forest** as a robust default choice
- **Consider SVM** for high-dimensional data
- **Apply Gradient Boosting** for maximum performance on tabular data
- **Use Neural Networks** for complex patterns and large datasets

Remember to always:

1. Understand your data first
2. Start with simple models
3. Use cross-validation for model selection
4. Tune hyperparameters systematically
5. Evaluate on multiple metrics
6. Consider computational constraints

For practical examples and hands-on tutorials, see the [tutorials section](../tutorials/) and [Jupyter notebooks](../../notebooks/).
