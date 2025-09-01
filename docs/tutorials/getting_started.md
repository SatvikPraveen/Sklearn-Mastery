# Getting Started Tutorial

Welcome to the Scikit-Learn Mastery Project! This tutorial will guide you through your first steps with the framework, from installation to building your first machine learning model.

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Your First Model](#your-first-model)
4. [Data Generation](#data-generation)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Saving and Loading Models](#saving-and-loading-models)
8. [Next Steps](#next-steps)

## Installation

### Prerequisites

Ensure you have Python 3.8 or higher installed:

```bash
python --version
```

### Clone and Install

```bash
# Clone the repository
git clone <repository-url>
cd scikit-learn-mastery

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

### Verify Installation

```python
# Test the installation
python -c "from src.models.supervised.classification import LogisticRegressionModel; print('Installation successful!')"
```

## Project Structure

Before we start, let's understand the project layout:

```
scikit-learn-mastery/
├── src/                    # Main source code
│   ├── data/              # Data generation and loading
│   ├── preprocessing/     # Data preprocessing
│   ├── models/           # ML model implementations
│   │   ├── supervised/   # Classification and regression
│   │   ├── unsupervised/ # Clustering and dimensionality reduction
│   │   └── ensemble/     # Ensemble methods
│   ├── pipelines/        # Custom transformers and pipelines
│   └── utils/           # Evaluation and visualization
├── tests/               # Test suite
├── notebooks/          # Jupyter tutorials
├── examples/           # Real-world examples
├── docs/              # Documentation
└── results/           # Output directory
```

## Your First Model

Let's build your first classification model step by step.

### Step 1: Import Required Modules

```python
# Core imports
import numpy as np
import matplotlib.pyplot as plt

# Project imports
from src.data.generators import ClassificationDataGenerator
from src.models.supervised.classification import LogisticRegressionModel
from src.utils.evaluation import ModelEvaluator
from src.utils.visualization import DataVisualizer, ModelVisualizer

# Set random seed for reproducibility
np.random.seed(42)
```

### Step 2: Generate Sample Data

```python
# Create data generator
generator = ClassificationDataGenerator()

# Generate a simple binary classification dataset
X, y = generator.generate_basic_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42
)

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class distribution: {np.bincount(y)}")
```

### Step 3: Visualize the Data

```python
# Create data visualizer
data_viz = DataVisualizer()

# Plot the dataset
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot of features
scatter = axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].set_title('Generated Classification Data')
plt.colorbar(scatter, ax=axes[0])

# Class distribution
data_viz.plot_class_distribution(y, ax=axes[1])

plt.tight_layout()
plt.show()
```

### Step 4: Split the Data

```python
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
```

## Data Generation

The project provides comprehensive data generation utilities. Let's explore different types of datasets:

### Classification Data

```python
# Binary classification
X_binary, y_binary = generator.generate_basic_classification(
    n_samples=500,
    n_features=4,
    n_classes=2,
    random_state=42
)

# Multi-class classification
X_multi, y_multi = generator.generate_multiclass_classification(
    n_samples=600,
    n_features=6,
    n_classes=4,
    n_informative=4,
    random_state=42
)

# Imbalanced classification
X_imbalanced, y_imbalanced = generator.generate_imbalanced_classification(
    n_samples=800,
    n_features=5,
    weights=[0.1, 0.3, 0.6],  # Class weights
    random_state=42
)

print("Generated datasets:")
print(f"Binary: {X_binary.shape}, Classes: {np.unique(y_binary)}")
print(f"Multi-class: {X_multi.shape}, Classes: {np.unique(y_multi)}")
print(f"Imbalanced: {X_imbalanced.shape}, Distribution: {np.bincount(y_imbalanced)}")
```

### Specialized Datasets

```python
# Non-linear separable data
X_nonlinear, y_nonlinear = generator.generate_moons_classification(
    n_samples=400,
    noise=0.1,
    random_state=42
)

# Concentric circles
X_circles, y_circles = generator.generate_circles_classification(
    n_samples=300,
    factor=0.3,
    noise=0.05,
    random_state=42
)

# Visualize specialized datasets
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X_nonlinear[:, 0], X_nonlinear[:, 1], c=y_nonlinear, cmap='viridis')
axes[0].set_title('Moons Dataset')

axes[1].scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='viridis')
axes[1].set_title('Circles Dataset')

plt.tight_layout()
plt.show()
```

## Model Training

Now let's train different types of models:

### Logistic Regression

```python
# Create and train logistic regression model
lr_model = LogisticRegressionModel(
    C=1.0,              # Regularization strength
    random_state=42,
    max_iter=1000
)

# Train the model
lr_model.train(X_train, y_train)

# Make predictions
lr_predictions = lr_model.predict(X_test)
lr_probabilities = lr_model.predict_proba(X_test)

print("Logistic Regression Model trained successfully!")
print(f"Model coefficients shape: {lr_model.model.coef_.shape}")
print(f"Model intercept: {lr_model.model.intercept_}")
```

### Random Forest

```python
from src.models.supervised.classification import RandomForestClassifierModel

# Create and train random forest model
rf_model = RandomForestClassifierModel(
    n_estimators=100,
    random_state=42,
    max_depth=5
)

# Train the model
rf_model.train(X_train, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test)
rf_probabilities = rf_model.predict_proba(X_test)

print("Random Forest Model trained successfully!")
print(f"Number of trees: {rf_model.model.n_estimators}")
print(f"Feature importances: {rf_model.model.feature_importances_}")
```

## Model Evaluation

### Basic Metrics

```python
# Create model evaluator
evaluator = ModelEvaluator(task_type='classification')

# Evaluate Logistic Regression
lr_metrics = evaluator.evaluate(lr_model, X_test, y_test)
print("Logistic Regression Metrics:")
for metric, value in lr_metrics.items():
    print(f"  {metric}: {value:.3f}")

# Evaluate Random Forest
rf_metrics = evaluator.evaluate(rf_model, X_test, y_test)
print("\nRandom Forest Metrics:")
for metric, value in rf_metrics.items():
    print(f"  {metric}: {value:.3f}")
```

### Detailed Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix

# Classification report
print("Logistic Regression - Classification Report:")
print(classification_report(y_test, lr_predictions))

print("\nRandom Forest - Classification Report:")
print(classification_report(y_test, rf_predictions))

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Logistic Regression confusion matrix
cm_lr = confusion_matrix(y_test, lr_predictions)
axes[0].imshow(cm_lr, interpolation='nearest', cmap=plt.cm.Blues)
axes[0].set_title('Logistic Regression\nConfusion Matrix')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# Add text annotations
for i in range(cm_lr.shape[0]):
    for j in range(cm_lr.shape[1]):
        axes[0].text(j, i, cm_lr[i, j], ha="center", va="center")

# Random Forest confusion matrix
cm_rf = confusion_matrix(y_test, rf_predictions)
axes[1].imshow(cm_rf, interpolation='nearest', cmap=plt.cm.Blues)
axes[1].set_title('Random Forest\nConfusion Matrix')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

# Add text annotations
for i in range(cm_rf.shape[0]):
    for j in range(cm_rf.shape[1]):
        axes[1].text(j, i, cm_rf[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()
```

### Visualization

```python
# Create model visualizer
model_viz = ModelVisualizer()

# Plot decision boundaries (for 2D data)
if X.shape[1] == 2:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Logistic Regression decision boundary
    model_viz.plot_decision_boundary(
        lr_model, X_test, y_test,
        title='Logistic Regression Decision Boundary',
        ax=axes[0]
    )

    # Random Forest decision boundary
    model_viz.plot_decision_boundary(
        rf_model, X_test, y_test,
        title='Random Forest Decision Boundary',
        ax=axes[1]
    )

    plt.tight_layout()
    plt.show()

# ROC curves
model_viz.plot_roc_curves({
    'Logistic Regression': lr_probabilities,
    'Random Forest': rf_probabilities
}, y_test)
```

## Saving and Loading Models

### Save Models

```python
import joblib
import os

# Create results directory if it doesn't exist
os.makedirs('results/models', exist_ok=True)

# Save models
joblib.dump(lr_model, 'results/models/logistic_regression_model.pkl')
joblib.dump(rf_model, 'results/models/random_forest_model.pkl')

print("Models saved successfully!")
```

### Load Models

```python
# Load models
loaded_lr = joblib.load('results/models/logistic_regression_model.pkl')
loaded_rf = joblib.load('results/models/random_forest_model.pkl')

# Verify loaded models work
test_predictions_lr = loaded_lr.predict(X_test[:5])
test_predictions_rf = loaded_rf.predict(X_test[:5])

print("Loaded models verification:")
print(f"LR predictions: {test_predictions_lr}")
print(f"RF predictions: {test_predictions_rf}")
```

### Model Persistence Best Practices

```python
# Save model with metadata
import json
from datetime import datetime

model_metadata = {
    'model_type': 'LogisticRegression',
    'training_date': datetime.now().isoformat(),
    'training_samples': len(X_train),
    'features': X_train.shape[1],
    'performance': lr_metrics,
    'hyperparameters': {
        'C': 1.0,
        'max_iter': 1000,
        'random_state': 42
    }
}

# Save metadata
with open('results/models/logistic_regression_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

print("Model metadata saved!")
```

## Quick Start Template

Here's a template you can use for future projects:

```python
# Quick Start Template
from src.data.generators import ClassificationDataGenerator
from src.models.supervised.classification import LogisticRegressionModel
from src.utils.evaluation import ModelEvaluator
import numpy as np

# Set random seed
np.random.seed(42)

# 1. Generate data
generator = ClassificationDataGenerator()
X, y = generator.generate_basic_classification(n_samples=1000, n_features=10)

# 2. Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train model
model = LogisticRegressionModel(random_state=42)
model.train(X_train, y_train)

# 4. Evaluate
evaluator = ModelEvaluator(task_type='classification')
metrics = evaluator.evaluate(model, X_test, y_test)

# 5. Print results
print("Model Performance:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.3f}")
```

## Next Steps

Congratulations! You've completed your first machine learning workflow with the Scikit-Learn Mastery framework. Here's what to explore next:

### Immediate Next Steps

1. **[Data Preprocessing Tutorial](data_preprocessing.md)** - Learn advanced data preparation techniques
2. **[Model Selection Tutorial](model_selection.md)** - Master systematic model comparison and selection
3. **Explore Notebooks** - Work through the Jupyter notebooks in the `notebooks/` directory

### Advanced Topics

4. **Ensemble Methods** - Combine multiple models for better performance
5. **Hyperparameter Tuning** - Optimize your models for best results
6. **Cross-Validation** - Robust model evaluation techniques
7. **Feature Engineering** - Create better features for improved performance

### Real-World Applications

8. **Check Examples** - Explore real-world scenarios in the `examples/` directory
9. **Custom Pipelines** - Build end-to-end ML pipelines
10. **Model Deployment** - Learn to deploy your models in production

### Getting Help

- **Documentation**: Check the `docs/` directory for detailed API reference
- **Examples**: Look at `examples/real_world_scenarios/` for practical use cases
- **Tests**: Review `tests/` directory to understand expected behavior
- **Issues**: If you encounter problems, check the project's issue tracker

### Learning Path Recommendations

**Beginner Path:**

1. Complete all tutorials in order
2. Work through notebooks 01-03
3. Try examples with different datasets

**Intermediate Path:**

1. Focus on model_selection.md and advanced preprocessing
2. Work through notebooks 04-06
3. Experiment with ensemble methods

**Advanced Path:**

1. Explore custom transformers and pipelines
2. Work through notebook 07
3. Contribute to the project or build custom extensions

---

**Happy Learning!** 🎉

Remember: The best way to learn machine learning is by doing. Don't hesitate to experiment with different datasets, models, and parameters. The framework is designed to make experimentation easy and systematic.
