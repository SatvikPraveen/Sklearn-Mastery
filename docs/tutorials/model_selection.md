# Model Selection Tutorial

Choosing the right machine learning algorithm is crucial for project success. This tutorial provides a systematic approach to model selection using the Scikit-Learn Mastery Project framework.

## Table of Contents

1. [Overview](#overview)
2. [Understanding Your Problem](#understanding-your-problem)
3. [Dataset Characteristics](#dataset-characteristics)
4. [Algorithm Categories](#algorithm-categories)
5. [Systematic Model Comparison](#systematic-model-comparison)
6. [Performance Evaluation](#performance-evaluation)
7. [Computational Considerations](#computational-considerations)
8. [Model Selection Framework](#model-selection-framework)
9. [Real-World Examples](#real-world-examples)
10. [Best Practices](#best-practices)

## Overview

Model selection involves choosing the most appropriate algorithm for your specific problem and dataset. This tutorial provides a structured approach using:

- **Problem analysis** - Understanding task requirements
- **Data profiling** - Analyzing dataset characteristics
- **Algorithm comparison** - Systematic evaluation of candidates
- **Performance metrics** - Comprehensive evaluation criteria
- **Practical constraints** - Considering real-world limitations

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Project imports
from src.data.generators import ClassificationDataGenerator, RegressionDataGenerator
from src.models.supervised.classification import *
from src.models.supervised.regression import *
from src.utils.evaluation import ModelEvaluator, PerformanceComparator
from src.utils.visualization import DataVisualizer, PerformanceVisualizer

# Set random seed for reproducibility
np.random.seed(42)
```

## Understanding Your Problem

### Problem Type Classification

```python
def analyze_problem_type(y):
    """Analyze the machine learning problem type."""

    analysis = {
        'problem_type': None,
        'n_samples': len(y),
        'n_unique_values': len(np.unique(y)),
        'data_type': str(y.dtype),
        'recommendations': []
    }

    # Determine problem type
    if np.issubdtype(y.dtype, np.integer) and analysis['n_unique_values'] < 20:
        analysis['problem_type'] = 'classification'
        analysis['n_classes'] = analysis['n_unique_values']

        if analysis['n_classes'] == 2:
            analysis['subtype'] = 'binary_classification'
        else:
            analysis['subtype'] = 'multiclass_classification'

    elif np.issubdtype(y.dtype, np.floating) or analysis['n_unique_values'] > 20:
        analysis['problem_type'] = 'regression'
        analysis['target_range'] = [float(np.min(y)), float(np.max(y))]
        analysis['target_distribution'] = 'continuous'

    # Add recommendations
    if analysis['problem_type'] == 'classification':
        if analysis['n_classes'] == 2:
            analysis['recommendations'].append("Consider: Logistic Regression, SVM, Random Forest")
        else:
            analysis['recommendations'].append("Consider: Random Forest, Gradient Boosting, Neural Networks")

        # Class balance analysis
        class_counts = pd.Series(y).value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min()

        if imbalance_ratio > 3:
            analysis['class_imbalance'] = True
            analysis['imbalance_ratio'] = float(imbalance_ratio)
            analysis['recommendations'].append("Handle class imbalance with resampling or class weights")
        else:
            analysis['class_imbalance'] = False

    elif analysis['problem_type'] == 'regression':
        analysis['recommendations'].append("Consider: Linear Regression, Random Forest, Gradient Boosting")

        # Check for skewness
        from scipy import stats
        skewness = stats.skew(y)
        if abs(skewness) > 1:
            analysis['target_skewness'] = float(skewness)
            analysis['recommendations'].append("Consider log transformation for skewed target")

    return analysis

# Example problem analysis
generator = ClassificationDataGenerator()
X_example, y_example = generator.generate_basic_classification(
    n_samples=1000, n_features=10, n_classes=3, random_state=42
)

problem_analysis = analyze_problem_type(y_example)
print("Problem Analysis:")
for key, value in problem_analysis.items():
    print(f"  {key}: {value}")
```

### Business Requirements Assessment

```python
def assess_business_requirements():
    """Template for assessing business requirements that affect model selection."""

    requirements = {
        'interpretability': {
            'level': 'high',  # 'low', 'medium', 'high'
            'reasoning': 'Need to explain predictions to stakeholders',
            'suggested_models': ['Logistic Regression', 'Decision Tree', 'Linear Regression']
        },
        'prediction_speed': {
            'requirement': 'real_time',  # 'batch', 'near_real_time', 'real_time'
            'max_latency_ms': 100,
            'suggested_models': ['Logistic Regression', 'Naive Bayes', 'Linear SVM']
        },
        'training_time': {
            'constraint': 'low',  # 'low', 'medium', 'high'
            'max_training_time': '1 hour',
            'retraining_frequency': 'daily'
        },
        'accuracy_requirements': {
            'minimum_accuracy': 0.85,
            'metric_priority': 'precision',  # 'accuracy', 'precision', 'recall', 'f1'
            'tolerance_for_complexity': 'medium'
        },
        'data_characteristics': {
            'updates_frequently': True,
            'missing_values_common': True,
            'feature_drift_expected': True
        }
    }

    return requirements

business_reqs = assess_business_requirements()
print("Business Requirements Example:")
for category, details in business_reqs.items():
    print(f"\n{category.upper()}:")
    for key, value in details.items():
        print(f"  {key}: {value}")
```

## Dataset Characteristics

### Data Profiling

```python
def profile_dataset(X, y, feature_names=None):
    """Comprehensive dataset profiling for model selection."""

    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    profile = {
        'dimensions': {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'samples_to_features_ratio': X.shape[0] / X.shape[1]
        },
        'data_quality': {
            'missing_percentage': (np.isnan(X).sum() / X.size) * 100,
            'duplicate_rows': 0,  # Would need to implement
            'constant_features': 0
        },
        'feature_characteristics': {},
        'target_analysis': {},
        'recommendations': []
    }

    # Analyze features
    for i, feature_name in enumerate(feature_names):
        feature_data = X[:, i]
        profile['feature_characteristics'][feature_name] = {
            'dtype': str(feature_data.dtype),
            'missing_pct': (np.isnan(feature_data).sum() / len(feature_data)) * 100,
            'unique_values': len(np.unique(feature_data[~np.isnan(feature_data)])),
            'variance': float(np.var(feature_data[~np.isnan(feature_data)])),
            'range': [float(np.nanmin(feature_data)), float(np.nanmax(feature_data))]
        }

    # Check for constant features
    variances = [info['variance'] for info in profile['feature_characteristics'].values()]
    profile['data_quality']['constant_features'] = sum(1 for v in variances if v < 1e-10)

    # Generate recommendations based on dataset characteristics
    if profile['dimensions']['samples_to_features_ratio'] < 10:
        profile['recommendations'].append("High-dimensional data: Consider regularized models (Ridge, Lasso, Elastic Net)")
        profile['recommendations'].append("Feature selection strongly recommended")

    if profile['dimensions']['n_samples'] < 1000:
        profile['recommendations'].append("Small dataset: Avoid complex models, consider cross-validation")
        profile['recommendations'].append("Simple models preferred: Logistic Regression, Naive Bayes")
    elif profile['dimensions']['n_samples'] > 100000:
        profile['recommendations'].append("Large dataset: Can use complex models (Neural Networks, Ensemble methods)")
        profile['recommendations'].append("Consider computational efficiency")

    if profile['data_quality']['missing_percentage'] > 10:
        profile['recommendations'].append("High missing data: Use robust imputation or models that handle missing values")

    return profile

# Profile our example dataset
dataset_profile = profile_dataset(X_example, y_example)
print("Dataset Profile:")
for category, details in dataset_profile.items():
    if category != 'feature_characteristics':  # Skip detailed feature info for brevity
        print(f"\n{category.upper()}:")
        if isinstance(details, dict):
            for key, value in details.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {details}")
```

### Data Distribution Analysis

```python
def analyze_data_distributions(X, y, sample_features=3):
    """Analyze data distributions to guide model selection."""

    # Sample a few features for analysis
    n_features = min(sample_features, X.shape[1])
    feature_indices = np.random.choice(X.shape[1], n_features, replace=False)

    fig, axes = plt.subplots(2, n_features, figsize=(15, 8))
    if n_features == 1:
        axes = axes.reshape(-1, 1)

    distribution_analysis = {}

    for i, feature_idx in enumerate(feature_indices):
        feature_data = X[:, feature_idx]
        feature_name = f'Feature {feature_idx}'

        # Distribution shape analysis
        from scipy import stats
        skewness = stats.skew(feature_data[~np.isnan(feature_data)])
        kurtosis = stats.kurtosis(feature_data[~np.isnan(feature_data)])

        distribution_analysis[feature_name] = {
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'distribution_type': 'normal' if abs(skewness) < 0.5 else 'skewed'
        }

        # Plot distributions
        axes[0, i].hist(feature_data, bins=30, alpha=0.7)
        axes[0, i].set_title(f'{feature_name} Distribution')
        axes[0, i].set_ylabel('Frequency')

        # Box plot by target class (for classification)
        if len(np.unique(y)) < 10:  # Assume classification
            for class_label in np.unique(y):
                class_data = feature_data[y == class_label]
                axes[1, i].boxplot(class_data, positions=[class_label], widths=0.3)
            axes[1, i].set_title(f'{feature_name} by Class')
            axes[1, i].set_xlabel('Class')
            axes[1, i].set_ylabel('Value')
        else:  # Regression
            axes[1, i].scatter(feature_data, y, alpha=0.5)
            axes[1, i].set_title(f'{feature_name} vs Target')
            axes[1, i].set_xlabel('Feature Value')
            axes[1, i].set_ylabel('Target')

    plt.tight_layout()
    plt.show()

    # Generate recommendations based on distributions
    recommendations = []

    skewed_features = sum(1 for analysis in distribution_analysis.values()
                         if abs(analysis['skewness']) > 1)

    if skewed_features > len(distribution_analysis) / 2:
        recommendations.append("Many skewed features: Consider data transformation or tree-based models")

    high_kurtosis_features = sum(1 for analysis in distribution_analysis.values()
                                if abs(analysis['kurtosis']) > 3)

    if high_kurtosis_features > 0:
        recommendations.append("High kurtosis detected: Consider robust models (Random Forest, SVM)")

    print("Distribution Analysis:")
    for feature, analysis in distribution_analysis.items():
        print(f"  {feature}: Skewness={analysis['skewness']:.2f}, Kurtosis={analysis['kurtosis']:.2f}")

    print("\nRecommendations:")
    for rec in recommendations:
        print(f"  - {rec}")

    return distribution_analysis, recommendations

# Analyze distributions
dist_analysis, dist_recommendations = analyze_data_distributions(X_example, y_example)
```

## Algorithm Categories

### Model Taxonomy

```python
def get_algorithm_taxonomy():
    """Comprehensive taxonomy of available algorithms."""

    taxonomy = {
        'classification': {
            'linear': {
                'algorithms': ['LogisticRegression'],
                'characteristics': ['Linear decision boundary', 'Fast training', 'Interpretable'],
                'best_for': ['Linearly separable data', 'High-dimensional data', 'Baseline models']
            },
            'tree_based': {
                'algorithms': ['DecisionTree', 'RandomForest', 'ExtraTrees', 'GradientBoosting'],
                'characteristics': ['Non-linear', 'Handle mixed data types', 'Feature importance'],
                'best_for': ['Structured data', 'Mixed data types', 'Feature interactions']
            },
            'instance_based': {
                'algorithms': ['KNN'],
                'characteristics': ['Non-parametric', 'Local patterns', 'Simple'],
                'best_for': ['Small datasets', 'Local patterns', 'Baseline comparison']
            },
            'kernel_based': {
                'algorithms': ['SVM'],
                'characteristics': ['Kernel trick', 'Memory efficient', 'Good generalization'],
                'best_for': ['High-dimensional data', 'Non-linear patterns', 'Small to medium datasets']
            },
            'probabilistic': {
                'algorithms': ['NaiveBayes'],
                'characteristics': ['Probabilistic', 'Fast', 'Handles missing values'],
                'best_for': ['Text data', 'Small datasets', 'Baseline models']
            },
            'neural': {
                'algorithms': ['NeuralNetwork'],
                'characteristics': ['Universal approximator', 'Complex patterns', 'Requires large data'],
                'best_for': ['Large datasets', 'Complex patterns', 'Image/text data']
            },
            'ensemble': {
                'algorithms': ['AdaBoost', 'Voting', 'Stacking'],
                'characteristics': ['Combines multiple models', 'Reduces overfitting', 'High performance'],
                'best_for': ['Competition settings', 'Maximum performance', 'Stable predictions']
            }
        },
        'regression': {
            'linear': {
                'algorithms': ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet'],
                'characteristics': ['Linear relationship', 'Interpretable', 'Regularization options'],
                'best_for': ['Linear relationships', 'Feature selection', 'Interpretability']
            },
            'tree_based': {
                'algorithms': ['DecisionTreeRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor'],
                'characteristics': ['Non-linear', 'Robust to outliers', 'Feature importance'],
                'best_for': ['Non-linear relationships', 'Structured data', 'Robust predictions']
            },
            'kernel_based': {
                'algorithms': ['SVMRegressor'],
                'characteristics': ['Non-linear via kernels', 'Robust', 'Memory efficient'],
                'best_for': ['Non-linear patterns', 'Robust predictions', 'High-dimensional data']
            },
            'neural': {
                'algorithms': ['NeuralNetworkRegressor'],
                'characteristics': ['Universal approximator', 'Complex patterns', 'Requires tuning'],
                'best_for': ['Complex non-linear patterns', 'Large datasets', 'Deep relationships']
            }
        }
    }

    return taxonomy

# Display algorithm taxonomy
taxonomy = get_algorithm_taxonomy()
print("Algorithm Taxonomy:")
for problem_type, categories in taxonomy.items():
    print(f"\n{problem_type.upper()}:")
    for category, info in categories.items():
        print(f"  {category}:")
        print(f"    Algorithms: {', '.join(info['algorithms'])}")
        print(f"    Best for: {', '.join(info['best_for'])}")
```

### Algorithm Selection Matrix

```python
def create_selection_matrix():
    """Create algorithm selection matrix based on dataset characteristics."""

    selection_matrix = pd.DataFrame({
        'Algorithm': [
            'LogisticRegression', 'RandomForest', 'SVM', 'GradientBoosting',
            'NeuralNetwork', 'NaiveBayes', 'KNN', 'DecisionTree'
        ],
        'Small_Dataset': [9, 7, 8, 6, 3, 9, 8, 7],
        'Large_Dataset': [8, 9, 6, 9, 9, 7, 4, 6],
        'High_Dimensional': [9, 7, 9, 7, 8, 8, 3, 5],
        'Interpretability': [9, 6, 4, 5, 2, 7, 8, 9],
        'Training_Speed': [9, 6, 7, 5, 3, 9, 9, 8],
        'Prediction_Speed': [9, 7, 8, 7, 8, 9, 6, 9],
        'Non_Linear': [3, 9, 8, 9, 9, 4, 8, 9],
        'Missing_Values': [5, 8, 5, 8, 6, 7, 4, 6],
        'Categorical_Features': [6, 9, 6, 9, 7, 8, 7, 9],
        'Imbalanced_Classes': [7, 8, 7, 8, 7, 6, 6, 7]
    })

    return selection_matrix

# Create and display selection matrix
selection_matrix = create_selection_matrix()
print("Algorithm Selection Matrix (1-10 scale, 10=best):")
print(selection_matrix.set_index('Algorithm'))

# Visualize selection matrix
plt.figure(figsize=(12, 8))
sns.heatmap(selection_matrix.set_index('Algorithm'),
            annot=True, cmap='RdYlGn', center=5.5,
            cbar_kws={'label': 'Suitability Score'})
plt.title('Algorithm Selection Matrix')
plt.xlabel('Dataset Characteristics')
plt.ylabel('Algorithms')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## Systematic Model Comparison

### Comprehensive Model Evaluation

```python
def systematic_model_comparison(X, y, problem_type='classification'):
    """Systematically compare multiple models."""

    # Define models based on problem type
    if problem_type == 'classification':
        models = {
            'Logistic Regression': LogisticRegressionModel(random_state=42),
            'Random Forest': RandomForestClassifierModel(n_estimators=100, random_state=42),
            'SVM': SVMClassifierModel(random_state=42),
            'Gradient Boosting': GradientBoostingClassifierModel(n_estimators=100, random_state=42),
            'Naive Bayes': NaiveBayesModel(),
            'KNN': KNNClassifierModel(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifierModel(random_state=42)
        }
        evaluator = ModelEvaluator(task_type='classification')
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']

    else:  # regression
        models = {
            'Linear Regression': LinearRegressionModel(),
            'Ridge': RidgeRegressionModel(random_state=42),
            'Random Forest': RandomForestRegressorModel(n_estimators=100, random_state=42),
            'SVM': SVMRegressorModel(random_state=42),
            'Gradient Boosting': GradientBoostingRegressorModel(n_estimators=100, random_state=42)
        }
        evaluator = ModelEvaluator(task_type='regression')
        scoring_metrics = ['neg_mean_squared_error', 'r2']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if problem_type == 'classification' else None
    )

    results = {}
    training_times = {}

    import time

    print(f"Comparing {len(models)} models...")
    print("-" * 50)

    for name, model in models.items():
        print(f"Training {name}...")

        # Time training
        start_time = time.time()
        model.train(X_train, y_train)
        training_time = time.time() - start_time
        training_times[name] = training_time

        # Evaluate model
        metrics = evaluator.evaluate(model, X_test, y_test)
        results[name] = metrics

        # Cross-validation scores
        cv_scores = cross_val_score(
            model.model, X, y, cv=5,
            scoring=scoring_metrics[0] if len(scoring_metrics) == 1 else 'accuracy'
        )
        results[name]['cv_mean'] = cv_scores.mean()
        results[name]['cv_std'] = cv_scores.std()
        results[name]['training_time'] = training_time

        print(f"  Completed in {training_time:.2f}s")

    return results, models, (X_train, X_test, y_train, y_test)

# Run systematic comparison
comparison_results, trained_models, data_splits = systematic_model_comparison(
    X_example, y_example, 'classification'
)

# Display results
print("\nModel Comparison Results:")
print("=" * 80)

results_df = pd.DataFrame(comparison_results).T
print(results_df[['accuracy', 'f1', 'cv_mean', 'training_time']].round(3))
```

### Performance Visualization

```python
def visualize_model_comparison(results, problem_type='classification'):
    """Visualize model comparison results."""

    # Convert results to DataFrame
    df = pd.DataFrame(results).T

    if problem_type == 'classification':
        metrics = ['accuracy', 'precision', 'recall', 'f1']
    else:
        metrics = ['r2', 'mse']

    # Create subplots
    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    # Performance metrics comparison
    for i, metric in enumerate(metrics):
        if i < len(axes) - 1 and metric in df.columns:
            df[metric].plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'{metric.capitalize()} Comparison')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].tick_params(axis='x', rotation=45)

    # Training time comparison
    if 'training_time' in df.columns:
        df['training_time'].plot(kind='bar', ax=axes[-1], color='orange')
        axes[-1].set_title('Training Time Comparison')
        axes[-1].set_ylabel('Time (seconds)')
        axes[-1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # Radar chart for top 3 models
    if problem_type == 'classification':
        # Select top 3 models by accuracy
        top_models = df.nlargest(3, 'accuracy')

        create_radar_chart(top_models, ['accuracy', 'precision', 'recall', 'f1'])

def create_radar_chart(data, metrics):
    """Create radar chart for model comparison."""

    from math import pi

    # Number of variables
    N = len(metrics)

    # Angles for each metric
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    colors = ['blue', 'red', 'green']

    for i, (model_name, model_data) in enumerate(data.iterrows()):
        values = [model_data[metric] for metric in metrics]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2,
                label=model_name, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

    # Add metric labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Comparison (Radar Chart)', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

    plt.tight_layout()
    plt.show()

# Visualize comparison results
visualize_model_comparison(comparison_results, 'classification')
```

## Performance Evaluation

### Comprehensive Evaluation Framework

```python
def comprehensive_evaluation(models, X_test, y_test, problem_type='classification'):
    """Comprehensive evaluation of trained models."""

    evaluation_results = {}

    for name, model in models.items():
        print(f"\nEvaluating {name}:")
        print("-" * 40)

        # Basic predictions
        predictions = model.predict(X_test)

        if problem_type == 'classification':
            # Classification metrics
            from sklearn.metrics import (accuracy_score, precision_score,
                                       recall_score, f1_score, roc_auc_score)

            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted')
            recall = recall_score(y_test, predictions, average='weighted')
            f1 = f1_score(y_test, predictions, average='weighted')

            print(f"Accuracy: {accuracy:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1-Score: {f1:.3f}")

            # ROC AUC for binary classification
            if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
                try:
                    probas = model.predict_proba(X_test)[:, 1]
                    roc_auc = roc_auc_score(y_test, probas)
                    print(f"ROC AUC: {roc_auc:.3f}")
                except:
                    roc_auc = None

            evaluation_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc if 'roc_auc' in locals() else None
            }

        else:  # regression
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            print(f"MSE: {mse:.3f}")
            print(f"RMSE: {rmse:.3f}")
            print(f"MAE: {mae:.3f}")
            print(f"R²: {r2:.3f}")

            evaluation_results[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }

    return evaluation_results

# Comprehensive evaluation
X_train, X_test, y_train, y_test = data_splits
comprehensive_results = comprehensive_evaluation(
    trained_models, X_test, y_test, 'classification'
)
```

### Model Ranking and Selection

```python
def rank_models(evaluation_results, ranking_criteria, problem_type='classification'):
    """Rank models based on multiple criteria."""

    if problem_type == 'classification':
        default_weights = {
            'accuracy': 0.3,
            'precision': 0.25,
            'recall': 0.25,
            'f1': 0.2
        }
    else:
        default_weights = {
            'r2': 0.4,
            'mse': -0.3,  # Negative because lower is better
            'mae': -0.3   # Negative because lower is better
        }

    # Use provided criteria or defaults
    weights = ranking_criteria if ranking_criteria else default_weights

    model_scores = {}

    for model_name, metrics in evaluation_results.items():
        score = 0
        for metric, weight in weights.items():
            if metric in metrics and metrics[metric] is not None:
                score += weight * metrics[metric]
        model_scores[model_name] = score

    # Sort by score (descending)
    ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

    print("Model Ranking:")
    print("-" * 40)
    for i, (model_name, score) in enumerate(ranked_models, 1):
        print(f"{i}. {model_name}: {score:.3f}")

    return ranked_models

# Rank models
ranking_criteria = {
    'accuracy': 0.4,
    'f1': 0.3,
    'precision': 0.2,
    'recall': 0.1
}

model_ranking = rank_models(comprehensive_results, ranking_criteria, 'classification')

# Select best model
best_model_name = model_ranking[0][0]
best_model = trained_models[best_model_name]

print(f"\nBest Model Selected: {best_model_name}")
```

## Computational Considerations

### Performance Profiling

```python
def profile_model_performance(models, X, y, n_runs=3):
    """Profile computational performance of models."""

    import time

    performance_profile = {}

    for name, model in models.items():
        print(f"Profiling {name}...")

        # Training time
        training_times = []
        for _ in range(n_runs):
            start_time = time.time()
            model.train(X, y)
            training_times.append(time.time() - start_time)

        # Prediction time
        prediction_times = []
        for _ in range(n_runs):
            start_time = time.time()
            _ = model.predict(X)
            prediction_times.append(time.time() - start_time)

        # Memory usage (simplified estimation)
        try:
            import sys
            model_size = sys.getsizeof(model)
        except:
            model_size = 0

        performance_profile[name] = {
            'avg_training_time': np.mean(training_times),
            'std_training_time': np.std(training_times),
            'avg_prediction_time': np.mean(prediction_times),
            'std_prediction_time': np.std(prediction_times),
            'model_size_bytes': model_size,
            'predictions_per_second': len(X) / np.mean(prediction_times)
        }

    return performance_profile

# Profile model performance
print("Profiling Model Performance...")
performance_data = profile_model_performance(trained_models, X_example, y_example)

# Display performance results
performance_df = pd.DataFrame(performance_data).T
print("\nPerformance Profile:")
print(performance_df[['avg_training_time', 'avg_prediction_time', 'predictions_per_second']].round(4))

# Visualize performance trade-offs
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(performance_df['avg_training_time'],
           [comparison_results[model]['accuracy'] for model in performance_df.index])
for i, model in enumerate(performance_df.index):
    plt.annotate(model, (performance_df.iloc[i]['avg_training_time'],
                        comparison_results[model]['accuracy']),
                xytext=(5, 5), textcoords='offset points')
plt.xlabel('Training Time (seconds)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Training Time')

plt.subplot(1, 2, 2)
plt.scatter(performance_df['predictions_per_second'],
           [comparison_results[model]['accuracy'] for model in performance_df.index])
for i, model in enumerate(performance_df.index):
    plt.annotate(model, (performance_df.iloc[i]['predictions_per_second'],
                        comparison_results[model]['accuracy']),
                xytext=(5, 5), textcoords='offset points')
plt.xlabel('Predictions per Second')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Prediction Speed')

plt.tight_layout()
plt.show()
```

## Model Selection Framework

### Automated Model Selection

```python
def automated_model_selection(X, y, business_requirements=None,
                             problem_type='classification', cv=5):
    """Automated model selection based on data characteristics and requirements."""

    # Step 1: Analyze dataset characteristics
    dataset_profile = profile_dataset(X, y)

    # Step 2: Filter models based on dataset characteristics
    if problem_type == 'classification':
        candidate_models = {
            'Logistic Regression': LogisticRegressionModel(random_state=42),
            'Random Forest': RandomForestClassifierModel(n_estimators=100, random_state=42),
            'SVM': SVMClassifierModel(random_state=42),
            'Gradient Boosting': GradientBoostingClassifierModel(n_estimators=100, random_state=42),
            'Naive Bayes': NaiveBayesModel(),
            'KNN': KNNClassifierModel(n_neighbors=5)
        }

    # Apply filtering based on dataset characteristics
    filtered_models = {}

    # Filter based on dataset size
    n_samples = dataset_profile['dimensions']['n_samples']

    if n_samples < 1000:
        # Small dataset - prefer simple models
        preferred = ['Logistic Regression', 'Naive Bayes', 'KNN']
        filtered_models = {name: model for name, model in candidate_models.items()
                          if name in preferred}
    elif n_samples > 10000:
        # Large dataset - can use complex models
        filtered_models = candidate_models
    else:
        # Medium dataset - exclude KNN (can be slow)
        filtered_models = {name: model for name, model in candidate_models.items()
                          if name != 'KNN'}

    # Filter based on interpretability requirements
    if business_requirements and business_requirements.get('interpretability', {}).get('level') == 'high':
        interpretable = ['Logistic Regression', 'Naive Bayes']
        filtered_models = {name: model for name, model in filtered_models.items()
                          if name in interpretable}

    print(f"Candidate models after filtering: {list(filtered_models.keys())}")

    # Step 3: Evaluate filtered models
    evaluation_results = {}

    for name, model in filtered_models.items():
        # Cross-validation
        scores = cross_val_score(model.model, X, y, cv=cv, scoring='accuracy')

        evaluation_results[name] = {
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'cv_scores': scores
        }

    # Step 4: Select best model
    best_model_name = max(evaluation_results.keys(),
                         key=lambda k: evaluation_results[k]['cv_mean'])

    best_model = filtered_models[best_model_name]

    # Step 5: Generate report
    selection_report = {
        'selected_model': best_model_name,
        'selection_reasoning': [],
        'performance': evaluation_results[best_model_name],
        'alternatives': {name: results for name, results in evaluation_results.items()
                        if name != best_model_name}
    }

    # Add reasoning
    if n_samples < 1000:
        selection_report['selection_reasoning'].append("Small dataset favors simple models")

    if business_requirements and business_requirements.get('interpretability', {}).get('level') == 'high':
        selection_report['selection_reasoning'].append("High interpretability requirement")

    selection_report['selection_reasoning'].append(
        f"Best cross-validation performance: {evaluation_results[best_model_name]['cv_mean']:.3f}"
    )

    return best_model, selection_report

# Run automated model selection
best_model_auto, selection_report = automated_model_selection(
    X_example, y_example, business_requirements=business_reqs
)

print("Automated Model Selection Report:")
print("=" * 50)
print(f"Selected Model: {selection_report['selected_model']}")
print(f"Performance: {selection_report['performance']['cv_mean']:.3f} ± {selection_report['performance']['cv_std']:.3f}")
print("\nSelection Reasoning:")
for reason in selection_report['selection_reasoning']:
    print(f"  - {reason}")

print("\nAlternative Models:")
for name, performance in selection_report['alternatives'].items():
    print(f"  {name}: {performance['cv_mean']:.3f} ± {performance['cv_std']:.3f}")
```

## Real-World Examples

### Example 1: Customer Churn Prediction

```python
def customer_churn_example():
    """Example: Customer churn prediction model selection."""

    print("Customer Churn Prediction - Model Selection")
    print("=" * 50)

    # Simulate customer churn dataset
    generator = ClassificationDataGenerator()
    X_churn, y_churn = generator.generate_imbalanced_classification(
        n_samples=5000,
        n_features=15,
        n_classes=2,
        weights=[0.8, 0.2],  # 80% no churn, 20% churn
        random_state=42
    )

    # Business requirements for churn prediction
    churn_requirements = {
        'interpretability': {'level': 'high'},  # Need to explain to business
        'class_imbalance': True,
        'precision_priority': True,  # False positives are costly
        'feature_importance': True   # Need to understand drivers
    }

    print("Dataset characteristics:")
    print(f"  Samples: {X_churn.shape[0]}")
    print(f"  Features: {X_churn.shape[1]}")
    print(f"  Class distribution: {np.bincount(y_churn)}")
    print(f"  Imbalance ratio: {np.bincount(y_churn)[0] / np.bincount(y_churn)[1]:.1f}:1")

    # Select appropriate models for churn prediction
    churn_models = {
        'Logistic Regression': LogisticRegressionModel(
            class_weight='balanced', random_state=42
        ),
        'Random Forest': RandomForestClassifierModel(
            class_weight='balanced', n_estimators=100, random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifierModel(
            n_estimators=100, random_state=42
        )
    }

    # Evaluate with appropriate metrics for imbalanced data
    X_train, X_test, y_train, y_test = train_test_split(
        X_churn, y_churn, test_size=0.2, random_state=42, stratify=y_churn
    )

    churn_results = {}

    for name, model in churn_models.items():
        model.train(X_train, y_train)
        predictions = model.predict(X_test)

        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

        precision = precision_score(y_test, predictions, pos_label=1)
        recall = recall_score(y_test, predictions, pos_label=1)
        f1 = f1_score(y_test, predictions, pos_label=1)

        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, probas)
        else:
            roc_auc = None

        churn_results[name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }

    print("\nChurn Prediction Results:")
    churn_df = pd.DataFrame(churn_results).T
    print(churn_df.round(3))

    # Recommendation
    best_churn_model = max(churn_results.keys(),
                          key=lambda k: churn_results[k]['precision'])

    print(f"\nRecommendation for churn prediction: {best_churn_model}")
    print("Reasoning: High precision prioritized for business cost considerations")

    return churn_results

# Run churn prediction example
churn_results = customer_churn_example()
```

## Best Practices

### Model Selection Checklist

```python
def model_selection_checklist():
    """Comprehensive checklist for model selection."""

    checklist = {
        'Problem Understanding': [
            '☐ Clearly defined problem type (classification/regression)',
            '☐ Business requirements documented',
            '☐ Success metrics identified',
            '☐ Constraints understood (time, interpretability, etc.)'
        ],
        'Data Analysis': [
            '☐ Dataset size and dimensionality analyzed',
            '☐ Data quality issues identified',
            '☐ Feature types and distributions understood',
            '☐ Class balance checked (for classification)',
            '☐ Target distribution analyzed (for regression)'
        ],
        'Model Evaluation': [
            '☐ Appropriate train/validation/test split',
            '☐ Cross-validation strategy defined',
            '☐ Relevant metrics selected',
            '☐ Baseline model established',
            '☐ Multiple algorithms compared'
        ],
        'Practical Considerations': [
            '☐ Computational requirements assessed',
            '☐ Training and prediction time measured',
            '☐ Model interpretability evaluated',
            '☐ Robustness to outliers tested',
            '☐ Scalability analyzed'
        ],
        'Validation and Testing': [
            '☐ Statistical significance tested',
            '☐ Model assumptions validated',
            '☐ Performance on held-out test set',
            '☐ Error analysis conducted',
            '☐ Edge cases considered'
        ],
        'Deployment Readiness': [
            '☐ Model serialization tested',
            '☐ Prediction pipeline validated',
            '☐ Monitoring strategy defined',
            '☐ Fallback plan established',
            '☐ Documentation completed'
        ]
    }

    return checklist

# Display checklist
checklist = model_selection_checklist()
print("Model Selection Checklist:")
print("=" * 50)

for category, items in checklist.items():
    print(f"\n{category}:")
    for item in items:
        print(f"  {item}")
```

## Summary

This comprehensive model selection tutorial covered:

✅ **Problem Analysis** - Understanding requirements and constraints  
✅ **Dataset Profiling** - Analyzing data characteristics that influence model choice  
✅ **Algorithm Taxonomy** - Systematic categorization of available algorithms  
✅ **Systematic Comparison** - Framework for comparing multiple models  
✅ **Performance Evaluation** - Comprehensive evaluation metrics and methods  
✅ **Computational Considerations** - Profiling and scalability analysis  
✅ **Automated Selection** - Framework for systematic model selection  
✅ **Real-world Examples** - Practical applications and case studies  
✅ **Best Practices** - Checklists and pitfall avoidance

### Key Takeaways

1. **Start with problem understanding** - Clear requirements guide algorithm choice
2. **Analyze your data** - Dataset characteristics strongly influence model suitability
3. **Compare systematically** - Use consistent evaluation methodology
4. **Consider practical constraints** - Real-world requirements matter as much as performance
5. **Use appropriate metrics** - Choose evaluation criteria that match your problem
6. **Avoid common pitfalls** - Be aware of data leakage and overfitting risks
7. **Document decisions** - Maintain clear reasoning for model choices

### Next Steps

- **[Hyperparameter Tuning](hyperparameter_tuning.md)** - Optimize your selected model
- **[Model Evaluation](model_evaluation.md)** - Deep dive into evaluation techniques
- **[Feature Engineering](feature_engineering.md)** - Improve model performance through better features

The framework and tools provided in this tutorial will help you make informed, systematic decisions about model selection for any machine learning project!
