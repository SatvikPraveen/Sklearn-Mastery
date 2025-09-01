# API Reference

Complete API documentation for all modules in the Scikit-Learn Mastery Project.

## Module Overview

| Module                            | Description                            | Key Classes                                                                         |
| --------------------------------- | -------------------------------------- | ----------------------------------------------------------------------------------- |
| [Data](data.md)                   | Data generation and loading utilities  | `ClassificationDataGenerator`, `RegressionDataGenerator`, `ClusteringDataGenerator` |
| [Preprocessing](preprocessing.md) | Data preprocessing and validation      | `DataPreprocessor`, `DataValidator`, `DataSplitter`                                 |
| [Models](models.md)               | Machine learning model implementations | All supervised and unsupervised models                                              |
| [Pipelines](pipelines.md)         | Pipeline components and transformers   | Custom transformers and pipeline utilities                                          |
| [Evaluation](evaluation.md)       | Model evaluation and metrics           | `ModelEvaluator`, `CrossValidator`, `MetricsCalculator`                             |
| [Visualization](visualization.md) | Plotting and visualization utilities   | Various visualizer classes                                                          |

## Quick Reference

### Common Imports

```python
# Data generation
from src.data.generators import ClassificationDataGenerator, RegressionDataGenerator

# Models
from src.models.supervised.classification import RandomForestClassifierModel
from src.models.supervised.regression import LinearRegressionModel
from src.models.unsupervised.clustering import KMeansModel

# Preprocessing
from src.preprocessing.preprocessor import DataPreprocessor
from src.pipelines.custom_transformers import ScalerTransformer, FeatureSelector

# Evaluation
from src.utils.evaluation import ModelEvaluator, CrossValidator
from src.utils.visualization import DataVisualizer, ModelVisualizer
```

### Basic Usage Pattern

```python
# 1. Generate or load data
generator = ClassificationDataGenerator()
X, y = generator.generate_basic_classification()

# 2. Preprocess data
preprocessor = DataPreprocessor()
X_processed = preprocessor.fit_transform(X)

# 3. Train model
model = RandomForestClassifierModel(n_estimators=100, random_state=42)
model.train(X_processed, y)

# 4. Evaluate model
evaluator = ModelEvaluator(task_type='classification')
metrics = evaluator.evaluate(model, X_processed, y)

# 5. Visualize results
visualizer = ModelVisualizer()
fig, ax = visualizer.plot_feature_importance(model, feature_names)
```

## Navigation

- **[Data Module](data.md)** - Data generation, loading, and splitting
- **[Preprocessing Module](preprocessing.md)** - Data preprocessing and validation
- **[Models Module](models.md)** - All machine learning models
- **[Pipelines Module](pipelines.md)** - Custom transformers and pipelines
- **[Evaluation Module](evaluation.md)** - Model evaluation and metrics
- **[Visualization Module](visualization.md)** - Plotting and visualization

## Conventions

### Method Naming

- `fit()` - Fit transformer/model to data
- `transform()` - Transform data using fitted transformer
- `fit_transform()` - Fit and transform in one step
- `train()` - Train machine learning model
- `predict()` - Make predictions
- `evaluate()` - Evaluate model performance

### Parameter Naming

- `X` - Feature matrix (2D array-like)
- `y` - Target vector (1D array-like)
- `random_state` - Random seed for reproducibility
- `n_estimators` - Number of estimators in ensemble methods
- `max_iter` - Maximum number of iterations

### Return Types

- Model methods return appropriate data types (arrays, dictionaries)
- Evaluation methods return dictionaries with metric names as keys
- Visualization methods return matplotlib Figure and Axes objects
