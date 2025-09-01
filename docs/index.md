# Scikit-Learn Mastery Project Documentation

Welcome to the comprehensive documentation for the Scikit-Learn Mastery Project - an advanced educational framework demonstrating machine learning best practices, implementations, and techniques.

## Overview

This project provides a complete ecosystem for learning and applying machine learning concepts using scikit-learn and related libraries. It includes custom implementations, advanced techniques, and real-world examples across all major ML domains.

## Quick Start

```python
from src.models.supervised.classification import RandomForestClassifierModel
from src.data.generators import ClassificationDataGenerator
from src.utils.evaluation import ModelEvaluator

# Generate sample data
generator = ClassificationDataGenerator()
X, y = generator.generate_basic_classification()

# Train model
model = RandomForestClassifierModel(n_estimators=100, random_state=42)
model.train(X, y)

# Evaluate
evaluator = ModelEvaluator(task_type='classification')
metrics = evaluator.evaluate(model, X, y)
print(f"Accuracy: {metrics['accuracy']:.3f}")
```

## Documentation Structure

### 📚 [Algorithm Guides](algorithm_guides/)

Detailed explanations and implementations of machine learning algorithms:

- [Classification Algorithms](algorithm_guides/classification.md)
- [Regression Algorithms](algorithm_guides/regression.md)
- [Clustering Algorithms](algorithm_guides/clustering.md)
- [Dimensionality Reduction](algorithm_guides/dimensionality_reduction.md)
- [Ensemble Methods](algorithm_guides/ensemble_methods.md)

### 🔧 [API Reference](api_reference/)

Complete API documentation for all modules:

- [Data Generation](api_reference/data.md)
- [Preprocessing](api_reference/preprocessing.md)
- [Models](api_reference/models.md)
- [Pipelines](api_reference/pipelines.md)
- [Evaluation](api_reference/evaluation.md)
- [Visualization](api_reference/visualization.md)

### 📖 [Tutorials](tutorials/)

Step-by-step guides for common machine learning tasks:

- [Getting Started](tutorials/getting_started.md)
- [Data Preprocessing](tutorials/data_preprocessing.md)
- [Model Selection](tutorials/model_selection.md)
- [Feature Engineering](tutorials/feature_engineering.md)
- [Hyperparameter Tuning](tutorials/hyperparameter_tuning.md)
- [Model Evaluation](tutorials/model_evaluation.md)
- [Building Pipelines](tutorials/building_pipelines.md)
- [Advanced Techniques](tutorials/advanced_techniques.md)

## Key Features

### 🎯 **Comprehensive Coverage**

- **Supervised Learning**: Classification and regression with 10+ algorithms
- **Unsupervised Learning**: Clustering and dimensionality reduction techniques
- **Ensemble Methods**: Voting, bagging, boosting, stacking, and blending
- **Custom Transformers**: 15+ preprocessing and feature engineering tools

### 🚀 **Advanced Capabilities**

- **Pipeline Integration**: Seamless sklearn pipeline compatibility
- **Model Persistence**: Save and load trained models
- **Cross-Validation**: Comprehensive validation strategies
- **Hyperparameter Tuning**: Grid search and random search implementations

### 📊 **Visualization & Analysis**

- **Performance Metrics**: 20+ evaluation metrics
- **Learning Curves**: Training progress analysis
- **Feature Importance**: Multiple importance calculation methods
- **Model Comparison**: Statistical significance testing

### 🔬 **Educational Focus**

- **Best Practices**: Industry-standard coding patterns
- **Documentation**: Comprehensive docstrings and examples
- **Testing**: 95%+ test coverage with pytest
- **Real-World Examples**: Practical use cases and datasets

## Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data generation and loading
│   ├── preprocessing/     # Data preprocessing utilities
│   ├── models/           # ML model implementations
│   ├── pipelines/        # Pipeline components
│   └── utils/           # Evaluation and visualization
├── tests/               # Comprehensive test suite
├── notebooks/          # Jupyter tutorials (7 notebooks)
├── examples/           # Real-world examples
├── docs/              # Documentation (this directory)
└── results/           # Output directory
```

## Installation

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Quick Install

```bash
# Clone the repository
git clone <repository-url>
cd scikit-learn-mastery

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Dependencies

```
scikit-learn >= 1.0.0
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
jupyter >= 1.0.0
pytest >= 6.0.0
```

## Usage Examples

### Classification Pipeline

```python
from sklearn.pipeline import Pipeline
from src.pipelines.custom_transformers import ScalerTransformer, FeatureSelector
from src.models.supervised.classification import RandomForestClassifierModel

# Create pipeline
pipeline = Pipeline([
    ('scaler', ScalerTransformer(method='standard')),
    ('selector', FeatureSelector(method='univariate', k=10)),
    ('classifier', RandomForestClassifierModel(n_estimators=100))
])

# Train and evaluate
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
```

### Ensemble Learning

```python
from src.models.ensemble.ensemble_methods import VotingEnsemble
from src.models.supervised.classification import *

# Create ensemble
estimators = [
    ('rf', RandomForestClassifierModel(n_estimators=50)),
    ('svm', SVMClassifierModel(kernel='rbf')),
    ('nb', NaiveBayesModel())
]

ensemble = VotingEnsemble(estimators=estimators, voting='soft')
ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)
```

### Feature Engineering

```python
from src.pipelines.custom_transformers import *

# Advanced preprocessing pipeline
preprocessor = Pipeline([
    ('imputer', MissingValueImputer(strategy='mean')),
    ('outlier_remover', OutlierRemover(method='iqr')),
    ('encoder', CategoryEncoder(method='onehot')),
    ('poly_features', PolynomialFeatureGenerator(degree=2)),
    ('feature_selector', FeatureSelector(method='rfe', k=15)),
    ('scaler', ScalerTransformer(method='robust'))
])
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](contributing.md) for details on:

- Code style and standards
- Testing requirements
- Documentation guidelines
- Pull request process

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Acknowledgments

Built with:

- [scikit-learn](https://scikit-learn.org/) - Machine learning library
- [NumPy](https://numpy.org/) - Numerical computing
- [Pandas](https://pandas.pydata.org/) - Data manipulation
- [Matplotlib](https://matplotlib.org/) - Plotting library
- [Seaborn](https://seaborn.pydata.org/) - Statistical visualization

## Support

- **Documentation**: [Full documentation](https://your-docs-url.com)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Examples**: See the [examples/](../examples/) directory

---

_This documentation is continuously updated. Last updated: 2024_
