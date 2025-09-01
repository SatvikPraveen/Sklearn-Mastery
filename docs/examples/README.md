# MyML Examples

**File Location:** `docs/examples/README.md`

This directory contains practical examples demonstrating MyML usage across different domains and use cases.

## Directory Structure

```
examples/
├── README.md                    # This file
├── basic_usage/                 # Basic algorithm usage
│   ├── classification_example.py
│   ├── regression_example.py
│   └── clustering_example.py
├── advanced_workflows/          # Complex ML workflows
│   ├── feature_engineering.py
│   ├── model_comparison.py
│   ├── hyperparameter_tuning.py
│   └── ensemble_methods.py
├── real_world_projects/         # Complete project examples
│   ├── credit_scoring/
│   ├── customer_segmentation/
│   ├── demand_forecasting/
│   └── anomaly_detection/
├── deployment_examples/         # Deployment scenarios
│   ├── flask_api/
│   ├── batch_processing/
│   └── cloud_deployment/
└── notebooks/                   # Jupyter notebooks
    ├── getting_started.ipynb
    ├── data_preprocessing.ipynb
    └── model_evaluation.ipynb
```

## Quick Start Examples

### Basic Classification

```python
# examples/basic_usage/classification_example.py
from myml.algorithms import RandomForestClassifier
from myml.data import load_dataset
from myml.evaluation import classification_report
from myml.utils import train_test_split

# Load dataset
X, y = load_dataset('iris', return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)
```

### Pipeline Example

```python
# examples/advanced_workflows/feature_engineering.py
from myml.pipelines import Pipeline
from myml.preprocessing import StandardScaler, SelectKBest
from myml.algorithms import SVM

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=5)),
    ('classifier', SVM(kernel='rbf'))
])

# Train pipeline
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

## Complete Project Examples

Each project includes:

- Data loading and exploration
- Feature engineering
- Model training and evaluation
- Results visualization
- Deployment considerations

### Credit Scoring Project

**Files:** `examples/real_world_projects/credit_scoring/`

- `credit_scoring.py` - Main analysis script
- `data_preprocessing.py` - Data cleaning utilities
- `model_evaluation.py` - Comprehensive evaluation
- `README.md` - Project documentation

### Customer Segmentation

**Files:** `examples/real_world_projects/customer_segmentation/`

- `segmentation_analysis.py` - Clustering analysis
- `visualization.py` - Segment visualization
- `business_insights.py` - Business metric analysis
- `README.md` - Project guide

## Running Examples

### Prerequisites

```bash
# Install MyML with examples dependencies
pip install myml[examples]

# Or install additional packages
pip install jupyter matplotlib seaborn plotly
```

### Running Scripts

```bash
# Run basic examples
python examples/basic_usage/classification_example.py

# Run advanced workflows
python examples/advanced_workflows/model_comparison.py

# Launch Jupyter notebooks
jupyter notebook examples/notebooks/
```

### Running Complete Projects

```bash
# Navigate to project directory
cd examples/real_world_projects/credit_scoring/

# Run main analysis
python credit_scoring.py

# View results
open results/model_comparison.html
```

## Example Categories

### 1. Basic Usage (`basic_usage/`)

- Single algorithm usage
- Simple train/test workflows
- Basic evaluation metrics

### 2. Advanced Workflows (`advanced_workflows/`)

- Pipeline construction
- Cross-validation strategies
- Hyperparameter optimization
- Ensemble methods

### 3. Real-World Projects (`real_world_projects/`)

- End-to-end ML projects
- Domain-specific applications
- Business problem solving
- Complete analysis workflows

### 4. Deployment Examples (`deployment_examples/`)

- API development
- Batch processing
- Cloud deployment
- Model monitoring

### 5. Jupyter Notebooks (`notebooks/`)

- Interactive tutorials
- Step-by-step guides
- Visualization examples
- Educational content

## Contributing Examples

To contribute new examples:

1. Choose appropriate directory
2. Follow existing code style
3. Include comprehensive documentation
4. Add requirements.txt if needed
5. Test examples thoroughly

### Example Template

```python
"""
Title: Brief description of the example
Author: Your Name
Date: YYYY-MM-DD

Description:
Detailed description of what this example demonstrates,
the problem it solves, and key learning objectives.

Requirements:
- myml>=1.0.0
- pandas>=1.3.0
- matplotlib>=3.0.0

Usage:
python example_name.py
"""

# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# MyML imports
from myml.algorithms import YourAlgorithm
from myml.evaluation import your_metric
from myml.utils import helper_function

def main():
    """Main example function."""

    # Step 1: Data preparation
    print("Step 1: Loading and preparing data...")
    # Your data loading code

    # Step 2: Model training
    print("Step 2: Training model...")
    # Your model training code

    # Step 3: Evaluation
    print("Step 3: Evaluating results...")
    # Your evaluation code

    # Step 4: Visualization (if applicable)
    print("Step 4: Creating visualizations...")
    # Your visualization code

    print("Example completed successfully!")

if __name__ == '__main__':
    main()
```

## Best Practices for Examples

1. **Clear Documentation**: Each example should have clear comments explaining the purpose and methodology

2. **Reproducible Results**: Use random seeds for consistent outputs

3. **Error Handling**: Include appropriate error handling for common issues

4. **Performance Considerations**: Note computational requirements for large examples

5. **Real Data**: Use realistic datasets when possible

6. **Business Context**: Explain the business problem being solved

7. **Multiple Approaches**: Show different ways to solve the same problem

## Getting Help

- Check existing examples for similar use cases
- Read the main documentation at `/docs/`
- Open issues for questions or improvements
- Contribute your own examples to help others

## License

All examples are provided under the same license as the main MyML project.
