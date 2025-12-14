# 🚀 Sklearn-Mastery: Comprehensive Scikit-Learn Learning Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 611](https://img.shields.io/badge/tests-611-brightgreen.svg)](tests/)
[![Test Coverage](https://img.shields.io/badge/coverage-%3E90%25-brightgreen.svg)](#-testing--quality-assurance)

> **A comprehensive one-stop learning solution for mastering Scikit-Learn: understand data generation, pipeline construction, algorithm implementations, model evaluation, and production-ready patterns.**

## 📋 Table of Contents

- [🎯 Project Vision](#-project-vision)
- [🌟 Key Highlights](#-key-highlights)
- [📁 Project Architecture](#-project-architecture)
- [🚀 Quick Start Guide](#-quick-start-guide)
- [🎮 Interactive Notebooks](#-interactive-notebooks)
- [🎯 Core Features](#-core-features)
- [🧪 Testing & Quality Assurance](#-testing--quality-assurance)
- [📚 Documentation & Learning Resources](#-documentation--learning-resources)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🎯 **Project Vision**

This project is a **comprehensive learning resource** for mastering Scikit-Learn through:

🔍 **Complete Framework Understanding** - Learn how to build and structure ML pipelines from data to deployment  
📊 **Algorithm Deep Dive** - Explore 50+ scikit-learn algorithms across classification, regression, clustering, and dimensionality reduction  
🔧 **Custom Implementations** - Understand transformer patterns and custom pipeline development  
📈 **Production Patterns** - Learn evaluation metrics, statistical testing, and deployment best practices  
🧪 **Hands-On Practice** - 7 interactive notebooks + 611 comprehensive tests for learning validation

---

## 🌟 **Key Highlights**

### ✨ **Learning-Focused Framework**

| Component | Description | Key Features |
|-----------|-------------|--------------|
| **50+ ML Algorithms** | Supervised, unsupervised, ensemble methods | Classification, regression, clustering, dimensionality reduction |
| **Custom Transformers** | sklearn-compatible pipeline components | Feature engineering, preprocessing, data validation |
| **Data Generation** | Algorithm-specific synthetic datasets | Perfect for testing, learning, and validation |
| **7 Interactive Notebooks** | Hands-on learning from data to deployment | Progressive complexity from basics to advanced |
| **611 Unit Tests** | Comprehensive test coverage for validation | Learn from test patterns and expected behaviors |
| **Advanced Evaluation** | Statistical metrics and visualization | Hypothesis testing, learning curves, model interpretation |

### 📚 **What You'll Learn**

```
✅ Building sklearn pipelines with custom transformers
✅ Creating synthetic datasets for algorithm testing
✅ Implementing supervised learning (classification & regression)
✅ Unsupervised learning (clustering & dimensionality reduction)
✅ Ensemble methods and meta-learning
✅ Hyperparameter tuning and model selection
✅ Statistical evaluation and significance testing
✅ Production-ready patterns and deployment considerations
```

---

## 📁 **Project Architecture**

<details>
<summary><strong>🏗️ Detailed Project Structure (Click to expand)</strong></summary>

```
sklearn-mastery/
├── 📦 src/                          # Core learning framework (~9,900 LoC)
│   ├── 🔢 data/                     # Data engineering
│   │   ├── generators.py            # Synthetic data generation (15+ methods)
│   │   ├── preprocessors.py         # Preprocessing utilities
│   │   └── validators.py            # Data validation
│   ├── 🔧 pipelines/                # Pipeline & transformation layer
│   │   ├── custom_transformers.py   # 20+ sklearn transformers
│   │   ├── pipeline_factory.py      # Pipeline creation patterns
│   │   ├── model_selection.py       # Model selection utilities
│   │   └── feature_union.py         # Feature composition patterns
│   ├── 🤖 models/                   # Algorithm implementations
│   │   ├── supervised/              # Classification & regression (30+ models)
│   │   ├── unsupervised/            # Clustering & dimensionality (25+ models)
│   │   └── ensemble/                # Ensemble methods (5 types)
│   ├── 📊 evaluation/               # Model evaluation framework
│   │   ├── metrics.py               # Evaluation metrics
│   │   ├── statistical_tests.py     # Hypothesis testing
│   │   ├── visualization.py         # Results visualization
│   │   └── utils.py                 # Evaluation utilities
│   ├── 🔐 preprocessing/            # Preprocessing wrapper
│   └── 🛠️ utils/                    # Utilities & helpers
├── 📓 notebooks/                    # 7 Interactive Jupyter notebooks
│   ├── 01_data_generation_showcase.ipynb
│   ├── 02_preprocessing_pipelines.ipynb
│   ├── 03_supervised_learning.ipynb
│   ├── 04_unsupervised_learning.ipynb
│   ├── 05_ensemble_methods.ipynb
│   ├── 06_model_selection_tuning.ipynb
│   └── 07_advanced_techniques.ipynb
├── 🧪 tests/                        # 611 comprehensive tests
│   ├── test_data/
│   ├── test_models/
│   ├── test_pipelines/
│   └── test_utils/
├── 📚 docs/                         # Documentation
│   ├── algorithm_guides/            # Algorithm-specific guides
│   ├── tutorials/                   # Step-by-step tutorials
│   └── examples/                    # Code examples
├── ⚙️ config/                       # Configuration management
├── 📄 setup.py                      # Package installation
├── 📋 requirements.txt              # Dependencies
└── 🧪 conftest.py                  # Pytest configuration
```

</details>

---

## 🚀 **Quick Start Guide**

### **Prerequisites**

- Python 3.8+ 🐍
- 8GB+ RAM recommended 💾
- Git version control 🔧

### **Installation Options**

<details>
<summary><strong>🔧 Standard Installation</strong></summary>

```bash
# 1. Clone the repository
git clone https://github.com/SatvikPraveen/sklearn-mastery.git
cd sklearn-mastery

# 2. Create virtual environment
python -m venv sklearn_env
source sklearn_env/bin/activate  # Windows: sklearn_env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install package in development mode
pip install -e .

# 5. Verify installation
python -c "import src; print('✅ Installation successful!')"

# 6. Test with a quick example
python -c "
from src.data.generators import SyntheticDataGenerator
gen = SyntheticDataGenerator()
X, y = gen.classification_complexity_spectrum('medium')
print(f'✅ Generated dataset: {X.shape[0]} samples, {X.shape[1]} features')
"
```

</details>

<details>
<summary><strong>🐳 Docker Installation</strong></summary>

```bash
# 1. Clone repository
git clone https://github.com/SatvikPraveen/sklearn-mastery.git
cd sklearn-mastery

# 2. Build Docker image
docker build -t sklearn-mastery .

# 3. Run container with Jupyter
docker run -p 8888:8888 -v $(pwd):/workspace sklearn-mastery

# 4. Access Jupyter at http://localhost:8888
```

</details>

<details>
<summary><strong>📦 Conda Installation</strong></summary>

```bash
# 1. Clone repository
git clone https://github.com/SatvikPraveen/sklearn-mastery.git
cd sklearn-mastery

# 2. Create conda environment
conda create -n sklearn-mastery python=3.9
conda activate sklearn-mastery

# 3. Install dependencies
conda install --file requirements.txt
pip install -e .

# 4. Launch Jupyter
jupyter notebook
```

</details>

<details>
<summary><strong>⚡ Minimal Installation</strong></summary>

```bash
# For basic functionality only
pip install -r requirements-minimal.txt
```

</details>

### **🔥 Why This Project?**

| Aspect | Traditional Learning | **Sklearn-Mastery** |
|--------|----------------------|---------------------|
| **Focus** | Theory & concepts | Hands-on sklearn implementation |
| **Data Generation** | Use static datasets | Create algorithm-specific synthetic data |
| **Pipeline Building** | Simple sklearn examples | Production-ready patterns + custom transformers |
| **Model Evaluation** | Basic metrics | Statistical testing + visualization |
| **Real Examples** | Single use case | Multiple patterns across algorithms |
| **Learning Path** | Self-directed | Structured notebooks + tests |
| **Test Coverage** | Rarely present | 611 tests validating behaviors |

### **30-Second Demo**

```python
from src.data.generators import SyntheticDataGenerator
from src.pipelines.pipeline_factory import PipelineFactory
from src.evaluation.metrics import ModelEvaluator

# 🎯 Generate algorithm-optimized data
generator = SyntheticDataGenerator(random_state=42)
X, y = generator.classification_complexity_spectrum('medium')

# 🔧 Create advanced pipeline with auto-tuning
factory = PipelineFactory()
pipeline = factory.create_pipeline_with_auto_tuning(
    algorithm='random_forest',
    task_type='classification',
    preprocessing_level='advanced'
)

# 📊 Train and evaluate
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)
print(f"🎉 Model accuracy: {score:.3f}")
```

---

## 🎮 **Interactive Notebooks**

Explore the project through **7 comprehensive Jupyter notebooks**:

| Notebook                                                                       | Focus Area                  | Key Features                                                       |
| ------------------------------------------------------------------------------ | --------------------------- | ------------------------------------------------------------------ |
| **[01_data_generation_showcase](notebooks/01_data_generation_showcase.ipynb)** | Data Engineering            | 15+ synthetic data generators, visualization, complexity analysis  |
| **[02_preprocessing_pipelines](notebooks/02_preprocessing_pipelines.ipynb)**   | Data Preprocessing          | Custom transformers, pipeline patterns, strategy comparisons       |
| **[03_supervised_learning](notebooks/03_supervised_learning.ipynb)**           | Supervised ML               | Classification/regression, hyperparameter tuning, model comparison |
| **[04_unsupervised_learning](notebooks/04_unsupervised_learning.ipynb)**       | Unsupervised ML             | Clustering, dimensionality reduction, anomaly detection            |
| **[05_ensemble_methods](notebooks/05_ensemble_methods.ipynb)**                 | Ensemble Learning           | Voting, stacking, blending, diversity analysis                     |
| **[06_model_selection_tuning](notebooks/06_model_selection_tuning.ipynb)**     | Hyperparameter Optimization | Grid search, random search, Bayesian optimization                  |
| **[07_advanced_techniques](notebooks/07_advanced_techniques.ipynb)**           | Production ML               | SHAP interpretation, model serialization, deployment               |

---

## 🎯 **Core Features**

### 🔧 **Advanced Pipeline System**

<details>
<summary><strong>Custom Transformers Library</strong></summary>

```python
from src.pipelines.custom_transformers import *

# 🔍 Intelligent outlier detection
outlier_remover = OutlierRemover(
    methods=['isolation_forest', 'lof', 'zscore'],
    contamination=0.1
)

# ⚡ Feature interaction creation
interaction_creator = FeatureInteractionCreator(
    interaction_types=['polynomial', 'pairwise', 'log_transform'],
    degree=2
)

# 🏷️ Domain-specific encoding
encoder = DomainSpecificEncoder(
    categorical_strategy='target_encoding',
    numerical_strategy='quantile_uniform'
)

# 🔄 Advanced imputation
imputer = AdvancedImputer(
    strategy='iterative',
    estimator='random_forest'
)
```

</details>

<details>
<summary><strong>Pipeline Factory Patterns</strong></summary>

```python
from src.pipelines.pipeline_factory import PipelineFactory

factory = PipelineFactory(random_state=42)

# 🚀 Speed-optimized pipeline
minimal_pipeline = factory.create_classification_pipeline(
    algorithm='logistic_regression',
    preprocessing_level='minimal',  # Basic scaling only
    n_jobs=-1
)

# ⚖️ Balanced performance pipeline
standard_pipeline = factory.create_classification_pipeline(
    algorithm='random_forest',
    preprocessing_level='standard',  # Standard preprocessing
    feature_selection=True,
    handle_imbalance=False
)

# 🎯 Maximum performance pipeline
advanced_pipeline = factory.create_classification_pipeline(
    algorithm='gradient_boosting',
    preprocessing_level='advanced',  # Full preprocessing suite
    feature_selection=True,
    handle_imbalance=True,  # SMOTE integration
    feature_engineering=True
)

# 🏭 Production pipeline with monitoring
production_pipeline = factory.create_production_pipeline(
    algorithm='xgboost',
    enable_monitoring=True,
    cache_transformations=True,
    parallel_preprocessing=True
)
```

</details>

### 🧠 **Intelligent Data Generation**

<details>
<summary><strong>Algorithm-Specific Datasets</strong></summary>

```python
from src.data.generators import SyntheticDataGenerator

generator = SyntheticDataGenerator(random_state=42)

# 📊 Perfect for Linear/Ridge/Lasso comparison
X_reg, y_reg, true_coef = generator.regression_with_collinearity(
    n_samples=1000,
    collinear_groups=[(0,1,2), (5,6,7,8)],  # Multicollinear features
    noise_variance=0.1,
    sparsity=0.3  # Sparse true coefficients
)

# 🎯 Ideal for SVM vs Neural Network comparison
X_nonlinear, y_nonlinear = generator.classification_complexity_spectrum('high')

# 🔍 Perfect for clustering algorithm comparison
X_blobs = generator.clustering_blobs_with_noise(
    n_clusters=4,
    outlier_fraction=0.1,
    cluster_std_range=(0.5, 2.0)
)

# 📈 High-dimensional sparse data for Naive Bayes
X_sparse, y_sparse = generator.high_dimensional_sparse_data(
    n_features=10000,
    sparsity=0.95,
    informative_features=100
)

# ⏰ Time series data for forecasting
ts_data = generator.time_series_with_seasonality(
    n_periods=1000,
    seasonal_periods=[7, 30, 365],  # Weekly, monthly, yearly
    trend_type='polynomial',
    noise_level=0.1
)
```

</details>

### 📊 **Comprehensive Evaluation Framework**

---

## 📚 **Documentation & Learning Resources**

### **Available Resources**

- 📖 **Algorithm Guides** - `docs/algorithm_guides/` - Deep dives into classification, regression, clustering, dimensionality reduction, and ensemble methods
- 🎓 **Tutorials** - `docs/tutorials/` - Step-by-step learning paths for getting started and model selection
- 📊 **Interactive Notebooks** - `notebooks/` - 7 hands-on Jupyter notebooks progressing from basics to advanced techniques
- 💻 **Examples** - `src/` - Production-ready code patterns and implementations

### **Learning Path**

**Beginner → Intermediate → Advanced**

1. **Start Here**: `notebooks/01_data_generation_showcase.ipynb` - Understand synthetic data
2. **Preprocessing**: `notebooks/02_preprocessing_pipelines.ipynb` - Build sklearn pipelines
3. **Supervised Learning**: `notebooks/03_supervised_learning.ipynb` - Classification and regression
4. **Unsupervised Learning**: `notebooks/04_unsupervised_learning.ipynb` - Clustering and dimensionality reduction
5. **Ensembles**: `notebooks/05_ensemble_methods.ipynb` - Combine multiple models
6. **Tuning**: `notebooks/06_model_selection_tuning.ipynb` - Hyperparameter optimization
7. **Advanced**: `notebooks/07_advanced_techniques.ipynb` - Production patterns and deployment

---

---

## 🤝 **Contributing**

We welcome contributions from the community! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute improvements, bug fixes, and new features.

### **Quick Start for Contributors**

```bash
# Clone the repository
git clone https://github.com/SatvikPraveen/sklearn-mastery.git
cd sklearn-mastery

# Create development environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Make your changes, test, and submit a PR
```

---

## 📄 **License**

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

Special thanks to:
- 🧠 **Scikit-learn Team** - For the incredible ML library
- 🌟 **Open Source Community** - For tools and inspiration
- 🤝 **Contributors** - For improvements and feedback

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

**🤖 Happy Machine Learning! 📊**

_Built with ❤️ by [Satvik Praveen](https://github.com/SatvikPraveen) and the community._

</div>
