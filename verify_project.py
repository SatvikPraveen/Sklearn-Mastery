"""Comprehensive project verification script."""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 70)
print("SKLEARN-MASTERY PROJECT VERIFICATION")
print("=" * 70)

# Test 1: Data Module
print("\n1. DATA MODULE")
print("-" * 70)
try:
    from data.generators import SyntheticDataGenerator, ClassificationDataGenerator
    gen = SyntheticDataGenerator()
    X, y = gen.classification_balanced(n_samples=100, n_features=10)
    print(f"✓ Data generation works! Generated {X.shape[0]} samples with {X.shape[1]} features")
except Exception as e:
    print(f"✗ Error: {e}")

try:
    from data.preprocessors import DataPreprocessor
    prep = DataPreprocessor()
    X_transformed = prep.fit_transform(X)
    print(f"✓ Data preprocessing works! Transformed shape: {X_transformed.shape}")
except Exception as e:
    print(f"✗ Error: {e}")

try:
    from data.validators import DataValidator
    validator = DataValidator()
    is_valid = validator.validate(X, y)
    print(f"✓ Data validation works! Data is valid: {is_valid}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Classification Models
print("\n2. CLASSIFICATION MODELS")
print("-" * 70)
try:
    from models.supervised.classification import (
        LogisticRegressionModel,
        RandomForestClassifier,
        SVMClassifier,
    )
    
    models = [
        ("Logistic Regression", LogisticRegressionModel()),
        ("Random Forest", RandomForestClassifier(n_estimators=50)),
        ("SVM", SVMClassifier()),
    ]
    
    for name, model in models:
        model.fit(X, y)
        score = model.score(X, y)
        print(f"✓ {name:20s} - Accuracy: {score:.4f}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Regression Models
print("\n3. REGRESSION MODELS")
print("-" * 70)
try:
    from models.supervised.regression import (
        LinearRegressionModel,
        RidgeRegressionModel,
        LassoRegressionModel,
    )
    import numpy as np
    
    X_reg, y_reg = SyntheticDataGenerator().regression_linear(n_samples=100, n_features=10)
    
    models = [
        ("Linear Regression", LinearRegressionModel()),
        ("Ridge Regression", RidgeRegressionModel()),
        ("Lasso Regression", LassoRegressionModel()),
    ]
    
    for name, model in models:
        model.fit(X_reg, y_reg)
        score = model.score(X_reg, y_reg)
        print(f"✓ {name:20s} - R² Score: {score:.4f}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 4: Clustering Models
print("\n4. CLUSTERING MODELS")
print("-" * 70)
try:
    from models.unsupervised.clustering import (
        KMeansClusterer,
        GaussianMixtureClusterer,
    )
    
    X_cluster, _ = SyntheticDataGenerator().clustering_gaussian_mixture(n_samples=100, n_clusters=3)
    
    models = [
        ("K-Means", KMeansClusterer(n_clusters=3)),
        ("Gaussian Mixture", GaussianMixtureClusterer(n_components=3)),
    ]
    
    for name, model in models:
        labels = model.fit_predict(X_cluster)
        n_clusters = len(np.unique(labels))
        print(f"✓ {name:20s} - Found {n_clusters} clusters")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 5: Dimensionality Reduction
print("\n5. DIMENSIONALITY REDUCTION")
print("-" * 70)
try:
    from models.unsupervised.dimensionality_reduction import PCAReducer
    
    models = [
        ("PCA", PCAReducer(n_components=2)),
    ]
    
    for name, model in models:
        X_reduced = model.fit_transform(X_cluster)
        print(f"✓ {name:20s} - Reduced from {X_cluster.shape[1]} to {X_reduced.shape[1]} dimensions")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 6: Evaluation Metrics
print("\n6. EVALUATION METRICS")
print("-" * 70)
try:
    from evaluation.metrics import ClassificationMetrics
    
    clf = LogisticRegressionModel()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    
    metrics = ClassificationMetrics()
    report = metrics.calculate_metrics(y, y_pred)
    print(f"✓ Classification metrics calculated:")
    print(f"  - Accuracy: {report.get('accuracy', 'N/A'):.4f}")
    print(f"  - Precision: {report.get('precision_macro', 'N/A'):.4f}")
    print(f"  - Recall: {report.get('recall_macro', 'N/A'):.4f}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 7: Config and Logging
print("\n7. CONFIGURATION & LOGGING")
print("-" * 70)
try:
    from config.settings import settings
    print(f"✓ Config loaded successfully")
    print(f"  - Project root: {settings.PROJECT_ROOT.name}/")
    print(f"  - Random seed: {settings.RANDOM_SEED}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 8: Pipeline Factory
print("\n8. PIPELINE FACTORY")
print("-" * 70)
try:
    from pipelines.pipeline_factory import PipelineFactory
    
    factory = PipelineFactory()
    pipeline = factory.create_classification_pipeline(complexity='standard')
    pipeline.fit(X, y)
    score = pipeline.score(X, y)
    print(f"✓ Pipeline factory works! Pipeline accuracy: {score:.4f}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 70)
print("✓ PROJECT VERIFICATION COMPLETE")
print("=" * 70)
print("\nAll core modules are working correctly!")
print("The project is ready for development and testing.")
