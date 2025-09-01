"""
Updated tests for data generation utilities.

This module contains comprehensive tests for the SyntheticDataGenerator
in the src/data/generators.py module.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from data.generators import SyntheticDataGenerator


class TestSyntheticDataGenerator:
    """Test SyntheticDataGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create a generator instance for testing."""
        return SyntheticDataGenerator(random_state=42)
    
    def test_initialization(self, generator):
        """Test SyntheticDataGenerator initialization."""
        assert generator.random_state == 42
        assert hasattr(generator, 'logger')
    
    def test_linear_regression_data(self, generator):
        """Test linear regression data generation."""
        X, y = generator.linear_regression_data(n_samples=100, n_features=20)
        
        assert X.shape == (100, 20)
        assert y.shape == (100,)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
    
    def test_linear_regression_with_params(self, generator):
        """Test linear regression with different parameters."""
        X, y = generator.linear_regression_data(
            n_samples=150,
            n_features=10,
            noise_level=0.5,
            effective_rank=5,
            bias=2.0
        )
        
        assert X.shape == (150, 10)
        assert y.shape == (150,)
        # Check that bias is applied (mean should be around bias value)
        assert abs(y.mean() - 2.0) < 1.0
    
    def test_regression_with_collinearity(self, generator):
        """Test regression data with multicollinearity."""
        X, y, true_coef = generator.regression_with_collinearity(
            n_samples=100,
            n_features=10,
            collinear_groups=[(0, 1, 2), (5, 6)]
        )
        
        assert X.shape == (100, 10)
        assert y.shape == (100,)
        assert true_coef.shape == (10,)
        
        # Check that collinearity exists
        corr_matrix = np.corrcoef(X.T)
        assert corr_matrix[0, 1] > 0.8  # Features 0 and 1 should be highly correlated
        assert corr_matrix[5, 6] > 0.8  # Features 5 and 6 should be highly correlated
        
        # Check that some coefficients are zero (sparsity)
        assert np.sum(true_coef == 0) > 0
    
    def test_classification_complexity_levels(self, generator):
        """Test classification data with different complexity levels."""
        complexities = ['linear', 'medium', 'high']
        
        for complexity in complexities:
            X, y = generator.classification_complexity_spectrum(
                complexity=complexity,
                n_samples=100,
                n_features=2
            )
            
            assert X.shape == (100, 2)
            assert y.shape == (100,)
            assert len(np.unique(y)) == 2
            assert set(np.unique(y)) <= {0, 1}
    
    def test_invalid_complexity_level(self, generator):
        """Test that invalid complexity level raises error."""
        with pytest.raises(ValueError, match="Unknown complexity level"):
            generator.classification_complexity_spectrum(complexity='invalid')
    
    def test_high_dimensional_sparse_data(self, generator):
        """Test high-dimensional sparse data generation."""
        X, y = generator.high_dimensional_sparse_data(
            n_samples=100,
            n_features=1000,
            sparsity=0.95,
            n_classes=3
        )
        
        assert X.shape == (100, 1000)
        assert y.shape == (100,)
        assert len(np.unique(y)) == 3
        
        # Check that data is positive (good for Multinomial Naive Bayes)
        assert np.all(X >= 0)
        
        # Check sparsity (most features should be zero or near-zero)
        # Since we take absolute value, we check for very small values
        small_values = np.sum(X < 0.1, axis=1)
        avg_small_ratio = np.mean(small_values / X.shape[1])
        assert avg_small_ratio > 0.5  # At least 50% should be small
    
    def test_clustering_blobs_with_noise(self, generator):
        """Test clustering data with noise and outliers."""
        X = generator.clustering_blobs_with_noise(
            n_clusters=4,
            n_samples=1000,
            outlier_fraction=0.1,
            cluster_std=1.0
        )
        
        assert X.shape == (1000, 2)
        assert isinstance(X, np.ndarray)
        
        # Check that data contains some spread (outliers + clusters)
        x_range = X[:, 0].max() - X[:, 0].min()
        y_range = X[:, 1].max() - X[:, 1].min()
        assert x_range > 5  # Should have significant spread due to outliers
        assert y_range > 5
    
    def test_clustering_moons(self, generator):
        """Test moon-shaped clustering data."""
        X = generator.clustering_moons(n_samples=200, noise=0.1)
        
        assert X.shape == (200, 2)
        assert isinstance(X, np.ndarray)
    
    def test_hierarchical_clustering_data(self, generator):
        """Test hierarchical clustering data generation."""
        X = generator.hierarchical_clustering_data(
            n_samples=100,
            n_levels=3
        )
        
        assert X.shape == (300, 2)  # 100 samples * 3 levels
        assert isinstance(X, np.ndarray)
    
    def test_time_series_with_seasonality(self, generator):
        """Test time series data generation."""
        X, y = generator.time_series_with_seasonality(
            n_samples=365,
            seasonal_periods=[7, 30],
            trend_coef=0.1,
            noise_level=0.1
        )
        
        assert isinstance(X, pd.DataFrame)
        assert len(X) == 365
        assert len(y) == 365
        
        # Check required columns
        required_cols = ['date', 'day_of_week', 'day_of_month', 'month']
        for col in required_cols:
            assert col in X.columns
        
        # Check that trend exists
        assert y[-1] > y[0]  # Should have upward trend
        
        # Check lag features
        lag_cols = [col for col in X.columns if col.startswith('lag_')]
        assert len(lag_cols) > 0
        
        # Check rolling features
        rolling_cols = [col for col in X.columns if col.startswith('rolling_')]
        assert len(rolling_cols) > 0
    
    def test_time_series_default_params(self, generator):
        """Test time series with default parameters."""
        X, y = generator.time_series_with_seasonality(n_samples=100)
        
        assert isinstance(X, pd.DataFrame)
        assert len(X) == 100
        assert len(y) == 100
    
    def test_imbalanced_classification_data(self, generator):
        """Test imbalanced classification data generation."""
        X, y = generator.imbalanced_classification_data(
            n_samples=1000,
            imbalance_ratio=0.1,
            n_features=20
        )
        
        assert X.shape == (1000, 20)
        assert y.shape == (1000,)
        assert len(np.unique(y)) == 2
        
        # Check imbalance ratio
        class_counts = np.bincount(y)
        minority_ratio = min(class_counts) / max(class_counts)
        assert 0.05 <= minority_ratio <= 0.15  # Should be approximately 0.1
    
    def test_mixed_data_types(self, generator):
        """Test mixed data types generation."""
        X, y = generator.mixed_data_types(
            n_samples=100,
            n_numerical=5,
            n_categorical=3,
            n_ordinal=2
        )
        
        assert isinstance(X, pd.DataFrame)
        assert X.shape == (100, 10)
        assert len(y) == 100
        
        # Check feature types
        num_cols = [col for col in X.columns if col.startswith('num_')]
        cat_cols = [col for col in X.columns if col.startswith('cat_')]
        ord_cols = [col for col in X.columns if col.startswith('ord_')]
        
        assert len(num_cols) == 5
        assert len(cat_cols) == 3
        assert len(ord_cols) == 2
        
        # Check data types
        for col in num_cols:
            assert pd.api.types.is_numeric_dtype(X[col])
        
        for col in cat_cols:
            assert pd.api.types.is_categorical_dtype(X[col])
        
        for col in ord_cols:
            assert pd.api.types.is_categorical_dtype(X[col])
            assert X[col].cat.ordered is True
    
    def test_feature_selection_showcase_data(self, generator):
        """Test feature selection data generation."""
        X, y, feature_importance = generator.feature_selection_showcase_data(
            n_samples=200,
            n_features=100,
            n_informative=15,
            n_redundant=15
        )
        
        assert X.shape == (200, 100)
        assert y.shape == (200,)
        assert feature_importance.shape == (100,)
        
        # Check importance structure
        # First 15 features should be informative (high importance)
        assert np.all(feature_importance[:15] >= 0.5)
        
        # Next 15 features should be redundant (medium importance)
        assert np.all(feature_importance[15:30] >= 0.1)
        assert np.all(feature_importance[15:30] < 0.5)
        
        # Remaining features should be noise (zero importance)
        assert np.all(feature_importance[30:] == 0)
    
    def test_feature_selection_default_noise(self, generator):
        """Test feature selection with default noise calculation."""
        X, y, feature_importance = generator.feature_selection_showcase_data(
            n_samples=100,
            n_features=50,
            n_informative=10,
            n_redundant=5
            # n_noise_features not specified, should be calculated
        )
        
        assert X.shape == (100, 50)
        assert feature_importance.shape == (50,)
        
        # Should have 35 noise features (50 - 10 - 5)
        noise_features = np.sum(feature_importance == 0)
        assert noise_features == 35
    
    def test_anomaly_detection_data(self, generator):
        """Test anomaly detection data generation."""
        X, y = generator.anomaly_detection_data(
            n_samples=1000,
            contamination=0.1,
            n_features=3
        )
        
        assert X.shape == (1000, 3)
        assert y.shape == (1000,)
        
        # Check labels
        assert set(np.unique(y)) == {-1, 1}
        
        # Check contamination ratio
        anomaly_ratio = np.sum(y == -1) / len(y)
        assert 0.08 <= anomaly_ratio <= 0.12  # Approximately 0.1
        
        # Check that data is shuffled (outliers not at the end)
        first_half_anomalies = np.sum(y[:500] == -1)
        assert first_half_anomalies > 0  # Should have some anomalies in first half
    
    def test_generate_dataset_suite(self, generator):
        """Test comprehensive dataset suite generation."""
        datasets = generator.generate_dataset_suite()
        
        assert isinstance(datasets, dict)
        assert len(datasets) >= 12  # Should have at least 12 different datasets
        
        # Check that all expected dataset types are present
        expected_datasets = [
            'linear_regression',
            'regression_collinear', 
            'classification_linear',
            'classification_medium',
            'classification_complex',
            'high_dimensional_sparse',
            'imbalanced_classification',
            'mixed_data_types',
            'clustering_blobs',
            'clustering_moons',
            'hierarchical_clusters',
            'feature_selection',
            'anomaly_detection',
            'time_series'
        ]
        
        for dataset_name in expected_datasets:
            assert dataset_name in datasets, f"Missing dataset: {dataset_name}"
        
        # Verify that each dataset has the expected structure
        # Regression datasets
        assert len(datasets['linear_regression']) == 2  # X, y
        assert len(datasets['regression_collinear']) == 3  # X, y, true_coef
        
        # Classification datasets  
        assert len(datasets['classification_linear']) == 2  # X, y
        assert len(datasets['mixed_data_types']) == 2  # X, y
        
        # Clustering datasets (only X)
        assert len(datasets['clustering_blobs']) == 1  # (X,)
        assert len(datasets['clustering_moons']) == 1  # (X,)
        
        # Special datasets
        assert len(datasets['feature_selection']) == 3  # X, y, feature_importance
        assert len(datasets['time_series']) == 2  # X, y
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        gen1 = SyntheticDataGenerator(random_state=42)
        gen2 = SyntheticDataGenerator(random_state=42)
        
        # Test multiple methods for reproducibility
        X1, y1 = gen1.linear_regression_data(n_samples=50, n_features=5)
        X2, y2 = gen2.linear_regression_data(n_samples=50, n_features=5)
        
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
        
        # Test classification
        X1, y1 = gen1.classification_complexity_spectrum('medium', n_samples=50)
        X2, y2 = gen2.classification_complexity_spectrum('medium', n_samples=50)
        
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        gen1 = SyntheticDataGenerator(random_state=42)
        gen2 = SyntheticDataGenerator(random_state=43)
        
        X1, y1 = gen1.linear_regression_data(n_samples=50, n_features=5)
        X2, y2 = gen2.linear_regression_data(n_samples=50, n_features=5)
        
        # Results should be different with different seeds
        assert not np.array_equal(X1, X2)
        assert not np.array_equal(y1, y2)
    
    def test_edge_cases(self, generator):
        """Test edge cases and boundary conditions."""
        # Minimum samples
        X, y = generator.linear_regression_data(n_samples=1, n_features=1)
        assert X.shape == (1, 1)
        assert y.shape == (1,)
        
        # Single feature
        X, y = generator.classification_complexity_spectrum('linear', n_samples=10, n_features=1)
        assert X.shape == (10, 1)
        
        # Very small time series
        X, y = generator.time_series_with_seasonality(n_samples=10)
        assert len(X) == 10
        assert len(y) == 10
    
    def test_parameter_validation(self, generator):
        """Test that invalid parameters are handled appropriately."""
        # Test with zero samples (should work but produce empty result)
        with pytest.raises((ValueError, Exception)):
            generator.linear_regression_data(n_samples=0)
        
        # Test with negative contamination
        with pytest.raises((ValueError, Exception)):
            generator.anomaly_detection_data(contamination=-0.1)
        
        # Test with contamination > 1
        with pytest.raises((ValueError, Exception)):
            generator.anomaly_detection_data(contamination=1.5)


class TestIntegrationWithSklearn:
    """Test integration with scikit-learn components."""
    
    def test_sklearn_compatibility(self):
        """Test that generated data works with sklearn models."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, r2_score
        
        generator = SyntheticDataGenerator(random_state=42)
        
        # Test classification data
        X_clf, y_clf = generator.classification_complexity_spectrum('linear')
        X_train, X_test, y_train, y_test = train_test_split(
            X_clf, y_clf, test_size=0.2, random_state=42
        )
        
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, clf.predict(X_test))
        assert accuracy > 0.5  # Should achieve reasonable accuracy
        
        # Test regression data
        X_reg, y_reg = generator.linear_regression_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        
        reg = RandomForestRegressor(n_estimators=10, random_state=42)
        reg.fit(X_train, y_train)
        r2 = r2_score(y_test, reg.predict(X_test))
        assert r2 > 0.5  # Should achieve reasonable R²
    
    def test_pipeline_integration(self):
        """Test integration with sklearn pipelines."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        generator = SyntheticDataGenerator(random_state=42)
        X, y = generator.classification_complexity_spectrum('linear', n_samples=200)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        scores = cross_val_score(pipeline, X, y, cv=3)
        assert np.mean(scores) > 0.7  # Should achieve good performance on linear data
    
    def test_clustering_with_sklearn(self):
        """Test clustering data with sklearn clustering algorithms."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        
        generator = SyntheticDataGenerator(random_state=42)
        X = generator.clustering_blobs_with_noise(
            n_clusters=3,
            n_samples=300,
            outlier_fraction=0.05
        )
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Should find reasonable clusters (though we don't have true labels)
        assert len(np.unique(labels)) <= 3
        assert len(np.unique(labels)) >= 2  # Should find at least 2 clusters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])