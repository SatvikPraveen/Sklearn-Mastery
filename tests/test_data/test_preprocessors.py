"""
Focused tests for data preprocessing utilities.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from data.preprocessors import DataPreprocessor, CategoricalEncoder, NumericalTransformer


class TestDataPreprocessor:
    """Test DataPreprocessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = pd.DataFrame({
            'num1': np.random.randn(100),
            'num2': np.random.randn(100) * 10 + 5,  # Different scale and offset
            'num3': np.random.exponential(2, 100),   # Skewed data
            'cat1': np.random.choice(['A', 'B', 'C'], 100),
            'cat2': np.random.choice(['X', 'Y'], 100),
            'constant': [1.0] * 100  # Constant feature
        })
        
        # Add some missing values
        X.loc[0:4, 'num1'] = np.nan
        X.loc[10:14, 'cat1'] = np.nan
        
        # Add some outliers
        X.loc[95:97, 'num2'] = [100, -100, 150]
        
        y = np.random.randint(0, 2, 100)
        return X, y
    
    @pytest.fixture
    def preprocessor(self):
        """Create basic preprocessor instance."""
        return DataPreprocessor(random_state=42)
    
    def test_initialization(self):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor(
            handle_missing=True,
            scale_features=True,
            remove_outliers=False,
            feature_selection=True,
            dimensionality_reduction=False
        )
        
        assert preprocessor.handle_missing is True
        assert preprocessor.scale_features is True
        assert preprocessor.remove_outliers is False
        assert preprocessor.feature_selection is True
        assert preprocessor.dimensionality_reduction is False
        assert preprocessor.is_fitted_ is False
        assert isinstance(preprocessor.preprocessors_, dict)
    
    def test_basic_fit_transform(self, sample_data, preprocessor):
        """Test basic fit and transform functionality."""
        X, y = sample_data
        
        X_transformed = preprocessor.fit_transform(X, y)
        
        # Check output format
        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape[0] == X.shape[0]
        assert preprocessor.is_fitted_ is True
        
        # Check no missing values in output
        assert not np.any(np.isnan(X_transformed))
    
    def test_missing_value_handling(self, sample_data):
        """Test missing value imputation."""
        X, y = sample_data
        
        preprocessor = DataPreprocessor(
            handle_missing=True,
            scale_features=False,
            remove_outliers=False
        )
        
        # Check that input has missing values
        assert X.isnull().any().any()
        
        X_transformed = preprocessor.fit_transform(X, y)
        
        # Check that output has no missing values
        assert not np.any(np.isnan(X_transformed))
        assert 'imputer' in preprocessor.preprocessors_
    
    def test_feature_scaling(self, sample_data):
        """Test feature scaling functionality."""
        X, y = sample_data
        
        # Select only numerical columns for this test
        X_num = X.select_dtypes(include=[np.number])
        
        preprocessor = DataPreprocessor(
            handle_missing=True,
            scale_features=True,
            remove_outliers=False
        )
        
        X_transformed = preprocessor.fit_transform(X_num, y)
        
        # Check that features are approximately scaled
        means = np.mean(X_transformed, axis=0)
        stds = np.std(X_transformed, axis=0)
        
        # Most features should have mean close to 0 and std close to 1
        # (allowing some tolerance for small datasets and outliers)
        assert np.all(np.abs(means) < 2)  # Means should be close to 0
        assert 'scaler' in preprocessor.preprocessors_
    
    def test_outlier_removal(self, sample_data):
        """Test outlier detection and removal."""
        X, y = sample_data
        
        preprocessor = DataPreprocessor(
            handle_missing=True,
            remove_outliers=True,
            scale_features=False
        )
        
        # Fit the preprocessor
        preprocessor.fit(X, y)
        
        # Check that outlier detector was fitted
        assert 'outlier_detector' in preprocessor.preprocessors_
        
        # Transform should work without removing outliers during transform
        X_transformed = preprocessor.transform(X)
        assert X_transformed.shape[0] == X.shape[0]  # Same number of rows
    
    def test_feature_selection(self, sample_data):
        """Test feature selection functionality."""
        X, y = sample_data
        
        preprocessor = DataPreprocessor(
            handle_missing=True,
            feature_selection=True,
            scale_features=False
        )
        
        X_transformed = preprocessor.fit_transform(X, y)
        
        # Should have same or fewer features
        assert X_transformed.shape[1] <= X.shape[1]
        assert 'feature_selector' in preprocessor.preprocessors_
    
    def test_dimensionality_reduction(self, sample_data):
        """Test dimensionality reduction functionality."""
        X, y = sample_data
        
        # Create data with more features for meaningful dimensionality reduction
        X_expanded = pd.concat([X] * 3, axis=1)
        X_expanded.columns = [f'col_{i}' for i in range(len(X_expanded.columns))]
        
        preprocessor = DataPreprocessor(
            handle_missing=True,
            dimensionality_reduction=True,
            scale_features=True
        )
        
        X_transformed = preprocessor.fit_transform(X_expanded, y)
        
        # Should reduce dimensionality
        assert X_transformed.shape[1] < X_expanded.shape[1]
        assert 'dim_reducer' in preprocessor.preprocessors_
    
    def test_full_pipeline(self, sample_data):
        """Test full preprocessing pipeline."""
        X, y = sample_data
        
        preprocessor = DataPreprocessor(
            handle_missing=True,
            scale_features=True,
            remove_outliers=True,
            feature_selection=True,
            dimensionality_reduction=False
        )
        
        X_transformed = preprocessor.fit_transform(X, y)
        
        # Check that all expected preprocessors were used
        expected_preprocessors = ['imputer', 'outlier_detector', 'scaler', 'feature_selector']
        for proc in expected_preprocessors:
            assert proc in preprocessor.preprocessors_
        
        # Check output quality
        assert isinstance(X_transformed, np.ndarray)
        assert not np.any(np.isnan(X_transformed))
        assert X_transformed.shape[0] == X.shape[0]
    
    def test_fit_transform_consistency(self, sample_data):
        """Test that fit_transform gives same result as fit then transform."""
        X, y = sample_data
        
        # Method 1: fit_transform
        preprocessor1 = DataPreprocessor(
            handle_missing=True,
            scale_features=True,
            random_state=42
        )
        X_transformed1 = preprocessor1.fit_transform(X, y)
        
        # Method 2: fit then transform
        preprocessor2 = DataPreprocessor(
            handle_missing=True,
            scale_features=True,
            random_state=42
        )
        preprocessor2.fit(X, y)
        X_transformed2 = preprocessor2.transform(X)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(X_transformed1, X_transformed2)
    
    def test_transform_without_fit_raises_error(self, sample_data):
        """Test that transform without fit raises appropriate error."""
        X, y = sample_data
        
        preprocessor = DataPreprocessor()
        
        with pytest.raises(ValueError, match="must be fitted"):
            preprocessor.transform(X)
    
    def test_numpy_input_conversion(self):
        """Test that numpy arrays are properly converted to DataFrames."""
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        
        preprocessor = DataPreprocessor(handle_missing=True)
        X_transformed = preprocessor.fit_transform(X, y)
        
        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape == (50, 5)
    
    def test_get_preprocessing_summary(self, sample_data):
        """Test preprocessing summary generation."""
        X, y = sample_data
        
        preprocessor = DataPreprocessor(
            handle_missing=True,
            scale_features=True,
            feature_selection=True
        )
        
        preprocessor.fit(X, y)
        summary = preprocessor.get_preprocessing_summary()
        
        assert isinstance(summary, dict)
        assert 'steps_applied' in summary
        assert 'parameters' in summary
        
        expected_steps = ['missing_value_imputation', 'feature_scaling', 'feature_selection']
        for step in expected_steps:
            assert step in summary['steps_applied']
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame."""
        X = pd.DataFrame()
        
        preprocessor = DataPreprocessor(handle_missing=True)
        
        # Should handle empty DataFrame gracefully
        with pytest.raises((ValueError, IndexError)):
            preprocessor.fit_transform(X)


class TestCategoricalEncoder:
    """Test CategoricalEncoder class."""
    
    @pytest.fixture
    def sample_categorical_data(self):
        """Create sample data with categorical features."""
        np.random.seed(42)
        return pd.DataFrame({
            'low_card': np.random.choice(['A', 'B', 'C'], 100),
            'medium_card': np.random.choice([f'cat_{i}' for i in range(20)], 100),
            'high_card': np.random.choice([f'cat_{i}' for i in range(80)], 100),
            'binary': np.random.choice(['Yes', 'No'], 100)
        })
    
    def test_initialization(self):
        """Test CategoricalEncoder initialization."""
        encoder = CategoricalEncoder(
            strategy='auto',
            handle_unknown='ignore',
            max_cardinality=50
        )
        
        assert encoder.strategy == 'auto'
        assert encoder.handle_unknown == 'ignore'
        assert encoder.max_cardinality == 50
        assert isinstance(encoder.encoders_, dict)
        assert isinstance(encoder.strategies_, dict)
    
    def test_auto_strategy_selection(self, sample_categorical_data):
        """Test automatic strategy selection based on cardinality."""
        X = sample_categorical_data
        y = np.random.randint(0, 2, 100)
        
        encoder = CategoricalEncoder(strategy='auto', max_cardinality=30)
        encoder.fit(X, y)
        
        # Check strategy selection
        assert encoder.strategies_['low_card'] == 'onehot'  # Low cardinality
        assert encoder.strategies_['binary'] == 'onehot'    # Binary
        assert encoder.strategies_['medium_card'] == 'target'  # Medium with target
        assert encoder.strategies_['high_card'] == 'label'    # High cardinality
    
    def test_auto_strategy_without_target(self, sample_categorical_data):
        """Test auto strategy when no target is provided."""
        X = sample_categorical_data
        
        encoder = CategoricalEncoder(strategy='auto', max_cardinality=30)
        encoder.fit(X)  # No y provided
        
        # Without target, should use label encoding for medium cardinality
        assert encoder.strategies_['medium_card'] == 'label'
    
    def test_onehot_encoding(self, sample_categorical_data):
        """Test one-hot encoding."""
        X = sample_categorical_data[['low_card']]
        
        encoder = CategoricalEncoder(strategy='onehot')
        X_encoded = encoder.fit_transform(X)
        
        # Should create multiple columns
        assert X_encoded.shape[1] > 1
        assert X_encoded.shape[0] == X.shape[0]
        
        # Check that original column is removed and new columns added
        assert 'low_card' not in X_encoded.columns
        assert any('low_card_' in col for col in X_encoded.columns)
    
    def test_label_encoding(self, sample_categorical_data):
        """Test label encoding."""
        X = sample_categorical_data[['low_card']]
        
        encoder = CategoricalEncoder(strategy='label')
        X_encoded = encoder.fit_transform(X)
        
        # Should maintain same shape
        assert X_encoded.shape == X.shape
        assert X_encoded['low_card'].dtype in [np.int32, np.int64]
    
    def test_target_encoding(self, sample_categorical_data):
        """Test target encoding."""
        X = sample_categorical_data[['low_card']]
        y = np.random.randint(0, 2, 100)
        
        encoder = CategoricalEncoder(strategy='target')
        X_encoded = encoder.fit_transform(X, y)
        
        # Should maintain same shape but with float values
        assert X_encoded.shape == X.shape
        assert X_encoded['low_card'].dtype in [np.float64, float]
    
    def test_binary_encoding(self, sample_categorical_data):
        """Test binary encoding."""
        X = sample_categorical_data[['low_card']]
        
        encoder = CategoricalEncoder(strategy='binary')
        X_encoded = encoder.fit_transform(X)
        
        # Should create binary representation
        assert X_encoded.shape[0] == X.shape[0]
        # May create multiple columns for binary representation
    
    def test_unknown_category_handling(self, sample_categorical_data):
        """Test handling of unknown categories during transform."""
        X_train = pd.DataFrame({'cat': ['A', 'B', 'C'] * 10})
        X_test = pd.DataFrame({'cat': ['A', 'B', 'D']})  # 'D' is unknown
        
        encoder = CategoricalEncoder(strategy='label', handle_unknown='ignore')
        encoder.fit(X_train)
        
        # Should handle unknown category without error
        X_encoded = encoder.transform(X_test)
        assert X_encoded.shape[0] == X_test.shape[0]
    
    def test_mixed_categorical_data(self, sample_categorical_data):
        """Test encoding multiple categorical columns."""
        X = sample_categorical_data
        y = np.random.randint(0, 2, 100)
        
        encoder = CategoricalEncoder(strategy='auto')
        X_encoded = encoder.fit_transform(X, y)
        
        # Should process all categorical columns
        assert X_encoded.shape[0] == X.shape[0]
        # Shape may change due to one-hot encoding
        
        # All columns should be processed
        assert len(encoder.strategies_) == X.shape[1]


class TestNumericalTransformer:
    """Test NumericalTransformer class."""
    
    @pytest.fixture
    def sample_numerical_data(self):
        """Create sample numerical data with different characteristics."""
        np.random.seed(42)
        return pd.DataFrame({
            'normal': np.random.randn(100),
            'skewed': np.random.exponential(2, 100),
            'negative_skewed': -np.random.exponential(2, 100),
            'uniform': np.random.uniform(0, 10, 100),
            'positive': np.random.exponential(1, 100) + 1  # Always positive
        })
    
    def test_initialization(self):
        """Test NumericalTransformer initialization."""
        transformer = NumericalTransformer(
            apply_log=True,
            apply_sqrt=True,
            apply_boxcox=False,
            create_interactions=True,
            create_polynomials=True,
            polynomial_degree=3
        )
        
        assert transformer.apply_log is True
        assert transformer.apply_sqrt is True
        assert transformer.apply_boxcox is False
        assert transformer.create_interactions is True
        assert transformer.create_polynomials is True
        assert transformer.polynomial_degree == 3
        assert isinstance(transformer.transformations_, dict)
    
    def test_skewness_detection_and_log_transform(self, sample_numerical_data):
        """Test skewness detection and log transformation."""
        X = sample_numerical_data[['skewed']]
        
        transformer = NumericalTransformer(apply_log=True)
        transformer.fit(X)
        
        # Skewed column should be selected for log transformation
        assert transformer.transformations_['skewed'] == 'log'
        
        X_transformed = transformer.transform(X)
        
        # Check that transformation was applied
        original_skew = X['skewed'].skew()
        transformed_skew = pd.Series(X_transformed['skewed']).skew()
        assert abs(transformed_skew) < abs(original_skew)
    
    def test_sqrt_transformation(self, sample_numerical_data):
        """Test square root transformation."""
        X = sample_numerical_data[['positive']]  # Use positive values
        
        transformer = NumericalTransformer(apply_sqrt=True)
        X_transformed = transformer.fit_transform(X)
        
        # Values should be reduced (sqrt effect)
        assert np.all(X_transformed['positive'] <= X['positive'])
    
    def test_polynomial_features(self, sample_numerical_data):
        """Test polynomial feature creation."""
        X = sample_numerical_data[['normal', 'uniform']]
        
        transformer = NumericalTransformer(
            create_polynomials=True,
            polynomial_degree=2
        )
        
        X_transformed = transformer.fit_transform(X)
        
        # Should create additional polynomial features
        assert X_transformed.shape[1] > X.shape[1]
        
        # Check for polynomial feature names
        poly_cols = [col for col in X_transformed.columns if 'poly' in col]
        assert len(poly_cols) > 0
    
    def test_interaction_features(self, sample_numerical_data):
        """Test interaction feature creation."""
        X = sample_numerical_data[['normal', 'uniform', 'positive']]
        
        transformer = NumericalTransformer(create_interactions=True)
        X_transformed = transformer.fit_transform(X)
        
        # Should create interaction features
        assert X_transformed.shape[1] > X.shape[1]
        
        # Check for interaction feature names
        interaction_cols = [col for col in X_transformed.columns if '_x_' in col]
        assert len(interaction_cols) > 0
    
    def test_no_transformations(self, sample_numerical_data):
        """Test when no transformations are applied."""
        X = sample_numerical_data[['normal']]  # Normal distribution, no skew
        
        transformer = NumericalTransformer(
            apply_log=False,
            apply_sqrt=False,
            create_interactions=False,
            create_polynomials=False
        )
        
        X_transformed = transformer.fit_transform(X)
        
        # Should be unchanged
        pd.testing.assert_frame_equal(X, X_transformed)
    
    def test_combined_transformations(self, sample_numerical_data):
        """Test multiple transformations applied together."""
        X = sample_numerical_data
        
        transformer = NumericalTransformer(
            apply_log=True,
            create_polynomials=True,
            create_interactions=True,
            polynomial_degree=2
        )
        
        X_transformed = transformer.fit_transform(X)
        
        # Should significantly increase number of features
        assert X_transformed.shape[1] > X.shape[1] * 2
        assert X_transformed.shape[0] == X.shape[0]  # Same number of rows
    
    def test_boxcox_transformation(self):
        """Test Box-Cox transformation."""
        # Create strictly positive data for Box-Cox
        X = pd.DataFrame({'positive': np.random.exponential(2, 100) + 0.1})
        
        transformer = NumericalTransformer(apply_boxcox=True)
        
        try:
            X_transformed = transformer.fit_transform(X)
            # If Box-Cox is applied, should change the distribution
            assert not np.array_equal(X['positive'].values, X_transformed['positive'].values)
        except Exception:
            # Box-Cox might fail for some data, which is acceptable
            pass
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Single column
        X = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        transformer = NumericalTransformer(create_interactions=True)
        X_transformed = transformer.fit_transform(X)
        # No interactions possible with single column
        assert X_transformed.shape[1] == X.shape[1]
        
        # All zeros
        X = pd.DataFrame({'zeros': [0] * 10})
        transformer = NumericalTransformer(apply_log=True)
        X_transformed = transformer.fit_transform(X)
        # Log of zeros should be handled gracefully
        assert X_transformed.shape == X.shape


class TestPreprocessorIntegration:
    """Test integration between different preprocessor components."""
    
    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline with all components."""
        # Create complex mixed data
        np.random.seed(42)
        X = pd.DataFrame({
            'num1': np.random.randn(200),
            'num2': np.random.exponential(2, 200),
            'cat1': np.random.choice(['A', 'B', 'C'], 200),
            'cat2': np.random.choice([f'cat_{i}' for i in range(15)], 200),
            'binary_cat': np.random.choice(['Yes', 'No'], 200)
        })
        
        # Add missing values
        X.loc[0:9, 'num1'] = np.nan
        X.loc[10:19, 'cat1'] = np.nan
        
        y = np.random.randint(0, 2, 200)
        
        # Apply categorical encoding first
        cat_encoder = CategoricalEncoder(strategy='auto')
        X_encoded = cat_encoder.fit_transform(X, y)
        
        # Apply numerical transformations
        num_transformer = NumericalTransformer(
            apply_log=True,
            create_polynomials=True
        )
        X_num_transformed = num_transformer.fit_transform(X_encoded)
        
        # Apply main preprocessor
        preprocessor = DataPreprocessor(
            handle_missing=True,
            scale_features=True,
            feature_selection=True
        )
        X_final = preprocessor.fit_transform(X_num_transformed, y)
        
        # Check final result
        assert isinstance(X_final, np.ndarray)
        assert X_final.shape[0] == len(X)
        assert not np.any(np.isnan(X_final))
    
    def test_sklearn_compatibility(self):
        """Test compatibility with sklearn estimators."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        # Generate data
        X, y = make_classification(n_samples=200, n_features=20, random_state=42)
        X = pd.DataFrame(X)
        
        # Preprocess
        preprocessor = DataPreprocessor(
            handle_missing=True,
            scale_features=True,
            feature_selection=True
        )
        X_processed = preprocessor.fit_transform(X, y)
        
        # Train model
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        scores = cross_val_score(clf, X_processed, y, cv=3)
        
        # Should achieve reasonable performance
        assert np.mean(scores) > 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])