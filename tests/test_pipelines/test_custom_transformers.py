"""
Unit tests for custom transformers.

Tests for all custom transformer classes and utilities.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import sys
import os
import tempfile
import warnings

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from preprocessing.custom_transformers import (
    OutlierRemover,
    FeatureSelector,
    DateTimeTransformer,
    CategoricalEncoder,
    NumericTransformer,
    TextTransformer,
    MissingValueHandler,
    FeatureInteractionCreator,
    PipelineDebugger,
    DataValidator,
    FeatureUnion,
    CustomScaler,
    BinningTransformer,
    PolynomialFeatureCreator,
    TargetEncoder
)


class TestOutlierRemover:
    """Test OutlierRemover transformer."""
    
    @pytest.fixture
    def data_with_outliers(self):
        """Generate data with outliers."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 3))
        # Add outliers
        X[95:, 0] = 10  # Extreme outliers in first column
        X[96:, 1] = -8  # Extreme outliers in second column
        return X
    
    @pytest.fixture
    def transformer(self):
        """Create transformer instance."""
        return OutlierRemover(method='iqr', threshold=1.5)
    
    def test_initialization(self, transformer):
        """Test transformer initialization."""
        assert transformer.method == 'iqr'
        assert transformer.threshold == 1.5
        assert hasattr(transformer, 'outlier_bounds_')
    
    def test_fit_transform(self, transformer, data_with_outliers):
        """Test fitting and transformation."""
        X_transformed = transformer.fit_transform(data_with_outliers)
        
        # Should remove outlier rows
        assert X_transformed.shape[0] < data_with_outliers.shape[0]
        assert X_transformed.shape[1] == data_with_outliers.shape[1]
    
    def test_different_methods(self, data_with_outliers):
        """Test different outlier detection methods."""
        methods = ['iqr', 'zscore', 'isolation_forest']
        
        for method in methods:
            transformer = OutlierRemover(method=method)
            X_transformed = transformer.fit_transform(data_with_outliers)
            
            assert X_transformed.shape[0] <= data_with_outliers.shape[0]
            assert X_transformed.shape[1] == data_with_outliers.shape[1]
    
    def test_no_outliers_data(self, transformer):
        """Test with data that has no outliers."""
        X = np.random.normal(0, 1, (50, 3))
        X_transformed = transformer.fit_transform(X)
        
        # Should keep most or all data
        assert X_transformed.shape[0] >= X.shape[0] * 0.8  # Keep at least 80%
    
    def test_get_outlier_mask(self, transformer, data_with_outliers):
        """Test outlier mask generation."""
        transformer.fit(data_with_outliers)
        mask = transformer.get_outlier_mask(data_with_outliers)
        
        assert len(mask) == len(data_with_outliers)
        assert mask.dtype == bool
        assert np.sum(~mask) > 0  # Should find some outliers


class TestFeatureSelector:
    """Test FeatureSelector transformer."""
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_classification(
            n_samples=100,
            n_features=20,
            n_informative=5,
            n_redundant=5,
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def transformer(self):
        """Create transformer instance."""
        return FeatureSelector(method='univariate', k=10)
    
    def test_initialization(self, transformer):
        """Test transformer initialization."""
        assert transformer.method == 'univariate'
        assert transformer.k == 10
        assert hasattr(transformer, 'selected_features_')
    
    def test_fit_transform(self, transformer, data):
        """Test fitting and transformation."""
        X, y = data
        X_transformed = transformer.fit_transform(X, y)
        
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == transformer.k
    
    def test_different_methods(self, data):
        """Test different feature selection methods."""
        X, y = data
        methods = ['univariate', 'rfe', 'from_model', 'variance_threshold']
        
        for method in methods:
            transformer = FeatureSelector(method=method, k=5)
            X_transformed = transformer.fit_transform(X, y)
            
            assert X_transformed.shape[0] == X.shape[0]
            assert X_transformed.shape[1] <= 5
    
    def test_get_selected_features(self, transformer, data):
        """Test getting selected feature indices."""
        X, y = data
        transformer.fit(X, y)
        
        selected = transformer.get_selected_features()
        
        assert len(selected) == transformer.k
        assert all(isinstance(idx, (int, np.integer)) for idx in selected)
        assert all(0 <= idx < X.shape[1] for idx in selected)
    
    def test_feature_scores(self, transformer, data):
        """Test feature importance scores."""
        X, y = data
        transformer.fit(X, y)
        
        if hasattr(transformer, 'get_feature_scores'):
            scores = transformer.get_feature_scores()
            assert len(scores) == X.shape[1]
            assert all(isinstance(score, (float, np.floating)) for score in scores)


class TestDateTimeTransformer:
    """Test DateTimeTransformer."""
    
    @pytest.fixture
    def datetime_data(self):
        """Generate datetime data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'date_col': dates,
            'datetime_col': pd.to_datetime(dates) + pd.Timedelta(hours=12),
            'other_col': np.random.randn(100)
        })
        return df
    
    @pytest.fixture
    def transformer(self):
        """Create transformer instance."""
        return DateTimeTransformer(
            datetime_columns=['date_col', 'datetime_col'],
            extract_features=['year', 'month', 'day', 'dayofweek']
        )
    
    def test_initialization(self, transformer):
        """Test transformer initialization."""
        assert transformer.datetime_columns == ['date_col', 'datetime_col']
        assert 'year' in transformer.extract_features
        assert hasattr(transformer, 'datetime_columns_')
    
    def test_fit_transform(self, transformer, datetime_data):
        """Test fitting and transformation."""
        X_transformed = transformer.fit_transform(datetime_data)
        
        # Should add new datetime features
        assert X_transformed.shape[0] == datetime_data.shape[0]
        assert X_transformed.shape[1] > datetime_data.shape[1]
    
    def test_feature_extraction(self, transformer, datetime_data):
        """Test specific datetime feature extraction."""
        X_transformed = transformer.fit_transform(datetime_data)
        
        # Check for expected columns
        expected_cols = ['date_col_year', 'date_col_month', 'date_col_day', 'date_col_dayofweek']
        for col in expected_cols:
            assert col in X_transformed.columns
    
    def test_cyclical_features(self, datetime_data):
        """Test cyclical encoding of datetime features."""
        transformer = DateTimeTransformer(
            datetime_columns=['date_col'],
            extract_features=['month', 'day'],
            cyclical_encoding=True
        )
        
        X_transformed = transformer.fit_transform(datetime_data)
        
        # Should have sin/cos pairs for cyclical features
        assert 'date_col_month_sin' in X_transformed.columns
        assert 'date_col_month_cos' in X_transformed.columns
    
    def test_drop_original(self, datetime_data):
        """Test dropping original datetime columns."""
        transformer = DateTimeTransformer(
            datetime_columns=['date_col'],
            extract_features=['year'],
            drop_original=True
        )
        
        X_transformed = transformer.fit_transform(datetime_data)
        
        assert 'date_col' not in X_transformed.columns
        assert 'date_col_year' in X_transformed.columns


class TestCategoricalEncoder:
    """Test CategoricalEncoder transformer."""
    
    @pytest.fixture
    def categorical_data(self):
        """Generate categorical data."""
        np.random.seed(42)
        df = pd.DataFrame({
            'cat_low': np.random.choice(['A', 'B', 'C'], 100),
            'cat_high': np.random.choice([f'cat_{i}' for i in range(50)], 100),
            'numeric': np.random.randn(100)
        })
        return df
    
    @pytest.fixture
    def transformer(self):
        """Create transformer instance."""
        return CategoricalEncoder(
            categorical_columns=['cat_low', 'cat_high'],
            encoding_method='onehot',
            handle_unknown='ignore'
        )
    
    def test_initialization(self, transformer):
        """Test transformer initialization."""
        assert transformer.categorical_columns == ['cat_low', 'cat_high']
        assert transformer.encoding_method == 'onehot'
        assert transformer.handle_unknown == 'ignore'
    
    def test_onehot_encoding(self, categorical_data):
        """Test one-hot encoding."""
        transformer = CategoricalEncoder(
            categorical_columns=['cat_low'],
            encoding_method='onehot'
        )
        
        X_transformed = transformer.fit_transform(categorical_data)
        
        # Should create multiple columns for one-hot
        assert X_transformed.shape[0] == categorical_data.shape[0]
        assert X_transformed.shape[1] > categorical_data.shape[1]
    
    def test_label_encoding(self, categorical_data):
        """Test label encoding."""
        transformer = CategoricalEncoder(
            categorical_columns=['cat_low'],
            encoding_method='label'
        )
        
        X_transformed = transformer.fit_transform(categorical_data)
        
        # Should keep same number of columns
        assert X_transformed.shape == categorical_data.shape
        assert X_transformed['cat_low'].dtype in [np.int32, np.int64]
    
    def test_target_encoding(self, categorical_data):
        """Test target encoding."""
        y = np.random.randn(len(categorical_data))
        
        transformer = CategoricalEncoder(
            categorical_columns=['cat_low'],
            encoding_method='target'
        )
        
        X_transformed = transformer.fit_transform(categorical_data, y)
        
        assert X_transformed.shape == categorical_data.shape
        assert X_transformed['cat_low'].dtype in [np.float32, np.float64]
    
    def test_high_cardinality_handling(self, categorical_data):
        """Test handling of high cardinality categories."""
        transformer = CategoricalEncoder(
            categorical_columns=['cat_high'],
            encoding_method='onehot',
            max_cardinality=10
        )
        
        X_transformed = transformer.fit_transform(categorical_data)
        
        # Should handle high cardinality appropriately
        assert X_transformed.shape[0] == categorical_data.shape[0]


class TestNumericTransformer:
    """Test NumericTransformer."""
    
    @pytest.fixture
    def numeric_data(self):
        """Generate numeric data."""
        np.random.seed(42)
        df = pd.DataFrame({
            'normal': np.random.normal(0, 1, 100),
            'skewed': np.random.exponential(2, 100),
            'with_zeros': np.random.choice([0, 1, 2, 3, 4], 100),
            'categorical': ['A', 'B', 'C'] * 33 + ['A']
        })
        return df
    
    @pytest.fixture
    def transformer(self):
        """Create transformer instance."""
        return NumericTransformer(
            numeric_columns=['normal', 'skewed', 'with_zeros'],
            scaling_method='standard',
            handle_outliers=True
        )
    
    def test_initialization(self, transformer):
        """Test transformer initialization."""
        assert transformer.numeric_columns == ['normal', 'skewed', 'with_zeros']
        assert transformer.scaling_method == 'standard'
        assert transformer.handle_outliers is True
    
    def test_standard_scaling(self, transformer, numeric_data):
        """Test standard scaling."""
        X_transformed = transformer.fit_transform(numeric_data)
        
        # Check that numeric columns are scaled
        for col in transformer.numeric_columns:
            assert abs(X_transformed[col].mean()) < 0.1  # Approximately 0
            assert abs(X_transformed[col].std() - 1.0) < 0.1  # Approximately 1
    
    def test_different_scaling_methods(self, numeric_data):
        """Test different scaling methods."""
        scaling_methods = ['standard', 'minmax', 'robust', 'quantile']
        
        for method in scaling_methods:
            transformer = NumericTransformer(
                numeric_columns=['normal'],
                scaling_method=method
            )
            
            X_transformed = transformer.fit_transform(numeric_data)
            
            assert X_transformed.shape == numeric_data.shape
            assert not X_transformed['normal'].equals(numeric_data['normal'])
    
    def test_transformation_methods(self, numeric_data):
        """Test data transformation methods."""
        transformer = NumericTransformer(
            numeric_columns=['skewed'],
            apply_transforms=['log', 'sqrt']
        )
        
        X_transformed = transformer.fit_transform(numeric_data)
        
        # Should apply transformations
        assert X_transformed.shape[0] == numeric_data.shape[0]
        # New columns should be created for transformations
        assert 'skewed_log' in X_transformed.columns or 'skewed_sqrt' in X_transformed.columns
    
    def test_outlier_handling(self, numeric_data):
        """Test outlier handling in numeric transformation."""
        # Add outliers
        numeric_data_with_outliers = numeric_data.copy()
        numeric_data_with_outliers.loc[0, 'normal'] = 100  # Extreme outlier
        
        transformer = NumericTransformer(
            numeric_columns=['normal'],
            handle_outliers=True,
            outlier_method='clip'
        )
        
        X_transformed = transformer.fit_transform(numeric_data_with_outliers)
        
        # Outlier should be clipped
        assert X_transformed['normal'].max() < 50


class TestMissingValueHandler:
    """Test MissingValueHandler transformer."""
    
    @pytest.fixture
    def data_with_missing(self):
        """Generate data with missing values."""
        np.random.seed(42)
        df = pd.DataFrame({
            'numeric_col': np.random.randn(100),
            'categorical_col': np.random.choice(['A', 'B', 'C'], 100),
            'mixed_col': np.random.randn(100)
        })
        
        # Introduce missing values
        df.loc[10:15, 'numeric_col'] = np.nan
        df.loc[20:25, 'categorical_col'] = np.nan
        df.loc[30:32, 'mixed_col'] = np.nan
        
        return df
    
    @pytest.fixture
    def transformer(self):
        """Create transformer instance."""
        return MissingValueHandler(
            strategy='auto',
            fill_value=None
        )
    
    def test_initialization(self, transformer):
        """Test transformer initialization."""
        assert transformer.strategy == 'auto'
        assert transformer.fill_value is None
    
    def test_fit_transform(self, transformer, data_with_missing):
        """Test fitting and transformation."""
        X_transformed = transformer.fit_transform(data_with_missing)
        
        # Should have no missing values
        assert not X_transformed.isnull().any().any()
        assert X_transformed.shape == data_with_missing.shape
    
    def test_different_strategies(self, data_with_missing):
        """Test different imputation strategies."""
        strategies = ['mean', 'median', 'mode', 'constant', 'knn']
        
        for strategy in strategies:
            transformer = MissingValueHandler(strategy=strategy, fill_value=0)
            X_transformed = transformer.fit_transform(data_with_missing)
            
            assert not X_transformed.isnull().any().any()
            assert X_transformed.shape == data_with_missing.shape
    
    def test_missing_indicator(self, data_with_missing):
        """Test missing value indicator creation."""
        transformer = MissingValueHandler(
            strategy='mean',
            add_indicator=True
        )
        
        X_transformed = transformer.fit_transform(data_with_missing)
        
        # Should add indicator columns
        assert X_transformed.shape[0] == data_with_missing.shape[0]
        assert X_transformed.shape[1] > data_with_missing.shape[1]
    
    def test_column_specific_strategies(self, data_with_missing):
        """Test column-specific imputation strategies."""
        transformer = MissingValueHandler(
            strategy={
                'numeric_col': 'mean',
                'categorical_col': 'mode',
                'mixed_col': 'median'
            }
        )
        
        X_transformed = transformer.fit_transform(data_with_missing)
        
        assert not X_transformed.isnull().any().any()


class TestFeatureInteractionCreator:
    """Test FeatureInteractionCreator transformer."""
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'categorical': np.random.choice(['A', 'B'], 100)
        })
        return df
    
    @pytest.fixture
    def transformer(self):
        """Create transformer instance."""
        return FeatureInteractionCreator(
            interaction_pairs=[('feature1', 'feature2')],
            interaction_types=['multiply', 'add']
        )
    
    def test_initialization(self, transformer):
        """Test transformer initialization."""
        assert transformer.interaction_pairs == [('feature1', 'feature2')]
        assert 'multiply' in transformer.interaction_types
    
    def test_fit_transform(self, transformer, data):
        """Test fitting and transformation."""
        X_transformed = transformer.fit_transform(data)
        
        # Should add interaction features
        assert X_transformed.shape[0] == data.shape[0]
        assert X_transformed.shape[1] > data.shape[1]
    
    def test_interaction_types(self, data):
        """Test different interaction types."""
        interaction_types = ['multiply', 'add', 'subtract', 'divide', 'ratio']
        
        transformer = FeatureInteractionCreator(
            interaction_pairs=[('feature1', 'feature2')],
            interaction_types=interaction_types
        )
        
        X_transformed = transformer.fit_transform(data)
        
        # Should create multiple interaction features
        expected_new_features = len(interaction_types)
        assert X_transformed.shape[1] >= data.shape[1] + expected_new_features
    
    def test_auto_interaction_discovery(self, data):
        """Test automatic interaction discovery."""
        transformer = FeatureInteractionCreator(
            auto_detect=True,
            max_interactions=5,
            numeric_columns=['feature1', 'feature2', 'feature3']
        )
        
        X_transformed = transformer.fit_transform(data)
        
        assert X_transformed.shape[0] == data.shape[0]
        assert X_transformed.shape[1] > data.shape[1]
    
    def test_polynomial_interactions(self, data):
        """Test polynomial feature interactions."""
        transformer = FeatureInteractionCreator(
            polynomial_degree=2,
            include_bias=False,
            interaction_only=True
        )
        
        X_transformed = transformer.fit_transform(data[['feature1', 'feature2']])
        
        # Should create polynomial interactions
        assert X_transformed.shape[0] == data.shape[0]
        assert X_transformed.shape[1] > 2


class TestPipelineDebugger:
    """Test PipelineDebugger transformer."""
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        return pd.DataFrame(X), y
    
    @pytest.fixture
    def transformer(self):
        """Create transformer instance."""
        return PipelineDebugger(
            step_name='test_step',
            log_shape=True,
            log_dtypes=True,
            log_missing=True
        )
    
    def test_initialization(self, transformer):
        """Test transformer initialization."""
        assert transformer.step_name == 'test_step'
        assert transformer.log_shape is True
        assert transformer.log_dtypes is True
    
    def test_fit_transform(self, transformer, data):
        """Test fitting and transformation."""
        X, y = data
        X_transformed = transformer.fit_transform(X)
        
        # Should pass through data unchanged
        pd.testing.assert_frame_equal(X, X_transformed)
    
    def test_logging_functionality(self, transformer, data):
        """Test logging functionality."""
        X, y = data
        
        # This should log information
        transformer.fit_transform(X)
        
        # Check that debug info is stored
        assert hasattr(transformer, 'debug_info_')
        assert 'shape' in transformer.debug_info_
    
    def test_in_pipeline(self, data):
        """Test debugger in sklearn pipeline."""
        X, y = data
        
        pipeline = Pipeline([
            ('debugger1', PipelineDebugger('input')),
            ('scaler', StandardScaler()),
            ('debugger2', PipelineDebugger('after_scaling')),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        
        assert len(predictions) == len(X)


class TestDataValidator:
    """Test DataValidator transformer."""
    
    @pytest.fixture
    def valid_data(self):
        """Generate valid test data."""
        return pd.DataFrame({
            'numeric_col': np.random.randn(100),
            'positive_col': np.random.exponential(1, 100),
            'categorical_col': np.random.choice(['A', 'B', 'C'], 100)
        })
    
    @pytest.fixture
    def transformer(self):
        """Create transformer instance."""
        return DataValidator(
            schema={
                'numeric_col': {'type': 'numeric', 'min': -5, 'max': 5},
                'positive_col': {'type': 'numeric', 'min': 0},
                'categorical_col': {'type': 'categorical', 'categories': ['A', 'B', 'C']}
            }
        )
    
    def test_initialization(self, transformer):
        """Test transformer initialization."""
        assert 'numeric_col' in transformer.schema
        assert transformer.schema['numeric_col']['type'] == 'numeric'
    
    def test_valid_data_passes(self, transformer, valid_data):
        """Test that valid data passes validation."""
        X_transformed = transformer.fit_transform(valid_data)
        
        # Should pass through unchanged
        pd.testing.assert_frame_equal(valid_data, X_transformed)
    
    def test_invalid_data_raises_error(self, transformer):
        """Test that invalid data raises validation error."""
        invalid_data = pd.DataFrame({
            'numeric_col': [100],  # Outside range
            'positive_col': [-1],  # Negative value
            'categorical_col': ['D']  # Invalid category
        })
        
        with pytest.raises((ValueError, AssertionError)):
            transformer.fit_transform(invalid_data)
    
    def test_missing_columns_handling(self, transformer):
        """Test handling of missing columns."""
        incomplete_data = pd.DataFrame({
            'numeric_col': [1, 2, 3]
            # Missing other required columns
        })
        
        with pytest.raises((ValueError, KeyError)):
            transformer.fit_transform(incomplete_data)
    
    def test_optional_validation(self, valid_data):
        """Test optional validation mode."""
        transformer = DataValidator(
            schema={'numeric_col': {'type': 'numeric'}},
            strict=False
        )
        
        # Should handle extra columns gracefully
        extra_data = valid_data.copy()
        extra_data['extra_col'] = [1] * len(extra_data)
        
        X_transformed = transformer.fit_transform(extra_data)
        assert 'extra_col' in X_transformed.columns


class TestCustomScaler:
    """Test CustomScaler transformer."""
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.normal(100, 20, 100),
            'feature2': np.random.exponential(5, 100),
            'feature3': np.random.uniform(-10, 10, 100)
        })
    
    def test_robust_scaling(self, data):
        """Test robust scaling method."""
        transformer = CustomScaler(method='robust')
        X_transformed = transformer.fit_transform(data)
        
        # Check that data is scaled
        assert X_transformed.shape == data.shape
        for col in data.columns:
            assert abs(X_transformed[col].median()) < 0.1
    
    def test_quantile_scaling(self, data):
        """Test quantile scaling method."""
        transformer = CustomScaler(method='quantile', n_quantiles=100)
        X_transformed = transformer.fit_transform(data)
        
        assert X_transformed.shape == data.shape
        # Quantile scaling should map to uniform distribution
        for col in data.columns:
            assert 0 <= X_transformed[col].min() <= 0.1
            assert 0.9 <= X_transformed[col].max() <= 1.0
    
    def test_unit_vector_scaling(self, data):
        """Test unit vector scaling."""
        transformer = CustomScaler(method='unit_vector')
        X_transformed = transformer.fit_transform(data)
        
        # Each row should have unit norm
        row_norms = np.linalg.norm(X_transformed.values, axis=1)
        np.testing.assert_array_almost_equal(row_norms, 1.0, decimal=5)
    
    def test_inverse_transform(self, data):
        """Test inverse transformation."""
        transformer = CustomScaler(method='standard')
        X_transformed = transformer.fit_transform(data)
        X_inverse = transformer.inverse_transform(X_transformed)
        
        # Should be close to original
        pd.testing.assert_frame_equal(data, X_inverse, check_exact=False, atol=1e-10)


class TestBinningTransformer:
    """Test BinningTransformer."""
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        np.random.seed(42)
        return pd.DataFrame({
            'continuous': np.random.normal(0, 1, 100),
            'skewed': np.random.exponential(2, 100),
            'uniform': np.random.uniform(0, 10, 100)
        })
    
    def test_equal_width_binning(self, data):
        """Test equal width binning."""
        transformer = BinningTransformer(
            columns=['continuous'],
            n_bins=5,
            strategy='uniform'
        )
        
        X_transformed = transformer.fit_transform(data)
        
        # Should create binned version
        assert 'continuous_binned' in X_transformed.columns
        assert X_transformed['continuous_binned'].nunique() <= 5
    
    def test_equal_frequency_binning(self, data):
        """Test equal frequency (quantile) binning."""
        transformer = BinningTransformer(
            columns=['skewed'],
            n_bins=4,
            strategy='quantile'
        )
        
        X_transformed = transformer.fit_transform(data)
        
        assert 'skewed_binned' in X_transformed.columns
        # Each bin should have approximately equal frequency
        bin_counts = X_transformed['skewed_binned'].value_counts()
        assert bin_counts.std() < bin_counts.mean() * 0.3  # Low variation in counts
    
    def test_custom_bins(self, data):
        """Test custom bin edges."""
        transformer = BinningTransformer(
            columns=['uniform'],
            bins=[0, 2.5, 5, 7.5, 10],
            labels=['low', 'medium-low', 'medium-high', 'high']
        )
        
        X_transformed = transformer.fit_transform(data)
        
        assert 'uniform_binned' in X_transformed.columns
        assert set(X_transformed['uniform_binned'].dropna()) <= set(['low', 'medium-low', 'medium-high', 'high'])
    
    def test_encode_bins(self, data):
        """Test bin encoding options."""
        transformer = BinningTransformer(
            columns=['continuous'],
            n_bins=3,
            encode='ordinal'
        )
        
        X_transformed = transformer.fit_transform(data)
        
        # Should create numeric encoding
        assert X_transformed['continuous_binned'].dtype in [np.int32, np.int64]


class TestPolynomialFeatureCreator:
    """Test PolynomialFeatureCreator transformer."""
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        np.random.seed(42)
        return pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
            'x3': np.random.randn(50)
        })
    
    def test_degree_2_polynomial(self, data):
        """Test degree 2 polynomial features."""
        transformer = PolynomialFeatureCreator(
            degree=2,
            include_bias=False,
            interaction_only=False
        )
        
        X_transformed = transformer.fit_transform(data)
        
        # Should include original + squared + interactions
        expected_features = 3 + 3 + 3  # original + squared + cross terms
        assert X_transformed.shape[1] == expected_features
        assert X_transformed.shape[0] == data.shape[0]
    
    def test_interaction_only(self, data):
        """Test interaction-only polynomial features."""
        transformer = PolynomialFeatureCreator(
            degree=2,
            include_bias=False,
            interaction_only=True
        )
        
        X_transformed = transformer.fit_transform(data)
        
        # Should include original + interactions (no squared terms)
        expected_features = 3 + 3  # original + cross terms
        assert X_transformed.shape[1] == expected_features
    
    def test_with_bias(self, data):
        """Test polynomial features with bias term."""
        transformer = PolynomialFeatureCreator(
            degree=2,
            include_bias=True
        )
        
        X_transformed = transformer.fit_transform(data)
        
        # Should include bias term
        assert X_transformed.shape[1] > data.shape[1]
        # First column should be all ones (bias)
        assert np.allclose(X_transformed.iloc[:, 0], 1.0)
    
    def test_feature_names(self, data):
        """Test polynomial feature naming."""
        transformer = PolynomialFeatureCreator(degree=2, include_bias=False)
        X_transformed = transformer.fit_transform(data)
        
        # Should have meaningful column names
        assert 'x1^2' in X_transformed.columns or 'x1 x1' in X_transformed.columns
        assert any('x1' in col and 'x2' in col for col in X_transformed.columns)


class TestTargetEncoder:
    """Test TargetEncoder transformer."""
    
    @pytest.fixture
    def categorical_target_data(self):
        """Generate categorical data with target."""
        np.random.seed(42)
        categories = ['A', 'B', 'C', 'D']
        df = pd.DataFrame({
            'category': np.random.choice(categories, 200),
            'high_card_cat': np.random.choice([f'cat_{i}' for i in range(20)], 200)
        })
        
        # Create target with some relationship to categories
        target_map = {'A': 0.1, 'B': 0.3, 'C': 0.6, 'D': 0.8}
        y = [target_map[cat] + np.random.normal(0, 0.1) for cat in df['category']]
        
        return df, np.array(y)
    
    def test_basic_target_encoding(self, categorical_target_data):
        """Test basic target encoding."""
        X, y = categorical_target_data
        
        transformer = TargetEncoder(
            categorical_columns=['category'],
            smoothing=0
        )
        
        X_transformed = transformer.fit_transform(X, y)
        
        # Should replace categorical with numeric
        assert X_transformed['category'].dtype in [np.float32, np.float64]
        assert X_transformed.shape == X.shape
    
    def test_smoothing_parameter(self, categorical_target_data):
        """Test smoothing in target encoding."""
        X, y = categorical_target_data
        
        transformer_no_smooth = TargetEncoder(['category'], smoothing=0)
        transformer_smooth = TargetEncoder(['category'], smoothing=10)
        
        X_no_smooth = transformer_no_smooth.fit_transform(X, y)
        X_smooth = transformer_smooth.fit_transform(X, y)
        
        # Smoothed version should be closer to global mean
        assert not np.array_equal(X_no_smooth['category'], X_smooth['category'])
    
    def test_cross_validation_encoding(self, categorical_target_data):
        """Test cross-validation target encoding."""
        X, y = categorical_target_data
        
        transformer = TargetEncoder(
            categorical_columns=['category'],
            cv_folds=3
        )
        
        X_transformed = transformer.fit_transform(X, y)
        
        # Should prevent overfitting through CV
        assert X_transformed['category'].dtype in [np.float32, np.float64]
        assert X_transformed.shape == X.shape
    
    def test_handle_unseen_categories(self, categorical_target_data):
        """Test handling of unseen categories."""
        X_train, y_train = categorical_target_data
        
        transformer = TargetEncoder(['category'])
        transformer.fit(X_train, y_train)
        
        # Create test data with unseen category
        X_test = pd.DataFrame({'category': ['E', 'A', 'F']})  # E and F are unseen
        X_transformed = transformer.transform(X_test)
        
        # Should handle unseen categories (typically with global mean)
        assert not X_transformed['category'].isnull().any()
    
    def test_multiple_columns(self, categorical_target_data):
        """Test encoding multiple categorical columns."""
        X, y = categorical_target_data
        
        transformer = TargetEncoder(['category', 'high_card_cat'])
        X_transformed = transformer.fit_transform(X, y)
        
        # Both columns should be encoded
        assert X_transformed['category'].dtype in [np.float32, np.float64]
        assert X_transformed['high_card_cat'].dtype in [np.float32, np.float64]


class TestTransformerIntegration:
    """Integration tests for custom transformers."""
    
    @pytest.fixture
    def complex_data(self):
        """Generate complex realistic dataset."""
        np.random.seed(42)
        
        # Create dates
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        
        df = pd.DataFrame({
            'date_col': dates,
            'numeric_col': np.random.normal(50, 15, 200),
            'skewed_col': np.random.exponential(2, 200),
            'categorical_low': np.random.choice(['A', 'B', 'C'], 200),
            'categorical_high': np.random.choice([f'cat_{i}' for i in range(30)], 200),
            'binary_col': np.random.choice([0, 1], 200),
            'text_col': [f'text_{i%10}' for i in range(200)]
        })
        
        # Add missing values
        df.loc[10:15, 'numeric_col'] = np.nan
        df.loc[20:25, 'categorical_low'] = np.nan
        
        # Add outliers
        df.loc[190:, 'numeric_col'] = 200
        
        return df
    
    def test_full_preprocessing_pipeline(self, complex_data):
        """Test complete preprocessing pipeline."""
        # Create a comprehensive pipeline
        pipeline = Pipeline([
            ('missing_handler', MissingValueHandler(strategy='auto')),
            ('datetime_transformer', DateTimeTransformer(
                datetime_columns=['date_col'],
                extract_features=['month', 'dayofweek']
            )),
            ('outlier_remover', OutlierRemover(method='iqr')),
            ('categorical_encoder', CategoricalEncoder(
                categorical_columns=['categorical_low', 'categorical_high'],
                encoding_method='onehot'
            )),
            ('numeric_transformer', NumericTransformer(
                numeric_columns=['numeric_col', 'skewed_col'],
                scaling_method='standard'
            )),
            ('feature_selector', FeatureSelector(method='variance_threshold', threshold=0.01))
        ])
        
        X_transformed = pipeline.fit_transform(complex_data)
        
        # Pipeline should handle all transformations
        assert X_transformed.shape[0] <= complex_data.shape[0]  # May remove outliers
        assert X_transformed.shape[1] != complex_data.shape[1]  # Features changed
        assert not pd.DataFrame(X_transformed).isnull().any().any()  # No missing values
    
    def test_custom_pipeline_with_target(self, complex_data):
        """Test pipeline that requires target variable."""
        # Create target
        y = np.random.randn(len(complex_data))
        
        pipeline = Pipeline([
            ('missing_handler', MissingValueHandler()),
            ('target_encoder', TargetEncoder(['categorical_low'])),
            ('feature_selector', FeatureSelector(method='univariate', k=5))
        ])
        
        X_transformed = pipeline.fit_transform(complex_data, y)
        
        assert X_transformed.shape[0] == complex_data.shape[0]
        assert X_transformed.shape[1] <= 5  # Selected features
    
    def test_transformer_serialization(self, complex_data):
        """Test transformer saving and loading."""
        transformer = NumericTransformer(
            numeric_columns=['numeric_col'],
            scaling_method='standard'
        )
        
        transformer.fit(complex_data)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save transformer
            import pickle
            save_path = os.path.join(tmp_dir, 'transformer.pkl')
            
            with open(save_path, 'wb') as f:
                pickle.dump(transformer, f)
            
            # Load transformer
            with open(save_path, 'rb') as f:
                loaded_transformer = pickle.load(f)
            
            # Test that loaded transformer works
            X_original = transformer.transform(complex_data)
            X_loaded = loaded_transformer.transform(complex_data)
            
            pd.testing.assert_frame_equal(
                pd.DataFrame(X_original), 
                pd.DataFrame(X_loaded)
            )
    
    def test_transformer_error_handling(self, complex_data):
        """Test error handling in transformers."""
        # Test with incompatible data types
        transformer = NumericTransformer(
            numeric_columns=['text_col'],  # Wrong column type
            scaling_method='standard'
        )
        
        with pytest.raises((ValueError, TypeError)):
            transformer.fit_transform(complex_data)
    
    def test_feature_name_preservation(self, complex_data):
        """Test that feature names are handled correctly."""
        # Select subset of data
        subset_data = complex_data[['numeric_col', 'categorical_low']].copy()
        
        transformer = CategoricalEncoder(
            categorical_columns=['categorical_low'],
            encoding_method='onehot'
        )
        
        X_transformed = transformer.fit_transform(subset_data)
        
        # Should have meaningful column names
        assert isinstance(X_transformed.columns[0], str)
        assert len(X_transformed.columns) > len(subset_data.columns)
    
    def test_memory_efficiency(self, complex_data):
        """Test memory efficiency of transformers."""
        # Create larger dataset
        large_data = pd.concat([complex_data] * 5, ignore_index=True)
        
        transformer = NumericTransformer(
            numeric_columns=['numeric_col'],
            scaling_method='standard'
        )
        
        # Should handle large datasets without memory issues
        X_transformed = transformer.fit_transform(large_data)
        
        assert X_transformed.shape[0] == large_data.shape[0]
        assert not pd.DataFrame(X_transformed).isnull().any().any()


class TestTransformerPerformance:
    """Performance tests for custom transformers."""
    
    def test_scaling_performance(self):
        """Test performance with different data sizes."""
        sizes = [100, 1000, 5000]
        
        for size in sizes:
            # Generate data of different sizes
            data = pd.DataFrame({
                'col1': np.random.randn(size),
                'col2': np.random.randn(size)
            })
            
            transformer = NumericTransformer(
                numeric_columns=['col1', 'col2'],
                scaling_method='standard'
            )
            
            import time
            start_time = time.time()
            transformer.fit_transform(data)
            execution_time = time.time() - start_time
            
            # Should complete reasonably quickly
            assert execution_time < 5.0  # Max 5 seconds
    
    def test_memory_usage(self):
        """Test memory usage of transformers."""
        # Create moderately large dataset
        data = pd.DataFrame({
            'numeric': np.random.randn(10000),
            'categorical': np.random.choice(['A', 'B', 'C'], 10000)
        })
        
        transformer = CategoricalEncoder(
            categorical_columns=['categorical'],
            encoding_method='onehot'
        )
        
        # Should not cause memory issues
        X_transformed = transformer.fit_transform(data)
        
        assert X_transformed.shape[0] == data.shape[0]
        assert X_transformed.shape[1] > data.shape[1]


if __name__ == "__main__":
    pytest.main([__file__])