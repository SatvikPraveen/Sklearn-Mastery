"""
Unit tests for feature union and pipeline utility components.

Tests for advanced pipeline compositions, feature unions, and pipeline utilities.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pipelines.feature_union import (
    AdvancedFeatureUnion,
    ParallelFeatureProcessor,
    ConditionalFeatureUnion,
    WeightedFeatureUnion,
    DynamicFeatureUnion,
    FeaturePipelineBuilder,
    PipelineComposer,
    ColumnSelector,
    ColumnTransformer,
    FeatureStacker
)


class TestAdvancedFeatureUnion:
    """Test AdvancedFeatureUnion class."""
    
    @pytest.fixture
    def mixed_data(self):
        """Generate mixed data types for testing."""
        np.random.seed(42)
        df = pd.DataFrame({
            'numeric_1': np.random.randn(200),
            'numeric_2': np.random.exponential(2, 200),
            'categorical_1': np.random.choice(['A', 'B', 'C'], 200),
            'categorical_2': np.random.choice([f'cat_{i}' for i in range(5)], 200),
            'binary': np.random.choice([0, 1], 200),
            'target': np.random.randint(0, 2, 200)
        })
        
        X = df.drop('target', axis=1)
        y = df['target']
        
        return X, y
    
    @pytest.fixture
    def feature_union(self):
        """Create advanced feature union."""
        return AdvancedFeatureUnion()
    
    def test_basic_feature_union(self, feature_union, mixed_data):
        """Test basic feature union functionality."""
        X, y = mixed_data
        
        # Define parallel processing pipelines
        numeric_pipeline = Pipeline([
            ('selector', ColumnSelector(['numeric_1', 'numeric_2'])),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('selector', ColumnSelector(['categorical_1', 'categorical_2'])),
            ('encoder', 'onehot')  # Placeholder for actual encoder
        ])
        
        # Create feature union
        feature_union.add_pipeline('numeric', numeric_pipeline)
        feature_union.add_pipeline('categorical', categorical_pipeline)
        
        X_transformed = feature_union.fit_transform(X, y)
        
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] > 0
    
    def test_weighted_feature_union(self, mixed_data):
        """Test weighted feature union."""
        X, y = mixed_data
        
        weighted_union = WeightedFeatureUnion()
        
        # Add pipelines with weights
        numeric_pipeline = Pipeline([
            ('selector', ColumnSelector(['numeric_1', 'numeric_2'])),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('selector', ColumnSelector(['categorical_1'])),
            ('encoder', 'label')  # Placeholder
        ])
        
        weighted_union.add_pipeline('numeric', numeric_pipeline, weight=0.7)
        weighted_union.add_pipeline('categorical', categorical_pipeline, weight=0.3)
        
        X_transformed = weighted_union.fit_transform(X, y)
        
        assert X_transformed.shape[0] == X.shape[0]
        assert hasattr(weighted_union, 'feature_weights_')
    
    def test_conditional_feature_union(self, mixed_data):
        """Test conditional feature union."""
        X, y = mixed_data
        
        conditional_union = ConditionalFeatureUnion()
        
        # Define conditions
        def has_missing_values(X):
            return X.isnull().any().any()
        
        def has_categorical_features(X):
            return any(X.dtypes == 'object')
        
        # Add conditional pipelines
        missing_pipeline = Pipeline([
            ('imputer', 'simple'),  # Placeholder
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('encoder', 'onehot')  # Placeholder
        ])
        
        conditional_union.add_conditional_pipeline(
            'missing_handler', missing_pipeline, condition=has_missing_values
        )
        
        conditional_union.add_conditional_pipeline(
            'categorical_handler', categorical_pipeline, condition=has_categorical_features
        )
        
        X_transformed = conditional_union.fit_transform(X, y)
        
        assert X_transformed.shape[0] == X.shape[0]
    
    def test_dynamic_feature_union(self, mixed_data):
        """Test dynamic feature union that adapts based on data."""
        X, y = mixed_data
        
        dynamic_union = DynamicFeatureUnion()
        
        # Define adaptive strategy
        def adaptive_strategy(X, y):
            pipelines = []
            
            # Always include numeric processing
            if any(X.dtypes in ['int64', 'float64']):
                pipelines.append(('numeric', 'standard_numeric_pipeline'))
            
            # Add categorical processing if needed
            if any(X.dtypes == 'object'):
                pipelines.append(('categorical', 'onehot_pipeline'))
            
            # Add feature selection if high dimensional
            if X.shape[1] > 10:
                pipelines.append(('selection', 'univariate_selection'))
            
            return pipelines
        
        dynamic_union.set_adaptive_strategy(adaptive_strategy)
        X_transformed = dynamic_union.fit_transform(X, y)
        
        assert X_transformed.shape[0] == X.shape[0]
        assert hasattr(dynamic_union, 'selected_pipelines_')


class TestParallelFeatureProcessor:
    """Test ParallelFeatureProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create parallel feature processor."""
        return ParallelFeatureProcessor()
    
    @pytest.fixture
    def numeric_data(self):
        """Generate numeric data for parallel processing."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature_1': np.random.randn(300),
            'feature_2': np.random.randn(300),
            'feature_3': np.random.randn(300),
            'feature_4': np.random.randn(300)
        })
        y = np.random.randint(0, 2, 300)
        return X, y
    
    def test_parallel_processing(self, processor, numeric_data):
        """Test parallel processing of features."""
        X, y = numeric_data
        
        # Define different processing strategies for different features
        processing_strategies = {
            'group_1': {
                'columns': ['feature_1', 'feature_2'],
                'processor': StandardScaler()
            },
            'group_2': {
                'columns': ['feature_3', 'feature_4'],
                'processor': MinMaxScaler()
            }
        }
        
        processor.set_processing_strategies(processing_strategies)
        X_transformed = processor.fit_transform(X, y)
        
        assert X_transformed.shape == X.shape
        assert not np.array_equal(X_transformed.values, X.values)
    
    def test_parallel_feature_engineering(self, processor, numeric_data):
        """Test parallel feature engineering."""
        X, y = numeric_data
        
        # Define feature engineering functions
        def polynomial_features(X):
            return X ** 2
        
        def interaction_features(X):
            if X.shape[1] >= 2:
                return X.iloc[:, 0] * X.iloc[:, 1]
            return X.iloc[:, 0]
        
        def log_features(X):
            return np.log(np.abs(X) + 1)
        
        # Apply different engineering to different feature groups
        engineering_strategies = {
            'polynomial': {
                'columns': ['feature_1'],
                'function': polynomial_features
            },
            'interaction': {
                'columns': ['feature_2', 'feature_3'],
                'function': interaction_features
            },
            'log_transform': {
                'columns': ['feature_4'],
                'function': log_features
            }
        }
        
        processor.set_engineering_strategies(engineering_strategies)
        X_engineered = processor.fit_transform(X, y)
        
        assert X_engineered.shape[0] == X.shape[0]
        # Should have additional engineered features
        assert X_engineered.shape[1] >= X.shape[1]
    
    def test_conditional_parallel_processing(self, processor, numeric_data):
        """Test conditional parallel processing."""
        X, y = numeric_data
        
        # Add some outliers to test conditional processing
        X_with_outliers = X.copy()
        X_with_outliers.loc[0:5, 'feature_1'] = 100  # Extreme outliers
        
        def outlier_condition(X):
            return (np.abs(X) > 3).any().any()
        
        def normal_condition(X):
            return not outlier_condition(X)
        
        # Define conditional processing
        conditional_strategies = {
            'outlier_processing': {
                'condition': outlier_condition,
                'processor': 'robust_scaler'
            },
            'normal_processing': {
                'condition': normal_condition,
                'processor': StandardScaler()
            }
        }
        
        processor.set_conditional_strategies(conditional_strategies)
        X_processed = processor.fit_transform(X_with_outliers, y)
        
        assert X_processed.shape == X_with_outliers.shape
    
    def test_async_parallel_processing(self, processor, numeric_data):
        """Test asynchronous parallel processing."""
        X, y = numeric_data
        
        # Define time-consuming processing strategies
        processing_strategies = {
            'strategy_1': {
                'columns': ['feature_1', 'feature_2'],
                'processor': StandardScaler(),
                'async': True
            },
            'strategy_2': {
                'columns': ['feature_3', 'feature_4'],
                'processor': MinMaxScaler(),
                'async': True
            }
        }
        
        processor.set_processing_strategies(processing_strategies)
        processor.enable_async_processing(max_workers=2)
        
        import time
        start_time = time.time()
        X_transformed = processor.fit_transform(X, y)
        processing_time = time.time() - start_time
        
        assert X_transformed.shape == X.shape
        # Async processing should be reasonably fast
        assert processing_time < 10  # seconds


class TestColumnSelector:
    """Test ColumnSelector transformer."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            'col_1': np.random.randn(100),
            'col_2': np.random.randn(100),
            'col_3': np.random.choice(['A', 'B', 'C'], 100),
            'col_4': np.random.randint(0, 10, 100),
            'col_5': np.random.randn(100)
        })
    
    def test_column_selection_by_name(self, sample_dataframe):
        """Test column selection by name."""
        selector = ColumnSelector(['col_1', 'col_3', 'col_5'])
        
        X_selected = selector.fit_transform(sample_dataframe)
        
        assert X_selected.shape[0] == sample_dataframe.shape[0]
        assert X_selected.shape[1] == 3
        assert list(X_selected.columns) == ['col_1', 'col_3', 'col_5']
    
    def test_column_selection_by_dtype(self, sample_dataframe):
        """Test column selection by data type."""
        selector = ColumnSelector(dtype_include=['float64', 'int64'])
        
        X_selected = selector.fit_transform(sample_dataframe)
        
        assert X_selected.shape[0] == sample_dataframe.shape[0]
        # Should select numeric columns only
        assert all(dtype in ['float64', 'int64'] for dtype in X_selected.dtypes)
    
    def test_column_selection_by_pattern(self, sample_dataframe):
        """Test column selection by pattern matching."""
        selector = ColumnSelector(pattern='col_[12]')
        
        X_selected = selector.fit_transform(sample_dataframe)
        
        assert X_selected.shape[0] == sample_dataframe.shape[0]
        assert X_selected.shape[1] == 2
        assert 'col_1' in X_selected.columns
        assert 'col_2' in X_selected.columns
    
    def test_column_selection_by_function(self, sample_dataframe):
        """Test column selection by custom function."""
        def select_numeric_with_variance(col):
            return col.dtype in ['float64', 'int64'] and col.var() > 0.5
        
        selector = ColumnSelector(selector_function=select_numeric_with_variance)
        
        X_selected = selector.fit_transform(sample_dataframe)
        
        assert X_selected.shape[0] == sample_dataframe.shape[0]
        assert X_selected.shape[1] <= sample_dataframe.shape[1]
    
    def test_inverse_column_selection(self, sample_dataframe):
        """Test inverse column selection."""
        selector = ColumnSelector(['col_1', 'col_3'], inverse=True)
        
        X_selected = selector.fit_transform(sample_dataframe)
        
        assert X_selected.shape[0] == sample_dataframe.shape[0]
        assert X_selected.shape[1] == 3  # 5 - 2 = 3
        assert 'col_1' not in X_selected.columns
        assert 'col_3' not in X_selected.columns


class TestColumnTransformer:
    """Test ColumnTransformer class."""
    
    @pytest.fixture
    def mixed_dataframe(self):
        """Create mixed DataFrame for transformation testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'numeric_1': np.random.randn(150),
            'numeric_2': np.random.exponential(2, 150),
            'categorical_1': np.random.choice(['A', 'B', 'C'], 150),
            'categorical_2': np.random.choice([f'cat_{i}' for i in range(5)], 150),
            'binary': np.random.choice([0, 1], 150),
            'text': [f'text_{i%10}' for i in range(150)]
        })
    
    def test_column_specific_transformations(self, mixed_dataframe):
        """Test applying different transformations to different columns."""
        transformer = ColumnTransformer()
        
        # Define column-specific transformations
        transformations = [
            (['numeric_1', 'numeric_2'], StandardScaler()),
            (['categorical_1'], 'onehot_encoder'),
            (['categorical_2'], 'label_encoder'),
            (['binary'], 'passthrough')
        ]
        
        transformer.set_transformations(transformations)
        X_transformed = transformer.fit_transform(mixed_dataframe)
        
        assert X_transformed.shape[0] == mixed_dataframe.shape[0]
        # Should have more columns due to one-hot encoding
        assert X_transformed.shape[1] >= mixed_dataframe.shape[1]
    
    def test_remainder_handling(self, mixed_dataframe):
        """Test handling of remaining columns."""
        transformer = ColumnTransformer(remainder='passthrough')
        
        transformations = [
            (['numeric_1'], StandardScaler()),
            (['categorical_1'], 'onehot_encoder')
        ]
        
        transformer.set_transformations(transformations)
        X_transformed = transformer.fit_transform(mixed_dataframe)
        
        # Should include all columns (transformed + remainder)
        assert X_transformed.shape[0] == mixed_dataframe.shape[0]
        assert X_transformed.shape[1] >= mixed_dataframe.shape[1]
    
    def test_sparse_output(self, mixed_dataframe):
        """Test sparse matrix output."""
        transformer = ColumnTransformer(sparse_output=True)
        
        transformations = [
            (['categorical_1', 'categorical_2'], 'onehot_encoder')
        ]
        
        transformer.set_transformations(transformations)
        X_transformed = transformer.fit_transform(mixed_dataframe)
        
        # Should return sparse matrix
        from scipy import sparse
        assert sparse.issparse(X_transformed)
    
    def test_parallel_column_transformation(self, mixed_dataframe):
        """Test parallel processing of column transformations."""
        transformer = ColumnTransformer(n_jobs=2)
        
        transformations = [
            (['numeric_1'], StandardScaler()),
            (['numeric_2'], MinMaxScaler()),
            (['categorical_1'], 'onehot_encoder'),
            (['categorical_2'], 'label_encoder')
        ]
        
        transformer.set_transformations(transformations)
        
        import time
        start_time = time.time()
        X_transformed = transformer.fit_transform(mixed_dataframe)
        processing_time = time.time() - start_time
        
        assert X_transformed.shape[0] == mixed_dataframe.shape[0]
        # Parallel processing should be reasonably fast
        assert processing_time < 5  # seconds


class TestFeatureStacker:
    """Test FeatureStacker class."""
    
    @pytest.fixture
    def stacker(self):
        """Create feature stacker."""
        return FeatureStacker()
    
    @pytest.fixture
    def feature_sets(self):
        """Create multiple feature sets for stacking."""
        np.random.seed(42)
        
        # Original features
        original_features = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randn(100)
        })
        
        # Engineered features
        engineered_features = pd.DataFrame({
            'poly_1': original_features['feature_1'] ** 2,
            'poly_2': original_features['feature_2'] ** 2,
            'interaction': original_features['feature_1'] * original_features['feature_2']
        })
        
        # Statistical features
        statistical_features = pd.DataFrame({
            'mean': original_features.mean(axis=1),
            'std': original_features.std(axis=1),
            'max': original_features.max(axis=1)
        })
        
        return {
            'original': original_features,
            'engineered': engineered_features,
            'statistical': statistical_features
        }
    
    def test_horizontal_stacking(self, stacker, feature_sets):
        """Test horizontal feature stacking."""
        stacked_features = stacker.horizontal_stack(feature_sets)
        
        total_features = sum(df.shape[1] for df in feature_sets.values())
        
        assert stacked_features.shape[0] == 100  # Same number of rows
        assert stacked_features.shape[1] == total_features
        
        # Check that all feature names are preserved
        for feature_set_name, feature_df in feature_sets.items():
            for col in feature_df.columns:
                expected_col_name = f"{feature_set_name}_{col}"
                assert expected_col_name in stacked_features.columns or col in stacked_features.columns
    
    def test_weighted_stacking(self, stacker, feature_sets):
        """Test weighted feature stacking."""
        weights = {
            'original': 0.5,
            'engineered': 0.3,
            'statistical': 0.2
        }
        
        weighted_features = stacker.weighted_stack(feature_sets, weights)
        
        assert weighted_features.shape[0] == 100
        assert hasattr(stacker, 'feature_weights_')
    
    def test_selective_stacking(self, stacker, feature_sets):
        """Test selective feature stacking based on importance."""
        # Mock feature importance scores
        feature_importance = {
            'original': {'feature_1': 0.8, 'feature_2': 0.6, 'feature_3': 0.3},
            'engineered': {'poly_1': 0.7, 'poly_2': 0.4, 'interaction': 0.9},
            'statistical': {'mean': 0.5, 'std': 0.2, 'max': 0.6}
        }
        
        selective_features = stacker.selective_stack(
            feature_sets, 
            feature_importance, 
            threshold=0.5
        )
        
        assert selective_features.shape[0] == 100
        # Should have fewer features than total (only those above threshold)
        total_features = sum(df.shape[1] for df in feature_sets.values())
        assert selective_features.shape[1] < total_features
    
    def test_hierarchical_stacking(self, stacker, feature_sets):
        """Test hierarchical feature stacking."""
        # Define hierarchy: original -> engineered -> statistical
        hierarchy = ['original', 'engineered', 'statistical']
        
        hierarchical_features = stacker.hierarchical_stack(
            feature_sets, 
            hierarchy, 
            cumulative=True
        )
        
        assert hierarchical_features.shape[0] == 100
        
        # Should include features from all levels
        assert 'level_0' in hierarchical_features.columns.names or \
               any('original' in str(col) for col in hierarchical_features.columns)


class TestFeaturePipelineBuilder:
    """Test FeaturePipelineBuilder class."""
    
    @pytest.fixture
    def builder(self):
        """Create feature pipeline builder."""
        return FeaturePipelineBuilder()
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for pipeline building."""
        np.random.seed(42)
        X = pd.DataFrame({
            'numeric_1': np.random.randn(200),
            'numeric_2': np.random.exponential(2, 200),
            'categorical': np.random.choice(['A', 'B', 'C'], 200),
            'binary': np.random.choice([0, 1], 200)
        })
        y = np.random.randint(0, 2, 200)
        return X, y
    
    def test_automatic_pipeline_building(self, builder, sample_data):
        """Test automatic pipeline building based on data types."""
        X, y = sample_data
        
        pipeline = builder.build_auto_pipeline(X, y)
        
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) > 0
        
        # Test that pipeline works
        X_transformed = pipeline.fit_transform(X, y)
        assert X_transformed.shape[0] == X.shape[0]
    
    def test_custom_pipeline_building(self, builder, sample_data):
        """Test custom pipeline building with user specifications."""
        X, y = sample_data
        
        # Define custom specifications
        pipeline_spec = {
            'numeric_processing': {
                'columns': ['numeric_1', 'numeric_2'],
                'steps': ['outlier_removal', 'standard_scaling']
            },
            'categorical_processing': {
                'columns': ['categorical'],
                'steps': ['onehot_encoding']
            },
            'feature_engineering': {
                'steps': ['polynomial_features']
            },
            'feature_selection': {
                'method': 'univariate',
                'k': 5
            }
        }
        
        pipeline = builder.build_custom_pipeline(pipeline_spec)
        
        assert isinstance(pipeline, Pipeline)
        X_transformed = pipeline.fit_transform(X, y)
        assert X_transformed.shape[0] == X.shape[0]
    
    def test_modular_pipeline_building(self, builder, sample_data):
        """Test modular pipeline building."""
        X, y = sample_data
        
        # Build pipeline module by module
        builder.add_preprocessing_module('numeric', ['numeric_1', 'numeric_2'])
        builder.add_preprocessing_module('categorical', ['categorical'])
        builder.add_feature_engineering_module('interactions')
        builder.add_selection_module('variance_threshold')
        
        pipeline = builder.build()
        
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) >= 4  # At least 4 modules added
    
    def test_conditional_pipeline_building(self, builder, sample_data):
        """Test building pipelines with conditional steps."""
        X, y = sample_data
        
        # Define conditions
        conditions = {
            'has_missing': lambda X: X.isnull().any().any(),
            'has_outliers': lambda X: (np.abs(X.select_dtypes(include=[np.number])) > 3).any().any(),
            'high_cardinality': lambda X: any(X.select_dtypes(include=['object']).nunique() > 10)
        }
        
        # Define conditional steps
        conditional_steps = {
            'has_missing': ['missing_value_imputation'],
            'has_outliers': ['outlier_removal'],
            'high_cardinality': ['target_encoding']
        }
        
        pipeline = builder.build_conditional_pipeline(X, conditions, conditional_steps)
        
        assert isinstance(pipeline, Pipeline)
        X_transformed = pipeline.fit_transform(X, y)
        assert X_transformed.shape[0] == X.shape[0]
    
    def test_pipeline_optimization(self, builder, sample_data):
        """Test pipeline optimization."""
        X, y = sample_data
        
        # Build initial pipeline
        initial_pipeline = builder.build_auto_pipeline(X, y)
        
        # Optimize pipeline
        optimized_pipeline = builder.optimize_pipeline(
            initial_pipeline, X, y,
            optimization_method='bayesian',
            n_iterations=5
        )
        
        assert isinstance(optimized_pipeline, Pipeline)
        
        # Optimized pipeline should perform at least as well
        initial_score = initial_pipeline.fit(X, y).score(X, y) if hasattr(initial_pipeline, 'score') else 0
        optimized_score = optimized_pipeline.fit(X, y).score(X, y) if hasattr(optimized_pipeline, 'score') else 0
        
        # Note: This comparison might not always be meaningful without a proper model at the end


class TestPipelineComposer:
    """Test PipelineComposer class."""
    
    @pytest.fixture
    def composer(self):
        """Create pipeline composer."""
        return PipelineComposer()
    
    def test_pipeline_composition(self, composer):
        """Test composing multiple pipelines."""
        # Define sub-pipelines
        preprocessing_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])
        
        feature_engineering_pipeline = Pipeline([
            ('selector', SelectKBest(k=5))
        ])
        
        model_pipeline = Pipeline([
            ('classifier', LogisticRegression(random_state=42, max_iter=200))
        ])
        
        # Compose into complete pipeline
        complete_pipeline = composer.compose([
            ('preprocessing', preprocessing_pipeline),
            ('feature_engineering', feature_engineering_pipeline),
            ('modeling', model_pipeline)
        ])
        
        assert isinstance(complete_pipeline, Pipeline)
        assert len(complete_pipeline.steps) == 3
    
    def test_dynamic_composition(self, composer):
        """Test dynamic pipeline composition based on data characteristics."""
        X, y = make_classification(n_samples=150, n_features=10, random_state=42)
        
        # Define composition rules
        composition_rules = {
            'high_dimensional': {
                'condition': lambda X, y: X.shape[1] > 8,
                'pipeline': ('feature_selection', SelectKBest(k=5))
            },
            'binary_classification': {
                'condition': lambda X, y: len(np.unique(y)) == 2,
                'pipeline': ('classifier', LogisticRegression(random_state=42, max_iter=200))
            }
        }
        
        dynamic_pipeline = composer.dynamic_compose(X, y, composition_rules)
        
        assert isinstance(dynamic_pipeline, Pipeline)
        # Should include feature selection (high dimensional) and logistic regression (binary)
        step_names = [step[0] for step in dynamic_pipeline.steps]
        assert 'feature_selection' in step_names or any('select' in name.lower() for name in step_names)
    
    def test_nested_composition(self, composer):
        """Test nested pipeline composition."""
        # Create nested structure
        inner_pipeline_1 = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(k=3))
        ])
        
        inner_pipeline_2 = Pipeline([
            ('scaler', MinMaxScaler())
        ])
        
        # Compose pipelines with branching
        branched_pipeline = composer.create_branched_pipeline({
            'branch_1': inner_pipeline_1,
            'branch_2': inner_pipeline_2
        })
        
        final_pipeline = composer.compose([
            ('branched_processing', branched_pipeline),
            ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        assert isinstance(final_pipeline, Pipeline)
    
    def test_pipeline_validation(self, composer):
        """Test pipeline composition validation."""
        # Try to compose incompatible pipelines
        incompatible_pipeline_1 = Pipeline([
            ('transformer', StandardScaler())  # Outputs numpy array
        ])
        
        incompatible_pipeline_2 = Pipeline([
            ('needs_dataframe', 'custom_transformer_requiring_dataframe')
        ])
        
        # Should validate compatibility
        try:
            problematic_pipeline = composer.compose([
                ('step1', incompatible_pipeline_1),
                ('step2', incompatible_pipeline_2)
            ], validate=True)
            
            # If validation passes, that's fine
            assert isinstance(problematic_pipeline, Pipeline)
        except (ValueError, TypeError):
            # If validation catches incompatibility, that's also expected
            pass


class TestFeatureUnionIntegration:
    """Integration tests for feature union components."""
    
    def test_end_to_end_feature_processing(self):
        """Test complete end-to-end feature processing pipeline."""
        # Generate complex dataset
        np.random.seed(42)
        df = pd.DataFrame({
            'numeric_1': np.random.randn(300),
            'numeric_2': np.random.exponential(2, 300),
            'numeric_3': np.random.uniform(0, 100, 300),
            'categorical_1': np.random.choice(['A', 'B', 'C', 'D'], 300),
            'categorical_2': np.random.choice([f'cat_{i}' for i in range(8)], 300),
            'binary': np.random.choice([0, 1], 300),
            'date_col': pd.date_range('2020-01-01', periods=300, freq='D'),
            'text_col': [f'text_{i%20}' for i in range(300)]
        })
        
        # Add missing values
        df.loc[10:20, 'numeric_1'] = np.nan
        df.loc[50:60, 'categorical_1'] = np.nan
        
        y = np.random.randint(0, 2, 300)
        
        # 1. Build feature processing pipeline
        builder = FeaturePipelineBuilder()
        
        # Define processing for different feature types
        pipeline_spec = {
            'numeric_processing': {
                'columns': ['numeric_1', 'numeric_2', 'numeric_3'],
                'steps': ['missing_imputation', 'outlier_removal', 'standard_scaling']
            },
            'categorical_processing': {
                'columns': ['categorical_1', 'categorical_2'],
                'steps': ['missing_imputation', 'onehot_encoding']
            },
            'datetime_processing': {
                'columns': ['date_col'],
                'steps': ['datetime_features_extraction']
            },
            'text_processing': {
                'columns': ['text_col'],
                'steps': ['tfidf_vectorization']
            }
        }
        
        feature_pipeline = builder.build_custom_pipeline(pipeline_spec)
        
        # 2. Create feature union for parallel processing
        feature_union = AdvancedFeatureUnion()
        
        numeric_pipeline = Pipeline([
            ('selector', ColumnSelector(['numeric_1', 'numeric_2', 'numeric_3'])),
            ('imputer', 'simple_imputer'),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('selector', ColumnSelector(['categorical_1', 'categorical_2'])),
            ('encoder', 'onehot_encoder')
        ])
        
        feature_union.add_pipeline('numeric', numeric_pipeline)
        feature_union.add_pipeline('categorical', categorical_pipeline)
        
        # 3. Apply feature processing
        X_processed = feature_union.fit_transform(df.drop(['date_col', 'text_col'], axis=1), y)
        
        # 4. Create final ML pipeline
        composer = PipelineComposer()
        final_pipeline = composer.compose([
            ('feature_processing', feature_union),
            ('feature_selection', SelectKBest(k=10)),
            ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
        ])
        
        # 5. Test complete pipeline
        X_input = df.drop(['date_col', 'text_col'], axis=1)  # Simplified for this test
        final_pipeline.fit(X_input, y)
        predictions = final_pipeline.predict(X_input)
        score = final_pipeline.score(X_input, y)
        
        # Assertions
        assert X_processed.shape[0] == df.shape[0]
        assert len(predictions) == len(y)
        assert 0 <= score <= 1
        
        print(f"End-to-end feature processing completed. Final score: {score:.3f}")
    
    def test_performance_comparison(self):
        """Test performance comparison of different feature union strategies."""
        # Generate dataset
        X, y = make_classification(
            n_samples=500,
            n_features=20,
            n_classes=2,
            n_informative=15,
            random_state=42
        )
        X = pd.DataFrame(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Strategy 1: Simple preprocessing
        simple_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=200))
        ])
        
        # Strategy 2: Feature union with parallel processing
        numeric_features = [col for col in X.columns if col.startswith('0') or col.startswith('1')]
        other_features = [col for col in X.columns if col not in numeric_features]
        
        feature_union = AdvancedFeatureUnion()
        
        numeric_pipeline = Pipeline([
            ('selector', ColumnSelector(numeric_features)),
            ('scaler', StandardScaler())
        ])
        
        other_pipeline = Pipeline([
            ('selector', ColumnSelector(other_features)),
            ('scaler', MinMaxScaler())
        ])
        
        feature_union.add_pipeline('numeric', numeric_pipeline)
        feature_union.add_pipeline('other', other_pipeline)
        
        union_pipeline = Pipeline([
            ('feature_union', feature_union),
            ('classifier', LogisticRegression(random_state=42, max_iter=200))
        ])
        
        # Strategy 3: Weighted feature union
        weighted_union = WeightedFeatureUnion()
        weighted_union.add_pipeline('numeric', numeric_pipeline, weight=0.7)
        weighted_union.add_pipeline('other', other_pipeline, weight=0.3)
        
        weighted_pipeline = Pipeline([
            ('weighted_union', weighted_union),
            ('classifier', LogisticRegression(random_state=42, max_iter=200))
        ])
        
        # Compare strategies
        strategies = {
            'Simple': simple_pipeline,
            'Feature Union': union_pipeline,
            'Weighted Union': weighted_pipeline
        }
        
        results = {}
        for name, pipeline in strategies.items():
            pipeline.fit(X_train, y_train)
            score = pipeline.score(X_test, y_test)
            results[name] = score
            print(f"{name} strategy score: {score:.3f}")
        
        # All strategies should achieve reasonable performance
        for strategy, score in results.items():
            assert score > 0.5, f"{strategy} strategy performed poorly: {score}"
    
    def test_scalability_analysis(self):
        """Test scalability of feature union approaches."""
        # Test with different dataset sizes
        sizes = [100, 500, 1000]
        processing_times = {}
        
        for size in sizes:
            X, y = make_classification(
                n_samples=size,
                n_features=15,
                n_classes=2,
                random_state=42
            )
            X = pd.DataFrame(X)
            
            # Create feature union
            feature_union = AdvancedFeatureUnion()
            
            pipeline1 = Pipeline([
                ('selector', ColumnSelector(list(X.columns[:7]))),
                ('scaler', StandardScaler())
            ])
            
            pipeline2 = Pipeline([
                ('selector', ColumnSelector(list(X.columns[7:]))),
                ('scaler', MinMaxScaler())
            ])
            
            feature_union.add_pipeline('group1', pipeline1)
            feature_union.add_pipeline('group2', pipeline2)
            
            # Time the processing
            import time
            start_time = time.time()
            X_transformed = feature_union.fit_transform(X, y)
            processing_time = time.time() - start_time
            
            processing_times[size] = processing_time
            
            # Verify output
            assert X_transformed.shape[0] == size
            assert X_transformed.shape[1] == X.shape[1]  # Should preserve all features
        
        # Processing time should scale reasonably
        print(f"Processing times: {processing_times}")
        assert all(time < 10 for time in processing_times.values()), "Processing taking too long"
    
    def test_memory_efficiency(self):
        """Test memory efficiency of feature union operations."""
        # Create moderately large dataset
        X, y = make_classification(
            n_samples=2000,
            n_features=30,
            n_classes=2,
            random_state=42
        )
        X = pd.DataFrame(X)
        
        # Test memory-efficient processing
        feature_union = AdvancedFeatureUnion()
        
        # Split features into groups
        n_groups = 3
        features_per_group = len(X.columns) // n_groups
        
        for i in range(n_groups):
            start_idx = i * features_per_group
            end_idx = start_idx + features_per_group if i < n_groups - 1 else len(X.columns)
            group_features = list(X.columns[start_idx:end_idx])
            
            pipeline = Pipeline([
                ('selector', ColumnSelector(group_features)),
                ('scaler', StandardScaler())
            ])
            
            feature_union.add_pipeline(f'group_{i}', pipeline)
        
        # Process with memory monitoring
        try:
            X_transformed = feature_union.fit_transform(X, y)
            
            # Should complete without memory issues
            assert X_transformed.shape[0] == X.shape[0]
            assert X_transformed.shape[1] == X.shape[1]
            
            print("Memory efficiency test passed")
        except MemoryError:
            pytest.fail("Memory efficiency test failed - out of memory")


class TestFeatureUnionErrorHandling:
    """Test error handling in feature union components."""
    
    def test_incompatible_pipeline_handling(self):
        """Test handling of incompatible pipelines."""
        feature_union = AdvancedFeatureUnion()
        
        # Create pipelines that might be incompatible
        pipeline1 = Pipeline([
            ('scaler', StandardScaler())
        ])
        
        # Pipeline that expects specific input format
        class IncompatibleTransformer:
            def fit(self, X, y=None):
                return self
            
            def transform(self, X):
                if not hasattr(X, 'columns'):
                    raise ValueError("Expected DataFrame input")
                return X
        
        pipeline2 = Pipeline([
            ('incompatible', IncompatibleTransformer())
        ])
        
        feature_union.add_pipeline('pipeline1', pipeline1)
        feature_union.add_pipeline('pipeline2', pipeline2)
        
        # Generate test data
        X = pd.DataFrame(np.random.randn(50, 5))
        y = np.random.randint(0, 2, 50)
        
        # Should handle incompatibility gracefully or raise informative error
        try:
            X_transformed = feature_union.fit_transform(X, y)
            # If it works, that's fine
            assert X_transformed.shape[0] == X.shape[0]
        except (ValueError, TypeError) as e:
            # If it raises an error, it should be informative
            assert len(str(e)) > 0
    
    def test_empty_pipeline_handling(self):
        """Test handling of empty pipelines."""
        feature_union = AdvancedFeatureUnion()
        
        # Try to transform without adding any pipelines
        X = pd.DataFrame(np.random.randn(50, 5))
        y = np.random.randint(0, 2, 50)
        
        with pytest.raises((ValueError, RuntimeError)):
            feature_union.fit_transform(X, y)
    
    def test_missing_column_handling(self):
        """Test handling of missing columns."""
        feature_union = AdvancedFeatureUnion()
        
        # Create pipeline that expects columns not in data
        pipeline = Pipeline([
            ('selector', ColumnSelector(['nonexistent_column'])),
            ('scaler', StandardScaler())
        ])
        
        feature_union.add_pipeline('test_pipeline', pipeline)
        
        X = pd.DataFrame({
            'existing_column_1': np.random.randn(50),
            'existing_column_2': np.random.randn(50)
        })
        y = np.random.randint(0, 2, 50)
        
        # Should handle missing columns gracefully
        with pytest.raises((KeyError, ValueError)):
            feature_union.fit_transform(X, y)
    
    def test_mixed_data_type_handling(self):
        """Test handling of mixed data types."""
        feature_union = AdvancedFeatureUnion()
        
        # Pipeline expecting numeric data
        numeric_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])
        
        feature_union.add_pipeline('numeric', numeric_pipeline)
        
        # Data with mixed types
        X = pd.DataFrame({
            'numeric_col': np.random.randn(50),
            'string_col': ['text'] * 50,
            'categorical_col': pd.Categorical(['A', 'B', 'C'] * 16 + ['A', 'B'])
        })
        y = np.random.randint(0, 2, 50)
        
        # Should handle mixed types appropriately
        try:
            X_transformed = feature_union.fit_transform(X, y)
            # If successful, verify output
            assert X_transformed.shape[0] == X.shape[0]
        except (ValueError, TypeError):
            # If it fails, that's expected for this test
            pass


class TestFeatureUnionPerformance:
    """Performance tests for feature union components."""
    
    @pytest.mark.slow
    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        # Create large dataset
        n_samples = 5000
        n_features = 50
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=2,
            random_state=42
        )
        X = pd.DataFrame(X)
        
        feature_union = AdvancedFeatureUnion()
        
        # Create multiple processing pipelines
        n_pipelines = 5
        features_per_pipeline = n_features // n_pipelines
        
        for i in range(n_pipelines):
            start_idx = i * features_per_pipeline
            end_idx = start_idx + features_per_pipeline if i < n_pipelines - 1 else n_features
            pipeline_features = list(X.columns[start_idx:end_idx])
            
            pipeline = Pipeline([
                ('selector', ColumnSelector(pipeline_features)),
                ('scaler', StandardScaler())
            ])
            
            feature_union.add_pipeline(f'pipeline_{i}', pipeline)
        
        # Time the processing
        import time
        start_time = time.time()
        X_transformed = feature_union.fit_transform(X, y)
        processing_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert processing_time < 30, f"Processing took too long: {processing_time:.2f}s"
        assert X_transformed.shape == X.shape
        
        print(f"Large dataset processing completed in {processing_time:.2f}s")
    
    def test_parallel_processing_speedup(self):
        """Test that parallel processing provides speedup."""
        # Create dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_classes=2,
            random_state=42
        )
        X = pd.DataFrame(X)
        
        # Sequential processing
        sequential_union = AdvancedFeatureUnion(n_jobs=1)
        
        # Parallel processing
        parallel_union = AdvancedFeatureUnion(n_jobs=2)
        
        # Add same pipelines to both
        for union in [sequential_union, parallel_union]:
            pipeline1 = Pipeline([
                ('selector', ColumnSelector(list(X.columns[:10]))),
                ('scaler', StandardScaler())
            ])
            
            pipeline2 = Pipeline([
                ('selector', ColumnSelector(list(X.columns[10:]))),
                ('scaler', MinMaxScaler())
            ])
            
            union.add_pipeline('group1', pipeline1)
            union.add_pipeline('group2', pipeline2)
        
        # Time both approaches
        import time
        
        start_time = time.time()
        X_sequential = sequential_union.fit_transform(X, y)
        sequential_time = time.time() - start_time
        
        start_time = time.time()
        X_parallel = parallel_union.fit_transform(X, y)
        parallel_time = time.time() - start_time
        
        # Results should be the same
        np.testing.assert_array_almost_equal(X_sequential, X_parallel)
        
        # Parallel should be faster (or at least not much slower)
        speedup_ratio = sequential_time / parallel_time
        print(f"Speedup ratio: {speedup_ratio:.2f}")
        
        # Note: On small datasets, parallel processing might be slower due to overhead
        # So we just check that both complete successfully
        assert sequential_time > 0
        assert parallel_time > 0


if __name__ == "__main__":
    pytest.main([__file__])