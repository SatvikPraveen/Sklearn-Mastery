"""
Unit tests for pipeline factory.

Tests for automated pipeline creation and configuration.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pipelines.pipeline_factory import (
    PipelineFactory,
    AutoMLPipelineBuilder,
    ClassificationPipelineFactory,
    RegressionPipelineFactory,
    CustomPipelineBuilder,
    PipelineConfig,
    PipelineOptimizer
)


class TestPipelineFactory:
    """Test PipelineFactory class."""
    
    @pytest.fixture
    def factory(self):
        """Create factory instance."""
        return PipelineFactory()
    
    @pytest.fixture
    def classification_data(self):
        """Generate classification test data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=3,
            n_informative=6,
            random_state=42
        )
        return pd.DataFrame(X), y
    
    @pytest.fixture
    def regression_data(self):
        """Generate regression test data."""
        X, y = make_regression(
            n_samples=200,
            n_features=8,
            noise=0.1,
            random_state=42
        )
        return pd.DataFrame(X), y
    
    def test_basic_classification_pipeline(self, factory, classification_data):
        """Test basic classification pipeline creation."""
        X, y = classification_data
        
        pipeline = factory.create_classification_pipeline(
            task_type='classification',
            model_type='random_forest'
        )
        
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) > 0
        
        # Test that pipeline works
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        assert len(predictions) == len(X)
    
    def test_basic_regression_pipeline(self, factory, regression_data):
        """Test basic regression pipeline creation."""
        X, y = regression_data
        
        pipeline = factory.create_regression_pipeline(
            task_type='regression',
            model_type='random_forest'
        )
        
        assert isinstance(pipeline, Pipeline)
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        assert len(predictions) == len(X)
    
    def test_custom_configuration(self, factory, classification_data):
        """Test pipeline with custom configuration."""
        X, y = classification_data
        
        config = {
            'preprocessing': {
                'scaler': 'standard',
                'feature_selection': 'univariate',
                'handle_missing': True
            },
            'model': {
                'type': 'logistic_regression',
                'hyperparameters': {'C': 1.0, 'random_state': 42}
            }
        }
        
        pipeline = factory.create_pipeline(config)
        
        assert isinstance(pipeline, Pipeline)
        pipeline.fit(X, y)
        score = pipeline.score(X, y)
        assert 0 <= score <= 1
    
    def test_auto_preprocessing_detection(self, factory):
        """Test automatic preprocessing step detection."""
        # Create data with specific characteristics
        df = pd.DataFrame({
            'numeric_col': np.random.randn(100),
            'categorical_col': np.random.choice(['A', 'B', 'C'], 100),
            'missing_col': [1 if i % 5 != 0 else np.nan for i in range(100)],
            'outlier_col': [100 if i == 99 else np.random.randn() for i in range(100)]
        })
        y = np.random.randint(0, 2, 100)
        
        pipeline = factory.auto_create_pipeline(df, y)
        
        assert isinstance(pipeline, Pipeline)
        # Should detect and handle missing values, categorical encoding, etc.
        step_names = [step[0] for step in pipeline.steps]
        assert any('missing' in name or 'imputer' in name for name in step_names)
    
    def test_pipeline_step_customization(self, factory, classification_data):
        """Test customizing individual pipeline steps."""
        X, y = classification_data
        
        custom_steps = {
            'preprocessing': ['standard_scaler', 'feature_selection'],
            'feature_engineering': ['polynomial_features'],
            'model': 'gradient_boosting'
        }
        
        pipeline = factory.create_custom_pipeline(custom_steps)
        
        assert isinstance(pipeline, Pipeline)
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        assert len(predictions) == len(X)
    
    def test_pipeline_validation(self, factory, classification_data):
        """Test pipeline configuration validation."""
        X, y = classification_data
        
        # Invalid configuration
        invalid_config = {
            'model': {'type': 'invalid_model_type'}
        }
        
        with pytest.raises((ValueError, KeyError)):
            factory.create_pipeline(invalid_config)
    
    def test_pipeline_serialization(self, factory, classification_data):
        """Test pipeline serialization and loading."""
        X, y = classification_data
        
        pipeline = factory.create_classification_pipeline(model_type='logistic_regression')
        pipeline.fit(X, y)
        
        # Test that pipeline can be serialized
        import pickle
        serialized = pickle.dumps(pipeline)
        loaded_pipeline = pickle.loads(serialized)
        
        # Predictions should be the same
        original_preds = pipeline.predict(X)
        loaded_preds = loaded_pipeline.predict(X)
        np.testing.assert_array_equal(original_preds, loaded_preds)


class TestAutoMLPipelineBuilder:
    """Test AutoMLPipelineBuilder class."""
    
    @pytest.fixture
    def builder(self):
        """Create builder instance."""
        return AutoMLPipelineBuilder()
    
    @pytest.fixture
    def mixed_data(self):
        """Generate mixed data types for testing."""
        np.random.seed(42)
        df = pd.DataFrame({
            'numeric_1': np.random.randn(150),
            'numeric_2': np.random.exponential(2, 150),
            'categorical_1': np.random.choice(['A', 'B', 'C', 'D'], 150),
            'categorical_2': np.random.choice([f'cat_{i}' for i in range(10)], 150),
            'binary': np.random.choice([0, 1], 150),
            'date_col': pd.date_range('2020-01-01', periods=150, freq='D'),
            'text_col': [f'text_{i%5}' for i in range(150)]
        })
        
        # Add missing values
        df.loc[10:15, 'numeric_1'] = np.nan
        df.loc[20:25, 'categorical_1'] = np.nan
        
        y = np.random.randint(0, 3, 150)
        return df, y
    
    def test_data_profiling(self, builder, mixed_data):
        """Test automatic data profiling."""
        X, y = mixed_data
        
        profile = builder.profile_data(X, y)
        
        assert 'numeric_columns' in profile
        assert 'categorical_columns' in profile
        assert 'datetime_columns' in profile
        assert 'missing_percentage' in profile
        assert 'target_type' in profile
    
    def test_preprocessing_recommendation(self, builder, mixed_data):
        """Test preprocessing step recommendation."""
        X, y = mixed_data
        
        recommendations = builder.recommend_preprocessing(X, y)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should recommend handling missing values
        step_names = [step['name'] for step in recommendations]
        assert any('missing' in name.lower() for name in step_names)
    
    def test_model_recommendation(self, builder, mixed_data):
        """Test model recommendation based on data characteristics."""
        X, y = mixed_data
        
        recommended_models = builder.recommend_models(X, y)
        
        assert isinstance(recommended_models, list)
        assert len(recommended_models) > 0
        
        for model_info in recommended_models:
            assert 'name' in model_info
            assert 'score' in model_info or 'priority' in model_info
    
    def test_full_automl_pipeline(self, builder, mixed_data):
        """Test complete AutoML pipeline creation."""
        X, y = mixed_data
        
        pipeline = builder.build_pipeline(X, y, max_time_mins=5)
        
        assert isinstance(pipeline, Pipeline)
        
        # Test pipeline performance
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        
        assert 0 <= score <= 1
    
    def test_feature_engineering_recommendations(self, builder, mixed_data):
        """Test feature engineering recommendations."""
        X, y = mixed_data
        
        feature_engineering = builder.recommend_feature_engineering(X, y)
        
        assert isinstance(feature_engineering, list)
        # Should recommend datetime feature extraction
        fe_names = [fe['name'] for fe in feature_engineering]
        assert any('datetime' in name.lower() for name in fe_names)
    
    def test_hyperparameter_optimization(self, builder, mixed_data):
        """Test hyperparameter optimization suggestion."""
        X, y = mixed_data
        
        # Create basic pipeline first
        pipeline = builder.build_pipeline(X, y, max_time_mins=1)
        
        # Get hyperparameter suggestions
        hp_suggestions = builder.suggest_hyperparameter_tuning(pipeline, X, y)
        
        assert isinstance(hp_suggestions, dict)
        assert 'param_grid' in hp_suggestions or 'param_distributions' in hp_suggestions


class TestClassificationPipelineFactory:
    """Test ClassificationPipelineFactory class."""
    
    @pytest.fixture
    def factory(self):
        """Create classification factory."""
        return ClassificationPipelineFactory()
    
    @pytest.fixture
    def binary_classification_data(self):
        """Generate binary classification data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        return pd.DataFrame(X), y
    
    @pytest.fixture
    def multiclass_data(self):
        """Generate multiclass classification data."""
        X, y = make_classification(
            n_samples=200,
            n_features=15,
            n_classes=5,
            random_state=42
        )
        return pd.DataFrame(X), y
    
    @pytest.fixture
    def imbalanced_data(self):
        """Generate imbalanced classification data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=2,
            weights=[0.9, 0.1],
            random_state=42
        )
        return pd.DataFrame(X), y
    
    def test_binary_classification_pipeline(self, factory, binary_classification_data):
        """Test binary classification pipeline."""
        X, y = binary_classification_data
        
        pipeline = factory.create_binary_classification_pipeline(
            algorithm='logistic_regression'
        )
        
        assert isinstance(pipeline, Pipeline)
        pipeline.fit(X, y)
        
        # Test probability predictions
        probabilities = pipeline.predict_proba(X)
        assert probabilities.shape == (len(X), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_multiclass_classification_pipeline(self, factory, multiclass_data):
        """Test multiclass classification pipeline."""
        X, y = multiclass_data
        
        pipeline = factory.create_multiclass_pipeline(
            algorithm='random_forest'
        )
        
        assert isinstance(pipeline, Pipeline)
        pipeline.fit(X, y)
        
        predictions = pipeline.predict(X)
        assert len(np.unique(predictions)) <= len(np.unique(y))
    
    def test_imbalanced_classification_pipeline(self, factory, imbalanced_data):
        """Test pipeline for imbalanced classification."""
        X, y = imbalanced_data
        
        pipeline = factory.create_imbalanced_pipeline(
            algorithm='random_forest',
            balancing_strategy='smote'
        )
        
        assert isinstance(pipeline, Pipeline)
        pipeline.fit(X, y)
        
        predictions = pipeline.predict(X)
        assert len(predictions) == len(X)
    
    def test_ensemble_classification_pipeline(self, factory, binary_classification_data):
        """Test ensemble classification pipeline."""
        X, y = binary_classification_data
        
        pipeline = factory.create_ensemble_pipeline(
            base_models=['logistic_regression', 'random_forest', 'svm'],
            ensemble_method='voting'
        )
        
        assert isinstance(pipeline, Pipeline)
        pipeline.fit(X, y)
        
        predictions = pipeline.predict(X)
        probabilities = pipeline.predict_proba(X)
        
        assert len(predictions) == len(X)
        assert probabilities.shape == (len(X), 2)
    
    def test_text_classification_pipeline(self, factory):
        """Test text classification pipeline."""
        # Create text data
        texts = [
            'This is a positive review',
            'This is a negative review',
            'Another positive comment',
            'Another negative comment'
        ] * 50
        
        labels = [1, 0, 1, 0] * 50
        
        X = pd.DataFrame({'text': texts})
        y = np.array(labels)
        
        pipeline = factory.create_text_classification_pipeline(
            text_column='text',
            vectorizer='tfidf'
        )
        
        assert isinstance(pipeline, Pipeline)
        pipeline.fit(X, y)
        
        predictions = pipeline.predict(X)
        assert len(predictions) == len(X)


class TestRegressionPipelineFactory:
    """Test RegressionPipelineFactory class."""
    
    @pytest.fixture
    def factory(self):
        """Create regression factory."""
        return RegressionPipelineFactory()
    
    @pytest.fixture
    def linear_regression_data(self):
        """Generate linear regression data."""
        X, y = make_regression(
            n_samples=200,
            n_features=5,
            noise=0.1,
            random_state=42
        )
        return pd.DataFrame(X), y
    
    @pytest.fixture
    def nonlinear_regression_data(self):
        """Generate nonlinear regression data."""
        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = X[:, 0]**2 + 2*X[:, 1] + np.sin(X[:, 2]) + np.random.randn(200) * 0.1
        return pd.DataFrame(X), y
    
    def test_linear_regression_pipeline(self, factory, linear_regression_data):
        """Test linear regression pipeline."""
        X, y = linear_regression_data
        
        pipeline = factory.create_linear_regression_pipeline(
            regularization='ridge'
        )
        
        assert isinstance(pipeline, Pipeline)
        pipeline.fit(X, y)
        
        predictions = pipeline.predict(X)
        assert len(predictions) == len(X)
        
        # Check R² score
        score = pipeline.score(X, y)
        assert score > 0.5  # Should be reasonably good
    
    def test_nonlinear_regression_pipeline(self, factory, nonlinear_regression_data):
        """Test nonlinear regression pipeline."""
        X, y = nonlinear_regression_data
        
        pipeline = factory.create_nonlinear_regression_pipeline(
            algorithm='random_forest'
        )
        
        assert isinstance(pipeline, Pipeline)
        pipeline.fit(X, y)
        
        predictions = pipeline.predict(X)
        score = pipeline.score(X, y)
        
        assert len(predictions) == len(X)
        assert score > 0.3  # Should capture some nonlinearity
    
    def test_polynomial_regression_pipeline(self, factory, linear_regression_data):
        """Test polynomial regression pipeline."""
        X, y = linear_regression_data
        
        pipeline = factory.create_polynomial_regression_pipeline(
            degree=2,
            include_interactions=True
        )
        
        assert isinstance(pipeline, Pipeline)
        pipeline.fit(X, y)
        
        predictions = pipeline.predict(X)
        assert len(predictions) == len(X)
    
    def test_ensemble_regression_pipeline(self, factory, linear_regression_data):
        """Test ensemble regression pipeline."""
        X, y = linear_regression_data
        
        pipeline = factory.create_ensemble_regression_pipeline(
            base_models=['linear_regression', 'random_forest', 'gradient_boosting'],
            ensemble_method='stacking'
        )
        
        assert isinstance(pipeline, Pipeline)
        pipeline.fit(X, y)
        
        predictions = pipeline.predict(X)
        score = pipeline.score(X, y)
        
        assert len(predictions) == len(X)
        assert score > 0.0


class TestCustomPipelineBuilder:
    """Test CustomPipelineBuilder class."""
    
    @pytest.fixture
    def builder(self):
        """Create custom builder."""
        return CustomPipelineBuilder()
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        X, y = make_classification(n_samples=100, n_features=8, random_state=42)
        return pd.DataFrame(X), y
    
    def test_step_by_step_building(self, builder, sample_data):
        """Test building pipeline step by step."""
        X, y = sample_data
        
        # Build pipeline incrementally
        builder.add_preprocessing_step('standard_scaler')
        builder.add_feature_engineering_step('polynomial_features', degree=2)
        builder.add_feature_selection_step('univariate', k=5)
        builder.add_model_step('logistic_regression', C=1.0)
        
        pipeline = builder.build()
        
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 4
        
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        assert len(predictions) == len(X)
    
    def test_conditional_steps(self, builder, sample_data):
        """Test conditional pipeline steps."""
        X, y = sample_data
        
        # Add step with condition
        builder.add_conditional_step(
            'outlier_removal',
            condition=lambda X, y: (X > 3).any().any(),  # If outliers exist
            step_config={'method': 'iqr'}
        )
        
        builder.add_model_step('random_forest')
        pipeline = builder.build()
        
        assert isinstance(pipeline, Pipeline)
        pipeline.fit(X, y)
    
    def test_parallel_processing_steps(self, builder, sample_data):
        """Test parallel processing in pipeline."""
        X, y = sample_data
        
        # Add parallel feature engineering
        builder.add_parallel_steps([
            ('numeric_processing', 'standard_scaler'),
            ('feature_creation', 'polynomial_features')
        ])
        
        builder.add_model_step('logistic_regression')
        pipeline = builder.build()
        
        assert isinstance(pipeline, Pipeline)
        pipeline.fit(X, y)
    
    def test_custom_transformer_integration(self, builder, sample_data):
        """Test integration with custom transformers."""
        X, y = sample_data
        
        # Define custom transformer
        from sklearn.base import BaseEstimator, TransformerMixin
        
        class CustomTransformer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self
            
            def transform(self, X):
                return X * 2  # Simple transformation
        
        builder.add_custom_step('custom_transform', CustomTransformer())
        builder.add_model_step('logistic_regression')
        
        pipeline = builder.build()
        
        assert isinstance(pipeline, Pipeline)
        pipeline.fit(X, y)
    
    def test_pipeline_branching(self, builder, sample_data):
        """Test pipeline branching for different data types."""
        X, y = sample_data
        
        # Create branched pipeline
        builder.add_branched_processing({
            'numeric_branch': ['standard_scaler', 'polynomial_features'],
            'categorical_branch': ['onehot_encoder']
        })
        
        builder.add_model_step('random_forest')
        pipeline = builder.build()
        
        assert isinstance(pipeline, Pipeline)


class TestPipelineConfig:
    """Test PipelineConfig class."""
    
    def test_config_creation(self):
        """Test configuration creation and validation."""
        config = PipelineConfig({
            'preprocessing': {
                'scaling': 'standard',
                'feature_selection': True,
                'handle_missing': 'auto'
            },
            'model': {
                'type': 'random_forest',
                'n_estimators': 100,
                'random_state': 42
            },
            'evaluation': {
                'cv_folds': 5,
                'scoring': 'accuracy'
            }
        })
        
        assert config.get_preprocessing_config()['scaling'] == 'standard'
        assert config.get_model_config()['type'] == 'random_forest'
        assert config.get_evaluation_config()['cv_folds'] == 5
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid configuration
        with pytest.raises((ValueError, KeyError)):
            PipelineConfig({
                'model': {'type': 'invalid_model'}
            })
    
    def test_config_templates(self):
        """Test predefined configuration templates."""
        # Quick start template
        quick_config = PipelineConfig.quick_start_classification()
        assert 'preprocessing' in quick_config.config
        assert 'model' in quick_config.config
        
        # Advanced template
        advanced_config = PipelineConfig.advanced_classification()
        assert 'feature_engineering' in advanced_config.config
        
        # Regression template
        regression_config = PipelineConfig.quick_start_regression()
        assert regression_config.get_model_config()['type'] in ['linear_regression', 'random_forest']
    
    def test_config_merging(self):
        """Test configuration merging."""
        base_config = PipelineConfig({'preprocessing': {'scaling': 'standard'}})
        override_config = {'preprocessing': {'scaling': 'minmax'}}
        
        merged_config = base_config.merge(override_config)
        
        assert merged_config.get_preprocessing_config()['scaling'] == 'minmax'


class TestPipelineOptimizer:
    """Test PipelineOptimizer class."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance."""
        return PipelineOptimizer()
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        return pd.DataFrame(X), y
    
    def test_hyperparameter_optimization(self, optimizer, sample_data):
        """Test hyperparameter optimization."""
        X, y = sample_data
        
        # Create basic pipeline
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Define parameter grid
        param_grid = {
            'classifier__n_estimators': [10, 50, 100],
            'classifier__max_depth': [3, 5, None]
        }
        
        optimized_pipeline = optimizer.optimize_hyperparameters(
            pipeline, X, y, param_grid, cv=3
        )
        
        assert hasattr(optimized_pipeline, 'best_params_')
        assert hasattr(optimized_pipeline, 'best_score_')
    
    def test_feature_selection_optimization(self, optimizer, sample_data):
        """Test feature selection optimization."""
        X, y = sample_data
        
        optimal_features = optimizer.optimize_feature_selection(
            X, y, max_features=5, method='univariate'
        )
        
        assert len(optimal_features) <= 5
        assert all(isinstance(idx, (int, np.integer)) for idx in optimal_features)
    
    def test_preprocessing_optimization(self, optimizer, sample_data):
        """Test preprocessing step optimization."""
        X, y = sample_data
        
        optimal_preprocessing = optimizer.optimize_preprocessing(
            X, y, 
            scalers=['standard', 'minmax', 'robust'],
            feature_selectors=['univariate', 'rfe']
        )
        
        assert 'best_scaler' in optimal_preprocessing
        assert 'best_selector' in optimal_preprocessing
        assert 'score' in optimal_preprocessing
    
    def test_end_to_end_optimization(self, optimizer, sample_data):
        """Test end-to-end pipeline optimization."""
        X, y = sample_data
        
        optimal_pipeline = optimizer.optimize_full_pipeline(
            X, y,
            algorithms=['logistic_regression', 'random_forest'],
            preprocessing_options=['standard', 'minmax'],
            max_time_minutes=5
        )
        
        assert isinstance(optimal_pipeline, Pipeline)
        
        # Test performance
        score = optimal_pipeline.score(X, y)
        assert 0 <= score <= 1


class TestPipelineIntegration:
    """Integration tests for pipeline components."""
    
    def test_factory_optimizer_integration(self):
        """Test integration between factory and optimizer."""
        # Generate data
        X, y = make_classification(n_samples=150, n_features=8, random_state=42)
        X = pd.DataFrame(X)
        
        # Create pipeline with factory
        factory = PipelineFactory()
        pipeline = factory.create_classification_pipeline(model_type='random_forest')
        
        # Optimize with optimizer
        optimizer = PipelineOptimizer()
        param_grid = {'classifier__n_estimators': [10, 50]}
        
        optimized_pipeline = optimizer.optimize_hyperparameters(
            pipeline, X, y, param_grid, cv=3
        )
        
        # Should work end-to-end
        assert hasattr(optimized_pipeline, 'best_score_')
        optimized_pipeline.fit(X, y)
        predictions = optimized_pipeline.predict(X)
        assert len(predictions) == len(X)
    
    def test_automl_custom_builder_integration(self):
        """Test integration between AutoML and custom builder."""
        X, y = make_classification(n_samples=100, n_features=6, random_state=42)
        X = pd.DataFrame(X)
        
        # Use AutoML to get recommendations
        automl = AutoMLPipelineBuilder()
        recommendations = automl.recommend_preprocessing(X, y)
        
        # Use recommendations in custom builder
        builder = CustomPipelineBuilder()
        
        for rec in recommendations:
            builder.add_preprocessing_step(rec['name'])
        
        builder.add_model_step('logistic_regression')
        pipeline = builder.build()
        
        # Should work together
        assert isinstance(pipeline, Pipeline)
        pipeline.fit(X, y)
    
    def test_config_driven_pipeline_creation(self):
        """Test configuration-driven pipeline creation."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X = pd.DataFrame(X)
        
        # Create config
        config = PipelineConfig({
            'preprocessing': {
                'scaling': 'standard',
                'feature_selection': 'univariate',
                'k_features': 3
            },
            'model': {
                'type': 'logistic_regression',
                'C': 1.0,
                'random_state': 42
            }
        })
        
        # Create pipeline from config
        factory = PipelineFactory()
        pipeline = factory.create_from_config(config)
        
        assert isinstance(pipeline, Pipeline)
        pipeline.fit(X, y)
        
        # Verify configuration was applied
        score = pipeline.score(X, y)
        assert 0 <= score <= 1


if __name__ == "__main__":
    pytest.main([__file__])