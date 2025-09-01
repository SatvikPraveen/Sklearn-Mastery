"""
Unit tests for regression models.

Tests for all regression model wrappers and utilities.
"""

import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os
import tempfile

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from models.supervised.regression import (
    RegressionModel,
    LinearRegressionModel,
    RidgeRegressionModel,
    LassoRegressionModel,
    ElasticNetModel,
    RandomForestRegressorModel,
    SVMRegressorModel,
    GradientBoostingRegressorModel,
    NeuralNetworkRegressorModel,
    DecisionTreeRegressorModel,
    ExtraTreesRegressorModel,
    AdaBoostRegressorModel
)


class TestRegressionModel:
    """Test base RegressionModel class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample regression data."""
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=8,
            noise=0.1,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        model = RegressionModel()
        
        with pytest.raises(NotImplementedError):
            model.train(None, None)
        
        with pytest.raises(NotImplementedError):
            model.predict(None)
    
    def test_evaluate_method(self, sample_data):
        """Test evaluate method with a concrete implementation."""
        X_train, X_test, y_train, y_test = sample_data
        
        # Use a concrete implementation
        model = LinearRegressionModel()
        model.train(X_train, y_train)
        
        metrics = model.evaluate(X_test, y_test)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        
        # Check metric properties
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['rmse'] == np.sqrt(metrics['mse'])
    
    def test_cross_validate(self, sample_data):
        """Test cross-validation functionality."""
        X_train, _, y_train, _ = sample_data
        
        model = LinearRegressionModel()
        scores = model.cross_validate(X_train, y_train, cv=3)
        
        assert 'test_score' in scores
        assert 'train_score' in scores
        assert 'fit_time' in scores
        assert len(scores['test_score']) == 3


class TestLinearRegressionModel:
    """Test LinearRegressionModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return LinearRegressionModel()
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_regression(
            n_samples=100,
            n_features=5,
            noise=0.1,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert hasattr(model, 'model')
        assert model.model is not None
    
    def test_train_predict(self, model, data):
        """Test training and prediction."""
        X, y = data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(y_test)
        assert predictions.dtype == np.float64
    
    def test_coefficients(self, model, data):
        """Test coefficient extraction."""
        X, y = data
        model.train(X, y)
        
        coef = model.get_coefficients()
        
        assert len(coef) == X.shape[1]
        assert isinstance(coef, np.ndarray)
    
    def test_intercept(self, model, data):
        """Test intercept extraction."""
        X, y = data
        model.train(X, y)
        
        intercept = model.get_intercept()
        
        assert isinstance(intercept, (float, np.float64))
    
    def test_save_load(self, model, data):
        """Test model saving and loading."""
        X, y = data
        model.train(X, y)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, 'test_model.pkl')
            model.save_model(model_path)
            
            # Load model and test
            loaded_model = LinearRegressionModel()
            loaded_model.load_model(model_path)
            
            # Test predictions are the same
            original_pred = model.predict(X)
            loaded_pred = loaded_model.predict(X)
            
            np.testing.assert_array_almost_equal(original_pred, loaded_pred)


class TestRidgeRegressionModel:
    """Test RidgeRegressionModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return RidgeRegressionModel(alpha=1.0, random_state=42)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_regression(
            n_samples=100,
            n_features=5,
            noise=0.1,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.alpha == 1.0
        assert model.random_state == 42
        assert hasattr(model, 'model')
    
    def test_train_predict(self, model, data):
        """Test training and prediction."""
        X, y = data
        model.train(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert predictions.dtype == np.float64
    
    def test_regularization_effect(self, data):
        """Test that regularization affects coefficients."""
        X, y = data
        
        # Train with different alpha values
        model_low = RidgeRegressionModel(alpha=0.01)
        model_high = RidgeRegressionModel(alpha=100.0)
        
        model_low.train(X, y)
        model_high.train(X, y)
        
        coef_low = model_low.get_coefficients()
        coef_high = model_high.get_coefficients()
        
        # Higher regularization should lead to smaller coefficients
        assert np.linalg.norm(coef_high) < np.linalg.norm(coef_low)


class TestLassoRegressionModel:
    """Test LassoRegressionModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return LassoRegressionModel(alpha=0.1, random_state=42)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_regression(
            n_samples=100,
            n_features=10,
            n_informative=5,
            noise=0.1,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.alpha == 0.1
        assert model.random_state == 42
        assert hasattr(model, 'model')
    
    def test_train_predict(self, model, data):
        """Test training and prediction."""
        X, y = data
        model.train(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert predictions.dtype == np.float64
    
    def test_feature_selection(self, model, data):
        """Test that Lasso performs feature selection."""
        X, y = data
        model.train(X, y)
        
        coef = model.get_coefficients()
        
        # Lasso should set some coefficients to zero
        assert np.sum(coef == 0) > 0


class TestElasticNetModel:
    """Test ElasticNetModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return ElasticNetModel(alpha=0.1, l1_ratio=0.5, random_state=42)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_regression(
            n_samples=100,
            n_features=5,
            noise=0.1,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.alpha == 0.1
        assert model.l1_ratio == 0.5
        assert model.random_state == 42
        assert hasattr(model, 'model')
    
    def test_train_predict(self, model, data):
        """Test training and prediction."""
        X, y = data
        model.train(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert predictions.dtype == np.float64
    
    def test_l1_ratio_effect(self, data):
        """Test that l1_ratio affects the model behavior."""
        X, y = data
        
        # Pure Lasso (l1_ratio=1)
        model_lasso = ElasticNetModel(alpha=0.1, l1_ratio=1.0, random_state=42)
        # Pure Ridge (l1_ratio=0)
        model_ridge = ElasticNetModel(alpha=0.1, l1_ratio=0.0, random_state=42)
        
        model_lasso.train(X, y)
        model_ridge.train(X, y)
        
        coef_lasso = model_lasso.get_coefficients()
        coef_ridge = model_ridge.get_coefficients()
        
        # Lasso should have more zero coefficients
        assert np.sum(coef_lasso == 0) >= np.sum(coef_ridge == 0)


class TestRandomForestRegressorModel:
    """Test RandomForestRegressorModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return RandomForestRegressorModel(random_state=42, n_estimators=10)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_regression(
            n_samples=100,
            n_features=5,
            noise=0.1,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.random_state == 42
        assert model.n_estimators == 10
        assert hasattr(model, 'model')
    
    def test_train_predict(self, model, data):
        """Test training and prediction."""
        X, y = data
        model.train(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert predictions.dtype == np.float64
    
    def test_feature_importance(self, model, data):
        """Test feature importance extraction."""
        X, y = data
        model.train(X, y)
        
        importance = model.get_feature_importance()
        
        assert len(importance) == X.shape[1]
        assert np.allclose(importance.sum(), 1.0)
        assert np.all(importance >= 0)


class TestSVMRegressorModel:
    """Test SVMRegressorModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return SVMRegressorModel(random_state=42)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_regression(
            n_samples=100,
            n_features=5,
            noise=0.1,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.random_state == 42
        assert hasattr(model, 'model')
    
    def test_train_predict(self, model, data):
        """Test training and prediction."""
        X, y = data
        model.train(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert predictions.dtype == np.float64
    
    def test_kernel_effect(self, data):
        """Test different kernel effects."""
        X, y = data
        
        model_linear = SVMRegressorModel(kernel='linear', random_state=42)
        model_rbf = SVMRegressorModel(kernel='rbf', random_state=42)
        
        model_linear.train(X, y)
        model_rbf.train(X, y)
        
        pred_linear = model_linear.predict(X)
        pred_rbf = model_rbf.predict(X)
        
        # Predictions should be different for different kernels
        assert not np.allclose(pred_linear, pred_rbf)


class TestGradientBoostingRegressorModel:
    """Test GradientBoostingRegressorModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return GradientBoostingRegressorModel(random_state=42, n_estimators=10)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_regression(
            n_samples=100,
            n_features=5,
            noise=0.1,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.random_state == 42
        assert model.n_estimators == 10
        assert hasattr(model, 'model')
    
    def test_train_predict(self, model, data):
        """Test training and prediction."""
        X, y = data
        model.train(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert predictions.dtype == np.float64
    
    def test_staged_prediction(self, model, data):
        """Test staged prediction functionality."""
        X, y = data
        model.train(X, y)
        
        staged_preds = list(model.staged_predict(X))
        
        assert len(staged_preds) == model.n_estimators
        assert all(len(pred) == len(y) for pred in staged_preds)
    
    def test_feature_importance(self, model, data):
        """Test feature importance extraction."""
        X, y = data
        model.train(X, y)
        
        importance = model.get_feature_importance()
        
        assert len(importance) == X.shape[1]
        assert np.all(importance >= 0)


class TestNeuralNetworkRegressorModel:
    """Test NeuralNetworkRegressorModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return NeuralNetworkRegressorModel(random_state=42, max_iter=100)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_regression(
            n_samples=100,
            n_features=5,
            noise=0.1,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.random_state == 42
        assert model.max_iter == 100
        assert hasattr(model, 'model')
    
    def test_train_predict(self, model, data):
        """Test training and prediction."""
        X, y = data
        model.train(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert predictions.dtype == np.float64


class TestDecisionTreeRegressorModel:
    """Test DecisionTreeRegressorModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return DecisionTreeRegressorModel(random_state=42, max_depth=3)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_regression(
            n_samples=100,
            n_features=5,
            noise=0.1,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.random_state == 42
        assert model.max_depth == 3
        assert hasattr(model, 'model')
    
    def test_train_predict(self, model, data):
        """Test training and prediction."""
        X, y = data
        model.train(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert predictions.dtype == np.float64
    
    def test_feature_importance(self, model, data):
        """Test feature importance extraction."""
        X, y = data
        model.train(X, y)
        
        importance = model.get_feature_importance()
        
        assert len(importance) == X.shape[1]
        assert np.all(importance >= 0)


class TestExtraTreesRegressorModel:
    """Test ExtraTreesRegressorModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return ExtraTreesRegressorModel(random_state=42, n_estimators=10)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_regression(
            n_samples=100,
            n_features=5,
            noise=0.1,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.random_state == 42
        assert model.n_estimators == 10
        assert hasattr(model, 'model')
    
    def test_train_predict(self, model, data):
        """Test training and prediction."""
        X, y = data
        model.train(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert predictions.dtype == np.float64


class TestAdaBoostRegressorModel:
    """Test AdaBoostRegressorModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return AdaBoostRegressorModel(random_state=42, n_estimators=10)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_regression(
            n_samples=100,
            n_features=5,
            noise=0.1,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.random_state == 42
        assert model.n_estimators == 10
        assert hasattr(model, 'model')
    
    def test_train_predict(self, model, data):
        """Test training and prediction."""
        X, y = data
        model.train(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert predictions.dtype == np.float64


class TestRegressionModelIntegration:
    """Integration tests for regression models."""
    
    @pytest.fixture
    def models(self):
        """Create all model instances."""
        return {
            'linear': LinearRegressionModel(),
            'ridge': RidgeRegressionModel(random_state=42),
            'lasso': LassoRegressionModel(random_state=42),
            'elastic': ElasticNetModel(random_state=42),
            'rf': RandomForestRegressorModel(random_state=42, n_estimators=10),
            'svm': SVMRegressorModel(random_state=42),
            'gb': GradientBoostingRegressorModel(random_state=42, n_estimators=10),
            'dt': DecisionTreeRegressorModel(random_state=42),
            'et': ExtraTreesRegressorModel(random_state=42, n_estimators=10),
            'ada': AdaBoostRegressorModel(random_state=42, n_estimators=10)
        }
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            noise=0.1,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_all_models_train_predict(self, models, data):
        """Test that all models can train and predict."""
        X_train, X_test, y_train, y_test = data
        
        for name, model in models.items():
            # Skip neural network for speed
            if name == 'nn':
                continue
                
            model.train(X_train, y_train)
            predictions = model.predict(X_test)
            
            assert len(predictions) == len(y_test), f"{name} failed prediction length test"
            assert predictions.dtype == np.float64, f"{name} failed prediction dtype test"
    
    def test_all_models_evaluate(self, models, data):
        """Test that all models can be evaluated."""
        X_train, X_test, y_train, y_test = data
        
        for name, model in models.items():
            # Skip neural network for speed
            if name == 'nn':
                continue
                
            model.train(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)
            
            assert 'mse' in metrics, f"{name} missing MSE metric"
            assert 'r2' in metrics, f"{name} missing R² metric"
            assert metrics['mse'] >= 0, f"{name} MSE should be non-negative"
    
    def test_hyperparameter_tuning(self, models, data):
        """Test hyperparameter tuning for select models."""
        X_train, _, y_train, _ = data
        
        # Test a few models that support hyperparameter tuning
        test_models = {
            'ridge': {
                'model': models['ridge'],
                'params': {'alpha': [0.1, 1.0, 10.0]}
            },
            'rf': {
                'model': models['rf'],
                'params': {'n_estimators': [5, 10], 'max_depth': [3, 5]}
            }
        }
        
        for name, config in test_models.items():
            model = config['model']
            param_grid = config['params']
            
            best_params, best_score = model.tune_hyperparameters(
                X_train, y_train, param_grid, cv=3
            )
            
            assert isinstance(best_params, dict), f"{name} should return dict for best params"
            assert isinstance(best_score, (float, np.float64)), f"{name} should return numeric score"
    
    def test_prediction_consistency(self, models, data):
        """Test that predictions are consistent across multiple calls."""
        X_train, X_test, y_train, y_test = data
        
        for name, model in models.items():
            # Skip neural network for speed and non-deterministic models
            if name in ['nn']:
                continue
                
            model.train(X_train, y_train)
            
            pred1 = model.predict(X_test)
            pred2 = model.predict(X_test)
            
            np.testing.assert_array_equal(
                pred1, pred2, 
                f"{name} predictions should be consistent"
            )