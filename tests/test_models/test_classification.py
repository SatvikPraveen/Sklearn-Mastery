"""
Unit tests for classification models.

Tests for all classification model wrappers and utilities.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import sys
import os
import joblib
import tempfile

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from models.supervised.classification import (
    ClassificationModel,
    LogisticRegressionModel,
    RandomForestClassifierModel,
    SVMClassifierModel,
    GradientBoostingClassifierModel,
    NeuralNetworkClassifierModel,
    NaiveBayesModel,
    KNNClassifierModel,
    DecisionTreeClassifierModel,
    ExtraTreesClassifierModel,
    AdaBoostClassifierModel
)


class TestClassificationModel:
    """Test base ClassificationModel class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        model = ClassificationModel()
        
        with pytest.raises(NotImplementedError):
            model.train(None, None)
        
        with pytest.raises(NotImplementedError):
            model.predict(None)
    
    def test_evaluate_method(self, sample_data):
        """Test evaluate method with a concrete implementation."""
        X_train, X_test, y_train, y_test = sample_data
        
        # Use a concrete implementation
        model = LogisticRegressionModel()
        model.train(X_train, y_train)
        
        metrics = model.evaluate(X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'confusion_matrix' in metrics
        assert 'classification_report' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
    
    def test_cross_validate(self, sample_data):
        """Test cross-validation functionality."""
        X_train, _, y_train, _ = sample_data
        
        model = LogisticRegressionModel()
        scores = model.cross_validate(X_train, y_train, cv=3)
        
        assert 'test_score' in scores
        assert 'train_score' in scores
        assert 'fit_time' in scores
        assert len(scores['test_score']) == 3


class TestLogisticRegressionModel:
    """Test LogisticRegressionModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return LogisticRegressionModel(random_state=42)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_classes=2,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.random_state == 42
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
        assert set(predictions).issubset(set(y))
    
    def test_predict_proba(self, model, data):
        """Test probability prediction."""
        X, y = data
        model.train(X, y)
        
        probas = model.predict_proba(X)
        
        assert probas.shape == (len(X), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
        assert np.all((probas >= 0) & (probas <= 1))
    
    def test_hyperparameter_tuning(self, model, data):
        """Test hyperparameter tuning."""
        X, y = data
        
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2']
        }
        
        best_params, best_score = model.tune_hyperparameters(
            X, y, param_grid, cv=3
        )
        
        assert 'C' in best_params
        assert 0 <= best_score <= 1
    
    def test_save_load(self, model, data):
        """Test model saving and loading."""
        X, y = data
        model.train(X, y)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, 'test_model.pkl')
            model.save_model(model_path)
            
            # Load model and test
            loaded_model = LogisticRegressionModel()
            loaded_model.load_model(model_path)
            
            # Test predictions are the same
            original_pred = model.predict(X)
            loaded_pred = loaded_model.predict(X)
            
            np.testing.assert_array_equal(original_pred, loaded_pred)


class TestRandomForestClassifierModel:
    """Test RandomForestClassifierModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return RandomForestClassifierModel(random_state=42, n_estimators=10)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_classes=2,
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
        assert set(predictions).issubset(set(y))
    
    def test_feature_importance(self, model, data):
        """Test feature importance extraction."""
        X, y = data
        model.train(X, y)
        
        importance = model.get_feature_importance()
        
        assert len(importance) == X.shape[1]
        assert np.allclose(importance.sum(), 1.0)
        assert np.all(importance >= 0)


class TestSVMClassifierModel:
    """Test SVMClassifierModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return SVMClassifierModel(random_state=42)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_classes=2,
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
        assert set(predictions).issubset(set(y))
    
    def test_probability_prediction(self, data):
        """Test probability prediction with probability=True."""
        X, y = data
        model = SVMClassifierModel(probability=True, random_state=42)
        model.train(X, y)
        
        probas = model.predict_proba(X)
        
        assert probas.shape == (len(X), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)


class TestGradientBoostingClassifierModel:
    """Test GradientBoostingClassifierModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return GradientBoostingClassifierModel(random_state=42, n_estimators=10)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_classes=2,
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
        assert set(predictions).issubset(set(y))
    
    def test_staged_prediction(self, model, data):
        """Test staged prediction functionality."""
        X, y = data
        model.train(X, y)
        
        staged_preds = list(model.staged_predict(X))
        
        assert len(staged_preds) == model.n_estimators
        assert all(len(pred) == len(y) for pred in staged_preds)


class TestNeuralNetworkClassifierModel:
    """Test NeuralNetworkClassifierModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return NeuralNetworkClassifierModel(random_state=42, max_iter=100)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_classes=2,
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
        assert set(predictions).issubset(set(y))


class TestNaiveBayesModel:
    """Test NaiveBayesModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return NaiveBayesModel()
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_classes=2,
            random_state=42
        )
        # Ensure positive values for Naive Bayes
        X = np.abs(X)
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert hasattr(model, 'model')
    
    def test_train_predict(self, model, data):
        """Test training and prediction."""
        X, y = data
        model.train(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert set(predictions).issubset(set(y))


class TestKNNClassifierModel:
    """Test KNNClassifierModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return KNNClassifierModel(n_neighbors=3)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_classes=2,
            random_state=42
        )
        return X, y
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.n_neighbors == 3
        assert hasattr(model, 'model')
    
    def test_train_predict(self, model, data):
        """Test training and prediction."""
        X, y = data
        model.train(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert set(predictions).issubset(set(y))


class TestDecisionTreeClassifierModel:
    """Test DecisionTreeClassifierModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return DecisionTreeClassifierModel(random_state=42, max_depth=3)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_classes=2,
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
        assert set(predictions).issubset(set(y))
    
    def test_feature_importance(self, model, data):
        """Test feature importance extraction."""
        X, y = data
        model.train(X, y)
        
        importance = model.get_feature_importance()
        
        assert len(importance) == X.shape[1]
        assert np.all(importance >= 0)


class TestExtraTreesClassifierModel:
    """Test ExtraTreesClassifierModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return ExtraTreesClassifierModel(random_state=42, n_estimators=10)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_classes=2,
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
        assert set(predictions).issubset(set(y))


class TestAdaBoostClassifierModel:
    """Test AdaBoostClassifierModel class."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return AdaBoostClassifierModel(random_state=42, n_estimators=10)
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_classes=2,
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
        assert set(predictions).issubset(set(y))
    
    def test_staged_decision_function(self, model, data):
        """Test staged decision function."""
        X, y = data
        model.train(X, y)
        
        staged_decisions = list(model.staged_decision_function(X))
        
        assert len(staged_decisions) == model.n_estimators
        assert all(len(decision) == len(y) for decision in staged_decisions)


class TestClassificationModelIntegration:
    """Integration tests for classification models."""
    
    @pytest.fixture
    def models(self):
        """Create all model instances."""
        return {
            'logistic': LogisticRegressionModel(random_state=42),
            'rf': RandomForestClassifierModel(random_state=42, n_estimators=10),
            'svm': SVMClassifierModel(random_state=42),
            'gb': GradientBoostingClassifierModel(random_state=42, n_estimators=10),
            'nb': NaiveBayesModel(),
            'knn': KNNClassifierModel(n_neighbors=3),
            'dt': DecisionTreeClassifierModel(random_state=42),
            'et': ExtraTreesClassifierModel(random_state=42, n_estimators=10),
            'ada': AdaBoostClassifierModel(random_state=42, n_estimators=10)
        }
    
    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=3,
            n_informative=8,
            random_state=42
        )
        # Ensure positive values for Naive Bayes
        X = np.abs(X)
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
            assert set(predictions).issubset(set(y_train)), f"{name} failed prediction values test"
    
    def test_all_models_evaluate(self, models, data):
        """Test that all models can be evaluated."""
        X_train, X_test, y_train, y_test = data
        
        for name, model in models.items():
            # Skip neural network for speed
            if name == 'nn':
                continue
                
            model.train(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)
            
            assert 'accuracy' in metrics, f"{name} missing accuracy metric"
            assert 0 <= metrics['accuracy'] <= 1, f"{name} accuracy out of range"
    
    def test_multiclass_classification(self, models, data):
        """Test multiclass classification capability."""
        X_train, X_test, y_train, y_test = data
        
        for name, model in models.items():
            # Skip neural network for speed
            if name == 'nn':
                continue
                
            model.train(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Check that all classes are handled
            unique_train = set(y_train)
            unique_pred = set(predictions)
            
            assert unique_pred.issubset(unique_train), f"{name} predicted unknown classes"