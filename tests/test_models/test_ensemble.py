"""
Unit tests for ensemble models.

Tests for all ensemble model wrappers and utilities.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import sys
import os
import tempfile

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from models.ensemble.ensemble_methods import (
    VotingEnsemble,
    BaggingEnsemble,
    BoostingEnsemble,
    StackingEnsemble,
    BlendingEnsemble
)


class TestVotingEnsemble:
    """Test VotingEnsemble class."""
    
    @pytest.fixture
    def classification_data(self):
        """Generate classification test data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=3,
            n_informative=8,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    @pytest.fixture
    def regression_data(self):
        """Generate regression test data."""
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            noise=0.1,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    @pytest.fixture
    def classification_estimators(self):
        """Create classification estimators."""
        return [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=200)),
            ('dt', DecisionTreeClassifier(random_state=42, max_depth=3))
        ]
    
    @pytest.fixture
    def regression_estimators(self):
        """Create regression estimators."""
        return [
            ('rf', RandomForestRegressor(n_estimators=10, random_state=42)),
            ('lr', LinearRegression()),
            ('dt', DecisionTreeRegressor(random_state=42, max_depth=3))
        ]
    
    def test_hard_voting_classification(self, classification_data, classification_estimators):
        """Test hard voting for classification."""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = VotingEnsemble(
            estimators=classification_estimators,
            voting='hard'
        )
        
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)
        
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset(set(y_train))
    
    def test_soft_voting_classification(self, classification_data, classification_estimators):
        """Test soft voting for classification."""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = VotingEnsemble(
            estimators=classification_estimators,
            voting='soft'
        )
        
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)
        probas = ensemble.predict_proba(X_test)
        
        assert len(predictions) == len(y_test)
        assert probas.shape == (len(y_test), len(set(y_train)))
        assert np.allclose(probas.sum(axis=1), 1.0)
    
    def test_voting_regression(self, regression_data, regression_estimators):
        """Test voting for regression."""
        X_train, X_test, y_train, y_test = regression_data
        
        ensemble = VotingEnsemble(
            estimators=regression_estimators,
            voting='soft'  # For regression, this is averaging
        )
        
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)
        
        assert len(predictions) == len(y_test)
        assert predictions.dtype == np.float64
    
    def test_individual_predictions(self, classification_data, classification_estimators):
        """Test individual estimator predictions."""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = VotingEnsemble(
            estimators=classification_estimators,
            voting='hard'
        )
        
        ensemble.fit(X_train, y_train)
        individual_preds = ensemble.get_individual_predictions(X_test)
        
        assert len(individual_preds) == len(classification_estimators)
        assert all(len(pred) == len(y_test) for pred in individual_preds)
    
    def test_save_load(self, classification_data, classification_estimators):
        """Test model saving and loading."""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = VotingEnsemble(
            estimators=classification_estimators,
            voting='hard'
        )
        ensemble.fit(X_train, y_train)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, 'ensemble.pkl')
            ensemble.save_model(model_path)
            
            loaded_ensemble = VotingEnsemble(estimators=[], voting='hard')
            loaded_ensemble.load_model(model_path)
            
            original_pred = ensemble.predict(X_test)
            loaded_pred = loaded_ensemble.predict(X_test)
            
            np.testing.assert_array_equal(original_pred, loaded_pred)


class TestBaggingEnsemble:
    """Test BaggingEnsemble class."""
    
    @pytest.fixture
    def classification_data(self):
        """Generate classification test data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    @pytest.fixture
    def regression_data(self):
        """Generate regression test data."""
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            noise=0.1,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_bagging_classification(self, classification_data):
        """Test bagging for classification."""
        X_train, X_test, y_train, y_test = classification_data
        
        base_estimator = DecisionTreeClassifier(random_state=42)
        ensemble = BaggingEnsemble(
            base_estimator=base_estimator,
            n_estimators=10,
            random_state=42
        )
        
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)
        
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset(set(y_train))
    
    def test_bagging_regression(self, regression_data):
        """Test bagging for regression."""
        X_train, X_test, y_train, y_test = regression_data
        
        base_estimator = DecisionTreeRegressor(random_state=42)
        ensemble = BaggingEnsemble(
            base_estimator=base_estimator,
            n_estimators=10,
            random_state=42
        )
        
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)
        
        assert len(predictions) == len(y_test)
        assert predictions.dtype == np.float64
    
    def test_oob_score(self, classification_data):
        """Test out-of-bag score calculation."""
        X_train, X_test, y_train, y_test = classification_data
        
        base_estimator = DecisionTreeClassifier(random_state=42)
        ensemble = BaggingEnsemble(
            base_estimator=base_estimator,
            n_estimators=10,
            random_state=42,
            oob_score=True
        )
        
        ensemble.fit(X_train, y_train)
        oob_score = ensemble.get_oob_score()
        
        assert isinstance(oob_score, (float, np.float64))
        assert 0 <= oob_score <= 1
    
    def test_feature_subsampling(self, classification_data):
        """Test feature subsampling."""
        X_train, X_test, y_train, y_test = classification_data
        
        base_estimator = DecisionTreeClassifier(random_state=42)
        ensemble = BaggingEnsemble(
            base_estimator=base_estimator,
            n_estimators=10,
            max_features=0.5,  # Use 50% of features
            random_state=42
        )
        
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)
        
        assert len(predictions) == len(y_test)
    
    def test_sample_subsampling(self, classification_data):
        """Test sample subsampling."""
        X_train, X_test, y_train, y_test = classification_data
        
        base_estimator = DecisionTreeClassifier(random_state=42)
        ensemble = BaggingEnsemble(
            base_estimator=base_estimator,
            n_estimators=10,
            max_samples=0.8,  # Use 80% of samples
            random_state=42
        )
        
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)
        
        assert len(predictions) == len(y_test)


class TestBoostingEnsemble:
    """Test BoostingEnsemble class."""
    
    @pytest.fixture
    def classification_data(self):
        """Generate classification test data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    @pytest.fixture
    def regression_data(self):
        """Generate regression test data."""
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            noise=0.1,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_adaboost_classification(self, classification_data):
        """Test AdaBoost for classification."""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = BoostingEnsemble(
            algorithm='adaboost',
            n_estimators=10,
            random_state=42
        )
        
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)
        
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset(set(y_train))
    
    def test_gradientboosting_classification(self, classification_data):
        """Test Gradient Boosting for classification."""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = BoostingEnsemble(
            algorithm='gradientboosting',
            n_estimators=10,
            random_state=42
        )
        
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)
        
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset(set(y_train))
    
    def test_gradientboosting_regression(self, regression_data):
        """Test Gradient Boosting for regression."""
        X_train, X_test, y_train, y_test = regression_data
        
        ensemble = BoostingEnsemble(
            algorithm='gradientboosting',
            n_estimators=10,
            random_state=42,
            task_type='regression'
        )
        
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)
        
        assert len(predictions) == len(y_test)
        assert predictions.dtype == np.float64
    
    def test_staged_predictions(self, classification_data):
        """Test staged predictions."""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = BoostingEnsemble(
            algorithm='gradientboosting',
            n_estimators=5,
            random_state=42
        )
        
        ensemble.fit(X_train, y_train)
        staged_preds = list(ensemble.staged_predict(X_test))
        
        assert len(staged_preds) == 5  # n_estimators
        assert all(len(pred) == len(y_test) for pred in staged_preds)
    
    def test_feature_importance(self, classification_data):
        """Test feature importance extraction."""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = BoostingEnsemble(
            algorithm='gradientboosting',
            n_estimators=10,
            random_state=42
        )
        
        ensemble.fit(X_train, y_train)
        importance = ensemble.get_feature_importance()
        
        assert len(importance) == X_train.shape[1]
        assert np.all(importance >= 0)
        assert np.sum(importance) > 0


class TestStackingEnsemble:
    """Test StackingEnsemble class."""
    
    @pytest.fixture
    def classification_data(self):
        """Generate classification test data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    @pytest.fixture
    def regression_data(self):
        """Generate regression test data."""
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            noise=0.1,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    @pytest.fixture
    def base_classifiers(self):
        """Create base classifiers."""
        return [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('dt', DecisionTreeClassifier(random_state=42, max_depth=3))
        ]
    
    @pytest.fixture
    def base_regressors(self):
        """Create base regressors."""
        return [
            ('rf', RandomForestRegressor(n_estimators=10, random_state=42)),
            ('dt', DecisionTreeRegressor(random_state=42, max_depth=3))
        ]
    
    def test_stacking_classification(self, classification_data, base_classifiers):
        """Test stacking for classification."""
        X_train, X_test, y_train, y_test = classification_data
        
    def test_stacking_classification(self, classification_data, base_classifiers):
        """Test stacking for classification."""
        X_train, X_test, y_train, y_test = classification_data
        
        meta_classifier = LogisticRegression(random_state=42)
        ensemble = StackingEnsemble(
            base_estimators=base_classifiers,
            meta_estimator=meta_classifier,
            cv=3
        )
        
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)
        
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset(set(y_train))
    
    def test_stacking_regression(self, regression_data, base_regressors):
        """Test stacking for regression."""
        X_train, X_test, y_train, y_test = regression_data
        
        meta_regressor = LinearRegression()
        ensemble = StackingEnsemble(
            base_estimators=base_regressors,
            meta_estimator=meta_regressor,
            cv=3
        )
        
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)
        
        assert len(predictions) == len(y_test)
        assert predictions.dtype == np.float64
    
    def test_base_predictions(self, classification_data, base_classifiers):
        """Test base model predictions."""
        X_train, X_test, y_train, y_test = classification_data
        
        meta_classifier = LogisticRegression(random_state=42)
        ensemble = StackingEnsemble(
            base_estimators=base_classifiers,
            meta_estimator=meta_classifier,
            cv=3
        )
        
        ensemble.fit(X_train, y_train)
        base_preds = ensemble.get_base_predictions(X_test)
        
        assert base_preds.shape == (len(y_test), len(base_classifiers))
    
    def test_cross_validation_predictions(self, classification_data, base_classifiers):
        """Test cross-validation predictions generation."""
        X_train, X_test, y_train, y_test = classification_data
        
        meta_classifier = LogisticRegression(random_state=42)
        ensemble = StackingEnsemble(
            base_estimators=base_classifiers,
            meta_estimator=meta_classifier,
            cv=3
        )
        
        ensemble.fit(X_train, y_train)
        
        # Check that base estimators were fitted
        assert len(ensemble.base_estimators_) == len(base_classifiers)
        assert ensemble.meta_estimator_ is not None
    
    def test_passthrough_features(self, classification_data, base_classifiers):
        """Test using original features alongside stacked predictions."""
        X_train, X_test, y_train, y_test = classification_data
        
        meta_classifier = LogisticRegression(random_state=42)
        ensemble = StackingEnsemble(
            base_estimators=base_classifiers,
            meta_estimator=meta_classifier,
            cv=3,
            passthrough=True
        )
        
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)
        
        assert len(predictions) == len(y_test)


class TestBlendingEnsemble:
    """Test BlendingEnsemble class."""
    
    @pytest.fixture
    def classification_data(self):
        """Generate classification test data."""
        X, y = make_classification(
            n_samples=300,  # Larger dataset for blending
            n_features=10,
            n_classes=2,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    @pytest.fixture
    def regression_data(self):
        """Generate regression test data."""
        X, y = make_regression(
            n_samples=300,
            n_features=10,
            noise=0.1,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    @pytest.fixture
    def base_classifiers(self):
        """Create base classifiers."""
        return [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('dt', DecisionTreeClassifier(random_state=42, max_depth=3))
        ]
    
    @pytest.fixture
    def base_regressors(self):
        """Create base regressors."""
        return [
            ('rf', RandomForestRegressor(n_estimators=10, random_state=42)),
            ('dt', DecisionTreeRegressor(random_state=42, max_depth=3))
        ]
    
    def test_blending_classification(self, classification_data, base_classifiers):
        """Test blending for classification."""
        X_train, X_test, y_train, y_test = classification_data
        
        meta_classifier = LogisticRegression(random_state=42)
        ensemble = BlendingEnsemble(
            base_estimators=base_classifiers,
            meta_estimator=meta_classifier,
            holdout_size=0.2
        )
        
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)
        
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset(set(y_train))
    
    def test_blending_regression(self, regression_data, base_regressors):
        """Test blending for regression."""
        X_train, X_test, y_train, y_test = regression_data
        
        meta_regressor = LinearRegression()
        ensemble = BlendingEnsemble(
            base_estimators=base_regressors,
            meta_estimator=meta_regressor,
            holdout_size=0.2
        )
        
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)
        
        assert len(predictions) == len(y_test)
        assert predictions.dtype == np.float64
    
    def test_holdout_split(self, classification_data, base_classifiers):
        """Test holdout set splitting."""
        X_train, X_test, y_train, y_test = classification_data
        
        meta_classifier = LogisticRegression(random_state=42)
        ensemble = BlendingEnsemble(
            base_estimators=base_classifiers,
            meta_estimator=meta_classifier,
            holdout_size=0.3,
            random_state=42
        )
        
        ensemble.fit(X_train, y_train)
        
        # Check that holdout set was created
        assert hasattr(ensemble, 'base_estimators_')
        assert hasattr(ensemble, 'meta_estimator_')
    
    def test_different_holdout_sizes(self, classification_data, base_classifiers):
        """Test different holdout sizes."""
        X_train, X_test, y_train, y_test = classification_data
        
        holdout_sizes = [0.1, 0.2, 0.3]
        
        for holdout_size in holdout_sizes:
            meta_classifier = LogisticRegression(random_state=42)
            ensemble = BlendingEnsemble(
                base_estimators=base_classifiers,
                meta_estimator=meta_classifier,
                holdout_size=holdout_size,
                random_state=42
            )
            
            ensemble.fit(X_train, y_train)
            predictions = ensemble.predict(X_test)
            
            assert len(predictions) == len(y_test)


class TestEnsembleIntegration:
    """Integration tests for ensemble methods."""
    
    @pytest.fixture
    def classification_data(self):
        """Generate classification test data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    @pytest.fixture
    def regression_data(self):
        """Generate regression test data."""
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            noise=0.1,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_ensemble_vs_base_models_classification(self, classification_data):
        """Test that ensembles can outperform base models."""
        X_train, X_test, y_train, y_test = classification_data
        
        # Base models
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        dt = DecisionTreeClassifier(random_state=42, max_depth=3)
        
        # Train base models
        rf.fit(X_train, y_train)
        dt.fit(X_train, y_train)
        
        rf_accuracy = np.mean(rf.predict(X_test) == y_test)
        dt_accuracy = np.mean(dt.predict(X_test) == y_test)
        base_best = max(rf_accuracy, dt_accuracy)
        
        # Voting ensemble
        estimators = [('rf', rf), ('dt', dt)]
        voting_ensemble = VotingEnsemble(estimators=estimators, voting='hard')
        voting_ensemble.fit(X_train, y_train)
        voting_accuracy = np.mean(voting_ensemble.predict(X_test) == y_test)
        
        # Ensemble should be competitive with base models
        assert voting_accuracy >= base_best * 0.9  # Allow some variance
    
    def test_ensemble_diversity_benefit(self, classification_data):
        """Test that diverse base models benefit ensemble performance."""
        X_train, X_test, y_train, y_test = classification_data
        
        # Diverse models
        diverse_estimators = [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=200)),
            ('dt', DecisionTreeClassifier(random_state=42, max_depth=3))
        ]
        
        # Similar models (all decision trees with same parameters)
        similar_estimators = [
            ('dt1', DecisionTreeClassifier(random_state=42, max_depth=3)),
            ('dt2', DecisionTreeClassifier(random_state=43, max_depth=3)),
            ('dt3', DecisionTreeClassifier(random_state=44, max_depth=3))
        ]
        
        diverse_ensemble = VotingEnsemble(estimators=diverse_estimators, voting='hard')
        similar_ensemble = VotingEnsemble(estimators=similar_estimators, voting='hard')
        
        diverse_ensemble.fit(X_train, y_train)
        similar_ensemble.fit(X_train, y_train)
        
        diverse_accuracy = np.mean(diverse_ensemble.predict(X_test) == y_test)
        similar_accuracy = np.mean(similar_ensemble.predict(X_test) == y_test)
        
        # Diverse ensemble should generally perform better or similar
        # (Allow some variance due to randomness)
        assert diverse_accuracy >= similar_accuracy * 0.95
    
    def test_ensemble_scalability(self, classification_data):
        """Test ensemble performance with different numbers of base estimators."""
        X_train, X_test, y_train, y_test = classification_data
        
        n_estimators_list = [3, 5, 10]
        accuracies = []
        
        for n_est in n_estimators_list:
            base_estimator = DecisionTreeClassifier(random_state=42, max_depth=3)
            ensemble = BaggingEnsemble(
                base_estimator=base_estimator,
                n_estimators=n_est,
                random_state=42
            )
            
            ensemble.fit(X_train, y_train)
            accuracy = np.mean(ensemble.predict(X_test) == y_test)
            accuracies.append(accuracy)
        
        # Generally, more estimators should not hurt performance significantly
        assert max(accuracies) >= min(accuracies) * 0.95
    
    def test_ensemble_overfitting_resistance(self, classification_data):
        """Test that ensembles are more resistant to overfitting."""
        X_train, X_test, y_train, y_test = classification_data
        
        # Deep decision tree (prone to overfitting)
        deep_tree = DecisionTreeClassifier(random_state=42, max_depth=None)
        
        # Ensemble of deep trees
        ensemble = BaggingEnsemble(
            base_estimator=DecisionTreeClassifier(random_state=42, max_depth=None),
            n_estimators=10,
            random_state=42
        )
        
        deep_tree.fit(X_train, y_train)
        ensemble.fit(X_train, y_train)
        
        # Training accuracies
        tree_train_acc = np.mean(deep_tree.predict(X_train) == y_train)
        ensemble_train_acc = np.mean(ensemble.predict(X_train) == y_train)
        
        # Test accuracies
        tree_test_acc = np.mean(deep_tree.predict(X_test) == y_test)
        ensemble_test_acc = np.mean(ensemble.predict(X_test) == y_test)
        
        # Calculate overfitting (train - test accuracy)
        tree_overfitting = tree_train_acc - tree_test_acc
        ensemble_overfitting = ensemble_train_acc - ensemble_test_acc
        
        # Ensemble should have less overfitting
        assert ensemble_overfitting <= tree_overfitting + 0.05  # Small tolerance
    
    def test_computational_efficiency(self, classification_data):
        """Test computational efficiency considerations."""
        X_train, X_test, y_train, y_test = classification_data
        
        import time
        
        # Single model
        single_model = RandomForestClassifier(n_estimators=50, random_state=42)
        start_time = time.time()
        single_model.fit(X_train, y_train)
        single_pred = single_model.predict(X_test)
        single_time = time.time() - start_time
        
        # Ensemble of smaller models
        estimators = [
            ('rf1', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('rf2', RandomForestClassifier(n_estimators=10, random_state=43)),
            ('rf3', RandomForestClassifier(n_estimators=10, random_state=44)),
            ('rf4', RandomForestClassifier(n_estimators=10, random_state=45)),
            ('rf5', RandomForestClassifier(n_estimators=10, random_state=46))
        ]
        
        ensemble = VotingEnsemble(estimators=estimators, voting='hard')
        start_time = time.time()
        ensemble.fit(X_train, y_train)
        ensemble_pred = ensemble.predict(X_test)
        ensemble_time = time.time() - start_time
        
        # Both should produce reasonable results
        single_acc = np.mean(single_pred == y_test)
        ensemble_acc = np.mean(ensemble_pred == y_test)
        
        assert single_acc > 0.5  # Sanity check
        assert ensemble_acc > 0.5  # Sanity check
        
        # Time comparison is informational (both approaches have merits)
        print(f"Single model time: {single_time:.3f}s, accuracy: {single_acc:.3f}")
        print(f"Ensemble time: {ensemble_time:.3f}s, accuracy: {ensemble_acc:.3f}")
    
    def test_ensemble_prediction_consistency(self, classification_data):
        """Test prediction consistency across multiple runs."""
        X_train, X_test, y_train, y_test = classification_data
        
        # Test with deterministic ensemble
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=200))
        ]
        
        ensemble1 = VotingEnsemble(estimators=estimators, voting='hard')
        ensemble2 = VotingEnsemble(estimators=estimators, voting='hard')
        
        ensemble1.fit(X_train, y_train)
        ensemble2.fit(X_train, y_train)
        
        pred1 = ensemble1.predict(X_test)
        pred2 = ensemble2.predict(X_test)
        
        # With same random states, predictions should be identical
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_ensemble_error_handling(self, classification_data):
        """Test error handling in ensemble methods."""
        X_train, X_test, y_train, y_test = classification_data
        
        # Test with empty estimators list
        with pytest.raises((ValueError, AttributeError)):
            ensemble = VotingEnsemble(estimators=[], voting='hard')
            ensemble.fit(X_train, y_train)
        
        # Test with incompatible estimators
        estimators = [
            ('classifier', LogisticRegression(random_state=42)),
            ('regressor', LinearRegression())  # Wrong type for classification
        ]
        
        ensemble = VotingEnsemble(estimators=estimators, voting='hard')
        
        # This should raise an error during fitting or prediction
        with pytest.raises((ValueError, AttributeError)):
            ensemble.fit(X_train, y_train)
            ensemble.predict(X_test)