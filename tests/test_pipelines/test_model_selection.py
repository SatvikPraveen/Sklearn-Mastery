"""
Unit tests for model selection pipeline components.

Tests for automated model selection, comparison, and optimization pipelines.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pipelines.model_selection import (
    ModelSelectionPipeline,
    AutoModelSelector,
    ModelComparator,
    HyperparameterOptimizer,
    CrossValidationPipeline,
    ModelEnsemblePipeline,
    PerformanceTracker,
    ModelRegistry,
    BayesianOptimizer,
    GridSearchPipeline
)


class TestModelSelectionPipeline:
    """Test ModelSelectionPipeline class."""
    
    @pytest.fixture
    def pipeline(self):
        """Create model selection pipeline."""
        return ModelSelectionPipeline()
    
    @pytest.fixture
    def classification_data(self):
        """Generate classification data."""
        X, y = make_classification(
            n_samples=300,
            n_features=15,
            n_classes=3,
            n_informative=10,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=300,
            n_features=12,
            noise=0.1,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_basic_model_selection(self, pipeline, classification_data):
        """Test basic model selection functionality."""
        X_train, X_test, y_train, y_test = classification_data
        
        # Define candidate models
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=200),
            'random_forest': RandomForestClassifier(n_estimators=50, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=50, random_state=42)
        }
        
        best_model, results = pipeline.select_best_model(
            models, X_train, y_train, 
            scoring='accuracy', cv=3
        )
        
        assert best_model in models
        assert 'scores' in results
        assert 'mean_score' in results
        assert len(results['scores']) == len(models)
    
    def test_model_selection_with_preprocessing(self, pipeline, classification_data):
        """Test model selection with preprocessing steps."""
        X_train, X_test, y_train, y_test = classification_data
        
        # Add preprocessing to pipeline
        preprocessing_steps = [
            ('scaler', 'standard'),
            ('feature_selection', 'univariate_5')
        ]
        
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=200),
            'svm': SVC(random_state=42)
        }
        
        best_model, results = pipeline.select_best_model(
            models, X_train, y_train,
            preprocessing=preprocessing_steps,
            scoring='f1_weighted', cv=3
        )
        
        assert best_model in models
        assert results['mean_score'][best_model] > 0
    
    def test_model_selection_multiple_metrics(self, pipeline, classification_data):
        """Test model selection with multiple evaluation metrics."""
        X_train, X_test, y_train, y_test = classification_data
        
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=200),
            'random_forest': RandomForestClassifier(n_estimators=30, random_state=42)
        }
        
        scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        best_model, results = pipeline.select_best_model(
            models, X_train, y_train,
            scoring=scoring_metrics, cv=3
        )
        
        assert best_model in models
        for metric in scoring_metrics:
            assert metric in results['detailed_scores']
    
    def test_model_selection_custom_scorer(self, pipeline, classification_data):
        """Test model selection with custom scoring function."""
        X_train, X_test, y_train, y_test = classification_data
        
        # Define custom scorer
        def custom_balanced_accuracy(y_true, y_pred):
            from sklearn.metrics import balanced_accuracy_score
            return balanced_accuracy_score(y_true, y_pred)
        
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=200),
            'random_forest': RandomForestClassifier(n_estimators=30, random_state=42)
        }
        
        best_model, results = pipeline.select_best_model(
            models, X_train, y_train,
            scoring=custom_balanced_accuracy, cv=3
        )
        
        assert best_model in models
        assert 'mean_score' in results
    
    def test_early_stopping(self, pipeline, classification_data):
        """Test early stopping in model selection."""
        X_train, X_test, y_train, y_test = classification_data
        
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=200),
            'random_forest': RandomForestClassifier(n_estimators=30, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=30, random_state=42)
        }
        
        best_model, results = pipeline.select_best_model(
            models, X_train, y_train,
            scoring='accuracy', cv=3,
            early_stopping=True,
            early_stopping_threshold=0.95
        )
        
        assert best_model in models
        # If early stopping triggered, not all models may be evaluated
        assert len(results['scores']) <= len(models)


class TestAutoModelSelector:
    """Test AutoModelSelector class."""
    
    @pytest.fixture
    def selector(self):
        """Create auto model selector."""
        return AutoModelSelector()
    
    @pytest.fixture
    def mixed_dataset(self):
        """Generate dataset with mixed characteristics."""
        np.random.seed(42)
        
        # Create dataset with specific characteristics
        X, y = make_classification(
            n_samples=500,
            n_features=20,
            n_classes=2,
            n_informative=15,
            n_redundant=5,
            random_state=42
        )
        
        # Add some noise and correlation
        X[:, -1] = X[:, 0] + np.random.normal(0, 0.1, 500)  # Correlated feature
        
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_dataset_profiling(self, selector, mixed_dataset):
        """Test automatic dataset profiling."""
        X_train, X_test, y_train, y_test = mixed_dataset
        
        profile = selector.profile_dataset(X_train, y_train)
        
        assert 'n_samples' in profile
        assert 'n_features' in profile
        assert 'n_classes' in profile
        assert 'class_balance' in profile
        assert 'feature_correlation' in profile
        assert 'dataset_complexity' in profile
    
    def test_model_recommendation(self, selector, mixed_dataset):
        """Test model recommendation based on dataset characteristics."""
        X_train, X_test, y_train, y_test = mixed_dataset
        
        recommended_models = selector.recommend_models(X_train, y_train)
        
        assert isinstance(recommended_models, list)
        assert len(recommended_models) > 0
        
        for model_info in recommended_models:
            assert 'name' in model_info
            assert 'model' in model_info
            assert 'reason' in model_info
            assert 'priority' in model_info
    
    def test_auto_selection_with_budget(self, selector, mixed_dataset):
        """Test automatic model selection with time budget."""
        X_train, X_test, y_train, y_test = mixed_dataset
        
        start_time = time.time()
        
        best_model, results = selector.auto_select(
            X_train, y_train,
            time_budget_minutes=2,
            max_models=3
        )
        
        elapsed_time = time.time() - start_time
        
        assert elapsed_time < 150  # Should respect time budget (with some buffer)
        assert best_model is not None
        assert 'evaluation_results' in results
        assert len(results['models_evaluated']) <= 3
    
    def test_feature_importance_based_selection(self, selector, mixed_dataset):
        """Test model selection based on feature importance analysis."""
        X_train, X_test, y_train, y_test = mixed_dataset
        
        # Focus on models that provide feature importance
        tree_based_selection = selector.select_interpretable_models(
            X_train, y_train,
            require_feature_importance=True
        )
        
        assert 'best_model' in tree_based_selection
        assert 'feature_importance' in tree_based_selection
        assert len(tree_based_selection['feature_importance']) == X_train.shape[1]
    
    def test_progressive_model_evaluation(self, selector, mixed_dataset):
        """Test progressive model evaluation strategy."""
        X_train, X_test, y_train, y_test = mixed_dataset
        
        # Start with fast models, progress to complex ones
        evaluation_results = selector.progressive_evaluation(
            X_train, y_train,
            start_simple=True,
            performance_threshold=0.8
        )
        
        assert 'evaluation_order' in evaluation_results
        assert 'best_model_at_each_stage' in evaluation_results
        assert 'stopping_reason' in evaluation_results


class TestModelComparator:
    """Test ModelComparator class."""
    
    @pytest.fixture
    def comparator(self):
        """Create model comparator."""
        return ModelComparator()
    
    @pytest.fixture
    def trained_models(self):
        """Create trained models for comparison."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=200),
            'random_forest': RandomForestClassifier(n_estimators=50, random_state=42),
            'svm': SVC(random_state=42, probability=True)
        }
        
        # Train all models
        for name, model in models.items():
            model.fit(X_train, y_train)
        
        return models, (X_train, X_test, y_train, y_test)
    
    def test_statistical_comparison(self, comparator, trained_models):
        """Test statistical significance testing between models."""
        models, (X_train, X_test, y_train, y_test) = trained_models
        
        comparison_results = comparator.statistical_comparison(
            models, X_test, y_test, cv=5
        )
        
        assert 'pairwise_comparisons' in comparison_results
        assert 'ranking' in comparison_results
        assert 'statistical_significance' in comparison_results
        
        # Check that p-values are present
        for comparison in comparison_results['pairwise_comparisons']:
            assert 'p_value' in comparison
            assert 0 <= comparison['p_value'] <= 1
    
    def test_performance_profiling(self, comparator, trained_models):
        """Test performance profiling of models."""
        models, (X_train, X_test, y_train, y_test) = trained_models
        
        profiling_results = comparator.profile_performance(
            models, X_test, y_test
        )
        
        assert 'training_time' in profiling_results
        assert 'prediction_time' in profiling_results
        assert 'memory_usage' in profiling_results
        assert 'model_complexity' in profiling_results
        
        for model_name in models.keys():
            assert model_name in profiling_results['training_time']
            assert model_name in profiling_results['prediction_time']
    
    def test_robustness_testing(self, comparator, trained_models):
        """Test model robustness comparison."""
        models, (X_train, X_test, y_train, y_test) = trained_models
        
        robustness_results = comparator.test_robustness(
            models, X_test, y_test,
            noise_levels=[0.01, 0.05, 0.1]
        )
        
        assert 'noise_sensitivity' in robustness_results
        assert 'outlier_sensitivity' in robustness_results
        assert 'stability_scores' in robustness_results
        
        for model_name in models.keys():
            assert model_name in robustness_results['stability_scores']
    
    def test_calibration_comparison(self, comparator, trained_models):
        """Test probability calibration comparison."""
        models, (X_train, X_test, y_train, y_test) = trained_models
        
        calibration_results = comparator.compare_calibration(
            models, X_test, y_test
        )
        
        assert 'brier_scores' in calibration_results
        assert 'calibration_errors' in calibration_results
        assert 'reliability_diagrams' in calibration_results
        
        for model_name in models.keys():
            if hasattr(models[model_name], 'predict_proba'):
                assert model_name in calibration_results['brier_scores']
    
    def test_comprehensive_comparison_report(self, comparator, trained_models):
        """Test comprehensive comparison report generation."""
        models, (X_train, X_test, y_train, y_test) = trained_models
        
        report = comparator.generate_comprehensive_report(
            models, X_test, y_test
        )
        
        assert 'summary' in report
        assert 'detailed_metrics' in report
        assert 'recommendations' in report
        assert 'visualizations' in report
        
        # Check that recommendations are actionable
        assert len(report['recommendations']) > 0
        for rec in report['recommendations']:
            assert 'action' in rec
            assert 'reasoning' in rec


class TestHyperparameterOptimizer:
    """Test HyperparameterOptimizer class."""
    
    @pytest.fixture
    def optimizer(self):
        """Create hyperparameter optimizer."""
        return HyperparameterOptimizer()
    
    @pytest.fixture
    def optimization_data(self):
        """Generate data for hyperparameter optimization."""
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_grid_search_optimization(self, optimizer, optimization_data):
        """Test grid search hyperparameter optimization."""
        X_train, X_test, y_train, y_test = optimization_data
        
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [3, 5, None],
            'min_samples_split': [2, 5]
        }
        
        best_params, results = optimizer.grid_search(
            model, param_grid, X_train, y_train, cv=3
        )
        
        assert 'n_estimators' in best_params
        assert 'max_depth' in best_params
        assert 'best_score' in results
        assert 'cv_results' in results
    
    def test_random_search_optimization(self, optimizer, optimization_data):
        """Test random search hyperparameter optimization."""
        X_train, X_test, y_train, y_test = optimization_data
        
        model = RandomForestClassifier(random_state=42)
        param_distributions = {
            'n_estimators': [10, 20, 50, 100, 200],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        best_params, results = optimizer.random_search(
            model, param_distributions, X_train, y_train,
            n_iter=20, cv=3
        )
        
        assert 'n_estimators' in best_params
        assert 'best_score' in results
        assert results['best_score'] > 0
    
    def test_bayesian_optimization(self, optimizer, optimization_data):
        """Test Bayesian optimization."""
        X_train, X_test, y_train, y_test = optimization_data
        
        model = RandomForestClassifier(random_state=42)
        
        # Define parameter space for Bayesian optimization
        param_space = {
            'n_estimators': (10, 200),
            'max_depth': (3, 10),
            'min_samples_split': (2, 20)
        }
        
        best_params, results = optimizer.bayesian_optimization(
            model, param_space, X_train, y_train,
            n_iterations=10, cv=3
        )
        
        assert 'n_estimators' in best_params
        assert 'optimization_history' in results
        assert len(results['optimization_history']) <= 10
    
    def test_multi_objective_optimization(self, optimizer, optimization_data):
        """Test multi-objective hyperparameter optimization."""
        X_train, X_test, y_train, y_test = optimization_data
        
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [3, 5, None]
        }
        
        # Optimize for both accuracy and model complexity
        objectives = ['accuracy', 'model_size']
        
        pareto_solutions, results = optimizer.multi_objective_optimization(
            model, param_grid, X_train, y_train,
            objectives=objectives, cv=3
        )
        
        assert len(pareto_solutions) > 0
        assert 'pareto_front' in results
        assert 'trade_off_analysis' in results
    
    def test_early_stopping_optimization(self, optimizer, optimization_data):
        """Test optimization with early stopping."""
        X_train, X_test, y_train, y_test = optimization_data
        
        model = RandomForestClassifier(random_state=42)
        param_distributions = {
            'n_estimators': [10, 20, 50, 100, 200],
            'max_depth': [3, 5, 7, None]
        }
        
        best_params, results = optimizer.random_search(
            model, param_distributions, X_train, y_train,
            n_iter=50, cv=3,
            early_stopping=True,
            early_stopping_rounds=5,
            early_stopping_threshold=0.95
        )
        
        assert 'n_estimators' in best_params
        assert 'early_stopped' in results
        # May have stopped early
        assert len(results.get('cv_results', [])) <= 50


class TestCrossValidationPipeline:
    """Test CrossValidationPipeline class."""
    
    @pytest.fixture
    def cv_pipeline(self):
        """Create cross-validation pipeline."""
        return CrossValidationPipeline()
    
    @pytest.fixture
    def cv_data(self):
        """Generate data for cross-validation testing."""
        X, y = make_classification(
            n_samples=400,
            n_features=12,
            n_classes=3,
            random_state=42
        )
        return X, y
    
    def test_stratified_cv(self, cv_pipeline, cv_data):
        """Test stratified cross-validation."""
        X, y = cv_data
        
        model = LogisticRegression(random_state=42, max_iter=200)
        
        cv_results = cv_pipeline.stratified_cross_validation(
            model, X, y, cv=5, scoring='accuracy'
        )
        
        assert 'scores' in cv_results
        assert 'mean_score' in cv_results
        assert 'std_score' in cv_results
        assert len(cv_results['scores']) == 5
        
        # Check that stratification preserved class distribution
        assert 'fold_distributions' in cv_results
    
    def test_time_series_cv(self, cv_pipeline):
        """Test time series cross-validation."""
        # Create time series data
        n_samples = 200
        X = np.random.randn(n_samples, 5)
        y = np.cumsum(np.random.randn(n_samples))  # Time series target
        
        model = LinearRegression()
        
        cv_results = cv_pipeline.time_series_cross_validation(
            model, X, y, n_splits=5, test_size=20
        )
        
        assert 'scores' in cv_results
        assert 'mean_score' in cv_results
        assert len(cv_results['scores']) == 5
    
    def test_nested_cv(self, cv_pipeline, cv_data):
        """Test nested cross-validation for unbiased performance estimation."""
        X, y = cv_data
        
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [10, 50],
            'max_depth': [3, 5]
        }
        
        nested_cv_results = cv_pipeline.nested_cross_validation(
            model, param_grid, X, y,
            inner_cv=3, outer_cv=3
        )
        
        assert 'outer_scores' in nested_cv_results
        assert 'best_params_per_fold' in nested_cv_results
        assert 'unbiased_score' in nested_cv_results
        assert len(nested_cv_results['outer_scores']) == 3
    
    def test_custom_cv_splitter(self, cv_pipeline, cv_data):
        """Test cross-validation with custom splitter."""
        X, y = cv_data
        
        from sklearn.model_selection import GroupKFold
        
        # Create artificial groups
        groups = np.random.randint(0, 10, len(y))
        
        model = LogisticRegression(random_state=42, max_iter=200)
        
        cv_results = cv_pipeline.custom_cross_validation(
            model, X, y,
            cv_splitter=GroupKFold(n_splits=3),
            groups=groups
        )
        
        assert 'scores' in cv_results
        assert len(cv_results['scores']) == 3
    
    def test_cross_validation_with_preprocessing(self, cv_pipeline, cv_data):
        """Test cross-validation with preprocessing pipeline."""
        X, y = cv_data
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import SelectKBest
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(k=8)),
            ('classifier', LogisticRegression(random_state=42, max_iter=200))
        ])
        
        cv_results = cv_pipeline.cross_validate_pipeline(
            pipeline, X, y, cv=5, scoring=['accuracy', 'f1_weighted']
        )
        
        assert 'accuracy' in cv_results
        assert 'f1_weighted' in cv_results
        assert len(cv_results['accuracy']['scores']) == 5


class TestModelEnsemblePipeline:
    """Test ModelEnsemblePipeline class."""
    
    @pytest.fixture
    def ensemble_pipeline(self):
        """Create ensemble pipeline."""
        return ModelEnsemblePipeline()
    
    @pytest.fixture
    def ensemble_data(self):
        """Generate data for ensemble testing."""
        X, y = make_classification(
            n_samples=300,
            n_features=15,
            n_classes=2,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_voting_ensemble(self, ensemble_pipeline, ensemble_data):
        """Test voting ensemble creation and evaluation."""
        X_train, X_test, y_train, y_test = ensemble_data
        
        base_models = [
            ('lr', LogisticRegression(random_state=42, max_iter=200)),
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('svm', SVC(random_state=42, probability=True))
        ]
        
        voting_ensemble = ensemble_pipeline.create_voting_ensemble(
            base_models, voting='soft'
        )
        
        voting_ensemble.fit(X_train, y_train)
        predictions = voting_ensemble.predict(X_test)
        probabilities = voting_ensemble.predict_proba(X_test)
        
        assert len(predictions) == len(X_test)
        assert probabilities.shape == (len(X_test), 2)
    
    def test_stacking_ensemble(self, ensemble_pipeline, ensemble_data):
        """Test stacking ensemble creation."""
        X_train, X_test, y_train, y_test = ensemble_data
        
        base_models = [
            ('lr', LogisticRegression(random_state=42, max_iter=200)),
            ('rf', RandomForestClassifier(n_estimators=30, random_state=42))
        ]
        
        meta_model = LogisticRegression(random_state=42, max_iter=200)
        
        stacking_ensemble = ensemble_pipeline.create_stacking_ensemble(
            base_models, meta_model, cv=3
        )
        
        stacking_ensemble.fit(X_train, y_train)
        predictions = stacking_ensemble.predict(X_test)
        
        assert len(predictions) == len(X_test)
    
    def test_bagging_ensemble(self, ensemble_pipeline, ensemble_data):
        """Test bagging ensemble creation."""
        X_train, X_test, y_train, y_test = ensemble_data
        
        base_model = LogisticRegression(random_state=42, max_iter=200)
        
        bagging_ensemble = ensemble_pipeline.create_bagging_ensemble(
            base_model, n_estimators=10, random_state=42
        )
        
        bagging_ensemble.fit(X_train, y_train)
        predictions = bagging_ensemble.predict(X_test)
        
        assert len(predictions) == len(X_test)
    
    def test_dynamic_ensemble_selection(self, ensemble_pipeline, ensemble_data):
        """Test dynamic ensemble selection."""
        X_train, X_test, y_train, y_test = ensemble_data
        
        base_models = {
            'lr': LogisticRegression(random_state=42, max_iter=200),
            'rf': RandomForestClassifier(n_estimators=30, random_state=42),
            'svm': SVC(random_state=42, probability=True)
        }
        
        # Train all base models
        for model in base_models.values():
            model.fit(X_train, y_train)
        
        dynamic_ensemble = ensemble_pipeline.create_dynamic_ensemble(
            base_models, selection_strategy='best_local_accuracy'
        )
        
        predictions = dynamic_ensemble.predict(X_test)
        
        assert len(predictions) == len(X_test)
    
    def test_ensemble_optimization(self, ensemble_pipeline, ensemble_data):
        """Test ensemble optimization."""
        X_train, X_test, y_train, y_test = ensemble_data
        
        base_models = [
            LogisticRegression(random_state=42, max_iter=200),
            RandomForestClassifier(n_estimators=30, random_state=42),
            SVC(random_state=42, probability=True)
        ]
        
        optimized_ensemble = ensemble_pipeline.optimize_ensemble(
            base_models, X_train, y_train,
            optimization_method='genetic_algorithm',
            cv=3
        )
        
        optimized_ensemble.fit(X_train, y_train)
        predictions = optimized_ensemble.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert hasattr(optimized_ensemble, 'weights_') or hasattr(optimized_ensemble, 'selected_models_')


class TestPerformanceTracker:
    """Test PerformanceTracker class."""
    
    @pytest.fixture
    def tracker(self):
        """Create performance tracker."""
        return PerformanceTracker()
    
    def test_experiment_tracking(self, tracker):
        """Test experiment tracking functionality."""
        # Log a mock experiment
        experiment_id = tracker.start_experiment(
            name='test_experiment',
            description='Testing model selection',
            tags=['classification', 'test']
        )
        
        # Log some metrics
        tracker.log_metric(experiment_id, 'accuracy', 0.85)
        tracker.log_metric(experiment_id, 'f1_score', 0.82)
        
        # Log parameters
        tracker.log_parameter(experiment_id, 'model_type', 'random_forest')
        tracker.log_parameter(experiment_id, 'n_estimators', 100)
        
        # End experiment
        tracker.end_experiment(experiment_id)
        
        # Retrieve experiment
        experiment_data = tracker.get_experiment(experiment_id)
        
        assert experiment_data['name'] == 'test_experiment'
        assert 'accuracy' in experiment_data['metrics']
        assert 'model_type' in experiment_data['parameters']
    
    def test_model_comparison_tracking(self, tracker):
        """Test tracking multiple model comparisons."""
        models_data = [
            {'name': 'LogisticRegression', 'accuracy': 0.85, 'f1': 0.82},
            {'name': 'RandomForest', 'accuracy': 0.89, 'f1': 0.87},
            {'name': 'SVM', 'accuracy': 0.83, 'f1': 0.80}
        ]
        
        comparison_id = tracker.start_model_comparison('classification_comparison')
        
        for model_data in models_data:
            tracker.log_model_performance(
                comparison_id,
                model_data['name'],
                {'accuracy': model_data['accuracy'], 'f1': model_data['f1']}
            )
        
        tracker.end_model_comparison(comparison_id)
        
        comparison_results = tracker.get_model_comparison(comparison_id)
        
        assert len(comparison_results['models']) == 3
        assert 'best_model' in comparison_results
        assert comparison_results['best_model']['name'] == 'RandomForest'
    
    def test_performance_history(self, tracker):
        """Test performance history tracking."""
        model_name = 'test_model'
        
        # Log performance over time
        for i in range(5):
            tracker.log_performance_snapshot(
                model_name,
                timestamp=f'2024-01-{i+1:02d}',
                metrics={'accuracy': 0.8 + i * 0.02, 'loss': 0.5 - i * 0.05}
            )
        
        history = tracker.get_performance_history(model_name)
        
        assert len(history) == 5
        assert history[-1]['metrics']['accuracy'] > history[0]['metrics']['accuracy']
    
    def test_performance_alerts(self, tracker):
        """Test performance degradation alerts."""
        model_name = 'monitored_model'
        
        # Set up performance monitoring
        tracker.setup_performance_monitoring(
            model_name,
            alert_threshold={'accuracy': 0.8, 'f1': 0.75},
            alert_callback=lambda alert: print(f"Alert: {alert}")
        )
        
        # Log performance that should trigger alert
        alert_triggered = tracker.check_performance_alert(
            model_name,
            {'accuracy': 0.75, 'f1': 0.70}  # Below thresholds
        )
        
        assert alert_triggered
    
    def test_experiment_comparison(self, tracker):
        """Test comparing multiple experiments."""
        # Create multiple experiments
        exp_ids = []
        for i in range(3):
            exp_id = tracker.start_experiment(f'experiment_{i}')
            tracker.log_metric(exp_id, 'accuracy', 0.8 + i * 0.05)
            tracker.log_metric(exp_id, 'f1', 0.75 + i * 0.04)
            tracker.end_experiment(exp_id)
            exp_ids.append(exp_id)
        
        comparison = tracker.compare_experiments(exp_ids, metric='accuracy')
        
        assert 'ranking' in comparison
        assert 'best_experiment' in comparison
        assert len(comparison['experiments']) == 3


class TestModelRegistry:
    """Test ModelRegistry class."""
    
    @pytest.fixture
    def registry(self):
        """Create model registry."""
        return ModelRegistry()
    
    def test_model_registration(self, registry):
        """Test registering models in the registry."""
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model.fit(X, y)
        
        # Register model
        model_id = registry.register_model(
            model=model,
            name='test_random_forest',
            version='1.0.0',
            description='Test Random Forest model',
            metadata={
                'training_data_size': len(X),
                'features': X.shape[1],
                'algorithm': 'RandomForest'
            }
        )
        
        assert model_id is not None
        
        # Retrieve model
        retrieved_model = registry.get_model(model_id)
        
        assert retrieved_model is not None
        assert hasattr(retrieved_model, 'predict')
    
    def test_model_versioning(self, registry):
        """Test model versioning system."""
        from sklearn.linear_model import LogisticRegression
        
        # Register multiple versions
        for version in ['1.0.0', '1.1.0', '2.0.0']:
            model = LogisticRegression(random_state=42, max_iter=200)
            X, y = make_classification(n_samples=50, n_features=3, random_state=42)
            model.fit(X, y)
            
            registry.register_model(
                model=model,
                name='logistic_model',
                version=version,
                description=f'Logistic model version {version}'
            )
        
        # Get all versions
        versions = registry.get_model_versions('logistic_model')
        
        assert len(versions) == 3
        assert '2.0.0' in [v['version'] for v in versions]
        
        # Get latest version
        latest_model = registry.get_latest_model('logistic_model')
        
        assert latest_model is not None
    
    def test_model_deployment_tracking(self, registry):
        """Test tracking model deployments."""
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model.fit(X, y)
        
        model_id = registry.register_model(
            model=model,
            name='deployment_test_model',
            version='1.0.0'
        )
        
        # Mark as deployed
        deployment_id = registry.deploy_model(
            model_id=model_id,
            environment='production',
            endpoint='api/v1/predict',
            deployment_config={'instances': 3, 'memory': '2GB'}
        )
        
        # Track deployment status
        deployment_info = registry.get_deployment_info(deployment_id)
        
        assert deployment_info['environment'] == 'production'
        assert deployment_info['status'] == 'deployed'
    
    def test_model_performance_tracking(self, registry):
        """Test tracking model performance in registry."""
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=30, random_state=42)
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model.fit(X, y)
        
        model_id = registry.register_model(
            model=model,
            name='performance_tracked_model',
            version='1.0.0'
        )
        
        # Log performance metrics
        registry.log_model_performance(
            model_id=model_id,
            metrics={'accuracy': 0.89, 'f1': 0.86, 'precision': 0.88},
            dataset='test_set_v1',
            timestamp='2024-01-15'
        )
        
        # Retrieve performance history
        performance_history = registry.get_performance_history(model_id)
        
        assert len(performance_history) == 1
        assert performance_history[0]['metrics']['accuracy'] == 0.89
    
    def test_model_search_and_discovery(self, registry):
        """Test model search and discovery features."""
        # Register models with different tags and metadata
        models_to_register = [
            {'name': 'classification_model_1', 'tags': ['classification', 'production']},
            {'name': 'classification_model_2', 'tags': ['classification', 'experimental']},
            {'name': 'regression_model_1', 'tags': ['regression', 'production']}
        ]
        
        for model_info in models_to_register:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            X, y = make_classification(n_samples=50, n_features=3, random_state=42)
            model.fit(X, y)
            
            registry.register_model(
                model=model,
                name=model_info['name'],
                version='1.0.0',
                tags=model_info['tags']
            )
        
        # Search by tags
        classification_models = registry.search_models(tags=['classification'])
        production_models = registry.search_models(tags=['production'])
        
        assert len(classification_models) == 2
        assert len(production_models) == 2
        
        # Search by name pattern
        classification_pattern = registry.search_models(name_pattern='classification*')
        
        assert len(classification_pattern) == 2


class TestBayesianOptimizer:
    """Test BayesianOptimizer class."""
    
    @pytest.fixture
    def optimizer(self):
        """Create Bayesian optimizer."""
        return BayesianOptimizer()
    
    @pytest.fixture
    def optimization_problem(self):
        """Create optimization problem setup."""
        X, y = make_classification(
            n_samples=200,
            n_features=8,
            n_classes=2,
            random_state=42
        )
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42)
        
        return model, X, y
    
    def test_basic_bayesian_optimization(self, optimizer, optimization_problem):
        """Test basic Bayesian optimization functionality."""
        model, X, y = optimization_problem
        
        # Define search space
        search_space = {
            'n_estimators': (10, 100),
            'max_depth': (3, 20),
            'min_samples_split': (2, 10)
        }
        
        # Run optimization
        best_params, results = optimizer.optimize(
            model=model,
            search_space=search_space,
            X=X, y=y,
            n_iterations=15,
            cv=3,
            scoring='accuracy'
        )
        
        assert 'n_estimators' in best_params
        assert 'max_depth' in best_params
        assert 'optimization_history' in results
        assert len(results['optimization_history']) <= 15
    
    def test_acquisition_functions(self, optimizer, optimization_problem):
        """Test different acquisition functions."""
        model, X, y = optimization_problem
        
        search_space = {
            'n_estimators': (10, 50),
            'max_depth': (3, 10)
        }
        
        acquisition_functions = ['expected_improvement', 'probability_of_improvement', 'upper_confidence_bound']
        
        for acq_func in acquisition_functions:
            best_params, results = optimizer.optimize(
                model=model,
                search_space=search_space,
                X=X, y=y,
                n_iterations=5,
                acquisition_function=acq_func,
                cv=3
            )
            
            assert 'n_estimators' in best_params
            assert 'best_score' in results
    
    def test_multi_objective_bayesian_optimization(self, optimizer, optimization_problem):
        """Test multi-objective Bayesian optimization."""
        model, X, y = optimization_problem
        
        search_space = {
            'n_estimators': (10, 100),
            'max_depth': (3, 15)
        }
        
        # Optimize for accuracy and model complexity (inverse of n_estimators)
        objectives = ['accuracy', 'model_simplicity']
        
        pareto_solutions, results = optimizer.multi_objective_optimize(
            model=model,
            search_space=search_space,
            X=X, y=y,
            objectives=objectives,
            n_iterations=10,
            cv=3
        )
        
        assert len(pareto_solutions) > 0
        assert 'pareto_front' in results
        assert 'hypervolume' in results
    
    def test_early_stopping_bayesian(self, optimizer, optimization_problem):
        """Test early stopping in Bayesian optimization."""
        model, X, y = optimization_problem
        
        search_space = {
            'n_estimators': (10, 200),
            'max_depth': (3, 20)
        }
        
        best_params, results = optimizer.optimize(
            model=model,
            search_space=search_space,
            X=X, y=y,
            n_iterations=30,
            early_stopping_rounds=5,
            early_stopping_threshold=0.95,
            cv=3
        )
        
        assert 'early_stopped' in results
        # May have stopped before 30 iterations
        assert len(results['optimization_history']) <= 30
    
    def test_warm_start_optimization(self, optimizer, optimization_problem):
        """Test warm start with previous optimization results."""
        model, X, y = optimization_problem
        
        search_space = {
            'n_estimators': (10, 100),
            'max_depth': (3, 15)
        }
        
        # First optimization run
        best_params_1, results_1 = optimizer.optimize(
            model=model,
            search_space=search_space,
            X=X, y=y,
            n_iterations=5,
            cv=3
        )
        
        # Second optimization run with warm start
        best_params_2, results_2 = optimizer.optimize(
            model=model,
            search_space=search_space,
            X=X, y=y,
            n_iterations=5,
            warm_start=True,
            previous_results=results_1['optimization_history'],
            cv=3
        )
        
        assert 'n_estimators' in best_params_2
        # Second run should benefit from previous knowledge
        assert results_2['best_score'] >= results_1['best_score']


class TestGridSearchPipeline:
    """Test GridSearchPipeline class."""
    
    @pytest.fixture
    def grid_pipeline(self):
        """Create grid search pipeline."""
        return GridSearchPipeline()
    
    @pytest.fixture
    def grid_search_data(self):
        """Generate data for grid search testing."""
        X, y = make_classification(
            n_samples=250,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_exhaustive_grid_search(self, grid_pipeline, grid_search_data):
        """Test exhaustive grid search."""
        X_train, X_test, y_train, y_test = grid_search_data
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42)
        
        param_grid = {
            'n_estimators': [10, 50],
            'max_depth': [3, 5, None],
            'min_samples_split': [2, 5]
        }
        
        best_model, results = grid_pipeline.exhaustive_search(
            model=model,
            param_grid=param_grid,
            X=X_train, y=y_train,
            cv=3,
            scoring='accuracy'
        )
        
        assert hasattr(best_model, 'predict')
        assert 'best_params' in results
        assert 'best_score' in results
        assert 'cv_results' in results
        
        # Should have tested all combinations
        expected_combinations = 2 * 3 * 2  # 12 combinations
        assert len(results['cv_results']) == expected_combinations
    
    def test_randomized_grid_search(self, grid_pipeline, grid_search_data):
        """Test randomized grid search."""
        X_train, X_test, y_train, y_test = grid_search_data
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42)
        
        param_distributions = {
            'n_estimators': [10, 20, 50, 100],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        best_model, results = grid_pipeline.randomized_search(
            model=model,
            param_distributions=param_distributions,
            X=X_train, y=y_train,
            n_iter=20,
            cv=3,
            scoring='f1'
        )
        
        assert hasattr(best_model, 'predict')
        assert 'best_params' in results
        assert len(results['cv_results']) == 20
    
    def test_adaptive_grid_search(self, grid_pipeline, grid_search_data):
        """Test adaptive grid search that refines search space."""
        X_train, X_test, y_train, y_test = grid_search_data
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42)
        
        initial_param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [3, 7, None]
        }
        
        best_model, results = grid_pipeline.adaptive_search(
            model=model,
            initial_param_grid=initial_param_grid,
            X=X_train, y=y_train,
            refinement_rounds=2,
            cv=3,
            scoring='accuracy'
        )
        
        assert hasattr(best_model, 'predict')
        assert 'refinement_history' in results
        assert len(results['refinement_history']) <= 2
    
    def test_parallel_grid_search(self, grid_pipeline, grid_search_data):
        """Test parallel grid search execution."""
        X_train, X_test, y_train, y_test = grid_search_data
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42)
        
        param_grid = {
            'n_estimators': [10, 30, 50],
            'max_depth': [3, 5]
        }
        
        import time
        start_time = time.time()
        
        best_model, results = grid_pipeline.parallel_search(
            model=model,
            param_grid=param_grid,
            X=X_train, y=y_train,
            cv=3,
            n_jobs=2,  # Use 2 parallel jobs
            scoring='accuracy'
        )
        
        execution_time = time.time() - start_time
        
        assert hasattr(best_model, 'predict')
        assert 'execution_time' in results
        # Parallel execution should be reasonably fast
        assert execution_time < 30  # seconds
    
    def test_constrained_grid_search(self, grid_pipeline, grid_search_data):
        """Test grid search with constraints."""
        X_train, X_test, y_train, y_test = grid_search_data
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42)
        
        param_grid = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [3, 5, 7, 10]
        }
        
        # Define constraint: n_estimators * max_depth <= 500
        def constraint_function(params):
            return params['n_estimators'] * params['max_depth'] <= 500
        
        best_model, results = grid_pipeline.constrained_search(
            model=model,
            param_grid=param_grid,
            constraint_function=constraint_function,
            X=X_train, y=y_train,
            cv=3,
            scoring='accuracy'
        )
        
        assert hasattr(best_model, 'predict')
        assert 'feasible_combinations' in results
        # Should have fewer combinations due to constraints
        assert results['feasible_combinations'] < 4 * 4  # Less than 16


class TestModelSelectionIntegration:
    """Integration tests for model selection pipeline components."""
    
    def test_end_to_end_model_selection(self):
        """Test complete end-to-end model selection pipeline."""
        # Generate comprehensive dataset
        X, y = make_classification(
            n_samples=400,
            n_features=20,
            n_classes=3,
            n_informative=15,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # 1. Auto model selection
        selector = AutoModelSelector()
        recommended_models = selector.recommend_models(X_train, y_train)
        
        # 2. Model comparison
        comparator = ModelComparator()
        models = {rec['name']: rec['model'] for rec in recommended_models[:3]}
        
        for model in models.values():
            model.fit(X_train, y_train)
        
        comparison_results = comparator.statistical_comparison(models, X_test, y_test)
        
        # 3. Hyperparameter optimization for best model
        best_model_name = comparison_results['ranking'][0]['model']
        best_model = models[best_model_name]
        
        optimizer = HyperparameterOptimizer()
        
        if 'RandomForest' in best_model_name:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None]
            }
        else:
            param_grid = {'C': [0.1, 1.0, 10.0]}
        
        optimized_params, opt_results = optimizer.grid_search(
            best_model, param_grid, X_train, y_train, cv=3
        )
        
        # 4. Final model evaluation
        final_model = type(best_model)(**optimized_params, random_state=42)
        final_model.fit(X_train, y_train)
        final_score = final_model.score(X_test, y_test)
        
        # Assertions
        assert len(recommended_models) > 0
        assert 'ranking' in comparison_results
        assert optimized_params is not None
        assert 0 <= final_score <= 1
        
        print(f"End-to-end pipeline completed. Final score: {final_score:.3f}")
    
    def test_model_selection_with_registry(self):
        """Test model selection with model registry integration."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Initialize components
        registry = ModelRegistry()
        selector = ModelSelectionPipeline()
        tracker = PerformanceTracker()
        
        # 1. Select and train models
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=200),
            'random_forest': RandomForestClassifier(n_estimators=50, random_state=42)
        }
        
        best_model_name, selection_results = selector.select_best_model(
            models, X_train, y_train, cv=3
        )
        
        # 2. Register best model
        best_model = models[best_model_name]
        best_model.fit(X_train, y_train)
        
        model_id = registry.register_model(
            model=best_model,
            name=f'selected_{best_model_name}',
            version='1.0.0',
            metadata={
                'selection_score': selection_results['mean_score'][best_model_name],
                'cv_std': selection_results.get('std_score', {}).get(best_model_name, 0)
            }
        )
        
        # 3. Track performance
        test_score = best_model.score(X_test, y_test)
        registry.log_model_performance(
            model_id=model_id,
            metrics={'test_accuracy': test_score},
            dataset='test_set'
        )
        
        # Verify integration
        assert model_id is not None
        retrieved_model = registry.get_model(model_id)
        assert retrieved_model is not None
        
        performance_history = registry.get_performance_history(model_id)
        assert len(performance_history) == 1
    
    def test_ensemble_pipeline_integration(self):
        """Test integration of ensemble methods in model selection."""
        X, y = make_classification(n_samples=300, n_features=12, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 1. Select base models
        selector = AutoModelSelector()
        recommended_models = selector.recommend_models(X_train, y_train)
        
        # 2. Create ensemble
        ensemble_pipeline = ModelEnsemblePipeline()
        
        base_models = [
            ('lr', LogisticRegression(random_state=42, max_iter=200)),
            ('rf', RandomForestClassifier(n_estimators=30, random_state=42)),
            ('svm', SVC(random_state=42, probability=True))
        ]
        
        # Train base models
        for name, model in base_models:
            model.fit(X_train, y_train)
        
        # Create voting ensemble
        voting_ensemble = ensemble_pipeline.create_voting_ensemble(
            base_models, voting='soft'
        )
        voting_ensemble.fit(X_train, y_train)
        
        # Create stacking ensemble
        stacking_ensemble = ensemble_pipeline.create_stacking_ensemble(
            base_models, 
            meta_model=LogisticRegression(random_state=42, max_iter=200),
            cv=3
        )
        stacking_ensemble.fit(X_train, y_train)
        
        # 3. Compare ensemble methods
        comparator = ModelComparator()
        ensemble_models = {
            'voting': voting_ensemble,
            'stacking': stacking_ensemble
        }
        
        comparison_results = comparator.statistical_comparison(
            ensemble_models, X_test, y_test
        )
        
        # Verify integration
        assert 'ranking' in comparison_results
        assert len(comparison_results['ranking']) == 2
        
        # Ensemble should perform reasonably well
        for model_name in ensemble_models:
            score = ensemble_models[model_name].score(X_test, y_test)
            assert score > 0.5  # Reasonable performance threshold


if __name__ == "__main__":
    pytest.main([__file__])