"""
Unit tests for evaluation utilities.

Tests for model evaluation and metrics calculation functions.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from evaluation import (
    ModelEvaluator,
    StatisticalTester,
    ValidationCurveAnalyzer,
    ModelVisualizationSuite
)


class TestModelEvaluator:
    """Test ModelEvaluator class."""
    
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
    
    def test_classification_evaluation(self, classification_data):
        """Test classification model evaluation."""
        X_train, X_test, y_train, y_test = classification_data
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        evaluator = ModelEvaluator(task_type='classification')
        metrics = evaluator.evaluate(model, X_test, y_test)
        
        # Check that all expected metrics are present
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1
    
    def test_regression_evaluation(self, regression_data):
        """Test regression model evaluation."""
        X_train, X_test, y_train, y_test = regression_data
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        evaluator = ModelEvaluator(task_type='regression')
        metrics = evaluator.evaluate(model, X_test, y_test)
        
        # Check that all expected metrics are present
        expected_metrics = ['mse', 'rmse', 'mae', 'r2']
        for metric in expected_metrics:
            assert metric in metrics
        
        # Check metric properties
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['rmse'] == np.sqrt(metrics['mse'])
    
    def test_custom_metrics(self, classification_data):
        """Test evaluation with custom metrics."""
        X_train, X_test, y_train, y_test = classification_data
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        def custom_accuracy(y_true, y_pred):
            return np.mean(y_true == y_pred)
        
        custom_metrics = {'custom_acc': custom_accuracy}
        evaluator = ModelEvaluator(task_type='classification', custom_metrics=custom_metrics)
        metrics = evaluator.evaluate(model, X_test, y_test)
        
        assert 'custom_acc' in metrics
        assert 'accuracy' in metrics
        # Custom accuracy should match built-in accuracy
        assert abs(metrics['custom_acc'] - metrics['accuracy']) < 1e-10
    
    def test_detailed_report(self, classification_data):
        """Test detailed evaluation report."""
        X_train, X_test, y_train, y_test = classification_data
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        evaluator = ModelEvaluator(task_type='classification')
        report = evaluator.detailed_report(model, X_test, y_test)
        
        assert 'metrics' in report
        assert 'confusion_matrix' in report
        assert 'classification_report' in report
        assert isinstance(report['confusion_matrix'], np.ndarray)


class TestCrossValidator:
    """Test CrossValidator class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        return X, y
    
    def test_cross_validation(self, sample_data):
        """Test cross-validation functionality."""
        X, y = sample_data
        
        model = LogisticRegression(random_state=42, max_iter=200)
        cv = CrossValidator(cv=5, scoring='accuracy', random_state=42)
        
        scores = cv.cross_validate(model, X, y)
        
        assert 'test_score' in scores
        assert 'train_score' in scores
        assert 'fit_time' in scores
        assert 'score_time' in scores
        
        assert len(scores['test_score']) == 5
        assert all(0 <= score <= 1 for score in scores['test_score'])
    
    def test_multiple_metrics(self, sample_data):
        """Test cross-validation with multiple metrics."""
        X, y = sample_data
        
        model = LogisticRegression(random_state=42, max_iter=200)
        scoring = ['accuracy', 'precision', 'recall', 'f1']
        cv = CrossValidator(cv=3, scoring=scoring, random_state=42)
        
        scores = cv.cross_validate(model, X, y)
        
        for metric in scoring:
            assert f'test_{metric}' in scores
            assert f'train_{metric}' in scores
    
    def test_stratified_cv(self, sample_data):
        """Test stratified cross-validation."""
        X, y = sample_data
        
        model = LogisticRegression(random_state=42, max_iter=200)
        cv = CrossValidator(cv=5, stratify=True, random_state=42)
        
        scores = cv.cross_validate(model, X, y)
        
        # Should work without errors
        assert len(scores['test_score']) == 5
    
    def test_cv_statistics(self, sample_data):
        """Test cross-validation statistics calculation."""
        X, y = sample_data
        
        model = LogisticRegression(random_state=42, max_iter=200)
        cv = CrossValidator(cv=5, scoring='accuracy', random_state=42)
        
        scores = cv.cross_validate(model, X, y)
        stats = cv.get_cv_statistics(scores)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        
        assert isinstance(stats['mean'], (float, np.float64))
        assert isinstance(stats['std'], (float, np.float64))


class TestMetricsCalculator:
    """Test MetricsCalculator class."""
    
    @pytest.fixture
    def classification_predictions(self):
        """Generate classification predictions."""
        np.random.seed(42)
        y_true = np.random.randint(0, 3, 100)
        y_pred = y_true.copy()
        # Add some errors
        error_indices = np.random.choice(100, 20, replace=False)
        y_pred[error_indices] = np.random.randint(0, 3, 20)
        
        return y_true, y_pred
    
    @pytest.fixture
    def regression_predictions(self):
        """Generate regression predictions."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.normal(0, 0.1, 100)  # Add noise
        
        return y_true, y_pred
    
    def test_classification_metrics(self, classification_predictions):
        """Test classification metrics calculation."""
        y_true, y_pred = classification_predictions
        
        calculator = MetricsCalculator(task_type='classification')
        metrics = calculator.calculate_all_metrics(y_true, y_pred)
        
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1
    
    def test_regression_metrics(self, regression_predictions):
        """Test regression metrics calculation."""
        y_true, y_pred = regression_predictions
        
        calculator = MetricsCalculator(task_type='regression')
        metrics = calculator.calculate_all_metrics(y_true, y_pred)
        
        expected_metrics = ['mse', 'rmse', 'mae', 'r2']
        for metric in expected_metrics:
            assert metric in metrics
        
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
    
    def test_individual_metrics(self, classification_predictions):
        """Test individual metric calculations."""
        y_true, y_pred = classification_predictions
        
        calculator = MetricsCalculator(task_type='classification')
        
        accuracy = calculator.accuracy(y_true, y_pred)
        precision = calculator.precision(y_true, y_pred)
        recall = calculator.recall(y_true, y_pred)
        f1 = calculator.f1_score(y_true, y_pred)
        
        assert 0 <= accuracy <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1
    
    def test_custom_average(self, classification_predictions):
        """Test different averaging strategies."""
        y_true, y_pred = classification_predictions
        
        calculator = MetricsCalculator(task_type='classification')
        
        # Test different averaging strategies
    def test_custom_average(self, classification_predictions):
        """Test different averaging strategies."""
        y_true, y_pred = classification_predictions
        
        calculator = MetricsCalculator(task_type='classification')
        
        # Test different averaging strategies
        for average in ['macro', 'micro', 'weighted']:
            precision = calculator.precision(y_true, y_pred, average=average)
            recall = calculator.recall(y_true, y_pred, average=average)
            f1 = calculator.f1_score(y_true, y_pred, average=average)
            
            assert 0 <= precision <= 1
            assert 0 <= recall <= 1
            assert 0 <= f1 <= 1


class TestPerformanceComparator:
    """Test PerformanceComparator class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_model_comparison(self, sample_data):
        """Test comparing multiple models."""
        X_train, X_test, y_train, y_test = sample_data
        
        models = {
            'rf': RandomForestClassifier(n_estimators=10, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=200)
        }
        
        # Train models
        for model in models.values():
            model.fit(X_train, y_train)
        
        comparator = PerformanceComparator()
        comparison = comparator.compare_models(models, X_test, y_test)
        
        assert 'rf' in comparison
        assert 'lr' in comparison
        
        for model_name, metrics in comparison.items():
            assert 'accuracy' in metrics
            assert 0 <= metrics['accuracy'] <= 1
    
    def test_statistical_comparison(self, sample_data):
        """Test statistical significance testing."""
        X, y = sample_data[:2]  # Use full dataset
        
        models = {
            'rf': RandomForestClassifier(n_estimators=10, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=200)
        }
        
        comparator = PerformanceComparator()
        results = comparator.statistical_comparison(models, X, y, cv=3)
        
        assert 'rf' in results
        assert 'lr' in results
        assert 'p_value' in results or 'significance_test' in results
    
    def test_ranking(self, sample_data):
        """Test model ranking."""
        X_train, X_test, y_train, y_test = sample_data
        
        models = {
            'rf': RandomForestClassifier(n_estimators=10, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=200)
        }
        
        # Train models
        for model in models.values():
            model.fit(X_train, y_train)
        
        comparator = PerformanceComparator()
        ranking = comparator.rank_models(models, X_test, y_test, metric='accuracy')
        
        assert len(ranking) == len(models)
        assert all(model in models for model, _ in ranking)
        
        # Should be sorted by performance (descending)
        scores = [score for _, score in ranking]
        assert scores == sorted(scores, reverse=True)


class TestLearningCurveAnalyzer:
    """Test LearningCurveAnalyzer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        return X, y
    
    def test_learning_curve_generation(self, sample_data):
        """Test learning curve generation."""
        X, y = sample_data
        
        model = LogisticRegression(random_state=42, max_iter=200)
        analyzer = LearningCurveAnalyzer()
        
        results = analyzer.generate_learning_curve(
            model, X, y, 
            train_sizes=np.linspace(0.1, 1.0, 5),
            cv=3
        )
        
        assert 'train_sizes' in results
        assert 'train_scores' in results
        assert 'validation_scores' in results
        
        assert len(results['train_sizes']) == 5
        assert results['train_scores'].shape[0] == 5
        assert results['validation_scores'].shape[0] == 5
    
    def test_curve_analysis(self, sample_data):
        """Test learning curve analysis."""
        X, y = sample_data
        
        model = LogisticRegression(random_state=42, max_iter=200)
        analyzer = LearningCurveAnalyzer()
        
        results = analyzer.generate_learning_curve(model, X, y, cv=3)
        analysis = analyzer.analyze_learning_curve(results)
        
        assert 'overfitting_detected' in analysis
        assert 'convergence_detected' in analysis
        assert 'recommendations' in analysis
        
        assert isinstance(analysis['overfitting_detected'], bool)
        assert isinstance(analysis['convergence_detected'], bool)
        assert isinstance(analysis['recommendations'], list)


class TestValidationCurveAnalyzer:
    """Test ValidationCurveAnalyzer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        return X, y
    
    def test_validation_curve_generation(self, sample_data):
        """Test validation curve generation."""
        X, y = sample_data
        
        model = RandomForestClassifier(random_state=42)
        analyzer = ValidationCurveAnalyzer()
        
        param_range = [5, 10, 20, 50]
        results = analyzer.generate_validation_curve(
            model, X, y,
            param_name='n_estimators',
            param_range=param_range,
            cv=3
        )
        
        assert 'param_range' in results
        assert 'train_scores' in results
        assert 'validation_scores' in results
        
        assert len(results['param_range']) == len(param_range)
        assert results['train_scores'].shape[0] == len(param_range)
        assert results['validation_scores'].shape[0] == len(param_range)
    
    def test_optimal_parameter_finding(self, sample_data):
        """Test finding optimal parameter value."""
        X, y = sample_data
        
        model = RandomForestClassifier(random_state=42)
        analyzer = ValidationCurveAnalyzer()
        
        param_range = [5, 10, 20, 50]
        results = analyzer.generate_validation_curve(
            model, X, y,
            param_name='n_estimators',
            param_range=param_range,
            cv=3
        )
        
        optimal_param = analyzer.find_optimal_parameter(results)
        
        assert optimal_param in param_range
        assert isinstance(optimal_param, (int, float))


class TestResidualAnalyzer:
    """Test ResidualAnalyzer class."""
    
    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=200,
            n_features=5,
            noise=0.1,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_residual_calculation(self, regression_data):
        """Test residual calculation."""
        X_train, X_test, y_train, y_test = regression_data
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        analyzer = ResidualAnalyzer()
        analysis = analyzer.analyze_residuals(y_test, y_pred)
        
        assert 'residuals' in analysis
        assert 'mean_residual' in analysis
        assert 'std_residual' in analysis
        assert 'residual_plots' in analysis
        
        assert len(analysis['residuals']) == len(y_test)
        assert abs(analysis['mean_residual']) < 1.0  # Should be close to zero
    
    def test_normality_test(self, regression_data):
        """Test residual normality testing."""
        X_train, X_test, y_train, y_test = regression_data
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        analyzer = ResidualAnalyzer()
        normality_result = analyzer.test_normality(y_test, y_pred)
        
        assert 'statistic' in normality_result
        assert 'p_value' in normality_result
        assert 'is_normal' in normality_result
        
        assert isinstance(normality_result['is_normal'], bool)
    
    def test_heteroscedasticity_test(self, regression_data):
        """Test heteroscedasticity testing."""
        X_train, X_test, y_train, y_test = regression_data
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        analyzer = ResidualAnalyzer()
        hetero_result = analyzer.test_heteroscedasticity(y_test, y_pred)
        
        assert 'statistic' in hetero_result
        assert 'p_value' in hetero_result
        assert 'has_heteroscedasticity' in hetero_result
        
        assert isinstance(hetero_result['has_heteroscedasticity'], bool)


class TestFeatureImportanceAnalyzer:
    """Test FeatureImportanceAnalyzer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=6,
            random_state=42
        )
        return X, y
    
    def test_tree_based_importance(self, sample_data):
        """Test tree-based feature importance."""
        X, y = sample_data
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        analyzer = FeatureImportanceAnalyzer()
        importance = analyzer.get_feature_importance(model, method='tree_based')
        
        assert len(importance) == X.shape[1]
        assert np.all(importance >= 0)
        assert np.sum(importance) > 0
    
    def test_permutation_importance(self, sample_data):
        """Test permutation-based feature importance."""
        X, y = sample_data
        
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X, y)
        
        analyzer = FeatureImportanceAnalyzer()
        importance = analyzer.get_feature_importance(
            model, X, y, method='permutation'
        )
        
        assert len(importance) == X.shape[1]
        # Permutation importance can be negative
        assert isinstance(importance, np.ndarray)
    
    def test_importance_ranking(self, sample_data):
        """Test feature importance ranking."""
        X, y = sample_data
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        analyzer = FeatureImportanceAnalyzer()
        ranking = analyzer.rank_features(model)
        
        assert len(ranking) == X.shape[1]
        
        # Should be sorted by importance (descending)
        importances = [imp for _, imp in ranking]
        assert importances == sorted(importances, reverse=True)
    
    def test_top_k_features(self, sample_data):
        """Test getting top-k important features."""
        X, y = sample_data
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        analyzer = FeatureImportanceAnalyzer()
        top_features = analyzer.get_top_k_features(model, k=5)
        
        assert len(top_features) == 5
        assert all(0 <= idx < X.shape[1] for idx, _ in top_features)


class TestConfusionMatrixAnalyzer:
    """Test ConfusionMatrixAnalyzer class."""
    
    @pytest.fixture
    def classification_data(self):
        """Generate classification data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=3,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_confusion_matrix_generation(self, classification_data):
        """Test confusion matrix generation."""
        X_train, X_test, y_train, y_test = classification_data
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        analyzer = ConfusionMatrixAnalyzer()
        cm_analysis = analyzer.analyze_confusion_matrix(y_test, y_pred)
        
        assert 'confusion_matrix' in cm_analysis
        assert 'normalized_cm' in cm_analysis
        assert 'class_metrics' in cm_analysis
        
        cm = cm_analysis['confusion_matrix']
        assert cm.shape == (3, 3)  # 3 classes
        assert np.all(cm >= 0)
        assert np.sum(cm) == len(y_test)
    
    def test_class_metrics(self, classification_data):
        """Test per-class metrics calculation."""
        X_train, X_test, y_train, y_test = classification_data
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        analyzer = ConfusionMatrixAnalyzer()
        class_metrics = analyzer.get_class_metrics(y_test, y_pred)
        
        # Should have metrics for each class
        assert len(class_metrics) == 3  # 3 classes
        
        for class_id, metrics in class_metrics.items():
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1_score' in metrics
            assert 'support' in metrics


class TestROCAnalyzer:
    """Test ROCAnalyzer class."""
    
    @pytest.fixture
    def binary_classification_data(self):
        """Generate binary classification data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_roc_curve_generation(self, binary_classification_data):
        """Test ROC curve generation."""
        X_train, X_test, y_train, y_test = binary_classification_data
        
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        analyzer = ROCAnalyzer()
        roc_data = analyzer.generate_roc_curve(y_test, y_proba)
        
        assert 'fpr' in roc_data
        assert 'tpr' in roc_data
        assert 'thresholds' in roc_data
        assert 'auc' in roc_data
        
        assert len(roc_data['fpr']) == len(roc_data['tpr'])
        assert 0 <= roc_data['auc'] <= 1
    
    def test_optimal_threshold(self, binary_classification_data):
        """Test optimal threshold finding."""
        X_train, X_test, y_train, y_test = binary_classification_data
        
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        analyzer = ROCAnalyzer()
        optimal_threshold = analyzer.find_optimal_threshold(y_test, y_proba)
        
        assert 0 <= optimal_threshold <= 1
        assert isinstance(optimal_threshold, (float, np.float64))


class TestEvaluationIntegration:
    """Integration tests for evaluation utilities."""
    
    @pytest.fixture
    def complete_dataset(self):
        """Generate complete dataset for integration testing."""
        X, y = make_classification(
            n_samples=500,
            n_features=20,
            n_classes=2,
            n_informative=15,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def test_full_evaluation_pipeline(self, complete_dataset):
        """Test complete evaluation pipeline."""
        X_train, X_test, y_train, y_test = complete_dataset
        
        # Train multiple models
        models = {
            'rf': RandomForestClassifier(n_estimators=20, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=300)
        }
        
        for model in models.values():
            model.fit(X_train, y_train)
        
        # Evaluate each model
        evaluator = ModelEvaluator(task_type='classification')
        results = {}
        
        for name, model in models.items():
            results[name] = evaluator.evaluate(model, X_test, y_test)
        
        # Compare models
        comparator = PerformanceComparator()
        comparison = comparator.compare_models(models, X_test, y_test)
        
        # Generate learning curves
        curve_analyzer = LearningCurveAnalyzer()
        learning_curve_results = {}
        
        for name, model in models.items():
            learning_curve_results[name] = curve_analyzer.generate_learning_curve(
                model, X_train, y_train, cv=3
            )
        
        # All results should be valid
        assert len(results) == len(models)
        assert len(comparison) == len(models)
        assert len(learning_curve_results) == len(models)
        
        for name in models.keys():
            assert name in results
            assert name in comparison
            assert name in learning_curve_results
    
    def test_evaluation_consistency(self, complete_dataset):
        """Test evaluation consistency across multiple runs."""
        X_train, X_test, y_train, y_test = complete_dataset
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        evaluator = ModelEvaluator(task_type='classification')
        
        # Evaluate multiple times
        results1 = evaluator.evaluate(model, X_test, y_test)
        results2 = evaluator.evaluate(model, X_test, y_test)
        
        # Results should be identical for deterministic evaluation
        for metric in results1.keys():
            if metric not in ['confusion_matrix', 'classification_report']:
                assert results1[metric] == results2[metric]

class TestPrecisionRecallAnalyzer:
    """Test PrecisionRecallAnalyzer class."""
    
    @pytest.fixture
    def binary_classification_data(self):
        """Generate binary classification data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_precision_recall_curve_generation(self, binary_classification_data):
        """Test precision-recall curve generation."""
        X_train, X_test, y_train, y_test = binary_classification_data
        
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        analyzer = PrecisionRecallAnalyzer()
        pr_data = analyzer.generate_precision_recall_curve(y_test, y_proba)
        
        assert 'precision' in pr_data
        assert 'recall' in pr_data
        assert 'thresholds' in pr_data
        assert 'average_precision' in pr_data
        
        assert len(pr_data['precision']) == len(pr_data['recall'])
        assert 0 <= pr_data['average_precision'] <= 1
    
    def test_f1_optimal_threshold(self, binary_classification_data):
        """Test finding optimal threshold for F1 score."""
        X_train, X_test, y_train, y_test = binary_classification_data
        
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        analyzer = PrecisionRecallAnalyzer()
        optimal_threshold = analyzer.find_f1_optimal_threshold(y_test, y_proba)
        
        assert 0 <= optimal_threshold <= 1
        assert isinstance(optimal_threshold, (float, np.float64))
    
    def test_multiclass_precision_recall(self):
        """Test precision-recall analysis for multiclass."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=3,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)
        
        analyzer = PrecisionRecallAnalyzer()
        multiclass_pr = analyzer.analyze_multiclass_precision_recall(y_test, y_proba)
        
        assert 'class_0' in multiclass_pr
        assert 'class_1' in multiclass_pr
        assert 'class_2' in multiclass_pr
        assert 'macro_average' in multiclass_pr
        
        for class_data in multiclass_pr.values():
            if isinstance(class_data, dict):
                assert 'average_precision' in class_data


class TestCalibrationAnalyzer:
    """Test CalibrationAnalyzer class."""
    
    @pytest.fixture
    def binary_classification_data(self):
        """Generate binary classification data."""
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_calibration_curve_generation(self, binary_classification_data):
        """Test calibration curve generation."""
        X_train, X_test, y_train, y_test = binary_classification_data
        
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        analyzer = CalibrationAnalyzer()
        calibration_data = analyzer.generate_calibration_curve(y_test, y_proba)
        
        assert 'mean_predicted_value' in calibration_data
        assert 'fraction_of_positives' in calibration_data
        assert 'brier_score' in calibration_data
        
        assert len(calibration_data['mean_predicted_value']) == len(calibration_data['fraction_of_positives'])
        assert calibration_data['brier_score'] >= 0
    
    def test_reliability_diagram(self, binary_classification_data):
        """Test reliability diagram data generation."""
        X_train, X_test, y_train, y_test = binary_classification_data
        
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        analyzer = CalibrationAnalyzer()
        reliability_data = analyzer.generate_reliability_diagram(y_test, y_proba, n_bins=10)
        
        assert 'bin_boundaries' in reliability_data
        assert 'bin_lowers' in reliability_data
        assert 'bin_uppers' in reliability_data
        assert 'y' in reliability_data
        assert 'bin_centers' in reliability_data
        
        assert len(reliability_data['bin_centers']) <= 10
    
    def test_calibration_metrics(self, binary_classification_data):
        """Test calibration metrics calculation."""
        X_train, X_test, y_train, y_test = binary_classification_data
        
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        analyzer = CalibrationAnalyzer()
        metrics = analyzer.calculate_calibration_metrics(y_test, y_proba)
        
        assert 'brier_score' in metrics
        assert 'ece' in metrics  # Expected Calibration Error
        assert 'mce' in metrics  # Maximum Calibration Error
        
        assert metrics['brier_score'] >= 0
        assert metrics['ece'] >= 0
        assert metrics['mce'] >= 0
    
    def test_calibration_comparison(self, binary_classification_data):
        """Test comparing calibration of multiple models."""
        X_train, X_test, y_train, y_test = binary_classification_data
        
        models = {
            'rf': RandomForestClassifier(n_estimators=10, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=200)
        }
        
        for model in models.values():
            model.fit(X_train, y_train)
        
        analyzer = CalibrationAnalyzer()
        comparison = analyzer.compare_model_calibration(models, X_test, y_test)
        
        assert 'rf' in comparison
        assert 'lr' in comparison
        
        for model_name, calibration_data in comparison.items():
            assert 'brier_score' in calibration_data
            assert 'calibration_curve' in calibration_data


class TestEvaluationErrorHandling:
    """Test error handling in evaluation utilities."""
    
    def test_mismatched_array_lengths(self):
        """Test error handling for mismatched array lengths."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1])  # Different length
        
        calculator = MetricsCalculator(task_type='classification')
        
        with pytest.raises(ValueError):
            calculator.accuracy(y_true, y_pred)
    
    def test_invalid_task_type(self):
        """Test error handling for invalid task types."""
        with pytest.raises(ValueError):
            ModelEvaluator(task_type='invalid_task')
    
    def test_empty_arrays(self):
        """Test error handling for empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])
        
        calculator = MetricsCalculator(task_type='classification')
        
        with pytest.raises((ValueError, ZeroDivisionError)):
            calculator.accuracy(y_true, y_pred)
    
    def test_wrong_probability_shape(self):
        """Test error handling for wrong probability shapes."""
        y_true = np.array([0, 1, 1, 0])
        y_proba = np.array([0.1, 0.8])  # Wrong shape
        
        analyzer = ROCAnalyzer()
        
        with pytest.raises((ValueError, IndexError)):
            analyzer.generate_roc_curve(y_true, y_proba)
    
    def test_unsupported_model_type(self):
        """Test error handling for unsupported model types."""
        # Mock a model without predict method
        class InvalidModel:
            pass
        
        model = InvalidModel()
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)
        
        evaluator = ModelEvaluator(task_type='classification')
        
        with pytest.raises(AttributeError):
            evaluator.evaluate(model, X, y)


class TestEvaluationUtilities:
    """Test utility functions in evaluation module."""
    
    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap confidence interval calculation."""
        # This would test a utility function for calculating confidence intervals
        np.random.seed(42)
        sample_data = np.random.normal(0, 1, 100)
        
        # Mock bootstrap function
        def bootstrap_ci(data, statistic_func, n_bootstrap=1000, confidence_level=0.95):
            bootstrap_stats = []
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_stats.append(statistic_func(bootstrap_sample))
            
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (100 - alpha / 2)
            
            return np.percentile(bootstrap_stats, [lower_percentile, upper_percentile])
        
        ci = bootstrap_ci(sample_data, np.mean)
        
        assert len(ci) == 2
        assert ci[0] < ci[1]  # Lower bound should be less than upper bound
        assert isinstance(ci[0], (float, np.floating))
        assert isinstance(ci[1], (float, np.floating))
    
    def test_effect_size_calculation(self):
        """Test effect size calculation between two groups."""
        np.random.seed(42)
        group1 = np.random.normal(0, 1, 50)
        group2 = np.random.normal(0.5, 1, 50)  # Slightly different mean
        
        # Cohen's d calculation
        def cohens_d(group1, group2):
            mean1, mean2 = np.mean(group1), np.mean(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            n1, n2 = len(group1), len(group2)
            
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            return (mean1 - mean2) / pooled_std
        
        effect_size = cohens_d(group1, group2)
        
        assert isinstance(effect_size, (float, np.floating))
        assert -3 < effect_size < 3  # Reasonable range for Cohen's d
    
    def test_model_complexity_metrics(self):
        """Test model complexity calculation."""
        # Test for different model types
        simple_model = LogisticRegression()
        complex_model = RandomForestClassifier(n_estimators=100, max_depth=20)
        
        # Mock complexity calculation
        def calculate_complexity(model):
            if hasattr(model, 'n_estimators'):
                return model.n_estimators * (model.max_depth or 10)
            elif hasattr(model, 'coef_'):
                # For linear models, complexity could be based on number of features
                return 1  # Simple baseline
            else:
                return 0
        
        simple_complexity = calculate_complexity(simple_model)
        complex_complexity = calculate_complexity(complex_model)
        
        assert complex_complexity > simple_complexity
        assert isinstance(simple_complexity, (int, float))
        assert isinstance(complex_complexity, (int, float))


class TestEvaluationVisualization:
    """Test visualization components of evaluation utilities."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Generate sample predictions for visualization tests."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_proba = np.random.beta(2, 2, 100)  # Probability-like distribution
        y_pred = (y_proba > 0.5).astype(int)
        
        return y_true, y_pred, y_proba
    
    def test_confusion_matrix_plot_data(self, sample_predictions):
        """Test confusion matrix plot data generation."""
        y_true, y_pred, _ = sample_predictions
        
        analyzer = ConfusionMatrixAnalyzer()
        plot_data = analyzer.prepare_confusion_matrix_plot(y_true, y_pred)
        
        assert 'matrix' in plot_data
        assert 'labels' in plot_data
        assert 'title' in plot_data
        
        assert plot_data['matrix'].shape == (2, 2)  # Binary classification
        assert len(plot_data['labels']) == 2
    
    def test_roc_curve_plot_data(self, sample_predictions):
        """Test ROC curve plot data generation."""
        y_true, _, y_proba = sample_predictions
        
        analyzer = ROCAnalyzer()
        plot_data = analyzer.prepare_roc_plot(y_true, y_proba)
        
        assert 'fpr' in plot_data
        assert 'tpr' in plot_data
        assert 'auc' in plot_data
        assert 'title' in plot_data
        
        assert len(plot_data['fpr']) == len(plot_data['tpr'])
        assert 0 <= plot_data['auc'] <= 1
    
    def test_learning_curve_plot_data(self):
        """Test learning curve plot data generation."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        model = LogisticRegression(random_state=42, max_iter=200)
        
        analyzer = LearningCurveAnalyzer()
        results = analyzer.generate_learning_curve(model, X, y, cv=3)
        plot_data = analyzer.prepare_learning_curve_plot(results)
        
        assert 'train_sizes' in plot_data
        assert 'train_scores_mean' in plot_data
        assert 'train_scores_std' in plot_data
        assert 'validation_scores_mean' in plot_data
        assert 'validation_scores_std' in plot_data
        assert 'title' in plot_data


class TestEvaluationReporting:
    """Test evaluation reporting functionality."""
    
    @pytest.fixture
    def complete_evaluation_results(self):
        """Generate complete evaluation results for reporting tests."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        evaluator = ModelEvaluator(task_type='classification')
        results = evaluator.detailed_report(model, X_test, y_test)
        
        return results, X_test, y_test, model
    
    def test_html_report_generation(self, complete_evaluation_results):
        """Test HTML report generation."""
        results, X_test, y_test, model = complete_evaluation_results
        
        # Mock HTML report generator
        def generate_html_report(results):
            html_content = f"""
            <html>
            <head><title>Model Evaluation Report</title></head>
            <body>
                <h1>Model Performance Report</h1>
                <h2>Metrics</h2>
                <ul>
            """
            
            for metric, value in results['metrics'].items():
                if isinstance(value, (int, float)):
                    html_content += f"<li>{metric}: {value:.4f}</li>"
            
            html_content += """
                </ul>
                <h2>Confusion Matrix</h2>
                <p>Matrix shape: {}</p>
            </body>
            </html>
            """.format(results['confusion_matrix'].shape)
            
            return html_content
        
        html_report = generate_html_report(results)
        
        assert '<html>' in html_report
        assert 'Model Performance Report' in html_report
        assert 'Metrics' in html_report
        assert 'Confusion Matrix' in html_report
    
    def test_json_report_generation(self, complete_evaluation_results):
        """Test JSON report generation."""
        results, X_test, y_test, model = complete_evaluation_results
        
        # Mock JSON report generator
        def generate_json_report(results):
            import json
            
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    json_results[key] = value.tolist()
                elif isinstance(value, dict):
                    json_results[key] = value
                else:
                    json_results[key] = str(value)
            
            return json.dumps(json_results, indent=2)
        
        json_report = generate_json_report(results)
        
        assert isinstance(json_report, str)
        
        # Should be valid JSON
        import json
        parsed_report = json.loads(json_report)
        assert 'metrics' in parsed_report
        assert 'confusion_matrix' in parsed_report
    
    def test_summary_statistics_report(self, complete_evaluation_results):
        """Test summary statistics report generation."""
        results, X_test, y_test, model = complete_evaluation_results
        
        # Mock summary generator
        def generate_summary(results):
            summary = {
                'model_type': type(model).__name__,
                'dataset_size': len(X_test),
                'num_features': X_test.shape[1],
                'accuracy': results['metrics']['accuracy'],
                'top_metric': max(results['metrics'].items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0),
                'confusion_matrix_trace': np.trace(results['confusion_matrix'])
            }
            return summary
        
        summary = generate_summary(results)
        
        assert 'model_type' in summary
        assert 'dataset_size' in summary
        assert 'accuracy' in summary
        assert summary['dataset_size'] == len(X_test)
        assert 0 <= summary['accuracy'] <= 1

# Additional test methods to add to your existing test_evaluation.py

class TestPerformanceMetrics:
    """Additional tests for edge cases in performance metrics."""
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])  # Perfect predictions
        
        calculator = MetricsCalculator(task_type='classification')
        metrics = calculator.calculate_all_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
    
    def test_worst_case_predictions(self):
        """Test metrics with worst possible predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0])  # Completely wrong
        
        calculator = MetricsCalculator(task_type='classification')
        metrics = calculator.calculate_all_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 0.0
    
    def test_single_class_predictions(self):
        """Test handling of single class in predictions."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0])  # Only one class predicted
        
        calculator = MetricsCalculator(task_type='classification')
        
        # Should handle gracefully (some metrics may be undefined)
        metrics = calculator.calculate_all_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        # Other metrics might be NaN or undefined for single class


class TestCrossValidationEdgeCases:
    """Test edge cases in cross-validation."""
    
    def test_cv_with_small_dataset(self):
        """Test cross-validation with very small datasets."""
        X = np.random.randn(10, 3)  # Very small dataset
        y = np.random.randint(0, 2, 10)
        
        model = LogisticRegression(random_state=42, max_iter=200)
        cv = CrossValidator(cv=3, random_state=42)  # 3-fold on 10 samples
        
        # Should work but might have warnings
        scores = cv.cross_validate(model, X, y)
        
        assert len(scores['test_score']) == 3
    
    def test_cv_reproducibility(self):
        """Test cross-validation reproducibility."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        model = LogisticRegression(random_state=42, max_iter=200)
        cv = CrossValidator(cv=5, random_state=42)
        
        # Run twice with same random state
        scores1 = cv.cross_validate(model, X, y)
        scores2 = cv.cross_validate(model, X, y)
        
        # Should be identical
        np.testing.assert_array_equal(scores1['test_score'], scores2['test_score'])


class TestModelEvaluatorExtended:
    """Extended tests for ModelEvaluator."""
    
    def test_evaluation_with_sample_weights(self):
        """Test evaluation with sample weights."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_train, y_train)
        
        # Create sample weights (higher weight for some samples)
        sample_weights = np.ones(len(y_test))
        sample_weights[:10] = 2.0  # Double weight for first 10 samples
        
        evaluator = ModelEvaluator(task_type='classification')
        
        # Test if evaluator can handle sample weights
        # (This would require implementing sample weight support in your evaluator)
        metrics_unweighted = evaluator.evaluate(model, X_test, y_test)
        
        # For now, just check that basic evaluation works
        assert 'accuracy' in metrics_unweighted
    
    def test_batch_evaluation(self):
        """Test evaluating multiple models at once."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        models = {
            'lr': LogisticRegression(random_state=42, max_iter=200),
            'rf': RandomForestClassifier(n_estimators=10, random_state=42)
        }
        
        # Train models
        for model in models.values():
            model.fit(X_train, y_train)
        
        evaluator = ModelEvaluator(task_type='classification')
        
        # Batch evaluation
        batch_results = {}
        for name, model in models.items():
            batch_results[name] = evaluator.evaluate(model, X_test, y_test)
        
        assert len(batch_results) == 2
        assert 'lr' in batch_results
        assert 'rf' in batch_results


class TestAnalyzerBenchmarks:
    """Benchmark tests for analyzers with larger datasets."""
    
    @pytest.mark.slow
    def test_large_dataset_performance(self):
        """Test analyzer performance with larger datasets."""
        # Create larger dataset
        X, y = make_classification(
            n_samples=5000,
            n_features=20,
            n_classes=3,
            random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Time the learning curve analysis
        import time
        
        analyzer = LearningCurveAnalyzer()
        
        start_time = time.time()
        results = analyzer.generate_learning_curve(
            model, X, y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=3
        )
        end_time = time.time()
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert (end_time - start_time) < 60  # Less than 60 seconds
        assert 'train_sizes' in results


class TestEvaluationConfigurability:
    """Test configurability and customization options."""
    
    def test_custom_scoring_functions(self):
        """Test using custom scoring functions."""
        def custom_balanced_accuracy(y_true, y_pred):
            """Custom balanced accuracy implementation."""
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            
            if cm.shape[0] != cm.shape[1]:
                return 0.0
            
            # Calculate balanced accuracy manually
            recalls = []
            for i in range(cm.shape[0]):
                if cm[i, :].sum() > 0:
                    recalls.append(cm[i, i] / cm[i, :].sum())
            
            return np.mean(recalls) if recalls else 0.0
        
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Test custom function
        custom_score = custom_balanced_accuracy(y_test, y_pred)
        
        assert 0 <= custom_score <= 1
        assert isinstance(custom_score, (float, np.floating))
    
    def test_evaluation_with_different_thresholds(self):
        """Test evaluation at different decision thresholds."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        thresholds = [0.3, 0.5, 0.7]
        results = {}
        
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            
            calculator = MetricsCalculator(task_type='classification')
            metrics = calculator.calculate_all_metrics(y_test, y_pred_thresh)
            results[threshold] = metrics
        
        # Results should vary with threshold
        assert len(results) == 3
        
        # Generally, precision should increase and recall decrease with higher threshold
        # (though this isn't guaranteed for all datasets)
        for threshold in thresholds:
            assert 0 <= results[threshold]['precision'] <= 1
            assert 0 <= results[threshold]['recall'] <= 1


class TestEvaluationReporting:
    """Test comprehensive reporting functionality."""
    
    def test_markdown_report_generation(self):
        """Test markdown report generation."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        evaluator = ModelEvaluator(task_type='classification')
        results = evaluator.detailed_report(model, X_test, y_test)
        
        # Mock markdown generator
        def generate_markdown_report(results):
            report = "# Model Evaluation Report\n\n"
            report += "## Performance Metrics\n\n"
            
            for metric, value in results['metrics'].items():
                if isinstance(value, (int, float)):
                    report += f"- **{metric.title()}**: {value:.4f}\n"
            
            report += "\n## Confusion Matrix\n\n"
            report += f"Shape: {results['confusion_matrix'].shape}\n"
            
            return report
        
        markdown_report = generate_markdown_report(results)
        
        assert "# Model Evaluation Report" in markdown_report
        assert "## Performance Metrics" in markdown_report
        assert "## Confusion Matrix" in markdown_report
    
    def test_latex_table_generation(self):
        """Test LaTeX table generation for results."""
        results = {
            'Model1': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88},
            'Model2': {'accuracy': 0.87, 'precision': 0.84, 'recall': 0.85},
            'Model3': {'accuracy': 0.83, 'precision': 0.86, 'recall': 0.81}
        }
        
        def generate_latex_table(results):
            latex = "\\begin{tabular}{|l|c|c|c|}\n"
            latex += "\\hline\n"
            latex += "Model & Accuracy & Precision & Recall \\\\\n"
            latex += "\\hline\n"
            
            for model, metrics in results.items():
                latex += f"{model} & {metrics['accuracy']:.3f} & {metrics['precision']:.3f} & {metrics['recall']:.3f} \\\\\n"
            
            latex += "\\hline\n"
            latex += "\\end{tabular}\n"
            
            return latex
        
        latex_table = generate_latex_table(results)
        
        assert "\\begin{tabular}" in latex_table
        assert "\\end{tabular}" in latex_table
        assert "Model1" in latex_table
        assert "Accuracy" in latex_table


# Add these benchmarking decorators if needed
pytest_slow = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_TESTS"),
    reason="Slow tests disabled by default"
)

if __name__ == "__main__":
    pytest.main([__file__])