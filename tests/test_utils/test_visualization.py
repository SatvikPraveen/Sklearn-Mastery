"""
Unit tests for visualization utilities.

Tests for data visualization and model visualization functions.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from evaluation import ModelVisualizationSuite


class TestDataVisualizer:
    """Test DataVisualizer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        X, y = make_classification(
            n_samples=200,
            n_features=4,
            n_classes=3,
            n_informative=3,
            random_state=42
        )
        feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
        return X, y, feature_names
    
    @pytest.fixture
    def regression_data(self):
        """Generate regression data for testing."""
        np.random.seed(42)
        X, y = make_regression(
            n_samples=200,
            n_features=3,
            noise=0.1,
            random_state=42
        )
        return X, y
    
    def test_plot_class_distribution(self, sample_data):
        """Test class distribution plotting."""
        X, y, feature_names = sample_data
        
        visualizer = DataVisualizer()
        fig, ax = plt.subplots()
        
        # Test basic plotting
        visualizer.plot_class_distribution(y, ax=ax)
        
        # Check that plot was created
        assert len(ax.patches) > 0  # Should have bars
        assert ax.get_xlabel() or ax.get_ylabel()  # Should have labels
        
        plt.close(fig)
    
    def test_plot_feature_distributions(self, sample_data):
        """Test feature distribution plotting."""
        X, y, feature_names = sample_data
        
        visualizer = DataVisualizer()
        fig = visualizer.plot_feature_distributions(X, feature_names=feature_names)
        
        # Should create subplots for each feature
        assert len(fig.axes) == len(feature_names)
        
        plt.close(fig)
    
    def test_plot_correlation_matrix(self, sample_data):
        """Test correlation matrix plotting."""
        X, y, feature_names = sample_data
        
        visualizer = DataVisualizer()
        fig, ax = plt.subplots(figsize=(8, 6))
        
        visualizer.plot_correlation_matrix(X, feature_names=feature_names, ax=ax)
        
        # Check that heatmap was created
        assert len(ax.collections) > 0  # Heatmap creates collections
        
        plt.close(fig)
    
    def test_plot_scatter_matrix(self, sample_data):
        """Test scatter matrix plotting."""
        X, y, feature_names = sample_data
        
        visualizer = DataVisualizer()
        
        # Test with subset of features to keep plot manageable
        X_subset = X[:, :3]
        feature_subset = feature_names[:3]
        
        fig = visualizer.plot_scatter_matrix(
            X_subset, y, feature_names=feature_subset
        )
        
        # Should create a matrix of subplots
        expected_subplots = len(feature_subset) ** 2
        assert len(fig.axes) == expected_subplots
        
        plt.close(fig)
    
    def test_plot_target_distribution(self, regression_data):
        """Test target distribution plotting for regression."""
        X, y = regression_data
        
        visualizer = DataVisualizer()
        fig, ax = plt.subplots()
        
        visualizer.plot_target_distribution(y, ax=ax)
        
        # Should create histogram
        assert len(ax.patches) > 0
        
        plt.close(fig)
    
    def test_plot_feature_target_relationship(self, sample_data):
        """Test feature-target relationship plotting."""
        X, y, feature_names = sample_data
        
        visualizer = DataVisualizer()
        fig, ax = plt.subplots()
        
        # Test with first feature
        visualizer.plot_feature_target_relationship(
            X[:, 0], y, feature_name=feature_names[0], ax=ax
        )
        
        # Should create scatter plot
        assert len(ax.collections) > 0
        
        plt.close(fig)
    
    def test_plot_missing_values_heatmap(self):
        """Test missing values heatmap."""
        # Create data with missing values
        np.random.seed(42)
        data = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [1, np.nan, 3, 4, np.nan],
            'C': [1, 2, 3, 4, 5]
        })
        
        visualizer = DataVisualizer()
        fig, ax = plt.subplots()
        
        visualizer.plot_missing_values_heatmap(data, ax=ax)
        
        # Should create heatmap
        assert len(ax.collections) > 0
        
        plt.close(fig)


class TestModelVisualizer:
    """Test ModelVisualizer class."""
    
    @pytest.fixture
    def trained_model_2d(self):
        """Train a model on 2D data for visualization."""
        X, y = make_classification(
            n_samples=200,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            random_state=42
        )
        
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        return model, X, y
    
    @pytest.fixture
    def trained_regression_model(self):
        """Train a regression model."""
        X, y = make_regression(
            n_samples=100,
            n_features=1,
            noise=0.1,
            random_state=42
        )
        
        model = LinearRegression()
        model.fit(X, y)
        
        return model, X, y
    
    def test_plot_decision_boundary(self, trained_model_2d):
        """Test decision boundary plotting."""
        model, X, y = trained_model_2d
        
        visualizer = ModelVisualizer()
        fig, ax = plt.subplots()
        
        visualizer.plot_decision_boundary(model, X, y, ax=ax)
        
        # Should create contour plot and scatter plot
        assert len(ax.collections) > 0  # Scatter plot
        
        plt.close(fig)
    
    def test_plot_regression_line(self, trained_regression_model):
        """Test regression line plotting."""
        model, X, y = trained_regression_model
        
        visualizer = ModelVisualizer()
        fig, ax = plt.subplots()
        
        visualizer.plot_regression_line(model, X, y, ax=ax)
        
        # Should have scatter plot and line
        assert len(ax.collections) > 0  # Scatter plot
        assert len(ax.lines) > 0  # Regression line
        
        plt.close(fig)
    
    def test_plot_feature_importance(self):
        """Test feature importance plotting."""
        # Create model with feature importance
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        visualizer = ModelVisualizer()
        fig, ax = plt.subplots()
        
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        visualizer.plot_feature_importance(
            model.feature_importances_, 
            feature_names=feature_names, 
            ax=ax
        )
        
        # Should create bar plot
        assert len(ax.patches) > 0
        
        plt.close(fig)
    
    def test_plot_residuals(self, trained_regression_model):
        """Test residuals plotting."""
        model, X, y = trained_regression_model
        
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        visualizer = ModelVisualizer()
        fig, ax = plt.subplots()
        
        visualizer.plot_residuals(y_pred, residuals, ax=ax)
        
        # Should create scatter plot
        assert len(ax.collections) > 0
        
        plt.close(fig)
    
    def test_plot_prediction_vs_actual(self, trained_regression_model):
        """Test prediction vs actual plotting."""
        model, X, y = trained_regression_model
        
        y_pred = model.predict(X)
        
        visualizer = ModelVisualizer()
        fig, ax = plt.subplots()
        
        visualizer.plot_prediction_vs_actual(y, y_pred, ax=ax)
        
        # Should have scatter plot and diagonal line
        assert len(ax.collections) > 0  # Scatter plot
        assert len(ax.lines) > 0  # Diagonal line
        
        plt.close(fig)


class TestPerformanceVisualizer:
    """Test PerformanceVisualizer class."""
    
    @pytest.fixture
    def classification_results(self):
        """Generate classification results for testing."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_proba = np.random.beta(2, 2, 100)
        y_pred = (y_proba > 0.5).astype(int)
        
        return y_true, y_pred, y_proba
    
    @pytest.fixture
    def learning_curve_data(self):
        """Generate learning curve data."""
        train_sizes = np.array([20, 40, 60, 80, 100])
        train_scores = np.random.uniform(0.7, 0.9, (5, 3))  # 5 sizes, 3 folds
        val_scores = np.random.uniform(0.6, 0.8, (5, 3))
        
        return {
            'train_sizes': train_sizes,
            'train_scores': train_scores,
            'validation_scores': val_scores
        }
    
    def test_plot_confusion_matrix(self, classification_results):
        """Test confusion matrix plotting."""
        y_true, y_pred, _ = classification_results
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        visualizer = PerformanceVisualizer()
        fig, ax = plt.subplots()
        
        visualizer.plot_confusion_matrix(cm, ax=ax)
        
        # Should create heatmap
        assert len(ax.collections) > 0
        
        plt.close(fig)
    
    def test_plot_roc_curve(self, classification_results):
        """Test ROC curve plotting."""
        y_true, _, y_proba = classification_results
        
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        visualizer = PerformanceVisualizer()
        fig, ax = plt.subplots()
        
        visualizer.plot_roc_curve(fpr, tpr, roc_auc, ax=ax)
        
        # Should have ROC curve and diagonal line
        assert len(ax.lines) >= 2
        
        plt.close(fig)
    
    def test_plot_precision_recall_curve(self, classification_results):
        """Test precision-recall curve plotting."""
        y_true, _, y_proba = classification_results
        
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        visualizer = PerformanceVisualizer()
        fig, ax = plt.subplots()
        
        visualizer.plot_precision_recall_curve(precision, recall, avg_precision, ax=ax)
        
        # Should create PR curve
        assert len(ax.lines) > 0
        
        plt.close(fig)
    
    def test_plot_learning_curves(self, learning_curve_data):
        """Test learning curves plotting."""
        visualizer = PerformanceVisualizer()
        fig, ax = plt.subplots()
        
        visualizer.plot_learning_curves(learning_curve_data, ax=ax)
        
        # Should have training and validation curves
        assert len(ax.lines) >= 2
        
        plt.close(fig)
    
    def test_plot_validation_curve(self):
        """Test validation curve plotting."""
        param_range = [1, 5, 10, 20, 50]
        train_scores = np.random.uniform(0.7, 0.9, (5, 3))
        val_scores = np.random.uniform(0.6, 0.8, (5, 3))
        
        visualizer = PerformanceVisualizer()
        fig, ax = plt.subplots()
        
        visualizer.plot_validation_curve(
            param_range, train_scores, val_scores, 
            param_name='n_estimators', ax=ax
        )
        
        # Should have training and validation curves
        assert len(ax.lines) >= 2
        
        plt.close(fig)
    
    def test_plot_calibration_curve(self, classification_results):
        """Test calibration curve plotting."""
        y_true, _, y_proba = classification_results
        
        from sklearn.calibration import calibration_curve
        fraction_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10)
        
        visualizer = PerformanceVisualizer()
        fig, ax = plt.subplots()
        
        visualizer.plot_calibration_curve(fraction_pos, mean_pred, ax=ax)
        
        # Should have calibration curve and diagonal line
        assert len(ax.lines) >= 2
        
        plt.close(fig)


class TestFeatureVisualizer:
    """Test FeatureVisualizer class."""
    
    @pytest.fixture
    def feature_data(self):
        """Generate feature data for testing."""
        np.random.seed(42)
        X, y = make_classification(
            n_samples=150,
            n_features=6,
            n_informative=4,
            random_state=42
        )
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        return X, y, feature_names
    
    def test_plot_feature_importance_ranking(self, feature_data):
        """Test feature importance ranking plot."""
        X, y, feature_names = feature_data
        
        # Create mock importance scores
        importance_scores = np.random.uniform(0, 1, len(feature_names))
        
        visualizer = FeatureVisualizer()
        fig, ax = plt.subplots()
        
        visualizer.plot_feature_importance_ranking(
            importance_scores, feature_names, ax=ax
        )
        
        # Should create horizontal bar plot
        assert len(ax.patches) > 0
        
        plt.close(fig)
    
    def test_plot_feature_correlation_with_target(self, feature_data):
        """Test feature-target correlation plotting."""
        X, y, feature_names = feature_data
        
        # Calculate correlations
        correlations = [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
        
        visualizer = FeatureVisualizer()
        fig, ax = plt.subplots()
        
        visualizer.plot_feature_correlation_with_target(
            correlations, feature_names, ax=ax
        )
        
        # Should create bar plot
        assert len(ax.patches) > 0
        
        plt.close(fig)
    
    def test_plot_pairwise_relationships(self, feature_data):
        """Test pairwise feature relationships."""
        X, y, feature_names = feature_data
        
        visualizer = FeatureVisualizer()
        
        # Test with subset to keep manageable
        X_subset = X[:, :3]
        feature_subset = feature_names[:3]
        
        fig = visualizer.plot_pairwise_relationships(X_subset, feature_subset)
        
        # Should create matrix of subplots
        assert len(fig.axes) > 0
        
        plt.close(fig)
    
    def test_plot_feature_distributions_by_class(self, feature_data):
        """Test feature distributions by class."""
        X, y, feature_names = feature_data
        
        visualizer = FeatureVisualizer()
        fig = visualizer.plot_feature_distributions_by_class(
            X, y, feature_names[:3]  # Test with subset
        )
        
        # Should create subplots for selected features
        assert len(fig.axes) == 3
        
        plt.close(fig)


class TestComparisonVisualizer:
    """Test ComparisonVisualizer class."""
    
    @pytest.fixture
    def model_comparison_data(self):
        """Generate model comparison data."""
        models = ['LogisticRegression', 'RandomForest', 'SVM']
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        # Random performance data
        np.random.seed(42)
        data = {}
        for model in models:
            data[model] = {metric: np.random.uniform(0.7, 0.95) for metric in metrics}
        
        return data
    
    def test_plot_model_comparison_bar(self, model_comparison_data):
        """Test model comparison bar plot."""
        visualizer = ComparisonVisualizer()
        fig, ax = plt.subplots()
        
        visualizer.plot_model_comparison_bar(
            model_comparison_data, metric='accuracy', ax=ax
        )
        
        # Should create bar plot
        assert len(ax.patches) > 0
        
        plt.close(fig)
    
    def test_plot_model_comparison_radar(self, model_comparison_data):
        """Test model comparison radar chart."""
        visualizer = ComparisonVisualizer()
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        
        # Test with subset of models
        subset_data = {k: v for k, v in list(model_comparison_data.items())[:2]}
        
        visualizer.plot_model_comparison_radar(subset_data, ax=ax)
        
        # Should create radar plot
        assert len(ax.lines) > 0
        
        plt.close(fig)
    
    def test_plot_performance_heatmap(self, model_comparison_data):
        """Test performance heatmap."""
        visualizer = ComparisonVisualizer()
        fig, ax = plt.subplots()
        
        visualizer.plot_performance_heatmap(model_comparison_data, ax=ax)
        
        # Should create heatmap
        assert len(ax.collections) > 0
        
        plt.close(fig)
    
    def test_plot_metric_distribution(self):
        """Test metric distribution plotting."""
        # Create distribution data
        np.random.seed(42)
        cv_scores = {
            'Model1': np.random.normal(0.85, 0.05, 10),
            'Model2': np.random.normal(0.80, 0.08, 10),
            'Model3': np.random.normal(0.88, 0.03, 10)
        }
        
        visualizer = ComparisonVisualizer()
        fig, ax = plt.subplots()
        
        visualizer.plot_metric_distribution(cv_scores, ax=ax)
        
        # Should create box plot or violin plot
        assert len(ax.collections) > 0 or len(ax.patches) > 0
        
        plt.close(fig)


class TestInteractiveVisualizer:
    """Test InteractiveVisualizer class."""
    
    @pytest.fixture
    def sample_interactive_data(self):
        """Generate sample data for interactive visualization testing."""
        np.random.seed(42)
        X, y = make_classification(
            n_samples=200,
            n_features=4,
            n_classes=3,
            n_informative=3,
            random_state=42
        )
        feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
        return X, y, feature_names
    
    def test_interactive_scatter_plot(self, sample_interactive_data):
        """Test interactive scatter plot creation."""
        X, y, feature_names = sample_interactive_data
        
        visualizer = InteractiveVisualizer()
        
        # Test creating interactive scatter plot
        fig = visualizer.create_interactive_scatter(
            X[:, 0], X[:, 1], y,
            x_label=feature_names[0],
            y_label=feature_names[1],
            color_label='class'
        )
        
        # Check that figure object is created
        assert fig is not None
        
        # If using plotly, check for plotly figure
        try:
            import plotly.graph_objects as go
            assert isinstance(fig, (go.Figure, dict))
        except ImportError:
            # If plotly not available, should return matplotlib figure or None
            assert fig is None or hasattr(fig, 'show')
    
    def test_interactive_feature_explorer(self, sample_interactive_data):
        """Test interactive feature exploration."""
        X, y, feature_names = sample_interactive_data
        
        visualizer = InteractiveVisualizer()
        
        # Test creating interactive feature explorer
        dashboard = visualizer.create_feature_explorer(
            X, y, feature_names=feature_names
        )
        
        # Should return some form of dashboard/widget
        assert dashboard is not None
    
    def test_interactive_model_comparison(self):
        """Test interactive model comparison dashboard."""
        # Create mock model comparison data
        models_data = {
            'LogisticRegression': {
                'accuracy': [0.85, 0.87, 0.83, 0.86, 0.84],
                'precision': [0.82, 0.84, 0.80, 0.83, 0.81],
                'recall': [0.88, 0.90, 0.86, 0.89, 0.87]
            },
            'RandomForest': {
                'accuracy': [0.89, 0.91, 0.87, 0.90, 0.88],
                'precision': [0.86, 0.88, 0.84, 0.87, 0.85],
                'recall': [0.92, 0.94, 0.90, 0.93, 0.91]
            },
            'SVM': {
                'accuracy': [0.83, 0.85, 0.81, 0.84, 0.82],
                'precision': [0.80, 0.82, 0.78, 0.81, 0.79],
                'recall': [0.86, 0.88, 0.84, 0.87, 0.85]
            }
        }
        
        visualizer = InteractiveVisualizer()
        
        # Test creating interactive comparison dashboard
        dashboard = visualizer.create_model_comparison_dashboard(models_data)
        
        # Should return dashboard object
        assert dashboard is not None
    
    def test_interactive_learning_curves(self):
        """Test interactive learning curves visualization."""
        # Mock learning curve data
        train_sizes = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
        
        models_curves = {
            'LogisticRegression': {
                'train_scores_mean': np.array([0.75, 0.78, 0.80, 0.82, 0.83, 0.84, 0.85, 0.85, 0.86, 0.86]),
                'train_scores_std': np.array([0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
                'val_scores_mean': np.array([0.70, 0.73, 0.75, 0.77, 0.78, 0.79, 0.80, 0.80, 0.80, 0.80]),
                'val_scores_std': np.array([0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])
            },
            'RandomForest': {
                'train_scores_mean': np.array([0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.95, 0.96, 0.96]),
                'train_scores_std': np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
                'val_scores_mean': np.array([0.78, 0.81, 0.83, 0.85, 0.86, 0.87, 0.87, 0.87, 0.87, 0.87]),
                'val_scores_std': np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])
            }
        }
        
        visualizer = InteractiveVisualizer()
        
        # Test creating interactive learning curves
        fig = visualizer.create_interactive_learning_curves(
            train_sizes, models_curves
        )
        
        # Should return figure object
        assert fig is not None
    
    def test_interactive_feature_importance(self):
        """Test interactive feature importance visualization."""
        # Mock feature importance data
        feature_names = [f'Feature_{i}' for i in range(10)]
        importance_data = {
            'RandomForest': np.random.uniform(0.01, 0.25, 10),
            'GradientBoosting': np.random.uniform(0.01, 0.25, 10),
            'ExtraTrees': np.random.uniform(0.01, 0.25, 10)
        }
        
        visualizer = InteractiveVisualizer()
        
        # Test creating interactive feature importance plot
        fig = visualizer.create_interactive_feature_importance(
            importance_data, feature_names
        )
        
        # Should return figure object
        assert fig is not None
    
    def test_interactive_confusion_matrix(self):
        """Test interactive confusion matrix visualization."""
        # Mock confusion matrix data
        cm_data = {
            'LogisticRegression': np.array([[45, 2, 1], [3, 38, 2], [1, 2, 46]]),
            'RandomForest': np.array([[47, 1, 0], [2, 39, 1], [0, 1, 48]]),
            'SVM': np.array([[44, 3, 1], [4, 37, 1], [2, 1, 47]])
        }
        
        class_names = ['Class_0', 'Class_1', 'Class_2']
        
        visualizer = InteractiveVisualizer()
        
        # Test creating interactive confusion matrix
        fig = visualizer.create_interactive_confusion_matrix(
            cm_data, class_names
        )
        
        # Should return figure object
        assert fig is not None
    
    def test_interactive_roc_curves(self):
        """Test interactive ROC curves visualization."""
        # Mock ROC curve data
        roc_data = {
            'LogisticRegression': {
                'fpr': np.linspace(0, 1, 50),
                'tpr': np.sort(np.random.uniform(0, 1, 50)),
                'auc': 0.85
            },
            'RandomForest': {
                'fpr': np.linspace(0, 1, 50),
                'tpr': np.sort(np.random.uniform(0, 1, 50)),
                'auc': 0.89
            },
            'SVM': {
                'fpr': np.linspace(0, 1, 50),
                'tpr': np.sort(np.random.uniform(0, 1, 50)),
                'auc': 0.83
            }
        }
        
        visualizer = InteractiveVisualizer()
        
        # Test creating interactive ROC curves
        fig = visualizer.create_interactive_roc_curves(roc_data)
        
        # Should return figure object
        assert fig is not None
    
    def test_interactive_hyperparameter_tuning(self):
        """Test interactive hyperparameter tuning visualization."""
        # Mock hyperparameter tuning results
        param_grid = {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [3, 5, 7, 10, None]
        }
        
        # Create grid of results
        results = []
        for n_est in param_grid['n_estimators']:
            for max_d in param_grid['max_depth']:
                score = np.random.uniform(0.7, 0.95)
                results.append({
                    'n_estimators': n_est,
                    'max_depth': max_d if max_d else 'None',
                    'score': score
                })
        
        visualizer = InteractiveVisualizer()
        
        # Test creating interactive hyperparameter visualization
        fig = visualizer.create_interactive_hyperparameter_heatmap(
            results, 'n_estimators', 'max_depth', 'score'
        )
        
        # Should return figure object
        assert fig is not None
    
    @pytest.mark.skipif(
        not pytest.importorskip("plotly", minversion="5.0"),
        reason="Plotly not available or version too old"
    )
    def test_plotly_integration(self, sample_interactive_data):
        """Test plotly integration if available."""
        import plotly.graph_objects as go
        
        X, y, feature_names = sample_interactive_data
        
        visualizer = InteractiveVisualizer()
        
        # Test that plotly figures are created properly
        fig = visualizer.create_interactive_scatter(
            X[:, 0], X[:, 1], y,
            x_label=feature_names[0],
            y_label=feature_names[1]
        )
        
        # Should be a plotly figure
        assert isinstance(fig, go.Figure)
        
        # Check basic properties
        assert len(fig.data) > 0  # Should have traces
        assert fig.layout.xaxis.title.text == feature_names[0]
        assert fig.layout.yaxis.title.text == feature_names[1]
    
    def test_fallback_to_matplotlib(self, sample_interactive_data):
        """Test fallback to matplotlib when plotly is not available."""
        X, y, feature_names = sample_interactive_data
        
        visualizer = InteractiveVisualizer()
        
        # Mock plotly unavailable
        original_import = __builtins__.__import__
        
        def mock_import(name, *args, **kwargs):
            if name.startswith('plotly'):
                raise ImportError("Plotly not available")
            return original_import(name, *args, **kwargs)
        
        __builtins__.__import__ = mock_import
        
        try:
            # Should fallback to matplotlib or return None
            fig = visualizer.create_interactive_scatter(
                X[:, 0], X[:, 1], y,
                x_label=feature_names[0],
                y_label=feature_names[1]
            )
            
            # Should handle gracefully
            assert fig is None or hasattr(fig, 'show')
            
        finally:
            __builtins__.__import__ = original_import
    
    def test_widget_integration(self):
        """Test integration with Jupyter widgets if available."""
        try:
            import ipywidgets as widgets
            
            visualizer = InteractiveVisualizer()
            
            # Test creating widget-based interface
            widget_interface = visualizer.create_model_comparison_widgets()
            
            # Should return widget container
            assert widget_interface is not None
            
        except ImportError:
            # Skip if ipywidgets not available
            pytest.skip("ipywidgets not available")
    
    def test_export_interactive_html(self, sample_interactive_data):
        """Test exporting interactive visualizations to HTML."""
        X, y, feature_names = sample_interactive_data
        
        visualizer = InteractiveVisualizer()
        
        # Create interactive plot
        fig = visualizer.create_interactive_scatter(
            X[:, 0], X[:, 1], y,
            x_label=feature_names[0],
            y_label=feature_names[1]
        )
        
        if fig is not None:
            # Test HTML export functionality
            html_content = visualizer.export_to_html(fig, include_plotlyjs=True)
            
            # Should return HTML string
            assert isinstance(html_content, str)
            assert '<html>' in html_content or '<div>' in html_content
    
    def test_interactive_data_table(self):
        """Test interactive data table creation."""
        # Mock model comparison data
        data = pd.DataFrame({
            'Model': ['LogisticRegression', 'RandomForest', 'SVM', 'GradientBoosting'],
            'Accuracy': [0.85, 0.89, 0.83, 0.87],
            'Precision': [0.82, 0.86, 0.80, 0.84],
            'Recall': [0.88, 0.92, 0.86, 0.90],
            'F1-Score': [0.85, 0.89, 0.83, 0.87],
            'Training_Time': [0.12, 1.45, 0.89, 2.34]
        })
        
        visualizer = InteractiveVisualizer()
        
        # Test creating interactive data table
        table = visualizer.create_interactive_table(
            data, 
            sortable=True, 
            filterable=True,
            highlight_best=['Accuracy', 'Precision', 'Recall', 'F1-Score']
        )
        
        # Should return table object
        assert table is not None


class TestVisualizationUtilities:
    """Test visualization utility functions."""
    
    def test_save_figure(self):
        """Test figure saving functionality."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        
        # Mock save function
        def save_figure(fig, filename, dpi=300, bbox_inches='tight'):
            # In real implementation, this would save the figure
            # For testing, just check that function can be called
            assert isinstance(fig, plt.Figure)
            assert isinstance(filename, str)
            assert isinstance(dpi, int)
            return True
        
        result = save_figure(fig, 'test_plot.png')
        assert result is True
        
        plt.close(fig)
    
    def test_setup_plot_style(self):
        """Test plot style setup."""
        # Mock style setup function
        def setup_plot_style(style='seaborn', figsize=(10, 6), dpi=100):
            plt.rcParams['figure.figsize'] = figsize
            plt.rcParams['figure.dpi'] = dpi
            return plt.rcParams
        
        original_figsize = plt.rcParams['figure.figsize']
        
        params = setup_plot_style(figsize=(12, 8))
        
        assert params['figure.figsize'] == (12, 8)
        
        # Reset to original
        plt.rcParams['figure.figsize'] = original_figsize
    
    def test_create_subplot_grid(self):
        """Test subplot grid creation."""
        def create_subplot_grid(n_plots, max_cols=3):
            n_rows = (n_plots - 1) // max_cols + 1
            n_cols = min(n_plots, max_cols)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
            
            if n_plots == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = list(axes)
            else:
                axes = axes.flatten()
            
            return fig, axes
        
        # Test with different numbers of plots
        for n_plots in [1, 3, 5, 8]:
            fig, axes = create_subplot_grid(n_plots)
            
            assert isinstance(fig, plt.Figure)
            assert len(axes) >= n_plots
            
            plt.close(fig)


class TestVisualizationErrorHandling:
    """Test error handling in visualization utilities."""
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        visualizer = DataVisualizer()
        
        # Empty arrays should be handled gracefully
        with pytest.raises((ValueError, IndexError)):
            fig, ax = plt.subplots()
            visualizer.plot_class_distribution(np.array([]), ax=ax)
            plt.close(fig)
    
    def test_mismatched_dimensions(self):
        """Test handling of mismatched data dimensions."""
        visualizer = ModelVisualizer()
        
        # Test with wrong dimensions for decision boundary
        X = np.random.randn(100, 3)  # 3D data
        y = np.random.randint(0, 2, 100)
        
        model = LogisticRegression()
        model.fit(X, y)
        
        fig, ax = plt.subplots()
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, NotImplementedError)):
            visualizer.plot_decision_boundary(model, X, y, ax=ax)
        
        plt.close(fig)
    
    def test_invalid_model_type(self):
        """Test handling of invalid model types."""
        visualizer = ModelVisualizer()
        
        # Mock invalid model
        class InvalidModel:
            pass
        
        model = InvalidModel()
        X = np.random.randn(50, 2)
        y = np.random.randint(0, 2, 50)
        
        fig, ax = plt.subplots()
        
        with pytest.raises(AttributeError):
            visualizer.plot_decision_boundary(model, X, y, ax=ax)
        
        plt.close(fig)
    
    def test_missing_feature_names(self):
        """Test handling when feature names are missing."""
        visualizer = DataVisualizer()
        
        X = np.random.randn(100, 4)
        
        # Should work without feature names (generate defaults)
        fig = visualizer.plot_feature_distributions(X)
        
        assert len(fig.axes) == X.shape[1]
        
        plt.close(fig)


class TestVisualizationPerformance:
    """Test visualization performance with large datasets."""
    
    def test_large_dataset_handling(self):
        """Test visualization with large datasets."""
        # Create moderately large dataset
        np.random.seed(42)
        X = np.random.randn(5000, 10)
        y = np.random.randint(0, 3, 5000)
        
        visualizer = DataVisualizer()
        
        # Should handle reasonably sized datasets
        fig, ax = plt.subplots()
        visualizer.plot_class_distribution(y, ax=ax)
        
        assert len(ax.patches) > 0
        
        plt.close(fig)
    
    def test_memory_efficient_plotting(self):
        """Test memory-efficient plotting strategies."""
        # Test sampling for very large datasets
        def sample_for_plotting(X, y, max_samples=1000):
            if len(X) > max_samples:
                indices = np.random.choice(len(X), max_samples, replace=False)
                return X[indices], y[indices]
            return X, y
        
        # Large dataset
        X = np.random.randn(10000, 5)
        y = np.random.randint(0, 2, 10000)
        
        X_sampled, y_sampled = sample_for_plotting(X, y)
        
        assert len(X_sampled) <= 1000
        assert len(y_sampled) <= 1000
        assert len(X_sampled) == len(y_sampled)


class TestVisualizationIntegration:
    """Integration tests for visualization components."""
    
    @pytest.fixture
    def complete_ml_pipeline(self):
        """Create complete ML pipeline for integration testing."""
        # Generate data
        X, y = make_classification(
            n_samples=300,
            n_features=8,
            n_classes=2,
            random_state=42
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'model': model,
            'feature_names': [f'feature_{i}' for i in range(X.shape[1])]
        }
    
    def test_complete_visualization_workflow(self, complete_ml_pipeline):
        """Test complete visualization workflow."""
        data = complete_ml_pipeline
        
        # Data visualization
        data_viz = DataVisualizer()
        fig1 = data_viz.plot_feature_distributions(
            data['X_train'], 
            feature_names=data['feature_names'][:4]  # Subset for testing
        )
        assert len(fig1.axes) == 4
        plt.close(fig1)
        
        # Model visualization
        model_viz = ModelVisualizer()
        fig2, ax2 = plt.subplots()
        model_viz.plot_feature_importance(
            data['model'].feature_importances_,
            feature_names=data['feature_names'],
            ax=ax2
        )
        assert len(ax2.patches) > 0
        plt.close(fig2)
        
        # Performance visualization
        perf_viz = PerformanceVisualizer()
        y_pred = data['model'].predict(data['X_test'])
        y_proba = data['model'].predict_proba(data['X_test'])[:, 1]
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(data['y_test'], y_pred)
        
        fig3, ax3 = plt.subplots()
        perf_viz.plot_confusion_matrix(cm, ax=ax3)
        assert len(ax3.collections) > 0
        plt.close(fig3)
    
    def test_visualization_consistency(self, complete_ml_pipeline):
        """Test consistency of visualizations across multiple calls."""
        data = complete_ml_pipeline
        
        visualizer = PerformanceVisualizer()
        
        # Generate same plot twice
        y_pred = data['model'].predict(data['X_test'])
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(data['y_test'], y_pred)
        
        fig1, ax1 = plt.subplots()
        visualizer.plot_confusion_matrix(cm, ax=ax1)
        
        fig2, ax2 = plt.subplots()
        visualizer.plot_confusion_matrix(cm, ax=ax2)
        
        # Should produce consistent results
        assert len(ax1.collections) == len(ax2.collections)
        
        plt.close(fig1)
        plt.close(fig2)


class TestInteractiveVisualizationAdvanced:
    """Advanced tests for interactive visualization features."""
    
    def test_interactive_3d_scatter(self):
        """Test 3D interactive scatter plot."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randint(0, 3, 100)
        
        visualizer = InteractiveVisualizer()
        
        # Test 3D scatter plot
        fig = visualizer.create_3d_scatter(
            X[:, 0], X[:, 1], X[:, 2], y,
            x_label='X', y_label='Y', z_label='Z'
        )
        
        assert fig is not None
    
    def test_animated_learning_curves(self):
        """Test animated learning curves."""
        # Mock time-series learning data
        epochs = np.arange(1, 51)
        train_acc = 0.5 + 0.4 * (1 - np.exp(-epochs/10)) + np.random.normal(0, 0.02, 50)
        val_acc = 0.5 + 0.35 * (1 - np.exp(-epochs/10)) + np.random.normal(0, 0.03, 50)
        
        visualizer = InteractiveVisualizer()
        
        # Test animated learning curves
        fig = visualizer.create_animated_learning_curves(
            epochs, train_acc, val_acc
        )
        
        assert fig is not None
    
    def test_interactive_correlation_network(self):
        """Test interactive correlation network visualization."""
        # Mock correlation data
        features = [f'feature_{i}' for i in range(8)]
        corr_matrix = np.random.uniform(-0.8, 0.8, (8, 8))
        np.fill_diagonal(corr_matrix, 1.0)
        
        visualizer = InteractiveVisualizer()
        
        # Test correlation network
        fig = visualizer.create_correlation_network(
            corr_matrix, features, threshold=0.5
        )
        
        assert fig is not None
    
    def test_interactive_decision_tree(self):
        """Test interactive decision tree visualization."""
        from sklearn.tree import DecisionTreeClassifier
        
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X, y)
        
        visualizer = InteractiveVisualizer()
        
        # Test interactive decision tree
        fig = visualizer.create_interactive_decision_tree(
            model, feature_names=[f'feature_{i}' for i in range(4)]
        )
        
        assert fig is not None
    
    def test_real_time_model_monitoring(self):
        """Test real-time model monitoring dashboard."""
        # Mock streaming performance data
        timestamps = pd.date_range('2024-01-01', periods=100, freq='H')
        performance_data = {
            'timestamp': timestamps,
            'accuracy': np.random.uniform(0.8, 0.95, 100),
            'latency_ms': np.random.uniform(10, 50, 100),
            'throughput': np.random.uniform(100, 500, 100)
        }
        
        visualizer = InteractiveVisualizer()
        
        # Test real-time monitoring dashboard
        dashboard = visualizer.create_monitoring_dashboard(performance_data)
        
        assert dashboard is not None


class TestInteractiveVisualizationIntegration:
    """Integration tests for interactive visualization with ML pipeline."""
    
    @pytest.fixture
    def ml_pipeline_results(self):
        """Create complete ML pipeline results for interactive testing."""
        X, y = make_classification(n_samples=500, n_features=10, n_classes=3, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train multiple models
        models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=200),
            'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42),
            'SVM': SVMClassifier(random_state=42) if 'SVMClassifier' in globals() else None
        }
        
        results = {}
        for name, model in models.items():
            if model is not None:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'probabilities': y_proba,
                    'test_score': model.score(X_test, y_test)
                }
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'models': results,
            'feature_names': [f'feature_{i}' for i in range(X.shape[1])]
        }
    
    def test_comprehensive_interactive_report(self, ml_pipeline_results):
        """Test comprehensive interactive ML report."""
        data = ml_pipeline_results
        
        visualizer = InteractiveVisualizer()
        
        # Create comprehensive interactive report
        report = visualizer.create_comprehensive_report(
            data['models'],
            data['X_test'],
            data['y_test'],
            feature_names=data['feature_names']
        )
        
        assert report is not None
        # Report should contain multiple interactive components
        assert hasattr(report, 'children') or isinstance(report, dict)
    
    def test_interactive_model_explainer(self, ml_pipeline_results):
        """Test interactive model explainer dashboard."""
        data = ml_pipeline_results
        
        # Get best performing model
        best_model_name = max(data['models'].keys(), 
                             key=lambda k: data['models'][k]['test_score'])
        best_model = data['models'][best_model_name]['model']
        
        visualizer = InteractiveVisualizer()
        
        # Create model explainer
        explainer = visualizer.create_model_explainer(
            best_model,
            data['X_test'],
            data['y_test'],
            feature_names=data['feature_names']
        )
        
        assert explainer is not None
    
    def test_interactive_feature_selection(self, ml_pipeline_results):
        """Test interactive feature selection tool."""
        data = ml_pipeline_results
        
        visualizer = InteractiveVisualizer()
        
        # Create feature selection tool
        feature_tool = visualizer.create_feature_selection_tool(
            data['X_train'],
            data['y_train'],
            feature_names=data['feature_names']
        )
        
        assert feature_tool is not None


class TestInteractiveVisualizationCustomization:
    """Test customization options for interactive visualizations."""
    
    def test_custom_color_schemes(self):
        """Test custom color scheme application."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = np.random.randint(0, 3, 100)
        
        visualizer = InteractiveVisualizer()
        
        # Test with custom color scheme
        custom_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        fig = visualizer.create_interactive_scatter(
            X[:, 0], X[:, 1], y,
            color_scheme=custom_colors
        )
        
        assert fig is not None
    
    def test_custom_styling(self):
        """Test custom styling options."""
        visualizer = InteractiveVisualizer()
        
        # Test applying custom theme
        theme_config = {
            'background_color': '#F8F9FA',
            'grid_color': '#E9ECEF',
            'text_color': '#343A40',
            'font_family': 'Arial, sans-serif'
        }
        
        styled_visualizer = visualizer.apply_custom_theme(theme_config)
        
        assert styled_visualizer is not None
    
    def test_responsive_layouts(self):
        """Test responsive layout configurations."""
        visualizer = InteractiveVisualizer()
        
        # Test different layout configurations
        layouts = ['mobile', 'tablet', 'desktop', 'presentation']
        
        for layout in layouts:
            config = visualizer.get_responsive_config(layout)
            assert config is not None
            assert 'width' in config or 'height' in config


if __name__ == "__main__":
    # Configure matplotlib for testing
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    pytest.main([__file__])