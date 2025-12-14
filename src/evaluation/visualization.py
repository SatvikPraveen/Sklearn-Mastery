"""Advanced visualization tools for model evaluation and data analysis."""

import sys
from pathlib import Path

# Handle imports that work in both package and direct import contexts
try:
    from ..config.settings import settings
    from ..config.logging_config import LoggerMixin
    from .utils import ensure_binary_classification, safe_division
except ImportError:
    # Fallback for direct imports outside package context
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.settings import settings
    from config.logging_config import LoggerMixin
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from evaluation.utils import ensure_binary_classification, safe_division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional, Union, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


class ModelVisualizationSuite(LoggerMixin):
    """Comprehensive visualization suite for model evaluation and analysis."""
    
    def __init__(self, style: str = None, figure_size: Tuple[int, int] = None):
        """Initialize visualization suite.
        
        Args:
            style: Matplotlib style to use.
            figure_size: Default figure size.
        """
        self.style = style or settings.STYLE
        self.figure_size = figure_size or settings.FIGURE_SIZE
        
        # Set matplotlib style
        try:
            plt.style.use(self.style)
        except:
            plt.style.use('default')
            
        # Set color palette
        self.colors = sns.color_palette(settings.COLOR_PALETTE, n_colors=10)
        sns.set_palette(self.colors)
        
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        normalize: bool = True,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot confusion matrix with annotations.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            class_names: Names of classes.
            normalize: Whether to normalize the matrix.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        self.logger.info("Creating confusion matrix plot")
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Plot heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt, 
            cmap='Blues',
            xticklabels=class_names or range(cm.shape[1]),
            yticklabels=class_names or range(cm.shape[0]),
            ax=ax
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=settings.DPI, bbox_inches='tight')
            
        return fig
    
    def plot_roc_curves(
        self,
        roc_data: Dict[str, Dict[str, np.ndarray]],
        title: str = "ROC Curves Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot ROC curves for multiple models.
        
        Args:
            roc_data: Dictionary with model names as keys and ROC curve data as values.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        self.logger.info("Creating ROC curves plot")
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        for i, (model_name, data) in enumerate(roc_data.items()):
            fpr = data['fpr']
            tpr = data['tpr']
            auc_score = np.trapz(tpr, fpr)
            
            ax.plot(
                fpr, tpr,
                color=self.colors[i % len(self.colors)],
                label=f'{model_name} (AUC = {auc_score:.3f})',
                linewidth=2
            )
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=settings.DPI, bbox_inches='tight')
            
        return fig
    
    def plot_precision_recall_curves(
        self,
        pr_data: Dict[str, Dict[str, np.ndarray]],
        title: str = "Precision-Recall Curves",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot precision-recall curves for multiple models.
        
        Args:
            pr_data: Dictionary with model names as keys and PR curve data as values.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        self.logger.info("Creating precision-recall curves plot")
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        for i, (model_name, data) in enumerate(pr_data.items()):
            precision = data['precision']
            recall = data['recall']
            auc_score = np.trapz(precision, recall)
            
            ax.plot(
                recall, precision,
                color=self.colors[i % len(self.colors)],
                label=f'{model_name} (AUC = {auc_score:.3f})',
                linewidth=2
            )
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=settings.DPI, bbox_inches='tight')
            
        return fig
    
    def plot_learning_curves(
        self,
        learning_data: Dict[str, Any],
        title: str = "Learning Curves",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot learning curves showing training and validation performance.
        
        Args:
            learning_data: Dictionary with learning curve data.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        self.logger.info("Creating learning curves plot")
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        train_sizes = learning_data['train_sizes']
        train_scores_mean = learning_data['train_scores_mean']
        train_scores_std = learning_data['train_scores_std']
        val_scores_mean = learning_data['val_scores_mean']
        val_scores_std = learning_data['val_scores_std']
        
        # Plot training scores
        ax.plot(train_sizes, train_scores_mean, 'o-', color=self.colors[0],
                label='Training Score', linewidth=2, markersize=6)
        ax.fill_between(train_sizes, 
                       np.array(train_scores_mean) - np.array(train_scores_std),
                       np.array(train_scores_mean) + np.array(train_scores_std),
                       alpha=0.1, color=self.colors[0])
        
        # Plot validation scores
        ax.plot(train_sizes, val_scores_mean, 'o-', color=self.colors[1],
                label='Validation Score', linewidth=2, markersize=6)
        ax.fill_between(train_sizes,
                       np.array(val_scores_mean) - np.array(val_scores_std),
                       np.array(val_scores_mean) + np.array(val_scores_std),
                       alpha=0.1, color=self.colors[1])
        
        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel(f'Score ({learning_data.get("scoring_metric", "Score")})', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=settings.DPI, bbox_inches='tight')
            
        return fig
    
    def plot_validation_curve(
        self,
        validation_data: Dict[str, Any],
        title: str = "Validation Curve",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot validation curve for hyperparameter analysis.
        
        Args:
            validation_data: Dictionary with validation curve data.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        self.logger.info("Creating validation curve plot")
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        param_range = validation_data['param_range']
        train_scores_mean = validation_data['train_scores_mean']
        train_scores_std = validation_data['train_scores_std']
        val_scores_mean = validation_data['val_scores_mean']
        val_scores_std = validation_data['val_scores_std']
        
        # Plot training scores
        ax.plot(param_range, train_scores_mean, 'o-', color=self.colors[0],
                label='Training Score', linewidth=2, markersize=6)
        ax.fill_between(param_range,
                       np.array(train_scores_mean) - np.array(train_scores_std),
                       np.array(train_scores_mean) + np.array(train_scores_std),
                       alpha=0.1, color=self.colors[0])
        
        # Plot validation scores
        ax.plot(param_range, val_scores_mean, 'o-', color=self.colors[1],
                label='Validation Score', linewidth=2, markersize=6)
        ax.fill_between(param_range,
                       np.array(val_scores_mean) - np.array(val_scores_std),
                       np.array(val_scores_mean) + np.array(val_scores_std),
                       alpha=0.1, color=self.colors[1])
        
        # Mark best parameter
        best_idx = validation_data.get('best_param_idx')
        if best_idx is not None:
            ax.axvline(x=param_range[best_idx], color='red', linestyle='--', alpha=0.7,
                      label=f'Best: {param_range[best_idx]}')
        
        ax.set_xlabel(validation_data['param_name'], fontsize=12)
        ax.set_ylabel(f'Score ({validation_data.get("scoring_metric", "Score")})', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        
        # Use log scale if parameter range spans multiple orders of magnitude
        if len(param_range) > 1:
            range_ratio = max(param_range) / min(param_range) if min(param_range) > 0 else 1
            if range_ratio > 100:
                ax.set_xscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=settings.DPI, bbox_inches='tight')
            
        return fig
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_scores: np.ndarray,
        title: str = "Feature Importance",
        max_features: int = 20,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot feature importance scores.
        
        Args:
            feature_names: Names of features.
            importance_scores: Importance scores for each feature.
            title: Plot title.
            max_features: Maximum number of features to display.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        self.logger.info("Creating feature importance plot")
        
        # Sort features by importance
        indices = np.argsort(importance_scores)[::-1]
        
        # Limit to top features
        top_indices = indices[:max_features]
        top_features = [feature_names[i] for i in top_indices]
        top_scores = importance_scores[top_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(self.figure_size[0], max(8, len(top_features) * 0.4)))
        
        # Horizontal bar plot
        bars = ax.barh(range(len(top_features)), top_scores, color=self.colors[0])
        
        # Customize plot
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, top_scores)):
            ax.text(score + max(top_scores) * 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', ha='left', va='center', fontsize=10)
        
        # Invert y-axis to show most important features at top
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=settings.DPI, bbox_inches='tight')
            
        return fig
    
    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric_column: str,
        title: str = "Model Performance Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot model performance comparison.
        
        Args:
            comparison_df: DataFrame with model comparison results.
            metric_column: Column name for the metric to plot.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        self.logger.info("Creating model comparison plot")
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Sort by metric
        sorted_df = comparison_df.sort_values(metric_column, ascending=True)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(sorted_df)), sorted_df[metric_column], 
                      color=self.colors[:len(sorted_df)])
        
        # Customize plot
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df['model_name'])
        ax.set_xlabel(metric_column.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, sorted_df[metric_column])):
            ax.text(score + max(sorted_df[metric_column]) * 0.01, 
                   bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=settings.DPI, bbox_inches='tight')
            
        return fig
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Residual Analysis",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot residuals for regression analysis.
        
        Args:
            y_true: True target values.
            y_pred: Predicted target values.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        self.logger.info("Creating residuals plot")
        
        residuals = y_true - y_pred
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.6, color=self.colors[0])
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted')
        ax1.grid(True, alpha=0.3)
        
        # 2. Q-Q plot for normality check
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normality Check)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Histogram of residuals
        ax3.hist(residuals, bins=30, alpha=0.7, color=self.colors[1], edgecolor='black')
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Residuals')
        ax3.grid(True, alpha=0.3)
        
        # 4. Actual vs Predicted
        ax4.scatter(y_true, y_pred, alpha=0.6, color=self.colors[2])
        # Perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax4.plot([min_val, max_val], [min_val, max_val], 'red', linestyle='--', alpha=0.7)
        ax4.set_xlabel('Actual Values')
        ax4.set_ylabel('Predicted Values')
        ax4.set_title('Actual vs Predicted')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=settings.DPI, bbox_inches='tight')
            
        return fig
    
    def plot_clustering_results(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        centers: Optional[np.ndarray] = None,
        title: str = "Clustering Results",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot clustering results with dimensionality reduction if needed.
        
        Args:
            X: Feature matrix.
            labels: Cluster labels.
            centers: Cluster centers (if available).
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        self.logger.info("Creating clustering results plot")
        
        # Reduce dimensionality for visualization if needed
        if X.shape[1] > 2:
            # Use PCA for dimensionality reduction
            pca = PCA(n_components=2)
            X_viz = pca.fit_transform(StandardScaler().fit_transform(X))
            
            if centers is not None and centers.shape[1] > 2:
                centers_viz = pca.transform(StandardScaler().fit_transform(centers))
            else:
                centers_viz = centers
        else:
            X_viz = X
            centers_viz = centers
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Plot points
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            if label == -1:  # Noise points (DBSCAN)
                color = 'black'
                marker = 'x'
                alpha = 0.5
                label_name = 'Noise'
            else:
                color = self.colors[i % len(self.colors)]
                marker = 'o'
                alpha = 0.7
                label_name = f'Cluster {label}'
            
            mask = labels == label
            ax.scatter(X_viz[mask, 0], X_viz[mask, 1], 
                      c=[color], marker=marker, alpha=alpha, 
                      label=label_name, s=50)
        
        # Plot cluster centers
        if centers_viz is not None:
            ax.scatter(centers_viz[:, 0], centers_viz[:, 1], 
                      c='red', marker='x', s=200, linewidths=3,
                      label='Centers')
        
        ax.set_xlabel('First Component' if X.shape[1] > 2 else 'Feature 1')
        ax.set_ylabel('Second Component' if X.shape[1] > 2 else 'Feature 2')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=settings.DPI, bbox_inches='tight')
            
        return fig
    
    def plot_decision_boundary(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        title: str = "Decision Boundary",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot decision boundary for 2D classification problems.
        
        Args:
            model: Trained classifier.
            X: Feature matrix (must be 2D).
            y: Target labels.
            feature_names: Names of features.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        if X.shape[1] != 2:
            raise ValueError("Decision boundary plotting requires exactly 2 features")
        
        self.logger.info("Creating decision boundary plot")
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Create mesh
        h = 0.02  # Step size in mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Make predictions on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        try:
            Z = model.predict(mesh_points)
        except Exception:
            # If model is part of a pipeline, we need to be more careful
            Z = model.predict(mesh_points)
        
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
        
        # Plot data points
        unique_labels = np.unique(y)
        for i, label in enumerate(unique_labels):
            mask = y == label
            ax.scatter(X[mask, 0], X[mask, 1], 
                      c=[self.colors[i % len(self.colors)]], 
                      label=f'Class {label}', alpha=0.8, s=50)
        
        ax.set_xlabel(feature_names[0] if feature_names else 'Feature 1')
        ax.set_ylabel(feature_names[1] if feature_names else 'Feature 2')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=settings.DPI, bbox_inches='tight')
            
        return fig
    
    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
        title: str = "Calibration Plot",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot calibration curve for probability calibration assessment.
        
        Args:
            y_true: True binary labels.
            y_prob: Predicted probabilities for positive class.
            n_bins: Number of bins for calibration curve.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        try:
            from sklearn.calibration import calibration_curve
        except ImportError:
            self.logger.warning("sklearn.calibration not available")
            return None
        
        self.logger.info("Creating calibration curve plot")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins
        )
        
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", 
                 color=self.colors[0], label="Model", linewidth=2, markersize=6)
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", linewidth=2)
        ax1.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax1.set_ylabel("Fraction of Positives", fontsize=12)
        ax1.set_title("Calibration Curve", fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Histogram of predictions
        ax2.hist(y_prob, bins=n_bins, alpha=0.7, color=self.colors[1], 
                 edgecolor='black')
        ax2.set_xlabel("Predicted Probability", fontsize=12)
        ax2.set_ylabel("Count", fontsize=12)
        ax2.set_title("Distribution of Predictions", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=settings.DPI, bbox_inches='tight')
            
        return fig
    
    def plot_feature_importance_waterfall(
        self,
        base_value: float,
        feature_contributions: np.ndarray,
        feature_names: List[str],
        prediction: float,
        title: str = "Feature Contribution Analysis",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create waterfall plot for feature contributions (SHAP-style).
        
        Args:
            base_value: Base prediction value.
            feature_contributions: Contribution of each feature.
            feature_names: Names of features.
            prediction: Final prediction value.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        self.logger.info("Creating feature importance waterfall plot")
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Sort by absolute contribution
        sorted_indices = np.argsort(np.abs(feature_contributions))[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_contributions = feature_contributions[sorted_indices]
        
        # Create waterfall chart
        cumulative = base_value
        x_pos = np.arange(len(sorted_features) + 2)
        
        # Base value
        ax.bar(0, base_value, color='gray', alpha=0.7, label='Base Value')
        
        # Feature contributions
        for i, (feature, contrib) in enumerate(zip(sorted_features, sorted_contributions)):
            color = 'green' if contrib > 0 else 'red'
            ax.bar(i + 1, contrib, bottom=cumulative, color=color, alpha=0.7)
            ax.text(i + 1, cumulative + contrib/2, f'{contrib:.3f}', 
                    ha='center', va='center', fontsize=8)
            cumulative += contrib
        
        # Final prediction
        ax.bar(len(sorted_features) + 1, 0, bottom=prediction, color='blue', 
               alpha=0.7, label=f'Prediction: {prediction:.3f}')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Base'] + sorted_features + ['Prediction'], rotation=45)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=settings.DPI, bbox_inches='tight')
            
        return fig
    
    def plot_fairness_metrics(
        self,
        fairness_results: Dict[str, float],
        sensitive_attribute: str,
        title: str = "Model Fairness Analysis",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize fairness metrics across groups.
        
        Args:
            fairness_results: Dictionary with fairness metrics.
            sensitive_attribute: Name of sensitive attribute.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        self.logger.info("Creating fairness metrics plot")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Group-wise performance
        group_metrics = {k: v for k, v in fairness_results.items() 
                        if k.startswith('accuracy_group_')}
        
        if group_metrics:
            groups = [k.split('_')[-1] for k in group_metrics.keys()]
            accuracies = list(group_metrics.values())
            
            bars = ax1.bar(groups, accuracies, color=self.colors[:len(groups)])
            ax1.set_xlabel(f'{sensitive_attribute} Groups', fontsize=12)
            ax1.set_ylabel('Accuracy', fontsize=12)
            ax1.set_title('Performance by Group', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Fairness metrics
        fairness_metrics = ['demographic_parity_diff']  # Add more as available
        fairness_values = [fairness_results.get(metric, 0) for metric in fairness_metrics 
                          if fairness_results.get(metric) is not None]
        fairness_names = [metric.replace('_', ' ').title() for metric in fairness_metrics 
                         if fairness_results.get(metric) is not None]
        
        if fairness_values:
            bars = ax2.bar(fairness_names, fairness_values, color='orange', alpha=0.7)
            ax2.set_ylabel('Metric Value', fontsize=12)
            ax2.set_title('Fairness Metrics', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Add threshold line for acceptable bias
            ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, 
                       label='Acceptable Threshold')
            ax2.legend()
            
            # Add value labels
            for bar, val in zip(bars, fairness_values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(fairness_values) * 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=settings.DPI, bbox_inches='tight')
            
        return fig
    
    def create_interactive_scatter_plot(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        target_name: str = "Target",
        title: str = "Interactive Data Exploration"
    ) -> go.Figure:
        """Create interactive scatter plot using Plotly.
        
        Args:
            X: Feature matrix.
            y: Target values.
            feature_names: Names of features.
            target_name: Name of target variable.
            title: Plot title.
            
        Returns:
            Plotly figure.
        """
        self.logger.info("Creating interactive scatter plot")
        
        # Prepare data
        if X.shape[1] < 2:
            raise ValueError("Need at least 2 features for scatter plot")
        
        # Use first two features for scatter plot
        df = pd.DataFrame({
            'x': X[:, 0],
            'y': X[:, 1],
            'target': y
        })
        
        if feature_names:
            df = df.rename(columns={'x': feature_names[0], 'y': feature_names[1]})
            x_col, y_col = feature_names[0], feature_names[1]
        else:
            x_col, y_col = 'x', 'y'
        
        # Create scatter plot
        fig = px.scatter(
            df, x=x_col, y=y_col, color='target',
            title=title,
            labels={'target': target_name},
            hover_data=[x_col, y_col, 'target']
        )
        
        fig.update_layout(
            width=800,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_model_performance_dashboard(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create comprehensive performance dashboard using Plotly.
        
        Args:
            results: Model evaluation results.
            save_path: Path to save the dashboard.
            
        Returns:
            Plotly figure with subplots.
        """
        self.logger.info("Creating model performance dashboard")
        
        task_type = results.get('task_type', 'classification')
        
        if task_type == 'classification':
            return self._create_classification_dashboard(results, save_path)
        elif task_type == 'regression':
            return self._create_regression_dashboard(results, save_path)
        else:
            return self._create_clustering_dashboard(results, save_path)
    
    def _create_classification_dashboard(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create classification performance dashboard."""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Confusion Matrix', 'ROC Curve', 'Metrics Comparison', 'Cross-Validation'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Confusion Matrix
        cm = results.get('confusion_matrix')
        if cm is not None:
            fig.add_trace(
                go.Heatmap(z=cm, colorscale='Blues', showscale=False),
                row=1, col=1
            )
        
        # 2. ROC Curve
        roc_data = results.get('roc_curve')
        if roc_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=roc_data['fpr'], y=roc_data['tpr'],
                    mode='lines', name='ROC Curve',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=2
            )
            # Add diagonal line
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines', name='Random',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=2
            )
        
        # 3. Metrics Comparison
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        metric_values = [results.get(metric, 0) for metric in metrics]
        metric_names = [metric.replace('test_', '').title() for metric in metrics]
        
        fig.add_trace(
            go.Bar(x=metric_names, y=metric_values, name='Metrics'),
            row=2, col=1
        )
        
        # 4. Cross-validation results (if available)
        cv_results = results.get('cross_validation', {})
        if cv_results:
            cv_metrics = list(cv_results.keys())[:4]  # Show first 4 metrics
            cv_means = [cv_results[metric]['mean'] if cv_results[metric] else 0 for metric in cv_metrics]
            cv_names = [metric.replace('_', ' ').title() for metric in cv_metrics]
            
            fig.add_trace(
                go.Bar(x=cv_names, y=cv_means, name='CV Metrics'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"Classification Performance Dashboard - {results.get('model_name', 'Model')}",
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def _create_regression_dashboard(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create regression performance dashboard."""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Metrics Comparison', 'Learning Curves', 'Cross-Validation', 'Residuals Info'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Metrics Comparison
        metrics = ['test_r2', 'test_rmse', 'test_mae']
        metric_values = [results.get(metric, 0) for metric in metrics]
        metric_names = [metric.replace('test_', '').upper() for metric in metrics]
        
        fig.add_trace(
            go.Bar(x=metric_names, y=metric_values, name='Test Metrics'),
            row=1, col=1
        )
        
        # 2. Learning Curves
        learning_data = results.get('learning_curves')
        if learning_data:
            train_sizes = learning_data['train_sizes']
            train_scores = learning_data['train_scores_mean']
            val_scores = learning_data['val_scores_mean']
            
            fig.add_trace(
                go.Scatter(x=train_sizes, y=train_scores, mode='lines+markers', 
                          name='Training', line=dict(color='blue')),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=train_sizes, y=val_scores, mode='lines+markers',
                          name='Validation', line=dict(color='red')),
                row=1, col=2
            )
        
        # 3. Cross-validation results
        cv_results = results.get('cross_validation', {})
        if cv_results:
            cv_metrics = list(cv_results.keys())[:3]
            cv_means = [cv_results[metric]['mean'] if cv_results[metric] else 0 for metric in cv_metrics]
            cv_names = [metric.replace('neg_', '').replace('_', ' ').title() for metric in cv_metrics]
            
            fig.add_trace(
                go.Bar(x=cv_names, y=cv_means, name='CV Metrics'),
                row=2, col=1
            )
        
        # 4. Residuals statistics
        residuals_info = results.get('residuals', {})
        if residuals_info:
            res_metrics = ['mean', 'std', 'min', 'max']
            res_values = [residuals_info.get(metric, 0) for metric in res_metrics]
            
            fig.add_trace(
                go.Bar(x=res_metrics, y=res_values, name='Residuals'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"Regression Performance Dashboard - {results.get('model_name', 'Model')}",
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def _create_clustering_dashboard(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create clustering performance dashboard."""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Clustering Metrics', 'Cluster Sizes', 'Performance Summary', 'Cluster Info')
        )
        
        # 1. Clustering metrics
        metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
        metric_values = [results.get(metric, 0) for metric in metrics if results.get(metric) is not None]
        metric_names = [metric.replace('_', ' ').title() for metric in metrics if results.get(metric) is not None]
        
        if metric_values:
            fig.add_trace(
                go.Bar(x=metric_names, y=metric_values, name='Metrics'),
                row=1, col=1
            )
        
        # 2. Cluster sizes
        cluster_sizes = results.get('cluster_sizes', {})
        if cluster_sizes:
            clusters = list(cluster_sizes.keys())
            sizes = list(cluster_sizes.values())
            
            fig.add_trace(
                go.Bar(x=[f'Cluster {c}' for c in clusters], y=sizes, name='Cluster Sizes'),
                row=1, col=2
            )
        
        # 3. Basic cluster information
        info_metrics = ['n_clusters']
        if 'n_noise_points' in results:
            info_metrics.append('n_noise_points')
        
        info_values = [results.get(metric, 0) for metric in info_metrics]
        info_names = [metric.replace('_', ' ').title() for metric in info_metrics]
        
        fig.add_trace(
            go.Bar(x=info_names, y=info_values, name='Cluster Info'),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"Clustering Performance Dashboard - {results.get('model_name', 'Model')}",
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig