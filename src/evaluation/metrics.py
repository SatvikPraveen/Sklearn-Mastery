"""Comprehensive evaluation metrics and model performance analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error,
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    calinski_harabasz_score, davies_bouldin_score
)
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.base import BaseEstimator
import warnings

from ..config.settings import settings
from ..config.logging_config import LoggerMixin
from .utils import (
    validate_evaluation_inputs, ensure_numpy_array, safe_metric_computation,
    bootstrap_metric, compute_metric_stability
)


class ModelEvaluator(LoggerMixin):
    """Comprehensive model evaluation with detailed metrics and analysis."""
    
    def __init__(self, task_type: str = 'classification'):
        """Initialize model evaluator.
        
        Args:
            task_type: Type of ML task ('classification', 'regression', 'clustering').
        """
        self.task_type = task_type
        self.results = {}
        
    def evaluate_model(
        self,
        model: BaseEstimator,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "Model",
        **kwargs
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation.
        
        Args:
            model: Trained model.
            X_train: Training features.
            X_test: Test features.
            y_train: Training targets.
            y_test: Test targets.
            model_name: Name of the model for logging.
            **kwargs: Additional evaluation options.
            
        Returns:
            Dictionary with comprehensive evaluation results.
        """
        self.logger.info(f"Evaluating {model_name} for {self.task_type}")
        
        # Validate inputs
        validate_evaluation_inputs(X_test, y_test, self.task_type)
        
        results = {
            'model_name': model_name,
            'task_type': self.task_type,
            'evaluation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        if self.task_type == 'classification':
            results.update(self._evaluate_classification(
                model, X_train, X_test, y_train, y_test
            ))
        elif self.task_type == 'regression':
            results.update(self._evaluate_regression(
                model, X_train, X_test, y_train, y_test
            ))
        elif self.task_type == 'clustering':
            results.update(self._evaluate_clustering(
                model, X_train, y_train if y_train is not None else None
            ))
        
        # Add cross-validation results
        if self.task_type != 'clustering':
            results['cross_validation'] = self._cross_validation_analysis(
                model, X_train, y_train
            )
        
        # Add learning curves
        if self.task_type != 'clustering':
            results['learning_curves'] = self._compute_learning_curves(
                model, X_train, y_train
            )
        
        # Enhanced evaluations (if requested in kwargs)
        if kwargs.get('include_calibration', False) and hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
            results['calibration'] = self.evaluate_calibration(y_test, y_pred_proba)
        
        if kwargs.get('include_fairness', False) and 'sensitive_attrs' in kwargs:
            y_pred = model.predict(X_test)
            results['fairness'] = self.evaluate_fairness(
                y_test, y_pred, kwargs['sensitive_attrs']
            )
        
        if kwargs.get('include_feature_importance', False):
            results['feature_importance'] = self.analyze_feature_importance(
                model, kwargs.get('feature_names')
            )
        
        self.results[model_name] = results
        return results
    
    def _evaluate_classification(
        self,
        model: BaseEstimator,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate classification model."""
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        
        # Get prediction probabilities if available
        try:
            y_pred_proba = model.predict_proba(X_test)
            y_train_pred_proba = model.predict_proba(X_train)
            has_proba = True
        except AttributeError:
            y_pred_proba = None
            y_train_pred_proba = None
            has_proba = False
        
        results = {}
        
        # Basic metrics
        results['test_accuracy'] = safe_metric_computation(accuracy_score, y_test, y_pred)
        results['train_accuracy'] = safe_metric_computation(accuracy_score, y_train, y_train_pred)
        
        # Determine average strategy based on number of classes
        n_classes = len(np.unique(y_train))
        avg_strategy = 'binary' if n_classes == 2 else 'macro'
        
        results['test_precision'] = safe_metric_computation(
            precision_score, y_test, y_pred, average=avg_strategy, zero_division=0
        )
        results['test_recall'] = safe_metric_computation(
            recall_score, y_test, y_pred, average=avg_strategy, zero_division=0
        )
        results['test_f1'] = safe_metric_computation(
            f1_score, y_test, y_pred, average=avg_strategy, zero_division=0
        )
        
        results['train_precision'] = safe_metric_computation(
            precision_score, y_train, y_train_pred, average=avg_strategy, zero_division=0
        )
        results['train_recall'] = safe_metric_computation(
            recall_score, y_train, y_train_pred, average=avg_strategy, zero_division=0
        )
        results['train_f1'] = safe_metric_computation(
            f1_score, y_train, y_train_pred, average=avg_strategy, zero_division=0
        )
        
        # ROC AUC (if probabilities available)
        if has_proba:
            try:
                if n_classes == 2:
                    results['test_roc_auc'] = safe_metric_computation(
                        roc_auc_score, y_test, y_pred_proba[:, 1]
                    )
                    results['train_roc_auc'] = safe_metric_computation(
                        roc_auc_score, y_train, y_train_pred_proba[:, 1]
                    )
                else:
                    # Check if we have enough samples per class
                    if len(np.unique(y_test)) == len(np.unique(y_train)):
                        results['test_roc_auc'] = safe_metric_computation(
                            roc_auc_score, y_test, y_pred_proba, 
                            multi_class='ovr', average='macro'
                        )
                        results['train_roc_auc'] = safe_metric_computation(
                            roc_auc_score, y_train, y_train_pred_proba,
                            multi_class='ovr', average='macro'
                        )
                    else:
                        results['test_roc_auc'] = None
                        results['train_roc_auc'] = None
            except (ValueError, IndexError) as e:
                self.logger.warning(f"Could not compute ROC AUC: {e}")
                results['test_roc_auc'] = None
                results['train_roc_auc'] = None
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        
        # Classification report
        results['classification_report'] = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        
        # ROC curve data (for binary classification)
        if has_proba and n_classes == 2:
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
            results['roc_curve'] = {
                'fpr': fpr.tolist(), 
                'tpr': tpr.tolist(), 
                'thresholds': thresholds.tolist()
            }
            
            # Precision-recall curve
            precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])
            results['pr_curve'] = {
                'precision': precision.tolist(), 
                'recall': recall.tolist(), 
                'thresholds': pr_thresholds.tolist()
            }
        
        # Overfitting analysis
        train_acc = results.get('train_accuracy', 0)
        test_acc = results.get('test_accuracy', 0)
        results['overfitting_score'] = train_acc - test_acc if train_acc and test_acc else None
        
        return results
    
    def _evaluate_regression(
        self,
        model: BaseEstimator,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate regression model."""
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        
        results = {}
        
        # Basic metrics
        results['test_mse'] = safe_metric_computation(mean_squared_error, y_test, y_pred)
        results['test_rmse'] = np.sqrt(results['test_mse']) if results['test_mse'] else None
        results['test_mae'] = safe_metric_computation(mean_absolute_error, y_test, y_pred)
        results['test_r2'] = safe_metric_computation(r2_score, y_test, y_pred)
        
        results['train_mse'] = safe_metric_computation(mean_squared_error, y_train, y_train_pred)
        results['train_rmse'] = np.sqrt(results['train_mse']) if results['train_mse'] else None
        results['train_mae'] = safe_metric_computation(mean_absolute_error, y_train, y_train_pred)
        results['train_r2'] = safe_metric_computation(r2_score, y_train, y_train_pred)
        
        # MAPE (if no zero values in y_test)
        if not np.any(y_test == 0):
            results['test_mape'] = safe_metric_computation(mean_absolute_percentage_error, y_test, y_pred)
            results['train_mape'] = safe_metric_computation(mean_absolute_percentage_error, y_train, y_train_pred)
        else:
            results['test_mape'] = None
            results['train_mape'] = None
        
        # Residual analysis
        residuals = y_test - y_pred
        results['residuals'] = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'min': float(np.min(residuals)),
            'max': float(np.max(residuals)),
            'q25': float(np.percentile(residuals, 25)),
            'q75': float(np.percentile(residuals, 75))
        }
        
        # Prediction vs actual correlation
        if len(y_test) > 1:
            corr_matrix = np.corrcoef(y_test, y_pred)
            results['pred_actual_corr'] = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else None
        else:
            results['pred_actual_corr'] = None
        
        # Overfitting analysis
        train_r2 = results.get('train_r2', 0)
        test_r2 = results.get('test_r2', 0)
        results['overfitting_score'] = train_r2 - test_r2 if train_r2 and test_r2 else None
        
        return results
    
    def _evaluate_clustering(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y_true: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Evaluate clustering model."""
        
        # Get cluster labels
        if hasattr(model, 'labels_'):
            labels = model.labels_
        else:
            labels = model.fit_predict(X)
        
        results = {}
        
        # Internal metrics (don't require true labels)
        results['silhouette_score'] = safe_metric_computation(silhouette_score, X, labels)
        results['calinski_harabasz_score'] = safe_metric_computation(calinski_harabasz_score, X, labels)
        results['davies_bouldin_score'] = safe_metric_computation(davies_bouldin_score, X, labels)
        
        # External metrics (require true labels)
        if y_true is not None:
            results['adjusted_rand_score'] = safe_metric_computation(adjusted_rand_score, y_true, labels)
            results['normalized_mutual_info'] = safe_metric_computation(normalized_mutual_info_score, y_true, labels)
        
        # Cluster statistics
        unique_labels, counts = np.unique(labels, return_counts=True)
        results['n_clusters'] = len(unique_labels)
        results['cluster_sizes'] = dict(zip(unique_labels.tolist(), counts.tolist()))
        
        # Noise points (for DBSCAN)
        if -1 in labels:
            results['n_noise_points'] = int(np.sum(labels == -1))
            results['noise_ratio'] = float(results['n_noise_points'] / len(labels))
        
        return results
    
    def _cross_validation_analysis(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = None
    ) -> Dict[str, Any]:
        """Perform cross-validation analysis."""
        
        cv_folds = cv_folds or settings.DEFAULT_CV_FOLDS
        
        # Select appropriate scoring metrics
        if self.task_type == 'classification':
            scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            try:
                # Add ROC AUC if binary classification
                if len(np.unique(y)) == 2:
                    scoring_metrics.append('roc_auc')
                else:
                    scoring_metrics.append('roc_auc_ovr')
            except Exception:
                pass
        else:  # regression
            scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        cv_results = {}
        
        for metric in scoring_metrics:
            try:
                scores = cross_val_score(
                    model, X, y, cv=cv_folds, scoring=metric, n_jobs=-1
                )
                stability = compute_metric_stability(scores)
                cv_results[metric] = {
                    'mean': stability['mean'],
                    'std': stability['std'],
                    'scores': scores.tolist(),
                    'stability': stability
                }
            except Exception as e:
                self.logger.warning(f"Could not compute CV score for {metric}: {e}")
                cv_results[metric] = None
        
        return cv_results
    
    def _compute_learning_curves(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        train_sizes: np.ndarray = None
    ) -> Optional[Dict[str, Any]]:
        """Compute learning curves for bias-variance analysis."""
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        # Select scoring metric
        scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
        
        try:
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y,
                train_sizes=train_sizes,
                cv=settings.DEFAULT_CV_FOLDS,
                scoring=scoring,
                n_jobs=-1,
                random_state=settings.RANDOM_SEED
            )
            
            return {
                'train_sizes': train_sizes_abs.tolist(),
                'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
                'train_scores_std': np.std(train_scores, axis=1).tolist(),
                'val_scores_mean': np.mean(val_scores, axis=1).tolist(),
                'val_scores_std': np.std(val_scores, axis=1).tolist(),
                'scoring_metric': scoring
            }
        except Exception as e:
            self.logger.warning(f"Could not compute learning curves: {e}")
            return None
    
    def evaluate_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
        """Evaluate model calibration using reliability diagrams.
        
        Args:
            y_true: True binary labels.
            y_pred_proba: Prediction probabilities.
            n_bins: Number of bins for calibration curve.
            
        Returns:
            Dictionary with calibration metrics.
        """
        try:
            from sklearn.calibration import calibration_curve
        except ImportError:
            self.logger.warning("sklearn.calibration not available")
            return {}
        
        # Ensure binary classification
        if len(np.unique(y_true)) != 2 or y_pred_proba.shape[1] != 2:
            self.logger.warning("Calibration evaluation requires binary classification")
            return {}
        
        y_prob = y_pred_proba[:, 1]  # Probability of positive class
        
        # Brier score
        brier_score = np.mean((y_prob - y_true) ** 2)
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins
        )
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return {
            'brier_score': float(brier_score),
            'expected_calibration_error': float(ece),
            'calibration_curve': {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            }
        }
    
    def analyze_feature_importance(self, model: BaseEstimator, feature_names: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Analyze feature importance if available.
        
        Args:
            model: Trained model.
            feature_names: Names of features.
            
        Returns:
            Dictionary with feature importance analysis.
        """
        if not hasattr(model, 'feature_importances_'):
            return None
        
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Summary statistics
        summary_stats = {
            'mean_importance': float(np.mean(importances)),
            'std_importance': float(np.std(importances)),
            'max_importance': float(np.max(importances)),
            'min_importance': float(np.min(importances)),
            'n_zero_importance': int(np.sum(importances == 0))
        }
        
        return {
            'feature_importance_df': importance_df.to_dict('records'),
            'summary_stats': summary_stats,
            'top_features': importance_df.head(10).to_dict('records')
        }
    
    def evaluate_fairness(self, y_true: np.ndarray, y_pred: np.ndarray, sensitive_attrs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Evaluate model fairness across sensitive groups.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            sensitive_attrs: Dictionary of sensitive attributes.
            
        Returns:
            Dictionary with fairness metrics.
        """
        fairness_results = {}
        
        for attr_name, attr_values in sensitive_attrs.items():
            if len(attr_values) != len(y_true):
                self.logger.warning(f"Sensitive attribute {attr_name} length mismatch")
                continue
            
            unique_groups = np.unique(attr_values)
            group_metrics = {}
            
            # Compute metrics for each group
            for group in unique_groups:
                mask = attr_values == group
                if np.sum(mask) > 0:
                    group_accuracy = safe_metric_computation(accuracy_score, y_true[mask], y_pred[mask])
                    group_metrics[f'accuracy_group_{group}'] = group_accuracy
            
            # Demographic parity difference
            group_positive_rates = {}
            for group in unique_groups:
                mask = attr_values == group
                if np.sum(mask) > 0:
                    positive_rate = np.mean(y_pred[mask] == 1)  # Assuming binary classification
                    group_positive_rates[group] = positive_rate
            
            if len(group_positive_rates) > 1:
                demographic_parity_diff = (
                    max(group_positive_rates.values()) - min(group_positive_rates.values())
                )
                group_metrics['demographic_parity_diff'] = demographic_parity_diff
            
            fairness_results[attr_name] = group_metrics
        
        return fairness_results
    
    def evaluate_time_series(self, y_true: np.ndarray, y_pred: np.ndarray, seasonal_period: Optional[int] = None) -> Dict[str, Any]:
        """Specialized evaluation for time series predictions.
        
        Args:
            y_true: True time series values.
            y_pred: Predicted time series values.
            seasonal_period: Seasonal period for MASE calculation.
            
        Returns:
            Dictionary with time series specific metrics.
        """
        results = {}
        
        # Standard regression metrics
        results['mse'] = safe_metric_computation(mean_squared_error, y_true, y_pred)
        results['rmse'] = np.sqrt(results['mse']) if results['mse'] else None
        results['mae'] = safe_metric_computation(mean_absolute_error, y_true, y_pred)
        results['r2'] = safe_metric_computation(r2_score, y_true, y_pred)
        
        # MAPE if no zeros
        if not np.any(y_true == 0):
            results['mape'] = safe_metric_computation(mean_absolute_percentage_error, y_true, y_pred)
        
        # Time series specific metrics
        if seasonal_period and seasonal_period < len(y_true):
            # Seasonal naive forecast for comparison
            seasonal_naive = np.roll(y_true, seasonal_period)
            seasonal_naive[:seasonal_period] = y_true[:seasonal_period]
            
            # MASE (Mean Absolute Scaled Error)
            mae_naive = np.mean(np.abs(y_true[seasonal_period:] - seasonal_naive[seasonal_period:]))
            mae_model = np.mean(np.abs(y_true - y_pred))
            results['mase'] = mae_model / mae_naive if mae_naive > 0 else np.inf
        
        return results
    
    def compare_models(self, results_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """Compare multiple model results.
        
        Args:
            results_list: List of evaluation results from different models.
            
        Returns:
            DataFrame with model comparison.
        """
        comparison_data = []
        
        for result in results_list:
            row = {'model_name': result['model_name']}
            
            if self.task_type == 'classification':
                row.update({
                    'test_accuracy': result.get('test_accuracy'),
                    'test_precision': result.get('test_precision'),
                    'test_recall': result.get('test_recall'),
                    'test_f1': result.get('test_f1'),
                    'test_roc_auc': result.get('test_roc_auc'),
                    'overfitting_score': result.get('overfitting_score')
                })
            elif self.task_type == 'regression':
                row.update({
                    'test_r2': result.get('test_r2'),
                    'test_rmse': result.get('test_rmse'),
                    'test_mae': result.get('test_mae'),
                    'test_mape': result.get('test_mape'),
                    'overfitting_score': result.get('overfitting_score')
                })
            elif self.task_type == 'clustering':
                row.update({
                    'silhouette_score': result.get('silhouette_score'),
                    'calinski_harabasz_score': result.get('calinski_harabasz_score'),
                    'davies_bouldin_score': result.get('davies_bouldin_score'),
                    'n_clusters': result.get('n_clusters'),
                    'adjusted_rand_score': result.get('adjusted_rand_score')
                })
            
            # Add cross-validation results
            cv_results = result.get('cross_validation', {})
            for metric, values in cv_results.items():
                if values is not None:
                    row[f'cv_{metric}_mean'] = values['mean']
                    row[f'cv_{metric}_std'] = values['std']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by primary metric
        if self.task_type == 'classification' and 'test_accuracy' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('test_accuracy', ascending=False)
        elif self.task_type == 'regression' and 'test_r2' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('test_r2', ascending=False)
        elif self.task_type == 'clustering' and 'silhouette_score' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('silhouette_score', ascending=False)
        
        return comparison_df
    
    def generate_evaluation_summary(self, result: Dict[str, Any]) -> str:
        """Generate a text summary of evaluation results.
        
        Args:
            result: Evaluation results dictionary.
            
        Returns:
            Formatted text summary.
        """
        model_name = result['model_name']
        task_type = result['task_type']
        
        summary_lines = [
            f"=== {model_name} Evaluation Summary ===",
            f"Task Type: {task_type.title()}",
            ""
        ]
        
        if task_type == 'classification':
            summary_lines.extend([
                f"Test Accuracy: {result.get('test_accuracy', 'N/A'):.4f}",
                f"Test Precision: {result.get('test_precision', 'N/A'):.4f}",
                f"Test Recall: {result.get('test_recall', 'N/A'):.4f}",
                f"Test F1-Score: {result.get('test_f1', 'N/A'):.4f}",
            ])
            
            if result.get('test_roc_auc') is not None:
                summary_lines.append(f"Test ROC AUC: {result['test_roc_auc']:.4f}")
            
            summary_lines.extend([
                "",
                f"Training Accuracy: {result.get('train_accuracy', 'N/A'):.4f}",
                f"Overfitting Score: {result.get('overfitting_score', 'N/A'):.4f}",
            ])
            
        elif task_type == 'regression':
            summary_lines.extend([
                f"Test R²: {result.get('test_r2', 'N/A'):.4f}",
                f"Test RMSE: {result.get('test_rmse', 'N/A'):.4f}",
                f"Test MAE: {result.get('test_mae', 'N/A'):.4f}",
            ])
            
            if result.get('test_mape') is not None:
                summary_lines.append(f"Test MAPE: {result['test_mape']:.4f}")
            
            summary_lines.extend([
                "",
                f"Training R²: {result.get('train_r2', 'N/A'):.4f}",
                f"Overfitting Score: {result.get('overfitting_score', 'N/A'):.4f}",
            ])
            
        elif task_type == 'clustering':
            summary_lines.extend([
                f"Number of Clusters: {result.get('n_clusters', 'N/A')}",
                f"Silhouette Score: {result.get('silhouette_score', 'N/A'):.4f}",
                f"Calinski-Harabasz Score: {result.get('calinski_harabasz_score', 'N/A'):.4f}",
                f"Davies-Bouldin Score: {result.get('davies_bouldin_score', 'N/A'):.4f}",
            ])
            
            if result.get('adjusted_rand_score') is not None:
                summary_lines.append(f"Adjusted Rand Score: {result['adjusted_rand_score']:.4f}")
        
        # Add cross-validation summary
        cv_results = result.get('cross_validation', {})
        if cv_results:
            summary_lines.extend(["", "Cross-Validation Results:"])
            for metric, values in cv_results.items():
                if values is not None:
                    summary_lines.append(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
        
        # Add enhanced results if available
        if result.get('calibration'):
            cal = result['calibration']
            summary_lines.extend([
                "",
                "Calibration Analysis:",
                f"  Brier Score: {cal.get('brier_score', 'N/A'):.4f}",
                f"  Expected Calibration Error: {cal.get('expected_calibration_error', 'N/A'):.4f}"
            ])
        
        if result.get('fairness'):
            summary_lines.extend(["", "Fairness Analysis:"])
            for attr_name, fairness_metrics in result['fairness'].items():
                if 'demographic_parity_diff' in fairness_metrics:
                    summary_lines.append(f"  {attr_name} Demographic Parity Diff: {fairness_metrics['demographic_parity_diff']:.4f}")
        
        return "\n".join(summary_lines)