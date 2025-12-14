"""Statistical significance testing and validation curve analysis."""

import sys
from pathlib import Path

# Handle imports that work in both package and direct import contexts
try:
    from ..config.settings import settings
    from ..config.logging_config import LoggerMixin
except ImportError:
    # Fallback for direct imports outside package context
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.settings import settings
    from config.logging_config import LoggerMixin

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.model_selection import validation_curve
from sklearn.base import BaseEstimator
import warnings


class StatisticalTester(LoggerMixin):
    """Statistical significance testing for model comparisons."""
    
    def __init__(self, alpha: float = 0.05):
        """Initialize statistical tester.
        
        Args:
            alpha: Significance level for statistical tests.
        """
        self.alpha = alpha
    
    def paired_t_test(
        self,
        scores1: np.ndarray,
        scores2: np.ndarray,
        model1_name: str = "Model 1",
        model2_name: str = "Model 2"
    ) -> Dict[str, Any]:
        """Perform paired t-test between two sets of CV scores.
        
        Args:
            scores1: Cross-validation scores for first model.
            scores2: Cross-validation scores for second model.
            model1_name: Name of first model.
            model2_name: Name of second model.
            
        Returns:
            Dictionary with test results.
        """
        from scipy import stats
        
        if len(scores1) != len(scores2):
            raise ValueError("Score arrays must have the same length")
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        # Calculate effect size (Cohen's d)
        diff = scores1 - scores2
        pooled_std = np.sqrt((np.var(scores1, ddof=1) + np.var(scores2, ddof=1)) / 2)
        cohens_d = np.mean(diff) / pooled_std if pooled_std > 0 else 0.0
        
        # Determine significance
        is_significant = p_value < self.alpha
        
        result = {
            'model1_name': model1_name,
            'model2_name': model2_name,
            'model1_mean': np.mean(scores1),
            'model2_mean': np.mean(scores2),
            'mean_difference': np.mean(diff),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'is_significant': is_significant,
            'alpha': self.alpha,
            'interpretation': self._interpret_test_result(p_value, cohens_d, model1_name, model2_name)
        }
        
        self.logger.info(f"Paired t-test: {model1_name} vs {model2_name}, p={p_value:.4f}")
        
        return result
    
    def mcnemar_test(
        self,
        y_true: np.ndarray,
        y_pred1: np.ndarray,
        y_pred2: np.ndarray,
        model1_name: str = "Model 1",
        model2_name: str = "Model 2"
    ) -> Dict[str, Any]:
        """Perform McNemar's test for classifier comparison.
        
        Args:
            y_true: True labels.
            y_pred1: Predictions from first classifier.
            y_pred2: Predictions from second classifier.
            model1_name: Name of first model.
            model2_name: Name of second model.
            
        Returns:
            Dictionary with test results.
        """
        from scipy import stats
        
        # Create contingency table
        correct1 = (y_pred1 == y_true)
        correct2 = (y_pred2 == y_true)
        
        # McNemar's table
        both_correct = np.sum(correct1 & correct2)
        model1_correct_only = np.sum(correct1 & ~correct2)
        model2_correct_only = np.sum(~correct1 & correct2)
        both_wrong = np.sum(~correct1 & ~correct2)
        
        contingency_table = np.array([
            [both_correct, model1_correct_only],
            [model2_correct_only, both_wrong]
        ])
        
        # Perform McNemar's test
        if model1_correct_only + model2_correct_only < 25:
            # Use exact binomial test for small samples
            n = model1_correct_only + model2_correct_only
            if n > 0:
                p_value = 2 * stats.binom.cdf(min(model1_correct_only, model2_correct_only), n, 0.5)
            else:
                p_value = 1.0
            chi2_stat = None
        else:
            # Use chi-square approximation
            chi2_stat = (abs(model1_correct_only - model2_correct_only) - 1)**2 / (model1_correct_only + model2_correct_only)
            p_value = 1 - stats.chi2.cdf(chi2_stat, 1)
        
        is_significant = p_value < self.alpha
        
        result = {
            'model1_name': model1_name,
            'model2_name': model2_name,
            'contingency_table': contingency_table,
            'model1_accuracy': np.mean(correct1),
            'model2_accuracy': np.mean(correct2),
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'is_significant': is_significant,
            'alpha': self.alpha,
            'interpretation': self._interpret_mcnemar_result(p_value, model1_name, model2_name)
        }
        
        self.logger.info(f"McNemar's test: {model1_name} vs {model2_name}, p={p_value:.4f}")
        
        return result
    
    def bootstrap_test(
        self,
        metric_func,
        y_true: np.ndarray,
        y_pred1: np.ndarray,
        y_pred2: np.ndarray,
        n_bootstrap: int = 1000,
        model1_name: str = "Model 1",
        model2_name: str = "Model 2"
    ) -> Dict[str, Any]:
        """Perform bootstrap test for model comparison.
        
        Args:
            metric_func: Metric function to compare (e.g., accuracy_score).
            y_true: True labels.
            y_pred1: Predictions from first model.
            y_pred2: Predictions from second model.
            n_bootstrap: Number of bootstrap samples.
            model1_name: Name of first model.
            model2_name: Name of second model.
            
        Returns:
            Dictionary with test results.
        """
        n_samples = len(y_true)
        differences = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred1_boot = y_pred1[indices]
            y_pred2_boot = y_pred2[indices]
            
            # Compute metric difference
            score1 = metric_func(y_true_boot, y_pred1_boot)
            score2 = metric_func(y_true_boot, y_pred2_boot)
            differences.append(score1 - score2)
        
        differences = np.array(differences)
        
        # Compute p-value (two-tailed test)
        p_value = 2 * min(
            np.mean(differences >= 0),
            np.mean(differences <= 0)
        )
        
        is_significant = p_value < self.alpha
        
        result = {
            'model1_name': model1_name,
            'model2_name': model2_name,
            'mean_difference': np.mean(differences),
            'std_difference': np.std(differences),
            'ci_lower': np.percentile(differences, 2.5),
            'ci_upper': np.percentile(differences, 97.5),
            'p_value': p_value,
            'is_significant': is_significant,
            'alpha': self.alpha,
            'n_bootstrap': n_bootstrap
        }
        
        self.logger.info(f"Bootstrap test: {model1_name} vs {model2_name}, p={p_value:.4f}")
        
        return result
    
    def compute_confidence_intervals(
        self, 
        scores: np.ndarray, 
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """Compute confidence intervals for metric scores.
        
        Args:
            scores: Array of metric scores.
            confidence: Confidence level.
            
        Returns:
            Dictionary with confidence interval information.
        """
        from scipy import stats
        
        mean_score = np.mean(scores)
        std_error = stats.sem(scores)
        dof = len(scores) - 1
        
        t_value = stats.t.ppf((1 + confidence) / 2, dof)
        margin_error = t_value * std_error
        
        return {
            'mean': mean_score,
            'std_error': std_error,
            'lower_bound': mean_score - margin_error,
            'upper_bound': mean_score + margin_error,
            'confidence_level': confidence,
            'margin_error': margin_error
        }
    
    def _interpret_test_result(
        self,
        p_value: float,
        cohens_d: float,
        model1_name: str,
        model2_name: str
    ) -> str:
        """Interpret paired t-test results."""
        
        if p_value >= self.alpha:
            return f"No significant difference between {model1_name} and {model2_name} (p={p_value:.4f})"
        
        # Determine which model is better
        if cohens_d > 0:
            better_model = model1_name
            worse_model = model2_name
        else:
            better_model = model2_name
            worse_model = model1_name
            cohens_d = abs(cohens_d)
        
        # Interpret effect size
        if cohens_d < 0.2:
            effect_size = "small"
        elif cohens_d < 0.5:
            effect_size = "medium"
        else:
            effect_size = "large"
        
        return f"{better_model} significantly outperforms {worse_model} with {effect_size} effect size (p={p_value:.4f}, d={cohens_d:.3f})"
    
    def _interpret_mcnemar_result(
        self,
        p_value: float,
        model1_name: str,
        model2_name: str
    ) -> str:
        """Interpret McNemar's test results."""
        
        if p_value >= self.alpha:
            return f"No significant difference in error rates between {model1_name} and {model2_name} (p={p_value:.4f})"
        else:
            return f"Significant difference in error rates between {model1_name} and {model2_name} (p={p_value:.4f})"


class ValidationCurveAnalyzer(LoggerMixin):
    """Analyze model performance across hyperparameter ranges."""
    
    def compute_validation_curve(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        param_name: str,
        param_range: List,
        cv_folds: int = None,
        scoring: str = None,
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """Compute validation curve for hyperparameter analysis.
        
        Args:
            model: Model to analyze.
            X: Feature matrix.
            y: Target vector.
            param_name: Name of parameter to vary.
            param_range: Range of parameter values.
            cv_folds: Number of CV folds.
            scoring: Scoring metric.
            n_jobs: Number of parallel jobs.
            
        Returns:
            Dictionary with validation curve results.
        """
        cv_folds = cv_folds or settings.DEFAULT_CV_FOLDS
        
        # Determine default scoring
        if scoring is None:
            scoring = self._determine_scoring_metric(model, X, y)
        
        try:
            train_scores, val_scores = validation_curve(
                model, X, y,
                param_name=param_name,
                param_range=param_range,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=n_jobs
            )
            
            # Find best parameter
            val_means = np.mean(val_scores, axis=1)
            best_idx = np.argmax(val_means)
            
            result = {
                'param_name': param_name,
                'param_range': param_range,
                'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
                'train_scores_std': np.std(train_scores, axis=1).tolist(),
                'val_scores_mean': val_means.tolist(),
                'val_scores_std': np.std(val_scores, axis=1).tolist(),
                'scoring_metric': scoring,
                'best_param_idx': int(best_idx),
                'best_param_value': param_range[best_idx],
                'best_score': float(val_means[best_idx]),
                'best_score_std': float(np.std(val_scores, axis=1)[best_idx])
            }
            
            self.logger.info(f"Validation curve computed for {param_name}, best value: {result['best_param_value']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error computing validation curve: {e}")
            return None
    
    def compute_learning_curve(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        train_sizes: Optional[np.ndarray] = None,
        cv_folds: int = None,
        scoring: str = None,
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """Compute learning curve for bias-variance analysis.
        
        Args:
            model: Model to analyze.
            X: Feature matrix.
            y: Target vector.
            train_sizes: Training set sizes to evaluate.
            cv_folds: Number of CV folds.
            scoring: Scoring metric.
            n_jobs: Number of parallel jobs.
            
        Returns:
            Dictionary with learning curve results.
        """
        from sklearn.model_selection import learning_curve
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        cv_folds = cv_folds or settings.DEFAULT_CV_FOLDS
        
        if scoring is None:
            scoring = self._determine_scoring_metric(model, X, y)
        
        try:
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y,
                train_sizes=train_sizes,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=n_jobs,
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
    
    def analyze_overfitting(
        self,
        learning_curve_results: Dict[str, Any],
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """Analyze overfitting from learning curve results.
        
        Args:
            learning_curve_results: Results from compute_learning_curve.
            threshold: Threshold for determining overfitting.
            
        Returns:
            Dictionary with overfitting analysis.
        """
        if not learning_curve_results:
            return None
        
        train_scores = np.array(learning_curve_results['train_scores_mean'])
        val_scores = np.array(learning_curve_results['val_scores_mean'])
        
        # Calculate gap between training and validation scores
        gap = train_scores - val_scores
        final_gap = gap[-1]  # Gap at largest training size
        
        # Determine if overfitting
        is_overfitting = final_gap > threshold
        
        # Find point where overfitting starts (if any)
        overfitting_start = None
        for i, g in enumerate(gap):
            if g > threshold:
                overfitting_start = i
                break
        
        return {
            'is_overfitting': is_overfitting,
            'final_gap': float(final_gap),
            'max_gap': float(np.max(gap)),
            'overfitting_threshold': threshold,
            'overfitting_start_idx': overfitting_start,
            'gap_trend': 'increasing' if gap[-1] > gap[0] else 'decreasing'
        }
    
    def _determine_scoring_metric(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> str:
        """Determine appropriate scoring metric based on model and data."""
        # Check if model has predict_proba (likely classifier)
        if hasattr(model, 'predict_proba'):
            return 'accuracy'
        
        # Try to determine if regression or classification
        try:
            # Fit on small sample to check output type
            sample_size = min(100, len(X))
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X[:sample_size], y[:sample_size])
            pred = model_copy.predict(X[:sample_size])
            
            # Check if predictions are likely classification (discrete) or regression (continuous)
            unique_pred = np.unique(pred)
            if len(unique_pred) <= 10 and np.all(pred == pred.astype(int)):
                return 'accuracy'  # Classification
            else:
                return 'r2'  # Regression
        except Exception:
            # Default fallback
            return 'accuracy'