"""Shared utilities for evaluation package."""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Callable
from sklearn.metrics import accuracy_score, mean_squared_error


def ensure_binary_classification(y_true: np.ndarray, y_pred_proba: np.ndarray) -> bool:
    """Ensure inputs are valid for binary classification analysis.
    
    Args:
        y_true: True binary labels.
        y_pred_proba: Prediction probabilities.
        
    Returns:
        True if inputs are valid for binary classification.
    """
    return (len(np.unique(y_true)) == 2 and 
            y_pred_proba.ndim == 2 and 
            y_pred_proba.shape[1] == 2)


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value for zero denominator.
    
    Args:
        numerator: Numerator value.
        denominator: Denominator value.
        default: Default value when denominator is zero.
        
    Returns:
        Division result or default value.
    """
    return numerator / denominator if denominator != 0 else default


def bootstrap_metric(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    metric_func: Callable,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, float]:
    """Compute bootstrap confidence intervals for a metric.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        metric_func: Metric function to bootstrap.
        n_bootstrap: Number of bootstrap samples.
        random_state: Random state for reproducibility.
        
    Returns:
        Dictionary with bootstrap statistics.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    bootstrap_scores = []
    n_samples = len(y_true)
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        try:
            score = metric_func(y_true_boot, y_pred_boot)
            bootstrap_scores.append(score)
        except Exception:
            # Skip invalid bootstrap samples
            continue
    
    if not bootstrap_scores:
        return {
            'mean': np.nan,
            'std': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_valid_samples': 0
        }
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    return {
        'mean': np.mean(bootstrap_scores),
        'std': np.std(bootstrap_scores),
        'ci_lower': np.percentile(bootstrap_scores, 2.5),
        'ci_upper': np.percentile(bootstrap_scores, 97.5),
        'n_valid_samples': len(bootstrap_scores)
    }


def validate_evaluation_inputs(
    X: np.ndarray, 
    y: np.ndarray, 
    task_type: str,
    allow_empty: bool = False
) -> None:
    """Validate inputs for evaluation.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        task_type: Type of ML task.
        allow_empty: Whether to allow empty datasets.
        
    Raises:
        ValueError: If inputs are invalid.
    """
    if not allow_empty and len(X) == 0:
        raise ValueError("Empty dataset provided")
    
    if len(X) != len(y):
        raise ValueError(f"X and y must have the same length: {len(X)} vs {len(y)}")
    
    valid_tasks = ['classification', 'regression', 'clustering']
    if task_type not in valid_tasks:
        raise ValueError(f"Unknown task type: {task_type}. Must be one of {valid_tasks}")
    
    # Check for NaN values
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Input data contains NaN values")
    
    # Task-specific validation
    if task_type == 'classification':
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ValueError("Classification requires at least 2 classes")
        
        # Check if labels are reasonable for classification
        if not np.all(y == y.astype(int)):
            raise ValueError("Classification labels should be integers")


def format_metric_value(value: float, metric_name: str) -> str:
    """Format metric values for display.
    
    Args:
        value: Metric value to format.
        metric_name: Name of the metric.
        
    Returns:
        Formatted string representation.
    """
    if pd.isna(value):
        return "N/A"
    
    # Percentage metrics (0-1 range)
    percentage_metrics = ['accuracy', 'precision', 'recall', 'f1', 'r2', 'roc_auc']
    if any(metric.lower() in metric_name.lower() for metric in percentage_metrics):
        return f"{value:.4f}"
    
    # Error metrics (usually larger values)
    error_metrics = ['mse', 'rmse', 'mae', 'mape']
    if any(metric.lower() in metric_name.lower() for metric in error_metrics):
        if abs(value) < 0.001:
            return f"{value:.2e}"
        elif abs(value) < 1:
            return f"{value:.6f}"
        else:
            return f"{value:.4f}"
    
    # Default formatting
    if abs(value) < 0.001:
        return f"{value:.2e}"
    else:
        return f"{value:.4f}"


def compute_effect_size(
    group1_scores: np.ndarray, 
    group2_scores: np.ndarray,
    method: str = 'cohens_d'
) -> float:
    """Compute effect size between two groups.
    
    Args:
        group1_scores: Scores for first group.
        group2_scores: Scores for second group.
        method: Method for computing effect size.
        
    Returns:
        Effect size value.
    """
    if method == 'cohens_d':
        # Cohen's d
        mean_diff = np.mean(group1_scores) - np.mean(group2_scores)
        pooled_std = np.sqrt(
            (np.var(group1_scores, ddof=1) + np.var(group2_scores, ddof=1)) / 2
        )
        return mean_diff / pooled_std if pooled_std > 0 else 0.0
    
    elif method == 'glass_delta':
        # Glass's Δ (uses control group std)
        mean_diff = np.mean(group1_scores) - np.mean(group2_scores)
        control_std = np.std(group2_scores, ddof=1)
        return mean_diff / control_std if control_std > 0 else 0.0
    
    else:
        raise ValueError(f"Unknown effect size method: {method}")


def interpret_effect_size(effect_size: float, method: str = 'cohens_d') -> str:
    """Interpret effect size magnitude.
    
    Args:
        effect_size: Effect size value.
        method: Method used to compute effect size.
        
    Returns:
        Interpretation string.
    """
    abs_effect = abs(effect_size)
    
    if method == 'cohens_d':
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"
    
    # Default interpretation
    if abs_effect < 0.2:
        return "negligible"
    elif abs_effect < 0.5:
        return "small"
    elif abs_effect < 0.8:
        return "medium"
    else:
        return "large"


def ensure_numpy_array(data: Union[np.ndarray, list, pd.Series]) -> np.ndarray:
    """Ensure data is a numpy array.
    
    Args:
        data: Input data in various formats.
        
    Returns:
        Data as numpy array.
    """
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, pd.Series):
        return data.values
    elif isinstance(data, list):
        return np.array(data)
    else:
        return np.array(data)


def check_consistent_length(*arrays) -> None:
    """Check that all arrays have consistent first dimension.
    
    Args:
        *arrays: Variable number of arrays to check.
        
    Raises:
        ValueError: If arrays have inconsistent lengths.
    """
    lengths = [len(arr) for arr in arrays if arr is not None]
    
    if len(set(lengths)) > 1:
        raise ValueError(f"Inconsistent array lengths: {lengths}")


def safe_metric_computation(
    metric_func: Callable,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    default_value: float = np.nan,
    **kwargs
) -> float:
    """Safely compute a metric with error handling.
    
    Args:
        metric_func: Metric function to compute.
        y_true: True labels.
        y_pred: Predicted labels.
        default_value: Value to return on error.
        **kwargs: Additional arguments for metric function.
        
    Returns:
        Metric value or default value on error.
    """
    try:
        return metric_func(y_true, y_pred, **kwargs)
    except Exception:
        return default_value


def create_evaluation_summary_dict(
    model_name: str,
    task_type: str,
    metrics: Dict[str, float],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a standardized evaluation summary dictionary.
    
    Args:
        model_name: Name of the model.
        task_type: Type of ML task.
        metrics: Dictionary of computed metrics.
        metadata: Additional metadata.
        
    Returns:
        Standardized summary dictionary.
    """
    summary = {
        'model_name': model_name,
        'task_type': task_type,
        'metrics': metrics,
        'evaluation_timestamp': pd.Timestamp.now().isoformat(),
        'n_metrics': len(metrics)
    }
    
    if metadata:
        summary['metadata'] = metadata
    
    return summary


def filter_valid_scores(scores: np.ndarray) -> np.ndarray:
    """Filter out invalid scores (NaN, inf) from array.
    
    Args:
        scores: Array of scores.
        
    Returns:
        Array with only valid scores.
    """
    return scores[np.isfinite(scores)]


def compute_metric_stability(scores: np.ndarray) -> Dict[str, float]:
    """Compute stability metrics for a set of scores.
    
    Args:
        scores: Array of scores (e.g., from cross-validation).
        
    Returns:
        Dictionary with stability metrics.
    """
    valid_scores = filter_valid_scores(scores)
    
    if len(valid_scores) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'cv': np.nan,
            'min': np.nan,
            'max': np.nan,
            'range': np.nan
        }
    
    mean_score = np.mean(valid_scores)
    std_score = np.std(valid_scores)
    
    return {
        'mean': mean_score,
        'std': std_score,
        'cv': std_score / abs(mean_score) if mean_score != 0 else np.inf,  # Coefficient of variation
        'min': np.min(valid_scores),
        'max': np.max(valid_scores),
        'range': np.max(valid_scores) - np.min(valid_scores)
    }


def rank_models_by_metric(
    results: List[Dict[str, Any]], 
    metric_name: str,
    ascending: bool = False
) -> List[Dict[str, Any]]:
    """Rank models by a specific metric.
    
    Args:
        results: List of evaluation results.
        metric_name: Name of metric to rank by.
        ascending: Whether to rank in ascending order.
        
    Returns:
        Sorted list of results.
    """
    def get_metric_value(result):
        # Try different possible locations for the metric
        for key_path in [metric_name, f'test_{metric_name}', f'metrics.{metric_name}']:
            value = result
            for key in key_path.split('.'):
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    value = None
                    break
            
            if value is not None and not pd.isna(value):
                return value
        
        return -np.inf if not ascending else np.inf  # Put missing values at end
    
    return sorted(results, key=get_metric_value, reverse=not ascending)