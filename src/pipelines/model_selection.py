"""Advanced model selection utilities and strategies."""

import sys
from pathlib import Path

# Handle imports that work in both package and direct import contexts
try:
    from ..config.settings import settings, ModelDefaults
    from ..config.logging_config import LoggerMixin
except ImportError:
    # Fallback for direct imports outside package context
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.settings import settings, ModelDefaults
    from config.logging_config import LoggerMixin

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score, 
    StratifiedKFold, KFold, TimeSeriesSplit, validation_curve,
    learning_curve, cross_validate
)
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import make_scorer
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Optional dependency
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


class AdvancedModelSelector(LoggerMixin):
    """Advanced model selection with multiple optimization strategies."""
    
    def __init__(
        self,
        cv_strategy: str = 'stratified',
        cv_folds: int = 5,
        scoring: Union[str, Callable] = 'accuracy',
        random_state: int = None,
        n_jobs: int = -1
    ):
        """Initialize advanced model selector.
        
        Args:
            cv_strategy: Cross-validation strategy ('stratified', 'kfold', 'timeseries').
            cv_folds: Number of cross-validation folds.
            scoring: Scoring metric or callable.
            random_state: Random state for reproducibility.
            n_jobs: Number of parallel jobs.
        """
        self.cv_strategy = cv_strategy
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state or settings.RANDOM_SEED
        self.n_jobs = n_jobs
        
        self.cv_splitter_ = self._get_cv_splitter()
        self.results_ = {}
        
    def select_best_model(
        self,
        models: Dict[str, BaseEstimator],
        X: np.ndarray,
        y: np.ndarray,
        param_grids: Optional[Dict[str, Dict]] = None
    ) -> Tuple[str, BaseEstimator, Dict[str, Any]]:
        """Select the best model from a collection of candidates.
        
        Args:
            models: Dictionary of model name to estimator.
            X: Feature matrix.
            y: Target vector.
            param_grids: Parameter grids for each model.
            
        Returns:
            Tuple of (best_model_name, best_estimator, results).
        """
        self.logger.info(f"Selecting best model from {len(models)} candidates")
        
        if param_grids is None:
            param_grids = {}
        
        model_scores = {}
        model_details = {}
        
        for model_name, model in models.items():
            self.logger.info(f"Evaluating model: {model_name}")
            
            try:
                # Get parameter grid for this model
                param_grid = param_grids.get(model_name, {})
                
                if param_grid:
                    # Hyperparameter optimization
                    optimized_model, best_params, best_score = self._optimize_hyperparameters(
                        model, X, y, param_grid, method='grid_search'
                    )
                    model_scores[model_name] = best_score
                    model_details[model_name] = {
                        'model': optimized_model,
                        'best_params': best_params,
                        'best_score': best_score
                    }
                else:
                    # Just cross-validate with default parameters
                    scores = cross_val_score(
                        model, X, y, cv=self.cv_splitter_, 
                        scoring=self.scoring, n_jobs=self.n_jobs
                    )
                    mean_score = np.mean(scores)
                    model_scores[model_name] = mean_score
                    model_details[model_name] = {
                        'model': model,
                        'best_params': {},
                        'best_score': mean_score,
                        'cv_scores': scores
                    }
                
                self.logger.info(f"{model_name}: CV score = {model_scores[model_name]:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
                model_scores[model_name] = -np.inf
                model_details[model_name] = {'error': str(e)}
        
        # Select best model
        best_model_name = max(model_scores, key=model_scores.get)
        best_model_details = model_details[best_model_name]
        
        self.results_ = {
            'model_scores': model_scores,
            'model_details': model_details,
            'best_model': best_model_name
        }
        
        self.logger.info(f"Best model: {best_model_name} (score: {model_scores[best_model_name]:.4f})")
        
        return best_model_name, best_model_details['model'], self.results_
    
    def optimize_hyperparameters(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict[str, List],
        method: str = 'grid_search',
        n_trials: int = 100
    ) -> Tuple[BaseEstimator, Dict[str, Any], float]:
        """Optimize hyperparameters for a single model.
        
        Args:
            model: Model to optimize.
            X: Feature matrix.
            y: Target vector.
            param_grid: Parameter grid or space.
            method: Optimization method ('grid_search', 'random_search', 'optuna').
            n_trials: Number of trials for random/optuna search.
            
        Returns:
            Tuple of (optimized_model, best_params, best_score).
        """
        return self._optimize_hyperparameters(model, X, y, param_grid, method, n_trials)
    
    def _optimize_hyperparameters(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict[str, List],
        method: str = 'grid_search',
        n_trials: int = 100
    ) -> Tuple[BaseEstimator, Dict[str, Any], float]:
        """Internal hyperparameter optimization."""
        
        if method == 'grid_search':
            search = GridSearchCV(
                model, param_grid, cv=self.cv_splitter_,
                scoring=self.scoring, n_jobs=self.n_jobs, verbose=0
            )
            search.fit(X, y)
            
            return search.best_estimator_, search.best_params_, search.best_score_
            
        elif method == 'random_search':
            search = RandomizedSearchCV(
                model, param_grid, cv=self.cv_splitter_,
                scoring=self.scoring, n_jobs=self.n_jobs,
                n_iter=n_trials, random_state=self.random_state, verbose=0
            )
            search.fit(X, y)
            
            return search.best_estimator_, search.best_params_, search.best_score_
            
        elif method == 'optuna':
            return self._optuna_optimization(model, X, y, param_grid, n_trials)
            
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _optuna_optimization(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        param_space: Dict[str, Any],
        n_trials: int
    ) -> Tuple[BaseEstimator, Dict[str, Any], float]:
        """Optimize hyperparameters using Optuna."""
        
        if not HAS_OPTUNA:
            raise ImportError("Optuna is required for 'optuna' optimization method. Install it with: pip install optuna")
        
        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_config in param_space.items():
                if isinstance(param_config, dict):
                    if param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
                    elif param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
                    elif param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
                    elif param_config['type'] == 'loguniform':
                        params[param_name] = trial.suggest_loguniform(param_name, param_config['low'], param_config['high'])
                else:
                    # Assume it's a list of values (categorical)
                    params[param_name] = trial.suggest_categorical(param_name, param_config)
            
            # Create model with sampled parameters
            model_instance = clone(model)
            model_instance.set_params(**params)
            
            # Cross-validate
            scores = cross_val_score(
                model_instance, X, y, cv=self.cv_splitter_,
                scoring=self.scoring, n_jobs=1  # Avoid nested parallelism
            )
            
            return np.mean(scores)
        
        # Create study
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # Get best parameters and create best model
        best_params = study.best_params
        best_model = clone(model)
        best_model.set_params(**best_params)
        best_model.fit(X, y)
        
        return best_model, best_params, study.best_value
    
    def _get_cv_splitter(self):
        """Get cross-validation splitter based on strategy."""
        if self.cv_strategy == 'stratified':
            return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        elif self.cv_strategy == 'kfold':
            return KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        elif self.cv_strategy == 'timeseries':
            return TimeSeriesSplit(n_splits=self.cv_folds)
        else:
            raise ValueError(f"Unknown CV strategy: {self.cv_strategy}")
    
    def get_model_comparison_report(self) -> pd.DataFrame:
        """Get detailed model comparison report.
        
        Returns:
            DataFrame with model comparison results.
        """
        if not self.results_:
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, details in self.results_['model_details'].items():
            if 'error' in details:
                continue
                
            row = {
                'model': model_name,
                'best_score': details['best_score'],
                'is_best': model_name == self.results_['best_model']
            }
            
            # Add best parameters (top 3 most important)
            best_params = details.get('best_params', {})
            for i, (param, value) in enumerate(list(best_params.items())[:3]):
                row[f'param_{i+1}'] = f"{param}={value}"
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        if not df.empty:
            df = df.sort_values('best_score', ascending=False)
        
        return df


class MultiObjectiveSelector(LoggerMixin):
    """Multi-objective model selection considering multiple metrics."""
    
    def __init__(
        self,
        metrics: List[str],
        weights: Optional[List[float]] = None,
        cv_folds: int = 5,
        random_state: int = None
    ):
        """Initialize multi-objective selector.
        
        Args:
            metrics: List of metrics to optimize.
            weights: Weights for each metric (if None, uses equal weights).
            cv_folds: Number of CV folds.
            random_state: Random state.
        """
        self.metrics = metrics
        self.weights = weights or [1.0] * len(metrics)
        self.cv_folds = cv_folds
        self.random_state = random_state or settings.RANDOM_SEED
        
        if len(self.weights) != len(self.metrics):
            raise ValueError("Number of weights must match number of metrics")
        
        self.cv_splitter_ = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
    def select_pareto_optimal_models(
        self,
        models: Dict[str, BaseEstimator],
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Select Pareto optimal models considering multiple objectives.
        
        Args:
            models: Dictionary of models to evaluate.
            X: Feature matrix.
            y: Target vector.
            
        Returns:
            Dictionary with Pareto optimal results.
        """
        self.logger.info(f"Multi-objective selection with metrics: {self.metrics}")
        
        model_scores = {}
        
        for model_name, model in models.items():
            self.logger.info(f"Evaluating {model_name}")
            
            scores = {}
            
            for metric in self.metrics:
                try:
                    cv_scores = cross_val_score(
                        model, X, y, cv=self.cv_splitter_,
                        scoring=metric, n_jobs=-1
                    )
                    scores[metric] = np.mean(cv_scores)
                except Exception as e:
                    self.logger.warning(f"Could not compute {metric} for {model_name}: {e}")
                    scores[metric] = -np.inf
            
            model_scores[model_name] = scores
        
        # Calculate weighted scores
        weighted_scores = {}
        for model_name, scores in model_scores.items():
            weighted_score = sum(
                scores[metric] * weight
                for metric, weight in zip(self.metrics, self.weights)
            )
            weighted_scores[model_name] = weighted_score
        
        # Find Pareto optimal models
        pareto_optimal = self._find_pareto_optimal(model_scores)
        
        # Select best overall model
        best_model = max(weighted_scores, key=weighted_scores.get)
        
        return {
            'model_scores': model_scores,
            'weighted_scores': weighted_scores,
            'pareto_optimal': pareto_optimal,
            'best_overall': best_model,
            'metrics': self.metrics,
            'weights': self.weights
        }
    
    def _find_pareto_optimal(self, model_scores: Dict[str, Dict[str, float]]) -> List[str]:
        """Find Pareto optimal models."""
        models = list(model_scores.keys())
        pareto_optimal = []
        
        for i, model1 in enumerate(models):
            is_dominated = False
            
            for j, model2 in enumerate(models):
                if i == j:
                    continue
                
                # Check if model1 is dominated by model2
                dominates = True
                strictly_better = False
                
                for metric in self.metrics:
                    score1 = model_scores[model1][metric]
                    score2 = model_scores[model2][metric]
                    
                    if score1 > score2:
                        dominates = False
                        break
                    elif score2 > score1:
                        strictly_better = True
                
                if dominates and strictly_better:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(model1)
        
        return pareto_optimal


class NestedCrossValidation(LoggerMixin):
    """Nested cross-validation for unbiased model selection and evaluation."""
    
    def __init__(
        self,
        outer_cv: int = 5,
        inner_cv: int = 3,
        random_state: int = None
    ):
        """Initialize nested cross-validation.
        
        Args:
            outer_cv: Number of outer CV folds.
            inner_cv: Number of inner CV folds.
            random_state: Random state.
        """
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.random_state = random_state or settings.RANDOM_SEED
        
    def evaluate_models(
        self,
        models: Dict[str, BaseEstimator],
        X: np.ndarray,
        y: np.ndarray,
        param_grids: Optional[Dict[str, Dict]] = None,
        scoring: str = 'accuracy'
    ) -> Dict[str, Any]:
        """Perform nested cross-validation for multiple models.
        
        Args:
            models: Dictionary of models to evaluate.
            X: Feature matrix.
            y: Target vector.
            param_grids: Parameter grids for hyperparameter optimization.
            scoring: Scoring metric.
            
        Returns:
            Nested CV results.
        """
        self.logger.info("Starting nested cross-validation")
        
        if param_grids is None:
            param_grids = {}
        
        # Outer CV splitter
        outer_cv_splitter = StratifiedKFold(
            n_splits=self.outer_cv, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        # Inner CV splitter
        inner_cv_splitter = StratifiedKFold(
            n_splits=self.inner_cv, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        nested_scores = {model_name: [] for model_name in models.keys()}
        selected_models = {model_name: [] for model_name in models.keys()}
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv_splitter.split(X, y)):
            self.logger.info(f"Outer fold {fold_idx + 1}/{self.outer_cv}")
            
            X_train_outer, X_test_outer = X[train_idx], X[test_idx]
            y_train_outer, y_test_outer = y[train_idx], y[test_idx]
            
            # For each model, perform inner CV for hyperparameter optimization
            for model_name, model in models.items():
                param_grid = param_grids.get(model_name, {})
                
                if param_grid:
                    # Hyperparameter optimization on inner train set
                    search = GridSearchCV(
                        model, param_grid,
                        cv=inner_cv_splitter,
                        scoring=scoring,
                        n_jobs=-1
                    )
                    search.fit(X_train_outer, y_train_outer)
                    
                    # Best model from inner CV
                    best_model = search.best_estimator_
                    selected_models[model_name].append(search.best_params_)
                else:
                    # No hyperparameter optimization
                    best_model = clone(model)
                    best_model.fit(X_train_outer, y_train_outer)
                    selected_models[model_name].append({})
                
                # Evaluate on outer test set
                if scoring == 'accuracy':
                    from sklearn.metrics import accuracy_score
                    y_pred = best_model.predict(X_test_outer)
                    score = accuracy_score(y_test_outer, y_pred)
                elif scoring == 'r2':
                    from sklearn.metrics import r2_score
                    y_pred = best_model.predict(X_test_outer)
                    score = r2_score(y_test_outer, y_pred)
                else:
                    # Use cross_val_score with single fold
                    score = cross_val_score(
                        best_model, X_test_outer, y_test_outer,
                        cv=2, scoring=scoring
                    ).mean()
                
                nested_scores[model_name].append(score)
        
        # Calculate final statistics
        results = {}
        for model_name in models.keys():
            scores = nested_scores[model_name]
            results[model_name] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'scores': scores,
                'selected_params': selected_models[model_name]
            }
        
        # Find best model
        best_model = max(results.keys(), key=lambda x: results[x]['mean_score'])
        
        return {
            'results': results,
            'best_model': best_model,
            'outer_cv_folds': self.outer_cv,
            'inner_cv_folds': self.inner_cv
        }


class LearningCurveAnalyzer(LoggerMixin):
    """Analyze learning curves for bias-variance analysis."""
    
    def __init__(self, cv_folds: int = 5, random_state: int = None):
        """Initialize learning curve analyzer.
        
        Args:
            cv_folds: Number of CV folds.
            random_state: Random state.
        """
        self.cv_folds = cv_folds
        self.random_state = random_state or settings.RANDOM_SEED
        
    def analyze_learning_curve(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        train_sizes: Optional[np.ndarray] = None,
        scoring: str = 'accuracy'
    ) -> Dict[str, Any]:
        """Analyze learning curve for bias-variance tradeoff.
        
        Args:
            model: Model to analyze.
            X: Feature matrix.
            y: Target vector.
            train_sizes: Training set sizes to evaluate.
            scoring: Scoring metric.
            
        Returns:
            Learning curve analysis results.
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        # Compute learning curve
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=self.cv_folds,
            scoring=scoring,
            n_jobs=-1,
            random_state=self.random_state
        )
        
        # Calculate statistics
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)
        
        # Analyze bias-variance
        final_train_score = train_scores_mean[-1]
        final_val_score = val_scores_mean[-1]
        overfitting_gap = final_train_score - final_val_score
        
        # Determine if model has high bias or high variance
        if final_val_score < 0.7:  # Arbitrary threshold
            bias_level = "high"
        elif final_val_score < 0.85:
            bias_level = "medium"
        else:
            bias_level = "low"
        
        if overfitting_gap > 0.1:  # Arbitrary threshold
            variance_level = "high"
        elif overfitting_gap > 0.05:
            variance_level = "medium"
        else:
            variance_level = "low"
        
        # Generate recommendations
        recommendations = []
        if bias_level == "high":
            recommendations.append("Consider using a more complex model or adding features")
        if variance_level == "high":
            recommendations.append("Consider regularization, more data, or simpler model")
        if len(recommendations) == 0:
            recommendations.append("Model appears well-balanced")
        
        return {
            'train_sizes': train_sizes_abs.tolist(),
            'train_scores_mean': train_scores_mean.tolist(),
            'train_scores_std': train_scores_std.tolist(),
            'val_scores_mean': val_scores_mean.tolist(),
            'val_scores_std': val_scores_std.tolist(),
            'final_train_score': float(final_train_score),
            'final_val_score': float(final_val_score),
            'overfitting_gap': float(overfitting_gap),
            'bias_level': bias_level,
            'variance_level': variance_level,
            'recommendations': recommendations,
            'scoring_metric': scoring
        }
    
    def compare_models_learning_curves(
        self,
        models: Dict[str, BaseEstimator],
        X: np.ndarray,
        y: np.ndarray,
        scoring: str = 'accuracy'
    ) -> Dict[str, Dict[str, Any]]:
        """Compare learning curves of multiple models.
        
        Args:
            models: Dictionary of models to compare.
            X: Feature matrix.
            y: Target vector.
            scoring: Scoring metric.
            
        Returns:
            Dictionary with learning curve analysis for each model.
        """
        results = {}
        
        for model_name, model in models.items():
            self.logger.info(f"Analyzing learning curve for {model_name}")
            try:
                results[model_name] = self.analyze_learning_curve(model, X, y, scoring=scoring)
            except Exception as e:
                self.logger.error(f"Error analyzing {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results


class ValidationCurveAnalyzer(LoggerMixin):
    """Analyze validation curves for hyperparameter sensitivity."""
    
    def __init__(self, cv_folds: int = 5, random_state: int = None):
        """Initialize validation curve analyzer.
        
        Args:
            cv_folds: Number of CV folds.
            random_state: Random state.
        """
        self.cv_folds = cv_folds
        self.random_state = random_state or settings.RANDOM_SEED
        
    def analyze_validation_curve(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        param_name: str,
        param_range: List,
        scoring: str = 'accuracy'
    ) -> Dict[str, Any]:
        """Analyze validation curve for a specific hyperparameter.
        
        Args:
            model: Model to analyze.
            X: Feature matrix.
            y: Target vector.
            param_name: Name of parameter to vary.
            param_range: Range of parameter values.
            scoring: Scoring metric.
            
        Returns:
            Validation curve analysis results.
        """
        # Compute validation curve
        train_scores, val_scores = validation_curve(
            model, X, y,
            param_name=param_name,
            param_range=param_range,
            cv=self.cv_folds,
            scoring=scoring,
            n_jobs=-1
        )
        
        # Calculate statistics
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)
        
        # Find optimal parameter value
        best_idx = np.argmax(val_scores_mean)
        best_param_value = param_range[best_idx]
        best_score = val_scores_mean[best_idx]
        
        # Analyze parameter sensitivity
        score_range = np.max(val_scores_mean) - np.min(val_scores_mean)
        if score_range > 0.1:
            sensitivity = "high"
        elif score_range > 0.05:
            sensitivity = "medium"
        else:
            sensitivity = "low"
        
        return {
            'param_name': param_name,
            'param_range': param_range,
            'train_scores_mean': train_scores_mean.tolist(),
            'train_scores_std': train_scores_std.tolist(),
            'val_scores_mean': val_scores_mean.tolist(),
            'val_scores_std': val_scores_std.tolist(),
            'best_param_value': best_param_value,
            'best_score': float(best_score),
            'sensitivity': sensitivity,
            'score_range': float(score_range),
            'scoring_metric': scoring
        }


class AutoMLSelector(LoggerMixin):
    """Automated machine learning model selection."""
    
    def __init__(
        self,
        time_budget: int = 300,  # seconds
        random_state: int = None
    ):
        """Initialize AutoML selector.
        
        Args:
            time_budget: Time budget in seconds.
            random_state: Random state.
        """
        self.time_budget = time_budget
        self.random_state = random_state or settings.RANDOM_SEED
        self.start_time = None
        self.results_ = {}
        
    def auto_select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str = 'auto'
    ) -> Dict[str, Any]:
        """Automatically select best model within time budget.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            task_type: Type of task ('auto', 'classification', 'regression').
            
        Returns:
            AutoML results.
        """
        self.start_time = time.time()
        self.logger.info(f"Starting AutoML with {self.time_budget}s budget")
        
        # Detect task type if auto
        if task_type == 'auto':
            unique_values = len(np.unique(y))
            if unique_values <= 20 and unique_values < len(y) * 0.1:
                task_type = 'classification'
            else:
                task_type = 'regression'
        
        # Get candidate models
        models = self._get_candidate_models(task_type)
        
        # Progressive evaluation: start with fast models
        model_priorities = self._prioritize_models(models, X.shape)
        
        best_model = None
        best_score = -np.inf
        evaluated_models = {}
        
        for model_name in model_priorities:
            if self._time_remaining() < 10:  # Keep 10s buffer
                break
                
            model = models[model_name]
            
            try:
                # Quick evaluation first
                start_eval = time.time()
                quick_score = self._quick_evaluate(model, X, y, task_type)
                eval_time = time.time() - start_eval
                
                evaluated_models[model_name] = {
                    'quick_score': quick_score,
                    'eval_time': eval_time,
                    'model': model
                }
                
                if quick_score > best_score:
                    best_score = quick_score
                    best_model = model_name
                
                self.logger.info(f"{model_name}: {quick_score:.4f} ({eval_time:.1f}s)")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
        
        # Hyperparameter optimization for best models if time allows
        if self._time_remaining() > 30 and evaluated_models:
            top_models = sorted(
                evaluated_models.items(),
                key=lambda x: x[1]['quick_score'],
                reverse=True
            )[:3]  # Top 3 models
            
            for model_name, model_info in top_models:
                if self._time_remaining() < 20:
                    break
                    
                self.logger.info(f"Optimizing {model_name}")
                try:
                    optimized_score = self._optimize_model(
                        model_info['model'], X, y, task_type
                    )
                    
                    if optimized_score > best_score:
                        best_score = optimized_score
                        best_model = model_name
                        
                    evaluated_models[model_name]['optimized_score'] = optimized_score
                    
                except Exception as e:
                    self.logger.error(f"Error optimizing {model_name}: {e}")
        
        total_time = time.time() - self.start_time
        
        return {
            'best_model': best_model,
            'best_score': best_score,
            'task_type': task_type,
            'evaluated_models': evaluated_models,
            'total_time': total_time,
            'time_budget': self.time_budget
        }
    
    def _get_candidate_models(self, task_type: str) -> Dict[str, BaseEstimator]:
        """Get candidate models based on task type."""
        if task_type == 'classification':
            return {
                'logistic': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'random_forest': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state),
                'svm': SVC(random_state=self.random_state),
                'naive_bayes': GaussianNB()
            }
        else:  # regression
            return {
                'linear': LinearRegression(),
                'ridge': Ridge(random_state=self.random_state),
                'random_forest': RandomForestRegressor(random_state=self.random_state, n_jobs=-1),
                'gradient_boosting': GradientBoostingRegressor(random_state=self.random_state),
                'svr': SVR()
            }
    
    def _prioritize_models(self, models: Dict[str, BaseEstimator], data_shape: Tuple[int, int]) -> List[str]:
        """Prioritize models based on data characteristics and speed."""
        n_samples, n_features = data_shape
        
        # Fast models first
        priority_order = []
        
        if n_samples < 1000:
            priority_order.extend(['naive_bayes', 'logistic', 'linear'])
        
        priority_order.extend(['random_forest', 'gradient_boosting'])
        
        if n_samples > 1000:
            priority_order.extend(['svm', 'svr'])
        
        # Remove duplicates and models not in the candidate set
        final_order = []
        for model_name in priority_order:
            if model_name in models and model_name not in final_order:
                final_order.append(model_name)
        
        # Add any remaining models
        for model_name in models:
            if model_name not in final_order:
                final_order.append(model_name)
        
        return final_order
    
    def _quick_evaluate(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray, task_type: str) -> float:
        """Quick model evaluation using 3-fold CV."""
        if task_type == 'classification':
            scoring = 'accuracy'
        else:
            scoring = 'r2'
        
        scores = cross_val_score(model, X, y, cv=3, scoring=scoring, n_jobs=-1)
        return np.mean(scores)
    
    def _optimize_model(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray, task_type: str) -> float:
        """Quick hyperparameter optimization."""
        # Simple parameter grids for quick optimization
        param_grids = {
            'RandomForestClassifier': {'n_estimators': [50, 100], 'max_depth': [5, 10, None]},
            'RandomForestRegressor': {'n_estimators': [50, 100], 'max_depth': [5, 10, None]},
            'Ridge': {'alpha': [0.1, 1.0, 10.0]},
            'SVC': {'C': [0.1, 1.0, 10.0]},
            'SVR': {'C': [0.1, 1.0, 10.0]}
        }
        
        model_class = type(model).__name__
        param_grid = param_grids.get(model_class, {})
        
        if not param_grid:
            return self._quick_evaluate(model, X, y, task_type)
        
        # Quick grid search
        if task_type == 'classification':
            scoring = 'accuracy'
        else:
            scoring = 'r2'
        
        search = GridSearchCV(
            model, param_grid, cv=3, scoring=scoring, n_jobs=-1
        )
        search.fit(X, y)
        
        return search.best_score_
    
    def _time_remaining(self) -> float:
        """Calculate remaining time in budget."""
        if self.start_time is None:
            return self.time_budget
        
        elapsed = time.time() - self.start_time
        return max(0, self.time_budget - elapsed)


# Export classes
__all__ = [
    'AdvancedModelSelector',
    'MultiObjectiveSelector',
    'NestedCrossValidation',
    'LearningCurveAnalyzer',
    'ValidationCurveAnalyzer',
    'AutoMLSelector'
]

# Additional model selection classes for backward compatibility

class ModelSelectionPipeline(LoggerMixin):
    """Pipeline for model selection and evaluation."""
    
    def __init__(self, models: List[BaseEstimator], cv: int = 5):
        self.models = models
        self.cv = cv
        self.results = {}
    
    def evaluate_all(self, X, y):
        """Evaluate all models."""
        for i, model in enumerate(self.models):
            scores = cross_val_score(model, X, y, cv=self.cv, scoring='accuracy')
            self.results[f'Model_{i}'] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }
        return self.results
    
    def best_model(self):
        """Get the best model."""
        if not self.results:
            raise ValueError("No results. Call evaluate_all() first")
        best = max(self.results.items(), key=lambda x: x[1]['mean'])
        return best


class AutoModelSelector(LoggerMixin):
    """Automatically select best model."""
    
    def __init__(self, models: List[BaseEstimator]):
        self.models = models
        self.best_model = None
    
    def select(self, X, y, cv: int = 5):
        """Select best model."""
        best_score = -np.inf
        for model in self.models:
            scores = cross_val_score(model, X, y, cv=cv)
            if np.mean(scores) > best_score:
                best_score = np.mean(scores)
                self.best_model = model
        return self.best_model


class ModelComparator(LoggerMixin):
    """Compare multiple models."""
    
    def __init__(self, models: Dict[str, BaseEstimator]):
        self.models = models
        self.comparison_results = None
    
    def compare(self, X, y, cv: int = 5):
        """Compare models."""
        results = {}
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            results[name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        self.comparison_results = results
        return results


class HyperparameterOptimizer(LoggerMixin):
    """Optimize hyperparameters."""
    
    def __init__(self, model: BaseEstimator, param_grid: Dict[str, list]):
        self.model = model
        self.param_grid = param_grid
        self.best_params = None
    
    def optimize(self, X, y, cv: int = 5):
        """Optimize hyperparameters."""
        gs = GridSearchCV(self.model, self.param_grid, cv=cv, n_jobs=-1)
        gs.fit(X, y)
        self.best_params = gs.best_params_
        return gs.best_estimator_


class CrossValidationPipeline(LoggerMixin):
    """Cross-validation pipeline."""
    
    def __init__(self, model: BaseEstimator, cv: int = 5):
        self.model = model
        self.cv = cv
        self.cv_results = None
    
    def run(self, X, y):
        """Run cross-validation."""
        scores = cross_val_score(self.model, X, y, cv=self.cv, scoring='accuracy')
        self.cv_results = {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores)
        }
        return self.cv_results


class ModelEnsemblePipeline(LoggerMixin):
    """Pipeline for ensemble model selection."""
    
    def __init__(self, models: List[BaseEstimator]):
        self.models = models
    
    def create_voting_ensemble(self):
        """Create voting ensemble."""
        from sklearn.ensemble import VotingClassifier
        return VotingClassifier(
            estimators=[(f'model_{i}', m) for i, m in enumerate(self.models)],
            voting='soft'
        )


class PerformanceTracker(LoggerMixin):
    """Track model performance."""
    
    def __init__(self):
        self.performance_history = []
    
    def track(self, model_name: str, metrics: Dict[str, float]):
        """Track performance metrics."""
        self.performance_history.append({
            'model': model_name,
            'metrics': metrics,
            'timestamp': time.time()
        })
    
    def get_history(self):
        """Get performance history."""
        return self.performance_history


# Update __all__ export
__all__ = [
    'AdvancedModelSelector',
    'MultiObjectiveSelector',
    'NestedCrossValidation',
    'LearningCurveAnalyzer',
    'ValidationCurveAnalyzer',
    'AutoMLSelector',
    'ModelSelectionPipeline',
    'AutoModelSelector',
    'ModelComparator',
    'HyperparameterOptimizer',
    'CrossValidationPipeline',
    'ModelEnsemblePipeline',
    'PerformanceTracker'
]


class ModelRegistry(LoggerMixin):
    """Registry for managing models."""
    
    def __init__(self):
        self.models = {}
    
    def register(self, name: str, model: BaseEstimator):
        """Register a model."""
        self.models[name] = model
    
    def get(self, name: str) -> BaseEstimator:
        """Get a registered model."""
        return self.models.get(name)
    
    def list_models(self):
        """List all registered models."""
        return list(self.models.keys())


class BayesianOptimizer(LoggerMixin):
    """Bayesian optimization for hyperparameter tuning."""
    
    def __init__(self, model: BaseEstimator, param_space: Dict[str, list]):
        self.model = model
        self.param_space = param_space
    
    def optimize(self, X, y, n_calls: int = 10):
        """Optimize using Bayesian search."""
        # Simple implementation - in practice, would use scikit-optimize
        from sklearn.model_selection import RandomizedSearchCV
        
        search = RandomizedSearchCV(
            self.model,
            self.param_space,
            n_iter=n_calls,
            cv=5,
            n_jobs=-1,
            random_state=42
        )
        search.fit(X, y)
        return search.best_estimator_, search.best_params_


class GridSearchPipeline(LoggerMixin):
    """Pipeline for grid search."""
    
    def __init__(self, model: BaseEstimator, param_grid: Dict[str, list]):
        self.model = model
        self.param_grid = param_grid
    
    def search(self, X, y, cv: int = 5):
        """Perform grid search."""
        gs = GridSearchCV(
            self.model,
            self.param_grid,
            cv=cv,
            n_jobs=-1
        )
        gs.fit(X, y)
        return {
            'best_estimator': gs.best_estimator_,
            'best_params': gs.best_params_,
            'best_score': gs.best_score_,
            'cv_results': gs.cv_results_
        }
