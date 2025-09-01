"""Advanced pipeline factory for creating optimized ML pipelines."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.base import BaseEstimator

from .custom_transformers import (
    OutlierRemover, FeatureInteractionCreator, DomainSpecificEncoder,
    AdvancedImputer, FeatureScaler, PipelineDebugger
)
from .model_selection import AdvancedModelSelector
from ..config.settings import settings
from ..config.logging_config import LoggerMixin


class PipelineFactory(LoggerMixin):
    """Factory for creating optimized ML pipelines based on data characteristics."""
    
    def __init__(self, random_state: int = None):
        """Initialize pipeline factory.
        
        Args:
            random_state: Random state for reproducibility.
        """
        self.random_state = random_state or settings.RANDOM_SEED
        
    def create_adaptive_pipeline(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
        task_type: str = 'auto',
        complexity_level: str = 'medium',
        include_feature_engineering: bool = True,
        include_outlier_removal: bool = True,
        debug_mode: bool = False
    ) -> Pipeline:
        """Create an adaptive pipeline based on data characteristics.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            task_type: Type of task ('auto', 'classification', 'regression').
            complexity_level: Complexity level ('simple', 'medium', 'advanced').
            include_feature_engineering: Whether to include feature engineering.
            include_outlier_removal: Whether to include outlier removal.
            debug_mode: Whether to include debug transformers.
            
        Returns:
            Configured pipeline.
        """
        self.logger.info(f"Creating adaptive pipeline with complexity: {complexity_level}")
        
        # Analyze data characteristics
        data_profile = self._profile_data(X, y)
        
        # Auto-detect task type
        if task_type == 'auto':
            task_type = self._detect_task_type(y) if y is not None else 'unsupervised'
        
        # Build pipeline steps
        steps = []
        
        # Debug step (if enabled)
        if debug_mode:
            steps.append(('debug_input', PipelineDebugger('Input Data')))
        
        # Data preprocessing
        preprocessing_steps = self._create_preprocessing_steps(
            data_profile, include_outlier_removal, debug_mode
        )
        steps.extend(preprocessing_steps)
        
        # Feature engineering
        if include_feature_engineering:
            feature_steps = self._create_feature_engineering_steps(
                data_profile, complexity_level, task_type, debug_mode
            )
            steps.extend(feature_steps)
        
        # Feature selection
        if complexity_level in ['medium', 'advanced']:
            selection_step = self._create_feature_selection_step(task_type, data_profile)
            if selection_step:
                steps.append(selection_step)
        
        # Final scaling
        steps.append(('scaler', FeatureScaler(strategy='auto')))
        
        # Final debug step
        if debug_mode:
            steps.append(('debug_final', PipelineDebugger('Final Features')))
        
        pipeline = Pipeline(steps)
        
        self.logger.info(f"Created pipeline with {len(steps)} steps")
        return pipeline
    
    def create_classification_pipeline(
        self,
        algorithm: str = 'random_forest',
        preprocessing_level: str = 'standard',
        feature_selection: bool = True
    ) -> Pipeline:
        """Create classification pipeline.
        
        Args:
            algorithm: Classification algorithm to use.
            preprocessing_level: Level of preprocessing ('minimal', 'standard', 'advanced').
            feature_selection: Whether to include feature selection.
            
        Returns:
            Classification pipeline.
        """
        steps = []
        
        # Preprocessing based on level
        if preprocessing_level == 'minimal':
            steps.extend([
                ('imputer', AdvancedImputer()),
                ('scaler', StandardScaler())
            ])
        elif preprocessing_level == 'standard':
            steps.extend([
                ('imputer', AdvancedImputer()),
                ('outlier_removal', OutlierRemover()),
                ('scaler', FeatureScaler())
            ])
        else:  # advanced
            steps.extend([
                ('imputer', AdvancedImputer()),
                ('outlier_removal', OutlierRemover()),
                ('feature_interactions', FeatureInteractionCreator()),
                ('scaler', FeatureScaler())
            ])
        
        # Feature selection
        if feature_selection:
            steps.append(('feature_selection', SelectKBest(score_func=f_classif, k=50)))
        
        # Classifier
        classifier = self._get_classifier(algorithm)
        steps.append(('classifier', classifier))
        
        return Pipeline(steps)
    
    def create_regression_pipeline(
        self,
        algorithm: str = 'random_forest',
        preprocessing_level: str = 'standard',
        feature_selection: bool = True
    ) -> Pipeline:
        """Create regression pipeline.
        
        Args:
            algorithm: Regression algorithm to use.
            preprocessing_level: Level of preprocessing.
            feature_selection: Whether to include feature selection.
            
        Returns:
            Regression pipeline.
        """
        steps = []
        
        # Preprocessing based on level
        if preprocessing_level == 'minimal':
            steps.extend([
                ('imputer', AdvancedImputer()),
                ('scaler', StandardScaler())
            ])
        elif preprocessing_level == 'standard':
            steps.extend([
                ('imputer', AdvancedImputer()),
                ('outlier_removal', OutlierRemover()),
                ('scaler', FeatureScaler())
            ])
        else:  # advanced
            steps.extend([
                ('imputer', AdvancedImputer()),
                ('outlier_removal', OutlierRemover()),
                ('feature_interactions', FeatureInteractionCreator()),
                ('scaler', FeatureScaler())
            ])
        
        # Feature selection
        if feature_selection:
            steps.append(('feature_selection', SelectKBest(score_func=f_regression, k=50)))
        
        # Regressor
        regressor = self._get_regressor(algorithm)
        steps.append(('regressor', regressor))
        
        return Pipeline(steps)
    
    def create_pipeline_with_auto_tuning(
        self,
        algorithm: str,
        task_type: str,
        preprocessing_level: str = 'standard'
    ) -> Pipeline:
        """Create pipeline with automatic hyperparameter tuning.
        
        Args:
            algorithm: Algorithm to use.
            task_type: Type of task ('classification', 'regression').
            preprocessing_level: Level of preprocessing.
            
        Returns:
            Pipeline with hyperparameter tuning.
        """
        # Create base pipeline
        if task_type == 'classification':
            pipeline = self.create_classification_pipeline(algorithm, preprocessing_level)
        else:
            pipeline = self.create_regression_pipeline(algorithm, preprocessing_level)
        
        # Wrap with model selector for auto-tuning
        models = {algorithm: pipeline}
        param_grids = self._get_param_grids(algorithm, task_type)
        
        selector = AdvancedModelSelector(
            cv_strategy='stratified' if task_type == 'classification' else 'kfold',
            scoring='accuracy' if task_type == 'classification' else 'r2',
            random_state=self.random_state
        )
        
        # This would be wrapped in a meta-estimator for auto-tuning
        return pipeline  # Simplified for now
    
    def _profile_data(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Profile data characteristics for pipeline adaptation."""
        profile = {}
        
        if isinstance(X, pd.DataFrame):
            profile['n_samples'], profile['n_features'] = X.shape
            profile['feature_types'] = {
                'numerical': len(X.select_dtypes(include=[np.number]).columns),
                'categorical': len(X.select_dtypes(include=['object', 'category']).columns)
            }
            profile['missing_values'] = X.isnull().sum().sum()
            profile['duplicate_rows'] = X.duplicated().sum()
        else:
            profile['n_samples'], profile['n_features'] = X.shape
            profile['feature_types'] = {'numerical': X.shape[1], 'categorical': 0}
            profile['missing_values'] = np.isnan(X).sum() if X.dtype.kind == 'f' else 0
            profile['duplicate_rows'] = 0
        
        # Data size categories
        if profile['n_samples'] < 1000:
            profile['size_category'] = 'small'
        elif profile['n_samples'] < 10000:
            profile['size_category'] = 'medium'
        else:
            profile['size_category'] = 'large'
        
        # Dimensionality
        if profile['n_features'] > 100:
            profile['dimensionality'] = 'high'
        elif profile['n_features'] > 20:
            profile['dimensionality'] = 'medium'
        else:
            profile['dimensionality'] = 'low'
        
        return profile
    
    def _detect_task_type(self, y: np.ndarray) -> str:
        """Detect task type from target variable."""
        unique_values = len(np.unique(y))
        
        if unique_values <= 20 and (y.dtype.kind in 'biO' or unique_values < len(y) * 0.1):
            return 'classification'
        else:
            return 'regression'
    
    def _create_preprocessing_steps(
        self, 
        data_profile: Dict[str, Any], 
        include_outlier_removal: bool,
        debug_mode: bool
    ) -> List[Tuple[str, BaseEstimator]]:
        """Create preprocessing steps based on data profile."""
        steps = []
        
        # Imputation (always needed)
        steps.append(('imputer', AdvancedImputer()))
        
        if debug_mode:
            steps.append(('debug_post_imputation', PipelineDebugger('Post-Imputation')))
        
        # Outlier removal for larger datasets
        if include_outlier_removal and data_profile['size_category'] != 'small':
            steps.append(('outlier_removal', OutlierRemover()))
            
            if debug_mode:
                steps.append(('debug_post_outliers', PipelineDebugger('Post-Outlier Removal')))
        
        return steps
    
    def _create_feature_engineering_steps(
        self,
        data_profile: Dict[str, Any],
        complexity_level: str,
        task_type: str,
        debug_mode: bool
    ) -> List[Tuple[str, BaseEstimator]]:
        """Create feature engineering steps."""
        steps = []
        
        # Domain-specific encoding for categorical features
        if data_profile['feature_types']['categorical'] > 0:
            steps.append(('encoder', DomainSpecificEncoder()))
            
            if debug_mode:
                steps.append(('debug_post_encoding', PipelineDebugger('Post-Encoding')))
        
        # Feature interactions for medium/advanced complexity
        if complexity_level in ['medium', 'advanced'] and data_profile['dimensionality'] != 'high':
            max_features = min(100, data_profile['n_features'] * 3)
            steps.append(('interactions', FeatureInteractionCreator(max_features=max_features)))
            
            if debug_mode:
                steps.append(('debug_post_interactions', PipelineDebugger('Post-Interactions')))
        
        return steps
    
    def _create_feature_selection_step(
        self, 
        task_type: str, 
        data_profile: Dict[str, Any]
    ) -> Optional[Tuple[str, BaseEstimator]]:
        """Create feature selection step if needed."""
        # Only use feature selection for high-dimensional data
        if data_profile['dimensionality'] == 'high':
            if task_type == 'classification':
                selector = SelectKBest(score_func=f_classif, k=min(50, data_profile['n_features'] // 2))
            else:
                selector = SelectKBest(score_func=f_regression, k=min(50, data_profile['n_features'] // 2))
            
            return ('feature_selection', selector)
        
        return None
    
    def _get_classifier(self, algorithm: str) -> BaseEstimator:
        """Get classifier instance."""
        if algorithm == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        elif algorithm == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(random_state=self.random_state)
        elif algorithm == 'logistic':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(random_state=self.random_state, max_iter=1000)
        elif algorithm == 'svm':
            from sklearn.svm import SVC
            return SVC(random_state=self.random_state, probability=True)
        else:
            raise ValueError(f"Unknown classifier: {algorithm}")
    
    def _get_regressor(self, algorithm: str) -> BaseEstimator:
        """Get regressor instance."""
        if algorithm == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        elif algorithm == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(random_state=self.random_state)
        elif algorithm == 'linear':
            from sklearn.linear_model import LinearRegression
            return LinearRegression()
        elif algorithm == 'ridge':
            from sklearn.linear_model import Ridge
            return Ridge(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown regressor: {algorithm}")
    
    def _get_param_grids(self, algorithm: str, task_type: str) -> Dict[str, Dict]:
        """Get parameter grids for hyperparameter tuning."""
        if task_type == 'classification':
            if algorithm == 'random_forest':
                return {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__max_depth': [5, 10, None],
                    'classifier__min_samples_split': [2, 5, 10]
                }
            elif algorithm == 'svm':
                return {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__gamma': ['scale', 'auto', 0.001, 0.01]
                }
        else:  # regression
            if algorithm == 'random_forest':
                return {
                    'regressor__n_estimators': [50, 100, 200],
                    'regressor__max_depth': [5, 10, None],
                    'regressor__min_samples_split': [2, 5, 10]
                }
            elif algorithm == 'ridge':
                return {
                    'regressor__alpha': [0.1, 1.0, 10.0, 100.0]
                }
        
        return {}


# Export classes
__all__ = ['PipelineFactory']