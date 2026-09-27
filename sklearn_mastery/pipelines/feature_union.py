"""Advanced feature union and pipeline composition utilities."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
import warnings


class AdvancedFeatureUnion(BaseEstimator, TransformerMixin):
    """Advanced feature union with weights and flexibility."""
    
    def __init__(
        self,
        transformers: List[Tuple[str, BaseEstimator]],
        weights: Optional[List[float]] = None,
        n_jobs: int = 1
    ):
        """Initialize advanced feature union."""
        self.transformers = transformers
        self.weights = weights
        self.n_jobs = n_jobs
        self.union = FeatureUnion(
            transformers=transformers,
            weights=weights,
            n_jobs=n_jobs
        )
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the feature union."""
        self.union.fit(X, y)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform using the feature union."""
        return self.union.transform(X)
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform."""
        return self.union.fit_transform(X, y)


class ParallelFeatureProcessor(BaseEstimator, TransformerMixin):
    """Process features in parallel."""
    
    def __init__(
        self,
        transformers: List[Tuple[str, BaseEstimator]],
        n_jobs: int = -1
    ):
        """Initialize parallel feature processor."""
        self.transformers = transformers
        self.n_jobs = n_jobs
        self.union = FeatureUnion(transformers=transformers, n_jobs=n_jobs)
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit all transformers."""
        self.union.fit(X, y)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform in parallel."""
        return self.union.transform(X)


class ConditionalFeatureUnion(BaseEstimator, TransformerMixin):
    """Feature union with conditional transformation."""
    
    def __init__(
        self,
        transformers: List[Tuple[str, BaseEstimator]],
        conditions: Optional[Dict[str, callable]] = None
    ):
        """Initialize conditional feature union."""
        self.transformers = transformers
        self.conditions = conditions or {}
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit transformers."""
        for name, transformer in self.transformers:
            if name in self.conditions:
                if self.conditions[name](X, y):
                    transformer.fit(X, y)
            else:
                transformer.fit(X, y)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform with conditions."""
        transformed = []
        for name, transformer in self.transformers:
            if name not in self.conditions or self.conditions[name](X, None):
                transformed.append(transformer.transform(X))
        return np.hstack(transformed) if transformed else X


class WeightedFeatureUnion(BaseEstimator, TransformerMixin):
    """Feature union with adaptive weights."""
    
    def __init__(
        self,
        transformers: List[Tuple[str, BaseEstimator]],
        initial_weights: Optional[List[float]] = None
    ):
        """Initialize weighted feature union."""
        self.transformers = transformers
        self.initial_weights = initial_weights
        n_transformers = len(transformers)
        if initial_weights is None:
            self.weights = [1.0 / n_transformers] * n_transformers
        else:
            self.weights = initial_weights
        self.union = FeatureUnion(
            transformers=transformers,
            weights=self.weights
        )
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the union."""
        self.union.fit(X, y)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform with weights."""
        return self.union.transform(X)


class DynamicFeatureUnion(BaseEstimator, TransformerMixin):
    """Dynamically select and apply transformers."""
    
    def __init__(
        self,
        transformers: List[Tuple[str, BaseEstimator]],
        selector: Optional[callable] = None
    ):
        """Initialize dynamic feature union."""
        self.transformers = transformers
        self.selector = selector
        self.selected_transformers = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit dynamic selection."""
        if self.selector:
            self.selected_transformers = self.selector(self.transformers, X, y)
        else:
            self.selected_transformers = self.transformers
        
        for name, transformer in self.selected_transformers:
            transformer.fit(X, y)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform with dynamic selection."""
        if self.selected_transformers is None:
            self.selected_transformers = self.transformers
        
        transformed = []
        for name, transformer in self.selected_transformers:
            transformed.append(transformer.transform(X))
        return np.hstack(transformed) if transformed else X


class FeaturePipelineBuilder(BaseEstimator):
    """Builder for constructing feature pipelines."""
    
    def __init__(self):
        """Initialize builder."""
        self.steps = []
    
    def add_transformer(self, name: str, transformer: BaseEstimator):
        """Add a transformer."""
        self.steps.append((name, transformer))
        return self
    
    def build(self) -> Pipeline:
        """Build the pipeline."""
        if not self.steps:
            raise ValueError("No steps added to pipeline")
        return Pipeline(self.steps)
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the built pipeline."""
        self.pipeline = self.build()
        self.pipeline.fit(X, y)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform using the pipeline."""
        if not hasattr(self, 'pipeline'):
            raise ValueError("Pipeline not built. Call fit() first.")
        return self.pipeline.transform(X)


class PipelineComposer(BaseEstimator):
    """Compose multiple pipelines."""
    
    def __init__(
        self,
        pipelines: List[Tuple[str, BaseEstimator]],
        final_estimator: Optional[BaseEstimator] = None
    ):
        """Initialize pipeline composer."""
        self.pipelines = pipelines
        self.final_estimator = final_estimator
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit all pipelines."""
        for name, pipeline in self.pipelines:
            pipeline.fit(X, y)
        
        if self.final_estimator:
            # Get predictions from all pipelines
            predictions = []
            for name, pipeline in self.pipelines:
                if hasattr(pipeline, 'predict_proba'):
                    predictions.append(pipeline.predict_proba(X))
                else:
                    predictions.append(pipeline.predict(X).reshape(-1, 1))
            meta_features = np.hstack(predictions)
            self.final_estimator.fit(meta_features, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using composed pipelines."""
        if not self.final_estimator:
            raise ValueError("No final estimator. Cannot predict.")
        
        predictions = []
        for name, pipeline in self.pipelines:
            if hasattr(pipeline, 'predict_proba'):
                predictions.append(pipeline.predict_proba(X))
            else:
                predictions.append(pipeline.predict(X).reshape(-1, 1))
        
        meta_features = np.hstack(predictions)
        return self.final_estimator.predict(meta_features)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select specific columns."""
    
    def __init__(self, columns: Union[List[int], List[str]]):
        """Initialize column selector."""
        self.columns = columns
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y=None):
        """Fit selector."""
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform by selecting columns."""
        if isinstance(X, pd.DataFrame):
            return X[self.columns].values
        else:
            return X[:, self.columns]


# Alias for backward compatibility
ColumnTransformer = ColumnSelector


class FeatureStacker(BaseEstimator, TransformerMixin):
    """Stack features from multiple transformers."""
    
    def __init__(
        self,
        transformers: List[Tuple[str, BaseEstimator]],
        axis: int = 1
    ):
        """Initialize feature stacker."""
        self.transformers = transformers
        self.axis = axis
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit all transformers."""
        for name, transformer in self.transformers:
            transformer.fit(X, y)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Stack transformed features."""
        transformed = []
        for name, transformer in self.transformers:
            t = transformer.transform(X)
            if t.ndim == 1:
                t = t.reshape(-1, 1)
            transformed.append(t)
        return np.concatenate(transformed, axis=self.axis)
