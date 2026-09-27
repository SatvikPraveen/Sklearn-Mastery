"""Pipelines package for sklearn-mastery project."""

from .custom_transformers import (
    AdvancedImputer,
    BinningTransformer,
    CategoricalEncoder,
    CustomScaler,
    DataValidator,
    DateTimeTransformer,
    DomainSpecificEncoder,
    FeatureInteractionCreator,
    FeatureScaler,
    FeatureSelector,
    FeatureUnion,
    MissingValueHandler,
    NumericTransformer,
    OutlierRemover,
    PipelineDebugger,
    PolynomialFeatureCreator,
    TargetEncoder,
    TextFeatureExtractor,
    TextTransformer,
    TimeSeriesFeatureCreator,
)
from .pipeline_factory import PipelineFactory
from .model_selection import *

__all__ = [
    # Custom transformers
    'AdvancedImputer',
    'BinningTransformer',
    'CategoricalEncoder',
    'CustomScaler',
    'DataValidator',
    'DateTimeTransformer',
    'DomainSpecificEncoder',
    'FeatureInteractionCreator',
    'FeatureScaler',
    'FeatureSelector',
    'FeatureUnion',
    'MissingValueHandler',
    'NumericTransformer',
    'OutlierRemover',
    'PipelineDebugger',
    'PolynomialFeatureCreator',
    'TargetEncoder',
    'TextFeatureExtractor',
    'TextTransformer',
    'TimeSeriesFeatureCreator',
    
    # Pipeline factory
    'PipelineFactory',
    
    # Model selection
    'AdvancedModelSelector',
    'MultiObjectiveSelector', 
    'NestedCrossValidation',
    'LearningCurveAnalyzer',
    'ValidationCurveAnalyzer',
    'AutoMLSelector'
]