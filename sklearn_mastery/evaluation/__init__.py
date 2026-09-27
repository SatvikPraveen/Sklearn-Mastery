"""Evaluation package for sklearn-mastery project."""

from .metrics import ModelEvaluator
from .statistical_tests import StatisticalTester, ValidationCurveAnalyzer  
from .visualization import ModelVisualizationSuite

__all__ = [
    'ModelEvaluator',
    'StatisticalTester', 
    'ValidationCurveAnalyzer',
    'ModelVisualizationSuite'
]

# Version information
__version__ = '1.0.0'

# Backward compatibility - maintain existing imports
# This ensures existing code continues to work
from .statistical_tests import StatisticalTester as StatisticalTester
from .statistical_tests import ValidationCurveAnalyzer as ValidationCurveAnalyzer