"""Model evaluation: metrics, cross-validation, diagnostics and statistical tests."""

from sklearn_mastery.evaluation.analyzers import (
    CalibrationAnalyzer,
    ConfusionMatrixAnalyzer,
    FeatureImportanceAnalyzer,
    PrecisionRecallAnalyzer,
    ResidualAnalyzer,
    ROCAnalyzer,
)
from sklearn_mastery.evaluation.comparison import PerformanceComparator
from sklearn_mastery.evaluation.cross_validation import (
    CrossValidator,
    LearningCurveAnalyzer,
    ValidationCurveAnalyzer,
)
from sklearn_mastery.evaluation.metrics import MetricsCalculator, ModelEvaluator
from sklearn_mastery.evaluation.statistical_tests import StatisticalTester
from sklearn_mastery.evaluation.visualization import ModelVisualizationSuite

__all__ = [
    "CalibrationAnalyzer",
    "ConfusionMatrixAnalyzer",
    "CrossValidator",
    "FeatureImportanceAnalyzer",
    "LearningCurveAnalyzer",
    "MetricsCalculator",
    "ModelEvaluator",
    "ModelVisualizationSuite",
    "PerformanceComparator",
    "PrecisionRecallAnalyzer",
    "ROCAnalyzer",
    "ResidualAnalyzer",
    "StatisticalTester",
    "ValidationCurveAnalyzer",
]

__version__ = "2.0.0"
