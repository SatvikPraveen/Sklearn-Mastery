"""Estimator wrappers with a uniform, sklearn-compatible interface.

Subpackages:

* :mod:`sklearn_mastery.models.supervised` - classification and regression wrappers
  plus the ``ClassificationModels`` / ``RegressionModels`` factories.
* :mod:`sklearn_mastery.models.unsupervised` - clustering and dimensionality
  reduction wrappers with model-selection helpers.
* :mod:`sklearn_mastery.models.ensemble` - voting, bagging, boosting, stacking
  and blending meta-estimators with diversity analysis.
"""

from sklearn_mastery.models.ensemble import (
    BaggingEnsemble,
    BlendingEnsemble,
    BoostingEnsemble,
    EnsembleAnalyzer,
    EnsembleMethods,
    StackingEnsemble,
    VotingEnsemble,
)
from sklearn_mastery.models.supervised import (
    AdvancedClassifier,
    ClassificationModel,
    ClassificationModels,
    RegressionModel,
    RegressionModels,
)
from sklearn_mastery.models.unsupervised import (
    ClusteringModel,
    ClusteringModels,
    DimensionalityReductionModel,
    estimate_eps,
    evaluate_clustering,
    find_optimal_k,
)

__all__ = [
    "AdvancedClassifier",
    "BaggingEnsemble",
    "BlendingEnsemble",
    "BoostingEnsemble",
    "ClassificationModel",
    "ClassificationModels",
    "ClusteringModel",
    "ClusteringModels",
    "DimensionalityReductionModel",
    "EnsembleAnalyzer",
    "EnsembleMethods",
    "RegressionModel",
    "RegressionModels",
    "StackingEnsemble",
    "VotingEnsemble",
    "estimate_eps",
    "evaluate_clustering",
    "find_optimal_k",
]
