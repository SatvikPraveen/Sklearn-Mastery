"""From-first-principles, NumPy-only implementations of the classic tree ensembles.

Every estimator in this subpackage is a faithful re-implementation of the
published algorithm - none of them wraps scikit-learn's or XGBoost's tree
code - while remaining fully scikit-learn compatible (``BaseEstimator`` +
mixins, ``fit`` returns ``self``, ``get_params``/``set_params``, ``clone``,
pickling, ``check_is_fitted``, ``n_features_in_``, ``classes_``,
``feature_importances_``). Each module docstring derives the maths of its
method; the code is written to be read alongside it.

Modules
-------
``decision_tree``
    CART (Breiman et al. 1984): exact greedy split search over sorted
    thresholds, gini/entropy and squared-error/Friedman criteria, per-split
    feature subsampling, sample weights, MDI importances, ``export_text`` and
    minimal cost-complexity (weakest-link) pruning.
``bagging``
    Bootstrap aggregating (Breiman 1996) with optional feature bagging,
    soft/hard voting and out-of-bag estimation.
``random_forest``
    Random forests (Breiman 2001): bagged CART trees with per-split feature
    subsampling, OOB estimate, MDI and permutation importances.
``adaboost``
    SAMME multiclass AdaBoost (Hastie, Rosset, Zhu and Zou 2009) and
    AdaBoost.R2 (Drucker 1997) with weighted-median prediction.
``gradient_boosting``
    Gradient boosting machines (Friedman 2001, 2002): squared / absolute /
    Huber losses with per-leaf line search, binomial and multinomial
    deviance with Newton leaf updates, shrinkage and stochastic subsampling.
``xgboost_scratch``
    Second-order boosting of Chen and Guestrin (2016): gain and leaf-weight
    formulas from the quadratic approximation of the objective, L2 and
    gamma regularisation, min_child_weight, row/column subsampling,
    sparsity-aware default directions for missing values and early stopping.

References
----------
Breiman, L., Friedman, J., Olshen, R. and Stone, C. (1984). *Classification
and Regression Trees*. Wadsworth.

Breiman, L. (1996). Bagging predictors. *Machine Learning* 24, 123-140.

Breiman, L. (2001). Random forests. *Machine Learning* 45, 5-32.

Drucker, H. (1997). Improving regressors using boosting techniques. *ICML*.

Hastie, T., Rosset, S., Zhu, J. and Zou, H. (2009). Multi-class AdaBoost.
*Statistics and Its Interface* 2, 349-360.

Friedman, J. H. (2001). Greedy function approximation: a gradient boosting
machine. *Annals of Statistics* 29(5), 1189-1232.

Friedman, J. H. (2002). Stochastic gradient boosting. *Computational
Statistics & Data Analysis* 38(4), 367-378.

Chen, T. and Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
*KDD '16*, 785-794.
"""

from sklearn_mastery.from_scratch.adaboost import AdaBoostClassifierScratch, AdaBoostRegressorScratch
from sklearn_mastery.from_scratch.bagging import BaggingClassifierScratch, BaggingRegressorScratch
from sklearn_mastery.from_scratch.decision_tree import (
    DecisionTreeClassifierScratch,
    DecisionTreeRegressorScratch,
)
from sklearn_mastery.from_scratch.gradient_boosting import (
    GradientBoostingClassifierScratch,
    GradientBoostingRegressorScratch,
)
from sklearn_mastery.from_scratch.random_forest import (
    RandomForestClassifierScratch,
    RandomForestRegressorScratch,
)
from sklearn_mastery.from_scratch.xgboost_scratch import XGBoostClassifierScratch, XGBoostRegressorScratch

__all__ = [
    "AdaBoostClassifierScratch",
    "AdaBoostRegressorScratch",
    "BaggingClassifierScratch",
    "BaggingRegressorScratch",
    "DecisionTreeClassifierScratch",
    "DecisionTreeRegressorScratch",
    "GradientBoostingClassifierScratch",
    "GradientBoostingRegressorScratch",
    "RandomForestClassifierScratch",
    "RandomForestRegressorScratch",
    "XGBoostClassifierScratch",
    "XGBoostRegressorScratch",
]
