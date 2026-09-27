# Ensembles

`sklearn_mastery.models.ensemble` provides scikit-learn compatible
meta-estimators for the five classic ensemble strategies, a factory of
pre-configured ensembles, and the diversity measures needed to explain *why*
an ensemble beats its members. For the algorithms themselves (bagging,
random forests, AdaBoost, gradient boosting, XGBoost) see
[From scratch](../from_scratch.md).

Every wrapper infers the task from its members (`task_type="auto"`), stores
constructor arguments verbatim, returns `self` from `fit`, exposes fitted
members with a trailing underscore and can be persisted with
`save_model` / `load_model`.

## Voting

[`VotingEnsemble`][sklearn_mastery.models.ensemble.ensemble_methods.VotingEnsemble]
combines heterogeneous estimators by majority vote (`voting="hard"`),
weighted probability averaging (`voting="soft"`) or, for regressors, weighted
averaging.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn_mastery.models.ensemble import VotingEnsemble

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

members = [
    ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
    ("tree", DecisionTreeClassifier(max_depth=4, random_state=0)),
    ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier())),
]
vote = VotingEnsemble(members, voting="soft", weights=[2, 1, 1]).fit(X_tr, y_tr)
print(vote.score(X_te, y_te))
print(vote.agreement_rate(X_te))                 # fraction of samples where all members agree
print(vote.transform(X_te[:3]))                  # (n_samples, n_estimators) matrix of member predictions
```

## Bagging

[`BaggingEnsemble`][sklearn_mastery.models.ensemble.ensemble_methods.BaggingEnsemble]
fits `n_estimators` clones of one base learner on bootstrap samples
(`max_samples`) and optional feature subsets (`max_features`,
`bootstrap_features`) and aggregates their predictions. With
`oob_score=True` it scores every sample with the replicas that did not see it.

```python
from sklearn_mastery.models.ensemble import BaggingEnsemble

bag = BaggingEnsemble(
    base_estimator=DecisionTreeClassifier(random_state=0),
    n_estimators=25, max_samples=0.8, max_features=0.8, oob_score=True, random_state=0,
).fit(X_tr, y_tr)
print(bag.score(X_te, y_te), round(bag.get_oob_score(), 3))
print(bag.get_feature_importance().shape)        # averaged over replicas that expose feature_importances_
print(len(bag.get_individual_predictions(X_te[:5])))
```

## Boosting

[`BoostingEnsemble`][sklearn_mastery.models.ensemble.ensemble_methods.BoostingEnsemble]
is one wrapper over the whole boosting family. `algorithm` is one of
`BOOSTING_ALGORITHMS == ("adaboost", "gradientboosting", "histgradientboosting",
"xgboost", "lightgbm")`; the last two need the optional packages and the
`HAS_XGBOOST` / `HAS_LIGHTGBM` flags report availability.

```python
from sklearn_mastery.models.ensemble import HAS_XGBOOST, BoostingEnsemble

gb = BoostingEnsemble(algorithm="gradientboosting", n_estimators=100, learning_rate=0.1,
                      max_depth=3, random_state=0).fit(X_tr, y_tr)
print(gb.score(X_te, y_te), gb.get_n_rounds())
stage_scores = [(pred == y_te).mean() for pred in gb.staged_predict(X_te)]
print(len(stage_scores), round(max(stage_scores), 3))

ada = BoostingEnsemble(algorithm="adaboost", n_estimators=50, random_state=0).fit(X_tr, y_tr)
print(ada.score(X_te, y_te))

if HAS_XGBOOST:
    xgb = BoostingEnsemble(algorithm="xgboost", n_estimators=100, max_depth=4, random_state=0).fit(X_tr, y_tr)
    print(xgb.score(X_te, y_te), xgb.get_feature_importance().shape)
```

`estimator_params` forwards back-end specific keyword arguments (for example
`{"subsample": 0.8}`), and `get_n_rounds` is aware of early stopping.

## Stacking and blending

[`StackingEnsemble`][sklearn_mastery.models.ensemble.ensemble_methods.StackingEnsemble]
trains a meta-estimator on *out-of-fold* predictions of the base estimators
(`cv` folds), then refits the base estimators on all data.
[`BlendingEnsemble`][sklearn_mastery.models.ensemble.ensemble_methods.BlendingEnsemble]
uses a single hold-out split (`holdout_size`) instead, which is cheaper but
wastes data. Both accept `stack_method` (`"auto"`, `"predict"`,
`"predict_proba"`, `"decision_function"`) and `passthrough=True` to append the
original features to the meta-features.

```python
from sklearn_mastery.models.ensemble import BlendingEnsemble, StackingEnsemble

stack = StackingEnsemble(members, meta_estimator=LogisticRegression(max_iter=1000),
                         cv=5, random_state=0).fit(X_tr, y_tr)
print(stack.score(X_te, y_te))
print(stack.oof_meta_features_.shape)            # (n_train, n_meta_features)
print(stack.get_feature_names_out()[:3])

blend = BlendingEnsemble(members, holdout_size=0.25, random_state=0).fit(X_tr, y_tr)
print(blend.score(X_te, y_te))
```

## The `EnsembleMethods` factory

[`EnsembleMethods`][sklearn_mastery.models.ensemble.ensemble_methods.EnsembleMethods]
returns pre-configured ensembles with diverse default members
(`default_classifiers()` / `default_regressors()`), so a comparison of the
strategies takes a few lines:

```python
from sklearn_mastery.models.ensemble import EnsembleMethods

em = EnsembleMethods(random_state=0)
candidates = {
    "voting": em.get_voting_classifier(voting="soft"),
    "bagging": em.get_bagging_classifier(n_estimators=25),
    "gradient_boosting": em.get_gradient_boosting_classifier(n_estimators=100),
    "stacking": em.get_stacking_classifier(cv=3),
    "blending": em.get_blending_classifier(),
}
print(em.compare_ensembles(candidates, X_tr, y_tr, X_te, y_te))    # {name: test score}
```

Other factory methods: `get_random_forest`, `get_extra_trees_classifier`,
`get_hist_gradient_boosting_classifier`, `get_ada_boost_classifier`,
`get_xgboost_classifier`, `get_lightgbm_classifier`, `get_multi_level_stacking`
and the regression counterparts.

## Diversity analysis

Ensembles help only when the members make *different* errors. The module
implements the standard pairwise and non-pairwise diversity measures of
Kuncheva and Whitaker (2003). For two classifiers with oracle outputs
(correct / incorrect) let $N^{11}, N^{10}, N^{01}, N^{00}$ be the counts of
samples both get right, only the first gets right, only the second gets
right and both get wrong. Then

$$
Q = \frac{N^{11}N^{00} - N^{01}N^{10}}{N^{11}N^{00} + N^{01}N^{10}}, \qquad
\rho = \frac{N^{11}N^{00} - N^{01}N^{10}}{\sqrt{(N^{11}+N^{10})(N^{01}+N^{00})(N^{11}+N^{01})(N^{10}+N^{00})}},
$$

$$
\text{disagreement} = \frac{N^{01} + N^{10}}{N}, \qquad
\text{double fault} = \frac{N^{00}}{N}.
$$

$Q$ and $\rho$ range over $[-1, 1]$ with $0$ for independent classifiers;
lower is more diverse. Disagreement is label-based and needs no oracle. For
$L$ classifiers the Kohavi-Wolpert variance and Kuncheva's entropy measure
summarise the whole ensemble.

```python
from sklearn_mastery.models.ensemble import (
    EnsembleAnalyzer, diversity_matrix, pairwise_diversity, q_statistic,
)

fitted = {name: est.fit(X_tr, y_tr) for name, est in members}
preds = {name: est.predict(X_te) for name, est in fitted.items()}

print(pairwise_diversity(preds["logreg"], preds["tree"], y_te))        # all pairwise measures
print(round(q_statistic(preds["logreg"], preds["knn"], y_te), 3))
print(diversity_matrix(list(preds.values()), y_te, measure="disagreement").round(3))

analyzer = EnsembleAnalyzer(task_type="classification")
summary = analyzer.calculate_diversity(fitted, X_te, y_te)
print(summary["individual_scores"])
print(round(summary["disagreement"], 3), round(summary["entropy"], 3), round(summary["kohavi_wolpert_variance"], 3))

report = analyzer.analyze_ensemble(vote, X_te, y_te)      # ensemble score vs. members, gain over best member
print(sorted(report))
```

## References

- Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24(2), 123-140.
- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241-259.
- Kuncheva, L. I., & Whitaker, C. J. (2003). Measures of diversity in
  classifier ensembles and their relationship with the ensemble accuracy.
  *Machine Learning*, 51(2), 181-207.
- Kohavi, R., & Wolpert, D. H. (1996). Bias plus variance decomposition for
  zero-one loss functions. *ICML*.
