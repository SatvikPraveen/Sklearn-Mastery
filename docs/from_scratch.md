# From scratch: trees, forests and boosting

`sklearn_mastery.from_scratch` re-implements the tree-based algorithms that
dominate tabular machine learning using only NumPy, so that every formula on
this page maps to a few lines of readable code. The estimators are
nonetheless fully scikit-learn compatible (`BaseEstimator` + mixins, `fit`
returns `self`, `clone`, `get_params`, pickling), so they drop into
pipelines, cross-validation and the [research layer](research/benchmarking.md).
Use them to *understand* the algorithms and to run controlled experiments;
use scikit-learn, XGBoost or LightGBM for speed.

All classes carry the `Scratch` suffix:

| Algorithm | Classes |
|---|---|
| CART decision trees | `DecisionTreeClassifierScratch`, `DecisionTreeRegressorScratch` |
| Bagging with out-of-bag estimation | `BaggingClassifierScratch`, `BaggingRegressorScratch` |
| Random forests with per-split feature subsampling | `RandomForestClassifierScratch`, `RandomForestRegressorScratch` |
| AdaBoost (SAMME, AdaBoost.R2) | `AdaBoostClassifierScratch`, `AdaBoostRegressorScratch` |
| Gradient boosting (Friedman) | `GradientBoostingClassifierScratch`, `GradientBoostingRegressorScratch` |
| XGBoost-style second-order boosting | `XGBoostClassifierScratch`, `XGBoostRegressorScratch` |

!!! note "Snippets on this page"
    The package is being finalised; the code blocks below are deliberately
    minimal and use the public names from the table. Consult the
    [API reference](api/from_scratch.md) for the exact signatures.

## CART decision trees

A CART tree (Breiman et al., 1984) is grown top-down by *exact greedy*
search. At a node with sample set $S$, weighted size $W = \sum_{i \in S} w_i$
and impurity $I(S)$, every feature $j$ is scanned: the samples are sorted by
$x_{ij}$ and every midpoint between consecutive distinct values is a
candidate threshold $\theta$. The partition
$S_L = \{i : x_{ij} \le \theta\}$, $S_R = S \setminus S_L$ is scored by the
weighted child impurity

$$
Q(j, \theta) = \frac{W_L}{W}\, I(S_L) + \frac{W_R}{W}\, I(S_R),
$$

and the split minimising $Q$ (equivalently maximising the impurity decrease
$\Delta I = I(S) - Q$) is chosen. Because the child impurities depend only on
cumulative sums of the sorted targets, the scan over all thresholds and
features is vectorised with a handful of `cumsum` calls per node.

Impurity criteria, with $p_k$ the weighted class proportions and $\bar y$
the weighted mean target:

$$
I_{\text{gini}} = 1 - \sum_k p_k^2, \qquad
I_{\text{entropy}} = -\sum_k p_k \log_2 p_k, \qquad
I_{\text{mse}} = \frac{1}{W} \sum_{i \in S} w_i (y_i - \bar y)^2 .
$$

`friedman_mse` keeps the variance as node impurity but ranks candidate
splits by Friedman's (2001) improvement score
$\frac{W_L W_R}{W_L + W_R} (\bar y_L - \bar y_R)^2$, the *unnormalised*
variance reduction used inside gradient boosting.

Growth stops when a node is pure, `max_depth` is reached or the
`min_samples_split` / `min_samples_leaf` constraints bite. `max_features`
draws a random feature subset *at each split* (this is what random forests
rely on). Feature importances are the mean decrease in impurity: the
weighted impurity decrease $W_t I_t - W_{t_L} I_{t_L} - W_{t_R} I_{t_R}$ of
every internal node is credited to its split feature and the totals are
normalised to sum to one. `ccp_alpha > 0` applies Breiman's minimal
cost-complexity (weakest-link) pruning.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn_mastery.from_scratch import DecisionTreeClassifierScratch

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

tree = DecisionTreeClassifierScratch(criterion="gini", max_depth=3, random_state=0).fit(X_tr, y_tr)
print(tree.score(X_te, y_te), tree.get_depth(), tree.get_n_leaves())
print(tree.feature_importances_.round(3))
print(tree.export_text(feature_names=load_iris().feature_names))
```

## Bagging and out-of-bag estimation

Bagging (Breiman, 1996) reduces the variance of an unstable learner by
fitting $B$ copies on bootstrap resamples $\mathcal{D}^{*b}$ drawn with
replacement and averaging:

$$
\hat f_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat f^{*b}(x).
$$

If the members have variance $\sigma^2$ and pairwise correlation $\rho$, the
variance of their average is

$$
\rho\,\sigma^2 + \frac{1 - \rho}{B}\,\sigma^2 ,
$$

so averaging removes the *independent* part of the variance but not the
correlated part. Classification uses soft voting (average of
`predict_proba`) or hard voting (label counts).

A bootstrap sample of size $n$ leaves each observation out with probability
$(1 - 1/n)^n \to e^{-1} \approx 0.368$. Every observation is therefore
**out-of-bag** for roughly 37 % of the members, and aggregating only those
members' predictions gives an honest estimate of generalisation error at no
extra cost (`oob_score=True` populates `oob_score_` and
`oob_decision_function_` / `oob_prediction_`).

```python
from sklearn_mastery.from_scratch import BaggingClassifierScratch, DecisionTreeClassifierScratch

bag = BaggingClassifierScratch(
    estimator=DecisionTreeClassifierScratch(), n_estimators=25, oob_score=True, random_state=0,
).fit(X_tr, y_tr)
print(bag.score(X_te, y_te), round(bag.oob_score_, 3))
print(len(bag.estimators_), bag.estimators_samples_[0][:5])
```

## Random forests

A random forest (Breiman, 2001) is bagging of CART trees with one extra
source of randomness: at *every split* of every tree only a random subset of
`max_features` features is eligible. Bagging alone leaves the trees highly
correlated because a few dominant features are chosen near the root of
almost every tree; in the variance formula above the $\rho\sigma^2$ term
then dominates and adding trees stops helping. Per-split feature subsampling
lowers $\rho$ at the price of a modest increase in each tree's own variance;
the net effect is a lower ensemble variance (Hastie et al., 2009, §15.4).
The subset is redrawn at each split, which distinguishes a forest from
bagging with a single feature subset per member
(`BaggingClassifierScratch(max_features=...)`).

Defaults follow the literature: $\sqrt{p}$ features per split for
classification and all $p$ features for regression.

Two importance measures are available: `feature_importances_` (mean
decrease in impurity, cheap but biased towards high-cardinality features and
computed on training data) and `permutation_importance(X, y)` (Breiman,
2001), which measures how much the score drops when one feature's values are
shuffled and can be evaluated on held-out data.

```python
from sklearn_mastery.from_scratch import RandomForestClassifierScratch

forest = RandomForestClassifierScratch(n_estimators=100, max_features="sqrt", oob_score=True,
                                       random_state=0).fit(X_tr, y_tr)
print(forest.score(X_te, y_te), round(forest.oob_score_, 3))
print(forest.feature_importances_.round(3))
print(forest.permutation_importance(X_te, y_te))
```

## AdaBoost: SAMME and AdaBoost.R2

Boosting fits an additive model $F(x) = \sum_m \alpha_m T_m(x)$ by forward
stagewise minimisation of the exponential loss. SAMME (Hastie, Rosset, Zhu &
Zou, 2009) is the multi-class form. With $K$ classes and sample weights $w_i$
initialised to $1/n$, round $m$ proceeds as follows:

1. fit a weak learner $T_m$ to the weighted data;
2. compute the weighted error
   $\varepsilon_m = \sum_i w_i\, \mathbb{1}[T_m(x_i) \ne y_i] \,/\, \sum_i w_i$;
3. set the learner weight

    $$
    \alpha_m = \eta \left[ \log\frac{1 - \varepsilon_m}{\varepsilon_m} + \log(K - 1) \right];
    $$

4. re-weight $w_i \leftarrow w_i \exp\big(\alpha_m\, \mathbb{1}[T_m(x_i) \ne y_i]\big)$
   and renormalise.

The $\log(K-1)$ term makes SAMME a proper multi-class generalisation:
$\alpha_m > 0$ as long as $\varepsilon_m < 1 - 1/K$, i.e. the weak learner
beats random guessing among $K$ classes; for $K = 2$ it reduces to Freund and
Schapire's AdaBoost.M1. Boosting stops early when $\varepsilon_m = 0$ or
$\varepsilon_m \ge 1 - 1/K$. Prediction is the weighted vote
$\arg\max_k \sum_m \alpha_m \mathbb{1}[T_m(x) = k]$, and class probabilities
follow the population minimiser
$p_k(x) \propto \exp\!\big(\tfrac{1}{K-1} f_k(x)\big)$.

AdaBoost.R2 (Drucker, 1997) handles regression: each round draws a weighted
bootstrap sample, fits the learner, computes each point's relative error
$e_i = |T_m(x_i) - y_i| / \max_j |T_m(x_j) - y_j| \in [0, 1]$ passed through a
loss $L$ (`linear`: $e$; `square`: $e^2$; `exponential`: $1 - e^{-e}$), and
with $\bar L = \sum_i w_i L_i < 1/2$ sets

$$
\beta_m = \frac{\bar L}{1 - \bar L}, \qquad
w_i \leftarrow w_i\, \beta_m^{\,\eta (1 - L_i)}, \qquad
\alpha_m = \eta \log(1/\beta_m).
$$

The ensemble predicts the weighted *median* of the members' predictions,
which is robust to an occasional wildly wrong member.

```python
from sklearn_mastery.from_scratch import AdaBoostClassifierScratch

ada = AdaBoostClassifierScratch(n_estimators=50, learning_rate=1.0, random_state=0).fit(X_tr, y_tr)
print(ada.score(X_te, y_te), len(ada.estimators_))
print(ada.estimator_weights_[:3].round(3), ada.estimator_errors_[:3].round(3))
staged = [(pred == y_te).mean() for pred in ada.staged_predict(X_te)]   # accuracy after each round
```

## Gradient boosting (Friedman)

Gradient boosting (Friedman, 2001) fits the additive model by steepest
descent in function space. Starting from a constant $F_0(x) = \arg\min_\gamma
\sum_i L(y_i, \gamma)$, each round $m$

1. computes the negative gradient (pseudo-residuals) of the loss at the
   current model,
   $$
   r_{im} = -\left.\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right|_{F = F_{m-1}},
   $$
2. fits a regression tree $h_m$ to $\{(x_i, r_{im})\}$ (with the
   `friedman_mse` criterion), partitioning the input into leaves $R_{jm}$,
3. replaces each leaf value by the line-search optimum
   $\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} L\big(y_i, F_{m-1}(x_i) + \gamma\big)$,
4. updates $F_m(x) = F_{m-1}(x) + \nu \sum_j \gamma_{jm}\, \mathbb{1}[x \in R_{jm}]$
   with learning rate (shrinkage) $\nu \in (0, 1]$.

For squared error the pseudo-residuals are the ordinary residuals
$y_i - F_{m-1}(x_i)$ and the leaf values are their means. For binary
classification with the logistic loss $L = \log(1 + e^{-2yF})$ the
pseudo-residuals are $y_i - p_{m-1}(x_i)$ and the Newton step gives
$\gamma_{jm} = \sum r_i \big/ \sum p_i (1 - p_i)$ over the leaf. Multi-class
problems fit one tree per class per round on the softmax gradients.
`subsample < 1` trains each tree on a random fraction of the rows
(stochastic gradient boosting, Friedman 2002), which regularises and gives an
out-of-bag improvement estimate.

```python
from sklearn_mastery.from_scratch import GradientBoostingClassifierScratch

gb = GradientBoostingClassifierScratch(n_estimators=100, learning_rate=0.1, max_depth=3,
                                       random_state=0).fit(X_tr, y_tr)
print(gb.score(X_te, y_te))
```

## XGBoost: second-order objective and the gain formula

XGBoost (Chen & Guestrin, 2016) replaces the first-order gradient step by a
second-order Taylor expansion of a *regularised* objective. With
$g_i = \partial_{\hat y} L(y_i, \hat y_i^{(m-1)})$ and
$h_i = \partial^2_{\hat y} L(y_i, \hat y_i^{(m-1)})$, the objective for the
tree added in round $m$ is

$$
\mathcal{L}^{(m)} \approx \sum_{i=1}^{n} \Big[ g_i f_m(x_i) + \tfrac{1}{2} h_i f_m(x_i)^2 \Big] + \Omega(f_m),
\qquad
\Omega(f) = \gamma T + \tfrac{1}{2} \lambda \sum_{j=1}^{T} w_j^2 ,
$$

where $T$ is the number of leaves and $w_j$ the leaf values. Writing
$G_j = \sum_{i \in I_j} g_i$ and $H_j = \sum_{i \in I_j} h_i$ for the
gradient statistics of leaf $j$, the optimal leaf weight and the resulting
objective are

$$
w_j^{*} = -\frac{G_j}{H_j + \lambda},
\qquad
\mathcal{L}^{*} = -\frac{1}{2} \sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T .
$$

A candidate split of a leaf into $L$ and $R$ is therefore scored by the
**gain**

$$
\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma ,
$$

and the split is only made when the gain is positive, which is how $\gamma$
prunes. The same statistics give the criterion's `min_child_weight`
constraint ($H_j \ge$ threshold). For squared error $g_i = \hat y_i - y_i$
and $h_i = 1$; for the logistic loss $g_i = p_i - y_i$ and
$h_i = p_i (1 - p_i)$, so leaves of a classification tree are Newton steps
shrunk by $\lambda$. Column subsampling (`colsample_bytree`), row subsampling
and the learning rate $\eta$ carry over from gradient boosting.

```python
from sklearn_mastery.from_scratch import XGBoostClassifierScratch

xgb = XGBoostClassifierScratch(n_estimators=100, learning_rate=0.1, max_depth=3,
                               reg_lambda=1.0, gamma=0.0, random_state=0).fit(X_tr, y_tr)
print(xgb.score(X_te, y_te))
```

## Comparing the implementations with the research layer

Because every class is a scikit-learn estimator, a controlled comparison
(same folds, paired tests) is a `BenchmarkSuite` call:

```python
from sklearn.datasets import load_wine

from sklearn_mastery.from_scratch import (
    AdaBoostClassifierScratch, BaggingClassifierScratch, DecisionTreeClassifierScratch,
    RandomForestClassifierScratch,
)
from sklearn_mastery.research import BenchmarkSuite, friedman_test

result = BenchmarkSuite(
    estimators={
        "tree": DecisionTreeClassifierScratch(random_state=0),
        "bagging": BaggingClassifierScratch(n_estimators=25, random_state=0),
        "forest": RandomForestClassifierScratch(n_estimators=50, random_state=0),
        "adaboost": AdaBoostClassifierScratch(n_estimators=50, random_state=0),
    },
    datasets={"iris": load_iris(return_X_y=True), "wine": load_wine(return_X_y=True)},
    scoring="accuracy", n_splits=3, random_state=0,
).run()
print(result.summary("accuracy"))
print(friedman_test(result.score_matrix("accuracy")).average_ranks)
```

## References

- Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). *Classification
  and Regression Trees*. Wadsworth.
- Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24, 123-140.
- Breiman, L. (1996). Out-of-bag estimation. Technical report, UC Berkeley.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5-32.
- Freund, Y., & Schapire, R. (1997). A decision-theoretic generalization of
  on-line learning and an application to boosting. *JCSS*, 55, 119-139.
- Hastie, T., Rosset, S., Zhu, J., & Zou, H. (2009). Multi-class AdaBoost.
  *Statistics and Its Interface*, 2, 349-360.
- Drucker, H. (1997). Improving regressors using boosting techniques. *ICML*.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting
  machine. *Annals of Statistics*, 29(5), 1189-1232.
- Friedman, J. H. (2002). Stochastic gradient boosting. *Computational
  Statistics & Data Analysis*, 38(4), 367-378.
- Chen, T., & Guestrin, C. (2016). XGBoost: a scalable tree boosting system.
  *KDD*.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
  Statistical Learning* (2nd ed.), chapters 10 and 15. Springer.
