# Models

`sklearn_mastery.models` wraps scikit-learn (and optionally XGBoost, LightGBM
and UMAP) estimators behind a uniform interface. Every wrapper is a real
scikit-learn estimator: `BaseEstimator` plus the right mixin, constructor
arguments stored verbatim (so `clone`, `GridSearchCV` and `Pipeline` work),
`fit` returns `self`, fitted attributes end in an underscore and the wrapped
estimator is available as `.model`.

The wrappers add what you repeatedly write by hand: `evaluate` with a metric
dictionary, `tune_hyperparameters` with a sensible default grid,
`get_feature_importance`, `cross_validate`, `save_model` / `load_model`, and
model selection helpers for clustering and dimensionality reduction.

## Supervised wrappers

### Classification

All classifiers derive from
[`ClassificationModel`][sklearn_mastery.models.supervised.classification.ClassificationModel].

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from sklearn_mastery.models.supervised import LogisticRegressionModel, RandomForestClassifierModel

X, y = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

rf = RandomForestClassifierModel(n_estimators=100, random_state=0).fit(X_tr, y_tr)
metrics = rf.evaluate(X_te, y_te)                      # accuracy, balanced_accuracy, precision, recall, f1, roc_auc, ...
print(metrics["accuracy"], metrics["f1"])
print(rf.get_top_features(k=3))                        # [(feature index, importance), ...]
print(rf.predict_proba(X_te[:2]).round(3))

cv = rf.cross_validate(X_tr, y_tr, cv=5, scoring=["accuracy", "f1_macro"])
print(cv["test_accuracy"].mean())

best_params, best_score = LogisticRegressionModel(max_iter=2000).tune_hyperparameters(X_tr, y_tr, cv=3)
print(best_params, round(best_score, 3))
```

`tune_hyperparameters` uses `get_hyperparameter_grid()` when no grid is given,
runs `GridSearchCV` (or `RandomizedSearchCV` when `n_iter` is set) on a clone,
then updates `self` with the best parameters and refitted estimator.
`save_model(path)` persists the wrapper with joblib; `ClassificationModel.load(path)`
restores it.

Available wrappers: `LogisticRegressionModel`, `DecisionTreeClassifierModel`,
`RandomForestClassifierModel`, `ExtraTreesClassifierModel`,
`GradientBoostingClassifierModel`, `AdaBoostClassifierModel`,
`SVMClassifierModel`, `KNNClassifierModel`, `NaiveBayesModel` (Gaussian,
multinomial, Bernoulli, complement via `nb_type`),
`NeuralNetworkClassifierModel`, and, when installed, `XGBoostClassifierModel`
and `LightGBMClassifierModel`. `AdvancedClassifier(estimator)` wraps any
scikit-learn-compatible classifier in the same interface.

### Regression

[`RegressionModel`][sklearn_mastery.models.supervised.regression.RegressionModel]
mirrors the classification base class with regression metrics (`mse`, `rmse`,
`mae`, `r2`, ...):

```python
from sklearn.datasets import load_diabetes

from sklearn_mastery.models.supervised import GradientBoostingRegressorModel, RidgeRegressionModel

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

ridge = RidgeRegressionModel(alpha=1.0).fit(X_tr, y_tr)
print(ridge.evaluate(X_te, y_te)["r2"])
print(ridge.get_coefficients().shape, ridge.get_intercept())

gbr = GradientBoostingRegressorModel(n_estimators=100, random_state=0).fit(X_tr, y_tr)
staged = [pred for pred in gbr.staged_predict(X_te[:1])]
print(len(staged))                                    # one prediction per boosting stage
```

Wrappers: `LinearRegressionModel`, `RidgeRegressionModel`,
`LassoRegressionModel`, `ElasticNetModel`, `DecisionTreeRegressorModel`,
`RandomForestRegressorModel`, `ExtraTreesRegressorModel`,
`GradientBoostingRegressorModel`, `AdaBoostRegressorModel`,
`SVMRegressorModel`, `NeuralNetworkRegressorModel`, plus
`XGBoostRegressorModel` and `LightGBMRegressorModel` when installed.

### Factories

[`ClassificationModels`][sklearn_mastery.models.supervised.classification.ClassificationModels]
and [`RegressionModels`][sklearn_mastery.models.supervised.regression.RegressionModels]
build wrappers by name, which is what the CLI `train` command and the
examples use:

```python
from sklearn_mastery.models.supervised import ClassificationModels, RegressionModels

print(ClassificationModels.available_models())
# ['logistic_regression', 'decision_tree', 'random_forest', 'extra_trees', 'gradient_boosting',
#  'ada_boost', 'xgboost', 'lightgbm', 'svm', 'knn', 'naive_bayes', 'neural_network']

clf = ClassificationModels().get_model("svm", C=10.0, kernel="rbf")
zoo = ClassificationModels().get_all_models(random_state=0)      # {name: wrapper}

print(RegressionModels.available_models())
reg, score = RegressionModels(random_state=0).train_model(X_tr, y_tr, algorithm="random_forest", n_estimators=50)
print(type(reg).__name__, round(score, 3))
```

Optional back-ends only appear in `available_models()` when the package is
installed. `ClassificationModels.register_custom_model(name, cls)` adds your
own wrapper class to the registry.

## Clustering

All clustering wrappers derive from
[`ClusteringModel`][sklearn_mastery.models.unsupervised.clustering.ClusteringModel]
and provide `fit`, `predict`, `fit_predict`, `get_labels`, `evaluate`,
`save_model` / `load_model`. Inductive algorithms (K-Means, mini-batch
K-Means, Gaussian mixtures, BIRCH, mean shift, affinity propagation) predict
on new data; transductive ones (DBSCAN, OPTICS, agglomerative, spectral)
assign new points to the nearest training cluster.

```python
from sklearn_mastery.data import SyntheticDataGenerator
from sklearn_mastery.models.unsupervised import (
    DBSCANEnhanced, GaussianMixtureModel, KMeansModel, evaluate_clustering, find_optimal_k, estimate_eps,
)

X, true_labels = SyntheticDataGenerator(random_state=0).clustering_dataset(n_samples=400, n_clusters=4, cluster_std=0.8)

km = KMeansModel(n_clusters=4, random_state=0).fit(X)
print(km.evaluate(X, y_true=true_labels))        # silhouette, calinski_harabasz, davies_bouldin, ARI, NMI, ...
print(km.get_cluster_centers().shape)

gmm = GaussianMixtureModel(n_components=4, random_state=0).fit(X)
print(gmm.bic(X), gmm.predict_proba(X[:1]).round(3))
```

### Choosing the number of clusters and DBSCAN's `eps`

[`find_optimal_k`][sklearn_mastery.models.unsupervised.clustering.find_optimal_k]
sweeps candidate cluster counts against a criterion: `silhouette` and
`calinski_harabasz` (maximised), `davies_bouldin`, `bic` and `aic`
(minimised) or `elbow` / `inertia` (knee of the within-cluster sum of squares).
[`estimate_eps`][sklearn_mastery.models.unsupervised.clustering.estimate_eps]
reads `eps` off the sorted *k*-distance graph.

```python
res = find_optimal_k(X, k_range=range(2, 8), criterion="silhouette", random_state=0)
print(res.best_k, res.criterion)
ks, scores = res.as_arrays()

res_bic = find_optimal_k(X, k_range=range(2, 8), criterion="bic", random_state=0)   # uses GaussianMixture
print(res_bic.best_k)

eps = estimate_eps(X, min_samples=5, method="knee")
db = DBSCANEnhanced(eps=None, min_samples=5).fit(X)      # eps=None -> estimated during fit
print(round(eps, 3), len(set(db.get_labels())))
```

The `*Enhanced` / `AdaptiveKMeans` variants embed this selection: leave
`n_clusters=None` (or `n_components=None` for `GaussianMixtureEnhanced`) and
pass `k_range` and `criterion`. `evaluate_clustering(X, labels, y_true=None)`
is the standalone metric function behind `ClusteringModel.evaluate`; noise
points carry the label `NOISE_LABEL == -1`.

### Factory

```python
from sklearn_mastery.models.unsupervised import ClusteringModels

print(ClusteringModels.available_models())
spectral = ClusteringModels(random_state=0).get_model("spectral", n_clusters=4)
everything = ClusteringModels(random_state=0).get_all_models(n_clusters=4, eps=0.5)
```

## Dimensionality reduction

[`DimensionalityReductionModel`][sklearn_mastery.models.unsupervised.dimensionality_reduction.DimensionalityReductionModel]
is a `TransformerMixin` with `fit`, `transform`, `fit_transform`,
`inverse_transform` (where the algorithm supports it), `get_feature_names_out`
(`["pca0", "pca1", ...]`), `get_components`, explained-variance accessors and
`reconstruction_mse`.

```python
from sklearn.datasets import load_digits
from sklearn_mastery.models.unsupervised import DimensionalityReduction, PCAModel, TSNEModel

X, y = load_digits(return_X_y=True)

pca = PCAModel(n_components=10).fit(X)
print(pca.get_explained_variance_ratio().round(3))
print(pca.n_components_for_variance(0.9))
print(pca.reconstruction_mse(X))
Z = pca.transform(X)

emb = TSNEModel(n_components=2, perplexity=30, random_state=0).fit_transform(X[:300])
print(emb.shape)

factory = DimensionalityReduction(random_state=0)
print(factory.available_methods())
embeddings = factory.compare(X[:300], methods=["pca", "isomap"], n_components=2)   # {method: embedding}
```

Wrappers: `PCAModel` (alias `EnhancedPCA`), `KernelPCAModel`,
`TruncatedSVDModel`, `ICAModel`, `FactorAnalysisModel`, `NMFModel`,
`DictionaryLearningModel`, `TSNEModel` (alias `AdaptiveTSNE`), `IsoMapModel`,
`LLEModel`, `MDSModel`, `SpectralEmbeddingModel`, `UMAPModel` (alias
`UMAPEnhanced`) and the dispatching `ManifoldLearning(method=...)`. `UMAPModel`
requires `umap-learn`; with `allow_fallback=True` (the default) it falls back
to Isomap and `HAS_UMAP` tells you which back-end is active.

## Using wrappers with the research layer

Because the wrappers are ordinary estimators they drop straight into
[`BenchmarkSuite`][sklearn_mastery.research.benchmark.BenchmarkSuite]:

```python
from sklearn_mastery.research import BenchmarkSuite
from sklearn_mastery.models.supervised import KNNClassifierModel, RandomForestClassifierModel

X, y = load_wine(return_X_y=True)
result = BenchmarkSuite(
    estimators={"rf": RandomForestClassifierModel(n_estimators=50), "knn": KNNClassifierModel()},
    datasets={"wine": (X, y)}, scoring="accuracy", n_splits=3, random_state=0,
).run()
print(result.summary("accuracy"))
```
