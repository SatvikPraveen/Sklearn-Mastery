# Evaluation

`sklearn_mastery.evaluation` covers the day-to-day evaluation toolbox:
metric calculation, cross-validation with timings, learning and validation
curves, hold-out comparison of several models, classical significance tests,
task-specific analyzers (ROC, precision-recall, confusion matrix, calibration,
residuals, feature importance) and a matplotlib visualisation suite.

For rigorous multi-dataset comparison use the
[research layer](../research/statistical_comparison.md); the classes here are
designed for single-dataset, interactive work and for the example scripts.

## Metrics

[`MetricsCalculator`][sklearn_mastery.evaluation.metrics.MetricsCalculator]
computes individual metrics or the full set for a task. `average="auto"`
resolves to binary averaging for two classes and macro averaging otherwise.

```python
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn_mastery.evaluation import MetricsCalculator

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
clf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_tr, y_tr)

calc = MetricsCalculator(task_type="classification")
print(calc.calculate_all_metrics(y_te, clf.predict(X_te), y_proba=clf.predict_proba(X_te)))
# accuracy, balanced_accuracy, precision, recall, f1, matthews_corrcoef, cohen_kappa, roc_auc
print(calc.f1_score(y_te, clf.predict(X_te), average="macro"))

Xr, yr = load_diabetes(return_X_y=True)
reg = RandomForestRegressor(n_estimators=50, random_state=0).fit(Xr[:300], yr[:300])
print(MetricsCalculator(task_type="regression").calculate_all_metrics(yr[300:], reg.predict(Xr[300:])))
# mse, rmse, mae, r2, explained_variance, median_absolute_error, mape
```

`clustering_metrics(X, labels, y_true=None)` returns silhouette,
Calinski-Harabasz and Davies-Bouldin scores plus ARI / NMI when ground truth
is available.

## `ModelEvaluator`

[`ModelEvaluator`][sklearn_mastery.evaluation.metrics.ModelEvaluator] scores a
fitted estimator and produces richer reports:

```python
from sklearn_mastery.evaluation import ModelEvaluator

ev = ModelEvaluator(task_type="classification")
print(ev.evaluate(clf, X_te, y_te))                       # metric dict for a fitted model
report = ev.detailed_report(clf, X_te, y_te)              # metrics + confusion matrix + per-class report
print(sorted(report))

full = ev.evaluate_model(clf, X_tr, X_te, y_tr, y_te, model_name="rf")   # train/test metrics, CV, learning curve
print(ev.generate_evaluation_summary(full))
print(ev.evaluate_calibration(y_te, clf.predict_proba(X_te))["expected_calibration_error"])
```

`evaluate_fairness(y_true, y_pred, sensitive_attrs)` reports per-group
accuracy and demographic-parity differences; `evaluate_time_series` adds MASE
relative to a seasonal-naive forecast; `custom_metrics={name: callable}`
extends every report.

## Cross-validation and curves

```python
from sklearn_mastery.evaluation import CrossValidator, LearningCurveAnalyzer, ValidationCurveAnalyzer

cv = CrossValidator(cv=5, scoring=["accuracy", "roc_auc"], random_state=0)
scores = cv.cross_validate(RandomForestClassifier(n_estimators=50, random_state=0), X, y)
print(sorted(scores))                                     # fit_time, score_time, test_accuracy, train_accuracy, ...
print(cv.summarize(scores)["test_accuracy"])              # mean, std, min, max, ...

lc = LearningCurveAnalyzer(scoring="accuracy", random_state=0)
curve = lc.generate_learning_curve(RandomForestClassifier(n_estimators=50, random_state=0), X, y,
                                   train_sizes=[0.2, 0.5, 1.0], cv=3)
print(lc.analyze_learning_curve(curve))                   # overfitting / convergence diagnosis

vc = ValidationCurveAnalyzer(scoring="accuracy", random_state=0)
vcurve = vc.generate_validation_curve(RandomForestClassifier(random_state=0), X, y,
                                      param_name="max_depth", param_range=[2, 4, 8], cv=3)
print(ValidationCurveAnalyzer.find_optimal_parameter(vcurve))
```

`CrossValidator` picks stratified folds for classifiers automatically
(`stratify=None`) and accepts any splitter or `groups`.

## Comparing models on one dataset

[`PerformanceComparator`][sklearn_mastery.evaluation.comparison.PerformanceComparator]
evaluates several models on identical folds and runs the appropriate test:
paired *t* (or Wilcoxon) for two models, Friedman followed by Nemenyi and
Holm-corrected pairwise tests for three or more.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn_mastery.evaluation import PerformanceComparator

models = {
    "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "svm": make_pipeline(StandardScaler(), SVC()),
    "rf": RandomForestClassifier(n_estimators=50, random_state=0),
}
pc = PerformanceComparator(task_type="classification", random_state=0)
stats = pc.statistical_comparison(models, X, y, cv=5, scoring="accuracy")
print(stats["best_model"], round(stats["p_value"], 4), stats["is_significant"])
print({name: round(stats[name]["mean"], 3) for name in models})

fitted = {name: m.fit(X_tr, y_tr) for name, m in models.items()}
print(pc.rank_models(fitted, X_te, y_te, metric="f1"))
print(PerformanceComparator.to_dataframe(pc.compare_models(fitted, X_te, y_te)).round(3))
```

## Statistical tests

[`StatisticalTester`][sklearn_mastery.evaluation.statistical_tests.StatisticalTester]
exposes the individual tests with dictionary results (`statistic`,
`p_value`, `is_significant`, `interpretation`, ...):

```python
import numpy as np
from sklearn_mastery.evaluation import StatisticalTester

st = StatisticalTester(alpha=0.05, random_state=0)
a = np.array([0.91, 0.89, 0.93, 0.90, 0.92])
b = np.array([0.88, 0.87, 0.90, 0.89, 0.90])
print(st.paired_t_test(a, b)["p_value"])
print(st.wilcoxon_test(a, b)["p_value"])
print(st.corrected_resampled_t_test(a, b, n_train=400, n_test=100)["p_value"])   # Nadeau-Bengio
print(st.compute_confidence_intervals(a, confidence=0.95))
print(st.bootstrap_confidence_interval(a, n_bootstrap=500)["ci_lower"])

matrix = np.array([[0.90, 0.88, 0.80], [0.85, 0.80, 0.75], [0.88, 0.85, 0.82], [0.92, 0.90, 0.85]])
fr = st.friedman_test(matrix, model_names=["a", "b", "c"])
print(fr["average_ranks"], round(fr["f_p_value"], 4))
print(st.nemenyi_post_hoc(matrix, model_names=["a", "b", "c"])["critical_difference"])
print(st.compare_multiple_models({"a": a, "b": b, "c": a - 0.05})["method"])

pred1, pred2 = clf.predict(X_te), LogisticRegression(max_iter=2000).fit(X_tr, y_tr).predict(X_te)
print(st.mcnemar_test(y_te, pred1, pred2)["p_value"])
```

The module-level helpers in `sklearn_mastery.evaluation.utils`
(`bootstrap_metric`, `compute_effect_size`, `holm_bonferroni`,
`is_higher_better`, ...) are shared by all of these classes.

## Analyzers

Each analyzer returns plain dictionaries and arrays so that results can be
logged or plotted with any library.

```python
from sklearn_mastery.evaluation import (
    CalibrationAnalyzer, ConfusionMatrixAnalyzer, FeatureImportanceAnalyzer,
    PrecisionRecallAnalyzer, ROCAnalyzer, ResidualAnalyzer,
)

proba = clf.predict_proba(X_te)[:, 1]
roc = ROCAnalyzer().generate_roc_curve(y_te, proba)
print(round(roc["auc"], 3), ROCAnalyzer().find_optimal_threshold(y_te, proba, method="youden"))

pr = PrecisionRecallAnalyzer().generate_precision_recall_curve(y_te, proba)
print(round(pr["average_precision"], 3), PrecisionRecallAnalyzer().find_f1_optimal_threshold(y_te, proba))

cm = ConfusionMatrixAnalyzer().analyze_confusion_matrix(y_te, clf.predict(X_te))
print(cm["confusion_matrix"], sorted(cm["class_metrics"][1]))

cal = CalibrationAnalyzer(n_bins=10).calculate_calibration_metrics(y_te, proba)
print({k: round(v, 3) for k, v in cal.items()})

fi = FeatureImportanceAnalyzer(random_state=0)
print(fi.get_top_k_features(clf, k=3, feature_names=load_breast_cancer().feature_names))

res = ResidualAnalyzer().analyze_residuals(yr[300:], reg.predict(Xr[300:]))
print(sorted(res))                                   # statistics, normality / heteroscedasticity tests, plot data
```

`FeatureImportanceAnalyzer` uses the model's native importances or
coefficients when available and falls back to permutation importance
(`method="permutation"`, requires `X`, `y`).

## Visualisation

[`ModelVisualizationSuite`][sklearn_mastery.evaluation.visualization.ModelVisualizationSuite]
turns the dictionaries above into matplotlib figures. Every method returns a
`Figure` and accepts `save_path`.

```python
import matplotlib
matplotlib.use("Agg")

from sklearn_mastery.evaluation import ModelVisualizationSuite

viz = ModelVisualizationSuite()
fig = viz.plot_confusion_matrix(y_te, clf.predict(X_te), class_names=["malignant", "benign"], normalize=True)
fig = viz.plot_roc_curves({"rf": {"fpr": roc["fpr"], "tpr": roc["tpr"], "auc": roc["auc"]}})
fig = viz.plot_calibration_curve(y_te, proba, n_bins=10)
fig = viz.plot_feature_importance(load_breast_cancer().feature_names, clf.feature_importances_, max_features=10)
fig = viz.plot_learning_curves(curve, title="Random forest")
fig = viz.plot_residuals(yr[300:], reg.predict(Xr[300:]))
fig.savefig("residuals.png", dpi=120)
```

Further plots: `plot_precision_recall_curves`, `plot_validation_curve`,
`plot_model_comparison`, `plot_decision_boundary` (2-D classifiers),
`plot_clustering_results` (PCA-projected when needed), `plot_fairness_metrics`,
`plot_feature_importance_waterfall` and
`create_model_performance_dashboard`. `create_interactive_scatter_plot` uses
plotly when it is installed.
