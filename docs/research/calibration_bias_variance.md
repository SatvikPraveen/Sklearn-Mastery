# Calibration and bias-variance

Accuracy tells you *how often* a model is right. Two further questions matter
for research use: are its **probabilities** trustworthy, and **where does its
error come from**? `sklearn_mastery.research.calibration` and
`sklearn_mastery.research.bias_variance` answer them.

## Probability calibration

A classifier is calibrated when, among the samples it assigns confidence
$p$, a fraction $p$ is actually correct. The functions below accept either a
1-D array of positive-class probabilities with 0/1 targets, or a 2-D
probability matrix for multi-class targets, in which case *top-label*
calibration is measured (the confidence is the maximum probability and the
outcome is whether the arg-max class is correct).

### Binning

Predictions are grouped into $M$ confidence bins $B_1, \dots, B_M$. Bins are
either equal-width (`strategy="uniform"`, the default) or equal-mass
(`strategy="quantile"`). For each bin the accuracy and the mean confidence are

$$
\mathrm{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbb{1}[\hat y_i = y_i],
\qquad
\mathrm{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat p_i .
$$

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn_mastery.research import compute_calibration_bins

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
clf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_tr, y_tr)
proba = clf.predict_proba(X_te)[:, 1]

bins = compute_calibration_bins(y_te, proba, n_bins=10, strategy="uniform")
print(bins.edges)              # n_bins + 1 edges
print(bins.counts)             # samples per bin
print(bins.mean_confidence, bins.mean_accuracy, bins.gaps)   # NaN for empty bins
```

### Expected and maximum calibration error

$$
\mathrm{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \,\big|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\big|,
\qquad
\mathrm{MCE} = \max_{m} \big|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\big| .
$$

ECE (Naeini et al., 2015; Guo et al., 2017) is the count-weighted mean gap and
is the usual headline number; MCE is the worst-bin gap and matters in
high-stakes settings. Both depend on the binning, so report `n_bins` and
`strategy` alongside them.

```python
from sklearn_mastery.research import expected_calibration_error, maximum_calibration_error

print(expected_calibration_error(y_te, proba, n_bins=10))
print(maximum_calibration_error(y_te, proba, n_bins=10))
print(expected_calibration_error(y_te, clf.predict_proba(X_te), n_bins=10, strategy="quantile"))  # 2-D input
```

### Brier score decomposition

The Brier score $\mathrm{BS} = \frac{1}{n}\sum_i (\hat p_i - y_i)^2$ decomposes
(Murphy, 1973) into

$$
\mathrm{BS} = \underbrace{\frac{1}{n}\sum_{m} |B_m|\,\big(\mathrm{conf}(B_m) - \mathrm{acc}(B_m)\big)^2}_{\text{reliability}}
\;-\;
\underbrace{\frac{1}{n}\sum_{m} |B_m|\,\big(\mathrm{acc}(B_m) - \bar y\big)^2}_{\text{resolution}}
\;+\;
\underbrace{\bar y (1 - \bar y)}_{\text{uncertainty}},
$$

where $\bar y$ is the base rate. *Reliability* is a calibration term (lower is
better), *resolution* rewards confident bins whose accuracy departs from the
base rate (higher is better) and *uncertainty* depends only on the class prior.

```python
from sklearn_mastery.research import brier_score_decomposition

dec = brier_score_decomposition(y_te, proba, n_bins=10)
print(dec.brier, dec.reliability, dec.resolution, dec.uncertainty)
```

The identity holds exactly only when the confidences inside each bin are
constant; with finite bins the reported `brier` is the exact mean squared
error and the three terms are the binned estimates.

### Reliability diagram

```python
import matplotlib
matplotlib.use("Agg")
from sklearn_mastery.research import reliability_diagram

ax = reliability_diagram(y_te, proba, n_bins=10, label="random forest")
ax.figure.savefig("reliability.png", dpi=150, bbox_inches="tight")
```

The plot shows accuracy against confidence per bin with the diagonal of
perfect calibration, annotates the ECE in the legend and adds a histogram of
confidences underneath (`show_histogram=False` to omit it). Pass an `ax` to
overlay several models.

!!! note "Calibration in the evaluation package"
    [`CalibrationAnalyzer`][sklearn_mastery.evaluation.analyzers.CalibrationAnalyzer]
    and `ModelEvaluator.evaluate_calibration` compute similar binary-only
    metrics as dictionaries and are used by the examples. The research
    functions above are the reference implementations and support
    multi-class top-label calibration.

## Bias-variance decomposition

`bias_variance_decomposition` estimates how much of an estimator's expected
loss is due to systematic error (bias) and how much to sensitivity to the
training sample (variance), following Domingos (2000). It draws `n_rounds`
bootstrap samples of the training set, fits a clone on each, predicts a fixed
test set and analyses the resulting `(n_rounds, n_test)` prediction matrix.

### Squared loss

With $\hat y^{(b)}$ the prediction of bootstrap model $b$ and
$\bar y = \frac{1}{B}\sum_b \hat y^{(b)}$ the main (mean) prediction,

$$
\mathbb{E}\big[(y - \hat y)^2\big]
= \underbrace{(\bar y - y)^2}_{\text{bias}^2}
+ \underbrace{\mathbb{E}\big[(\hat y - \bar y)^2\big]}_{\text{variance}}
+ \sigma^2_{\text{noise}} ,
$$

averaged over the test points. The noise-free target is unknown, so the
reported `bias` absorbs the irreducible noise term; `expected_loss = bias +
variance` holds exactly.

```python
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor

from sklearn_mastery.research import bias_variance_decomposition

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

bv = bias_variance_decomposition(
    DecisionTreeRegressor(max_depth=3), X_tr, y_tr, X_te, y_te,
    loss="mse", n_rounds=50, random_state=0,
)
print(bv.expected_loss, bv.bias, bv.variance)      # expected_loss == bias + variance
print(bv.as_dict())
```

### 0-1 loss

For classification the main prediction is the majority vote across bootstrap
models. Bias is the 0-1 loss of the main prediction; the variance of a test
point is the probability that a bootstrap model disagrees with the main
prediction. Domingos (2000, Theorem 1) shows that variance *helps* on biased
points and *hurts* on unbiased points, so the decomposition uses the net
variance

$$
\mathbb{E}[L_{0\text{-}1}] = \text{bias} + \underbrace{V_u - V_b}_{\text{net variance}},
$$

where $V_u$ is the variance contribution of unbiased points and $V_b$ that of
biased points. The result exposes both components.

```python
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

bv = bias_variance_decomposition(
    DecisionTreeClassifier(random_state=0), X_tr, y_tr, X_te, y_te,
    loss="0-1", n_rounds=50, random_state=0, return_predictions=True,
)
print(bv.expected_loss, bv.bias, bv.variance)          # variance is the *net* variance
print(bv.variance_unbiased, bv.variance_biased)
print(bv.predictions.shape)                            # (n_rounds, n_test)
```

Typical use: compare a deep and a shallow tree, or a single tree and a bagged
ensemble, to see variance drop as bias rises. Use `n_jobs` to parallelise the
bootstrap rounds.

## References

- Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining well
  calibrated probabilities using Bayesian binning. *AAAI*.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of
  modern neural networks. *ICML*.
- Murphy, A. H. (1973). A new vector partition of the probability score.
  *Journal of Applied Meteorology*, 12(4), 595-600.
- Domingos, P. (2000). A unified bias-variance decomposition and its
  applications. *ICML*.
