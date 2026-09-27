"""XGBoost-style second-order gradient boosting (Chen and Guestrin, 2016) from scratch.

Objective and its second-order approximation
--------------------------------------------
XGBoost minimises a regularised objective over an additive model of trees,

$$
\\mathcal{L} = \\sum_i L(y_i, \\hat y_i) + \\sum_m \\Omega(f_m), \\qquad
\\Omega(f) = \\gamma T + \\tfrac{1}{2}\\lambda \\sum_{j=1}^{T} w_j^2,
$$

where $T$ is the number of leaves of tree $f$ and $w_j$
its leaf weights. At round $m$ the prediction is
$\\hat y_i^{(m)} = \\hat y_i^{(m-1)} + f_m(x_i)$ and a second-order
Taylor expansion of the loss around $\\hat y_i^{(m-1)}$ gives

$$
\\mathcal{L}^{(m)} \\approx \\sum_i \\big[ L(y_i, \\hat y_i^{(m-1)})
    + g_i f_m(x_i) + \\tfrac{1}{2} h_i f_m(x_i)^2 \\big] + \\Omega(f_m),
\\quad g_i = \\partial_{\\hat y} L, \\; h_i = \\partial^2_{\\hat y} L.
$$

Dropping the constant and grouping the samples by leaf
($I_j = \\{i : x_i \\in \\text{leaf } j\\}$, $G_j = \\sum_{I_j} g_i$,
$H_j = \\sum_{I_j} h_i$):

$$
\\tilde{\\mathcal{L}}^{(m)} = \\sum_{j=1}^{T}
    \\big[ G_j w_j + \\tfrac12 (H_j + \\lambda) w_j^2 \\big] + \\gamma T.
$$

This is a separable quadratic in the $w_j$, so for a *fixed* tree
structure the optimal leaf weight and the resulting objective are

$$
w_j^* = -\\frac{G_j}{H_j + \\lambda}, \\qquad
\\tilde{\\mathcal{L}}^* = -\\tfrac12 \\sum_j \\frac{G_j^2}{H_j + \\lambda} + \\gamma T.
$$

The structure itself is found greedily: splitting a node with sums
$(G, H)$ into $(G_L, H_L)$ and $(G_R, H_R)$ changes the
objective by

$$
\\text{Gain} = \\tfrac12 \\left[ \\frac{G_L^2}{H_L + \\lambda}
    + \\frac{G_R^2}{H_R + \\lambda}
    - \\frac{(G_L + G_R)^2}{H_L + H_R + \\lambda} \\right] - \\gamma,
$$

and a split is accepted only if its gain is positive. Unlike CART, the
split criterion therefore *is* the loss being optimised - no impurity
proxy - and the same tree builder serves every twice-differentiable loss.

Losses implemented here ($\\hat y$ is the raw margin):

* squared error: $L = \\tfrac12 (y - \\hat y)^2$, $g = \\hat y - y$, $h = 1$;
* binary logistic: $p = \\sigma(\\hat y)$, $g = p - y$, $h = p(1 - p)$;
* multiclass softmax (one tree per class per round): $p_k = \\operatorname{softmax}(\\hat y)_k$,
  $g_k = p_k - y_k$, $h_k = p_k (1 - p_k)$.

Further ingredients of the paper that are implemented:

* **Shrinkage** $\\eta$ scales every new leaf weight (``learning_rate``).
* **Row and column subsampling** (``subsample``, ``colsample_bytree``,
  ``colsample_bynode``).
* **min_child_weight**: a child must have $H \\ge$ this value - for
  logistic loss $H$ is the sum of $p(1-p)$, i.e. a measure of how
  many *uncertain* samples the leaf contains.
* **Sparsity-aware split finding**: at each split the missing values are
  tried on the left and on the right, and the direction with higher gain
  becomes the node's *default direction* used at prediction time.
* **Early stopping** on an evaluation set, with ``evals_result_`` and
  ``best_iteration_`` as in the XGBoost API.

Note on constants: the XGBoost library omits the factor $\\tfrac12$
inside its gain and compares that doubled quantity with ``gamma``, and it
applies ``gamma`` by post-pruning. Here the paper's formula is used
verbatim (gamma as a pre-pruning threshold on the halved gain), so for
``gamma > 0`` the two are not numerically identical; for ``gamma = 0`` they
build the same trees.

References
----------
Chen, T. and Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
*KDD '16*, 785-794.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

__all__ = ["XGBoostClassifierScratch", "XGBoostRegressorScratch"]

_MAX_SEED = np.iinfo(np.int32).max


# --------------------------------------------------------------------------- #
# Tree
# --------------------------------------------------------------------------- #
@dataclass
class _XGBNode:
    """A node of a second-order boosted tree.

    Attributes:
        depth: Depth of the node.
        sum_grad: $G$ over the node's samples.
        sum_hess: $H$ over the node's samples (the node's *cover*).
        weight: Leaf weight $\\eta \\cdot w^*$ (meaningful for leaves).
        feature: Split feature (``-1`` for leaves).
        threshold: Split threshold; ``x < threshold`` goes left.
        missing_left: Default direction for NaN values.
        gain: Gain of the split (0 for leaves).
        left: Left child.
        right: Right child.
    """

    depth: int
    sum_grad: float
    sum_hess: float
    weight: float
    feature: int = -1
    threshold: float = np.nan
    missing_left: bool = False
    gain: float = 0.0
    left: Optional[_XGBNode] = None
    right: Optional[_XGBNode] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None


class _XGBTree:
    """One regression tree grown on gradients and Hessians.

    The tree is built by exact greedy search: at each node the candidate
    features are sorted, prefix sums of ``g`` and ``h`` give $(G_L, H_L)$
    for every cut, and the gain formula from the module docstring is
    evaluated for both possible default directions of the missing values.

    Args:
        max_depth: Maximum depth (root is depth 0).
        reg_lambda: L2 penalty $\\lambda$ on leaf weights.
        gamma: Minimum gain $\\gamma$ required to make a split.
        min_child_weight: Minimum $H$ in each child.
        learning_rate: Shrinkage $\\eta$ folded into the leaf weights.
        colsample_bynode: Fraction of features considered at each split.
    """

    def __init__(
        self,
        max_depth: int,
        reg_lambda: float,
        gamma: float,
        min_child_weight: float,
        learning_rate: float,
        colsample_bynode: float,
    ) -> None:
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.learning_rate = learning_rate
        self.colsample_bynode = colsample_bynode
        self.root: Optional[_XGBNode] = None

    # ------------------------------------------------------------ growing
    def fit(
        self, X: np.ndarray, g: np.ndarray, h: np.ndarray, features: np.ndarray, rng: np.random.RandomState
    ) -> _XGBTree:
        """Grow the tree.

        Args:
            X: Feature matrix (may contain NaN).
            g: First-order gradients.
            h: Second-order gradients (Hessians).
            features: Indices of the features available to this tree
                (``colsample_bytree`` is applied by the caller).
            rng: Random state for ``colsample_bynode``.
        """
        self.rng = rng
        self.features = features
        self.root = self._grow(X, g, h, np.arange(X.shape[0]), depth=0)
        return self

    def _leaf_weight(self, G: float, H: float) -> float:
        return -self.learning_rate * G / (H + self.reg_lambda)

    def _grow(self, X: np.ndarray, g: np.ndarray, h: np.ndarray, idx: np.ndarray, depth: int) -> _XGBNode:
        G, H = float(g[idx].sum()), float(h[idx].sum())
        node = _XGBNode(depth=depth, sum_grad=G, sum_hess=H, weight=self._leaf_weight(G, H))
        if depth >= self.max_depth or idx.size < 2:
            return node

        n_cols = max(1, round(self.colsample_bynode * self.features.size))
        cols = (
            self.features
            if n_cols >= self.features.size
            else self.rng.choice(self.features, n_cols, replace=False)
        )
        split = self._best_split(X[np.ix_(idx, cols)], g[idx], h[idx], G, H)
        if split is None:
            return node
        col, threshold, missing_left, gain = split
        feature = int(cols[col])
        x = X[idx, feature]
        go_left = np.where(np.isnan(x), missing_left, x < threshold)
        node.feature, node.threshold, node.missing_left, node.gain = feature, threshold, missing_left, gain
        node.left = self._grow(X, g, h, idx[go_left], depth + 1)
        node.right = self._grow(X, g, h, idx[~go_left], depth + 1)
        return node

    def _best_split(
        self, Xc: np.ndarray, g: np.ndarray, h: np.ndarray, G: float, H: float
    ) -> Optional[Tuple[int, float, bool, float]]:
        """Vectorised exact greedy search over all cuts of all candidate columns.

        Returns:
            ``(column, threshold, missing_left, gain)`` or ``None``.
        """
        n, m = Xc.shape
        lam = self.reg_lambda
        order = np.argsort(Xc, axis=0, kind="stable")  # NaN sorts last
        Xs = np.take_along_axis(Xc, order, axis=0)
        gs, hs = g[order], h[order]
        present = ~np.isnan(Xs)  # (n, m)

        # Prefix sums over non-missing rows only (missing rows contribute 0).
        G_left = np.cumsum(np.where(present, gs, 0.0), axis=0)[:-1]  # (n-1, m)
        H_left = np.cumsum(np.where(present, hs, 0.0), axis=0)[:-1]
        G_present = np.sum(np.where(present, gs, 0.0), axis=0)  # (m,)
        H_present = np.sum(np.where(present, hs, 0.0), axis=0)
        G_missing, H_missing = G - G_present, H - H_present
        G_right = G_present - G_left
        H_right = H_present - H_left

        # Cut i separates sorted rows [0..i] from [i+1..]; both rows must be
        # present and distinct for the midpoint to be a meaningful threshold.
        valid = present[1:] & (Xs[1:] > Xs[:-1])
        parent_term = G**2 / (H + lam)

        def gain(GL, HL, GR, HR):
            with np.errstate(divide="ignore", invalid="ignore"):
                value = 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam) - parent_term) - self.gamma
            ok = valid & (HL >= self.min_child_weight) & (HR >= self.min_child_weight)
            return np.where(ok, value, -np.inf)

        gain_missing_right = gain(G_left, H_left, G_right + G_missing, H_right + H_missing)
        gain_missing_left = gain(G_left + G_missing, H_left + H_missing, G_right, H_right)
        best_right = int(np.argmax(gain_missing_right))
        best_left = int(np.argmax(gain_missing_left))
        value_right = gain_missing_right.flat[best_right]
        value_left = gain_missing_left.flat[best_left]
        if max(value_right, value_left) <= 0.0:
            return None
        # When the node has no missing values both directions tie; XGBoost's
        # exact method then sends future missing values left, and so do we.
        missing_left = bool(value_left >= value_right)
        flat = best_left if missing_left else best_right
        i, j = np.unravel_index(flat, (n - 1, m))
        threshold = 0.5 * (Xs[i, j] + Xs[i + 1, j])
        if threshold <= Xs[i, j]:  # floating-point guard: keep x_i strictly left
            threshold = Xs[i + 1, j]
        return int(j), float(threshold), missing_left, float(max(value_left, value_right))

    # ---------------------------------------------------------- inference
    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.empty(X.shape[0])
        self._route(self.root, X, np.arange(X.shape[0]), out)  # type: ignore[arg-type]
        return out

    def _route(self, node: _XGBNode, X: np.ndarray, idx: np.ndarray, out: np.ndarray) -> None:
        if node.is_leaf:
            out[idx] = node.weight
            return
        x = X[idx, node.feature]
        go_left = np.where(np.isnan(x), node.missing_left, x < node.threshold)
        self._route(node.left, X, idx[go_left], out)  # type: ignore[arg-type]
        self._route(node.right, X, idx[~go_left], out)  # type: ignore[arg-type]

    def nodes(self) -> Iterator[_XGBNode]:
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node is None:
                continue
            yield node
            if not node.is_leaf:
                stack.append(node.right)
                stack.append(node.left)


# --------------------------------------------------------------------------- #
# Boosters
# --------------------------------------------------------------------------- #
class _BaseXGBoostScratch(BaseEstimator):
    """Shared boosting loop, subsampling, early stopping and importances."""

    _is_classifier: bool = False

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.3,
        max_depth: int = 6,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        colsample_bynode: float = 1.0,
        base_score: Optional[float] = None,
        early_stopping_rounds: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bynode = colsample_bynode
        self.base_score = base_score
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state

    # ----------------------------------------------------------- abstract
    def _prepare_targets(self, y: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _initial_margin(self, y: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Return the initial raw margin(s), shape ``(n_outputs,)``."""
        raise NotImplementedError

    def _grad_hess(self, y: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        """Return ``g, h`` of shape ``(n_samples, n_outputs)``."""
        raise NotImplementedError

    def _metric(self, y: np.ndarray, F: np.ndarray) -> float:  # pragma: no cover
        raise NotImplementedError

    _metric_name: str = ""

    # ---------------------------------------------------------------- fit
    def _validate(self) -> None:
        if not isinstance(self.n_estimators, numbers.Integral) or self.n_estimators < 1:
            raise ValueError(f"n_estimators must be a positive integer; got {self.n_estimators!r}")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not isinstance(self.max_depth, numbers.Integral) or self.max_depth < 1:
            raise ValueError("max_depth must be a positive integer")
        if self.reg_lambda < 0 or self.gamma < 0 or self.min_child_weight < 0:
            raise ValueError("reg_lambda, gamma and min_child_weight must be non-negative")
        for name in ("subsample", "colsample_bytree", "colsample_bynode"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]; got {value}")
        if self.early_stopping_rounds is not None and self.early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be a positive integer or None")

    def _check_X(self, X: np.ndarray, fitted: bool = True) -> np.ndarray:
        if fitted:
            check_is_fitted(self, "estimators_")
        X = check_array(X, dtype=np.float64, ensure_all_finite=False)
        if np.isinf(X).any():
            raise ValueError("X contains infinite values; only NaN is treated as missing")
        if fitted and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was fitted with {self.n_features_in_}"
            )
        return X

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> _BaseXGBoostScratch:
        """Run the boosting rounds.

        Args:
            X: Feature matrix; NaN entries are treated as missing.
            y: Targets.
            eval_set: Optional ``(X_val, y_val)`` on which the metric is tracked
                every round (``evals_result_["validation"]``) and used for
                early stopping when ``early_stopping_rounds`` is set.

        Returns:
            The fitted model.

        Raises:
            ValueError: If ``early_stopping_rounds`` is set without ``eval_set``.
        """
        X, y = check_X_y(
            X, y, dtype=np.float64, ensure_all_finite="allow-nan", y_numeric=not self._is_classifier
        )
        self._validate()
        if self.early_stopping_rounds is not None and eval_set is None:
            raise ValueError("early_stopping_rounds requires eval_set=(X_val, y_val)")
        self.n_features_in_ = X.shape[1]
        n_samples, n_features = X.shape
        rng = check_random_state(self.random_state)
        y_arr = self._prepare_targets(y)

        self.init_margin_ = self._initial_margin(y_arr)
        n_outputs = self.init_margin_.shape[0]
        F = np.tile(self.init_margin_, (n_samples, 1))

        X_val = y_val = F_val = None
        if eval_set is not None:
            X_val = self._check_X(eval_set[0], fitted=False)
            y_val = self._prepare_targets(np.asarray(eval_set[1]), fitting=False)  # type: ignore[call-arg]
            F_val = np.tile(self.init_margin_, (X_val.shape[0], 1))

        self.estimators_: List[List[_XGBTree]] = []
        self.evals_result_: Dict[str, Dict[str, List[float]]] = {"train": {self._metric_name: []}}
        if eval_set is not None:
            self.evals_result_["validation"] = {self._metric_name: []}
        best_score, best_iteration, rounds_without_gain = np.inf, 0, 0

        n_cols_tree = max(1, round(self.colsample_bytree * n_features))
        n_rows = max(1, round(self.subsample * n_samples))
        for m in range(self.n_estimators):
            g, h = self._grad_hess(y_arr, F)
            rows = (
                np.arange(n_samples) if n_rows >= n_samples else rng.choice(n_samples, n_rows, replace=False)
            )
            features = (
                np.arange(n_features)
                if n_cols_tree >= n_features
                else np.sort(rng.choice(n_features, n_cols_tree, replace=False))
            )
            stage: List[_XGBTree] = []
            for k in range(n_outputs):
                tree = _XGBTree(
                    max_depth=self.max_depth,
                    reg_lambda=self.reg_lambda,
                    gamma=self.gamma,
                    min_child_weight=self.min_child_weight,
                    learning_rate=self.learning_rate,
                    colsample_bynode=self.colsample_bynode,
                ).fit(X[rows], g[rows, k], h[rows, k], features, rng)
                stage.append(tree)
                F[:, k] += tree.predict(X)
            self.estimators_.append(stage)
            self.evals_result_["train"][self._metric_name].append(self._metric(y_arr, F))

            if eval_set is not None:
                for k, tree in enumerate(stage):
                    F_val[:, k] += tree.predict(X_val)  # type: ignore[index]
                score = self._metric(y_val, F_val)  # type: ignore[arg-type]
                self.evals_result_["validation"][self._metric_name].append(score)
                if score < best_score:
                    best_score, best_iteration, rounds_without_gain = score, m, 0
                else:
                    rounds_without_gain += 1
                    if (
                        self.early_stopping_rounds is not None
                        and rounds_without_gain >= self.early_stopping_rounds
                    ):
                        break

        if self.early_stopping_rounds is not None:
            self.best_iteration_ = best_iteration
            self.best_score_ = float(best_score)
            self.estimators_ = self.estimators_[: best_iteration + 1]
        self.n_estimators_ = len(self.estimators_)
        self.feature_importances_ = self._normalised(self.get_feature_importance("gain"))
        return self

    # -------------------------------------------------------- importances
    def get_feature_importance(self, importance_type: str = "gain") -> np.ndarray:
        """Raw per-feature importance totals.

        Args:
            importance_type: ``"gain"`` (total split gain), ``"weight"``
                (number of splits) or ``"cover"`` (total Hessian sum of the
                split nodes).

        Returns:
            Array of shape ``(n_features_in_,)``.
        """
        check_is_fitted(self, "estimators_")
        if importance_type not in ("gain", "weight", "cover"):
            raise ValueError("importance_type must be 'gain', 'weight' or 'cover'")
        totals = np.zeros(self.n_features_in_)
        for stage in self.estimators_:
            for tree in stage:
                for node in tree.nodes():
                    if node.is_leaf:
                        continue
                    if importance_type == "gain":
                        totals[node.feature] += node.gain
                    elif importance_type == "weight":
                        totals[node.feature] += 1.0
                    else:
                        totals[node.feature] += node.sum_hess
        return totals

    @staticmethod
    def _normalised(values: np.ndarray) -> np.ndarray:
        total = values.sum()
        return values / total if total > 0 else values

    # ---------------------------------------------------------- inference
    def _staged_margin(self, X: np.ndarray) -> Iterator[np.ndarray]:
        F = np.tile(self.init_margin_, (X.shape[0], 1))
        for stage in self.estimators_:
            for k, tree in enumerate(stage):
                F[:, k] += tree.predict(X)
            yield F.copy()

    def _margin(self, X: np.ndarray) -> np.ndarray:
        F = None
        for F in self._staged_margin(X):
            pass
        return F  # type: ignore[return-value]

    def __sklearn_tags__(self):
        """Advertise NaN support to scikit-learn's estimator checks (sklearn >= 1.6)."""
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class XGBoostRegressorScratch(RegressorMixin, _BaseXGBoostScratch):
    """Second-order boosted trees for squared-error regression (see module docstring).

    Args:
        n_estimators: Maximum number of boosting rounds.
        learning_rate: Shrinkage $\\eta$.
        max_depth: Maximum tree depth.
        reg_lambda: L2 penalty $\\lambda$ on leaf weights.
        gamma: Minimum gain to split.
        min_child_weight: Minimum Hessian sum per child.
        subsample: Row fraction per tree.
        colsample_bytree: Feature fraction per tree.
        colsample_bynode: Feature fraction per split.
        base_score: Initial prediction; ``None`` uses the mean of ``y``.
        early_stopping_rounds: Stop when the validation RMSE has not improved
            for this many rounds (requires ``eval_set``).
        random_state: Seed for subsampling.

    Attributes:
        estimators_: List of rounds; each round holds one :class:`_XGBTree`.
        init_margin_: Initial prediction as a length-1 array.
        evals_result_: ``{"train": {"rmse": [...]}, "validation": {...}}``.
        best_iteration_, best_score_: Set when early stopping is enabled.
        n_estimators_: Number of rounds actually kept.
        feature_importances_: Normalised total gain per feature.
    """

    _metric_name = "rmse"

    def _prepare_targets(self, y: np.ndarray, fitting: bool = True) -> np.ndarray:
        return np.asarray(y, dtype=np.float64)

    def _initial_margin(self, y: np.ndarray) -> np.ndarray:
        return np.array([float(np.mean(y)) if self.base_score is None else float(self.base_score)])

    def _grad_hess(self, y: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return F - y[:, None], np.ones_like(F)

    def _metric(self, y: np.ndarray, F: np.ndarray) -> float:
        return float(np.sqrt(np.mean((F[:, 0] - y) ** 2)))

    def staged_predict(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield predictions after each boosting round."""
        X = self._check_X(X)
        for F in self._staged_margin(X):
            yield F[:, 0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with all kept rounds."""
        return self._margin(self._check_X(X))[:, 0]


class XGBoostClassifierScratch(ClassifierMixin, _BaseXGBoostScratch):
    """Second-order boosted trees for classification (see module docstring).

    Two classes use the binary logistic loss with a single tree per round;
    ``K > 2`` classes use the softmax loss with ``K`` trees per round.

    Args:
        n_estimators: Maximum number of boosting rounds.
        learning_rate: Shrinkage $\\eta$.
        max_depth: Maximum tree depth.
        reg_lambda: L2 penalty $\\lambda$ on leaf weights.
        gamma: Minimum gain to split.
        min_child_weight: Minimum Hessian sum per child.
        subsample: Row fraction per tree.
        colsample_bytree: Feature fraction per tree.
        colsample_bynode: Feature fraction per split.
        base_score: Initial *probability* (binary; the margin is its logit) or
            ``None`` for 0.5 (zero margin). Ignored for multiclass, which
            starts from zero margins.
        early_stopping_rounds: Stop when the validation log-loss has not
            improved for this many rounds (requires ``eval_set``).
        random_state: Seed for subsampling.

    Attributes:
        classes_: Sorted unique class labels.
        n_classes_: Number of classes.
        estimators_: List of rounds; each round holds 1 or ``K`` trees.
        evals_result_: Per-round ``logloss`` / ``mlogloss``.
        best_iteration_, best_score_: Set when early stopping is enabled.
        n_estimators_: Number of rounds actually kept.
        feature_importances_: Normalised total gain per feature.
    """

    _is_classifier = True

    @property
    def _metric_name(self) -> str:  # type: ignore[override]
        return "mlogloss" if getattr(self, "n_classes_", 2) > 2 else "logloss"

    def _prepare_targets(self, y: np.ndarray, fitting: bool = True) -> np.ndarray:
        if fitting:
            self.classes_, y_enc = np.unique(y, return_inverse=True)
            self.n_classes_ = len(self.classes_)
            if self.n_classes_ < 2:
                raise ValueError("Need at least two classes")
            return y_enc.astype(np.intp)
        lookup = {label: k for k, label in enumerate(self.classes_)}
        try:
            return np.array([lookup[label] for label in y], dtype=np.intp)
        except KeyError as exc:
            raise ValueError(f"eval_set contains a label unseen during training: {exc}") from None

    def _initial_margin(self, y: np.ndarray) -> np.ndarray:
        if self.n_classes_ == 2:
            p = 0.5 if self.base_score is None else float(self.base_score)
            if not 0.0 < p < 1.0:
                raise ValueError("base_score must lie in (0, 1) for classification")
            return np.array([np.log(p / (1.0 - p))])
        return np.zeros(self.n_classes_)

    def _proba(self, F: np.ndarray) -> np.ndarray:
        if self.n_classes_ == 2:
            p = 1.0 / (1.0 + np.exp(-F[:, 0]))
            return np.column_stack([1.0 - p, p])
        z = F - F.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _grad_hess(self, y: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        proba = self._proba(F)
        if self.n_classes_ == 2:
            p = proba[:, 1:2]
            return p - y[:, None], p * (1.0 - p)
        onehot = np.zeros_like(proba)
        onehot[np.arange(y.shape[0]), y] = 1.0
        return proba - onehot, proba * (1.0 - proba)

    def _metric(self, y: np.ndarray, F: np.ndarray) -> float:
        proba = np.clip(self._proba(F), 1e-15, 1.0)
        return float(-np.mean(np.log(proba[np.arange(y.shape[0]), y])))

    def staged_predict_proba(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield class probabilities after each boosting round."""
        X = self._check_X(X)
        for F in self._staged_margin(X):
            yield self._proba(F)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Sigmoid / softmax of the boosted margins."""
        return self._proba(self._margin(self._check_X(X)))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Raw margins: ``(n_samples,)`` for binary, ``(n_samples, K)`` otherwise."""
        F = self._margin(self._check_X(X))
        return F[:, 0] if self.n_classes_ == 2 else F

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the most probable class."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def staged_predict(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield class predictions after each boosting round."""
        for proba in self.staged_predict_proba(X):
            yield self.classes_[np.argmax(proba, axis=1)]
