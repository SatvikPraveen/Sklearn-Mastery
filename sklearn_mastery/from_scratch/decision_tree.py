"""CART decision trees implemented from first principles with NumPy.

This module re-implements the Classification And Regression Trees (CART)
algorithm of Breiman, Friedman, Olshen and Stone (1984) without using any of
scikit-learn's tree machinery. The estimators are nonetheless fully
scikit-learn compatible (``BaseEstimator`` + mixins, ``fit`` returns ``self``,
``get_params``/``set_params``, ``clone`` and pickling all work) so that they
can be dropped into pipelines, cross-validation and the ensemble methods of
:mod:`sklearn_mastery.from_scratch`.

Algorithm
---------
A CART tree is grown top-down by *exact greedy* search. At every node with
sample set $S$, weighted node size $W = \\sum_{i \\in S} w_i$ and
impurity $I(S)$, every feature $j$ is scanned: the samples are
sorted by $x_{ij}$ and every midpoint between two consecutive *distinct*
values is a candidate threshold $\\theta$. The candidate partitions
$S_L = \\{i : x_{ij} \\le \\theta\\}$ and $S_R = S \\setminus S_L$
are scored by the weighted child impurity

$$
Q(j, \\theta) = \\frac{W_L}{W} I(S_L) + \\frac{W_R}{W} I(S_R),
$$

and the split minimising $Q$ (equivalently, maximising the impurity
decrease $\\Delta I = I(S) - Q$) is chosen. Because the child impurities
depend only on cumulative sums of the sorted targets, the scan over all
thresholds *and* all features is vectorised into a handful of NumPy
``cumsum`` calls per node.

Impurity criteria (``p_k`` are the weighted class proportions in a node,
``\\bar y`` the weighted mean target):

* ``gini``:      $I = 1 - \\sum_k p_k^2$
* ``entropy``:   $I = -\\sum_k p_k \\log_2 p_k$
* ``squared_error`` (``mse``): $I = \\frac{1}{W}\\sum_i w_i (y_i - \\bar y)^2$
* ``friedman_mse``: node impurity is the weighted variance as above, but
  candidate splits are ranked by Friedman's (2001) improvement score
  $\\frac{W_L W_R}{W_L + W_R}(\\bar y_L - \\bar y_R)^2$, which is
  the *unnormalised* variance reduction and the criterion used inside
  gradient boosting.

Feature importances follow scikit-learn: for every internal node ``t`` the
weighted impurity decrease
$W_t I_t - W_{t_L} I_{t_L} - W_{t_R} I_{t_R}$ is credited to the
split feature; the per-feature totals are then normalised to sum to one
(mean decrease in impurity, MDI).

Minimal cost-complexity pruning (``ccp_alpha``) implements Breiman's
weakest-link algorithm; see :meth:`_BaseTree._prune_weakest_link` for the
derivation of the effective alphas.

References
----------
Breiman, L., Friedman, J., Olshen, R. and Stone, C. (1984).
*Classification and Regression Trees*. Wadsworth.

Friedman, J. H. (2001). Greedy function approximation: a gradient boosting
machine. *Annals of Statistics* 29(5), 1189-1232.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_sample_weight, check_array, check_is_fitted, check_X_y

__all__ = ["DecisionTreeClassifierScratch", "DecisionTreeRegressorScratch"]

MaxFeatures = Union[int, float, str, None]


# --------------------------------------------------------------------------- #
# Tree data structure
# --------------------------------------------------------------------------- #
@dataclass
class _Node:
    """A node of a binary CART tree.

    Attributes:
        depth: Depth of the node (root is 0).
        n_samples: Number of training samples reaching the node.
        weighted_n_samples: Sum of the sample weights reaching the node.
        impurity: Node impurity under the tree's criterion.
        value: Leaf prediction. For classification, the weighted class
            proportions ``(n_classes,)``; for regression, a length-1 array
            holding the weighted mean target.
        feature: Index of the split feature (``-1`` for leaves).
        threshold: Split threshold; samples with ``x[feature] <= threshold``
            go left.
        left: Left child (``None`` for leaves).
        right: Right child (``None`` for leaves).
        node_id: Pre-order index assigned after the tree is finalised.
    """

    depth: int
    n_samples: int
    weighted_n_samples: float
    impurity: float
    value: np.ndarray
    feature: int = -1
    threshold: float = np.nan
    left: Optional[_Node] = None
    right: Optional[_Node] = None
    node_id: int = -1

    @property
    def is_leaf(self) -> bool:
        return self.left is None

    def make_leaf(self) -> None:
        """Turn this node into a leaf by discarding its subtree."""
        self.left = None
        self.right = None
        self.feature = -1
        self.threshold = np.nan


def _preorder(node: _Node) -> Iterator[_Node]:
    """Yield the nodes of a subtree in pre-order (parent, left, right)."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        if not current.is_leaf:
            stack.append(current.right)  # type: ignore[arg-type]
            stack.append(current.left)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Impurity functions (vectorised over leading axes)
# --------------------------------------------------------------------------- #
def _gini(counts: np.ndarray) -> np.ndarray:
    """Gini impurity ``1 - sum_k p_k^2`` from weighted class counts on the last axis."""
    total = counts.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = counts / total
    return 1.0 - np.sum(p * p, axis=-1)


def _entropy(counts: np.ndarray) -> np.ndarray:
    """Shannon entropy ``-sum_k p_k log2 p_k`` (with ``0 log 0 = 0``)."""
    total = counts.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = counts / total
        plogp = np.where(p > 0, p * np.log2(p), 0.0)
    return -np.sum(plogp, axis=-1)


_CLASSIFICATION_CRITERIA = {"gini": _gini, "entropy": _entropy, "log_loss": _entropy}
_REGRESSION_CRITERIA = ("squared_error", "mse", "friedman_mse")


def _resolve_max_features(max_features: MaxFeatures, n_features: int) -> int:
    """Translate the ``max_features`` option into an integer number of features."""
    if max_features is None:
        return n_features
    if isinstance(max_features, str):
        if max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if max_features == "log2":
            return max(1, int(np.log2(n_features)))
        raise ValueError(
            f"max_features must be 'sqrt', 'log2', an int, a float or None; got {max_features!r}"
        )
    if isinstance(max_features, numbers.Integral):
        if not 1 <= int(max_features) <= n_features:
            raise ValueError(f"max_features={max_features} must be in [1, {n_features}]")
        return int(max_features)
    if isinstance(max_features, numbers.Real):
        if not 0.0 < float(max_features) <= 1.0:
            raise ValueError(f"A float max_features must lie in (0, 1]; got {max_features}")
        return max(1, int(float(max_features) * n_features))
    raise TypeError(f"Unsupported max_features type: {type(max_features).__name__}")


# --------------------------------------------------------------------------- #
# Base tree
# --------------------------------------------------------------------------- #
class _BaseTree(BaseEstimator):
    """Shared CART machinery for the classifier and regressor.

    Subclasses set the class attribute ``_is_classifier`` and implement
    ``_encode_targets`` (target validation) and ``_node_stats`` (impurity
    and prediction value of a node) and ``_split_scores`` (vectorised score of
    every candidate threshold). Everything else - recursive growth, feature
    subsampling, pruning, prediction, importances - lives here.
    """

    _is_classifier: bool = False

    def __init__(
        self,
        criterion: str,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: MaxFeatures = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        ccp_alpha: float = 0.0,
    ) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.ccp_alpha = ccp_alpha

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> _BaseTree:
        """Grow the tree on ``(X, y)``.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Targets of shape ``(n_samples,)``.
            sample_weight: Optional non-negative weights. All impurities,
                node values and importances are weighted; ``min_samples_*``
                constraints count *unweighted* samples (as in scikit-learn).

        Returns:
            The fitted estimator.

        Raises:
            ValueError: On invalid hyper-parameters or malformed inputs.
        """
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=not self._is_classifier)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64)
        if np.any(sample_weight == 0):  # zero-weight rows are ignored entirely, as in scikit-learn
            keep = sample_weight > 0
            X, y, sample_weight = X[keep], y[keep], sample_weight[keep]
        self._validate_hyperparameters()
        self.n_features_in_ = X.shape[1]
        self.max_features_ = _resolve_max_features(self.max_features, self.n_features_in_)
        y_enc = self._encode_targets(y)
        rng = check_random_state(self.random_state)

        self.tree_ = self._grow(X, y_enc, sample_weight, depth=0, rng=rng)
        if self.ccp_alpha > 0.0:
            self._prune_weakest_link(self.tree_, self.ccp_alpha)
        self._finalize()
        return self

    def _validate_hyperparameters(self) -> None:
        if self._is_classifier:
            if self.criterion not in _CLASSIFICATION_CRITERIA:
                raise ValueError(
                    f"criterion must be one of {sorted(_CLASSIFICATION_CRITERIA)}; got {self.criterion!r}"
                )
        elif self.criterion not in _REGRESSION_CRITERIA:
            raise ValueError(f"criterion must be one of {_REGRESSION_CRITERIA}; got {self.criterion!r}")
        if self.max_depth is not None and (
            not isinstance(self.max_depth, numbers.Integral) or self.max_depth < 1
        ):
            raise ValueError(f"max_depth must be a positive integer or None; got {self.max_depth!r}")
        if not isinstance(self.min_samples_split, numbers.Integral) or self.min_samples_split < 2:
            raise ValueError(f"min_samples_split must be an integer >= 2; got {self.min_samples_split!r}")
        if not isinstance(self.min_samples_leaf, numbers.Integral) or self.min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be an integer >= 1; got {self.min_samples_leaf!r}")
        if self.ccp_alpha < 0.0:
            raise ValueError(f"ccp_alpha must be non-negative; got {self.ccp_alpha}")

    # ------------------------------------------------------------- growing
    def _grow(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, depth: int, rng: np.random.RandomState
    ) -> _Node:
        """Recursively grow a subtree on the given samples."""
        impurity, value = self._node_stats(y, w)
        node = _Node(
            depth=depth,
            n_samples=X.shape[0],
            weighted_n_samples=float(w.sum()),
            impurity=float(impurity),
            value=value,
        )
        n = X.shape[0]
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n < self.min_samples_split
            or n < 2 * self.min_samples_leaf
            or impurity <= 1e-12
        ):
            return node

        split = self._find_best_split(X, y, w, rng)
        if split is None:  # no feature admits a valid partition (e.g. constant X)
            return node

        feature, threshold = split
        go_left = X[:, feature] <= threshold
        node.feature = feature
        node.threshold = threshold
        node.left = self._grow(X[go_left], y[go_left], w[go_left], depth + 1, rng)
        node.right = self._grow(X[~go_left], y[~go_left], w[~go_left], depth + 1, rng)
        return node

    def _find_best_split(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, rng: np.random.RandomState
    ) -> Optional[Tuple[int, float]]:
        """Exact greedy split search over a random subset of ``max_features_`` features.

        As in scikit-learn, if none of the drawn features admits a valid split
        the remaining features are inspected, so a node only becomes a leaf
        for lack of a split when *no* feature can partition it.
        """
        n_features = X.shape[1]
        if self.max_features_ >= n_features:
            candidates = np.arange(n_features)
            rest = np.empty(0, dtype=int)
        else:
            perm = rng.permutation(n_features)
            candidates, rest = perm[: self.max_features_], perm[self.max_features_ :]

        best = self._search_features(X, y, w, candidates)
        if best is None and rest.size:
            best = self._search_features(X, y, w, rest)
        return best

    def _search_features(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, features: np.ndarray
    ) -> Optional[Tuple[int, float]]:
        """Score every threshold of every feature in ``features`` in one vectorised pass.

        Returns:
            ``(feature_index, threshold)`` of the best valid split, or ``None``.
        """
        n = X.shape[0]
        Xf = X[:, features]
        order = np.argsort(Xf, axis=0, kind="stable")  # (n, m)
        Xs = np.take_along_axis(Xf, order, axis=0)
        ws = w[order]  # (n, m) weights in sorted order

        # Candidate i splits sorted samples [0..i] | [i+1..n-1]; the threshold
        # is the midpoint, so it is only meaningful where consecutive values differ.
        n_left = np.arange(1, n)[:, None]
        valid = (Xs[1:] > Xs[:-1]) & (n_left >= self.min_samples_leaf) & (n - n_left >= self.min_samples_leaf)
        if not valid.any():
            return None

        scores = self._split_scores(y, ws, order)  # (n-1, m); lower is better
        scores = np.where(valid & np.isfinite(scores), scores, np.inf)
        i, j = np.unravel_index(int(np.argmin(scores)), scores.shape)
        if not np.isfinite(scores[i, j]):
            return None

        threshold = 0.5 * (Xs[i, j] + Xs[i + 1, j])
        if threshold >= Xs[i + 1, j]:  # guard against floating-point round-up
            threshold = Xs[i, j]
        return int(features[j]), float(threshold)

    # ------------------------------------------------------------- pruning
    def _prune_weakest_link(self, root: _Node, alpha: float) -> None:
        """Minimal cost-complexity pruning (Breiman et al. 1984, ch. 3).

        For a tree $T$ define the total (weighted) leaf impurity
        $R(T) = \\sum_{t \\in \\text{leaves}(T)} \\frac{W_t}{W_{root}} I_t$
        and the cost-complexity $R_\\alpha(T) = R(T) + \\alpha |T|$,
        where $|T|$ is the number of leaves. Collapsing the subtree
        $T_t$ rooted at an internal node $t$ into a single leaf
        raises the impurity term by $R(t) - R(T_t)$ and lowers the
        complexity term by $\\alpha (|T_t| - 1)$. The two effects cancel
        at the *effective alpha*

        $$
        \\alpha_{\\text{eff}}(t) = \\frac{R(t) - R(T_t)}{|T_t| - 1},
        $$

        so a subtree is worth keeping only while $\\alpha <
        \\alpha_{\\text{eff}}(t)$. Weakest-link pruning repeatedly collapses the
        internal node with the *smallest* effective alpha and recomputes the
        alphas of its ancestors, stopping once every remaining
        $\\alpha_{\\text{eff}}(t)$ exceeds ``ccp_alpha``. Nodes whose
        effective alpha equals ``ccp_alpha`` are pruned, matching scikit-learn.
        """
        total_weight = root.weighted_n_samples

        def leaf_risk(node: _Node) -> float:
            return node.weighted_n_samples / total_weight * node.impurity

        def collect(node: _Node, candidates: List[Tuple[float, _Node]]) -> Tuple[float, int]:
            """Return (R(T_t), |T_t|) and record alpha_eff of every internal node (post-order)."""
            if node.is_leaf:
                return leaf_risk(node), 1
            risk_l, leaves_l = collect(node.left, candidates)  # type: ignore[arg-type]
            risk_r, leaves_r = collect(node.right, candidates)  # type: ignore[arg-type]
            risk, leaves = risk_l + risk_r, leaves_l + leaves_r
            candidates.append(((leaf_risk(node) - risk) / (leaves - 1), node))
            return risk, leaves

        while True:
            candidates: List[Tuple[float, _Node]] = []
            collect(root, candidates)
            if not candidates:
                return
            alpha_eff, weakest = min(candidates, key=lambda item: item[0])
            if alpha_eff > alpha:
                return
            weakest.make_leaf()

    # ------------------------------------------------------------ finalise
    def _finalize(self) -> None:
        """Assign pre-order node ids, cache the value table and compute importances."""
        self._nodes: List[_Node] = list(_preorder(self.tree_))
        for node_id, node in enumerate(self._nodes):
            node.node_id = node_id
        self.node_count_ = len(self._nodes)
        self._values = np.stack([node.value for node in self._nodes])
        self.feature_importances_ = self._compute_feature_importances()

    def _compute_feature_importances(self) -> np.ndarray:
        """Normalised total weighted impurity decrease per feature (MDI)."""
        importances = np.zeros(self.n_features_in_)
        for node in self._nodes:
            if node.is_leaf:
                continue
            left, right = node.left, node.right
            decrease = (
                node.weighted_n_samples * node.impurity
                - left.weighted_n_samples * left.impurity  # type: ignore[union-attr]
                - right.weighted_n_samples * right.impurity  # type: ignore[union-attr]
            )
            importances[node.feature] += decrease
        total = importances.sum()
        if total > 0:
            importances /= total
        return importances

    def _set_leaf_values(self, leaf_values: Dict[int, np.ndarray]) -> None:
        """Overwrite leaf predictions (used by gradient boosting's line search).

        Args:
            leaf_values: Mapping from leaf ``node_id`` to the new value.
        """
        for node_id, value in leaf_values.items():
            node = self._nodes[node_id]
            if not node.is_leaf:
                raise ValueError(f"node {node_id} is not a leaf")
            node.value = np.atleast_1d(np.asarray(value, dtype=np.float64))
        self._values = np.stack([node.value for node in self._nodes])

    # ---------------------------------------------------------- inference
    def _check_X(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "tree_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the tree was fitted with {self.n_features_in_}"
            )
        return X

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Return the pre-order id of the leaf each sample ends up in.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with leaf node ids.
        """
        X = self._check_X(X)
        leaf_ids = np.empty(X.shape[0], dtype=np.intp)
        self._route(self.tree_, X, np.arange(X.shape[0]), leaf_ids)
        return leaf_ids

    def _route(self, node: _Node, X: np.ndarray, idx: np.ndarray, out: np.ndarray) -> None:
        """Push the samples ``idx`` down the subtree, writing leaf ids into ``out``."""
        if node.is_leaf:
            out[idx] = node.node_id
            return
        go_left = X[idx, node.feature] <= node.threshold
        self._route(node.left, X, idx[go_left], out)  # type: ignore[arg-type]
        self._route(node.right, X, idx[~go_left], out)  # type: ignore[arg-type]

    def _leaf_values(self, X: np.ndarray) -> np.ndarray:
        leaf_ids = self.apply(X)  # validates X and the fitted state
        return self._values[leaf_ids]

    def get_depth(self) -> int:
        """Return the depth of the tree (a single leaf has depth 0)."""
        check_is_fitted(self, "tree_")
        return max(node.depth for node in self._nodes)

    def get_n_leaves(self) -> int:
        """Return the number of leaves."""
        check_is_fitted(self, "tree_")
        return sum(node.is_leaf for node in self._nodes)

    def export_text(self, feature_names: Optional[Sequence[str]] = None, decimals: int = 3) -> str:
        """Render the tree as an indented list of rules (like ``sklearn.tree.export_text``).

        Args:
            feature_names: Optional names for the features.
            decimals: Number of decimals used for thresholds and values.

        Returns:
            A multi-line string.
        """
        check_is_fitted(self, "tree_")
        names = (
            list(feature_names)
            if feature_names is not None
            else [f"feature_{i}" for i in range(self.n_features_in_)]
        )
        lines: List[str] = []

        def describe_leaf(node: _Node) -> str:
            if self._is_classifier:
                return f"class: {self.classes_[int(np.argmax(node.value))]}"  # type: ignore[attr-defined]
            return f"value: [{node.value[0]:.{decimals}f}]"

        def walk(node: _Node, indent: str) -> None:
            if node.is_leaf:
                lines.append(f"{indent}|--- {describe_leaf(node)}")
                return
            name = names[node.feature]
            lines.append(f"{indent}|--- {name} <= {node.threshold:.{decimals}f}")
            walk(node.left, indent + "|   ")  # type: ignore[arg-type]
            lines.append(f"{indent}|--- {name} >  {node.threshold:.{decimals}f}")
            walk(node.right, indent + "|   ")  # type: ignore[arg-type]

        walk(self.tree_, "")
        return "\n".join(lines) + "\n"

    # ----------------------------------------------------------- abstract
    def _encode_targets(self, y: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _node_stats(self, y: np.ndarray, w: np.ndarray) -> Tuple[float, np.ndarray]:  # pragma: no cover
        raise NotImplementedError

    def _split_scores(
        self, y: np.ndarray, ws: np.ndarray, order: np.ndarray
    ) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #
class DecisionTreeClassifierScratch(ClassifierMixin, _BaseTree):
    """CART classification tree built from scratch.

    Splits are chosen by exact greedy search minimising the weighted child
    impurity under the ``gini`` or ``entropy`` criterion; see the module
    docstring for the maths. Leaves store the weighted class proportions, so
    :meth:`predict_proba` is the empirical class distribution of the leaf.

    Args:
        criterion: ``"gini"`` or ``"entropy"`` (``"log_loss"`` is an alias).
        max_depth: Maximum depth; ``None`` grows until leaves are pure or
            the ``min_samples_*`` constraints bite.
        min_samples_split: Minimum number of samples a node needs to be split.
        min_samples_leaf: Minimum number of samples in each child.
        max_features: Number of features drawn at random at *each split*:
            an int, a fraction in ``(0, 1]``, ``"sqrt"``, ``"log2"`` or
            ``None`` (all features).
        random_state: Seed for the feature subsampling.
        ccp_alpha: Complexity parameter for minimal cost-complexity pruning;
            ``0`` disables pruning.

    Attributes:
        classes_: Sorted unique class labels.
        n_classes_: Number of classes.
        n_features_in_: Number of features seen during fit.
        max_features_: Resolved number of features per split.
        tree_: Root :class:`_Node`.
        node_count_: Number of nodes after pruning.
        feature_importances_: Normalised mean decrease in impurity.

    Example:
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> clf = DecisionTreeClassifierScratch(max_depth=3).fit(X, y)
        >>> clf.predict(X[:2]).tolist()
        [0, 0]
    """

    _is_classifier = True

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: MaxFeatures = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        ccp_alpha: float = 0.0,
    ) -> None:
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            ccp_alpha=ccp_alpha,
        )

    def _encode_targets(self, y: np.ndarray) -> np.ndarray:
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return y_enc.astype(np.intp)

    def _impurity_fn(self):
        return _CLASSIFICATION_CRITERIA[self.criterion]

    def _node_stats(self, y: np.ndarray, w: np.ndarray) -> Tuple[float, np.ndarray]:
        counts = np.bincount(y, weights=w, minlength=self.n_classes_)
        total = counts.sum()
        value = counts / total if total > 0 else np.full(self.n_classes_, 1.0 / self.n_classes_)
        return float(self._impurity_fn()(counts)), value

    def _split_scores(self, y: np.ndarray, ws: np.ndarray, order: np.ndarray) -> np.ndarray:
        """Weighted child impurity ``(W_L I_L + W_R I_R) / W`` for every threshold and feature.

        ``order`` has shape ``(n, m)``; column ``j`` is the sort permutation of
        feature ``j``. Weighted one-hot targets are cumulated along the sorted
        order to obtain the left-child class counts for every cut position.
        """
        n = y.shape[0]
        onehot = np.zeros((n, self.n_classes_))
        onehot[np.arange(n), y] = 1.0
        sorted_onehot = onehot[order] * ws[..., None]  # (n, m, K)
        left = np.cumsum(sorted_onehot, axis=0)[:-1]  # (n-1, m, K)
        right = left[-1] + sorted_onehot[-1] - left
        w_left = left.sum(axis=-1)
        w_right = right.sum(axis=-1)
        impurity = self._impurity_fn()
        with np.errstate(divide="ignore", invalid="ignore"):
            score = (w_left * impurity(left) + w_right * impurity(right)) / (w_left + w_right)
        score[(w_left <= 0) | (w_right <= 0)] = np.inf
        return score

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the weighted class proportions of the leaf each sample falls in."""
        return self._leaf_values(X)

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ``log(predict_proba(X))``."""
        return np.log(self.predict_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the majority class of the leaf each sample falls in."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# --------------------------------------------------------------------------- #
# Regressor
# --------------------------------------------------------------------------- #
class DecisionTreeRegressorScratch(RegressorMixin, _BaseTree):
    """CART regression tree built from scratch.

    Splits minimise the weighted within-child variance (``squared_error``) or
    maximise Friedman's improvement score (``friedman_mse``); leaves predict
    the weighted mean of their training targets. See the module docstring.

    Args:
        criterion: ``"squared_error"`` (alias ``"mse"``) or ``"friedman_mse"``.
        max_depth: Maximum depth; ``None`` for unlimited.
        min_samples_split: Minimum number of samples a node needs to be split.
        min_samples_leaf: Minimum number of samples in each child.
        max_features: Features drawn at random at each split (see classifier).
        random_state: Seed for the feature subsampling.
        ccp_alpha: Complexity parameter for minimal cost-complexity pruning.

    Attributes:
        n_features_in_: Number of features seen during fit.
        max_features_: Resolved number of features per split.
        tree_: Root :class:`_Node`.
        node_count_: Number of nodes after pruning.
        feature_importances_: Normalised mean decrease in impurity.
    """

    _is_classifier = False

    def __init__(
        self,
        criterion: str = "squared_error",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: MaxFeatures = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        ccp_alpha: float = 0.0,
    ) -> None:
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            ccp_alpha=ccp_alpha,
        )

    def _encode_targets(self, y: np.ndarray) -> np.ndarray:
        return y.astype(np.float64)

    def _node_stats(self, y: np.ndarray, w: np.ndarray) -> Tuple[float, np.ndarray]:
        total = w.sum()
        if total <= 0:
            return 0.0, np.array([0.0])
        mean = float(np.dot(w, y) / total)
        variance = float(np.dot(w, (y - mean) ** 2) / total)
        return variance, np.array([mean])

    def _split_scores(self, y: np.ndarray, ws: np.ndarray, order: np.ndarray) -> np.ndarray:
        """Score every cut of every feature from cumulative weighted moments.

        With ``S = sum w y`` and ``Q = sum w y^2`` over a child,
        ``W * var = Q - S^2 / W``, so the weighted child variance of a cut is
        ``(Q_L - S_L^2/W_L + Q_R - S_R^2/W_R) / W`` and needs only prefix sums.
        For ``friedman_mse`` the negative improvement
        ``-(W_L W_R / (W_L + W_R)) (mean_L - mean_R)^2`` is returned instead.
        """
        ys = y[order]  # (n, m)
        wy = ws * ys
        w_left = np.cumsum(ws, axis=0)[:-1]
        s_left = np.cumsum(wy, axis=0)[:-1]
        q_left = np.cumsum(wy * ys, axis=0)[:-1]
        w_total, s_total, q_total = ws.sum(axis=0), wy.sum(axis=0), (wy * ys).sum(axis=0)
        w_right, s_right, q_right = w_total - w_left, s_total - s_left, q_total - q_left
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.criterion == "friedman_mse":
                diff = s_left / w_left - s_right / w_right
                score = -(w_left * w_right / (w_left + w_right)) * diff * diff
            else:
                sse = (q_left - s_left**2 / w_left) + (q_right - s_right**2 / w_right)
                score = np.maximum(sse, 0.0) / w_total
        score[(w_left <= 0) | (w_right <= 0)] = np.inf
        return score

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the weighted mean target of the leaf each sample falls in."""
        return self._leaf_values(X)[:, 0]
