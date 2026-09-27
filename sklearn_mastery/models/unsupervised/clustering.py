"""Clustering wrappers with a uniform, scikit-learn-compatible interface.

Every wrapper in this module derives from :class:`ClusteringModel`, which
combines :class:`sklearn.base.BaseEstimator` and
:class:`sklearn.base.ClusterMixin`. The wrappers therefore work inside
scikit-learn pipelines, ``clone``, ``GridSearchCV`` and friends while adding
a few conveniences on top of the raw estimators:

* ``fit`` / ``fit_predict`` / ``predict`` on every model, including the
  transductive algorithms (hierarchical, spectral, DBSCAN, OPTICS) that
  scikit-learn ships without ``predict``. For those, prediction on the
  training data returns the fitted ``labels_`` and prediction on new data is
  a nearest-neighbour assignment to the labelled training points.
* A shared :meth:`ClusteringModel.evaluate` that reports internal validity
  indices (silhouette, Calinski-Harabasz, Davies-Bouldin) and, when ground
  truth is available, external agreement scores (ARI, AMI, homogeneity, ...).
* ``save_model`` / ``load_model`` for joblib persistence.
* Automatic hyper-parameter selection helpers: :func:`find_optimal_k`
  (silhouette / Calinski-Harabasz / Davies-Bouldin / BIC / AIC / elbow) and
  :func:`estimate_eps` (k-distance knee heuristic for DBSCAN), together with
  the ``*Enhanced`` / :class:`AdaptiveKMeans` wrappers that use them when the
  cluster count or ``eps`` is left unspecified.

Example:
    >>> from sklearn.datasets import make_blobs
    >>> X, y = make_blobs(n_samples=200, centers=3, random_state=0)
    >>> model = KMeansModel(n_clusters=3, random_state=0).fit(X)
    >>> labels = model.predict(X)
    >>> metrics = model.evaluate(X, y_true=y)
    >>> round(metrics["adjusted_rand_score"], 2)
    1.0
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
    SpectralClustering,
    cluster_optics_dbscan,
)
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array, check_is_fitted

try:  # scikit-learn >= 1.6
    from sklearn.utils.validation import validate_data as _sk_validate_data
except ImportError:  # pragma: no cover - older scikit-learn
    _sk_validate_data = None

from sklearn_mastery.config.logging_config import LoggerMixin, get_logger

logger = get_logger(__name__)

#: Label used by density-based algorithms for points that belong to no cluster.
NOISE_LABEL = -1

ArrayLike = Union[np.ndarray, "list[list[float]]"]
EstimatorFactory = Callable[[int], Any]

_MAXIMISE_CRITERIA = ("silhouette", "calinski_harabasz")
_MINIMISE_CRITERIA = ("davies_bouldin", "bic", "aic")
_KNEE_CRITERIA = ("elbow", "inertia")
VALID_K_CRITERIA = _MAXIMISE_CRITERIA + _MINIMISE_CRITERIA + _KNEE_CRITERIA


# --------------------------------------------------------------------------- #
# Evaluation and model-selection helpers
# --------------------------------------------------------------------------- #
def evaluate_clustering(
    X: ArrayLike,
    labels: ArrayLike,
    y_true: Optional[ArrayLike] = None,
) -> Dict[str, float]:
    """Compute internal and (optionally) external clustering quality metrics.

    Internal indices are computed on the non-noise points only (labels other
    than :data:`NOISE_LABEL`) and are ``nan`` whenever fewer than two clusters
    or too few labelled points are available. External indices use all points
    so that noise assignments are penalised.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        labels: Predicted cluster labels of shape ``(n_samples,)``.
        y_true: Optional ground-truth labels of shape ``(n_samples,)``.

    Returns:
        Dictionary with ``n_clusters``, ``n_noise``, ``noise_fraction``,
        ``silhouette_score``, ``calinski_harabasz_score`` and
        ``davies_bouldin_score``; plus ``adjusted_rand_score``,
        ``adjusted_mutual_info_score``, ``homogeneity_score``,
        ``completeness_score`` and ``v_measure_score`` when ``y_true`` is given.

    Raises:
        ValueError: If ``labels`` (or ``y_true``) does not match ``X`` in length.
    """
    X = check_array(X)
    labels = np.asarray(labels).ravel()
    if labels.shape[0] != X.shape[0]:
        raise ValueError(f"labels has {labels.shape[0]} entries but X has {X.shape[0]} samples")

    mask = labels != NOISE_LABEL
    n_labelled = int(mask.sum())
    n_clusters = int(np.unique(labels[mask]).size)
    n_noise = int(labels.shape[0] - n_labelled)

    metrics: Dict[str, float] = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_fraction": n_noise / labels.shape[0],
    }

    if 2 <= n_clusters <= n_labelled - 1:
        X_l, labels_l = X[mask], labels[mask]
        metrics["silhouette_score"] = float(silhouette_score(X_l, labels_l))
        metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X_l, labels_l))
        metrics["davies_bouldin_score"] = float(davies_bouldin_score(X_l, labels_l))
    else:
        metrics["silhouette_score"] = float("nan")
        metrics["calinski_harabasz_score"] = float("nan")
        metrics["davies_bouldin_score"] = float("nan")

    if y_true is not None:
        y_true = np.asarray(y_true).ravel()
        if y_true.shape[0] != labels.shape[0]:
            raise ValueError(f"y_true has {y_true.shape[0]} entries but X has {X.shape[0]} samples")
        metrics["adjusted_rand_score"] = float(adjusted_rand_score(y_true, labels))
        metrics["adjusted_mutual_info_score"] = float(adjusted_mutual_info_score(y_true, labels))
        metrics["homogeneity_score"] = float(homogeneity_score(y_true, labels))
        metrics["completeness_score"] = float(completeness_score(y_true, labels))
        metrics["v_measure_score"] = float(v_measure_score(y_true, labels))

    return metrics


def _knee_index(values: np.ndarray) -> int:
    """Return the index of the knee/elbow of a monotone curve.

    Uses the "kneedle"-style heuristic: normalise the curve to the unit
    square and pick the point with the largest perpendicular distance from the
    chord joining its end points.

    Args:
        values: One-dimensional array of curve values sampled at equal steps.

    Returns:
        Index of the knee. ``0`` when the curve is flat or has fewer than
        three points.
    """
    y = np.asarray(values, dtype=float).ravel()
    n = y.shape[0]
    if n < 3:
        return 0
    y_range = np.ptp(y)
    if y_range == 0:
        return 0
    x = np.linspace(0.0, 1.0, n)
    y_norm = (y - y.min()) / y_range
    dx, dy = x[-1] - x[0], y_norm[-1] - y_norm[0]
    chord_len = float(np.hypot(dx, dy))
    if chord_len == 0:
        return 0
    distances = np.abs(dx * (y_norm - y_norm[0]) - dy * (x - x[0])) / chord_len
    return int(np.argmax(distances))


@dataclass(frozen=True)
class OptimalKResult:
    """Outcome of :func:`find_optimal_k`.

    Attributes:
        best_k: Selected number of clusters.
        scores: Mapping ``k -> criterion value`` for every evaluated ``k``.
        criterion: Name of the selection criterion that was used.
    """

    best_k: int
    scores: Dict[int, float]
    criterion: str

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the evaluated ``k`` values and their scores as arrays."""
        ks = np.array(sorted(self.scores), dtype=int)
        return ks, np.array([self.scores[k] for k in ks], dtype=float)


def find_optimal_k(
    X: ArrayLike,
    k_range: Iterable[int] = range(2, 11),
    criterion: str = "silhouette",
    estimator_factory: Optional[EstimatorFactory] = None,
    random_state: Optional[int] = None,
) -> OptimalKResult:
    """Select the number of clusters by sweeping ``k`` against a criterion.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        k_range: Candidate cluster counts. Values outside ``[1, n_samples - 1]``
            are ignored.
        criterion: One of ``"silhouette"`` and ``"calinski_harabasz"``
            (maximised), ``"davies_bouldin"``, ``"bic"`` and ``"aic"``
            (minimised), or ``"elbow"`` / ``"inertia"`` (knee of the
            within-cluster sum of squares curve).
        estimator_factory: Callable ``k -> estimator``. The estimator must
            implement ``fit_predict`` (all criteria), ``bic``/``aic`` (for the
            information criteria) or expose ``inertia_`` (for the elbow).
            Defaults to :class:`sklearn.cluster.KMeans`, or
            :class:`sklearn.mixture.GaussianMixture` for ``"bic"``/``"aic"``.
        random_state: Seed forwarded to the default estimators.

    Returns:
        An :class:`OptimalKResult` with the chosen ``k`` and the score curve.

    Raises:
        ValueError: If ``criterion`` is unknown, ``k_range`` is empty after
            filtering, or no candidate produced a finite score.
    """
    if criterion not in VALID_K_CRITERIA:
        raise ValueError(f"criterion must be one of {VALID_K_CRITERIA}, got {criterion!r}")

    X = check_array(X)
    n_samples = X.shape[0]
    ks = sorted({int(k) for k in k_range if 1 <= int(k) <= n_samples - 1})
    if not ks:
        raise ValueError(f"k_range contains no valid values for {n_samples} samples")

    if estimator_factory is None:
        if criterion in ("bic", "aic"):

            def estimator_factory(k: int) -> Any:
                return GaussianMixture(n_components=k, random_state=random_state)

        else:

            def estimator_factory(k: int) -> Any:
                return KMeans(n_clusters=k, n_init=10, random_state=random_state)

    scores: Dict[int, float] = {}
    for k in ks:
        estimator = estimator_factory(k)
        if criterion in ("bic", "aic"):
            estimator.fit(X)
            scores[k] = float(getattr(estimator, criterion)(X))
        elif criterion in _KNEE_CRITERIA:
            estimator.fit(X)
            scores[k] = float(estimator.inertia_)
        else:
            labels = np.asarray(estimator.fit_predict(X))
            scores[k] = _internal_index(X, labels, criterion)

    values = np.array([scores[k] for k in ks], dtype=float)
    if criterion in _KNEE_CRITERIA:
        best_idx = _knee_index(values)
    else:
        if np.all(np.isnan(values)):
            raise ValueError(f"No candidate k in {ks} produced a valid {criterion} score")
        best_idx = int(np.nanargmax(values) if criterion in _MAXIMISE_CRITERIA else np.nanargmin(values))

    result = OptimalKResult(best_k=ks[best_idx], scores=scores, criterion=criterion)
    logger.debug("find_optimal_k(%s): best_k=%d over %s", criterion, result.best_k, ks)
    return result


def _internal_index(X: np.ndarray, labels: np.ndarray, criterion: str) -> float:
    """Evaluate a single internal validity index, returning ``nan`` if undefined."""
    mask = labels != NOISE_LABEL
    n_clusters = np.unique(labels[mask]).size
    if not 2 <= n_clusters <= int(mask.sum()) - 1:
        return float("nan")
    scorer = {
        "silhouette": silhouette_score,
        "calinski_harabasz": calinski_harabasz_score,
        "davies_bouldin": davies_bouldin_score,
    }[criterion]
    return float(scorer(X[mask], labels[mask]))


def estimate_eps(
    X: ArrayLike,
    min_samples: int = 5,
    method: str = "knee",
    percentile: float = 95.0,
) -> float:
    """Estimate DBSCAN's ``eps`` from the sorted k-distance graph.

    The distance from every point to its ``min_samples``-th nearest neighbour
    (counting the point itself, as DBSCAN does) is sorted in increasing order.
    The knee of that curve is the classical heuristic for ``eps``.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        min_samples: The ``min_samples`` that will be used with DBSCAN.
        method: ``"knee"`` for the curve knee or ``"percentile"`` to take the
            given percentile of the k-distances.
        percentile: Percentile used when ``method="percentile"``.

    Returns:
        A strictly positive ``eps`` estimate.

    Raises:
        ValueError: If ``method`` is unknown, ``min_samples < 1``, fewer than two
            samples are given, or all k-distances are zero.
    """
    if method not in ("knee", "percentile"):
        raise ValueError(f"method must be 'knee' or 'percentile', got {method!r}")
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")

    X = check_array(X)
    n_samples = X.shape[0]
    if n_samples < 2:
        raise ValueError("estimate_eps needs at least two samples")

    n_neighbors = min(max(min_samples, 2), n_samples)
    distances, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(X).kneighbors(X)
    k_distances = np.sort(distances[:, -1])

    if method == "knee":
        eps = float(k_distances[_knee_index(k_distances)])
    else:
        eps = float(np.percentile(k_distances, percentile))

    if eps <= 0:
        positive = k_distances[k_distances > 0]
        if positive.size == 0:
            raise ValueError("All k-distances are zero; eps cannot be estimated")
        eps = float(positive[0])
    logger.debug("estimate_eps(min_samples=%d, method=%s) -> %.4f", min_samples, method, eps)
    return eps


# --------------------------------------------------------------------------- #
# Base classes
# --------------------------------------------------------------------------- #
class ClusteringModel(BaseEstimator, ClusterMixin, LoggerMixin):
    """Abstract base class for all clustering wrappers.

    Subclasses must implement :meth:`fit` and :meth:`predict`; the mixin
    supplies ``fit_predict``, :meth:`evaluate`, persistence helpers and the
    :attr:`model` accessor for the underlying scikit-learn estimator.

    Fitted attributes (set by concrete subclasses):
        model_: The fitted scikit-learn estimator.
        labels_: Cluster labels of the training samples.
        n_clusters_: Number of clusters found (noise excluded).
        n_noise_: Number of training samples labelled as noise.
        n_features_in_: Number of features seen during ``fit``.
    """

    def _build_estimator(self, **overrides: Any) -> Any:
        """Create an unfitted underlying estimator from the current parameters."""
        raise NotImplementedError(f"{self.__class__.__name__} does not define an underlying estimator")

    def fit(self, X: ArrayLike, y: Any = None) -> ClusteringModel:
        """Fit the model to ``X``.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Ignored; present for API compatibility.

        Returns:
            The fitted estimator.

        Raises:
            NotImplementedError: Always, on the abstract base class.
        """
        raise NotImplementedError("Subclasses must implement fit")

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict cluster labels for ``X``.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Integer labels of shape ``(n_samples,)``.

        Raises:
            NotImplementedError: Always, on the abstract base class.
        """
        raise NotImplementedError("Subclasses must implement predict")

    @property
    def model(self) -> Any:
        """The underlying scikit-learn estimator (fitted when available)."""
        fitted = getattr(self, "model_", None)
        if fitted is not None:
            return fitted
        return self._build_estimator()

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "model_")

    def get_labels(self) -> np.ndarray:
        """Return the training labels found during ``fit``.

        Returns:
            Copy of ``labels_``.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        check_is_fitted(self)
        return np.asarray(self.labels_).copy()

    def evaluate(
        self,
        X: ArrayLike,
        y_true: Optional[ArrayLike] = None,
        y_pred: Optional[ArrayLike] = None,
    ) -> Dict[str, float]:
        """Evaluate clustering quality on ``X``.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y_true: Optional ground-truth labels for external metrics.
            y_pred: Optional precomputed labels; when omitted the model's
                :meth:`predict` is used.

        Returns:
            Metric dictionary as produced by :func:`evaluate_clustering`.
        """
        if y_pred is None:
            check_is_fitted(self)
            y_pred = self.predict(X)
        return evaluate_clustering(X, y_pred, y_true=y_true)

    def save_model(self, path: Union[str, Path]) -> Path:
        """Persist the fitted wrapper with joblib.

        Args:
            path: Destination file path; parent directories are created.

        Returns:
            The resolved path the model was written to.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        check_is_fitted(self)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        self.logger.info("Saved %s to %s", self.__class__.__name__, path)
        return path

    def load_model(self, path: Union[str, Path]) -> ClusteringModel:
        """Load a wrapper saved by :meth:`save_model` into this instance.

        Args:
            path: File written by :meth:`save_model`.

        Returns:
            ``self``, now carrying the loaded parameters and fitted state.

        Raises:
            TypeError: If the file does not contain a wrapper of this class.
        """
        loaded = joblib.load(path)
        if type(loaded) is not type(self):
            raise TypeError(
                f"{path} contains a {type(loaded).__name__}, cannot load into {self.__class__.__name__}"
            )
        self.__dict__.clear()
        self.__dict__.update(loaded.__dict__)
        self.logger.info("Loaded %s from %s", self.__class__.__name__, path)
        return self

    def _validate_X(self, X: ArrayLike, *, reset: bool) -> np.ndarray:
        """Validate ``X`` and track ``n_features_in_`` across scikit-learn versions."""
        if _sk_validate_data is not None:
            return _sk_validate_data(self, X, reset=reset)
        return self._validate_data(X, reset=reset)  # type: ignore[attr-defined]


class _SklearnClusteringModel(ClusteringModel):
    """Template base wrapping a single scikit-learn estimator class.

    Subclasses set ``_estimator_cls`` and implement ``_estimator_params``;
    everything else (validation, fitting, bookkeeping) is shared.
    """

    _estimator_cls: Type[Any]
    #: Name of the constructor argument holding the cluster count, if any.
    _k_param: Optional[str] = None

    def _estimator_params(self) -> Dict[str, Any]:
        """Return keyword arguments for the underlying estimator class."""
        raise NotImplementedError

    def _build_estimator(self, **overrides: Any) -> Any:
        params = self._estimator_params()
        extra = getattr(self, "estimator_params", None)
        if extra:
            params.update(extra)
        params.update(overrides)
        return self._estimator_cls(**params)

    def _labels_from_fitted(self, estimator: Any, X: np.ndarray) -> np.ndarray:
        return np.asarray(estimator.labels_)

    def _post_fit(self, X: np.ndarray) -> None:
        """Hook for subclasses that need training-time state."""

    def _record_labels(self, labels: np.ndarray) -> None:
        labels = np.asarray(labels)
        self.labels_ = labels
        self.n_clusters_ = int(np.unique(labels[labels != NOISE_LABEL]).size)
        self.n_noise_ = int(np.sum(labels == NOISE_LABEL))

    def fit(self, X: ArrayLike, y: Any = None) -> _SklearnClusteringModel:
        """Fit the underlying estimator.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Ignored; present for API compatibility.

        Returns:
            The fitted wrapper.
        """
        X = self._validate_X(X, reset=True)
        estimator = self._build_estimator()
        estimator.fit(X)
        self.model_ = estimator
        self._record_labels(self._labels_from_fitted(estimator, X))
        self._post_fit(X)
        self.logger.debug(
            "%s fitted on %d samples: %d clusters, %d noise points",
            self.__class__.__name__,
            X.shape[0],
            self.n_clusters_,
            self.n_noise_,
        )
        return self


class _InductiveClusteringModel(_SklearnClusteringModel):
    """Base for estimators that natively implement ``predict``."""

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Assign new samples to the fitted clusters.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Integer labels of shape ``(n_samples,)``.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        return np.asarray(self.model_.predict(X))


class _TransductiveClusteringModel(_SklearnClusteringModel):
    """Base for estimators without a native ``predict``.

    Predicting on the exact training matrix returns ``labels_``. Any other
    input is labelled by its nearest reference point (see
    :meth:`_reference_points`); a distance threshold, when defined by the
    subclass, turns far-away points into noise.
    """

    def _reference_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(points, labels)`` used for nearest-neighbour prediction."""
        mask = self.labels_ != NOISE_LABEL
        return self.training_data_[mask], self.labels_[mask]

    def _max_assignment_distance(self) -> Optional[float]:
        """Distance beyond which new points are labelled as noise (``None`` = never)."""
        return None

    def _post_fit(self, X: np.ndarray) -> None:
        self.training_data_ = np.array(X, copy=True)
        points, labels = self._reference_points()
        self._reference_labels_ = labels
        self._nn_index_ = NearestNeighbors(n_neighbors=1).fit(points) if points.shape[0] else None

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Assign samples to the fitted clusters.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Integer labels of shape ``(n_samples,)``; ``-1`` marks noise.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        if X.shape == self.training_data_.shape and np.array_equal(X, self.training_data_):
            return self.labels_.copy()
        if self._nn_index_ is None:
            return np.full(X.shape[0], NOISE_LABEL, dtype=int)
        distances, indices = self._nn_index_.kneighbors(X)
        labels = np.asarray(self._reference_labels_[indices[:, 0]]).copy()
        max_distance = self._max_assignment_distance()
        if max_distance is not None and np.isfinite(max_distance):
            labels[distances[:, 0] > max_distance] = NOISE_LABEL
        return labels

    def get_cluster_centers(self) -> np.ndarray:
        """Return the centroid of each cluster's training members (noise excluded).

        Returns:
            Array of shape ``(n_clusters_, n_features)`` ordered by label.

        Raises:
            NotFittedError: If the model has not been fitted.
        """
        check_is_fitted(self)
        return _centroids_from_labels(self.training_data_, self.labels_)


def _centroids_from_labels(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute per-label centroids, skipping :data:`NOISE_LABEL`."""
    unique = np.unique(labels[labels != NOISE_LABEL])
    if unique.size == 0:
        return np.empty((0, X.shape[1]), dtype=float)
    return np.vstack([X[labels == label].mean(axis=0) for label in unique])


class _AutoKMixin:
    """Select the cluster count automatically when it is left as ``None``.

    Hosts must define ``_k_param``, ``k_range`` and ``criterion`` and inherit
    from :class:`_SklearnClusteringModel`. The selection is recorded in
    ``selected_k_`` and ``selection_result_``.
    """

    def _auto_k_enabled(self) -> bool:
        return getattr(self, self._k_param) is None  # type: ignore[attr-defined]

    def fit(self, X: ArrayLike, y: Any = None) -> Any:
        X = self._validate_X(X, reset=True)  # type: ignore[attr-defined]
        if self._auto_k_enabled():
            lo, hi = self.k_range  # type: ignore[attr-defined]
            result = find_optimal_k(
                X,
                k_range=range(int(lo), int(hi) + 1),
                criterion=self.criterion,  # type: ignore[attr-defined]
                estimator_factory=lambda k: self._build_estimator(**{self._k_param: k}),  # type: ignore[attr-defined]
            )
            self.selection_result_ = result
            self.selected_k_ = result.best_k
            self.logger.info(  # type: ignore[attr-defined]
                "%s selected k=%d by %s", self.__class__.__name__, result.best_k, result.criterion
            )
        else:
            self.selected_k_ = getattr(self, self._k_param)  # type: ignore[attr-defined]
        return super().fit(X, y)  # type: ignore[misc]

    def _build_estimator(self, **overrides: Any) -> Any:
        selected = getattr(self, "selected_k_", None)
        if selected is not None and self._k_param not in overrides:  # type: ignore[attr-defined]
            overrides[self._k_param] = selected  # type: ignore[attr-defined]
        return super()._build_estimator(**overrides)  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Centroid-based models
# --------------------------------------------------------------------------- #
class _CentroidModel(TransformerMixin, _InductiveClusteringModel):
    """Shared behaviour for K-Means-style estimators exposing ``cluster_centers_``."""

    def get_cluster_centers(self) -> np.ndarray:
        """Return the fitted cluster centres of shape ``(n_clusters, n_features)``."""
        check_is_fitted(self)
        return np.asarray(self.model_.cluster_centers_)

    def get_inertia(self) -> float:
        """Return the within-cluster sum of squared distances."""
        check_is_fitted(self)
        return float(self.model_.inertia_)

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Transform ``X`` to cluster-distance space.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Distances of shape ``(n_samples, n_clusters)``.
        """
        check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        return np.asarray(self.model_.transform(X))

    def score(self, X: ArrayLike, y: Any = None) -> float:
        """Return the negative inertia of ``X`` (higher is better)."""
        check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        return float(self.model_.score(X))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Feature names for the cluster-distance representation."""
        check_is_fitted(self)
        prefix = self.__class__.__name__.lower()
        n_out = self.model_.cluster_centers_.shape[0]
        return np.asarray([f"{prefix}{i}" for i in range(n_out)], dtype=object)


class KMeansModel(_CentroidModel):
    """K-Means clustering (Lloyd / Elkan) wrapper.

    Args:
        n_clusters: Number of clusters.
        init: Initialisation strategy (``"k-means++"``, ``"random"`` or array).
        n_init: Number of centroid seeds to try (``"auto"`` or an int).
        max_iter: Maximum number of EM iterations per run.
        tol: Relative tolerance on inertia used to declare convergence.
        algorithm: ``"lloyd"`` or ``"elkan"``.
        random_state: Seed for centroid initialisation.
        estimator_params: Extra keyword arguments forwarded to
            :class:`sklearn.cluster.KMeans`.
    """

    _estimator_cls = KMeans
    _k_param = "n_clusters"

    def __init__(
        self,
        n_clusters: Optional[int] = 8,
        init: Union[str, np.ndarray] = "k-means++",
        n_init: Union[str, int] = "auto",
        max_iter: int = 300,
        tol: float = 1e-4,
        algorithm: str = "lloyd",
        random_state: Optional[int] = None,
        estimator_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.algorithm = algorithm
        self.random_state = random_state
        self.estimator_params = estimator_params

    def _estimator_params(self) -> Dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "init": self.init,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "algorithm": self.algorithm,
            "random_state": self.random_state,
        }


class AdaptiveKMeans(_AutoKMixin, KMeansModel):
    """K-Means that chooses ``n_clusters`` automatically when it is ``None``.

    Args:
        n_clusters: Fixed cluster count, or ``None`` to select one from
            ``k_range`` with :func:`find_optimal_k`.
        k_range: Inclusive ``(min_k, max_k)`` bounds for the search.
        criterion: Selection criterion accepted by :func:`find_optimal_k`.
        init, n_init, max_iter, tol, algorithm, random_state, estimator_params:
            See :class:`KMeansModel`.
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        k_range: Tuple[int, int] = (2, 10),
        criterion: str = "silhouette",
        init: Union[str, np.ndarray] = "k-means++",
        n_init: Union[str, int] = "auto",
        max_iter: int = 300,
        tol: float = 1e-4,
        algorithm: str = "lloyd",
        random_state: Optional[int] = None,
        estimator_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            algorithm=algorithm,
            random_state=random_state,
            estimator_params=estimator_params,
        )
        self.k_range = k_range
        self.criterion = criterion


class MiniBatchKMeansModel(_CentroidModel):
    """Mini-batch K-Means wrapper with ``partial_fit`` support.

    Args:
        n_clusters: Number of clusters.
        init: Initialisation strategy.
        max_iter: Maximum number of iterations over the full dataset.
        batch_size: Size of the mini batches.
        tol: Relative tolerance on centre changes (``0`` disables the check).
        max_no_improvement: Consecutive batches without inertia improvement
            before early stopping.
        n_init: Number of random initialisations (``"auto"`` or an int).
        reassignment_ratio: Fraction of the maximum centre count to reassign.
        random_state: Seed for initialisation and batch sampling.
        estimator_params: Extra keyword arguments forwarded to
            :class:`sklearn.cluster.MiniBatchKMeans`.
    """

    _estimator_cls = MiniBatchKMeans
    _k_param = "n_clusters"

    def __init__(
        self,
        n_clusters: int = 8,
        init: Union[str, np.ndarray] = "k-means++",
        max_iter: int = 100,
        batch_size: int = 1024,
        tol: float = 0.0,
        max_no_improvement: Optional[int] = 10,
        n_init: Union[str, int] = "auto",
        reassignment_ratio: float = 0.01,
        random_state: Optional[int] = None,
        estimator_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.max_no_improvement = max_no_improvement
        self.n_init = n_init
        self.reassignment_ratio = reassignment_ratio
        self.random_state = random_state
        self.estimator_params = estimator_params

    def _estimator_params(self) -> Dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "init": self.init,
            "max_iter": self.max_iter,
            "batch_size": self.batch_size,
            "tol": self.tol,
            "max_no_improvement": self.max_no_improvement,
            "n_init": self.n_init,
            "reassignment_ratio": self.reassignment_ratio,
            "random_state": self.random_state,
        }

    def partial_fit(self, X: ArrayLike, y: Any = None) -> MiniBatchKMeansModel:
        """Update the centres with one mini batch.

        The first call creates the underlying estimator; subsequent calls keep
        refining it. ``labels_`` reflects the most recent batch only.

        Args:
            X: Mini batch of shape ``(n_samples, n_features)``.
            y: Ignored; present for API compatibility.

        Returns:
            The (partially) fitted wrapper.
        """
        first_call = not hasattr(self, "model_")
        X = self._validate_X(X, reset=first_call)
        if first_call:
            self.model_ = self._build_estimator()
        self.model_.partial_fit(X)
        self._record_labels(self.model_.labels_)
        return self


# --------------------------------------------------------------------------- #
# Hierarchical / graph-based models
# --------------------------------------------------------------------------- #
class HierarchicalClusteringModel(_TransductiveClusteringModel):
    """Agglomerative (hierarchical) clustering wrapper.

    New samples are labelled by their nearest labelled training point.

    Args:
        n_clusters: Number of clusters, or ``None`` when ``distance_threshold``
            is given.
        linkage: ``"ward"``, ``"complete"``, ``"average"`` or ``"single"``.
        metric: Distance metric; must be ``"euclidean"`` with Ward linkage.
        distance_threshold: Merge threshold used instead of ``n_clusters``.
        compute_distances: Store merge distances in ``distances_``.
        connectivity: Optional connectivity matrix or callable.
        estimator_params: Extra keyword arguments forwarded to
            :class:`sklearn.cluster.AgglomerativeClustering`.
    """

    _estimator_cls = AgglomerativeClustering
    _k_param = "n_clusters"

    def __init__(
        self,
        n_clusters: Optional[int] = 2,
        linkage: str = "ward",
        metric: str = "euclidean",
        distance_threshold: Optional[float] = None,
        compute_distances: bool = False,
        connectivity: Any = None,
        estimator_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        self.distance_threshold = distance_threshold
        self.compute_distances = compute_distances
        self.connectivity = connectivity
        self.estimator_params = estimator_params

    def _estimator_params(self) -> Dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "linkage": self.linkage,
            "metric": self.metric,
            "distance_threshold": self.distance_threshold,
            "compute_distances": self.compute_distances,
            "connectivity": self.connectivity,
        }

    def get_children(self) -> np.ndarray:
        """Return the merge tree (``children_``) of shape ``(n_samples - 1, 2)``."""
        check_is_fitted(self)
        return np.asarray(self.model_.children_)

    def get_distances(self) -> np.ndarray:
        """Return merge distances; requires ``compute_distances=True`` or a threshold.

        Raises:
            AttributeError: If distances were not computed during ``fit``.
        """
        check_is_fitted(self)
        if not hasattr(self.model_, "distances_"):
            raise AttributeError("Merge distances are only available with compute_distances=True")
        return np.asarray(self.model_.distances_)


class HierarchicalEnhanced(_AutoKMixin, HierarchicalClusteringModel):
    """Agglomerative clustering with automatic ``n_clusters`` selection.

    Selection runs when both ``n_clusters`` and ``distance_threshold`` are
    ``None``.

    Args:
        n_clusters: Fixed cluster count or ``None`` for automatic selection.
        k_range: Inclusive ``(min_k, max_k)`` bounds for the search.
        criterion: Internal index used for selection (``"silhouette"``,
            ``"calinski_harabasz"`` or ``"davies_bouldin"``).
        linkage, metric, distance_threshold, compute_distances, connectivity,
        estimator_params: See :class:`HierarchicalClusteringModel`.
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        k_range: Tuple[int, int] = (2, 10),
        criterion: str = "silhouette",
        linkage: str = "ward",
        metric: str = "euclidean",
        distance_threshold: Optional[float] = None,
        compute_distances: bool = False,
        connectivity: Any = None,
        estimator_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric,
            distance_threshold=distance_threshold,
            compute_distances=compute_distances,
            connectivity=connectivity,
            estimator_params=estimator_params,
        )
        self.k_range = k_range
        self.criterion = criterion

    def _auto_k_enabled(self) -> bool:
        return self.n_clusters is None and self.distance_threshold is None


class SpectralClusteringModel(_TransductiveClusteringModel):
    """Spectral clustering wrapper.

    New samples are labelled by their nearest labelled training point.

    Args:
        n_clusters: Number of clusters.
        affinity: ``"rbf"``, ``"nearest_neighbors"``, ``"precomputed"`` or a
            kernel name / callable.
        gamma: Kernel coefficient for RBF-type affinities.
        n_neighbors: Neighbour count for the ``"nearest_neighbors"`` affinity.
        assign_labels: ``"kmeans"``, ``"discretize"`` or ``"cluster_qr"``.
        n_init: K-Means restarts when ``assign_labels="kmeans"``.
        random_state: Seed for the eigen decomposition and label assignment.
        n_jobs: Parallel jobs for the affinity computation.
        estimator_params: Extra keyword arguments forwarded to
            :class:`sklearn.cluster.SpectralClustering`.
    """

    _estimator_cls = SpectralClustering
    _k_param = "n_clusters"

    def __init__(
        self,
        n_clusters: Optional[int] = 8,
        affinity: Union[str, Callable[..., Any]] = "rbf",
        gamma: float = 1.0,
        n_neighbors: int = 10,
        assign_labels: str = "kmeans",
        n_init: int = 10,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        estimator_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.assign_labels = assign_labels
        self.n_init = n_init
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.estimator_params = estimator_params

    def _estimator_params(self) -> Dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "affinity": self.affinity,
            "gamma": self.gamma,
            "n_neighbors": self.n_neighbors,
            "assign_labels": self.assign_labels,
            "n_init": self.n_init,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

    def get_affinity_matrix(self) -> np.ndarray:
        """Return the affinity matrix used for the spectral embedding."""
        check_is_fitted(self)
        matrix = self.model_.affinity_matrix_
        return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


class SpectralEnhanced(_AutoKMixin, SpectralClusteringModel):
    """Spectral clustering with automatic ``n_clusters`` selection.

    Args:
        n_clusters: Fixed cluster count or ``None`` for automatic selection.
        k_range: Inclusive ``(min_k, max_k)`` bounds for the search.
        criterion: Internal index used for selection.
        affinity, gamma, n_neighbors, assign_labels, n_init, random_state,
        n_jobs, estimator_params: See :class:`SpectralClusteringModel`.
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        k_range: Tuple[int, int] = (2, 10),
        criterion: str = "silhouette",
        affinity: Union[str, Callable[..., Any]] = "rbf",
        gamma: float = 1.0,
        n_neighbors: int = 10,
        assign_labels: str = "kmeans",
        n_init: int = 10,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        estimator_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            n_clusters=n_clusters,
            affinity=affinity,
            gamma=gamma,
            n_neighbors=n_neighbors,
            assign_labels=assign_labels,
            n_init=n_init,
            random_state=random_state,
            n_jobs=n_jobs,
            estimator_params=estimator_params,
        )
        self.k_range = k_range
        self.criterion = criterion


# --------------------------------------------------------------------------- #
# Density-based models
# --------------------------------------------------------------------------- #
class DBSCANModel(_TransductiveClusteringModel):
    """DBSCAN wrapper.

    New samples inherit the label of their nearest core sample when it lies
    within ``eps``; otherwise they are labelled as noise (``-1``).

    Args:
        eps: Neighbourhood radius.
        min_samples: Minimum neighbourhood size (including the point itself)
            for a core sample.
        metric: Distance metric.
        algorithm: Nearest-neighbour algorithm (``"auto"``, ``"ball_tree"``,
            ``"kd_tree"`` or ``"brute"``).
        leaf_size: Leaf size for tree-based neighbour searches.
        p: Minkowski power parameter.
        n_jobs: Parallel jobs for the neighbour search.
        estimator_params: Extra keyword arguments forwarded to
            :class:`sklearn.cluster.DBSCAN`.
    """

    _estimator_cls = DBSCAN

    def __init__(
        self,
        eps: Optional[float] = 0.5,
        min_samples: int = 5,
        metric: Union[str, Callable[..., float]] = "euclidean",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: Optional[float] = None,
        n_jobs: Optional[int] = None,
        estimator_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs
        self.estimator_params = estimator_params

    def _estimator_params(self) -> Dict[str, Any]:
        return {
            "eps": self.eps,
            "min_samples": self.min_samples,
            "metric": self.metric,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            "p": self.p,
            "n_jobs": self.n_jobs,
        }

    def _reference_points(self) -> Tuple[np.ndarray, np.ndarray]:
        core_idx = np.asarray(self.model_.core_sample_indices_, dtype=int)
        return self.training_data_[core_idx], self.labels_[core_idx]

    def _max_assignment_distance(self) -> Optional[float]:
        return float(self.model_.eps)

    def get_core_samples(self) -> np.ndarray:
        """Return indices of the core samples found during ``fit``."""
        check_is_fitted(self)
        return np.asarray(self.model_.core_sample_indices_, dtype=int)

    def get_components(self) -> np.ndarray:
        """Return the feature vectors of the core samples."""
        check_is_fitted(self)
        return np.asarray(self.model_.components_)


class DBSCANEnhanced(DBSCANModel):
    """DBSCAN that estimates ``eps`` from the k-distance graph when it is ``None``.

    The chosen value is stored in ``eps_``.

    Args:
        eps: Fixed radius or ``None`` to call :func:`estimate_eps`.
        eps_method: ``"knee"`` or ``"percentile"`` (see :func:`estimate_eps`).
        eps_percentile: Percentile used with ``eps_method="percentile"``.
        min_samples, metric, algorithm, leaf_size, p, n_jobs,
        estimator_params: See :class:`DBSCANModel`.
    """

    def __init__(
        self,
        eps: Optional[float] = None,
        eps_method: str = "knee",
        eps_percentile: float = 95.0,
        min_samples: int = 5,
        metric: Union[str, Callable[..., float]] = "euclidean",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: Optional[float] = None,
        n_jobs: Optional[int] = None,
        estimator_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            n_jobs=n_jobs,
            estimator_params=estimator_params,
        )
        self.eps_method = eps_method
        self.eps_percentile = eps_percentile

    def fit(self, X: ArrayLike, y: Any = None) -> DBSCANEnhanced:
        """Estimate ``eps`` if needed, then fit DBSCAN.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Ignored; present for API compatibility.

        Returns:
            The fitted wrapper.
        """
        X = self._validate_X(X, reset=True)
        if self.eps is None:
            self.eps_ = estimate_eps(
                X, min_samples=self.min_samples, method=self.eps_method, percentile=self.eps_percentile
            )
            self.logger.info("%s estimated eps=%.4f", self.__class__.__name__, self.eps_)
        else:
            self.eps_ = float(self.eps)
        return super().fit(X, y)

    def _build_estimator(self, **overrides: Any) -> Any:
        if self.eps is None and hasattr(self, "eps_") and "eps" not in overrides:
            overrides["eps"] = self.eps_
        return super()._build_estimator(**overrides)


class OPTICSModel(_TransductiveClusteringModel):
    """OPTICS wrapper exposing the reachability plot and DBSCAN-style extraction.

    New samples inherit the label of their nearest labelled training point;
    with ``cluster_method="dbscan"`` points farther than ``eps`` are noise.

    Args:
        min_samples: Neighbourhood size for core points (int or fraction).
        max_eps: Maximum neighbourhood radius considered.
        metric: Distance metric.
        cluster_method: ``"xi"`` or ``"dbscan"``.
        eps: Radius used when ``cluster_method="dbscan"`` (defaults to ``max_eps``).
        xi: Steepness threshold for the ``"xi"`` method.
        min_cluster_size: Minimum cluster size (int or fraction).
        n_jobs: Parallel jobs for the neighbour search.
        estimator_params: Extra keyword arguments forwarded to
            :class:`sklearn.cluster.OPTICS`.
    """

    _estimator_cls = OPTICS

    def __init__(
        self,
        min_samples: Union[int, float] = 5,
        max_eps: float = np.inf,
        metric: Union[str, Callable[..., float]] = "minkowski",
        cluster_method: str = "xi",
        eps: Optional[float] = None,
        xi: float = 0.05,
        min_cluster_size: Optional[Union[int, float]] = None,
        n_jobs: Optional[int] = None,
        estimator_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.metric = metric
        self.cluster_method = cluster_method
        self.eps = eps
        self.xi = xi
        self.min_cluster_size = min_cluster_size
        self.n_jobs = n_jobs
        self.estimator_params = estimator_params

    def _estimator_params(self) -> Dict[str, Any]:
        return {
            "min_samples": self.min_samples,
            "max_eps": self.max_eps,
            "metric": self.metric,
            "cluster_method": self.cluster_method,
            "eps": self.eps,
            "xi": self.xi,
            "min_cluster_size": self.min_cluster_size,
            "n_jobs": self.n_jobs,
        }

    def _max_assignment_distance(self) -> Optional[float]:
        if self.cluster_method == "dbscan":
            return float(self.eps if self.eps is not None else self.max_eps)
        return None

    def get_reachability(self) -> np.ndarray:
        """Return reachability distances (``inf`` for points never reached)."""
        check_is_fitted(self)
        return np.asarray(self.model_.reachability_).copy()

    def get_core_distances(self) -> np.ndarray:
        """Return core distances of every training sample."""
        check_is_fitted(self)
        return np.asarray(self.model_.core_distances_).copy()

    def get_ordering(self) -> np.ndarray:
        """Return the cluster-ordered sample indices."""
        check_is_fitted(self)
        return np.asarray(self.model_.ordering_).copy()

    def extract_clusters(self, eps: float) -> np.ndarray:
        """Extract DBSCAN-like clusters at a given ``eps`` from the fitted ordering.

        Args:
            eps: Neighbourhood radius for the extraction.

        Returns:
            Labels of shape ``(n_samples,)`` for the training data.
        """
        check_is_fitted(self)
        return np.asarray(
            cluster_optics_dbscan(
                reachability=self.model_.reachability_,
                core_distances=self.model_.core_distances_,
                ordering=self.model_.ordering_,
                eps=eps,
            )
        )


class MeanShiftModel(_InductiveClusteringModel):
    """Mean-shift clustering wrapper.

    Args:
        bandwidth: Kernel bandwidth; estimated with
            :func:`sklearn.cluster.estimate_bandwidth` when ``None``.
        bin_seeding: Seed from a binned grid instead of every point.
        cluster_all: Assign orphans to the nearest kernel (else ``-1``).
        max_iter: Maximum iterations per seed.
        n_jobs: Parallel jobs for the seed shifts.
        estimator_params: Extra keyword arguments forwarded to
            :class:`sklearn.cluster.MeanShift`.
    """

    _estimator_cls = MeanShift

    def __init__(
        self,
        bandwidth: Optional[float] = None,
        bin_seeding: bool = False,
        cluster_all: bool = True,
        max_iter: int = 300,
        n_jobs: Optional[int] = None,
        estimator_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.bandwidth = bandwidth
        self.bin_seeding = bin_seeding
        self.cluster_all = cluster_all
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.estimator_params = estimator_params

    def _estimator_params(self) -> Dict[str, Any]:
        return {
            "bandwidth": self.bandwidth,
            "bin_seeding": self.bin_seeding,
            "cluster_all": self.cluster_all,
            "max_iter": self.max_iter,
            "n_jobs": self.n_jobs,
        }

    def get_cluster_centers(self) -> np.ndarray:
        """Return the modes found, shape ``(n_clusters_, n_features)``."""
        check_is_fitted(self)
        return np.asarray(self.model_.cluster_centers_)


class AffinityPropagationModel(_InductiveClusteringModel):
    """Affinity propagation wrapper.

    Args:
        damping: Damping factor in ``[0.5, 1)``.
        max_iter: Maximum number of message-passing iterations.
        convergence_iter: Iterations without change to declare convergence.
        preference: Exemplar preferences (scalar or array); defaults to the
            median similarity.
        affinity: ``"euclidean"`` or ``"precomputed"``.
        random_state: Seed for the tie-breaking noise.
        estimator_params: Extra keyword arguments forwarded to
            :class:`sklearn.cluster.AffinityPropagation`.
    """

    _estimator_cls = AffinityPropagation

    def __init__(
        self,
        damping: float = 0.5,
        max_iter: int = 200,
        convergence_iter: int = 15,
        preference: Optional[Union[float, np.ndarray]] = None,
        affinity: str = "euclidean",
        random_state: Optional[int] = None,
        estimator_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.preference = preference
        self.affinity = affinity
        self.random_state = random_state
        self.estimator_params = estimator_params

    def _estimator_params(self) -> Dict[str, Any]:
        return {
            "damping": self.damping,
            "max_iter": self.max_iter,
            "convergence_iter": self.convergence_iter,
            "preference": self.preference,
            "affinity": self.affinity,
            "random_state": self.random_state,
        }

    def get_cluster_centers_indices(self) -> np.ndarray:
        """Return the training indices of the exemplars."""
        check_is_fitted(self)
        return np.asarray(self.model_.cluster_centers_indices_, dtype=int)

    def get_cluster_centers(self) -> np.ndarray:
        """Return the exemplar feature vectors, shape ``(n_clusters_, n_features)``."""
        check_is_fitted(self)
        return np.asarray(self.model_.cluster_centers_)


class BirchModel(TransformerMixin, _InductiveClusteringModel):
    """BIRCH wrapper with ``partial_fit`` and global-cluster transforms.

    ``transform`` returns distances to the *global* cluster centres
    (``cluster_centers_``, the mean of each cluster's CF-subcluster centroids),
    giving ``n_clusters`` columns rather than one per subcluster.

    Args:
        threshold: Maximum subcluster radius.
        branching_factor: Maximum CF subclusters per node.
        n_clusters: Global clustering step: an int (agglomerative), a fitted
            clusterer, or ``None`` to keep the raw subclusters.
        estimator_params: Extra keyword arguments forwarded to
            :class:`sklearn.cluster.Birch`.
    """

    _estimator_cls = Birch
    _k_param = "n_clusters"

    def __init__(
        self,
        threshold: float = 0.5,
        branching_factor: int = 50,
        n_clusters: Optional[Union[int, Any]] = 3,
        estimator_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.estimator_params = estimator_params

    def _estimator_params(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold,
            "branching_factor": self.branching_factor,
            "n_clusters": self.n_clusters,
        }

    def _update_centers(self) -> None:
        sub_centers = np.asarray(self.model_.subcluster_centers_)
        sub_labels = np.asarray(self.model_.subcluster_labels_)
        self.cluster_centers_ = _centroids_from_labels(sub_centers, sub_labels)

    def _post_fit(self, X: np.ndarray) -> None:
        self._update_centers()

    def partial_fit(self, X: ArrayLike, y: Any = None) -> BirchModel:
        """Insert one batch into the CF tree and refresh the global clustering.

        Args:
            X: Batch of shape ``(n_samples, n_features)``.
            y: Ignored; present for API compatibility.

        Returns:
            The (partially) fitted wrapper.
        """
        first_call = not hasattr(self, "model_")
        X = self._validate_X(X, reset=first_call)
        if first_call:
            self.model_ = self._build_estimator()
        self.model_.partial_fit(X)
        self._record_labels(self.model_.labels_)
        self._update_centers()
        return self

    def get_cluster_centers(self) -> np.ndarray:
        """Return the global cluster centres, shape ``(n_clusters_, n_features)``."""
        check_is_fitted(self)
        return self.cluster_centers_

    def get_subcluster_centers(self) -> np.ndarray:
        """Return the CF-subcluster centroids of the leaves."""
        check_is_fitted(self)
        return np.asarray(self.model_.subcluster_centers_)

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Distances from each sample to every global cluster centre.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Array of shape ``(n_samples, n_clusters_)``.
        """
        check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        return euclidean_distances(X, self.cluster_centers_)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Feature names for the cluster-distance representation."""
        check_is_fitted(self)
        return np.asarray([f"birchmodel{i}" for i in range(self.cluster_centers_.shape[0])], dtype=object)


# --------------------------------------------------------------------------- #
# Probabilistic models
# --------------------------------------------------------------------------- #
class GaussianMixtureModel(_InductiveClusteringModel):
    """Gaussian mixture model wrapper with soft assignments and model selection.

    Args:
        n_components: Number of mixture components.
        covariance_type: ``"full"``, ``"tied"``, ``"diag"`` or ``"spherical"``.
        tol: EM convergence threshold on the lower bound.
        reg_covar: Non-negative regularisation added to the covariance diagonal.
        max_iter: Maximum EM iterations.
        n_init: Number of initialisations to try.
        init_params: ``"kmeans"``, ``"k-means++"``, ``"random"`` or
            ``"random_from_data"``.
        random_state: Seed for initialisation.
        estimator_params: Extra keyword arguments forwarded to
            :class:`sklearn.mixture.GaussianMixture`.
    """

    _estimator_cls = GaussianMixture
    _k_param = "n_components"

    def __init__(
        self,
        n_components: Optional[int] = 1,
        covariance_type: str = "full",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = "kmeans",
        random_state: Optional[int] = None,
        estimator_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.estimator_params = estimator_params

    def _estimator_params(self) -> Dict[str, Any]:
        return {
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "tol": self.tol,
            "reg_covar": self.reg_covar,
            "max_iter": self.max_iter,
            "n_init": self.n_init,
            "init_params": self.init_params,
            "random_state": self.random_state,
        }

    def _labels_from_fitted(self, estimator: Any, X: np.ndarray) -> np.ndarray:
        return np.asarray(estimator.predict(X))

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Posterior component responsibilities of shape ``(n_samples, n_components)``."""
        check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        return np.asarray(self.model_.predict_proba(X))

    def score(self, X: ArrayLike, y: Any = None) -> float:
        """Average per-sample log-likelihood of ``X``."""
        check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        return float(self.model_.score(X))

    def score_samples(self, X: ArrayLike) -> np.ndarray:
        """Per-sample log-likelihood of shape ``(n_samples,)``."""
        check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        return np.asarray(self.model_.score_samples(X))

    def bic(self, X: ArrayLike) -> float:
        """Bayesian information criterion on ``X`` (lower is better)."""
        check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        return float(self.model_.bic(X))

    def aic(self, X: ArrayLike) -> float:
        """Akaike information criterion on ``X`` (lower is better)."""
        check_is_fitted(self)
        X = self._validate_X(X, reset=False)
        return float(self.model_.aic(X))

    def sample(self, n_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Draw samples from the fitted mixture.

        Args:
            n_samples: Number of samples to generate.

        Returns:
            ``(X, component_labels)`` as produced by the underlying estimator.
        """
        check_is_fitted(self)
        X, labels = self.model_.sample(n_samples)
        return np.asarray(X), np.asarray(labels)

    def get_cluster_centers(self) -> np.ndarray:
        """Component means of shape ``(n_components, n_features)``."""
        check_is_fitted(self)
        return np.asarray(self.model_.means_)

    def get_covariances(self) -> np.ndarray:
        """Component covariances in the layout implied by ``covariance_type``."""
        check_is_fitted(self)
        return np.asarray(self.model_.covariances_)


class GaussianMixtureEnhanced(_AutoKMixin, GaussianMixtureModel):
    """Gaussian mixture with automatic ``n_components`` selection (BIC by default).

    Args:
        n_components: Fixed component count or ``None`` for automatic selection.
        k_range: Inclusive ``(min_k, max_k)`` bounds for the search.
        criterion: ``"bic"``, ``"aic"`` or an internal index name.
        covariance_type, tol, reg_covar, max_iter, n_init, init_params,
        random_state, estimator_params: See :class:`GaussianMixtureModel`.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        k_range: Tuple[int, int] = (1, 10),
        criterion: str = "bic",
        covariance_type: str = "full",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = "kmeans",
        random_state: Optional[int] = None,
        estimator_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            estimator_params=estimator_params,
        )
        self.k_range = k_range
        self.criterion = criterion


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
class ClusteringModels(LoggerMixin):
    """Factory returning configured clustering wrappers by name.

    Args:
        random_state: Default seed injected into every model that accepts
            ``random_state`` unless the caller overrides it.

    Example:
        >>> factory = ClusteringModels(random_state=0)
        >>> kmeans = factory.get_kmeans(n_clusters=4)
        >>> dbscan = factory.get_model("dbscan", eps=0.8)
    """

    _REGISTRY: Dict[str, Type[ClusteringModel]] = {
        "kmeans": KMeansModel,
        "adaptive_kmeans": AdaptiveKMeans,
        "mini_batch_kmeans": MiniBatchKMeansModel,
        "hierarchical": HierarchicalClusteringModel,
        "agglomerative": HierarchicalClusteringModel,
        "dbscan": DBSCANModel,
        "optics": OPTICSModel,
        "gaussian_mixture": GaussianMixtureModel,
        "gmm": GaussianMixtureModel,
        "spectral": SpectralClusteringModel,
        "affinity_propagation": AffinityPropagationModel,
        "mean_shift": MeanShiftModel,
        "birch": BirchModel,
    }

    def __init__(self, random_state: Optional[int] = None) -> None:
        self.random_state = random_state

    @classmethod
    def available_models(cls) -> List[str]:
        """Return the registered model names."""
        return sorted(cls._REGISTRY)

    def get_model(self, name: str, **kwargs: Any) -> ClusteringModel:
        """Instantiate a registered model.

        Args:
            name: Registry key (case-insensitive), e.g. ``"kmeans"``.
            **kwargs: Constructor arguments for the wrapper.

        Returns:
            A new, unfitted wrapper instance.

        Raises:
            ValueError: If ``name`` is not registered.
        """
        key = name.lower().strip()
        try:
            cls = self._REGISTRY[key]
        except KeyError:
            raise ValueError(
                f"Unknown clustering model {name!r}; choose from {self.available_models()}"
            ) from None
        if (
            self.random_state is not None
            and "random_state" not in kwargs
            and "random_state" in inspect.signature(cls.__init__).parameters
        ):
            kwargs["random_state"] = self.random_state
        self.logger.debug("Creating %s with %s", cls.__name__, kwargs)
        return cls(**kwargs)

    def get_kmeans(self, **kwargs: Any) -> KMeansModel:
        """Return a :class:`KMeansModel`."""
        return self.get_model("kmeans", **kwargs)  # type: ignore[return-value]

    def get_mini_batch_kmeans(self, **kwargs: Any) -> MiniBatchKMeansModel:
        """Return a :class:`MiniBatchKMeansModel`."""
        return self.get_model("mini_batch_kmeans", **kwargs)  # type: ignore[return-value]

    def get_hierarchical(self, **kwargs: Any) -> HierarchicalClusteringModel:
        """Return a :class:`HierarchicalClusteringModel`."""
        return self.get_model("hierarchical", **kwargs)  # type: ignore[return-value]

    def get_dbscan(self, **kwargs: Any) -> DBSCANModel:
        """Return a :class:`DBSCANModel`."""
        return self.get_model("dbscan", **kwargs)  # type: ignore[return-value]

    def get_optics(self, **kwargs: Any) -> OPTICSModel:
        """Return an :class:`OPTICSModel`."""
        return self.get_model("optics", **kwargs)  # type: ignore[return-value]

    def get_gaussian_mixture(self, **kwargs: Any) -> GaussianMixtureModel:
        """Return a :class:`GaussianMixtureModel`."""
        return self.get_model("gaussian_mixture", **kwargs)  # type: ignore[return-value]

    def get_spectral(self, **kwargs: Any) -> SpectralClusteringModel:
        """Return a :class:`SpectralClusteringModel`."""
        return self.get_model("spectral", **kwargs)  # type: ignore[return-value]

    def get_affinity_propagation(self, **kwargs: Any) -> AffinityPropagationModel:
        """Return an :class:`AffinityPropagationModel`."""
        return self.get_model("affinity_propagation", **kwargs)  # type: ignore[return-value]

    def get_mean_shift(self, **kwargs: Any) -> MeanShiftModel:
        """Return a :class:`MeanShiftModel`."""
        return self.get_model("mean_shift", **kwargs)  # type: ignore[return-value]

    def get_birch(self, **kwargs: Any) -> BirchModel:
        """Return a :class:`BirchModel`."""
        return self.get_model("birch", **kwargs)  # type: ignore[return-value]

    def get_all_models(self, n_clusters: int = 3, eps: float = 0.5) -> Dict[str, ClusteringModel]:
        """Return one sensibly configured instance of every distinct algorithm.

        Args:
            n_clusters: Cluster count for the algorithms that need one.
            eps: Radius for DBSCAN.

        Returns:
            Mapping from model name to an unfitted wrapper.
        """
        return {
            "kmeans": self.get_kmeans(n_clusters=n_clusters),
            "mini_batch_kmeans": self.get_mini_batch_kmeans(n_clusters=n_clusters),
            "hierarchical": self.get_hierarchical(n_clusters=n_clusters),
            "dbscan": self.get_dbscan(eps=eps),
            "optics": self.get_optics(),
            "gaussian_mixture": self.get_gaussian_mixture(n_components=n_clusters),
            "spectral": self.get_spectral(n_clusters=n_clusters),
            "affinity_propagation": self.get_affinity_propagation(),
            "mean_shift": self.get_mean_shift(),
            "birch": self.get_birch(n_clusters=n_clusters),
        }


#: Backwards-compatible aliases.
KMeansClusterer = KMeansModel
GaussianMixtureClusterer = GaussianMixtureModel

__all__ = [
    "NOISE_LABEL",
    "VALID_K_CRITERIA",
    "AdaptiveKMeans",
    "AffinityPropagationModel",
    "BirchModel",
    "ClusteringModel",
    "ClusteringModels",
    "DBSCANEnhanced",
    "DBSCANModel",
    "GaussianMixtureClusterer",
    "GaussianMixtureEnhanced",
    "GaussianMixtureModel",
    "HierarchicalClusteringModel",
    "HierarchicalEnhanced",
    "KMeansClusterer",
    "KMeansModel",
    "MeanShiftModel",
    "MiniBatchKMeansModel",
    "OPTICSModel",
    "OptimalKResult",
    "SpectralClusteringModel",
    "SpectralEnhanced",
    "estimate_eps",
    "evaluate_clustering",
    "find_optimal_k",
]
