"""Dimensionality reduction wrappers built on scikit-learn.

Every wrapper in this module is a thin, sklearn-compatible estimator
(:class:`~sklearn.base.BaseEstimator` + :class:`~sklearn.base.TransformerMixin`)
around a scikit-learn (or ``umap-learn``) reducer. The wrappers add:

* a uniform ``fit`` / ``transform`` / ``fit_transform`` / ``inverse_transform``
  surface with clear ``NotImplementedError`` messages where the underlying
  method does not support out-of-sample transformation or inversion;
* float64 outputs regardless of the backend's native precision;
* pass-through fitted attributes (``components_``, ``embedding_``,
  ``explained_variance_ratio_``, ...) plus ``get_*`` convenience accessors;
* ``get_feature_names_out`` for use inside :class:`sklearn.pipeline.Pipeline`
  and :class:`sklearn.compose.ColumnTransformer`;
* graceful handling of the optional ``umap-learn`` dependency via
  :data:`HAS_UMAP` (the module imports without it).

Example
-------
>>> from sklearn.datasets import load_iris
>>> from sklearn_mastery.models.unsupervised.dimensionality_reduction import PCAModel
>>> X = load_iris().data
>>> pca = PCAModel(n_components=2, random_state=0).fit(X)
>>> pca.transform(X).shape
(150, 2)
>>> float(pca.get_explained_variance_ratio().sum()) > 0.9
True
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import (
    NMF,
    PCA,
    DictionaryLearning,
    FactorAnalysis,
    FastICA,
    KernelPCA,
    TruncatedSVD,
)
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.utils.validation import check_array, check_is_fitted

from sklearn_mastery.config.logging_config import LoggerMixin, get_logger
from sklearn_mastery.config.settings import settings

try:  # optional heavy dependency
    import umap as _umap

    HAS_UMAP = True
except ImportError:  # pragma: no cover - exercised only when umap-learn is absent
    _umap = None
    HAS_UMAP = False

__all__ = [
    "HAS_UMAP",
    "MODEL_REGISTRY",
    "AdaptiveTSNE",
    "DictionaryLearningModel",
    "DimensionalityReduction",
    "DimensionalityReductionModel",
    "EnhancedPCA",
    "FactorAnalysisModel",
    "ICAModel",
    "IsoMapModel",
    "KernelPCAModel",
    "LLEModel",
    "MDSModel",
    "ManifoldLearning",
    "NMFModel",
    "PCAModel",
    "SpectralEmbeddingModel",
    "TSNEModel",
    "TruncatedSVDModel",
    "UMAPEnhanced",
    "UMAPModel",
]

_logger = get_logger(__name__)

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]
RandomState = Optional[Union[int, np.random.RandomState]]


def _filter_kwargs(estimator_cls: type, **kwargs: Any) -> Dict[str, Any]:
    """Drop keyword arguments that ``estimator_cls.__init__`` does not accept.

    scikit-learn renames and removes constructor parameters between releases
    (for example ``TSNE(n_iter=...)`` became ``max_iter``). Filtering keeps the
    wrappers importable and functional across a range of scikit-learn versions
    while logging the dropped names at debug level.

    Args:
        estimator_cls: Estimator class to inspect.
        **kwargs: Candidate constructor arguments.

    Returns:
        Dictionary containing only the accepted keyword arguments.
    """
    params = inspect.signature(estimator_cls.__init__).parameters
    accepted = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(accepted))
    if dropped:
        _logger.debug("%s: ignoring unsupported kwargs %s", estimator_cls.__name__, dropped)
    return accepted


class DimensionalityReductionModel(BaseEstimator, TransformerMixin, LoggerMixin):
    """Abstract base class for all dimensionality reduction wrappers.

    Subclasses implement :meth:`_build_estimator`, which returns an *unfitted*
    scikit-learn-style estimator configured from the wrapper's constructor
    parameters. Everything else (validation, fitting, transformation, fitted
    attribute pass-through and feature naming) is handled here.

    Class-level flags control behaviour:

    * ``_supports_transform``: whether the backend can embed *new* samples.
      Methods such as t-SNE only expose ``fit_transform``; for these
      :meth:`transform` raises :class:`NotImplementedError`.
    * ``_accept_sparse``: sparse formats accepted by :func:`check_array`.
    * ``_feature_prefix``: prefix used by :meth:`get_feature_names_out`.

    Args:
        n_components: Target dimensionality. ``None`` keeps the backend's
            default (usually all components).
        random_state: Seed or ``RandomState`` for reproducible results.

    Attributes:
        model_: The fitted backend estimator.
        n_features_in_: Number of input features seen during :meth:`fit`.
        n_components_out_: Number of output dimensions produced by
            :meth:`transform` / :meth:`fit_transform`.

    Raises:
        NotImplementedError: When :meth:`fit` or :meth:`transform` is invoked
            on the abstract base class itself.
    """

    _supports_transform: bool = False
    _accept_sparse: Union[bool, str, Sequence[str]] = False
    _feature_prefix: str = "component"

    def __init__(self, n_components: Optional[int] = 2, random_state: RandomState = None) -> None:
        self.n_components = n_components
        self.random_state = random_state

    # ------------------------------------------------------------------ hooks
    def _build_estimator(self, n_samples: Optional[int] = None) -> Any:
        """Return an unfitted backend estimator.

        Args:
            n_samples: Number of training samples, when known. Wrappers whose
                hyper-parameters depend on the dataset size (perplexity,
                neighbourhood size) use it to clip invalid settings.

        Returns:
            An estimator exposing ``fit`` and, where applicable, ``transform``.

        Raises:
            NotImplementedError: Always, on the abstract base class.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is abstract; use a concrete wrapper such as PCAModel."
        )

    # -------------------------------------------------------------- properties
    @property
    def model(self) -> Any:
        """The backend estimator: fitted after :meth:`fit`, otherwise a fresh instance."""
        if hasattr(self, "model_"):
            return self.model_
        return self._build_estimator()

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "model_")

    def _fitted_attr(self, name: str) -> Any:
        """Fetch an attribute of the fitted backend estimator.

        Args:
            name: Attribute name on the backend (e.g. ``"components_"``).

        Returns:
            The attribute value.

        Raises:
            sklearn.exceptions.NotFittedError: If the wrapper is not fitted.
            AttributeError: If the backend has no such attribute.
        """
        check_is_fitted(self, "model_")
        try:
            return getattr(self.model_, name)
        except AttributeError as exc:
            raise AttributeError(
                f"{type(self).__name__} (backend {type(self.model_).__name__}) has no attribute '{name}'"
            ) from exc

    @property
    def components_(self) -> np.ndarray:
        """Component / loading matrix of shape ``(n_components, n_features)``."""
        return self._fitted_attr("components_")

    @property
    def embedding_(self) -> np.ndarray:
        """Training-set embedding of shape ``(n_samples, n_components)``."""
        return self._fitted_attr("embedding_")

    @property
    def explained_variance_(self) -> np.ndarray:
        """Variance explained by each component (variance-based methods only)."""
        return self._fitted_attr("explained_variance_")

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """Fraction of total variance explained by each component."""
        return self._fitted_attr("explained_variance_ratio_")

    # -------------------------------------------------------------- internals
    def _validate_X(self, X: ArrayLike, *, reset: bool) -> np.ndarray:
        """Validate ``X`` and (optionally) record ``n_features_in_``."""
        X = check_array(X, accept_sparse=self._accept_sparse, dtype=np.float64)
        if reset:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} was fitted with "
                f"{self.n_features_in_} features."
            )
        return X

    @staticmethod
    def _as_float64(X: Any) -> np.ndarray:
        """Return ``X`` as a dense float64 array."""
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float64)

    def _infer_n_components_out(self, transformed: Optional[np.ndarray] = None) -> int:
        """Determine the output dimensionality of the fitted backend."""
        if transformed is not None:
            return int(transformed.shape[1])
        est = self.model_
        if getattr(est, "n_components_", None) is not None:
            return int(est.n_components_)
        components = getattr(est, "components_", None)
        if components is not None:
            return int(np.shape(components)[0])
        embedding = getattr(est, "embedding_", None)
        if embedding is not None:
            return int(np.shape(embedding)[1])
        if self.n_components is not None:
            return int(self.n_components)
        return int(self.n_features_in_)

    def _finalize_fit(self, estimator: Any, transformed: Optional[np.ndarray] = None) -> None:
        """Store the fitted backend and derived bookkeeping attributes."""
        self.model_ = estimator
        self.n_components_out_ = self._infer_n_components_out(transformed)
        self.logger.debug(
            "%s fitted: %d features -> %d components",
            type(self).__name__,
            self.n_features_in_,
            self.n_components_out_,
        )

    # -------------------------------------------------------------- public API
    def fit(self, X: ArrayLike, y: Any = None) -> DimensionalityReductionModel:
        """Fit the reducer to ``X``.

        Args:
            X: Training data of shape ``(n_samples, n_features)``.
            y: Ignored; present for pipeline compatibility.

        Returns:
            The fitted wrapper (``self``).

        Raises:
            NotImplementedError: On the abstract base class.
        """
        # Build first so the abstract base fails fast with NotImplementedError
        # even when ``X`` is not a valid array.
        n_samples = getattr(X, "shape", (None,))[0] if X is not None else None
        estimator = self._build_estimator(n_samples=n_samples)
        X = self._validate_X(X, reset=True)
        estimator.fit(X)
        self._finalize_fit(estimator)
        return self

    def fit_transform(self, X: ArrayLike, y: Any = None, **fit_params: Any) -> np.ndarray:
        """Fit the reducer and return the embedding of ``X``.

        Uses the backend's own ``fit_transform`` when available, which is both
        faster and (for t-SNE-like methods) the only way to obtain an embedding.

        Args:
            X: Training data of shape ``(n_samples, n_features)``.
            y: Ignored; present for pipeline compatibility.
            **fit_params: Ignored; accepted for API compatibility.

        Returns:
            Embedded data of shape ``(n_samples, n_components)`` as float64.
        """
        n_samples = getattr(X, "shape", (None,))[0] if X is not None else None
        estimator = self._build_estimator(n_samples=n_samples)
        X = self._validate_X(X, reset=True)
        if hasattr(estimator, "fit_transform"):
            transformed = self._as_float64(estimator.fit_transform(X))
        else:
            transformed = self._as_float64(estimator.fit(X).transform(X))
        self._finalize_fit(estimator, transformed)
        return transformed

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Embed new samples using the fitted reducer.

        Args:
            X: Data of shape ``(n_samples, n_features)``.

        Returns:
            Embedded data of shape ``(n_samples, n_components)`` as float64.

        Raises:
            NotImplementedError: If the backend cannot embed unseen samples.
            sklearn.exceptions.NotFittedError: If called before :meth:`fit`.
        """
        if not self._supports_transform:
            raise NotImplementedError(
                f"{type(self).__name__} does not support transforming unseen samples; "
                "use fit_transform on the full dataset instead."
            )
        check_is_fitted(self, "model_")
        X = self._validate_X(X, reset=False)
        return self._as_float64(self.model_.transform(X))

    def inverse_transform(self, X: ArrayLike) -> np.ndarray:
        """Map embedded samples back to the original feature space.

        Args:
            X: Embedded data of shape ``(n_samples, n_components)``.

        Returns:
            Reconstructed data of shape ``(n_samples, n_features)`` as float64.

        Raises:
            NotImplementedError: If the backend provides no inverse mapping.
            sklearn.exceptions.NotFittedError: If called before :meth:`fit`.
        """
        check_is_fitted(self, "model_")
        if not hasattr(self.model_, "inverse_transform"):
            raise NotImplementedError(
                f"{type(self).__name__} (backend {type(self.model_).__name__}) has no inverse_transform."
            )
        X = check_array(X, dtype=np.float64)
        return self._as_float64(self.model_.inverse_transform(X))

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        """Return output feature names such as ``["pca0", "pca1", ...]``.

        Args:
            input_features: Ignored; present for sklearn API compatibility.

        Returns:
            Object array of length ``n_components_out_``.
        """
        check_is_fitted(self, "model_")
        return np.asarray([f"{self._feature_prefix}{i}" for i in range(self.n_components_out_)], dtype=object)

    # ------------------------------------------------------------ accessors
    def get_components(self) -> np.ndarray:
        """Return the component matrix of shape ``(n_components, n_features)``."""
        return np.asarray(self.components_)

    def get_embedding(self) -> np.ndarray:
        """Return the training-set embedding of shape ``(n_samples, n_components)``."""
        return self._as_float64(self.embedding_)

    def get_explained_variance(self) -> np.ndarray:
        """Return the variance explained by each component."""
        return np.asarray(self.explained_variance_)

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Return the fraction of variance explained by each component."""
        return np.asarray(self.explained_variance_ratio_)

    def get_cumulative_explained_variance_ratio(self) -> np.ndarray:
        """Return the cumulative explained-variance ratio across components."""
        return np.cumsum(self.get_explained_variance_ratio())

    def reconstruction_mse(self, X: ArrayLike) -> float:
        """Mean squared reconstruction error of ``X`` after a round trip.

        Args:
            X: Data of shape ``(n_samples, n_features)``.

        Returns:
            ``mean((X - inverse_transform(transform(X))) ** 2)``.

        Raises:
            NotImplementedError: If the backend lacks ``transform`` or
                ``inverse_transform``.
        """
        X = self._validate_X(X, reset=False)
        X_hat = self.inverse_transform(self.transform(X))
        return float(np.mean((X - X_hat) ** 2))


# --------------------------------------------------------------------------- #
# Linear / matrix-factorisation methods
# --------------------------------------------------------------------------- #


class PCAModel(DimensionalityReductionModel):
    """Principal Component Analysis wrapper around :class:`sklearn.decomposition.PCA`.

    Args:
        n_components: Number of components, a variance fraction in ``(0, 1)``
            (with ``svd_solver="full"``), ``"mle"``, or ``None`` for all.
        whiten: Scale components to unit variance.
        svd_solver: One of ``"auto"``, ``"full"``, ``"arpack"``, ``"randomized"``.
        tol: Tolerance for singular values (``arpack`` only).
        iterated_power: Power iterations for the randomized solver.
        random_state: Seed for the randomized / arpack solvers.
    """

    _supports_transform = True
    _feature_prefix = "pca"

    def __init__(
        self,
        n_components: Optional[Union[int, float, str]] = None,
        whiten: bool = False,
        svd_solver: str = "auto",
        tol: float = 0.0,
        iterated_power: Union[int, str] = "auto",
        random_state: RandomState = None,
    ) -> None:
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

    def _build_estimator(self, n_samples: Optional[int] = None) -> PCA:
        return PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            tol=self.tol,
            iterated_power=self.iterated_power,
            random_state=self.random_state,
        )

    def n_components_for_variance(self, threshold: float = 0.95) -> int:
        """Smallest number of fitted components explaining ``threshold`` of the variance.

        Args:
            threshold: Target cumulative explained-variance ratio in ``(0, 1]``.

        Returns:
            Number of components (at least 1, at most the number fitted).

        Raises:
            ValueError: If ``threshold`` is outside ``(0, 1]``.
        """
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        cumulative = self.get_cumulative_explained_variance_ratio()
        if cumulative[-1] < threshold:
            return len(cumulative)
        return int(np.searchsorted(cumulative, threshold, side="left")) + 1

    def get_singular_values(self) -> np.ndarray:
        """Return the singular values of the fitted components."""
        return np.asarray(self._fitted_attr("singular_values_"))

    def get_noise_variance(self) -> float:
        """Return the estimated noise variance (Tipping & Bishop PPCA)."""
        return float(self._fitted_attr("noise_variance_"))

    def score(self, X: ArrayLike, y: Any = None) -> float:
        """Average log-likelihood of ``X`` under the probabilistic PCA model."""
        check_is_fitted(self, "model_")
        return float(self.model_.score(self._validate_X(X, reset=False)))


class TruncatedSVDModel(DimensionalityReductionModel):
    """Truncated SVD (LSA) wrapper around :class:`sklearn.decomposition.TruncatedSVD`.

    Works on sparse matrices and does not centre the data, unlike PCA.

    Args:
        n_components: Number of singular vectors to keep (``< n_features``).
        algorithm: ``"randomized"`` or ``"arpack"``.
        n_iter: Power iterations for the randomized solver.
        tol: Tolerance for the arpack solver.
        random_state: Seed for the solver.
    """

    _supports_transform = True
    _accept_sparse = ("csr", "csc")
    _feature_prefix = "svd"

    def __init__(
        self,
        n_components: int = 2,
        algorithm: str = "randomized",
        n_iter: int = 5,
        tol: float = 0.0,
        random_state: RandomState = None,
    ) -> None:
        self.n_components = n_components
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

    def _build_estimator(self, n_samples: Optional[int] = None) -> TruncatedSVD:
        return TruncatedSVD(
            n_components=self.n_components,
            algorithm=self.algorithm,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
        )

    def get_singular_values(self) -> np.ndarray:
        """Return the retained singular values."""
        return np.asarray(self._fitted_attr("singular_values_"))


class ICAModel(DimensionalityReductionModel):
    """Independent Component Analysis wrapper around :class:`sklearn.decomposition.FastICA`.

    Args:
        n_components: Number of independent components (``None`` = all).
        algorithm: ``"parallel"`` or ``"deflation"``.
        whiten: Whitening strategy (``"unit-variance"``, ``"arbitrary-variance"``
            or ``False``).
        fun: Contrast function: ``"logcosh"``, ``"exp"`` or ``"cube"``.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance on the un-mixing matrix update.
        random_state: Seed for the initial un-mixing matrix.
    """

    _supports_transform = True
    _feature_prefix = "ica"

    def __init__(
        self,
        n_components: Optional[int] = None,
        algorithm: str = "parallel",
        whiten: Union[str, bool] = "unit-variance",
        fun: Union[str, Callable[..., Any]] = "logcosh",
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: RandomState = None,
    ) -> None:
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _build_estimator(self, n_samples: Optional[int] = None) -> FastICA:
        return FastICA(
            n_components=self.n_components,
            algorithm=self.algorithm,
            whiten=self.whiten,
            fun=self.fun,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )

    def get_mixing_matrix(self) -> np.ndarray:
        """Return the mixing matrix ``A`` of shape ``(n_features, n_components)``."""
        return np.asarray(self._fitted_attr("mixing_"))

    def get_unmixing_matrix(self) -> np.ndarray:
        """Return the un-mixing matrix ``W`` of shape ``(n_components, n_features)``."""
        return self.get_components()

    def get_n_iter(self) -> int:
        """Return the number of iterations run by FastICA."""
        return int(self._fitted_attr("n_iter_"))


class NMFModel(DimensionalityReductionModel):
    """Non-negative Matrix Factorisation wrapper around :class:`sklearn.decomposition.NMF`.

    Requires non-negative input; produces non-negative codes and components.

    Args:
        n_components: Number of components. ``"auto"`` infers it from ``init``
            when ``init="custom"``, otherwise keeps all features.
        init: Initialisation: ``None``, ``"random"``, ``"nndsvd"``, ``"nndsvda"``,
            ``"nndsvdar"`` or ``"custom"``.
        solver: ``"cd"`` (coordinate descent) or ``"mu"`` (multiplicative update).
        beta_loss: Beta divergence to minimise (``"frobenius"``,
            ``"kullback-leibler"``, ``"itakura-saito"`` or a float).
        max_iter: Maximum number of iterations.
        tol: Stopping tolerance.
        alpha_W: Regularisation strength on ``W``.
        alpha_H: Regularisation strength on ``H`` (``"same"`` copies ``alpha_W``).
        l1_ratio: Mix between L1 (1.0) and L2 (0.0) penalties.
        random_state: Seed for initialisation and the ``mu`` solver.
    """

    _supports_transform = True
    _accept_sparse = ("csr", "csc")
    _feature_prefix = "nmf"

    def __init__(
        self,
        n_components: Optional[Union[int, str]] = "auto",
        init: Optional[str] = None,
        solver: str = "cd",
        beta_loss: Union[str, float] = "frobenius",
        max_iter: int = 200,
        tol: float = 1e-4,
        alpha_W: float = 0.0,
        alpha_H: Union[float, str] = "same",
        l1_ratio: float = 0.0,
        random_state: RandomState = None,
    ) -> None:
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.beta_loss = beta_loss
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.l1_ratio = l1_ratio
        self.random_state = random_state

    def _build_estimator(self, n_samples: Optional[int] = None) -> NMF:
        return NMF(
            **_filter_kwargs(
                NMF,
                n_components=self.n_components,
                init=self.init,
                solver=self.solver,
                beta_loss=self.beta_loss,
                max_iter=self.max_iter,
                tol=self.tol,
                alpha_W=self.alpha_W,
                alpha_H=self.alpha_H,
                l1_ratio=self.l1_ratio,
                random_state=self.random_state,
            )
        )

    def get_reconstruction_error(self) -> float:
        """Return the Frobenius (or beta-divergence) reconstruction error after fitting."""
        return float(self._fitted_attr("reconstruction_err_"))

    def get_n_iter(self) -> int:
        """Return the number of iterations run by the solver."""
        return int(self._fitted_attr("n_iter_"))


class DictionaryLearningModel(DimensionalityReductionModel):
    """Sparse dictionary learning wrapper around :class:`sklearn.decomposition.DictionaryLearning`.

    Learns an over- or under-complete dictionary and encodes data as sparse
    codes. ``inverse_transform`` reconstructs data as ``code @ dictionary``.

    Args:
        n_components: Number of dictionary atoms (``None`` = ``n_features``).
        alpha: Sparsity controlling parameter.
        max_iter: Maximum number of iterations.
        tol: Tolerance for numerical error.
        fit_algorithm: ``"lars"`` or ``"cd"``.
        transform_algorithm: Encoding algorithm (``"omp"``, ``"lasso_lars"``,
            ``"lasso_cd"``, ``"lars"`` or ``"threshold"``).
        transform_n_nonzero_coefs: Non-zero coefficients per sample (``omp``/``lars``).
        transform_alpha: Penalty for the ``lasso_*`` / ``threshold`` encoders.
        positive_code: Enforce non-negative codes.
        positive_dict: Enforce non-negative dictionary atoms.
        n_jobs: Parallel jobs.
        random_state: Seed for dictionary initialisation.
    """

    _supports_transform = True
    _feature_prefix = "dict"

    def __init__(
        self,
        n_components: Optional[int] = None,
        alpha: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-8,
        fit_algorithm: str = "lars",
        transform_algorithm: str = "omp",
        transform_n_nonzero_coefs: Optional[int] = None,
        transform_alpha: Optional[float] = None,
        positive_code: bool = False,
        positive_dict: bool = False,
        n_jobs: Optional[int] = None,
        random_state: RandomState = None,
    ) -> None:
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_algorithm = fit_algorithm
        self.transform_algorithm = transform_algorithm
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.transform_alpha = transform_alpha
        self.positive_code = positive_code
        self.positive_dict = positive_dict
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _build_estimator(self, n_samples: Optional[int] = None) -> DictionaryLearning:
        return DictionaryLearning(
            n_components=self.n_components,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_algorithm=self.fit_algorithm,
            transform_algorithm=self.transform_algorithm,
            transform_n_nonzero_coefs=self.transform_n_nonzero_coefs,
            transform_alpha=self.transform_alpha,
            positive_code=self.positive_code,
            positive_dict=self.positive_dict,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

    def get_dictionary(self) -> np.ndarray:
        """Return the learned dictionary of shape ``(n_components, n_features)``."""
        return self.get_components()

    def inverse_transform(self, X: ArrayLike) -> np.ndarray:
        """Reconstruct samples from sparse codes as ``codes @ dictionary``.

        Args:
            X: Sparse codes of shape ``(n_samples, n_components)``.

        Returns:
            Reconstruction of shape ``(n_samples, n_features)``.
        """
        check_is_fitted(self, "model_")
        codes = check_array(X, dtype=np.float64)
        return codes @ self.get_components()

    def get_n_iter(self) -> int:
        """Return the number of iterations run."""
        return int(self._fitted_attr("n_iter_"))


class FactorAnalysisModel(DimensionalityReductionModel):
    """Factor Analysis wrapper around :class:`sklearn.decomposition.FactorAnalysis`.

    A linear-Gaussian latent variable model with per-feature noise variance.
    ``inverse_transform`` reconstructs data as ``Z @ loadings + mean``.

    Args:
        n_components: Number of latent factors (``None`` = ``n_features``).
        tol: Stopping tolerance for log-likelihood increase.
        max_iter: Maximum number of EM iterations.
        noise_variance_init: Initial guess for the per-feature noise variance.
        svd_method: ``"lapack"`` or ``"randomized"``.
        iterated_power: Power iterations for the randomized SVD.
        rotation: ``None``, ``"varimax"`` or ``"quartimax"``.
        random_state: Seed for the randomized SVD.
    """

    _supports_transform = True
    _feature_prefix = "factor"

    def __init__(
        self,
        n_components: Optional[int] = None,
        tol: float = 1e-2,
        max_iter: int = 1000,
        noise_variance_init: Optional[ArrayLike] = None,
        svd_method: str = "randomized",
        iterated_power: int = 3,
        rotation: Optional[str] = None,
        random_state: RandomState = None,
    ) -> None:
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.noise_variance_init = noise_variance_init
        self.svd_method = svd_method
        self.iterated_power = iterated_power
        self.rotation = rotation
        self.random_state = random_state

    def _build_estimator(self, n_samples: Optional[int] = None) -> FactorAnalysis:
        return FactorAnalysis(
            n_components=self.n_components,
            tol=self.tol,
            max_iter=self.max_iter,
            noise_variance_init=self.noise_variance_init,
            svd_method=self.svd_method,
            iterated_power=self.iterated_power,
            rotation=self.rotation,
            random_state=self.random_state,
        )

    def get_loadings(self) -> np.ndarray:
        """Return factor loadings of shape ``(n_components, n_features)``."""
        return self.get_components()

    def get_noise_variance(self) -> np.ndarray:
        """Return the estimated per-feature noise variance (length ``n_features``)."""
        return np.asarray(self._fitted_attr("noise_variance_"))

    def get_covariance(self) -> np.ndarray:
        """Return the model covariance ``W^T W + diag(psi)``."""
        check_is_fitted(self, "model_")
        return np.asarray(self.model_.get_covariance())

    def score(self, X: ArrayLike, y: Any = None) -> float:
        """Average log-likelihood of ``X`` under the fitted factor model.

        Args:
            X: Data of shape ``(n_samples, n_features)``.
            y: Ignored.

        Returns:
            Mean per-sample log-likelihood.
        """
        check_is_fitted(self, "model_")
        return float(self.model_.score(self._validate_X(X, reset=False)))

    def score_samples(self, X: ArrayLike) -> np.ndarray:
        """Per-sample log-likelihood of ``X`` under the fitted factor model."""
        check_is_fitted(self, "model_")
        return np.asarray(self.model_.score_samples(self._validate_X(X, reset=False)))

    def inverse_transform(self, X: ArrayLike) -> np.ndarray:
        """Reconstruct samples from factor scores as ``Z @ loadings + mean``.

        Args:
            X: Factor scores of shape ``(n_samples, n_components)``.

        Returns:
            Reconstruction of shape ``(n_samples, n_features)``.
        """
        check_is_fitted(self, "model_")
        Z = check_array(X, dtype=np.float64)
        return Z @ self.get_components() + np.asarray(self._fitted_attr("mean_"))


class KernelPCAModel(DimensionalityReductionModel):
    """Kernel PCA wrapper around :class:`sklearn.decomposition.KernelPCA`.

    Args:
        n_components: Number of components (``None`` = all non-zero).
        kernel: ``"linear"``, ``"poly"``, ``"rbf"``, ``"sigmoid"``, ``"cosine"``,
            ``"precomputed"`` or a callable.
        gamma: Kernel coefficient for ``rbf`` / ``poly`` / ``sigmoid``.
        degree: Degree of the polynomial kernel.
        coef0: Independent term for ``poly`` / ``sigmoid``.
        kernel_params: Extra parameters for a callable kernel.
        alpha: Ridge regularisation for the inverse transform.
        fit_inverse_transform: Learn the pre-image mapping so that
            :meth:`inverse_transform` is available (default ``True``; not
            allowed with ``kernel="precomputed"``).
        eigen_solver: ``"auto"``, ``"dense"``, ``"arpack"`` or ``"randomized"``.
        tol: Tolerance for arpack.
        max_iter: Maximum iterations for arpack.
        remove_zero_eig: Drop components with zero eigenvalues.
        n_jobs: Parallel jobs for the kernel computation.
        random_state: Seed for ``arpack`` / ``randomized`` solvers.
    """

    _supports_transform = True
    _feature_prefix = "kpca"

    def __init__(
        self,
        n_components: Optional[int] = None,
        kernel: Union[str, Callable[..., Any]] = "linear",
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 1.0,
        kernel_params: Optional[Dict[str, Any]] = None,
        alpha: float = 1.0,
        fit_inverse_transform: bool = True,
        eigen_solver: str = "auto",
        tol: float = 0.0,
        max_iter: Optional[int] = None,
        remove_zero_eig: bool = False,
        n_jobs: Optional[int] = None,
        random_state: RandomState = None,
    ) -> None:
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.remove_zero_eig = remove_zero_eig
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _build_estimator(self, n_samples: Optional[int] = None) -> KernelPCA:
        fit_inverse = self.fit_inverse_transform and self.kernel != "precomputed"
        return KernelPCA(
            n_components=self.n_components,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            kernel_params=self.kernel_params,
            alpha=self.alpha,
            fit_inverse_transform=fit_inverse,
            eigen_solver=self.eigen_solver,
            tol=self.tol,
            max_iter=self.max_iter,
            remove_zero_eig=self.remove_zero_eig,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

    def inverse_transform(self, X: ArrayLike) -> np.ndarray:
        """Approximate pre-images of embedded samples.

        Raises:
            NotImplementedError: If the wrapper was built with
                ``fit_inverse_transform=False`` or a precomputed kernel.
        """
        check_is_fitted(self, "model_")
        if not getattr(self.model_, "fit_inverse_transform", False):
            raise NotImplementedError(
                "KernelPCAModel.inverse_transform requires fit_inverse_transform=True "
                "and a non-precomputed kernel."
            )
        return super().inverse_transform(X)

    def get_eigenvalues(self) -> np.ndarray:
        """Return the eigenvalues of the centred kernel matrix, descending."""
        return np.asarray(self._fitted_attr("eigenvalues_"))

    def get_eigenvectors(self) -> np.ndarray:
        """Return the eigenvectors of the centred kernel matrix."""
        return np.asarray(self._fitted_attr("eigenvectors_"))

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Eigenvalue share of each retained component (kernel-space variance).

        Kernel PCA has no ``explained_variance_ratio_`` of its own; this returns
        each retained eigenvalue divided by the sum of retained eigenvalues.
        """
        eig = np.clip(self.get_eigenvalues(), 0.0, None)
        total = eig.sum()
        return eig / total if total > 0 else eig


# --------------------------------------------------------------------------- #
# Manifold learning
# --------------------------------------------------------------------------- #


class TSNEModel(DimensionalityReductionModel):
    """t-SNE wrapper around :class:`sklearn.manifold.TSNE`.

    t-SNE has no out-of-sample extension: only :meth:`fit_transform` yields an
    embedding, and :meth:`transform` raises :class:`NotImplementedError`.
    The perplexity is clipped automatically when it is not strictly smaller
    than the number of samples.

    Args:
        n_components: Embedding dimension (``barnes_hut`` requires ``<= 3``;
            larger values switch to the ``exact`` method automatically).
        perplexity: Effective number of neighbours per point.
        early_exaggeration: Cluster tightness during the early phase.
        learning_rate: Gradient-descent step size or ``"auto"``.
        max_iter: Maximum optimisation iterations (``>= 250``).
        n_iter: Deprecated alias for ``max_iter`` kept for backward compatibility;
            when given it overrides ``max_iter``.
        init: ``"pca"``, ``"random"`` or an array.
        metric: Distance metric for the input space.
        method: ``"barnes_hut"`` or ``"exact"``.
        angle: Barnes-Hut trade-off between speed and accuracy.
        n_jobs: Parallel jobs for neighbour search.
        random_state: Seed for initialisation.
    """

    _supports_transform = False
    _feature_prefix = "tsne"

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        early_exaggeration: float = 12.0,
        learning_rate: Union[float, str] = "auto",
        max_iter: int = 1000,
        n_iter: Optional[int] = None,
        init: Union[str, np.ndarray] = "pca",
        metric: str = "euclidean",
        method: str = "barnes_hut",
        angle: float = 0.5,
        n_jobs: Optional[int] = None,
        random_state: RandomState = None,
    ) -> None:
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.n_iter = n_iter
        self.init = init
        self.metric = metric
        self.method = method
        self.angle = angle
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _effective_perplexity(self, n_samples: Optional[int]) -> float:
        """Clip perplexity so that it is strictly below ``n_samples``."""
        if n_samples is None or self.perplexity < n_samples:
            return float(self.perplexity)
        clipped = max(1.0, (n_samples - 1) / 3.0)
        self.logger.warning(
            "perplexity=%s must be < n_samples=%d; using perplexity=%.2f", self.perplexity, n_samples, clipped
        )
        return clipped

    def _build_estimator(self, n_samples: Optional[int] = None) -> TSNE:
        max_iter = self.max_iter if self.n_iter is None else int(self.n_iter)
        method = self.method
        if method == "barnes_hut" and self.n_components > 3:
            self.logger.warning("barnes_hut supports n_components <= 3; switching to method='exact'")
            method = "exact"
        kwargs = dict(
            n_components=self.n_components,
            perplexity=self._effective_perplexity(n_samples),
            early_exaggeration=self.early_exaggeration,
            learning_rate=self.learning_rate,
            init=self.init,
            metric=self.metric,
            method=method,
            angle=self.angle,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        params = inspect.signature(TSNE.__init__).parameters
        kwargs["max_iter" if "max_iter" in params else "n_iter"] = max_iter
        return TSNE(**_filter_kwargs(TSNE, **kwargs))

    def get_kl_divergence(self) -> float:
        """Return the final Kullback-Leibler divergence of the embedding."""
        return float(self._fitted_attr("kl_divergence_"))

    def get_n_iter(self) -> int:
        """Return the number of optimisation iterations run."""
        return int(self._fitted_attr("n_iter_"))


class UMAPModel(DimensionalityReductionModel):
    """UMAP wrapper around :class:`umap.UMAP` with an optional Isomap fallback.

    ``umap-learn`` is an optional dependency (``pip install umap-learn``). When
    it is unavailable and ``allow_fallback`` is ``True`` (the default), the
    wrapper logs a warning and fits :class:`sklearn.manifold.Isomap` with the
    same ``n_components`` / ``n_neighbors`` so that pipelines keep running;
    the attribute ``backend_`` records which backend produced the embedding.
    Set ``allow_fallback=False`` to raise :class:`ImportError` instead.

    Args:
        n_components: Embedding dimension.
        n_neighbors: Size of the local neighbourhood used for manifold
            approximation.
        min_dist: Minimum distance between embedded points.
        metric: Distance metric in the input space.
        n_epochs: Number of optimisation epochs (``None`` = automatic).
        learning_rate: Initial learning rate for the SGD optimiser.
        spread: Effective scale of embedded points.
        random_state: Seed. Setting it disables UMAP's multi-threaded
            (non-deterministic) optimisation.
        allow_fallback: Use Isomap when ``umap-learn`` is not installed.

    Attributes:
        backend_: ``"umap"`` or ``"isomap"`` after fitting.
    """

    _supports_transform = True
    _feature_prefix = "umap"

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        n_epochs: Optional[int] = None,
        learning_rate: float = 1.0,
        spread: float = 1.0,
        random_state: RandomState = None,
        allow_fallback: bool = True,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.spread = spread
        self.random_state = random_state
        self.allow_fallback = allow_fallback

    def _build_estimator(self, n_samples: Optional[int] = None) -> Any:
        n_neighbors = self.n_neighbors
        if n_samples is not None and n_neighbors >= n_samples:
            n_neighbors = max(2, n_samples - 1)
            self.logger.warning(
                "n_neighbors=%d must be < n_samples=%d; using n_neighbors=%d",
                self.n_neighbors,
                n_samples,
                n_neighbors,
            )
        if HAS_UMAP:
            self._backend = "umap"
            return _umap.UMAP(
                n_components=self.n_components,
                n_neighbors=n_neighbors,
                min_dist=self.min_dist,
                metric=self.metric,
                n_epochs=self.n_epochs,
                learning_rate=self.learning_rate,
                spread=self.spread,
                random_state=self.random_state,
            )
        if not self.allow_fallback:
            raise ImportError(
                "UMAPModel requires the optional dependency 'umap-learn' (pip install umap-learn); "
                "pass allow_fallback=True to use an Isomap approximation instead."
            )
        self.logger.warning("umap-learn is not installed; UMAPModel is falling back to sklearn Isomap.")
        self._backend = "isomap"
        return Isomap(n_components=self.n_components, n_neighbors=n_neighbors)

    def _finalize_fit(self, estimator: Any, transformed: Optional[np.ndarray] = None) -> None:
        super()._finalize_fit(estimator, transformed)
        self.backend_ = self._backend

    def inverse_transform(self, X: ArrayLike) -> np.ndarray:
        """Approximate pre-images (UMAP backend only).

        Raises:
            NotImplementedError: When running on the Isomap fallback.
        """
        check_is_fitted(self, "model_")
        if self.backend_ != "umap":
            raise NotImplementedError("inverse_transform is only available with the umap-learn backend.")
        return super().inverse_transform(X)


class IsoMapModel(DimensionalityReductionModel):
    """Isomap wrapper around :class:`sklearn.manifold.Isomap`.

    Isomap is deterministic (no ``random_state``) and supports out-of-sample
    :meth:`transform`.

    Args:
        n_components: Embedding dimension.
        n_neighbors: Neighbours per point for the graph (ignored when ``radius`` is set).
        radius: Neighbourhood radius (alternative to ``n_neighbors``).
        eigen_solver: ``"auto"``, ``"arpack"`` or ``"dense"``.
        tol: Convergence tolerance for arpack.
        max_iter: Maximum iterations for arpack.
        path_method: Shortest-path algorithm: ``"auto"``, ``"FW"`` or ``"D"``.
        neighbors_algorithm: Nearest-neighbour search algorithm.
        metric: Distance metric for the neighbour graph.
        p: Minkowski power parameter.
        n_jobs: Parallel jobs.
    """

    _supports_transform = True
    _feature_prefix = "isomap"

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: Optional[int] = 5,
        radius: Optional[float] = None,
        eigen_solver: str = "auto",
        tol: float = 0.0,
        max_iter: Optional[int] = None,
        path_method: str = "auto",
        neighbors_algorithm: str = "auto",
        metric: str = "minkowski",
        p: int = 2,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.path_method = path_method
        self.neighbors_algorithm = neighbors_algorithm
        self.metric = metric
        self.p = p
        self.n_jobs = n_jobs

    def _build_estimator(self, n_samples: Optional[int] = None) -> Isomap:
        return Isomap(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            radius=self.radius,
            eigen_solver=self.eigen_solver,
            tol=self.tol,
            max_iter=self.max_iter,
            path_method=self.path_method,
            neighbors_algorithm=self.neighbors_algorithm,
            metric=self.metric,
            p=self.p,
            n_jobs=self.n_jobs,
        )

    def get_reconstruction_error(self) -> float:
        """Return Isomap's reconstruction error of the geodesic distance matrix."""
        check_is_fitted(self, "model_")
        return float(self.model_.reconstruction_error())

    def get_geodesic_distances(self) -> np.ndarray:
        """Return the training-set geodesic distance matrix."""
        return np.asarray(self._fitted_attr("dist_matrix_"))


class LLEModel(DimensionalityReductionModel):
    """Locally Linear Embedding wrapper around :class:`sklearn.manifold.LocallyLinearEmbedding`.

    Args:
        n_components: Embedding dimension.
        n_neighbors: Neighbours per point (must exceed ``n_components``;
            the ``hessian`` variant needs ``> n_components * (n_components + 3) / 2``).
        reg: Regularisation constant.
        eigen_solver: ``"auto"``, ``"arpack"`` or ``"dense"``.
        tol: Tolerance for arpack.
        max_iter: Maximum iterations for arpack.
        method: ``"standard"``, ``"hessian"``, ``"modified"`` or ``"ltsa"``.
        hessian_tol: Tolerance for the Hessian eigen-mapping.
        modified_tol: Tolerance for modified LLE.
        neighbors_algorithm: Nearest-neighbour search algorithm.
        n_jobs: Parallel jobs.
        random_state: Seed for the arpack solver.
    """

    _supports_transform = True
    _feature_prefix = "lle"

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 5,
        reg: float = 1e-3,
        eigen_solver: str = "auto",
        tol: float = 1e-6,
        max_iter: int = 100,
        method: str = "standard",
        hessian_tol: float = 1e-4,
        modified_tol: float = 1e-12,
        neighbors_algorithm: str = "auto",
        n_jobs: Optional[int] = None,
        random_state: RandomState = None,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.reg = reg
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.method = method
        self.hessian_tol = hessian_tol
        self.modified_tol = modified_tol
        self.neighbors_algorithm = neighbors_algorithm
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _build_estimator(self, n_samples: Optional[int] = None) -> LocallyLinearEmbedding:
        return LocallyLinearEmbedding(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            reg=self.reg,
            eigen_solver=self.eigen_solver,
            tol=self.tol,
            max_iter=self.max_iter,
            method=self.method,
            hessian_tol=self.hessian_tol,
            modified_tol=self.modified_tol,
            neighbors_algorithm=self.neighbors_algorithm,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

    def get_reconstruction_error(self) -> float:
        """Return the LLE reconstruction error associated with the embedding."""
        return float(self._fitted_attr("reconstruction_error_"))


class SpectralEmbeddingModel(DimensionalityReductionModel):
    """Laplacian Eigenmaps wrapper around :class:`sklearn.manifold.SpectralEmbedding`.

    Spectral embedding has no out-of-sample extension; use :meth:`fit_transform`.

    Args:
        n_components: Embedding dimension.
        affinity: ``"nearest_neighbors"``, ``"rbf"``, ``"precomputed"``,
            ``"precomputed_nearest_neighbors"`` or a callable.
        gamma: Kernel coefficient for the ``rbf`` affinity.
        eigen_solver: ``None``, ``"arpack"``, ``"lobpcg"`` or ``"amg"``.
        n_neighbors: Neighbours for the ``nearest_neighbors`` affinity
            (``None`` = ``max(n_samples / 10, 1)``).
        n_jobs: Parallel jobs.
        random_state: Seed for the eigen-solver initialisation.
    """

    _supports_transform = False
    _feature_prefix = "spectral"

    def __init__(
        self,
        n_components: int = 2,
        affinity: Union[str, Callable[..., Any]] = "nearest_neighbors",
        gamma: Optional[float] = None,
        eigen_solver: Optional[str] = None,
        n_neighbors: Optional[int] = None,
        n_jobs: Optional[int] = None,
        random_state: RandomState = None,
    ) -> None:
        self.n_components = n_components
        self.affinity = affinity
        self.gamma = gamma
        self.eigen_solver = eigen_solver
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _build_estimator(self, n_samples: Optional[int] = None) -> SpectralEmbedding:
        return SpectralEmbedding(
            n_components=self.n_components,
            affinity=self.affinity,
            gamma=self.gamma,
            eigen_solver=self.eigen_solver,
            n_neighbors=self.n_neighbors,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

    def get_affinity_matrix(self) -> Any:
        """Return the affinity matrix built during fitting (may be sparse)."""
        return self._fitted_attr("affinity_matrix_")


class MDSModel(DimensionalityReductionModel):
    """Multidimensional Scaling wrapper around :class:`sklearn.manifold.MDS`.

    MDS has no out-of-sample extension; use :meth:`fit_transform`. Parameter
    names that changed across scikit-learn releases (``metric`` vs
    ``metric_mds``, ``dissimilarity`` vs ``metric``) are mapped automatically.

    Args:
        n_components: Embedding dimension.
        metric: ``True`` for metric MDS, ``False`` for non-metric MDS.
        n_init: Number of SMACOF initialisations; the best (lowest stress) is kept.
        init: Initialisation strategy (``"random"`` or ``"classical_mds"``) on
            scikit-learn releases that support it.
        max_iter: Maximum SMACOF iterations per initialisation.
        eps: Relative stress tolerance for convergence.
        dissimilarity: ``"euclidean"`` or ``"precomputed"``.
        normalized_stress: Whether to report normalised stress (``"auto"`` or bool).
        n_jobs: Parallel jobs across initialisations.
        random_state: Seed for the random initialisations.
    """

    _supports_transform = False
    _feature_prefix = "mds"

    def __init__(
        self,
        n_components: int = 2,
        metric: bool = True,
        n_init: int = 1,
        init: str = "random",
        max_iter: int = 300,
        eps: float = 1e-6,
        dissimilarity: str = "euclidean",
        normalized_stress: Union[bool, str] = "auto",
        n_jobs: Optional[int] = None,
        random_state: RandomState = None,
    ) -> None:
        self.n_components = n_components
        self.metric = metric
        self.n_init = n_init
        self.init = init
        self.max_iter = max_iter
        self.eps = eps
        self.dissimilarity = dissimilarity
        self.normalized_stress = normalized_stress
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _build_estimator(self, n_samples: Optional[int] = None) -> MDS:
        params = inspect.signature(MDS.__init__).parameters
        kwargs: Dict[str, Any] = dict(
            n_components=self.n_components,
            n_init=self.n_init,
            init=self.init,
            max_iter=self.max_iter,
            eps=self.eps,
            normalized_stress=self.normalized_stress,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        if "metric_mds" in params:  # scikit-learn >= 1.8: metric_mds + metric (distance)
            kwargs["metric_mds"] = self.metric
            kwargs["metric"] = self.dissimilarity
        else:  # older releases: metric (bool) + dissimilarity
            kwargs["metric"] = self.metric
            kwargs["dissimilarity"] = self.dissimilarity
        return MDS(**_filter_kwargs(MDS, **kwargs))

    def get_stress(self) -> float:
        """Return the final stress value of the best SMACOF run."""
        return float(self._fitted_attr("stress_"))

    def get_dissimilarity_matrix(self) -> np.ndarray:
        """Return the pairwise dissimilarities used for fitting."""
        return np.asarray(self._fitted_attr("dissimilarity_matrix_"))


# --------------------------------------------------------------------------- #
# Dispatchers, factories and notebook-facing aliases
# --------------------------------------------------------------------------- #

MODEL_REGISTRY: Dict[str, type] = {
    "pca": PCAModel,
    "svd": TruncatedSVDModel,
    "truncated_svd": TruncatedSVDModel,
    "ica": ICAModel,
    "nmf": NMFModel,
    "dictionary_learning": DictionaryLearningModel,
    "factor_analysis": FactorAnalysisModel,
    "kernel_pca": KernelPCAModel,
    "tsne": TSNEModel,
    "umap": UMAPModel,
    "isomap": IsoMapModel,
    "lle": LLEModel,
    "spectral": SpectralEmbeddingModel,
    "spectral_embedding": SpectralEmbeddingModel,
    "mds": MDSModel,
}
"""Mapping from short method names to wrapper classes."""

_MANIFOLD_METHODS = ("isomap", "lle", "tsne", "spectral", "spectral_embedding", "mds", "umap")


class ManifoldLearning(DimensionalityReductionModel):
    """Dispatching wrapper that selects a manifold learner by name.

    Useful for notebooks and comparison loops where the algorithm is chosen
    from a string, e.g. ``ManifoldLearning(method="isomap", n_components=2)``.

    Args:
        method: One of ``"isomap"``, ``"lle"``, ``"tsne"``, ``"spectral"``,
            ``"mds"`` or ``"umap"``.
        n_components: Embedding dimension.
        n_neighbors: Neighbourhood size, forwarded to methods that use one.
        random_state: Seed, forwarded to methods that accept one.

    Attributes:
        reducer_: The concrete wrapper used for fitting.

    Raises:
        ValueError: If ``method`` is not a recognised manifold learner.
    """

    _feature_prefix = "manifold"

    def __init__(
        self,
        method: str = "isomap",
        n_components: int = 2,
        n_neighbors: Optional[int] = None,
        random_state: RandomState = None,
    ) -> None:
        self.method = method
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    @property
    def _supports_transform(self) -> bool:  # type: ignore[override]
        return self._make_reducer()._supports_transform

    def _make_reducer(self) -> DimensionalityReductionModel:
        method = str(self.method).lower()
        if method not in _MANIFOLD_METHODS:
            raise ValueError(f"Unknown manifold method '{self.method}'. Choose from {_MANIFOLD_METHODS}.")
        cls = MODEL_REGISTRY[method]
        kwargs: Dict[str, Any] = {"n_components": self.n_components}
        params = inspect.signature(cls.__init__).parameters
        if "random_state" in params:
            kwargs["random_state"] = self.random_state
        if self.n_neighbors is not None and "n_neighbors" in params:
            kwargs["n_neighbors"] = self.n_neighbors
        return cls(**kwargs)

    def _build_estimator(self, n_samples: Optional[int] = None) -> Any:
        reducer = self._make_reducer()
        self._reducer = reducer
        return reducer._build_estimator(n_samples=n_samples)

    def _finalize_fit(self, estimator: Any, transformed: Optional[np.ndarray] = None) -> None:
        super()._finalize_fit(estimator, transformed)
        reducer = self._reducer
        reducer.n_features_in_ = self.n_features_in_
        reducer._finalize_fit(estimator, transformed)
        self.reducer_ = reducer

    def __getattr__(self, name: str) -> Any:
        # Delegate ``get_*`` accessors (get_stress, get_kl_divergence, ...) to the fitted reducer.
        if name.startswith("get_") and "reducer_" in self.__dict__:
            return getattr(self.__dict__["reducer_"], name)
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")


class DimensionalityReduction(LoggerMixin):
    """Factory for configured dimensionality reduction wrappers.

    Mirrors the ``get_<method>`` convenience style used elsewhere in the
    package (``dim_reducer.get_pca(n_components=10)``). Every factory method
    forwards keyword arguments to the wrapper and injects the factory's
    default ``random_state`` when the wrapper accepts one.

    Args:
        random_state: Default seed used by every model this factory creates.
    """

    def __init__(self, random_state: RandomState = settings.RANDOM_SEED) -> None:
        self.random_state = random_state

    @staticmethod
    def available_methods() -> List[str]:
        """Return the sorted list of method names accepted by :meth:`get_model`."""
        return sorted(MODEL_REGISTRY)

    def get_model(self, method: str, **kwargs: Any) -> DimensionalityReductionModel:
        """Instantiate the wrapper registered under ``method``.

        Args:
            method: Key of :data:`MODEL_REGISTRY` (case-insensitive).
            **kwargs: Constructor arguments for the wrapper.

        Returns:
            An unfitted wrapper.

        Raises:
            ValueError: If ``method`` is unknown.
        """
        key = str(method).lower()
        if key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown method '{method}'. Available: {self.available_methods()}")
        cls = MODEL_REGISTRY[key]
        if "random_state" in inspect.signature(cls.__init__).parameters:
            kwargs.setdefault("random_state", self.random_state)
        self.logger.debug("Creating %s with %s", cls.__name__, kwargs)
        return cls(**kwargs)

    def get_pca(self, n_components: Optional[Union[int, float, str]] = None, **kwargs: Any) -> PCAModel:
        """Create a :class:`PCAModel`."""
        return self.get_model("pca", n_components=n_components, **kwargs)

    def get_svd(self, n_components: int = 2, **kwargs: Any) -> TruncatedSVDModel:
        """Create a :class:`TruncatedSVDModel`."""
        return self.get_model("svd", n_components=n_components, **kwargs)

    def get_ica(self, n_components: Optional[int] = None, **kwargs: Any) -> ICAModel:
        """Create an :class:`ICAModel`."""
        return self.get_model("ica", n_components=n_components, **kwargs)

    def get_nmf(self, n_components: Optional[Union[int, str]] = "auto", **kwargs: Any) -> NMFModel:
        """Create an :class:`NMFModel`."""
        return self.get_model("nmf", n_components=n_components, **kwargs)

    def get_dictionary_learning(
        self, n_components: Optional[int] = None, **kwargs: Any
    ) -> DictionaryLearningModel:
        """Create a :class:`DictionaryLearningModel`."""
        return self.get_model("dictionary_learning", n_components=n_components, **kwargs)

    def get_factor_analysis(self, n_components: Optional[int] = None, **kwargs: Any) -> FactorAnalysisModel:
        """Create a :class:`FactorAnalysisModel`."""
        return self.get_model("factor_analysis", n_components=n_components, **kwargs)

    def get_kernel_pca(
        self, n_components: Optional[int] = None, kernel: str = "rbf", **kwargs: Any
    ) -> KernelPCAModel:
        """Create a :class:`KernelPCAModel`."""
        return self.get_model("kernel_pca", n_components=n_components, kernel=kernel, **kwargs)

    def get_tsne(self, n_components: int = 2, **kwargs: Any) -> TSNEModel:
        """Create a :class:`TSNEModel`."""
        return self.get_model("tsne", n_components=n_components, **kwargs)

    def get_umap(self, n_components: int = 2, **kwargs: Any) -> UMAPModel:
        """Create a :class:`UMAPModel` (falls back to Isomap without ``umap-learn``)."""
        return self.get_model("umap", n_components=n_components, **kwargs)

    def get_isomap(self, n_components: int = 2, **kwargs: Any) -> IsoMapModel:
        """Create an :class:`IsoMapModel`."""
        return self.get_model("isomap", n_components=n_components, **kwargs)

    def get_lle(self, n_components: int = 2, **kwargs: Any) -> LLEModel:
        """Create an :class:`LLEModel`."""
        return self.get_model("lle", n_components=n_components, **kwargs)

    def get_spectral_embedding(self, n_components: int = 2, **kwargs: Any) -> SpectralEmbeddingModel:
        """Create a :class:`SpectralEmbeddingModel`."""
        return self.get_model("spectral", n_components=n_components, **kwargs)

    def get_mds(self, n_components: int = 2, **kwargs: Any) -> MDSModel:
        """Create an :class:`MDSModel`."""
        return self.get_model("mds", n_components=n_components, **kwargs)

    def compare(
        self,
        X: ArrayLike,
        methods: Optional[Sequence[str]] = None,
        n_components: int = 2,
        **kwargs: Any,
    ) -> Dict[str, np.ndarray]:
        """Embed ``X`` with several methods and return the embeddings.

        Methods that fail (for example NMF on data with negative entries) are
        logged and skipped rather than aborting the comparison.

        Args:
            X: Data of shape ``(n_samples, n_features)``.
            methods: Registry keys to run; defaults to ``("pca", "tsne", "isomap")``.
            n_components: Embedding dimension used for every method.
            **kwargs: Extra constructor arguments forwarded to every wrapper
                that accepts them.

        Returns:
            Mapping ``method -> embedding`` for the methods that succeeded.
        """
        methods = tuple(methods) if methods is not None else ("pca", "tsne", "isomap")
        results: Dict[str, np.ndarray] = {}
        for method in methods:
            cls = MODEL_REGISTRY.get(str(method).lower())
            if cls is None:
                self.logger.warning("Skipping unknown method '%s'", method)
                continue
            accepted = _filter_kwargs(cls, **kwargs)
            try:
                model = self.get_model(method, n_components=n_components, **accepted)
                results[str(method)] = model.fit_transform(X)
            except (ValueError, ImportError, NotImplementedError) as exc:
                self.logger.warning("Method '%s' failed: %s", method, exc)
        return results


# Notebook-facing aliases (kept for backward compatibility with older notebooks).
EnhancedPCA = PCAModel
AdaptiveTSNE = TSNEModel
UMAPEnhanced = UMAPModel
