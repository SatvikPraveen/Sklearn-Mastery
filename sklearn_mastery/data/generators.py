"""Synthetic data generators for the sklearn-mastery learning suite.

The central class is :class:`SyntheticDataGenerator`, which produces datasets
that each isolate one modelling challenge (multicollinearity, class imbalance,
non-linear decision boundaries, sparsity, seasonality, outliers, ...). Every
method is deterministic for a given ``random_state``: the generator derives a
dedicated :class:`numpy.random.RandomState` per call from its seed and the
method name, so results do not depend on the order in which methods are
invoked and different methods never share a noise stream.

``DataGenerator``, :class:`ClassificationDataGenerator`,
:class:`RegressionDataGenerator` and :class:`ClusteringDataGenerator` are
thin named views over the same functionality, kept for API compatibility with
the documentation and notebooks.

Example:
    >>> from sklearn_mastery.data.generators import SyntheticDataGenerator
    >>> gen = SyntheticDataGenerator(random_state=0)
    >>> X, y = gen.classification_complexity_spectrum("medium", n_samples=200)
    >>> X.shape, sorted(set(y))
    ((200, 2), [0, 1])
"""

from __future__ import annotations

import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import (
    make_blobs,
    make_circles,
    make_classification,
    make_friedman1,
    make_low_rank_matrix,
    make_moons,
    make_multilabel_classification,
    make_regression,
)

from sklearn_mastery.config.logging_config import LoggerMixin
from sklearn_mastery.config.settings import settings

RandomStateLike = Union[None, int, np.random.RandomState, np.random.Generator]
"""Accepted forms of a random state: ``None``, an integer seed or a NumPy RNG."""

COMPLEXITY_LEVELS: Tuple[str, ...] = ("linear", "medium", "high")
"""Recognised decision-boundary complexity levels, easiest first."""

_COMPLEXITY_ALIASES: Dict[str, str] = {"simple": "linear", "nonlinear": "medium", "complex": "high"}
_ORDINAL_LEVEL_NAMES: Dict[int, Tuple[str, ...]] = {
    2: ("low", "high"),
    3: ("low", "medium", "high"),
    4: ("low", "medium", "high", "very_high"),
    5: ("very_low", "low", "medium", "high", "very_high"),
}
_MAX_SEED = 2**32


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _check_int(name: str, value: Any, minimum: int = 1) -> int:
    """Validate that ``value`` is an integer ``>= minimum`` and return it as ``int``.

    Raises:
        TypeError: If ``value`` is not an integer (booleans are rejected).
        ValueError: If ``value`` is below ``minimum``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _check_float(
    name: str,
    value: Any,
    low: float = 0.0,
    high: float = 1.0,
    *,
    inclusive_low: bool = True,
    inclusive_high: bool = True,
) -> float:
    """Validate that ``value`` is a real number inside ``[low, high]`` and return it.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` lies outside the permitted interval.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    below = value < low if inclusive_low else value <= low
    above = value > high if inclusive_high else value >= high
    if below or above:
        lo_b, hi_b = ("[" if inclusive_low else "("), ("]" if inclusive_high else ")")
        raise ValueError(f"{name} must lie in {lo_b}{low}, {high}{hi_b}, got {value}")
    return value


def _derive_seed(seed: int, salt: str) -> int:
    """Combine an integer seed with a method name into a stable 32-bit seed."""
    return (int(seed) + zlib.crc32(salt.encode("utf-8"))) % _MAX_SEED


def _shuffle_rows(rng: np.random.RandomState, *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Apply one shared random permutation to the first axis of every array."""
    n = arrays[0].shape[0]
    order = rng.permutation(n)
    return tuple(a[order] for a in arrays)


def _unit_directions(rng: np.random.RandomState, n: int, n_features: int) -> np.ndarray:
    """Draw ``n`` random unit vectors in ``n_features`` dimensions."""
    directions = rng.standard_normal((n, n_features))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return directions / norms


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
class SyntheticDataGenerator(LoggerMixin):
    """Deterministic synthetic dataset factory covering the sklearn-mastery curriculum.

    Every public method returns plain NumPy arrays (or a :class:`pandas.DataFrame`
    where heterogeneous columns are the point of the dataset) and accepts an
    optional ``random_state`` override. The instance-level seed is combined with
    the method name, so each method draws from its own reproducible stream.

    Args:
        random_state: Integer seed that makes every method reproducible. Defaults
            to ``settings.RANDOM_SEED``. Pass ``None`` for non-deterministic output.

    Attributes:
        random_state: The seed given at construction (``None`` when unseeded).
        logger: Class-scoped logger provided by :class:`LoggerMixin`.
    """

    def __init__(self, random_state: Optional[int] = settings.RANDOM_SEED) -> None:
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer))
        ):
            raise TypeError(f"random_state must be an int or None, got {type(random_state).__name__}")
        self.random_state: Optional[int] = None if random_state is None else int(random_state)
        self.logger.debug("SyntheticDataGenerator initialised (random_state=%s)", self.random_state)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(random_state={self.random_state!r})"

    # ------------------------------------------------------------------ RNG
    def _rng(self, random_state: RandomStateLike, salt: str) -> np.random.RandomState:
        """Return the RandomState used by one method call.

        Args:
            random_state: Per-call override. ``None`` falls back to the instance seed.
            salt: Method name mixed into the seed so that different methods do not
                share a noise stream.

        Returns:
            A :class:`numpy.random.RandomState`. When both the override and the
            instance seed are ``None`` the state is drawn from OS entropy.
        """
        seed: RandomStateLike = self.random_state if random_state is None else random_state
        if seed is None:
            return np.random.RandomState()
        if isinstance(seed, np.random.RandomState):
            return seed
        if isinstance(seed, np.random.Generator):
            return np.random.RandomState(int(seed.integers(_MAX_SEED)))
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError(
                f"random_state must be an int, RandomState, Generator or None, got {type(seed).__name__}"
            )
        return np.random.RandomState(_derive_seed(int(seed), salt))

    def _log_dataset(self, name: str, X: Any, y: Any = None) -> None:
        shape = getattr(X, "shape", None)
        if y is None:
            self.logger.debug("%s: X=%s", name, shape)
        else:
            self.logger.debug("%s: X=%s, y=%s", name, shape, getattr(y, "shape", len(y)))

    # ------------------------------------------------------------ regression
    def linear_regression_data(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = settings.DEFAULT_N_FEATURES,
        noise_level: float = settings.DEFAULT_NOISE_LEVEL,
        effective_rank: Optional[int] = None,
        bias: float = 0.0,
        n_informative: Optional[int] = None,
        tail_strength: float = 0.5,
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a linear regression problem ``y = X w + bias + eps``.

        Features are centred, so ``bias`` equals the expected target value. The
        coefficients of informative features are drawn from ``U(-1, 1)`` and all
        other coefficients are zero.

        Args:
            n_samples: Number of rows.
            n_features: Number of columns.
            noise_level: Standard deviation of the Gaussian target noise.
            effective_rank: Approximate number of singular vectors needed to
                explain the design matrix (``None`` for a well-conditioned,
                i.i.d. Gaussian design).
            bias: Intercept added to the target.
            n_informative: Number of features with a non-zero coefficient.
                Defaults to ``min(10, n_features)``.
            tail_strength: Relative importance of the noisy singular-value tail
                when ``effective_rank`` is set.
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, y)`` with ``X`` of shape ``(n_samples, n_features)`` and
            ``y`` of shape ``(n_samples,)``.

        Raises:
            ValueError: If a size argument is non-positive or ``n_informative``
                exceeds ``n_features``.
        """
        n_samples = _check_int("n_samples", n_samples)
        n_features = _check_int("n_features", n_features)
        noise_level = _check_float("noise_level", noise_level, 0.0, np.inf)
        n_informative = (
            min(10, n_features) if n_informative is None else _check_int("n_informative", n_informative)
        )
        if n_informative > n_features:
            raise ValueError(f"n_informative ({n_informative}) cannot exceed n_features ({n_features})")
        rng = self._rng(random_state, "linear_regression_data")

        if effective_rank is None:
            X = rng.standard_normal((n_samples, n_features))
        else:
            effective_rank = _check_int("effective_rank", effective_rank)
            X = make_low_rank_matrix(
                n_samples=n_samples,
                n_features=n_features,
                effective_rank=effective_rank,
                tail_strength=tail_strength,
                random_state=rng,
            )
            X /= max(float(X.std()), 1e-12)  # unit overall scale, low-rank structure preserved
        if n_samples > 1:
            X = X - X.mean(axis=0)

        coef = np.zeros(n_features)
        informative = rng.choice(n_features, size=n_informative, replace=False)
        coef[informative] = rng.uniform(-1.0, 1.0, size=n_informative)
        y = X @ coef + bias + rng.normal(0.0, noise_level, size=n_samples)
        self._log_dataset("linear_regression_data", X, y)
        return X, y

    def regression_linear(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = settings.DEFAULT_N_FEATURES,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Alias of :meth:`linear_regression_data` (legacy name)."""
        return self.linear_regression_data(n_samples=n_samples, n_features=n_features, **kwargs)

    def regression_dataset(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = settings.DEFAULT_N_FEATURES,
        n_informative: Optional[int] = None,
        noise: float = settings.DEFAULT_NOISE_LEVEL,
        bias: float = 0.0,
        effective_rank: Optional[int] = None,
        tail_strength: float = 0.5,
        n_targets: int = 1,
        shuffle: bool = True,
        coef: bool = False,
        random_state: RandomStateLike = None,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Thin, seeded wrapper around :func:`sklearn.datasets.make_regression`.

        Args:
            n_samples: Number of rows.
            n_features: Number of columns.
            n_informative: Informative features (defaults to all of them).
            noise: Standard deviation of the Gaussian target noise.
            bias: Intercept.
            effective_rank: See :func:`sklearn.datasets.make_regression`.
            tail_strength: See :func:`sklearn.datasets.make_regression`.
            n_targets: Number of regression targets.
            shuffle: Shuffle samples and features.
            coef: Also return the true coefficients.
            random_state: Per-call seed override.

        Returns:
            ``(X, y)`` or ``(X, y, coef)`` when ``coef=True``.
        """
        n_samples = _check_int("n_samples", n_samples)
        n_features = _check_int("n_features", n_features)
        n_informative = (
            n_features
            if n_informative is None
            else min(_check_int("n_informative", n_informative), n_features)
        )
        rng = self._rng(random_state, "regression_dataset")
        result = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_targets=n_targets,
            bias=bias,
            effective_rank=effective_rank,
            tail_strength=tail_strength,
            noise=noise,
            shuffle=shuffle,
            coef=coef,
            random_state=rng,
        )
        self._log_dataset("regression_dataset", result[0], result[1])
        return result

    def generate_regression_data(self, *args: Any, **kwargs: Any) -> Any:
        """Alias of :meth:`regression_dataset` (documentation name)."""
        return self.regression_dataset(*args, **kwargs)

    def regression_with_collinearity(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = settings.DEFAULT_N_FEATURES,
        collinear_groups: Optional[Sequence[Sequence[int]]] = None,
        collinearity_strength: float = 0.95,
        noise_variance: float = 0.1,
        coefficient_sparsity: float = 0.3,
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a regression problem with blocks of highly correlated features.

        Within each group the first feature is drawn independently and the
        remaining members are noisy copies of it, so that the pairwise Pearson
        correlation inside a group is approximately ``collinearity_strength``.
        A fraction ``coefficient_sparsity`` of the true coefficients is exactly
        zero, which lets Lasso-style estimators be evaluated against the truth.

        Args:
            n_samples: Number of rows.
            n_features: Number of columns.
            collinear_groups: Iterable of index tuples; each tuple lists features
                that should be mutually collinear. Defaults to ``[(0, 1, 2), (3, 4)]``
                restricted to valid indices.
            collinearity_strength: Target within-group correlation in ``(0, 1)``.
            noise_variance: Variance of the Gaussian target noise.
            coefficient_sparsity: Fraction of coefficients forced to zero in ``[0, 1)``.
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, y, true_coef)``.

        Raises:
            ValueError: If a group is malformed, references an out-of-range feature,
                or a feature appears in more than one group.
        """
        n_samples = _check_int("n_samples", n_samples)
        n_features = _check_int("n_features", n_features)
        collinearity_strength = _check_float(
            "collinearity_strength",
            collinearity_strength,
            0.0,
            1.0,
            inclusive_low=False,
            inclusive_high=False,
        )
        noise_variance = _check_float("noise_variance", noise_variance, 0.0, np.inf)
        coefficient_sparsity = _check_float(
            "coefficient_sparsity", coefficient_sparsity, 0.0, 1.0, inclusive_high=False
        )

        if collinear_groups is None:
            collinear_groups = [g for g in ((0, 1, 2), (3, 4)) if max(g) < n_features]
        groups: List[Tuple[int, ...]] = [tuple(int(i) for i in g) for g in collinear_groups]
        seen: set = set()
        for group in groups:
            if len(group) < 2:
                raise ValueError(f"Each collinear group needs at least two features, got {group}")
            for idx in group:
                if not 0 <= idx < n_features:
                    raise ValueError(
                        f"Feature index {idx} in group {group} is out of range for n_features={n_features}"
                    )
                if idx in seen:
                    raise ValueError(f"Feature {idx} appears in more than one collinear group")
                seen.add(idx)

        rng = self._rng(random_state, "regression_with_collinearity")
        X = rng.standard_normal((n_samples, n_features))
        copy_noise_std = float(np.sqrt(1.0 / collinearity_strength**2 - 1.0))
        for group in groups:
            base = X[:, group[0]]
            for idx in group[1:]:
                X[:, idx] = base + rng.normal(0.0, copy_noise_std, size=n_samples)

        true_coef = rng.uniform(-3.0, 3.0, size=n_features)
        n_zero = round(coefficient_sparsity * n_features)
        if coefficient_sparsity > 0 and n_features > 1:
            n_zero = min(max(n_zero, 1), n_features - 1)
        if n_zero:
            true_coef[rng.choice(n_features, size=n_zero, replace=False)] = 0.0

        y = X @ true_coef + rng.normal(0.0, np.sqrt(noise_variance), size=n_samples)
        self._log_dataset("regression_with_collinearity", X, y)
        return X, y, true_coef

    def high_dimensional_regression(
        self,
        n_samples: int = 200,
        n_features: int = 500,
        n_informative: int = 10,
        noise_variance: float = 0.1,
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a ``p >> n`` style regression with a sparse coefficient vector.

        Args:
            n_samples: Number of rows.
            n_features: Number of columns (typically larger than ``n_samples``).
            n_informative: Number of non-zero coefficients.
            noise_variance: Variance of the Gaussian target noise.
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, y, true_coef)`` where ``true_coef`` has exactly
            ``n_informative`` non-zero entries.
        """
        n_samples = _check_int("n_samples", n_samples)
        n_features = _check_int("n_features", n_features)
        n_informative = _check_int("n_informative", n_informative)
        if n_informative > n_features:
            raise ValueError(f"n_informative ({n_informative}) cannot exceed n_features ({n_features})")
        noise_variance = _check_float("noise_variance", noise_variance, 0.0, np.inf)
        rng = self._rng(random_state, "high_dimensional_regression")
        X = rng.standard_normal((n_samples, n_features))
        true_coef = np.zeros(n_features)
        active = rng.choice(n_features, size=n_informative, replace=False)
        true_coef[active] = rng.choice([-1.0, 1.0], size=n_informative) * rng.uniform(
            1.0, 3.0, size=n_informative
        )
        y = X @ true_coef + rng.normal(0.0, np.sqrt(noise_variance), size=n_samples)
        self._log_dataset("high_dimensional_regression", X, y)
        return X, y, true_coef

    def nonlinear_regression(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = 10,
        nonlinearity_type: str = "polynomial",
        noise_level: float = settings.DEFAULT_NOISE_LEVEL,
        noise_variance: Optional[float] = None,
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a regression target with a non-linear dependence on the features.

        Args:
            n_samples: Number of rows.
            n_features: Number of columns, drawn from ``U(-2, 2)``.
            nonlinearity_type: One of ``"polynomial"`` (cubic terms plus a pairwise
                interaction), ``"sinusoidal"`` (sum of sines) or ``"friedman"``
                (the Friedman #1 benchmark, needs ``n_features >= 5``).
            noise_level: Standard deviation of the target noise. Ignored when
                ``noise_variance`` is given.
            noise_variance: Optional variance of the target noise.
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, y)``.

        Raises:
            ValueError: For an unknown ``nonlinearity_type`` or an incompatible
                ``n_features``.
        """
        n_samples = _check_int("n_samples", n_samples)
        n_features = _check_int("n_features", n_features)
        std = (
            float(np.sqrt(_check_float("noise_variance", noise_variance, 0.0, np.inf)))
            if noise_variance is not None
            else _check_float("noise_level", noise_level, 0.0, np.inf)
        )
        rng = self._rng(random_state, "nonlinear_regression")

        if nonlinearity_type == "friedman":
            if n_features < 5:
                raise ValueError("nonlinearity_type='friedman' requires n_features >= 5")
            X, y = make_friedman1(n_samples=n_samples, n_features=n_features, noise=std, random_state=rng)
            self._log_dataset("nonlinear_regression", X, y)
            return X, y

        X = rng.uniform(-2.0, 2.0, size=(n_samples, n_features))
        n_active = min(n_features, 5)
        w = rng.uniform(0.5, 1.5, size=(3, n_active))
        if nonlinearity_type == "polynomial":
            Z = X[:, :n_active]
            y = (w[0] * Z + w[1] * Z**2 * 0.5 + w[2] * Z**3 * 0.2).sum(axis=1)
            if n_active >= 2:
                y = y + Z[:, 0] * Z[:, 1]
        elif nonlinearity_type == "sinusoidal":
            Z = X[:, :n_active]
            phases = rng.uniform(0.0, np.pi, size=n_active)
            y = (w[0] * np.sin(w[1] * 2.0 * Z + phases)).sum(axis=1)
        else:
            raise ValueError(
                f"Unknown nonlinearity_type '{nonlinearity_type}'. Choose from 'polynomial', 'sinusoidal', 'friedman'."
            )
        y = y + rng.normal(0.0, std, size=n_samples)
        self._log_dataset("nonlinear_regression", X, y)
        return X, y

    def regression_with_outliers(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = 10,
        outlier_fraction: float = 0.1,
        outlier_strength: float = 3.0,
        noise_level: float = settings.DEFAULT_NOISE_LEVEL,
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a linear regression whose targets contain gross outliers.

        A fraction of the targets is displaced by ``outlier_strength`` times the
        standard deviation of the clean target, in a random direction.

        Args:
            n_samples: Number of rows.
            n_features: Number of columns.
            outlier_fraction: Share of contaminated targets in ``[0, 1)``.
            outlier_strength: Displacement size in units of the clean target std.
            noise_level: Standard deviation of the Gaussian noise on clean targets.
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, y, true_coef)``.
        """
        n_samples = _check_int("n_samples", n_samples)
        n_features = _check_int("n_features", n_features)
        outlier_fraction = _check_float("outlier_fraction", outlier_fraction, 0.0, 1.0, inclusive_high=False)
        outlier_strength = _check_float("outlier_strength", outlier_strength, 0.0, np.inf)
        rng = self._rng(random_state, "regression_with_outliers")
        X = rng.standard_normal((n_samples, n_features))
        true_coef = rng.uniform(-2.0, 2.0, size=n_features)
        y = X @ true_coef + rng.normal(0.0, noise_level, size=n_samples)
        n_outliers = round(outlier_fraction * n_samples)
        if n_outliers:
            idx = rng.choice(n_samples, size=n_outliers, replace=False)
            scale = float(np.std(y)) if n_samples > 1 else 1.0
            y[idx] += (
                rng.choice([-1.0, 1.0], size=n_outliers)
                * outlier_strength
                * scale
                * rng.uniform(1.0, 2.0, size=n_outliers)
            )
        self._log_dataset("regression_with_outliers", X, y)
        return X, y, true_coef

    # -------------------------------------------------------- classification
    @staticmethod
    def _resolve_classification_layout(
        n_features: int,
        n_informative: Optional[int],
        n_redundant: Optional[int],
        n_repeated: int,
        n_classes: int,
        n_clusters_per_class: Optional[int],
    ) -> Tuple[int, int, int]:
        """Pick a valid ``(n_informative, n_redundant, n_clusters_per_class)`` triple.

        Defaults follow scikit-learn where possible but scale with ``n_features``
        and guarantee ``n_classes * n_clusters_per_class <= 2**n_informative``.
        """
        if n_informative is None:
            n_informative = max(1, min(n_features - n_repeated, round(0.75 * n_features)))
        else:
            n_informative = _check_int("n_informative", n_informative)
        if n_redundant is None:
            n_redundant = max(0, min(2, n_features - n_informative - n_repeated))
        else:
            n_redundant = _check_int("n_redundant", n_redundant, minimum=0)
        if n_clusters_per_class is None:
            n_clusters_per_class = 2 if 2**n_informative >= 2 * n_classes else 1
        else:
            n_clusters_per_class = _check_int("n_clusters_per_class", n_clusters_per_class)
        if n_informative + n_redundant + n_repeated > n_features:
            raise ValueError(
                "n_informative + n_redundant + n_repeated must not exceed n_features "
                f"({n_informative} + {n_redundant} + {n_repeated} > {n_features})"
            )
        if n_classes * n_clusters_per_class > 2**n_informative:
            raise ValueError(
                f"n_classes * n_clusters_per_class ({n_classes * n_clusters_per_class}) must be <= "
                f"2**n_informative ({2**n_informative}); increase n_informative or reduce the number of clusters"
            )
        return n_informative, n_redundant, n_clusters_per_class

    def classification_dataset(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = settings.DEFAULT_N_FEATURES,
        n_informative: Optional[int] = None,
        n_redundant: Optional[int] = None,
        n_repeated: int = 0,
        n_classes: int = 2,
        n_clusters_per_class: Optional[int] = None,
        weights: Optional[Sequence[float]] = None,
        flip_y: float = 0.01,
        class_sep: float = 1.0,
        hypercube: bool = True,
        shift: float = 0.0,
        scale: float = 1.0,
        shuffle: bool = True,
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Seeded wrapper around :func:`sklearn.datasets.make_classification`.

        Unlike scikit-learn, ``n_informative``/``n_redundant`` default to values
        that scale with ``n_features`` (75 % informative, up to two redundant) and
        ``n_clusters_per_class`` is lowered to one automatically when the
        requested number of classes would otherwise be infeasible.

        Args:
            n_samples: Number of rows.
            n_features: Number of columns.
            n_informative: Informative features.
            n_redundant: Linear combinations of informative features.
            n_repeated: Duplicated features.
            n_classes: Number of classes.
            n_clusters_per_class: Gaussian clusters per class.
            weights: Class proportions (``None`` for balanced classes).
            flip_y: Fraction of randomly flipped labels.
            class_sep: Class separation multiplier.
            hypercube: Place clusters on hypercube vertices.
            shift: Feature shift.
            scale: Feature scale.
            shuffle: Shuffle samples and features.
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, y)``.
        """
        n_samples = _check_int("n_samples", n_samples)
        n_features = _check_int("n_features", n_features)
        n_classes = _check_int("n_classes", n_classes)
        n_repeated = _check_int("n_repeated", n_repeated, minimum=0)
        n_informative, n_redundant, n_clusters_per_class = self._resolve_classification_layout(
            n_features, n_informative, n_redundant, n_repeated, n_classes, n_clusters_per_class
        )
        rng = self._rng(random_state, "classification_dataset")
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=n_repeated,
            n_classes=n_classes,
            n_clusters_per_class=n_clusters_per_class,
            weights=None if weights is None else list(weights),
            flip_y=flip_y,
            class_sep=class_sep,
            hypercube=hypercube,
            shift=shift,
            scale=scale,
            shuffle=shuffle,
            random_state=rng,
        )
        self._log_dataset("classification_dataset", X, y)
        return X, y

    def generate_classification_data(
        self, *args: Any, noise: Optional[float] = None, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Alias of :meth:`classification_dataset`; ``noise`` maps to ``flip_y``."""
        if noise is not None:
            kwargs.setdefault("flip_y", noise)
        return self.classification_dataset(*args, **kwargs)

    def generate_basic_classification(self, *args: Any, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Alias of :meth:`classification_dataset` (documentation name)."""
        return self.classification_dataset(*args, **kwargs)

    def classification_balanced(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = settings.DEFAULT_N_FEATURES,
        n_classes: int = 2,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Balanced multi-class classification data (equal class proportions)."""
        kwargs.pop("weights", None)
        return self.classification_dataset(
            n_samples=n_samples, n_features=n_features, n_classes=n_classes, weights=None, **kwargs
        )

    def generate_multiclass_classification(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = settings.DEFAULT_N_FEATURES,
        n_classes: int = 3,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Multi-class variant of :meth:`classification_dataset` (three classes by default)."""
        return self.classification_dataset(
            n_samples=n_samples, n_features=n_features, n_classes=n_classes, **kwargs
        )

    def generate_imbalanced_classification(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = settings.DEFAULT_N_FEATURES,
        n_classes: int = 2,
        weights: Optional[Sequence[float]] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Imbalanced classification defined through explicit class ``weights``.

        Args:
            n_samples: Number of rows.
            n_features: Number of columns.
            n_classes: Number of classes.
            weights: Class proportions summing to one. Defaults to a geometric
                decay (``0.9/0.1`` for two classes).
            **kwargs: Forwarded to :meth:`classification_dataset`.

        Returns:
            Tuple ``(X, y)``.
        """
        n_classes = _check_int("n_classes", n_classes)
        if weights is None:
            raw = 0.3 ** np.arange(n_classes)[::-1] if n_classes > 2 else np.array([0.9, 0.1])
            weights = list(raw / raw.sum())
        if len(weights) != n_classes:
            raise ValueError(f"weights must have {n_classes} entries, got {len(weights)}")
        kwargs.setdefault("flip_y", 0.0)
        return self.classification_dataset(
            n_samples=n_samples, n_features=n_features, n_classes=n_classes, weights=weights, **kwargs
        )

    def imbalanced_classification_data(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        imbalance_ratio: float = 0.1,
        n_features: int = settings.DEFAULT_N_FEATURES,
        n_informative: Optional[int] = None,
        n_redundant: Optional[int] = None,
        class_sep: float = 1.0,
        flip_y: float = 0.0,
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Binary classification where the positive class is the minority.

        Args:
            n_samples: Number of rows.
            imbalance_ratio: Ratio ``n_minority / n_majority`` in ``(0, 1]``. The
                minority class is label ``1``.
            n_features: Number of columns.
            n_informative: Informative features (see :meth:`classification_dataset`).
            n_redundant: Redundant features (see :meth:`classification_dataset`).
            class_sep: Class separation multiplier.
            flip_y: Fraction of randomly flipped labels (zero keeps the ratio exact).
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, y)`` with ``y`` in ``{0, 1}``.
        """
        imbalance_ratio = _check_float("imbalance_ratio", imbalance_ratio, 0.0, 1.0, inclusive_low=False)
        minority_share = imbalance_ratio / (1.0 + imbalance_ratio)
        X, y = self.classification_dataset(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=2,
            weights=[1.0 - minority_share, minority_share],
            flip_y=flip_y,
            class_sep=class_sep,
            random_state=random_state
            if random_state is not None
            else self._rng(None, "imbalanced_classification_data"),
        )
        if np.sum(y == 1) == 0:
            self.logger.warning(
                "imbalance_ratio=%.4f with n_samples=%d produced no minority samples",
                imbalance_ratio,
                n_samples,
            )
        return X, y

    def imbalanced_classification(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = settings.DEFAULT_N_FEATURES,
        imbalance_ratio: float = 0.1,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Alias of :meth:`imbalanced_classification_data` with notebook argument order."""
        return self.imbalanced_classification_data(
            n_samples=n_samples, imbalance_ratio=imbalance_ratio, n_features=n_features, **kwargs
        )

    def classification_complexity_spectrum(
        self,
        complexity: str = "linear",
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = 2,
        noise: Optional[float] = None,
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Binary classification with a decision boundary of increasing complexity.

        * ``"linear"`` - two Gaussian clusters that are (nearly) linearly separable.
          ``noise`` is the fraction of flipped labels (default ``0.0``).
        * ``"medium"`` - interleaving half-moons. ``noise`` is the Gaussian jitter
          (default ``0.2``).
        * ``"high"`` - concentric circles. ``noise`` is the Gaussian jitter
          (default ``0.1``).

        For ``"medium"`` and ``"high"`` the shape lives in the first two features
        and any additional features are uninformative standard-normal noise.
        ``"simple"``, ``"nonlinear"`` and ``"complex"`` are accepted aliases.

        Args:
            complexity: One of :data:`COMPLEXITY_LEVELS`.
            n_samples: Number of rows.
            n_features: Number of columns (``>= 2`` for the non-linear levels).
            noise: Level-specific noise parameter (see above).
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, y)`` with ``y`` in ``{0, 1}``.

        Raises:
            ValueError: If ``complexity`` is unknown or ``n_features`` is too small.
        """
        level = _COMPLEXITY_ALIASES.get(str(complexity).lower(), str(complexity).lower())
        if level not in COMPLEXITY_LEVELS:
            raise ValueError(
                f"Unknown complexity level '{complexity}'. Choose from {list(COMPLEXITY_LEVELS)}."
            )
        n_samples = _check_int("n_samples", n_samples)
        n_features = _check_int("n_features", n_features)
        rng = self._rng(random_state, f"classification_complexity_spectrum:{level}")

        if level == "linear":
            flip_y = 0.0 if noise is None else _check_float("noise", noise, 0.0, 1.0)
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_features,
                n_redundant=0,
                n_repeated=0,
                n_classes=2,
                n_clusters_per_class=1,
                class_sep=2.0,
                flip_y=flip_y,
                random_state=rng,
            )
        else:
            if n_features < 2:
                raise ValueError(f"complexity='{level}' requires n_features >= 2, got {n_features}")
            if level == "medium":
                jitter = 0.2 if noise is None else _check_float("noise", noise, 0.0, np.inf)
                X2, y = make_moons(n_samples=n_samples, noise=jitter, random_state=rng)
            else:
                jitter = 0.1 if noise is None else _check_float("noise", noise, 0.0, np.inf)
                X2, y = make_circles(n_samples=n_samples, noise=jitter, factor=0.5, random_state=rng)
            if n_features > 2:
                X = np.hstack([X2, rng.standard_normal((n_samples, n_features - 2))])
            else:
                X = X2
        y = y.astype(int)
        self._log_dataset(f"classification_complexity_spectrum[{level}]", X, y)
        return X, y

    def classification_with_noise(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = settings.DEFAULT_N_FEATURES,
        n_informative: int = 10,
        noise_features: int = 5,
        flip_y: float = 0.05,
        n_classes: int = 2,
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Classification with explicit pure-noise features and label noise.

        Args:
            n_samples: Number of rows.
            n_features: Total number of columns.
            n_informative: Informative features.
            noise_features: Features carrying no signal. The remainder
                (``n_features - n_informative - noise_features``) is redundant.
            flip_y: Fraction of randomly flipped labels.
            n_classes: Number of classes.
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, y)``.
        """
        n_features = _check_int("n_features", n_features)
        n_informative = _check_int("n_informative", n_informative)
        noise_features = _check_int("noise_features", noise_features, minimum=0)
        n_redundant = n_features - n_informative - noise_features
        if n_redundant < 0:
            raise ValueError("n_informative + noise_features must not exceed n_features")
        return self.classification_dataset(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_classes=n_classes,
            flip_y=flip_y,
            random_state=random_state
            if random_state is not None
            else self._rng(None, "classification_with_noise"),
        )

    def high_dimensional_sparse_data(
        self,
        n_samples: int = 100,
        n_features: int = 1000,
        sparsity: float = 0.95,
        n_classes: int = 3,
        n_informative: Optional[int] = None,
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Non-negative, mostly-zero features resembling bag-of-words counts.

        A fraction ``sparsity`` of the entries is exactly zero. Each class owns a
        disjoint block of ``n_informative // n_classes`` features that are
        activated more often and with larger magnitude for its samples, so the
        data is separable by multinomial Naive Bayes or linear SVMs. Classes are
        balanced.

        Args:
            n_samples: Number of rows.
            n_features: Number of columns.
            sparsity: Fraction of zero entries in ``[0, 1)``.
            n_classes: Number of (balanced) classes.
            n_informative: Number of class-specific features. Defaults to
                ``max(n_classes, n_features // 20)``.
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, y)`` with ``X >= 0``.
        """
        n_samples = _check_int("n_samples", n_samples)
        n_features = _check_int("n_features", n_features)
        n_classes = _check_int("n_classes", n_classes)
        sparsity = _check_float("sparsity", sparsity, 0.0, 1.0, inclusive_high=False)
        if n_informative is None:
            n_informative = min(n_features, max(n_classes, n_features // 20))
        n_informative = _check_int("n_informative", n_informative)
        if n_informative > n_features:
            raise ValueError(f"n_informative ({n_informative}) cannot exceed n_features ({n_features})")
        if n_classes > n_samples:
            raise ValueError(f"n_classes ({n_classes}) cannot exceed n_samples ({n_samples})")
        rng = self._rng(random_state, "high_dimensional_sparse_data")

        y = rng.permutation(np.arange(n_samples) % n_classes)
        density = 1.0 - sparsity
        X = np.where(
            rng.uniform(size=(n_samples, n_features)) < density,
            rng.exponential(1.0, size=(n_samples, n_features)),
            0.0,
        )

        informative = rng.choice(n_features, size=n_informative, replace=False)
        blocks = np.array_split(informative, n_classes)
        boosted_density = min(1.0, 4.0 * density)
        for label, cols in enumerate(blocks):
            if cols.size == 0:
                continue
            rows = np.flatnonzero(y == label)
            active = rng.uniform(size=(rows.size, cols.size)) < boosted_density
            values = rng.uniform(1.0, 3.0, size=(rows.size, cols.size)) * active
            X[np.ix_(rows, cols)] = np.where(active, values, X[np.ix_(rows, cols)])
        self._log_dataset("high_dimensional_sparse_data", X, y)
        return X, y

    def sparse_classification(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = 500,
        n_informative: Optional[int] = None,
        sparsity: float = 0.95,
        n_classes: int = 2,
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Binary-by-default alias of :meth:`high_dimensional_sparse_data`."""
        return self.high_dimensional_sparse_data(
            n_samples=n_samples,
            n_features=n_features,
            sparsity=sparsity,
            n_classes=n_classes,
            n_informative=n_informative,
            random_state=random_state,
        )

    def multilabel_classification(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = settings.DEFAULT_N_FEATURES,
        n_classes: int = 5,
        n_labels_per_sample: int = 2,
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Seeded wrapper around :func:`sklearn.datasets.make_multilabel_classification`.

        Args:
            n_samples: Number of rows.
            n_features: Number of columns.
            n_classes: Number of labels.
            n_labels_per_sample: Average number of active labels per row.
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, Y)`` where ``Y`` is a binary indicator matrix of shape
            ``(n_samples, n_classes)``.
        """
        rng = self._rng(random_state, "multilabel_classification")
        X, Y = make_multilabel_classification(
            n_samples=_check_int("n_samples", n_samples),
            n_features=_check_int("n_features", n_features),
            n_classes=_check_int("n_classes", n_classes),
            n_labels=_check_int("n_labels_per_sample", n_labels_per_sample),
            allow_unlabeled=False,
            random_state=rng,
        )
        self._log_dataset("multilabel_classification", X, Y)
        return X, Y

    def concept_drift_classification(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = 10,
        n_drift_points: int = 2,
        drift_severity: float = 0.5,
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Streaming binary classification whose decision boundary drifts over time.

        The stream is split into ``n_drift_points + 1`` segments. Each segment
        uses a linear boundary ``w_k``; at a drift point the direction is rotated
        towards a random direction by ``drift_severity`` (``0`` = no drift,
        ``1`` = unrelated boundary).

        Args:
            n_samples: Number of rows, ordered in time.
            n_features: Number of columns.
            n_drift_points: Number of abrupt drifts.
            drift_severity: Drift magnitude in ``[0, 1]``.
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, y, drift_points)`` where ``drift_points`` holds the row
            indices at which a new concept starts.
        """
        n_samples = _check_int("n_samples", n_samples)
        n_features = _check_int("n_features", n_features)
        n_drift_points = _check_int("n_drift_points", n_drift_points, minimum=0)
        drift_severity = _check_float("drift_severity", drift_severity, 0.0, 1.0)
        rng = self._rng(random_state, "concept_drift_classification")

        X = rng.standard_normal((n_samples, n_features))
        boundaries = np.linspace(0, n_samples, n_drift_points + 2).round().astype(int)
        drift_points = boundaries[1:-1].copy()
        y = np.empty(n_samples, dtype=int)
        w = _unit_directions(rng, 1, n_features)[0]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if start > boundaries[0]:
                target = _unit_directions(rng, 1, n_features)[0]
                w = (1.0 - drift_severity) * w + drift_severity * target
                w /= max(np.linalg.norm(w), 1e-12)
            margin = X[start:end] @ w + rng.normal(0.0, 0.2, size=end - start)
            y[start:end] = (margin > 0).astype(int)
        self._log_dataset("concept_drift_classification", X, y)
        return X, y, drift_points

    def generate_moons_classification(
        self, n_samples: int = 400, noise: float = 0.1, random_state: RandomStateLike = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Two interleaving half-moons (see :func:`sklearn.datasets.make_moons`)."""
        rng = self._rng(random_state, "generate_moons_classification")
        X, y = make_moons(n_samples=_check_int("n_samples", n_samples), noise=noise, random_state=rng)
        return X, y.astype(int)

    def generate_circles_classification(
        self,
        n_samples: int = 400,
        factor: float = 0.5,
        noise: float = 0.1,
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Two concentric circles (see :func:`sklearn.datasets.make_circles`)."""
        rng = self._rng(random_state, "generate_circles_classification")
        X, y = make_circles(
            n_samples=_check_int("n_samples", n_samples), factor=factor, noise=noise, random_state=rng
        )
        return X, y.astype(int)

    # ------------------------------------------------------------ clustering
    def clustering_dataset(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = 2,
        n_clusters: int = 3,
        cluster_std: Union[float, Sequence[float]] = 1.0,
        center_box: Tuple[float, float] = (-10.0, 10.0),
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Isotropic Gaussian blobs with ground-truth labels.

        Args:
            n_samples: Number of rows.
            n_features: Number of columns.
            n_clusters: Number of blobs.
            cluster_std: Standard deviation per blob (scalar or one per blob).
            center_box: Bounding box for the blob centres.
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, labels)``.
        """
        rng = self._rng(random_state, "clustering_dataset")
        X, y = make_blobs(
            n_samples=_check_int("n_samples", n_samples),
            n_features=_check_int("n_features", n_features),
            centers=_check_int("n_clusters", n_clusters),
            cluster_std=cluster_std,
            center_box=center_box,
            random_state=rng,
        )
        self._log_dataset("clustering_dataset", X, y)
        return X, y

    def generate_clustering_data(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = 2,
        n_clusters: int = 3,
        n_centers: Optional[int] = None,
        cluster_std: Union[float, Sequence[float]] = 1.0,
        center_box: Tuple[float, float] = (-10.0, 10.0),
        shuffle: bool = True,
        random_state: RandomStateLike = None,
        return_centers: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Documentation-facing wrapper around :func:`sklearn.datasets.make_blobs`.

        ``n_centers`` overrides ``n_clusters`` when given. With
        ``return_centers=True`` the blob centres are returned as a third item.
        """
        rng = self._rng(random_state, "generate_clustering_data")
        result = make_blobs(
            n_samples=_check_int("n_samples", n_samples),
            n_features=_check_int("n_features", n_features),
            centers=_check_int("n_centers", n_clusters if n_centers is None else n_centers),
            cluster_std=cluster_std,
            center_box=center_box,
            shuffle=shuffle,
            random_state=rng,
            return_centers=return_centers,
        )
        self._log_dataset("generate_clustering_data", result[0], result[1])
        return result

    def make_blobs_advanced(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        centers: Union[int, np.ndarray] = 3,
        cluster_std: Union[float, Sequence[float]] = 1.0,
        n_features: int = 2,
        center_box: Tuple[float, float] = (-10.0, 10.0),
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Gaussian blobs with explicit ``centers`` (count or coordinate array)."""
        rng = self._rng(random_state, "make_blobs_advanced")
        X, y = make_blobs(
            n_samples=_check_int("n_samples", n_samples),
            n_features=_check_int("n_features", n_features),
            centers=centers,
            cluster_std=cluster_std,
            center_box=center_box,
            random_state=rng,
        )
        self._log_dataset("make_blobs_advanced", X, y)
        return X, y

    def clustering_gaussian_mixture(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_clusters: int = 3,
        n_features: int = 2,
        center_box: Tuple[float, float] = (-10.0, 10.0),
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Mixture of anisotropic Gaussians (each component has its own covariance).

        Args:
            n_samples: Number of rows, spread evenly across components.
            n_clusters: Number of mixture components.
            n_features: Number of columns.
            center_box: Bounding box for the component means.
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, labels)``.
        """
        n_samples = _check_int("n_samples", n_samples)
        n_clusters = _check_int("n_clusters", n_clusters)
        n_features = _check_int("n_features", n_features)
        rng = self._rng(random_state, "clustering_gaussian_mixture")
        counts = np.bincount(np.arange(n_samples) % n_clusters, minlength=n_clusters)
        means = rng.uniform(center_box[0], center_box[1], size=(n_clusters, n_features))
        chunks, labels = [], []
        for k in range(n_clusters):
            A = rng.standard_normal((n_features, n_features))
            cov = A @ A.T / n_features + 0.2 * np.eye(n_features)
            chunks.append(rng.multivariate_normal(means[k], cov, size=counts[k]))
            labels.append(np.full(counts[k], k))
        X, y = _shuffle_rows(rng, np.vstack(chunks), np.concatenate(labels))
        self._log_dataset("clustering_gaussian_mixture", X, y)
        return X, y

    def clustering_blobs_with_noise(
        self,
        n_clusters: int = 4,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        outlier_fraction: float = 0.1,
        cluster_std: Union[float, Sequence[float]] = 1.0,
        n_features: int = 2,
        center_box: Tuple[float, float] = (-10.0, 10.0),
        return_labels: bool = False,
        random_state: RandomStateLike = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Gaussian blobs contaminated with uniformly distributed outliers.

        Outliers are drawn uniformly from the bounding box of the blobs enlarged
        by 50 % on every side, so they sit both between and beyond the clusters.

        Args:
            n_clusters: Number of blobs.
            n_samples: Total number of rows (inliers plus outliers).
            outlier_fraction: Share of outlier rows in ``[0, 1)``.
            cluster_std: Standard deviation per blob.
            n_features: Number of columns.
            center_box: Bounding box for the blob centres.
            return_labels: Also return labels (``-1`` marks outliers).
            random_state: Per-call seed override.

        Returns:
            ``X`` of shape ``(n_samples, n_features)``, or ``(X, labels)`` when
            ``return_labels=True``.
        """
        n_clusters = _check_int("n_clusters", n_clusters)
        n_samples = _check_int("n_samples", n_samples)
        n_features = _check_int("n_features", n_features)
        outlier_fraction = _check_float("outlier_fraction", outlier_fraction, 0.0, 1.0, inclusive_high=False)
        n_outliers = round(outlier_fraction * n_samples)
        n_inliers = n_samples - n_outliers
        if n_inliers < n_clusters:
            raise ValueError(f"Too few inlier samples ({n_inliers}) for {n_clusters} clusters")
        rng = self._rng(random_state, "clustering_blobs_with_noise")
        X_in, y_in = make_blobs(
            n_samples=n_inliers,
            n_features=n_features,
            centers=n_clusters,
            cluster_std=cluster_std,
            center_box=center_box,
            random_state=rng,
        )
        lo, hi = X_in.min(axis=0), X_in.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        X_out = rng.uniform(lo - 0.5 * span, hi + 0.5 * span, size=(n_outliers, n_features))
        X, y = _shuffle_rows(rng, np.vstack([X_in, X_out]), np.concatenate([y_in, np.full(n_outliers, -1)]))
        self._log_dataset("clustering_blobs_with_noise", X, y)
        return (X, y) if return_labels else X

    def clustering_moons(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        noise: float = 0.1,
        return_labels: bool = False,
        random_state: RandomStateLike = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Two interleaving half-moons for density-based clustering demos.

        Args:
            n_samples: Number of rows.
            noise: Gaussian jitter standard deviation.
            return_labels: Also return the moon index of every row.
            random_state: Per-call seed override.

        Returns:
            ``X`` of shape ``(n_samples, 2)``, or ``(X, labels)``.
        """
        rng = self._rng(random_state, "clustering_moons")
        X, y = make_moons(n_samples=_check_int("n_samples", n_samples), noise=noise, random_state=rng)
        self._log_dataset("clustering_moons", X, y)
        return (X, y) if return_labels else X

    def make_moons_advanced(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        noise: float = 0.1,
        n_clusters: int = 2,
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Half-moon clusters; more than two moons are laid out as shifted pairs.

        Args:
            n_samples: Total number of rows.
            noise: Gaussian jitter standard deviation.
            n_clusters: Number of moons (``>= 2``).
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, labels)``.
        """
        n_samples = _check_int("n_samples", n_samples)
        n_clusters = _check_int("n_clusters", n_clusters, minimum=2)
        rng = self._rng(random_state, "make_moons_advanced")
        per_moon = np.bincount(np.arange(n_samples) % n_clusters, minlength=n_clusters)
        chunks, labels = [], []
        for p in range(int(np.ceil(n_clusters / 2))):
            upper, lower = per_moon[2 * p], per_moon[2 * p + 1] if 2 * p + 1 < n_clusters else 0
            Xp, yp = make_moons(n_samples=(int(upper), int(lower)), noise=noise, random_state=rng)
            Xp[:, 0] += 3.5 * p
            chunks.append(Xp)
            labels.append(yp + 2 * p)
        X, y = _shuffle_rows(rng, np.vstack(chunks), np.concatenate(labels))
        self._log_dataset("make_moons_advanced", X, y)
        return X, y

    def make_circles_advanced(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        noise: float = 0.1,
        factor: float = 0.5,
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Concentric circles (see :func:`sklearn.datasets.make_circles`)."""
        rng = self._rng(random_state, "make_circles_advanced")
        X, y = make_circles(
            n_samples=_check_int("n_samples", n_samples), noise=noise, factor=factor, random_state=rng
        )
        self._log_dataset("make_circles_advanced", X, y)
        return X, y

    def make_spirals(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        noise: float = 0.1,
        n_spirals: int = 2,
        n_turns: float = 1.5,
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interleaved Archimedean spirals, a classic non-convex clustering benchmark.

        Args:
            n_samples: Total number of rows.
            noise: Gaussian jitter standard deviation.
            n_spirals: Number of arms.
            n_turns: Number of revolutions per arm.
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, labels)`` with ``X`` of shape ``(n_samples, 2)``.
        """
        n_samples = _check_int("n_samples", n_samples)
        n_spirals = _check_int("n_spirals", n_spirals)
        rng = self._rng(random_state, "make_spirals")
        y = np.arange(n_samples) % n_spirals
        t = np.sqrt(rng.uniform(0.05, 1.0, size=n_samples)) * n_turns * 2.0 * np.pi
        angle = t + 2.0 * np.pi * y / n_spirals
        X = np.column_stack([t * np.cos(angle), t * np.sin(angle)]) / (2.0 * np.pi)
        X += rng.normal(0.0, noise, size=X.shape)
        X, y = _shuffle_rows(rng, X, y)
        self._log_dataset("make_spirals", X, y)
        return X, y

    def make_density_clusters(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_centers: int = 3,
        cluster_density_ratio: float = 0.3,
        noise_ratio: float = 0.1,
        n_features: int = 2,
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Clusters of strongly varying density plus uniform background noise.

        Cluster standard deviations are spaced geometrically between ``1`` and
        ``cluster_density_ratio`` (a ratio ``< 1`` makes the last cluster the
        tightest). Background points carry the label ``-1``.

        Args:
            n_samples: Total number of rows (clusters plus noise).
            n_centers: Number of clusters.
            cluster_density_ratio: Ratio of the smallest to the largest cluster std.
            noise_ratio: Share of uniformly distributed background rows in ``[0, 1)``.
            n_features: Number of columns.
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, labels)``.
        """
        n_samples = _check_int("n_samples", n_samples)
        n_centers = _check_int("n_centers", n_centers)
        n_features = _check_int("n_features", n_features)
        cluster_density_ratio = _check_float(
            "cluster_density_ratio", cluster_density_ratio, 0.0, np.inf, inclusive_low=False
        )
        noise_ratio = _check_float("noise_ratio", noise_ratio, 0.0, 1.0, inclusive_high=False)
        rng = self._rng(random_state, "make_density_clusters")
        n_noise = round(noise_ratio * n_samples)
        n_clustered = n_samples - n_noise
        if n_clustered < n_centers:
            raise ValueError(f"Too few clustered samples ({n_clustered}) for {n_centers} clusters")
        stds = np.geomspace(1.0, cluster_density_ratio, num=n_centers)
        X_c, y_c = make_blobs(
            n_samples=n_clustered,
            n_features=n_features,
            centers=n_centers,
            cluster_std=stds,
            center_box=(-8.0, 8.0),
            random_state=rng,
        )
        lo, hi = X_c.min(axis=0), X_c.max(axis=0)
        X_n = rng.uniform(lo, hi, size=(n_noise, n_features))
        X, y = _shuffle_rows(rng, np.vstack([X_c, X_n]), np.concatenate([y_c, np.full(n_noise, -1)]))
        self._log_dataset("make_density_clusters", X, y)
        return X, y

    def varied_density_clusters(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = 2,
        n_clusters: int = 3,
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Clusters with different sizes and spreads (no background noise).

        Cluster ``k`` receives a share of the rows proportional to ``k + 1`` and a
        standard deviation of ``0.5 * (k + 1)``, which makes the largest cluster
        also the most diffuse.

        Args:
            n_samples: Total number of rows.
            n_features: Number of columns.
            n_clusters: Number of clusters.
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, labels)``.
        """
        n_samples = _check_int("n_samples", n_samples)
        n_features = _check_int("n_features", n_features)
        n_clusters = _check_int("n_clusters", n_clusters)
        rng = self._rng(random_state, "varied_density_clusters")
        shares = np.arange(1, n_clusters + 1, dtype=float)
        counts = np.floor(shares / shares.sum() * n_samples).astype(int)
        counts[np.argmax(counts)] += n_samples - counts.sum()
        if np.any(counts < 1):
            raise ValueError(f"n_samples={n_samples} is too small for {n_clusters} clusters of varied size")
        centers = rng.uniform(-10.0, 10.0, size=(n_clusters, n_features))
        X, y = make_blobs(
            n_samples=counts.tolist(),
            n_features=n_features,
            centers=centers,
            cluster_std=0.5 * shares,
            random_state=rng,
        )
        self._log_dataset("varied_density_clusters", X, y)
        return X, y

    def complex_shapes(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        shape_type: str = "moons",
        noise: float = 0.1,
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dispatch to a two-dimensional non-convex shape generator.

        Args:
            n_samples: Number of rows.
            shape_type: ``"moons"``, ``"circles"``, ``"spirals"`` or ``"blobs"``.
            noise: Jitter standard deviation (cluster std for ``"blobs"``).
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, labels)``.

        Raises:
            ValueError: For an unknown ``shape_type``.
        """
        shape = str(shape_type).lower()
        if shape == "moons":
            return self.make_moons_advanced(n_samples=n_samples, noise=noise, random_state=random_state)
        if shape == "circles":
            return self.make_circles_advanced(n_samples=n_samples, noise=noise, random_state=random_state)
        if shape == "spirals":
            return self.make_spirals(n_samples=n_samples, noise=noise, random_state=random_state)
        if shape == "blobs":
            return self.clustering_dataset(
                n_samples=n_samples, n_features=2, cluster_std=max(noise, 1e-3), random_state=random_state
            )
        raise ValueError(
            f"Unknown shape_type '{shape_type}'. Choose from 'moons', 'circles', 'spirals', 'blobs'."
        )

    @staticmethod
    def _hierarchical_centers(
        rng: np.random.RandomState, n_levels: int, branching_factor: int, n_features: int, spread: float
    ) -> List[np.ndarray]:
        """Build a tree of cluster centres; level ``l`` holds ``branching_factor**l`` centres."""
        levels = [np.zeros((1, n_features))]
        for level in range(1, n_levels):
            parents = levels[-1]
            offsets = _unit_directions(rng, parents.shape[0] * branching_factor, n_features)
            scale = spread / (2.0 ** (level - 1))
            levels.append(np.repeat(parents, branching_factor, axis=0) + offsets * scale)
        return levels

    def hierarchical_clustering_data(
        self,
        n_samples: int = 100,
        n_levels: int = 3,
        n_features: int = 2,
        branching_factor: int = 2,
        spread: float = 10.0,
        return_labels: bool = False,
        random_state: RandomStateLike = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Multi-resolution data for agglomerative clustering demos.

        Level ``0`` is one coarse cluster around the origin; every subsequent
        level splits each cluster of the previous level into ``branching_factor``
        tighter sub-clusters located closer together. Every level contributes
        ``n_samples`` rows, so the result has ``n_samples * n_levels`` rows.

        Args:
            n_samples: Rows generated per level.
            n_levels: Number of hierarchy levels.
            n_features: Number of columns.
            branching_factor: Children per cluster.
            spread: Distance between level-1 centres and the origin.
            return_labels: Also return a unique cluster id per row.
            random_state: Per-call seed override.

        Returns:
            ``X`` of shape ``(n_samples * n_levels, n_features)`` or ``(X, labels)``.
        """
        n_samples = _check_int("n_samples", n_samples)
        n_levels = _check_int("n_levels", n_levels)
        n_features = _check_int("n_features", n_features)
        branching_factor = _check_int("branching_factor", branching_factor, minimum=2)
        spread = _check_float("spread", spread, 0.0, np.inf, inclusive_low=False)
        rng = self._rng(random_state, "hierarchical_clustering_data")
        levels = self._hierarchical_centers(rng, n_levels, branching_factor, n_features, spread)
        chunks, labels, offset = [], [], 0
        for level, centers in enumerate(levels):
            std = spread / (2.0**level) / 2.0
            assignment = rng.permutation(np.arange(n_samples) % centers.shape[0])
            chunks.append(centers[assignment] + rng.normal(0.0, std, size=(n_samples, n_features)))
            labels.append(assignment + offset)
            offset += centers.shape[0]
        X, y = np.vstack(chunks), np.concatenate(labels)
        self._log_dataset("hierarchical_clustering_data", X, y)
        return (X, y) if return_labels else X

    def make_hierarchical_clusters(
        self,
        n_samples: int = 500,
        n_levels: int = 3,
        branching_factor: int = 2,
        n_features: int = 2,
        spread: float = 10.0,
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Nested clusters with ``branching_factor ** n_levels`` leaf clusters.

        All rows are drawn around the leaf centres of a ``n_levels``-deep tree, so
        the data exhibits a genuine dendrogram structure.

        Args:
            n_samples: Total number of rows, spread evenly across leaves.
            n_levels: Depth of the tree below the root.
            branching_factor: Children per node.
            n_features: Number of columns.
            spread: Distance between first-level centres and the origin.
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, leaf_labels)``.
        """
        n_samples = _check_int("n_samples", n_samples)
        n_levels = _check_int("n_levels", n_levels)
        branching_factor = _check_int("branching_factor", branching_factor, minimum=2)
        n_features = _check_int("n_features", n_features)
        rng = self._rng(random_state, "make_hierarchical_clusters")
        leaves = self._hierarchical_centers(rng, n_levels + 1, branching_factor, n_features, spread)[-1]
        std = spread / (2.0**n_levels) / 3.0
        y = rng.permutation(np.arange(n_samples) % leaves.shape[0])
        X = leaves[y] + rng.normal(0.0, std, size=(n_samples, n_features))
        self._log_dataset("make_hierarchical_clusters", X, y)
        return X, y

    # ----------------------------------------------------------- time series
    def time_series_with_seasonality(
        self,
        n_samples: int = 365,
        seasonal_periods: Sequence[int] = (7, 30),
        trend_coef: float = 0.05,
        noise_level: float = settings.DEFAULT_NOISE_LEVEL,
        seasonal_amplitudes: Optional[Sequence[float]] = None,
        n_lags: int = 7,
        rolling_windows: Sequence[int] = (7, 30),
        start_date: str = "2020-01-01",
        freq: str = "D",
        random_state: RandomStateLike = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Daily series with linear trend, additive seasonality and engineered features.

        The target is ``trend_coef * t + sum_i a_i sin(2 pi t / p_i + phi_i) + eps``.
        The feature frame contains calendar columns, ``lag_k`` columns and
        ``rolling_mean_w`` / ``rolling_std_w`` columns computed on past values
        only (no target leakage). Rows without enough history are padded with the
        first observation (lags) or zero (rolling std) so the frame has no
        missing values.

        Args:
            n_samples: Number of time steps.
            seasonal_periods: Period (in steps) of every seasonal component.
            trend_coef: Linear trend slope per step.
            noise_level: Standard deviation of the Gaussian noise.
            seasonal_amplitudes: Amplitude per period (defaults to ``1.0`` each).
            n_lags: Number of lag features.
            rolling_windows: Window sizes for rolling statistics.
            start_date: Timestamp of the first row.
            freq: Pandas frequency string.
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, y)`` with ``X`` a :class:`pandas.DataFrame` containing the
            columns ``date``, ``time_index``, ``day_of_week``, ``day_of_month``,
            ``month``, ``day_of_year``, ``lag_*`` and ``rolling_*``.
        """
        n_samples = _check_int("n_samples", n_samples)
        n_lags = _check_int("n_lags", n_lags, minimum=0)
        noise_level = _check_float("noise_level", noise_level, 0.0, np.inf)
        periods = [_check_int("seasonal_period", p) for p in seasonal_periods]
        amplitudes = (
            [1.0] * len(periods) if seasonal_amplitudes is None else [float(a) for a in seasonal_amplitudes]
        )
        if len(amplitudes) != len(periods):
            raise ValueError("seasonal_amplitudes must have one entry per seasonal period")
        rng = self._rng(random_state, "time_series_with_seasonality")

        t = np.arange(n_samples, dtype=float)
        y = trend_coef * t
        for period, amplitude in zip(periods, amplitudes):
            y = y + amplitude * np.sin(2.0 * np.pi * t / period + rng.uniform(0.0, 2.0 * np.pi))
        y = y + rng.normal(0.0, noise_level, size=n_samples)

        dates = pd.date_range(start=start_date, periods=n_samples, freq=freq)
        X = pd.DataFrame(
            {
                "date": dates,
                "time_index": t.astype(int),
                "day_of_week": dates.dayofweek.to_numpy(),
                "day_of_month": dates.day.to_numpy(),
                "month": dates.month.to_numpy(),
                "day_of_year": dates.dayofyear.to_numpy(),
            }
        )
        series = pd.Series(y)
        for k in range(1, n_lags + 1):
            X[f"lag_{k}"] = series.shift(k).fillna(y[0]).to_numpy()
        past = series.shift(1)
        for window in rolling_windows:
            window = _check_int("rolling_window", window)
            X[f"rolling_mean_{window}"] = past.rolling(window, min_periods=1).mean().fillna(y[0]).to_numpy()
            X[f"rolling_std_{window}"] = past.rolling(window, min_periods=2).std().fillna(0.0).to_numpy()
        self._log_dataset("time_series_with_seasonality", X, y)
        return X, y

    def time_series_features(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = 12,
        trend: bool = True,
        seasonality: bool = True,
        noise_level: float = settings.DEFAULT_NOISE_LEVEL,
        seasonal_period: int = 12,
        random_state: RandomStateLike = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Autoregressive feature frame for tabular forecasting.

        Args:
            n_samples: Number of time steps.
            n_features: Number of lag columns (``lag_1`` ... ``lag_n``).
            trend: Include a linear trend.
            seasonality: Include a sinusoidal seasonal component.
            noise_level: Standard deviation of the Gaussian noise.
            seasonal_period: Period of the seasonal component in steps.
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, y)`` where ``X`` is a :class:`pandas.DataFrame` of lagged
            targets (padded with the first observation) and ``y`` the series.
        """
        n_samples = _check_int("n_samples", n_samples)
        n_features = _check_int("n_features", n_features)
        rng = self._rng(random_state, "time_series_features")
        t = np.arange(n_samples, dtype=float)
        y = np.zeros(n_samples)
        if trend:
            y += 0.02 * t
        if seasonality:
            y += np.sin(2.0 * np.pi * t / seasonal_period)
        y += rng.normal(0.0, noise_level, size=n_samples)
        series = pd.Series(y)
        X = pd.DataFrame(
            {f"lag_{k}": series.shift(k).fillna(y[0]).to_numpy() for k in range(1, n_features + 1)}
        )
        self._log_dataset("time_series_features", X, y)
        return X, y

    def time_series_classification(
        self,
        n_samples: int = 400,
        n_timesteps: int = 50,
        n_features: int = 3,
        n_classes: int = 4,
        pattern_type: str = "seasonal",
        noise_level: float = 0.2,
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Labelled multivariate sequences with class-specific temporal patterns.

        Args:
            n_samples: Number of sequences.
            n_timesteps: Length of every sequence.
            n_features: Channels per time step.
            n_classes: Number of (balanced) classes.
            pattern_type: ``"seasonal"`` (class-specific frequency), ``"trend"``
                (class-specific slope) or ``"mixed"`` (both).
            noise_level: Standard deviation of the Gaussian noise.
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, y)`` with ``X`` of shape ``(n_samples, n_timesteps,
            n_features)``; reshape to ``(n_samples, -1)`` for tabular estimators.

        Raises:
            ValueError: For an unknown ``pattern_type``.
        """
        n_samples = _check_int("n_samples", n_samples)
        n_timesteps = _check_int("n_timesteps", n_timesteps)
        n_features = _check_int("n_features", n_features)
        n_classes = _check_int("n_classes", n_classes)
        pattern = str(pattern_type).lower()
        if pattern not in {"seasonal", "trend", "mixed"}:
            raise ValueError(
                f"Unknown pattern_type '{pattern_type}'. Choose from 'seasonal', 'trend', 'mixed'."
            )
        rng = self._rng(random_state, "time_series_classification")
        y = rng.permutation(np.arange(n_samples) % n_classes)
        t = np.linspace(0.0, 1.0, n_timesteps)
        X = rng.normal(0.0, noise_level, size=(n_samples, n_timesteps, n_features))
        channel_phase = rng.uniform(0.0, np.pi, size=n_features)
        for label in range(n_classes):
            rows = y == label
            signal = np.zeros((n_timesteps, n_features))
            if pattern in {"seasonal", "mixed"}:
                freq = 1.0 + label
                signal += np.sin(2.0 * np.pi * freq * t)[:, None] * np.cos(channel_phase)[None, :]
            if pattern in {"trend", "mixed"}:
                slope = label - (n_classes - 1) / 2.0
                signal += slope * t[:, None] * np.ones((1, n_features))
            X[rows] += signal[None, :, :]
        self._log_dataset("time_series_classification", X, y)
        return X, y

    def generate_time_series_data(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = 5,
        trend: str = "linear",
        seasonality: Optional[int] = 12,
        noise_level: float = settings.DEFAULT_NOISE_LEVEL,
        anomaly_rate: float = 0.0,
        random_state: RandomStateLike = None,
    ) -> pd.DataFrame:
        """Multivariate time-series frame with a target and optional anomalies.

        Args:
            n_samples: Number of time steps.
            n_features: Number of exogenous ``feature_i`` columns.
            trend: ``"linear"``, ``"quadratic"`` or ``"none"``.
            seasonality: Seasonal period in steps (``None`` disables it).
            noise_level: Standard deviation of the Gaussian noise.
            anomaly_rate: Share of rows whose target is displaced by a large shock.
            random_state: Per-call seed override.

        Returns:
            :class:`pandas.DataFrame` with ``timestamp``, ``feature_*``, ``target``
            and ``is_anomaly`` columns.
        """
        n_samples = _check_int("n_samples", n_samples)
        n_features = _check_int("n_features", n_features, minimum=0)
        anomaly_rate = _check_float("anomaly_rate", anomaly_rate, 0.0, 1.0, inclusive_high=False)
        trend_kind = str(trend).lower()
        if trend_kind not in {"linear", "quadratic", "none"}:
            raise ValueError(f"Unknown trend '{trend}'. Choose from 'linear', 'quadratic', 'none'.")
        rng = self._rng(random_state, "generate_time_series_data")
        t = np.arange(n_samples, dtype=float)
        base = np.zeros(n_samples)
        if trend_kind == "linear":
            base += 0.02 * t
        elif trend_kind == "quadratic":
            base += 0.02 * t + 1e-4 * t**2
        if seasonality:
            base += np.sin(2.0 * np.pi * t / _check_int("seasonality", seasonality))
        features = (
            np.column_stack([np.cumsum(rng.normal(0.0, 0.1, size=n_samples)) for _ in range(n_features)])
            if n_features
            else np.empty((n_samples, 0))
        )
        weights = rng.uniform(-1.0, 1.0, size=n_features)
        target = base + features @ weights + rng.normal(0.0, noise_level, size=n_samples)
        is_anomaly = np.zeros(n_samples, dtype=bool)
        n_anomalies = round(anomaly_rate * n_samples)
        if n_anomalies:
            idx = rng.choice(n_samples, size=n_anomalies, replace=False)
            scale = max(float(np.std(target)), 1e-6)
            target[idx] += (
                rng.choice([-1.0, 1.0], size=n_anomalies) * rng.uniform(3.0, 6.0, size=n_anomalies) * scale
            )
            is_anomaly[idx] = True
        frame = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(n_features)])
        frame.insert(0, "timestamp", pd.date_range("2020-01-01", periods=n_samples, freq="D"))
        frame["target"] = target
        frame["is_anomaly"] = is_anomaly
        self._log_dataset("generate_time_series_data", frame)
        return frame

    # ------------------------------------------------------ special purpose
    def mixed_data_types(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_numerical: int = 5,
        n_categorical: int = 3,
        n_ordinal: int = 2,
        n_binary: int = 0,
        n_classes: int = 2,
        n_categories: int = 4,
        n_ordinal_levels: int = 5,
        missing_rate: float = 0.0,
        outlier_rate: float = 0.0,
        return_feature_types: bool = False,
        random_state: RandomStateLike = None,
    ) -> Union[Tuple[pd.DataFrame, np.ndarray], Tuple[pd.DataFrame, np.ndarray, List[str]]]:
        """Heterogeneous tabular data for preprocessing-pipeline demos.

        Returns ``(X, y)`` by default; pass ``return_feature_types=True`` to also
        receive the ``{column: kind}`` mapping as a third element.

        Columns are named ``num_i`` (float), ``cat_i`` (unordered
        :class:`pandas.CategoricalDtype`), ``ord_i`` (ordered categorical) and
        ``bin_i`` (``int`` in ``{0, 1}``). The target depends on every column
        through a random linear score with per-category effects, so encoders
        matter for downstream accuracy.

        Args:
            n_samples: Number of rows.
            n_numerical: Number of numeric columns.
            n_categorical: Number of nominal columns.
            n_ordinal: Number of ordered categorical columns.
            n_binary: Number of binary integer columns.
            n_classes: Number of target classes.
            n_categories: Levels per nominal column.
            n_ordinal_levels: Levels per ordinal column.
            missing_rate: Share of cells set to missing in numeric and nominal columns.
            outlier_rate: Share of numeric cells replaced by extreme values.
            return_feature_types: Also return the column type per feature
                (``"numerical"``, ``"categorical"``, ``"ordinal"``, ``"binary"``).
            random_state: Per-call seed override.

        Returns:
            ``(X, y)`` or ``(X, y, feature_types)``.
        """
        n_samples = _check_int("n_samples", n_samples)
        counts = {
            "numerical": _check_int("n_numerical", n_numerical, minimum=0),
            "categorical": _check_int("n_categorical", n_categorical, minimum=0),
            "ordinal": _check_int("n_ordinal", n_ordinal, minimum=0),
            "binary": _check_int("n_binary", n_binary, minimum=0),
        }
        if sum(counts.values()) == 0:
            raise ValueError("At least one feature column is required")
        n_classes = _check_int("n_classes", n_classes)
        n_categories = _check_int("n_categories", n_categories, minimum=2)
        n_ordinal_levels = _check_int("n_ordinal_levels", n_ordinal_levels, minimum=2)
        missing_rate = _check_float("missing_rate", missing_rate, 0.0, 1.0, inclusive_high=False)
        outlier_rate = _check_float("outlier_rate", outlier_rate, 0.0, 1.0, inclusive_high=False)
        rng = self._rng(random_state, "mixed_data_types")

        columns: Dict[str, Any] = {}
        feature_types: List[str] = []
        score = np.zeros((n_samples, n_classes))

        for i in range(counts["numerical"]):
            values = rng.normal(rng.uniform(-2.0, 2.0), rng.uniform(0.5, 2.0), size=n_samples)
            columns[f"num_{i}"] = values
            feature_types.append("numerical")
            z = (values - values.mean()) / (values.std() + 1e-12)
            score += z[:, None] * rng.normal(0.0, 1.0, size=(1, n_classes))

        category_labels = [chr(ord("A") + k) for k in range(n_categories)]
        for i in range(counts["categorical"]):
            codes = rng.randint(n_categories, size=n_samples)
            columns[f"cat_{i}"] = pd.Categorical.from_codes(codes, categories=category_labels)
            feature_types.append("categorical")
            score += rng.normal(0.0, 1.0, size=(n_categories, n_classes))[codes]

        level_names = list(
            _ORDINAL_LEVEL_NAMES.get(n_ordinal_levels, tuple(f"level_{k}" for k in range(n_ordinal_levels)))
        )
        for i in range(counts["ordinal"]):
            codes = rng.randint(n_ordinal_levels, size=n_samples)
            columns[f"ord_{i}"] = pd.Categorical.from_codes(codes, categories=level_names, ordered=True)
            feature_types.append("ordinal")
            score += codes[:, None] * rng.normal(0.0, 0.5, size=(1, n_classes))

        for i in range(counts["binary"]):
            bits = rng.randint(2, size=n_samples)
            columns[f"bin_{i}"] = bits
            feature_types.append("binary")
            score += bits[:, None] * rng.normal(0.0, 1.0, size=(1, n_classes))

        score += rng.normal(0.0, 0.5, size=score.shape)
        y = np.argmax(score, axis=1) if n_classes > 1 else np.zeros(n_samples, dtype=int)
        X = pd.DataFrame(columns)

        if outlier_rate > 0:
            for i in range(counts["numerical"]):
                col = f"num_{i}"
                mask = rng.uniform(size=n_samples) < outlier_rate
                spread = X[col].std() if n_samples > 1 else 1.0
                X.loc[mask, col] = (
                    X[col].mean()
                    + rng.choice([-1.0, 1.0], size=mask.sum())
                    * rng.uniform(5.0, 10.0, size=mask.sum())
                    * spread
                )
        if missing_rate > 0:
            for col in X.columns:
                if col.startswith(("num_", "cat_")):
                    mask = rng.uniform(size=n_samples) < missing_rate
                    X.loc[mask, col] = np.nan
        self._log_dataset("mixed_data_types", X, y)
        return (X, y, feature_types) if return_feature_types else (X, y)

    def feature_selection_showcase_data(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = 100,
        n_informative: int = 15,
        n_redundant: int = 15,
        n_noise_features: Optional[int] = None,
        n_classes: int = 2,
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Classification data with a known feature-importance ground truth.

        Columns are ordered informative, redundant, noise (no column shuffling),
        and ``feature_importance`` encodes that structure: informative features
        score in ``[0.5, 1.0]``, redundant ones in ``[0.1, 0.5)`` and noise
        features exactly ``0``.

        Args:
            n_samples: Number of rows.
            n_features: Total number of columns.
            n_informative: Informative features.
            n_redundant: Linear combinations of informative features.
            n_noise_features: Pure-noise features; defaults to the remainder.
            n_classes: Number of classes.
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, y, feature_importance)``.

        Raises:
            ValueError: If the feature counts do not add up to ``n_features``.
        """
        n_samples = _check_int("n_samples", n_samples)
        n_features = _check_int("n_features", n_features)
        n_informative = _check_int("n_informative", n_informative)
        n_redundant = _check_int("n_redundant", n_redundant, minimum=0)
        if n_noise_features is None:
            n_noise_features = n_features - n_informative - n_redundant
            if n_noise_features < 0:
                raise ValueError("n_informative + n_redundant must not exceed n_features")
        else:
            n_noise_features = _check_int("n_noise_features", n_noise_features, minimum=0)
            if n_informative + n_redundant + n_noise_features != n_features:
                raise ValueError("n_informative + n_redundant + n_noise_features must equal n_features")
        rng = self._rng(random_state, "feature_selection_showcase_data")
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=0,
            n_classes=n_classes,
            n_clusters_per_class=1,
            shuffle=False,
            random_state=rng,
        )
        X, y = _shuffle_rows(rng, X, y)
        importance = np.zeros(n_features)
        importance[:n_informative] = np.sort(rng.uniform(0.5, 1.0, size=n_informative))[::-1]
        importance[n_informative : n_informative + n_redundant] = np.sort(
            rng.uniform(0.1, 0.45, size=n_redundant)
        )[::-1]
        self._log_dataset("feature_selection_showcase_data", X, y)
        return X, y, importance

    def anomaly_detection_data(
        self,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        contamination: float = 0.1,
        n_features: int = 3,
        n_normal_clusters: int = 1,
        cluster_std: float = 1.0,
        random_state: RandomStateLike = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inlier clusters plus uniformly scattered anomalies, shuffled together.

        Args:
            n_samples: Total number of rows.
            contamination: Share of anomalies in ``(0, 0.5]`` (scikit-learn convention).
            n_features: Number of columns.
            n_normal_clusters: Number of inlier Gaussian clusters.
            cluster_std: Standard deviation of the inlier clusters.
            random_state: Per-call seed override.

        Returns:
            Tuple ``(X, y)`` with ``y = 1`` for inliers and ``y = -1`` for
            anomalies, matching the scikit-learn outlier-detection convention.

        Raises:
            ValueError: If ``contamination`` is outside ``(0, 0.5]``.
        """
        n_samples = _check_int("n_samples", n_samples)
        n_features = _check_int("n_features", n_features)
        n_normal_clusters = _check_int("n_normal_clusters", n_normal_clusters)
        contamination = _check_float("contamination", contamination, 0.0, 0.5, inclusive_low=False)
        n_anomalies = round(contamination * n_samples)
        n_inliers = n_samples - n_anomalies
        if n_inliers < n_normal_clusters:
            raise ValueError(f"Too few inlier samples ({n_inliers}) for {n_normal_clusters} clusters")
        rng = self._rng(random_state, "anomaly_detection_data")
        X_in, _ = make_blobs(
            n_samples=n_inliers,
            n_features=n_features,
            centers=n_normal_clusters,
            cluster_std=cluster_std,
            center_box=(-5.0, 5.0),
            random_state=rng,
        )
        lo, hi = X_in.min(axis=0), X_in.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        X_out = rng.uniform(lo - span, hi + span, size=(n_anomalies, n_features))
        X, y = _shuffle_rows(
            rng,
            np.vstack([X_in, X_out]),
            np.concatenate([np.ones(n_inliers, dtype=int), -np.ones(n_anomalies, dtype=int)]),
        )
        self._log_dataset("anomaly_detection_data", X, y)
        return X, y

    # ------------------------------------------------------------------ suite
    def generate_dataset_suite(
        self, n_samples: int = 500, random_state: RandomStateLike = None
    ) -> Dict[str, Tuple[Any, ...]]:
        """Generate one representative dataset per modelling challenge.

        Args:
            n_samples: Base number of rows used for most datasets.
            random_state: Per-call seed override applied to every dataset.

        Returns:
            Mapping from dataset name to the tuple returned by the corresponding
            method (``(X, y)``, ``(X, y, extra)`` or ``(X,)`` for unlabeled data).
        """
        n_samples = _check_int("n_samples", n_samples)
        rs = random_state
        suite: Dict[str, Tuple[Any, ...]] = {
            "linear_regression": self.linear_regression_data(n_samples=n_samples, random_state=rs),
            "regression_collinear": self.regression_with_collinearity(n_samples=n_samples, random_state=rs),
            "classification_linear": self.classification_complexity_spectrum(
                "linear", n_samples=n_samples, random_state=rs
            ),
            "classification_medium": self.classification_complexity_spectrum(
                "medium", n_samples=n_samples, random_state=rs
            ),
            "classification_complex": self.classification_complexity_spectrum(
                "high", n_samples=n_samples, random_state=rs
            ),
            "high_dimensional_sparse": self.high_dimensional_sparse_data(
                n_samples=min(n_samples, 200), n_features=500, random_state=rs
            ),
            "imbalanced_classification": self.imbalanced_classification_data(
                n_samples=n_samples, random_state=rs
            ),
            "mixed_data_types": self.mixed_data_types(n_samples=n_samples, random_state=rs),
            "clustering_blobs": (self.clustering_blobs_with_noise(n_samples=n_samples, random_state=rs),),
            "clustering_moons": (self.clustering_moons(n_samples=n_samples, random_state=rs),),
            "hierarchical_clusters": (
                self.hierarchical_clustering_data(n_samples=max(n_samples // 3, 1), random_state=rs),
            ),
            "feature_selection": self.feature_selection_showcase_data(n_samples=n_samples, random_state=rs),
            "anomaly_detection": self.anomaly_detection_data(n_samples=n_samples, random_state=rs),
            "time_series": self.time_series_with_seasonality(n_samples=max(n_samples, 60), random_state=rs),
        }
        self.logger.info("Generated dataset suite with %d datasets", len(suite))
        return suite


class ClassificationDataGenerator(SyntheticDataGenerator):
    """Named view of :class:`SyntheticDataGenerator` focused on classification.

    Provides ``generate_basic_classification``, ``generate_multiclass_classification``,
    ``generate_imbalanced_classification``, ``generate_moons_classification`` and
    ``generate_circles_classification`` alongside every other generator method.
    """


class RegressionDataGenerator(SyntheticDataGenerator):
    """Named view of :class:`SyntheticDataGenerator` focused on regression.

    Provides ``linear_regression_data``, ``regression_with_collinearity``,
    ``nonlinear_regression``, ``high_dimensional_regression`` and
    ``regression_with_outliers`` alongside every other generator method.
    """


class ClusteringDataGenerator(SyntheticDataGenerator):
    """Named view of :class:`SyntheticDataGenerator` focused on clustering.

    Provides ``clustering_dataset``, ``clustering_blobs_with_noise``,
    ``clustering_moons``, ``make_spirals``, ``make_density_clusters`` and
    ``hierarchical_clustering_data`` alongside every other generator method.
    """


DataGenerator = SyntheticDataGenerator
"""Backwards-compatible alias used throughout the documentation."""


__all__ = [
    "COMPLEXITY_LEVELS",
    "ClassificationDataGenerator",
    "ClusteringDataGenerator",
    "DataGenerator",
    "RegressionDataGenerator",
    "SyntheticDataGenerator",
]
