"""Global configuration for sklearn-mastery.

Settings are resolved from (highest precedence first): environment variables
prefixed with ``SKLEARN_MASTERY_``, a ``.env`` file in the working directory,
and the defaults below. ``PROJECT_ROOT`` defaults to the current working
directory so that an installed package never writes into ``site-packages``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_root() -> Path:
    return Path(os.environ.get("SKLEARN_MASTERY_ROOT", Path.cwd())).resolve()


class Settings(BaseSettings):
    """Project-wide settings with lazy directory creation."""

    model_config = SettingsConfigDict(
        env_prefix="SKLEARN_MASTERY_",
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    # ------------------------------------------------------------------ paths
    PROJECT_ROOT: Path = Field(default_factory=_default_root)
    RESULTS_DIRNAME: str = "results"

    # --------------------------------------------------------- reproducibility
    RANDOM_SEED: int = 42
    NUMPY_SEED: int = 42

    # ------------------------------------------------------- model defaults
    DEFAULT_TEST_SIZE: float = 0.2
    DEFAULT_CV_FOLDS: int = 5
    DEFAULT_N_JOBS: int = -1

    # ----------------------------------------------------- data generation
    DEFAULT_N_SAMPLES: int = 1000
    DEFAULT_N_FEATURES: int = 20
    DEFAULT_NOISE_LEVEL: float = 0.1

    # -------------------------------------------------------- visualization
    FIGURE_SIZE: Tuple[int, int] = (12, 8)
    DPI: int = 300
    STYLE: str = "seaborn-v0_8"
    COLOR_PALETTE: str = "husl"
    SAVE_FIGURES_BY_DEFAULT: bool = True
    FIGURE_FORMATS: List[str] = ["png", "pdf"]
    ADD_TIMESTAMP_TO_FIGURES: bool = True
    AUTO_CREATE_SUBDIRS: bool = True

    # --------------------------------------------------------------- mlflow
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "sklearn-mastery"

    # ----------------------------------------------------------- evaluation
    CLASSIFICATION_METRICS: List[str] = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc_ovr",
    ]
    REGRESSION_METRICS: List[str] = ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]

    # ------------------------------------------------------------- tuning
    MAX_ITER_OPTUNA: int = 100
    N_TRIALS_OPTUNA: int = 50

    # ---------------------------------------------------- feature selection
    MAX_FEATURES_SELECT: int = 50
    FEATURE_SELECTION_METHODS: List[str] = [
        "univariate",
        "rfe",
        "from_model",
        "variance_threshold",
    ]

    ENABLE_PIPELINE_CACHING: bool = True

    FIGURE_SUBDIRS: List[str] = [
        "data_generation",
        "preprocessing",
        "classification",
        "regression",
        "clustering",
        "model_comparison",
        "hyperparameter_optimization",
        "interpretability",
    ]

    @field_validator("PROJECT_ROOT", mode="before")
    @classmethod
    def _expand_root(cls, value: object) -> Path:
        return Path(str(value)).expanduser().resolve()

    # ------------------------------------------------------ derived paths
    @property
    def RESULTS_DIR(self) -> Path:  # noqa: N802 - keep legacy upper-case API
        return self.PROJECT_ROOT / self.RESULTS_DIRNAME

    @property
    def DATA_DIR(self) -> Path:  # noqa: N802
        return self.PROJECT_ROOT / "data"

    @property
    def MODELS_DIR(self) -> Path:  # noqa: N802
        return self.RESULTS_DIR / "models"

    @property
    def FIGURES_DIR(self) -> Path:  # noqa: N802
        return self.RESULTS_DIR / "figures"

    @property
    def REPORTS_DIR(self) -> Path:  # noqa: N802
        return self.RESULTS_DIR / "reports"

    @property
    def CACHE_DIR(self) -> Path:  # noqa: N802
        return self.RESULTS_DIR / "cache"

    @property
    def LOGS_DIR(self) -> Path:  # noqa: N802
        return self.PROJECT_ROOT / "logs"

    def ensure_directories(self) -> Dict[str, Path]:
        """Create all output directories and return them keyed by name."""
        dirs = {
            "data": self.DATA_DIR,
            "models": self.MODELS_DIR,
            "figures": self.FIGURES_DIR,
            "reports": self.REPORTS_DIR,
            "cache": self.CACHE_DIR,
            "logs": self.LOGS_DIR,
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        if self.AUTO_CREATE_SUBDIRS:
            for sub in self.FIGURE_SUBDIRS:
                (self.FIGURES_DIR / sub).mkdir(parents=True, exist_ok=True)
        return dirs


class ModelDefaults:
    """Default hyperparameter search spaces used by the pipeline factory."""

    CLASSIFICATION_MODELS: Dict[str, Dict[str, list]] = {
        "logistic_regression": {
            "C": [0.1, 1.0, 10.0],
            "solver": ["lbfgs", "liblinear"],
            "max_iter": [1000],
        },
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
        },
        "svm": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
        "gradient_boosting": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        },
        "mlp": {
            "hidden_layer_sizes": [(100,), (100, 50), (50, 50, 50)],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate": ["constant", "adaptive"],
        },
    }

    REGRESSION_MODELS: Dict[str, Dict[str, list]] = {
        "linear_regression": {},
        "ridge": {"alpha": [0.1, 1.0, 10.0, 100.0]},
        "lasso": {"alpha": [0.1, 1.0, 10.0, 100.0]},
        "elastic_net": {"alpha": [0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5, 0.7, 0.9]},
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
        },
        "svr": {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
    }

    CLUSTERING_MODELS: Dict[str, Dict[str, list]] = {
        "kmeans": {
            "n_clusters": list(range(2, 11)),
            "init": ["k-means++", "random"],
            "algorithm": ["lloyd", "elkan"],
        },
        "dbscan": {"eps": [0.3, 0.5, 0.7, 1.0], "min_samples": [3, 5, 10, 15]},
        "gaussian_mixture": {
            "n_components": list(range(2, 11)),
            "covariance_type": ["full", "tied", "diag", "spherical"],
        },
    }


settings = Settings()

__all__ = ["ModelDefaults", "Settings", "settings"]
