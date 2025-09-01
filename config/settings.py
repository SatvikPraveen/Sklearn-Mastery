"""Configuration settings for sklearn-mastery project."""

from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseSettings, Field
import os


class Settings(BaseSettings):
    """Global settings for the project."""
    
    # Project paths
    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    MODELS_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "results" / "models")
    FIGURES_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "results" / "figures")
    REPORTS_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "results" / "reports")
    
    # Additional paths for better organization
    CACHE_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "results" / "cache")
    LOGS_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    
    # Visualization settings (expand existing section)
    SAVE_FIGURES_BY_DEFAULT: bool = True
    FIGURE_FORMATS: List[str] = ["png", "pdf"]
    ADD_TIMESTAMP_TO_FIGURES: bool = True
    AUTO_CREATE_SUBDIRS: bool = True
    
    # Random seeds for reproducibility
    RANDOM_SEED: int = 42
    NUMPY_SEED: int = 42
    
    # Default model parameters
    DEFAULT_TEST_SIZE: float = 0.2
    DEFAULT_CV_FOLDS: int = 5
    DEFAULT_N_JOBS: int = -1
    
    # Data generation defaults
    DEFAULT_N_SAMPLES: int = 1000
    DEFAULT_N_FEATURES: int = 20
    DEFAULT_NOISE_LEVEL: float = 0.1
    
    # Visualization settings
    FIGURE_SIZE: tuple = (12, 8)
    DPI: int = 300
    STYLE: str = "seaborn-v0_8"
    COLOR_PALETTE: str = "husl"
    
    # MLflow settings
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "sklearn-mastery"
    
    # Model evaluation
    CLASSIFICATION_METRICS: List[str] = [
        "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr"
    ]
    REGRESSION_METRICS: List[str] = [
        "neg_mean_squared_error", "neg_mean_absolute_error", "r2"
    ]
    
    # Hyperparameter tuning
    MAX_ITER_OPTUNA: int = 100
    N_TRIALS_OPTUNA: int = 50
    
    # Feature selection
    MAX_FEATURES_SELECT: int = 50
    FEATURE_SELECTION_METHODS: List[str] = [
        "univariate", "rfe", "from_model", "variance_threshold"
    ]
    
    # Pipeline caching
    ENABLE_PIPELINE_CACHING: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True


class ModelDefaults:
    """Default hyperparameters for different models."""
    
    CLASSIFICATION_MODELS = {
        "logistic_regression": {
            "C": [0.1, 1.0, 10.0],
            "solver": ["lbfgs", "liblinear"],
            "max_iter": [1000]
        },
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10]
        },
        "svm": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"]
        },
        "gradient_boosting": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7]
        },
        "mlp": {
            "hidden_layer_sizes": [(100,), (100, 50), (50, 50, 50)],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate": ["constant", "adaptive"]
        }
    }
    
    REGRESSION_MODELS = {
        "linear_regression": {},
        "ridge": {
            "alpha": [0.1, 1.0, 10.0, 100.0]
        },
        "lasso": {
            "alpha": [0.1, 1.0, 10.0, 100.0]
        },
        "elastic_net": {
            "alpha": [0.1, 1.0, 10.0],
            "l1_ratio": [0.1, 0.5, 0.7, 0.9]
        },
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10]
        },
        "svr": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"]
        }
    }
    
    CLUSTERING_MODELS = {
        "kmeans": {
            "n_clusters": range(2, 11),
            "init": ["k-means++", "random"],
            "algorithm": ["auto", "full", "elkan"]
        },
        "dbscan": {
            "eps": [0.3, 0.5, 0.7, 1.0],
            "min_samples": [3, 5, 10, 15]
        },
        "gaussian_mixture": {
            "n_components": range(2, 11),
            "covariance_type": ["full", "tied", "diag", "spherical"]
        }
    }


# Create global settings instance
settings = Settings()

# Ensure directories exist
# Ensure directories exist (replace the existing lines at the end)
for directory in [settings.DATA_DIR, settings.MODELS_DIR, settings.FIGURES_DIR, 
                  settings.REPORTS_DIR, settings.CACHE_DIR, settings.LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Create figure subdirectories
figure_subdirs = [
    "data_generation", "preprocessing", "classification", "regression",
    "clustering", "model_comparison", "hyperparameter_optimization", "interpretability"
]
for subdir in figure_subdirs:
    (settings.FIGURES_DIR / subdir).mkdir(parents=True, exist_ok=True)