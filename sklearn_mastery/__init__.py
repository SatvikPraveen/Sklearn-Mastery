"""sklearn-mastery: a research-grade toolkit built on scikit-learn.

The package provides synthetic data generators, composable preprocessing and
model-selection pipelines, thin estimator wrappers, an evaluation and
statistical-testing layer, and a ``research`` subpackage for reproducible
benchmarking with rigorous multi-model comparison.

Importing the package has no side effects: logging is configured only when
:func:`sklearn_mastery.setup_logging` is called explicitly, and result
directories are created lazily via :meth:`Settings.ensure_directories`.
"""

from __future__ import annotations

from sklearn_mastery.config.logging_config import get_logger, setup_logging
from sklearn_mastery.config.settings import ModelDefaults, settings

__version__ = "2.0.0"
__author__ = "Satvik Praveen"
__email__ = "satvikpraveen707@gmail.com"

__all__ = [
    "ModelDefaults",
    "__author__",
    "__email__",
    "__version__",
    "get_logger",
    "settings",
    "setup_logging",
]
