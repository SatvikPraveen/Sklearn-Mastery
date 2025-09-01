"""Scikit-Learn Mastery: A Comprehensive ML Portfolio Project.

This package provides advanced machine learning utilities, custom transformers,
and comprehensive evaluation tools built on top of scikit-learn.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core imports for easy access
from .config.settings import settings
from .config.logging_config import setup_logging, get_logger

# Initialize logging
setup_logging()

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "settings",
    "setup_logging",
    "get_logger"
]