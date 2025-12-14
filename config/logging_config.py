"""Logging configuration for sklearn-mastery project."""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any

try:
    from rich.logging import RichHandler
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
    rich_console: bool = True
) -> None:
    """Setup logging configuration with optional file output and rich formatting."""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Default log file
    if log_file is None:
        from .settings import settings
        log_file = settings.LOGS_DIR / "sklearn_mastery.log"
    
    # Configure formatters
    formatters = {
        "detailed": {
            "format": "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {
            "format": "%(levelname)s | %(name)s | %(message)s",
        },
    }
    
    # Configure handlers - use RichHandler if available, otherwise standard StreamHandler
    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(log_file),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8",
        },
    }
    
    # Configure loggers
    loggers = {
        "sklearn_mastery": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False,
        },
        # Suppress verbose third-party loggers
        "matplotlib": {"level": "WARNING"},
        "urllib3": {"level": "WARNING"},
        "requests": {"level": "WARNING"},
    }
    
    # Complete logging configuration
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "loggers": loggers,
        "root": {
            "level": log_level,
            "handlers": ["console"],
        },
    }
    
    # Apply configuration
    logging.config.dictConfig(logging_config)
    
    # Get the main logger
    logger = logging.getLogger("sklearn_mastery")
    logger.info(f"Logging initialized with level: {log_level}")
    logger.info(f"Log file: {log_file}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name under the sklearn_mastery namespace."""
    return logging.getLogger(f"sklearn_mastery.{name}")


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)


# Performance logging utilities
class PerformanceLogger:
    """Context manager for logging performance metrics."""
    
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        if exc_type is None:
            self.logger.info(f"Completed {self.operation} in {duration:.2f}s")
        else:
            self.logger.error(f"Failed {self.operation} after {duration:.2f}s: {exc_val}")


def log_model_performance(logger: logging.Logger, model_name: str, metrics: Dict[str, float]) -> None:
    """Log model performance metrics in a structured format."""
    logger.info(f"Model Performance - {model_name}")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")


def log_data_info(logger: logging.Logger, X, y=None, dataset_name: str = "Dataset") -> None:
    """Log information about dataset dimensions and types."""
    logger.info(f"{dataset_name} Info:")
    logger.info(f"  Features shape: {X.shape}")
    if y is not None:
        if hasattr(y, 'shape'):
            logger.info(f"  Target shape: {y.shape}")
        logger.info(f"  Target type: {type(y).__name__}")
    logger.info(f"  Feature types: {X.dtypes.value_counts().to_dict() if hasattr(X, 'dtypes') else 'NumPy array'}")


def log_figure_saved(logger: logging.Logger, filepath: str, subfolder: str = None) -> None:
    """Log when a figure is saved with standardized format."""
    if subfolder:
        logger.info(f"📊 Figure saved: {subfolder}/{Path(filepath).name}")
    else:
        logger.info(f"📊 Figure saved: {Path(filepath).name}")


def log_experiment_start(logger: logging.Logger, experiment_name: str, parameters: Dict[str, Any] = None) -> None:
    """Log the start of an experiment with parameters."""
    logger.info(f"🧪 Starting experiment: {experiment_name}")
    if parameters:
        for key, value in parameters.items():
            logger.info(f"  {key}: {value}")


def log_experiment_result(logger: logging.Logger, experiment_name: str, metrics: Dict[str, float]) -> None:
    """Log experiment results in a structured format."""
    logger.info(f"✅ Experiment completed: {experiment_name}")
    for metric, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")


# Initialize default logging on module import
setup_logging()