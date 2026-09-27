"""Logging configuration for sklearn-mastery.

The package logger (``sklearn_mastery``) carries a ``NullHandler`` by default,
following library best practice. Call :func:`setup_logging` from applications,
notebooks, or the CLI to attach console (optionally Rich) and rotating-file
handlers.
"""

from __future__ import annotations

import logging
import logging.config
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:  # optional pretty console output
    from rich.logging import RichHandler  # noqa: F401

    HAS_RICH = True
except ImportError:  # pragma: no cover - exercised only without rich installed
    HAS_RICH = False

PACKAGE_LOGGER = "sklearn_mastery"
logging.getLogger(PACKAGE_LOGGER).addHandler(logging.NullHandler())


def setup_logging(
    log_level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    rich_console: bool = True,
    log_to_file: bool = False,
) -> logging.Logger:
    """Configure logging for the package.

    Args:
        log_level: Console log level (name or numeric).
        log_file: Explicit log-file path. When given, file logging is enabled.
        rich_console: Use ``rich`` for the console handler when available.
        log_to_file: Enable file logging at ``settings.LOGS_DIR`` even when
            ``log_file`` is not given.

    Returns:
        The configured package logger.
    """
    level_name = logging.getLevelName(log_level) if isinstance(log_level, int) else str(log_level).upper()

    formatters: Dict[str, Any] = {
        "detailed": {
            "format": "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {"format": "%(levelname)s | %(name)s | %(message)s"},
    }

    handlers: Dict[str, Any] = {}
    if rich_console and HAS_RICH:
        handlers["console"] = {
            "class": "rich.logging.RichHandler",
            "level": level_name,
            "rich_tracebacks": True,
            "markup": False,
            "show_path": False,
        }
    else:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "level": level_name,
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        }

    package_handlers = ["console"]
    resolved_file: Optional[Path] = None
    if log_file is not None or log_to_file:
        if log_file is None:
            from sklearn_mastery.config.settings import settings

            resolved_file = settings.LOGS_DIR / "sklearn_mastery.log"
        else:
            resolved_file = Path(log_file)
        resolved_file.parent.mkdir(parents=True, exist_ok=True)
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(resolved_file),
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
            "encoding": "utf8",
        }
        package_handlers.append("file")

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": formatters,
            "handlers": handlers,
            "loggers": {
                PACKAGE_LOGGER: {"level": "DEBUG", "handlers": package_handlers, "propagate": False},
                "matplotlib": {"level": "WARNING"},
                "urllib3": {"level": "WARNING"},
            },
            "root": {"level": level_name, "handlers": ["console"]},
        }
    )

    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.debug("Logging initialised (level=%s, file=%s)", level_name, resolved_file)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger in the ``sklearn_mastery`` namespace."""
    if name.startswith(PACKAGE_LOGGER):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER}.{name}")


class LoggerMixin:
    """Mixin providing a class-scoped ``logger`` property."""

    @property
    def logger(self) -> logging.Logger:
        return get_logger(self.__class__.__name__)


class PerformanceLogger:
    """Context manager that logs the wall-clock duration of an operation."""

    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time: Optional[float] = None
        self.duration: Optional[float] = None

    def __enter__(self) -> PerformanceLogger:
        self.start_time = time.perf_counter()
        self.logger.info("Starting %s", self.operation)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.duration = time.perf_counter() - (self.start_time or time.perf_counter())
        if exc_type is None:
            self.logger.info("Completed %s in %.2fs", self.operation, self.duration)
        else:
            self.logger.error("Failed %s after %.2fs: %s", self.operation, self.duration, exc_val)


def log_model_performance(logger: logging.Logger, model_name: str, metrics: Dict[str, float]) -> None:
    """Log model metrics in a structured format."""
    logger.info("Model performance - %s", model_name)
    for metric, value in metrics.items():
        logger.info("  %s: %.4f", metric, value)


def log_data_info(logger: logging.Logger, X, y=None, dataset_name: str = "Dataset") -> None:
    """Log dataset dimensions and dtypes."""
    logger.info("%s info:", dataset_name)
    logger.info("  Features shape: %s", getattr(X, "shape", None))
    if y is not None:
        logger.info("  Target shape: %s (%s)", getattr(y, "shape", None), type(y).__name__)
    if hasattr(X, "dtypes"):
        logger.info("  Feature dtypes: %s", X.dtypes.value_counts().to_dict())


def log_figure_saved(
    logger: logging.Logger, filepath: Union[str, Path], subfolder: Optional[str] = None
) -> None:
    """Log that a figure was written to disk."""
    name = Path(filepath).name
    logger.info("Figure saved: %s", f"{subfolder}/{name}" if subfolder else name)


def log_experiment_start(
    logger: logging.Logger, experiment_name: str, parameters: Optional[Dict[str, Any]] = None
) -> None:
    """Log the start of an experiment together with its parameters."""
    logger.info("Starting experiment: %s", experiment_name)
    for key, value in (parameters or {}).items():
        logger.info("  %s: %s", key, value)


def log_experiment_result(logger: logging.Logger, experiment_name: str, metrics: Dict[str, Any]) -> None:
    """Log experiment results."""
    logger.info("Experiment completed: %s", experiment_name)
    for metric, value in metrics.items():
        logger.info("  %s: %s", metric, f"{value:.4f}" if isinstance(value, float) else value)


__all__ = [
    "HAS_RICH",
    "LoggerMixin",
    "PerformanceLogger",
    "get_logger",
    "log_data_info",
    "log_experiment_result",
    "log_experiment_start",
    "log_figure_saved",
    "log_model_performance",
    "setup_logging",
]
