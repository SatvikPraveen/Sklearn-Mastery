"""Reproducibility utilities: seeding, environment capture, run manifests.

A benchmark result is only as credible as its provenance. This module makes
provenance cheap to record:

* :func:`set_global_seed` seeds Python, NumPy and (if installed) optional
  libraries in one call.
* :func:`capture_environment` snapshots interpreter, platform, key package
  versions and the current git commit.
* :class:`RunManifest` bundles a configuration dictionary with the environment
  snapshot and a content hash so runs can be de-duplicated and diffed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union

import numpy as np

__all__ = [
    "RunManifest",
    "capture_environment",
    "config_hash",
    "get_git_revision",
    "set_global_seed",
    "to_jsonable",
]

_DEFAULT_PACKAGES: tuple = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "joblib",
    "matplotlib",
    "xgboost",
    "lightgbm",
    "optuna",
    "imbalanced-learn",
)


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed every random source the toolkit may use.

    Args:
        seed: Non-negative integer seed.

    Returns:
        A fresh :class:`numpy.random.Generator` seeded with ``seed`` for callers
        that prefer the modern generator API.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global RNG used by scikit-learn
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # pragma: no cover - optional dependency
        import torch  # type: ignore

        torch.manual_seed(seed)
    except Exception:
        pass
    return np.random.default_rng(seed)


def get_git_revision(path: Union[str, Path, None] = None) -> Optional[str]:
    """Return the current git commit hash for ``path`` (or cwd), if available."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(path) if path else None,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _package_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def capture_environment(packages: Iterable[str] = _DEFAULT_PACKAGES) -> Dict[str, Any]:
    """Snapshot the execution environment.

    Args:
        packages: Distribution names whose versions should be recorded.

    Returns:
        Dictionary with interpreter, platform, package versions, git revision
        and a UTC timestamp.
    """
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": {name: _package_version(name) for name in packages},
        "git_revision": get_git_revision(),
    }


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays, paths and sets to JSON-safe types."""
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(to_jsonable(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "get_params") and callable(obj.get_params):
        return {"__estimator__": type(obj).__name__, "params": to_jsonable(obj.get_params(deep=False))}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def config_hash(config: Mapping[str, Any], length: int = 12) -> str:
    """Stable SHA-256 prefix of a JSON-serialisable configuration."""
    payload = json.dumps(to_jsonable(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


@dataclass
class RunManifest:
    """Provenance record for a single experimental run.

    Attributes:
        name: Human-readable run name.
        config: Full configuration (estimators, protocol, seeds, ...).
        seed: Master random seed.
        environment: Output of :func:`capture_environment`.
        tags: Free-form key/value metadata.
    """

    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    environment: Dict[str, Any] = field(default_factory=capture_environment)
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def hash(self) -> str:
        """Content hash of ``config`` and ``seed`` (environment excluded)."""
        return config_hash({"config": self.config, "seed": self.seed})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "hash": self.hash,
            "seed": self.seed,
            "config": to_jsonable(self.config),
            "environment": self.environment,
            "tags": dict(self.tags),
        }

    def save(self, path: Union[str, Path]) -> Path:
        """Write the manifest as JSON and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> RunManifest:
        data = json.loads(Path(path).read_text())
        return cls(
            name=data["name"],
            config=data.get("config", {}),
            seed=int(data.get("seed", 42)),
            environment=data.get("environment", {}),
            tags=data.get("tags", {}),
        )
