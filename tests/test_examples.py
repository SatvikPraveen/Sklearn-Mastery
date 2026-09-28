"""Run every real-world example script headlessly and assert it exits cleanly.

These tests are marked ``slow`` (deselect with ``-m "not slow"``). Each script
under ``examples/real_world_scenarios/<domain>/`` is executed in a fresh
interpreter with ``MPLBACKEND=Agg`` and a 300 s timeout (override with
``SKLEARN_MASTERY_SCRIPT_TIMEOUT``); a script passes when
its exit code is 0. Scripts save their figures under ``<repo>/results`` and
never open a window, so the suite runs on CI and over SSH.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples" / "real_world_scenarios"
SCRIPT_TIMEOUT_SECONDS = int(os.environ.get("SKLEARN_MASTERY_SCRIPT_TIMEOUT", "300"))


def _discover_example_scripts() -> List[Path]:
    """Return every runnable domain script, excluding utilities and packages."""
    scripts = [
        path
        for path in sorted(EXAMPLES_DIR.glob("*/*.py"))
        if path.parent.name != "utilities" and path.name != "__init__.py"
    ]
    return scripts


EXAMPLE_SCRIPTS: List[Path] = _discover_example_scripts()


def _script_environment() -> Dict[str, str]:
    """Headless matplotlib plus an importable checked-out package."""
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env.setdefault("PYTHONHASHSEED", "0")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
    return env


def test_example_scripts_are_discovered() -> None:
    """The discovery glob must find the domain scripts (guards against renames)."""
    assert len(EXAMPLE_SCRIPTS) >= 20, f"only found {len(EXAMPLE_SCRIPTS)} scripts under {EXAMPLES_DIR}"
    names = {path.name for path in EXAMPLE_SCRIPTS}
    assert "customer_churn_prediction.py" in names
    assert not any(name.startswith("custome_") for name in names)


@pytest.mark.slow
@pytest.mark.parametrize(
    "script_path",
    EXAMPLE_SCRIPTS,
    ids=[f"{p.parent.name}/{p.stem}" for p in EXAMPLE_SCRIPTS],
)
def test_example_script_runs_to_completion(script_path: Path) -> None:
    """Every example script exits with status 0 within the timeout."""
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=REPO_ROOT,
        env=_script_environment(),
        capture_output=True,
        text=True,
        timeout=SCRIPT_TIMEOUT_SECONDS,
        check=False,
    )
    tail = "\n".join(result.stderr.strip().splitlines()[-30:])
    assert result.returncode == 0, (
        f"{script_path.relative_to(REPO_ROOT)} exited with {result.returncode}\n--- stderr tail ---\n{tail}"
    )
