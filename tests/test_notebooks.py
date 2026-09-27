"""Execute every tutorial notebook headlessly and assert that no cell raises.

These tests are marked ``slow`` (deselect with ``-m "not slow"``). Each notebook
is executed in a fresh Jupyter kernel with ``nbclient``; a notebook passes when
every code cell runs to completion without an ``error`` output. Notebooks write
their artefacts under ``<repo>/results`` exactly as they do interactively, so
the kernel is started with the notebook directory as its working directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

from nbclient import NotebookClient  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
NOTEBOOKS: List[Path] = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
CELL_TIMEOUT_SECONDS = 600


def _kernel_environment() -> Dict[str, str]:
    """Environment for the kernel: headless matplotlib and an importable package.

    ``PYTHONPATH`` is prefixed with the repository root so the notebooks import
    the checked-out ``sklearn_mastery`` even when the editable install is not
    picked up by the kernel's interpreter.
    """
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
    return env


def _error_outputs(notebook: Any) -> List[str]:
    """Collect ``"<cell index>: <ename>: <evalue>"`` for every error output."""
    errors: List[str] = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append(f"cell {index}: {output.get('ename')}: {output.get('evalue')}")
    return errors


@pytest.mark.slow
@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_notebook_executes_without_errors(notebook_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every notebook under ``notebooks/`` runs end-to-end in a fresh kernel."""
    for key, value in _kernel_environment().items():
        monkeypatch.setenv(key, value)

    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=CELL_TIMEOUT_SECONDS,
        kernel_name="python3",
        allow_errors=True,
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()

    errors = _error_outputs(notebook)
    assert not errors, f"{notebook_path.name} raised in {len(errors)} cell(s):\n" + "\n".join(errors)


def test_notebooks_are_saved_without_outputs() -> None:
    """Committed notebooks must be clean: no outputs and no execution counts."""
    assert NOTEBOOKS, f"no notebooks found under {NOTEBOOK_DIR}"
    dirty: List[str] = []
    for path in NOTEBOOKS:
        notebook = nbformat.read(path, as_version=4)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            if cell.get("outputs") or cell.get("execution_count") is not None:
                dirty.append(f"{path.name} cell {index}")
    assert not dirty, "notebooks with saved outputs/execution counts:\n" + "\n".join(dirty)


def test_notebooks_do_not_manipulate_sys_path() -> None:
    """Notebooks rely on the installed package, never on ``sys.path`` hacks."""
    offenders: List[str] = []
    for path in NOTEBOOKS:
        notebook = nbformat.read(path, as_version=4)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code" and "sys.path" in cell.source:
                offenders.append(f"{path.name} cell {index}")
    assert not offenders, "sys.path manipulation found in:\n" + "\n".join(offenders)


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    sys.exit(pytest.main([__file__, "-m", "slow", "-v"]))
