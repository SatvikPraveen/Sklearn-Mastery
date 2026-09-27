"""Root pytest configuration: shared fixtures and deterministic seeding."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_numpy():
    """Seed NumPy before every test so stochastic tests are reproducible."""
    np.random.seed(42)
    yield


@pytest.fixture(autouse=True)
def _headless_matplotlib(monkeypatch):
    """Force a non-interactive matplotlib backend and close figures after tests."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
