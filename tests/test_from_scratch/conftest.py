"""Shared fixtures for the from-scratch estimator tests."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, load_diabetes, make_classification, make_regression
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="session")
def binary_data():
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5, n_redundant=2, flip_y=0.05, random_state=0
    )
    return train_test_split(X, y, test_size=0.3, random_state=0)


@pytest.fixture(scope="session")
def multiclass_data():
    X, y = make_classification(
        n_samples=600, n_features=12, n_informative=6, n_redundant=2, n_classes=3, flip_y=0.03, random_state=1
    )
    return train_test_split(X, y, test_size=0.3, random_state=1)


@pytest.fixture(scope="session")
def regression_data():
    X, y = make_regression(n_samples=500, n_features=10, n_informative=5, noise=10.0, random_state=2)
    return train_test_split(X, y, test_size=0.3, random_state=2)


@pytest.fixture(scope="session")
def cancer_data():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, random_state=3, stratify=y)


@pytest.fixture(scope="session")
def diabetes_data():
    X, y = load_diabetes(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, random_state=4)


def _api_smoke(estimator, X, y, classifier: bool):
    """Common scikit-learn API checks every scratch estimator must pass."""
    # unfitted state
    with pytest.raises(NotFittedError):
        estimator.predict(X)
    # get_params / set_params round trip and clone
    params = estimator.get_params()
    estimator.set_params(**params)
    assert clone(estimator).get_params() == params
    # fit returns self and sets fitted attributes
    fitted = estimator.fit(X, y)
    assert fitted is estimator
    assert estimator.n_features_in_ == X.shape[1]
    pred = estimator.predict(X)
    assert pred.shape == (X.shape[0],)
    if classifier:
        assert np.array_equal(estimator.classes_, np.unique(y))
        proba = estimator.predict_proba(X)
        assert proba.shape == (X.shape[0], len(estimator.classes_))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
        assert np.all(proba >= 0)
        assert set(pred) <= set(estimator.classes_)
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        assert importances.shape == (X.shape[1],)
        assert np.all(importances >= 0)
        assert np.isclose(importances.sum(), 1.0)
    # pickling preserves predictions
    restored = pickle.loads(pickle.dumps(estimator))
    np.testing.assert_array_equal(restored.predict(X), pred)
    # wrong feature count is rejected
    with pytest.raises(ValueError):
        estimator.predict(X[:, :-1])
    return estimator


@pytest.fixture
def api_smoke():
    return _api_smoke
