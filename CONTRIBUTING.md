# Contributing to MyML

**File Location:** `docs/CONTRIBUTING.md`

Thank you for your interest in contributing to MyML! This document provides guidelines and information for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

   ```bash
   git clone https://github.com/YOUR_USERNAME/myml.git
   cd myml
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/myml.git
   ```

## Development Setup

### Environment Setup

1. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Project Structure

```
myml/
├── myml/                 # Main package
│   ├── algorithms/       # Algorithm implementations
│   ├── data/            # Data processing utilities
│   ├── evaluation/      # Evaluation metrics
│   ├── pipelines/       # ML pipelines
│   └── utils/           # Utility functions
├── tests/               # Test suite
├── docs/                # Documentation
├── examples/            # Example scripts
├── setup.py             # Package setup
└── requirements.txt     # Dependencies
```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

1. **Bug fixes**: Fix existing issues in the codebase
2. **New features**: Add new algorithms or functionality
3. **Documentation**: Improve or add documentation
4. **Tests**: Add or improve test coverage
5. **Examples**: Create usage examples and tutorials
6. **Performance improvements**: Optimize existing code

### Before You Start

1. **Check existing issues**: Look for existing issues or discussions
2. **Create an issue**: For new features or major changes, create an issue first
3. **Discuss your approach**: Get feedback before starting major work

## Code Style

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with these additions:

- Line length: 88 characters (Black formatter default)
- Use type hints for function signatures
- Use descriptive variable names
- Add docstrings for all public functions and classes

### Code Formatting

We use automated code formatting tools:

```bash
# Format code with Black
black myml/ tests/

# Sort imports with isort
isort myml/ tests/

# Check style with flake8
flake8 myml/ tests/
```

### Example Function Format

```python
from typing import Optional, Tuple, Union
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class MyMLAlgorithm(BaseEstimator, ClassifierMixin):
    """
    Brief description of the algorithm.

    Longer description explaining the algorithm's purpose,
    key features, and when to use it.

    Parameters
    ----------
    param1 : float, default=1.0
        Description of param1.
    param2 : str, default='auto'
        Description of param2.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    attribute1_ : ndarray of shape (n_features,)
        Description of fitted attribute.

    Examples
    --------
    >>> from myml.algorithms import MyMLAlgorithm
    >>> clf = MyMLAlgorithm(param1=0.5)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    """

    def __init__(
        self,
        param1: float = 1.0,
        param2: str = 'auto',
        random_state: Optional[int] = None
    ):
        self.param1 = param1
        self.param2 = param2
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> 'MyMLAlgorithm':
        """
        Fit the model to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : MyMLAlgorithm
            Fitted estimator.
        """
        # Implementation here
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        # Implementation here
        pass
```

## Testing

### Test Structure

Tests are located in the `tests/` directory and mirror the package structure:

```
tests/
├── test_algorithms/
├── test_data/
├── test_evaluation/
├── test_pipelines/
└── test_utils/
```

### Writing Tests

1. Use `pytest` for testing
2. Write unit tests for all public functions
3. Include integration tests for workflows
4. Test edge cases and error conditions
5. Aim for high test coverage (>90%)

### Test Example

```python
import pytest
import numpy as np
from sklearn.datasets import make_classification
from myml.algorithms import MyMLAlgorithm


class TestMyMLAlgorithm:
    """Test suite for MyMLAlgorithm."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X, y = make_classification(
            n_samples=100, n_features=4, n_classes=2, random_state=42
        )
        return X, y

    def test_init(self):
        """Test algorithm initialization."""
        clf = MyMLAlgorithm(param1=0.5, param2='test')
        assert clf.param1 == 0.5
        assert clf.param2 == 'test'

    def test_fit(self, sample_data):
        """Test model fitting."""
        X, y = sample_data
        clf = MyMLAlgorithm()

        # Should return self
        result = clf.fit(X, y)
        assert result is clf

        # Should have fitted attributes
        assert hasattr(clf, 'attribute1_')

    def test_predict(self, sample_data):
        """Test prediction functionality."""
        X, y = sample_data
        clf = MyMLAlgorithm()
        clf.fit(X, y)

        predictions = clf.predict(X)

        # Check output shape
        assert predictions.shape == (len(X),)

        # Check output type
        assert isinstance(predictions, np.ndarray)

    def test_invalid_input(self):
        """Test handling of invalid inputs."""
        clf = MyMLAlgorithm()

        with pytest.raises(ValueError):
            clf.fit([[1, 2]], [])  # Mismatched dimensions

    @pytest.mark.parametrize("param1", [0.1, 0.5, 1.0, 2.0])
    def test_different_parameters(self, sample_data, param1):
        """Test algorithm with different parameter values."""
        X, y = sample_data
        clf = MyMLAlgorithm(param1=param1)
        clf.fit(X, y)
        predictions = clf.predict(X)

        assert predictions is not None
        assert len(predictions) == len(X)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=myml

# Run specific test file
pytest tests/test_algorithms/test_classification.py

# Run with verbose output
pytest -v
```

## Documentation

### Docstring Style

We use NumPy-style docstrings:

```python
def example_function(param1: float, param2: str = 'default') -> bool:
    """
    One-line summary of the function.

    Longer description of what the function does,
    including any important details or caveats.

    Parameters
    ----------
    param1 : float
        Description of param1.
    param2 : str, default='default'
        Description of param2.

    Returns
    -------
    bool
        Description of return value.

    Raises
    ------
    ValueError
        When param1 is negative.

    Examples
    --------
    >>> result = example_function(1.5, 'test')
    >>> print(result)
    True
    """
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs/
mkdocs build

# Serve documentation locally
mkdocs serve
```

## Pull Request Process

### Before Submitting

1. **Sync with upstream**:

   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create feature branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make changes and commit**:

   ```bash
   git add .
   git commit -m "Add: Brief description of changes"
   ```

4. **Run tests and checks**:
   ```bash
   pytest
   black myml/ tests/
   flake8 myml/ tests/
   ```

### Commit Message Guidelines

Use conventional commit format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions or changes
- `refactor:` Code refactoring
- `style:` Code style changes
- `perf:` Performance improvements

Examples:

```
feat: Add support for multi-class SVM
fix: Handle edge case in decision tree splitting
docs: Update installation instructions
test: Add integration tests for ensemble methods
```

### Pull Request Template

When submitting a PR, include:

1. **Description**: Clear description of changes
2. **Motivation**: Why the changes are needed
3. **Testing**: How you tested the changes
4. **Documentation**: Any documentation updates
5. **Breaking changes**: List any breaking changes

### Review Process

1. Automated checks must pass
2. At least one maintainer review required
3. Address reviewer feedback
4. Squash commits if requested
5. Maintainer will merge when approved

## Issue Reporting

### Bug Reports

Include the following information:

1. **Environment**: Python version, OS, package versions
2. **Reproduction**: Minimal code to reproduce the issue
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Error messages**: Full error traceback

### Feature Requests

For new features, describe:

1. **Use case**: Why the feature is needed
2. **Proposed solution**: How it should work
3. **Alternatives**: Other solutions considered
4. **Implementation ideas**: If you have any

### Issue Template

````markdown
## Bug Report

**Environment:**

- Python version:
- MyML version:
- Operating System:

**To Reproduce:**

```python
# Minimal code example
```
````

**Expected Behavior:**
Description of expected behavior

**Actual Behavior:**
Description of what actually happened

**Error Message:**

```
Full error traceback
```

**Additional Context:**
Any other relevant information

```

## Community Guidelines

### Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct. Please read and follow these guidelines:

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Give constructive feedback
- Focus on what's best for the community

### Communication

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Pull Request Comments**: Code review discussions

### Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Documentation acknowledgments

## Development Workflow

### Typical Workflow

1. Find or create an issue
2. Fork and clone the repository
3. Create a feature branch
4. Make changes with tests
5. Update documentation
6. Submit pull request
7. Address review feedback
8. Merge when approved

### Release Process

1. Version bumping follows [Semantic Versioning](https://semver.org/)
2. Releases are tagged and published to PyPI
3. Release notes summarize changes
4. Documentation is updated automatically

## Getting Help

If you need help:

1. Check existing documentation
2. Search through issues
3. Ask questions in GitHub Discussions
4. Reach out to maintainers

Thank you for contributing to MyML! Your contributions help make machine learning more accessible to everyone.
```
