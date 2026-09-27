# Contributing to sklearn-mastery

Thanks for considering a contribution. This project aims to be a research-grade
toolkit, so the bar is: correct, tested, documented, and reproducible.

## Development setup

```bash
git clone https://github.com/SatvikPraveen/Sklearn-Mastery.git
cd Sklearn-Mastery
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"
pre-commit install
```

## Workflow

1. Open an issue describing the change (bug, feature, or methodology addition).
2. Create a branch from `main`.
3. Make the change with tests and docstrings.
4. Run the checks below; CI runs the same ones.
5. Open a pull request with a clear description and, for statistical or
   algorithmic additions, the reference(s) you implemented.

```bash
make lint          # ruff check + ruff format --check
make type-check    # mypy on typed subpackages
make test          # pytest with coverage (fast tier)
pytest -m slow     # notebooks and example scripts (several minutes)
make docs          # mkdocs build --strict
```

## Code standards

- **Style**: `ruff` (line length 110) for linting and formatting. No `print()`
  in library code; use `LoggerMixin` / `get_logger`.
- **Imports**: absolute (`from sklearn_mastery.x import y`). Never modify
  `sys.path`. Optional heavy dependencies (xgboost, lightgbm, optuna, shap,
  imbalanced-learn, plotly, umap) are imported inside `try/except ImportError`
  with a module-level `HAS_<NAME>` flag; the module must import without them.
- **scikit-learn compatibility**: estimators subclass `BaseEstimator` plus the
  right mixin, store constructor arguments unchanged, use trailing-underscore
  fitted attributes, return `self` from `fit`, and work with `clone`,
  `Pipeline` and `GridSearchCV`.
- **Determinism**: accept `random_state` and derive every random draw from it.
- **Docstrings**: Google style (`Args:` / `Returns:` / `Raises:`), one parameter
  per line, with the relevant citation for any published method. Maths may be
  written in MathJax (`$...$`, `$$...$$`) and renders in the docs.
- **Tests**: every public behaviour has a test; numerical procedures are
  checked against hand-computed values or a reference implementation
  (scipy, scikit-learn, xgboost). Tests are deterministic and fast; anything
  above a few seconds goes in the `slow` tier (`@pytest.mark.slow`).

## Adding a statistical procedure

Add it to `sklearn_mastery/research/comparison.py` (or a new module), document
the null hypothesis, assumptions, and reference in the docstring, expose it in
`sklearn_mastery/research/__init__.py`, add a test that reproduces a published
worked example or agrees with scipy where one exists, and describe when to use
it in `docs/research/statistical_comparison.md`.

## Commit messages

Conventional-commit style: `feat(research): ...`, `fix(pipelines): ...`,
`docs: ...`, `test: ...`, `ci: ...`, `refactor: ...`. One logical change per
commit.

## Reporting issues

Include the version (`sklearn-mastery --version`), the environment
(`sklearn-mastery info`), a minimal reproducible example, and the full
traceback.
