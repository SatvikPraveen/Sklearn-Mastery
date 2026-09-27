# Contributing

Thank you for considering a contribution. This page is the documentation
version of the repository's `CONTRIBUTING.md`; the workflow below is what
continuous integration enforces.

## Development setup

```bash
git clone https://github.com/<your-user>/Sklearn-Mastery.git
cd Sklearn-Mastery
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"
pre-commit install
```

or simply `make install-dev`, which runs the last two commands. Python 3.9+
is required; CI tests 3.10 to 3.12.

## Repository layout

```
sklearn_mastery/     the package (research, data, models, pipelines, evaluation, from_scratch, config, cli)
tests/               pytest suite mirroring the package (test_research, test_data, test_models, ...)
docs/ + mkdocs.yml   this site
examples/            domain scripts built on the model and evaluation layers
notebooks/           tutorial notebooks
```

## Everyday commands

| Task | Command |
|---|---|
| Run the test suite in parallel | `make test` (`pytest -n auto --cov -q`) |
| Fast run, stop at first failure | `make test-fast` |
| Lint and check formatting | `make lint` (`ruff check` + `ruff format --check`) |
| Auto-format | `make format` |
| Type-check the typed subpackages | `make type-check` (`mypy sklearn_mastery/research sklearn_mastery/config`) |
| Everything CI runs | `make check` |
| Build the documentation | `make docs` (`mkdocs build --strict`) |
| Serve the documentation locally | `make serve-docs` |
| Build sdist and wheel | `make build` |

## Coding standards

- **Style and imports** are enforced by [ruff](https://docs.astral.sh/ruff/)
  (line length 110; rule sets `E, F, W, I, B, UP, NPY, RUF`). Run
  `ruff check --fix` and `ruff format` before committing; the pre-commit hook
  does this for you.
- **Type hints** on every public signature. `sklearn_mastery/research` and
  `sklearn_mastery/config` are checked with mypy in CI and by the pre-commit
  hook.
- **Docstrings** use Google style (`Args:`, `Returns:`, `Raises:`,
  `Attributes:`, `Example:`), which is what the API reference is rendered
  from. Module docstrings should cite the relevant papers.
- **scikit-learn compatibility** for estimators: inherit from
  `BaseEstimator` and the appropriate mixin, store constructor arguments
  verbatim, return `self` from `fit`, suffix fitted attributes with `_`, use
  `check_is_fitted`, and implement `get_feature_names_out` for transformers.
- **Determinism**: expose `random_state` and derive every random draw from
  it.
- **No print** in library code; use `get_logger(__name__)` or the
  `LoggerMixin.logger` property from `sklearn_mastery.config.logging_config`.
- **Optional dependencies** (`xgboost`, `lightgbm`, `optuna`, `shap`,
  `imbalanced-learn`, `plotly`, `umap-learn`) must be guarded with
  `try/except ImportError` and a module-level `HAS_*` flag so that every
  module imports without them.
- **Absolute imports** only (`from sklearn_mastery.config.settings import settings`).

## Tests

Tests live under `tests/` and mirror the package structure. Add tests for
every new public function or class, cover edge cases and error paths, keep
them deterministic (`random_state`) and fast; mark long-running tests with
`@pytest.mark.slow` (deselect with `-m "not slow"`).

```bash
pytest tests/test_research -q                 # one package
pytest -n 4 -x -q                             # parallel, stop on first failure
pytest --cov=sklearn_mastery --cov-report=term-missing
```

## Documentation

The site is built with MkDocs (Material theme) and mkdocstrings. Guides live
in `docs/`, the API reference pages contain `::: sklearn_mastery.<module>`
directives and are generated from docstrings.

```bash
pip install -e ".[docs]"      # or: pip install -r requirements-docs.txt
mkdocs serve                  # live preview at http://127.0.0.1:8000
mkdocs build --strict         # what CI runs; warnings fail the build
```

Every code snippet in the guides must import real symbols and run against
the current package. Formulas use MathJax (`$$ ... $$`).

## Pull requests

1. Open an issue first for new features or behaviour changes.
2. Create a branch from `main` (`git checkout -b feat/short-name`).
3. Write code, tests and docs; run `make check`.
4. Use [conventional commit](https://www.conventionalcommits.org/) messages:
   `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `chore:`.
5. Describe the motivation, the change and how you tested it in the PR;
   list breaking changes explicitly and add a line to `CHANGELOG.md`.
6. CI (lint, type-check, test matrix, distribution build, docs) must pass
   and one maintainer review is required before merging.

## Reporting bugs

Include the output of `sklearn-mastery info`, a minimal reproducible
example, the expected and actual behaviour, and the full traceback.

## Code of conduct

This project follows the
[Contributor Covenant](https://www.contributor-covenant.org/). Be respectful,
welcome newcomers and give constructive feedback.
