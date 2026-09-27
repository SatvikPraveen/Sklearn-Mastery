# Developer workflow for sklearn-mastery
PYTHON ?= python
PIP    ?= $(PYTHON) -m pip

.PHONY: help install install-dev test test-fast lint format type-check check clean docs serve-docs benchmark-demo build

help:
	@echo "install       - install the package"
	@echo "install-dev   - editable install with dev extras + pre-commit hooks"
	@echo "test          - run the full test suite with coverage"
	@echo "test-fast     - run tests in parallel, stop on first failure"
	@echo "lint          - ruff lint + format check"
	@echo "format        - auto-format with ruff"
	@echo "type-check    - mypy on typed subpackages"
	@echo "check         - lint + type-check + test (CI equivalent)"
	@echo "benchmark-demo- run the research benchmark demo"
	@echo "docs          - build the mkdocs site"
	@echo "build         - build sdist and wheel"

install:
	$(PIP) install .

install-dev:
	$(PIP) install -e ".[dev,docs]"
	pre-commit install

test:
	$(PYTHON) -m pytest -n auto --cov --cov-report=term-missing -q

test-fast:
	$(PYTHON) -m pytest -n auto -x -q

lint:
	ruff check sklearn_mastery tests
	ruff format --check sklearn_mastery tests

format:
	ruff check --fix sklearn_mastery tests
	ruff format sklearn_mastery tests

type-check:
	mypy sklearn_mastery/research sklearn_mastery/config

check: lint type-check test

benchmark-demo:
	$(PYTHON) -m sklearn_mastery.cli benchmark --quick

docs:
	mkdocs build --strict

serve-docs:
	mkdocs serve

build:
	$(PYTHON) -m build

clean:
	rm -rf build dist *.egg-info .coverage coverage.xml htmlcov .pytest_cache .mypy_cache .ruff_cache site
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
