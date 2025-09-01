# Makefile for sklearn-mastery project

.PHONY: help install install-dev test lint format type-check clean docs serve-docs demo

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install package and dependencies"
	@echo "  install-dev  - Install package in development mode with dev dependencies"
	@echo "  test         - Run tests with coverage"
	@echo "  lint         - Run linting (flake8)"
	@echo "  format       - Format code (black + isort)"
	@echo "  type-check   - Run type checking (mypy)"
	@echo "  clean        - Clean temporary files and caches"
	@echo "  docs         - Build documentation"
	@echo "  serve-docs   - Serve documentation locally"
	@echo "  demo         - Run demonstration"
	@echo "  all          - Run format, lint, type-check, and test"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v

test-fast:
	pytest tests/ -x -v

# Code quality
lint:
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	
format:
	black src/ tests/ notebooks/
	isort src/ tests/

format-check:
	black --check src/ tests/
	isort --check-only src/ tests/

type-check:
	mypy src/ --ignore-missing-imports

# Development workflow
all: format lint type-check test

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf results/
	rm -rf logs/

# Documentation
docs:
	cd docs && make html

serve-docs:
	cd docs/_build/html && python -m http.server 8000

# Demo and examples
demo:
	python -m src.cli demo

generate-sample-data:
	python -m src.cli generate-data --dataset-type classification --complexity medium --output sample_data.csv
	python -m src.cli generate-data --dataset-type regression --complexity high --output sample_regression.csv

# Jupyter notebooks
notebooks:
	jupyter lab --notebook-dir=notebooks --no-browser

# Docker commands (if using Docker)
docker-build:
	docker build -t sklearn-mastery .

docker-run:
	docker run -it --rm -v $(PWD):/workspace sklearn-mastery

# Package distribution
build:
	python setup.py sdist bdist_wheel

upload-test:
	twine upload --repository testpypi dist/*

upload:
	twine upload dist/*

# Security checks
security:
	safety check
	bandit -r src/

# Performance profiling
profile:
	python -m cProfile -o profile_results.prof -m src.cli demo
	snakeviz profile_results.prof

# Git hooks setup
setup-hooks:
	pre-commit install
	pre-commit install --hook-type commit-msg

# Environment setup
setup-env:
	python -m venv venv
	@echo "Activate virtual environment with:"
	@echo "source venv/bin/activate  # On Unix/macOS"
	@echo "venv\\Scripts\\activate     # On Windows"

# CI simulation
ci-local:
	@echo "Running local CI simulation..."
	$(MAKE) format-check
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) test
	@echo "✅ Local CI simulation passed!"

# Project statistics
stats:
	@echo "Project Statistics:"
	@echo "==================="
	@echo "Lines of code:"
	find src/ -name "*.py" -exec wc -l {} + | tail -1
	@echo ""
	@echo "Number of Python files:"
	find src/ -name "*.py" | wc -l
	@echo ""
	@echo "Test coverage:"
	pytest tests/ --cov=src --cov-report=term-missing | grep TOTAL

# Quick start for new contributors
quickstart: setup-env install-dev
	@echo ""
	@echo "🎉 Quick start complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Activate virtual environment: source venv/bin/activate"
	@echo "2. Run demo: make demo"
	@echo "3. Run tests: make test"
	@echo "4. Start notebooks: make notebooks"
	@echo ""
	@echo "For more commands: make help"