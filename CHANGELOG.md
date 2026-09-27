# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/).

## [2.0.0] - 2026-09-27

### Added
- `sklearn_mastery.research`: reproducible `BenchmarkSuite` with shared
  repeated stratified splits and optional nested tuning; Friedman /
  Iman-Davenport, Nemenyi, Wilcoxon-Holm, Nadeau-Bengio corrected t-test and
  Bayesian correlated t-test; critical-difference diagrams; calibration
  diagnostics (ECE, MCE, Brier decomposition, reliability diagrams);
  bootstrap bias-variance decomposition; Markdown/LaTeX result tables;
  environment capture and hashed run manifests.
- `sklearn_mastery.data`: synthetic data generators, `DataPreprocessor`,
  `CategoricalEncoder`, `NumericalTransformer`, `DataValidator`.
- `sklearn_mastery.models`: uniform sklearn-compatible wrappers for
  classification, regression, clustering, dimensionality reduction and
  ensembles, with optional XGBoost / LightGBM / UMAP backends.
- GitHub Actions CI (lint, type-check, multi-version tests, build),
  `CITATION.cff`, ruff-based pre-commit configuration.

### Changed
- Package renamed from `src` to `sklearn_mastery`; `config/` moved inside the
  package. Packaging migrated from `setup.py` to `pyproject.toml`.
- Importing the package no longer configures logging or creates directories.
  `settings.PROJECT_ROOT` defaults to the working directory and can be set
  with `SKLEARN_MASTERY_ROOT`.
- Pipelines, evaluation and visualization modules refactored so that the test
  suite is the API specification and passes.

### Fixed
- `.gitignore` pattern that hid the package's `models/` directory.
- Commit attribution normalised to a single author identity.

## [1.0.0]
- Initial learning-oriented framework.
