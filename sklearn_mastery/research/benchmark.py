"""Reproducible multi-dataset, multi-estimator benchmarking.

The :class:`BenchmarkSuite` evaluates a set of estimators on a set of datasets
under a fixed resampling protocol (repeated, stratified k-fold by default) and
returns tidy, long-form results that feed directly into the statistical
comparison tools in :mod:`sklearn_mastery.research.comparison`.

Design principles
-----------------
* **Tidy output.** One row per (dataset, estimator, repeat, fold, metric).
* **Identical splits for every estimator.** Splits are generated once per
  dataset and repeat, so paired tests are valid.
* **Provenance.** Every run carries a :class:`RunManifest` with seeds,
  configuration hash and environment snapshot.
* **Optional nested tuning.** Pass ``param_grids`` to tune each estimator
  inside every outer fold, which is the only unbiased way to report tuned
  performance (Cawley & Talbot, 2010).

References
----------
Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection
and subsequent selection bias in performance evaluation. *JMLR*, 11, 2079-2107.
Demšar, J. (2006). Statistical comparisons of classifiers over multiple data
sets. *JMLR*, 7, 1-30.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.metrics import check_scoring, get_scorer
from sklearn.model_selection import (
    BaseCrossValidator,
    GridSearchCV,
    RepeatedKFold,
    RepeatedStratifiedKFold,
)

from sklearn_mastery.config.logging_config import LoggerMixin
from sklearn_mastery.research.reproducibility import RunManifest, to_jsonable

__all__ = ["BenchmarkResult", "BenchmarkSuite", "DatasetSpec"]

DatasetLike = Union[Tuple[np.ndarray, np.ndarray], Callable[[], Tuple[np.ndarray, np.ndarray]]]


@dataclass
class DatasetSpec:
    """A named dataset, given eagerly as ``(X, y)`` or lazily as a callable."""

    name: str
    loader: DatasetLike
    _cache: Optional[Tuple[np.ndarray, np.ndarray]] = field(default=None, repr=False)

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._cache is None:
            data = self.loader() if callable(self.loader) else self.loader
            X, y = data
            self._cache = (np.asarray(X), np.asarray(y))
        return self._cache


@dataclass
class BenchmarkResult:
    """Container for benchmark output.

    Attributes:
        results: Long-form DataFrame with columns
            ``dataset, estimator, repeat, fold, metric, value, fit_time, score_time``.
        manifest: Provenance manifest.
        best_params: Nested dict ``{dataset: {estimator: [params per fold]}}``
            populated when nested tuning was used.
    """

    results: pd.DataFrame
    manifest: RunManifest
    best_params: Dict[str, Dict[str, List[Dict[str, Any]]]] = field(default_factory=dict)

    # ---------------------------------------------------------------- views
    def summary(self, metric: Optional[str] = None) -> pd.DataFrame:
        """Mean ± std of each metric per (dataset, estimator).

        Args:
            metric: Restrict to one metric. Defaults to all metrics.

        Returns:
            DataFrame indexed by dataset with a column per estimator when a
            single metric is requested, otherwise a multi-index table.
        """
        df = self.results if metric is None else self.results[self.results["metric"] == metric]
        agg = df.groupby(["dataset", "estimator", "metric"])["value"].agg(["mean", "std", "count"])
        if metric is None:
            return agg
        return agg.reset_index().pivot(index="dataset", columns="estimator", values="mean")

    def score_matrix(self, metric: str, aggregate: Union[str, Callable] = "mean") -> pd.DataFrame:
        """Dataset x estimator matrix of aggregated fold scores for one metric.

        This is the input expected by the Friedman/Nemenyi procedures.
        """
        df = self.results[self.results["metric"] == metric]
        if df.empty:
            raise KeyError(
                f"metric {metric!r} not found; available: {sorted(self.results['metric'].unique())}"
            )
        return df.pivot_table(index="dataset", columns="estimator", values="value", aggfunc=aggregate)  # type: ignore[arg-type]

    def paired_scores(self, metric: str, dataset: str) -> pd.DataFrame:
        """(repeat, fold) x estimator matrix of raw fold scores for one dataset.

        Rows are aligned across estimators because the suite shares splits,
        making the output valid input for paired tests.
        """
        df = self.results[(self.results["metric"] == metric) & (self.results["dataset"] == dataset)]
        return df.pivot_table(index=["repeat", "fold"], columns="estimator", values="value")

    def rank_table(self, metric: str, higher_is_better: bool = True) -> pd.DataFrame:
        """Per-dataset ranks (1 = best) and the average rank per estimator."""
        scores = self.score_matrix(metric)
        ranks = scores.rank(axis=1, ascending=not higher_is_better, method="average")
        ranks.loc["average_rank"] = ranks.mean(axis=0)
        return ranks

    # ------------------------------------------------------------------ io
    def save(self, directory: Union[str, Path]) -> Path:
        """Persist results (CSV), manifest (JSON) and best params (JSON)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(directory / "results.csv", index=False)
        self.manifest.save(directory / "manifest.json")
        (directory / "best_params.json").write_text(json.dumps(to_jsonable(self.best_params), indent=2))
        return directory

    @classmethod
    def load(cls, directory: Union[str, Path]) -> BenchmarkResult:
        directory = Path(directory)
        results = pd.read_csv(directory / "results.csv")
        manifest = RunManifest.load(directory / "manifest.json")
        best_params_path = directory / "best_params.json"
        best_params = json.loads(best_params_path.read_text()) if best_params_path.exists() else {}
        return cls(results=results, manifest=manifest, best_params=best_params)


def _evaluate_fold(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    scorers: Mapping[str, Callable],
    param_grid: Optional[Mapping[str, Sequence[Any]]],
    inner_cv: int,
    tuning_scoring: Optional[str],
    seed: int,
) -> Tuple[Dict[str, float], float, float, Optional[Dict[str, Any]]]:
    """Fit one estimator on one split and score it with every scorer."""
    est = clone(estimator)
    if hasattr(est, "random_state") and "random_state" in est.get_params():
        est.set_params(random_state=seed)
    X_tr, X_te, y_tr, y_te = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    best_params: Optional[Dict[str, Any]] = None
    t0 = time.perf_counter()
    if param_grid:
        search = GridSearchCV(est, param_grid, cv=inner_cv, scoring=tuning_scoring, n_jobs=1, refit=True)
        search.fit(X_tr, y_tr)
        est = search.best_estimator_
        best_params = dict(search.best_params_)
    else:
        est.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores = {name: float(scorer(est, X_te, y_te)) for name, scorer in scorers.items()}
    score_time = time.perf_counter() - t0
    return scores, fit_time, score_time, best_params


class BenchmarkSuite(LoggerMixin):
    """Evaluate many estimators on many datasets under one shared protocol.

    Args:
        estimators: Mapping of estimator name to (unfitted) estimator.
        datasets: Mapping of dataset name to ``(X, y)`` or a zero-argument
            loader returning ``(X, y)``.
        scoring: Scorer names or callables (``{name: scorer}``) or a single
            string. Defaults to accuracy for classifiers and R² for regressors.
        n_splits: Outer folds.
        n_repeats: Repetitions of the outer CV with different shuffles.
        cv: Explicit outer cross-validator (overrides ``n_splits``/``n_repeats``).
        param_grids: Optional ``{estimator name: grid}`` for nested tuning.
        inner_cv: Folds for the inner (tuning) loop.
        tuning_scoring: Scorer used for inner selection (defaults to the first
            entry of ``scoring``).
        random_state: Master seed; every split and estimator seed derives from it.
        n_jobs: Parallel jobs across (dataset, repeat, fold, estimator) tasks.
        name: Run name recorded in the manifest.

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from sklearn.datasets import load_iris, load_wine
        >>> suite = BenchmarkSuite(
        ...     estimators={"logreg": LogisticRegression(max_iter=500), "tree": DecisionTreeClassifier()},
        ...     datasets={"iris": load_iris(return_X_y=True), "wine": load_wine(return_X_y=True)},
        ...     scoring=["accuracy", "f1_macro"], n_splits=5, n_repeats=2, random_state=0)
        >>> result = suite.run()
        >>> result.score_matrix("accuracy").shape
        (2, 2)
    """

    def __init__(
        self,
        estimators: Mapping[str, BaseEstimator],
        datasets: Mapping[str, DatasetLike],
        scoring: Union[str, Iterable[str], Mapping[str, Callable], None] = None,
        n_splits: int = 5,
        n_repeats: int = 1,
        cv: Optional[BaseCrossValidator] = None,
        param_grids: Optional[Mapping[str, Mapping[str, Sequence[Any]]]] = None,
        inner_cv: int = 3,
        tuning_scoring: Optional[str] = None,
        random_state: int = 42,
        n_jobs: int = 1,
        name: str = "benchmark",
    ) -> None:
        if not estimators:
            raise ValueError("at least one estimator is required")
        if not datasets:
            raise ValueError("at least one dataset is required")
        self.estimators: Dict[str, BaseEstimator] = dict(estimators)
        self.datasets: List[DatasetSpec] = [DatasetSpec(name, loader) for name, loader in datasets.items()]
        self.scoring = scoring
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.cv = cv
        self.param_grids: Dict[str, Mapping[str, Sequence[Any]]] = dict(param_grids or {})
        self.inner_cv = inner_cv
        self.tuning_scoring = tuning_scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.name = name

        unknown = set(self.param_grids) - set(self.estimators)
        if unknown:
            raise KeyError(f"param_grids given for unknown estimators: {sorted(unknown)}")

    # ------------------------------------------------------------ helpers
    def _resolve_scorers(self, estimator: BaseEstimator) -> Dict[str, Callable]:
        scoring = self.scoring
        if scoring is None:
            return {"score": check_scoring(estimator)}
        if isinstance(scoring, str):
            return {scoring: get_scorer(scoring)}
        if isinstance(scoring, Mapping):
            return {k: (get_scorer(v) if isinstance(v, str) else v) for k, v in scoring.items()}
        return {s: get_scorer(s) for s in scoring}

    def _make_cv(self, estimator: BaseEstimator, repeat_seed: int) -> BaseCrossValidator:
        if self.cv is not None:
            return self.cv
        if is_classifier(estimator):
            return RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=1, random_state=repeat_seed)
        return RepeatedKFold(n_splits=self.n_splits, n_repeats=1, random_state=repeat_seed)

    def _manifest(self) -> RunManifest:
        config = {
            "estimators": {k: to_jsonable(v) for k, v in self.estimators.items()},
            "datasets": [d.name for d in self.datasets],
            "scoring": to_jsonable(self.scoring)
            if not isinstance(self.scoring, Mapping)
            else sorted(self.scoring),
            "n_splits": self.n_splits,
            "n_repeats": self.n_repeats,
            "cv": repr(self.cv) if self.cv is not None else None,
            "param_grids": to_jsonable(self.param_grids),
            "inner_cv": self.inner_cv,
            "tuning_scoring": self.tuning_scoring,
        }
        return RunManifest(name=self.name, config=config, seed=self.random_state)

    # ---------------------------------------------------------------- run
    def run(self) -> BenchmarkResult:
        """Execute the benchmark and return tidy results with provenance."""
        first_estimator = next(iter(self.estimators.values()))
        scorers = self._resolve_scorers(first_estimator)
        tuning_scoring = self.tuning_scoring or next(iter(scorers))
        rng = np.random.default_rng(self.random_state)

        tasks: List[Tuple[str, str, int, int, Any]] = []
        for spec in self.datasets:
            X, y = spec.load()
            for repeat in range(self.n_repeats):
                repeat_seed = int(rng.integers(0, 2**31 - 1))
                cv = self._make_cv(first_estimator, repeat_seed)
                splits = list(cv.split(X, y))
                for fold, (tr, te) in enumerate(splits):
                    for est_name in self.estimators:
                        tasks.append((spec.name, est_name, repeat, fold, (tr, te, repeat_seed)))

        self.logger.info(
            "Running %d fits (%d datasets x %d estimators x %d repeats x %d folds)",
            len(tasks),
            len(self.datasets),
            len(self.estimators),
            self.n_repeats,
            self.n_splits if self.cv is None else self.cv.get_n_splits(),
        )
        data_cache = {spec.name: spec.load() for spec in self.datasets}

        def _job(task):
            ds, est_name, _repeat, fold, (tr, te, seed) = task
            X, y = data_cache[ds]
            return task, _evaluate_fold(
                self.estimators[est_name],
                X,
                y,
                tr,
                te,
                scorers,
                self.param_grids.get(est_name),
                self.inner_cv,
                tuning_scoring,
                seed + fold,
            )

        outputs = Parallel(n_jobs=self.n_jobs)(delayed(_job)(t) for t in tasks)

        rows: List[Dict[str, Any]] = []
        best_params: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for (ds, est_name, repeat, fold, _), (scores, fit_time, score_time, params) in outputs:
            for metric, value in scores.items():
                rows.append(
                    {
                        "dataset": ds,
                        "estimator": est_name,
                        "repeat": repeat,
                        "fold": fold,
                        "metric": metric,
                        "value": value,
                        "fit_time": fit_time,
                        "score_time": score_time,
                    }
                )
            if params is not None:
                best_params.setdefault(ds, {}).setdefault(est_name, []).append(params)

        results = pd.DataFrame(rows).sort_values(["dataset", "estimator", "metric", "repeat", "fold"])
        results = results.reset_index(drop=True)
        return BenchmarkResult(results=results, manifest=self._manifest(), best_params=best_params)
