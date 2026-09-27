"""Command-line interface for sklearn-mastery.

Commands
--------
``generate-data``   Write a synthetic dataset to CSV.
``train``           Train and evaluate a named estimator on a CSV file.
``benchmark``       Run a reproducible multi-estimator benchmark and save results.
``compare``         Statistically compare estimators from a saved benchmark.
``info``            Print the captured execution environment.
``launch-notebooks`` Start JupyterLab in the notebooks directory.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import click
import numpy as np
import pandas as pd

from sklearn_mastery import __version__
from sklearn_mastery.config.logging_config import get_logger, setup_logging
from sklearn_mastery.config.settings import settings

_BUILTIN_DATASETS = {
    "iris": ("sklearn.datasets", "load_iris", "classification"),
    "wine": ("sklearn.datasets", "load_wine", "classification"),
    "breast_cancer": ("sklearn.datasets", "load_breast_cancer", "classification"),
    "digits": ("sklearn.datasets", "load_digits", "classification"),
    "diabetes": ("sklearn.datasets", "load_diabetes", "regression"),
}


def _load_csv(path: str, target: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if target not in df.columns:
        raise click.ClickException(
            f"target column {target!r} not found in {path}; columns: {list(df.columns)}"
        )
    X = df.drop(columns=[target]).to_numpy()
    y = df[target].to_numpy()
    return X, y


def _load_builtin(name: str) -> Tuple[np.ndarray, np.ndarray, str]:
    import importlib

    module, func, task = _BUILTIN_DATASETS[name]
    X, y = getattr(importlib.import_module(module), func)(return_X_y=True)
    return X, y, task


def _default_estimators(task: str) -> Dict[str, object]:
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, SVR

    seed = settings.RANDOM_SEED
    if task == "classification":
        return {
            "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            "svm_rbf": make_pipeline(StandardScaler(), SVC()),
            "knn": make_pipeline(StandardScaler(), KNeighborsClassifier()),
            "random_forest": RandomForestClassifier(n_estimators=200, random_state=seed),
            "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        }
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge()),
        "svr": make_pipeline(StandardScaler(), SVR()),
        "knn": make_pipeline(StandardScaler(), KNeighborsRegressor()),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=seed),
        "gradient_boosting": GradientBoostingRegressor(random_state=seed),
    }


@click.group()
@click.version_option(__version__, prog_name="sklearn-mastery")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.option("--log-file", type=click.Path(dir_okay=False), default=None, help="Also log to this file.")
def cli(verbose: bool, log_file: Optional[str]) -> None:
    """sklearn-mastery: reproducible scikit-learn experiments."""
    setup_logging(log_level="DEBUG" if verbose else "INFO", log_file=log_file, rich_console=True)


@cli.command("generate-data")
@click.option(
    "--dataset-type",
    type=click.Choice(["classification", "regression", "clustering"]),
    default="classification",
    show_default=True,
)
@click.option("--n-samples", default=1000, show_default=True)
@click.option("--n-features", default=20, show_default=True)
@click.option(
    "--complexity", type=click.Choice(["linear", "medium", "high"]), default="medium", show_default=True
)
@click.option("--seed", default=None, type=int, help="Random seed (defaults to settings.RANDOM_SEED).")
@click.option(
    "--output", "-o", type=click.Path(dir_okay=False), default="generated_data.csv", show_default=True
)
def generate_data(
    dataset_type: str, n_samples: int, n_features: int, complexity: str, seed: Optional[int], output: str
) -> None:
    """Generate a synthetic dataset and write it to CSV."""
    from sklearn_mastery.data.generators import SyntheticDataGenerator

    gen = SyntheticDataGenerator(random_state=settings.RANDOM_SEED if seed is None else seed)
    y: Optional[np.ndarray]
    if dataset_type == "regression":
        if complexity == "linear":
            X, y = gen.linear_regression_data(n_samples=n_samples, n_features=n_features)
        else:
            X, y, _ = gen.regression_with_collinearity(n_samples=n_samples, n_features=n_features)
    elif dataset_type == "classification":
        X, y = gen.classification_complexity_spectrum(complexity, n_samples=n_samples, n_features=n_features)
    else:
        X = gen.clustering_blobs_with_noise(n_samples=n_samples, n_features=n_features)
        y = None

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    if y is not None:
        df["target"] = y
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    click.echo(f"Wrote {dataset_type} dataset with shape {X.shape} to {out}")


@cli.command()
@click.argument("data_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--target", default="target", show_default=True, help="Target column name.")
@click.option("--algorithm", default="random_forest", show_default=True)
@click.option("--task-type", type=click.Choice(["classification", "regression"]), default="classification")
@click.option("--test-size", default=0.2, show_default=True)
@click.option("--cv-folds", default=5, show_default=True)
@click.option("--output-dir", "-o", type=click.Path(file_okay=False), default=None)
def train(
    data_file: str,
    target: str,
    algorithm: str,
    task_type: str,
    test_size: float,
    cv_folds: int,
    output_dir: Optional[str],
) -> None:
    """Train a named estimator on a CSV file and report hold-out and CV scores."""
    from sklearn.model_selection import cross_val_score, train_test_split

    from sklearn_mastery.research.reproducibility import RunManifest, to_jsonable

    logger = get_logger("cli")
    X, y = _load_csv(data_file, target)
    if task_type == "classification":
        from sklearn_mastery.models.supervised.classification import ClassificationModels

        model = ClassificationModels().get_model(algorithm, random_state=settings.RANDOM_SEED)
    else:
        from sklearn_mastery.models.supervised.regression import RegressionModels

        model = RegressionModels().get_model(algorithm, random_state=settings.RANDOM_SEED)

    stratify = y if task_type == "classification" else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=settings.RANDOM_SEED, stratify=stratify
    )
    logger.info("Training %s on %d samples", algorithm, len(y_tr))
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv_folds)
    model.fit(X_tr, y_tr)
    holdout = float(model.score(X_te, y_te))
    metrics = {
        "algorithm": algorithm,
        "task_type": task_type,
        "holdout_score": holdout,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std(ddof=1)) if cv_folds > 1 else 0.0,
        "cv_scores": cv_scores.tolist(),
        "n_train": len(y_tr),
        "n_test": len(y_te),
    }
    click.echo(json.dumps(metrics, indent=2))

    if output_dir:
        import joblib

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out / f"{algorithm}.joblib")
        (out / f"{algorithm}_metrics.json").write_text(json.dumps(to_jsonable(metrics), indent=2))
        RunManifest(name=f"train-{algorithm}", config={"metrics": metrics, "data_file": data_file}).save(
            out / f"{algorithm}_manifest.json"
        )
        click.echo(f"Saved model, metrics and manifest to {out}")


@cli.command()
@click.option(
    "--dataset",
    "datasets",
    multiple=True,
    type=click.Choice(sorted(_BUILTIN_DATASETS)),
    help="Built-in dataset(s); repeatable. Defaults to iris, wine, breast_cancer.",
)
@click.option("--csv", "csv_files", multiple=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--target", default="target", show_default=True, help="Target column for CSV datasets.")
@click.option("--task-type", type=click.Choice(["classification", "regression"]), default="classification")
@click.option("--scoring", multiple=True, help="Scorer name(s); repeatable.")
@click.option("--n-splits", default=5, show_default=True)
@click.option("--n-repeats", default=3, show_default=True)
@click.option("--n-jobs", default=1, show_default=True)
@click.option("--seed", default=None, type=int)
@click.option("--quick", is_flag=True, help="Small protocol (3 folds, 1 repeat) for smoke testing.")
@click.option("--output-dir", "-o", type=click.Path(file_okay=False), default=None)
def benchmark(
    datasets: Tuple[str, ...],
    csv_files: Tuple[str, ...],
    target: str,
    task_type: str,
    scoring: Tuple[str, ...],
    n_splits: int,
    n_repeats: int,
    n_jobs: int,
    seed: Optional[int],
    quick: bool,
    output_dir: Optional[str],
) -> None:
    """Benchmark the default estimator zoo on built-in and/or CSV datasets."""
    from sklearn_mastery.research import BenchmarkSuite, friedman_test, results_to_markdown

    if quick:
        n_splits, n_repeats = 3, 1
    if not datasets and not csv_files:
        datasets = ("iris", "wine", "breast_cancer") if task_type == "classification" else ("diabetes",)

    data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name in datasets:
        X, y, task = _load_builtin(name)
        if task != task_type:
            raise click.ClickException(f"dataset {name!r} is a {task} problem, not {task_type}")
        data[name] = (X, y)
    for path in csv_files:
        data[Path(path).stem] = _load_csv(path, target)

    scorers = list(scoring) or (["accuracy", "f1_macro"] if task_type == "classification" else ["r2"])
    suite = BenchmarkSuite(
        estimators=_default_estimators(task_type),
        datasets=data,
        scoring=scorers,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=settings.RANDOM_SEED if seed is None else seed,
        n_jobs=n_jobs,
        name="cli-benchmark",
    )
    result = suite.run()
    primary = scorers[0]
    click.echo(results_to_markdown(result.results, primary, caption=f"{primary} (mean ± std over folds)"))
    if len(data) >= 2:
        fr = friedman_test(result.score_matrix(primary))
        click.echo(
            f"Friedman (Iman-Davenport) F = {fr.iman_davenport_statistic:.3f}, "
            f"p = {fr.iman_davenport_p_value:.4f} -> "
            + ("reject equal performance" if fr.reject() else "cannot reject equal performance")
        )
    if output_dir:
        out = result.save(output_dir)
        click.echo(f"Saved results, manifest and best params to {out}")


@cli.command()
@click.argument("results_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--metric", default=None, help="Metric to compare (defaults to the first in the results).")
@click.option("--alpha", default=0.05, show_default=True)
@click.option("--rope", default=0.01, show_default=True, help="ROPE half-width for the Bayesian test.")
@click.option("--cd-diagram", type=click.Path(dir_okay=False), default=None, help="Save a CD diagram PNG.")
def compare(
    results_dir: str, metric: Optional[str], alpha: float, rope: float, cd_diagram: Optional[str]
) -> None:
    """Statistically compare estimators from a saved benchmark directory."""
    from sklearn_mastery.research import (
        BenchmarkResult,
        bayesian_correlated_ttest,
        friedman_test,
        nemenyi_critical_difference,
        plot_critical_difference_diagram,
        wilcoxon_holm,
    )

    result = BenchmarkResult.load(results_dir)
    metric = metric or str(result.results["metric"].iloc[0])
    scores = result.score_matrix(metric)
    click.echo(f"Metric: {metric}\n")
    click.echo(scores.round(4).to_string())

    if scores.shape[0] >= 2:
        fr = friedman_test(scores)
        click.echo("\nAverage ranks (1 = best):")
        click.echo(fr.average_ranks.sort_values().round(3).to_string())
        click.echo(
            f"\nFriedman chi2 = {fr.statistic:.3f} (p = {fr.p_value:.4f}); "
            f"Iman-Davenport F = {fr.iman_davenport_statistic:.3f} (p = {fr.iman_davenport_p_value:.4f})"
        )
        cd = nemenyi_critical_difference(fr.n_estimators, fr.n_datasets, alpha)
        click.echo(f"Nemenyi critical difference at alpha={alpha}: {cd:.3f}")
        click.echo("\nPairwise Wilcoxon signed-rank with Holm correction:")
        click.echo(wilcoxon_holm(scores, alpha=alpha).round(4).to_string(index=False))
        if cd_diagram:
            import matplotlib

            matplotlib.use("Agg")
            ax = plot_critical_difference_diagram(fr.average_ranks, cd, title=f"{metric} (alpha={alpha})")
            ax.figure.savefig(cd_diagram, dpi=150, bbox_inches="tight")
            click.echo(f"\nSaved critical-difference diagram to {cd_diagram}")
    else:
        dataset = scores.index[0]
        paired = result.paired_scores(metric, dataset)
        n_splits = int(result.results["fold"].max()) + 1
        best = paired.mean().idxmax()
        click.echo(f"\nSingle dataset: Bayesian correlated t-test vs. best estimator ({best}), ROPE = {rope}")
        for other in paired.columns:
            if other == best:
                continue
            bc = bayesian_correlated_ttest(paired[best], paired[other], rope=rope, n_splits=n_splits)
            click.echo(
                f"  {best} vs {other}: P(left)={bc.p_left:.3f} P(rope)={bc.p_rope:.3f} "
                f"P(right)={bc.p_right:.3f} -> {bc.decision()}"
            )


@cli.command()
def info() -> None:
    """Print the captured execution environment as JSON."""
    from sklearn_mastery.research.reproducibility import capture_environment

    click.echo(json.dumps(capture_environment(), indent=2))


@cli.command("launch-notebooks")
@click.option("--port", default=8888, show_default=True)
@click.option("--notebook-dir", default="notebooks", show_default=True)
def launch_notebooks(port: int, notebook_dir: str) -> None:
    """Start JupyterLab in the notebooks directory."""
    path = Path(notebook_dir)
    if not path.exists():
        raise click.ClickException(f"notebook directory not found: {path}")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "jupyterlab",
                "--port",
                str(port),
                "--notebook-dir",
                str(path),
                "--no-browser",
            ],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise click.ClickException(
            f"failed to launch JupyterLab ({exc}); install with 'pip install jupyterlab'"
        ) from exc
    except KeyboardInterrupt:
        click.echo("JupyterLab stopped")


def main() -> None:
    """Console-script entry point."""
    cli()


if __name__ == "__main__":
    main()
