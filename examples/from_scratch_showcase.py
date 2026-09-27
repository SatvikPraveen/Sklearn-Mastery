"""Showcase: from-scratch tree ensembles versus their scikit-learn / XGBoost counterparts.

Trains every estimator in :mod:`sklearn_mastery.from_scratch` and its reference
implementation on identical train/validation splits, prints a comparison table
(accuracy or R-squared, fit time) and saves a figure with

* random-forest feature importances, scratch versus scikit-learn, side by side;
* staged train/validation loss of the scratch gradient-boosting classifier;
* staged train/validation log-loss of the scratch XGBoost-style classifier.

Run from the repository root::

    .venv/bin/python examples/from_scratch_showcase.py

The whole script runs in well under a minute and never opens a window
(matplotlib's Agg backend is forced).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import accuracy_score, log_loss, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn_mastery.from_scratch import (  # noqa: E402
    AdaBoostClassifierScratch,
    AdaBoostRegressorScratch,
    BaggingClassifierScratch,
    BaggingRegressorScratch,
    DecisionTreeClassifierScratch,
    DecisionTreeRegressorScratch,
    GradientBoostingClassifierScratch,
    GradientBoostingRegressorScratch,
    RandomForestClassifierScratch,
    RandomForestRegressorScratch,
    XGBoostClassifierScratch,
    XGBoostRegressorScratch,
)

try:
    import xgboost

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover - xgboost is optional
    HAS_XGBOOST = False

FIGURE_PATH = REPO_ROOT / "results" / "figures" / "examples" / "from_scratch_showcase.png"
SEED = 0
N_JOBS = 4

# Fixed categorical palette (slot 1 = scratch, slot 2 = reference), text in ink tokens.
COLOR_SCRATCH = "#2a78d6"
COLOR_REFERENCE = "#eb6834"
COLOR_TEXT = "#0b0b0b"
COLOR_MUTED = "#52514e"
COLOR_GRID = "#e6e5e1"


def timed_fit(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray, **fit_kwargs) -> float:
    """Fit ``estimator`` in place and return the wall-clock time in seconds."""
    start = time.perf_counter()
    estimator.fit(X, y, **fit_kwargs)
    return time.perf_counter() - start


def reference_booster(classifier: bool) -> Tuple[str, BaseEstimator]:
    """The real XGBoost when available, otherwise scikit-learn's histogram booster."""
    if HAS_XGBOOST:
        cls = xgboost.XGBClassifier if classifier else xgboost.XGBRegressor
        return "xgboost", cls(
            n_estimators=100, max_depth=3, learning_rate=0.3, tree_method="exact", random_state=SEED
        )
    cls = HistGradientBoostingClassifier if classifier else HistGradientBoostingRegressor
    return "sklearn (HistGB)", cls(max_iter=100, max_depth=3, learning_rate=0.3, random_state=SEED)


def build_pairs(classifier: bool) -> List[Tuple[str, BaseEstimator, str, BaseEstimator]]:
    """Return ``(name, scratch estimator, reference library, reference estimator)`` tuples."""
    ref_name, ref_booster = reference_booster(classifier)
    if classifier:
        return [
            (
                "Decision tree (depth 6)",
                DecisionTreeClassifierScratch(max_depth=6),
                "sklearn",
                DecisionTreeClassifier(max_depth=6, random_state=SEED),
            ),
            (
                "Bagging (20 trees)",
                BaggingClassifierScratch(n_estimators=20, random_state=SEED, n_jobs=N_JOBS),
                "sklearn",
                BaggingClassifier(n_estimators=20, random_state=SEED, n_jobs=N_JOBS),
            ),
            (
                "Random forest (50 trees)",
                RandomForestClassifierScratch(n_estimators=50, random_state=SEED, n_jobs=N_JOBS),
                "sklearn",
                RandomForestClassifier(n_estimators=50, random_state=SEED, n_jobs=N_JOBS),
            ),
            (
                "AdaBoost SAMME (50 stumps)",
                AdaBoostClassifierScratch(n_estimators=50, random_state=SEED),
                "sklearn",
                AdaBoostClassifier(n_estimators=50, random_state=SEED),
            ),
            (
                "Gradient boosting (100 x depth 3)",
                GradientBoostingClassifierScratch(n_estimators=100, random_state=SEED),
                "sklearn",
                GradientBoostingClassifier(n_estimators=100, random_state=SEED),
            ),
            (
                "XGBoost-style (100 x depth 3)",
                XGBoostClassifierScratch(n_estimators=100, max_depth=3, random_state=SEED),
                ref_name,
                ref_booster,
            ),
        ]
    return [
        (
            "Decision tree (depth 6)",
            DecisionTreeRegressorScratch(max_depth=6),
            "sklearn",
            DecisionTreeRegressor(max_depth=6, random_state=SEED),
        ),
        (
            "Bagging (20 trees)",
            BaggingRegressorScratch(n_estimators=20, random_state=SEED, n_jobs=N_JOBS),
            "sklearn",
            BaggingRegressor(n_estimators=20, random_state=SEED, n_jobs=N_JOBS),
        ),
        (
            "Random forest (50 trees)",
            RandomForestRegressorScratch(n_estimators=50, random_state=SEED, n_jobs=N_JOBS),
            "sklearn",
            RandomForestRegressor(n_estimators=50, random_state=SEED, n_jobs=N_JOBS),
        ),
        (
            "AdaBoost.R2 (50 x depth 3)",
            AdaBoostRegressorScratch(n_estimators=50, random_state=SEED),
            "sklearn",
            AdaBoostRegressor(
                estimator=DecisionTreeRegressor(max_depth=3), n_estimators=50, random_state=SEED
            ),
        ),
        (
            "Gradient boosting (100 x depth 3)",
            GradientBoostingRegressorScratch(n_estimators=100, random_state=SEED),
            "sklearn",
            GradientBoostingRegressor(n_estimators=100, random_state=SEED),
        ),
        (
            "XGBoost-style (100 x depth 3)",
            XGBoostRegressorScratch(n_estimators=100, max_depth=3, random_state=SEED),
            ref_name,
            ref_booster,
        ),
    ]


def run_comparison(classifier: bool, data) -> dict:
    """Fit every pair, print the table and return the fitted models keyed by name."""
    X_tr, X_va, y_tr, y_va = data
    metric: Callable = accuracy_score if classifier else r2_score
    metric_name = "accuracy" if classifier else "R^2"
    task = "Classification" if classifier else "Regression"
    print(f"\n{task}: {X_tr.shape[0]} train / {X_va.shape[0]} validation rows, {X_tr.shape[1]} features")
    header = f"{'Model':<36} {'scratch ' + metric_name:>18} {'fit s':>7}   {'reference ' + metric_name:>20} {'fit s':>7}  {'ref lib':<16}"
    print(header)
    print("-" * len(header))
    fitted = {}
    for name, scratch, ref_lib, reference in build_pairs(classifier):
        t_scratch = timed_fit(scratch, X_tr, y_tr)
        t_ref = timed_fit(reference, X_tr, y_tr)
        s_score = metric(y_va, scratch.predict(X_va))
        r_score = metric(y_va, reference.predict(X_va))
        print(
            f"{name:<36} {s_score:>18.4f} {t_scratch:>7.2f}   {r_score:>20.4f} {t_ref:>7.2f}  {ref_lib:<16}"
        )
        fitted[name] = (scratch, reference)
    return fitted


def staged_log_loss(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([log_loss(y, proba) for proba in model.staged_predict_proba(X)])


def style_axes(ax) -> None:
    ax.set_facecolor("#fcfcfb")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.title.set_color(COLOR_TEXT)
    ax.xaxis.label.set_color(COLOR_MUTED)
    ax.yaxis.label.set_color(COLOR_MUTED)


def make_figure(clf_models: dict, clf_data) -> None:
    """Importances side by side, GB staged loss and XGB staged loss."""
    X_tr, X_va, y_tr, y_va = clf_data
    rf_scratch, rf_ref = clf_models["Random forest (50 trees)"]
    gb_scratch, _ = clf_models["Gradient boosting (100 x depth 3)"]

    # Re-fit the scratch XGB with an eval set so evals_result_ holds both curves.
    xgb_scratch = XGBoostClassifierScratch(n_estimators=100, max_depth=3, random_state=SEED)
    xgb_scratch.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    xgb_train = xgb_scratch.evals_result_["train"]["logloss"]
    xgb_val = xgb_scratch.evals_result_["validation"]["logloss"]

    gb_train = staged_log_loss(gb_scratch, X_tr, y_tr)
    gb_val = staged_log_loss(gb_scratch, X_va, y_va)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#fcfcfb")

    # Panel 1: RF importances (top 10 by scratch MDI).
    ax = axes[0]
    order = np.argsort(rf_scratch.feature_importances_)[::-1][:10]
    positions = np.arange(len(order))
    height = 0.38
    ax.barh(
        positions - height / 2,
        rf_scratch.feature_importances_[order],
        height=height,
        color=COLOR_SCRATCH,
        label="scratch RF",
        edgecolor="#fcfcfb",
        linewidth=1,
    )
    ax.barh(
        positions + height / 2,
        rf_ref.feature_importances_[order],
        height=height,
        color=COLOR_REFERENCE,
        label="sklearn RF",
        edgecolor="#fcfcfb",
        linewidth=1,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels([f"feature {i}" for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("mean decrease in impurity")
    ax.set_title("Random forest feature importances (top 10)", fontsize=11, loc="left")
    ax.xaxis.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, labelcolor=COLOR_TEXT)
    style_axes(ax)
    ax.yaxis.grid(False)

    # Panel 2 and 3: staged losses.
    for ax, train, val, title in (
        (axes[1], gb_train, gb_val, "Gradient boosting (scratch): binomial deviance"),
        (axes[2], xgb_train, xgb_val, "XGBoost-style (scratch): log-loss"),
    ):
        rounds = np.arange(1, len(train) + 1)
        ax.plot(rounds, train, color=COLOR_SCRATCH, linewidth=2, label="train")
        ax.plot(rounds, val, color=COLOR_REFERENCE, linewidth=2, label="validation")
        best = int(np.argmin(val))
        ax.scatter([rounds[best]], [val[best]], s=36, color=COLOR_REFERENCE, edgecolor="#fcfcfb", zorder=3)
        on_right = best > 0.7 * len(val)
        ax.annotate(
            f"best round {rounds[best]}: {val[best]:.3f}",
            (rounds[best], val[best]),
            textcoords="offset points",
            xytext=(-8, 8) if on_right else (8, 8),
            ha="right" if on_right else "left",
            fontsize=9,
            color=COLOR_TEXT,
        )
        ax.set_xlabel("boosting round")
        ax.set_ylabel("log-loss")
        ax.set_title(title, fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=9, labelcolor=COLOR_TEXT)
        style_axes(ax)

    fig.suptitle(
        "From-scratch tree ensembles versus reference implementations",
        fontsize=13,
        color=COLOR_TEXT,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=130)
    plt.close(fig)
    print(f"\nFigure saved to {FIGURE_PATH.relative_to(REPO_ROOT)}")


def main() -> None:
    start = time.perf_counter()
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10, n_redundant=4, flip_y=0.03, random_state=SEED
    )
    clf_data = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    X, y = make_regression(n_samples=1000, n_features=20, n_informative=10, noise=10.0, random_state=SEED)
    reg_data = train_test_split(X, y, test_size=0.25, random_state=SEED)

    clf_models = run_comparison(classifier=True, data=clf_data)
    run_comparison(classifier=False, data=reg_data)
    make_figure(clf_models, clf_data)
    print(f"Total wall time: {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
