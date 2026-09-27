"""Reproducible benchmarking and rigorous statistical model comparison.

Typical workflow::

    from sklearn_mastery.research import BenchmarkSuite, friedman_test, nemenyi_critical_difference

    result = BenchmarkSuite(estimators, datasets, scoring="accuracy", n_repeats=3).run()
    scores = result.score_matrix("accuracy")
    fr = friedman_test(scores)
    cd = nemenyi_critical_difference(fr.n_estimators, fr.n_datasets)
"""

from sklearn_mastery.research.benchmark import BenchmarkResult, BenchmarkSuite, DatasetSpec
from sklearn_mastery.research.bias_variance import BiasVarianceResult, bias_variance_decomposition
from sklearn_mastery.research.calibration import (
    BrierDecomposition,
    CalibrationBins,
    brier_score_decomposition,
    compute_calibration_bins,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_diagram,
)
from sklearn_mastery.research.comparison import (
    BayesianComparison,
    FriedmanResult,
    bayesian_correlated_ttest,
    corrected_resampled_ttest,
    friedman_test,
    holm_correction,
    nemenyi_critical_difference,
    nemenyi_posthoc,
    plot_critical_difference_diagram,
    wilcoxon_holm,
)
from sklearn_mastery.research.reporting import format_mean_std_table, results_to_latex, results_to_markdown
from sklearn_mastery.research.reproducibility import (
    RunManifest,
    capture_environment,
    config_hash,
    get_git_revision,
    set_global_seed,
    to_jsonable,
)

__all__ = [
    "BayesianComparison",
    "BenchmarkResult",
    "BenchmarkSuite",
    "BiasVarianceResult",
    "BrierDecomposition",
    "CalibrationBins",
    "DatasetSpec",
    "FriedmanResult",
    "RunManifest",
    "bayesian_correlated_ttest",
    "bias_variance_decomposition",
    "brier_score_decomposition",
    "capture_environment",
    "compute_calibration_bins",
    "config_hash",
    "corrected_resampled_ttest",
    "expected_calibration_error",
    "format_mean_std_table",
    "friedman_test",
    "get_git_revision",
    "holm_correction",
    "maximum_calibration_error",
    "nemenyi_critical_difference",
    "nemenyi_posthoc",
    "plot_critical_difference_diagram",
    "reliability_diagram",
    "results_to_latex",
    "results_to_markdown",
    "set_global_seed",
    "to_jsonable",
    "wilcoxon_holm",
]
