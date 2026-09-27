"""Tests for statistical comparison procedures (validated against known values)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from sklearn_mastery.research.comparison import (
    bayesian_correlated_ttest,
    corrected_resampled_ttest,
    friedman_test,
    holm_correction,
    nemenyi_critical_difference,
    nemenyi_posthoc,
    plot_critical_difference_diagram,
    wilcoxon_holm,
)


@pytest.fixture
def demsar_table():
    """Accuracy table from Demšar (2006), Table 6 (C4.5 vs. three variants, 14 datasets)."""
    data = {
        "C4.5": [
            0.763,
            0.599,
            0.954,
            0.628,
            0.882,
            0.936,
            0.661,
            0.583,
            0.775,
            1.000,
            0.940,
            0.619,
            0.972,
            0.957,
        ],
        "C4.5+m": [
            0.768,
            0.591,
            0.971,
            0.661,
            0.888,
            0.931,
            0.668,
            0.583,
            0.838,
            1.000,
            0.962,
            0.666,
            0.981,
            0.978,
        ],
        "C4.5+cf": [
            0.771,
            0.590,
            0.968,
            0.654,
            0.886,
            0.916,
            0.609,
            0.563,
            0.866,
            1.000,
            0.965,
            0.614,
            0.975,
            0.946,
        ],
        "C4.5+m+cf": [
            0.798,
            0.569,
            0.967,
            0.657,
            0.898,
            0.931,
            0.685,
            0.625,
            0.875,
            1.000,
            0.962,
            0.669,
            0.975,
            0.970,
        ],
    }
    return pd.DataFrame(data)


class TestFriedman:
    def test_iman_davenport_consistency(self, demsar_table):
        res = friedman_test(demsar_table)
        n, k = demsar_table.shape
        # average ranks must sum to k(k+1)/2 and the best variant must be C4.5+m+cf or C4.5+m
        assert res.average_ranks.sum() == pytest.approx(k * (k + 1) / 2)
        assert res.average_ranks.idxmin() in {"C4.5+m+cf", "C4.5+m"}
        chi2 = res.statistic
        f_expected = (n - 1) * chi2 / (n * (k - 1) - chi2)
        assert res.iman_davenport_statistic == pytest.approx(f_expected)
        assert res.iman_davenport_p_value == pytest.approx(stats.f.sf(f_expected, k - 1, (k - 1) * (n - 1)))
        assert res.n_datasets == n and res.n_estimators == k
        # the F correction is less conservative than the chi-square approximation
        assert res.iman_davenport_p_value < res.p_value

    def test_agrees_with_scipy_chi2_statistic(self):
        rng = np.random.default_rng(0)
        scores = rng.normal(size=(10, 3))
        res = friedman_test(scores)
        scipy_stat, scipy_p = stats.friedmanchisquare(*scores.T)
        assert res.statistic == pytest.approx(scipy_stat)
        assert res.p_value == pytest.approx(scipy_p)

    def test_lower_is_better_flips_ranks(self):
        scores = pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})
        assert friedman_test(scores, higher_is_better=True).average_ranks["b"] == 1
        assert friedman_test(scores, higher_is_better=False).average_ranks["a"] == 1

    def test_input_validation(self):
        with pytest.raises(ValueError):
            friedman_test(np.ones((1, 3)))
        with pytest.raises(ValueError):
            friedman_test(np.ones((5, 1)))
        with pytest.raises(ValueError):
            friedman_test(pd.DataFrame({"a": [1, np.nan], "b": [1, 2]}))


class TestNemenyi:
    def test_critical_difference_demsar(self):
        # Demšar: q_0.05 for k=4 is 2.569, CD = 2.569 * sqrt(4*5/(6*14)) = 1.25
        assert nemenyi_critical_difference(4, 14, alpha=0.05) == pytest.approx(1.25, abs=0.01)

    def test_q_alpha_table_values(self):
        # q_alpha values from Demšar (2006) Table 5, alpha=0.05
        expected = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 10: 3.164}
        for k, q in expected.items():
            cd = nemenyi_critical_difference(k, 1, alpha=0.05)
            assert cd / np.sqrt(k * (k + 1) / 6.0) == pytest.approx(q, abs=2e-3)

    def test_posthoc_symmetric_and_bounded(self, demsar_table):
        p = nemenyi_posthoc(demsar_table)
        assert p.shape == (4, 4)
        assert np.allclose(p.values, p.values.T)
        assert (p.values >= 0).all() and (p.values <= 1).all()
        assert np.allclose(np.diag(p.values), 1.0)
        # C4.5 vs C4.5+m+cf differ by 1.179 average ranks (< CD 1.25) -> not significant at 0.05
        assert p.loc["C4.5", "C4.5+m+cf"] > 0.05
        # C4.5 vs C4.5+m differ by 1.143 -> also not significant
        assert p.loc["C4.5", "C4.5+m"] > 0.05

    def test_cd_diagram_renders(self, demsar_table):
        res = friedman_test(demsar_table)
        cd = nemenyi_critical_difference(4, 14)
        ax = plot_critical_difference_diagram(res.average_ranks, cd, title="demo")
        assert ax.get_title() == "demo"


class TestWilcoxonHolm:
    def test_holm_matches_reference(self):
        p = [0.01, 0.04, 0.03, 0.005]
        adj = holm_correction(p)
        # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04->max(0.06,0.04)=0.06
        np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])
        assert holm_correction([]).size == 0

    def test_pairwise_table(self, demsar_table):
        out = wilcoxon_holm(demsar_table)
        assert len(out) == 6
        assert set(out.columns) >= {
            "estimator_a",
            "estimator_b",
            "p_value",
            "p_adjusted",
            "significant",
            "winner",
        }
        assert (out["p_adjusted"] >= out["p_value"]).all()

    def test_identical_columns_give_p_one(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
        out = wilcoxon_holm(df)
        assert out.loc[0, "p_value"] == 1.0 and out.loc[0, "winner"] is None


class TestCorrelatedTTests:
    def test_corrected_variance_is_larger_than_naive(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0.85, 0.02, size=10)
        b = rng.normal(0.83, 0.02, size=10)
        t_corr, p_corr = corrected_resampled_ttest(a, b, n_splits=10)
        t_naive, p_naive = stats.ttest_rel(a, b)
        assert abs(t_corr) < abs(t_naive)
        assert p_corr > p_naive

    def test_geometry_arguments(self):
        a = np.array([0.9, 0.91, 0.89, 0.92])
        b = np.array([0.88, 0.9, 0.87, 0.9])
        t1, _ = corrected_resampled_ttest(a, b, n_splits=5)
        t2, _ = corrected_resampled_ttest(a, b, n_train=80, n_test=20)
        assert t1 == pytest.approx(t2)
        with pytest.raises(ValueError):
            corrected_resampled_ttest(a, b)
        with pytest.raises(ValueError):
            corrected_resampled_ttest(a, b[:-1], n_splits=5)

    def test_bayesian_probabilities_sum_to_one(self):
        rng = np.random.default_rng(2)
        a = rng.normal(0.90, 0.01, size=30)
        b = rng.normal(0.80, 0.01, size=30)
        res = bayesian_correlated_ttest(a, b, rope=0.01, n_splits=10)
        assert res.p_left + res.p_rope + res.p_right == pytest.approx(1.0)
        assert res.p_right > 0.99
        assert res.decision() == "A > B"
        res_rev = bayesian_correlated_ttest(b, a, rope=0.01, n_splits=10)
        assert res_rev.decision() == "A < B"

    def test_bayesian_equivalence(self):
        a = np.linspace(0.80, 0.81, 20)
        b = a + 0.0005
        res = bayesian_correlated_ttest(a, b, rope=0.05, n_splits=10)
        assert res.decision() == "equivalent"

    def test_bayesian_zero_variance(self):
        a = np.full(5, 0.9)
        b = np.full(5, 0.8)
        res = bayesian_correlated_ttest(a, b, rope=0.01, n_splits=5)
        assert res.p_right == 1.0
