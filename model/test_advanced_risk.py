from __future__ import annotations

import math
import unittest

import numpy as np

from capital import economic_capital, irb_corporate_capital, reconcile_tail_capital
from decisioning import Candidate, ChallengerMetrics, compare_challengers, optimize_exact, sorted_equal_exposure_frontier
from ifrs9 import EclPolicy, cumulative_to_marginal, determine_stage, scenario_weighted_ecl
from macro_stress import MacroObservation, asof_join_macro, fit_univariate_logit, stress_pd_path
from portfolio_risk import simulate_portfolio


class PortfolioRiskTests(unittest.TestCase):
    def test_independent_mean_and_variance_match_analytic_toy(self) -> None:
        pd = np.full(80, 0.08)
        lgd = np.full(80, 0.45)
        ead = np.full(80, 1000.0)
        result = simulate_portfolio(pd, lgd, ead, rho=0.0, scenarios=50000, seed=17, chunk_size=731)
        analytic = result["independentBaseline"]
        self.assertLess(abs(result["expectedLoss"] - analytic["expectedLoss"]), 5 * result["monteCarlo"]["expectedLossStdError"])
        self.assertLess(abs(result["unexpectedLoss"] - analytic["unexpectedLoss"]), analytic["unexpectedLoss"] * 0.05)

    def test_seeded_results_are_chunk_size_invariant(self) -> None:
        pd = np.linspace(0.01, 0.20, 25)
        lgd = np.linspace(0.30, 0.75, 25)
        ead = np.linspace(1000, 5000, 25)
        left = simulate_portfolio(pd, lgd, ead, rho=0.18, scenarios=12000, seed=1234, chunk_size=127)
        right = simulate_portfolio(pd, lgd, ead, rho=0.18, scenarios=12000, seed=1234, chunk_size=2048)
        self.assertEqual(left["lossDigest"], right["lossDigest"])
        self.assertEqual(left["expectedLoss"], right["expectedLoss"])
        self.assertEqual(left["var"]["0.99"], right["var"]["0.99"])

    def test_999_tail_is_withheld_below_precision_floor(self) -> None:
        result = simulate_portfolio([0.05] * 20, [0.5] * 20, [1000] * 20, rho=0.2, scenarios=20000, seed=1)
        self.assertFalse(result["tailSupport"]["0.999"]["supported"])
        self.assertIsNone(result["var"]["0.999"])
        self.assertIn("100000", result["tailSupport"]["0.999"]["reason"])

    def test_tail_contributions_reconcile_to_expected_shortfall(self) -> None:
        result = simulate_portfolio(
            [0.03, 0.07, 0.12], [0.4, 0.5, 0.6], [1000, 2500, 5000], rho=0.25,
            scenarios=30000, seed=8, tail_contributions=True, quantiles=(0.95,),
        )
        self.assertAlmostEqual(sum(result["tailContributions"]["0.95"]), result["expectedShortfall"]["0.95"], places=10)

    def test_invalid_correlation_and_negative_exposure_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            simulate_portfolio([0.1], [0.5], [100], rho=1.01, scenarios=1000)
        with self.assertRaises(ValueError):
            simulate_portfolio([0.1], [0.5], [-1], rho=0.1, scenarios=1000)


class MacroStressTests(unittest.TestCase):
    def test_asof_join_never_uses_unpublished_value(self) -> None:
        observations = [
            MacroObservation("2020-01-01", "2020-02-01", 4.0),
            MacroObservation("2020-02-01", "2020-03-01", 5.0),
        ]
        joined = asof_join_macro(["2020-02-15", "2020-03-01"], observations)
        self.assertEqual(joined.tolist(), [4.0, 5.0])

    def test_univariate_logit_recovers_positive_macro_relationship(self) -> None:
        x = np.repeat(np.array([4.0, 6.0, 8.0, 10.0]), 1500)
        p = 1 / (1 + np.exp(-(-4.0 + 0.35 * x)))
        rng = np.random.default_rng(42)
        y = rng.binomial(1, p)
        fit = fit_univariate_logit(x, y)
        self.assertGreater(fit["coefficient"], 0)
        self.assertLess(fit["coefficientCi95"][0], fit["coefficient"])
        self.assertGreater(fit["coefficientCi95"][1], fit["coefficient"])

    def test_stress_path_identity_and_extrapolation(self) -> None:
        base_pd = np.array([0.05, 0.05, 0.05])
        baseline = np.array([5.0, 5.5, 6.0])
        identity = stress_pd_path(base_pd, baseline, baseline, coefficient=0.4, scale=1.0, support=(4.0, 10.0))
        adverse = stress_pd_path(base_pd, baseline, baseline + 2.0, coefficient=0.4, scale=1.0, support=(4.0, 7.0))
        self.assertTrue(np.allclose(identity["pd"], base_pd))
        self.assertTrue(np.all(adverse["pd"] > base_pd))
        self.assertTrue(adverse["extrapolative"])


class Ifrs9Tests(unittest.TestCase):
    def test_cumulative_pd_converts_to_marginal_once(self) -> None:
        marginal = cumulative_to_marginal([0.10, 0.19, 0.271])
        np.testing.assert_allclose(marginal, [0.10, 0.09, 0.081], rtol=0, atol=1e-12)
        with self.assertRaises(ValueError):
            cumulative_to_marginal([0.2, 0.1])

    def test_stage_policy_covers_sicr_dpd_default_and_probation(self) -> None:
        policy = EclPolicy(relative_pd_multiplier=2.0, absolute_pd_increase=0.05, stage2_dpd=30, stage3_dpd=90, cure_probation_months=3)
        self.assertEqual(determine_stage(0.04, 0.05, 0, False, policy=policy)["stage"], 1)
        self.assertEqual(determine_stage(0.04, 0.09, 0, False, policy=policy)["stage"], 2)
        self.assertEqual(determine_stage(0.04, 0.05, 45, False, policy=policy)["stage"], 2)
        self.assertEqual(determine_stage(0.04, 0.05, 0, True, policy=policy)["stage"], 3)
        self.assertEqual(determine_stage(0.04, 0.05, 0, False, previous_stage=2, months_since_cure=1, policy=policy)["stage"], 2)

    def test_scenario_weighted_ecl_matches_hand_calculation(self) -> None:
        scenarios = {
            "baseline": {"weight": 0.75, "cumulativePd": [0.10, 0.19], "lgd": [0.5, 0.5], "ead": [100.0, 80.0]},
            "adverse": {"weight": 0.25, "cumulativePd": [0.20, 0.36], "lgd": [0.6, 0.6], "ead": [100.0, 80.0]},
        }
        result = scenario_weighted_ecl(scenarios, stage=2, annual_eir=0.0, interval_months=12, policy_version="ifrs9-research-v1")
        baseline = 0.10 * 0.5 * 100 + 0.09 * 0.5 * 80
        adverse = 0.20 * 0.6 * 100 + 0.16 * 0.6 * 80
        self.assertAlmostEqual(result["ecl"], 0.75 * baseline + 0.25 * adverse, places=12)
        self.assertAlmostEqual(sum(result["scenarioContributions"].values()), result["ecl"], places=12)

    def test_invalid_scenario_weights_fail(self) -> None:
        with self.assertRaises(ValueError):
            scenario_weighted_ecl({"a": {"weight": 0.8, "cumulativePd": [0.1], "lgd": [0.5], "ead": [100]}}, stage=1)


class CapitalTests(unittest.TestCase):
    def test_zero_ead_has_zero_irb_style_capital(self) -> None:
        result = irb_corporate_capital(0.02, 0.45, 0.0, maturity_years=2.5)
        self.assertEqual(result["capital"], 0.0)
        self.assertEqual(result["rwaStyle"], 0.0)

    def test_irb_style_capital_increases_with_lgd_for_same_exposure(self) -> None:
        low = irb_corporate_capital(0.02, 0.25, 1_000_000, maturity_years=2.5)
        high = irb_corporate_capital(0.02, 0.55, 1_000_000, maturity_years=2.5)
        self.assertGreater(high["capital"], low["capital"])
        self.assertEqual(low["formulaVersion"], "basel-irb-corporate-research-v1")
        self.assertIn("not regulatory compliance", low["status"].lower())

    def test_economic_capital_keeps_expected_loss_separate(self) -> None:
        result = economic_capital(expected_loss=100.0, tail_loss=260.0, confidence=0.99, tail_measure="VaR")
        self.assertEqual(result["economicCapital"], 160.0)
        self.assertEqual(result["expectedLoss"], 100.0)

    def test_tail_capital_contributions_reconcile(self) -> None:
        result = reconcile_tail_capital([80.0, 120.0], [30.0, 40.0])
        self.assertAlmostEqual(sum(result["economicCapitalContributions"]), result["economicCapital"], places=12)


class DecisioningTests(unittest.TestCase):
    def test_higher_auc_challenger_can_fail_promotion_gate(self) -> None:
        incumbent = ChallengerMetrics("incumbent", "same", 0.70, 0.16, 1.02, 0.01, 0.05, 0.17)
        flashy = ChallengerMetrics("flashy", "same", 0.75, 0.15, 1.55, 0.02, 0.04, 0.16)
        result = compare_challengers([incumbent, flashy], incumbent_id="incumbent")
        self.assertEqual(result["recommendedModelId"], "incumbent")
        self.assertIn("calibration", " ".join(result["models"]["flashy"]["failedGates"]).lower())

    def test_exact_optimizer_matches_known_toy_optimum(self) -> None:
        candidates = [
            Candidate("a", 60, 10, 1, "x"),
            Candidate("b", 50, 9, 1, "x"),
            Candidate("c", 40, 8, 1, "y"),
        ]
        result = optimize_exact(candidates, budget=100, max_expected_loss=3, max_segment_share=0.7)
        self.assertEqual(result["status"], "optimal")
        self.assertEqual(set(result["selectedIds"]), {"a", "c"})
        self.assertAlmostEqual(result["expectedReturn"], 18.0)

    def test_ineligible_candidate_is_never_selected_and_infeasible_is_explicit(self) -> None:
        candidates = [Candidate("blocked", 10, 100, 0, "x", eligible=False)]
        result = optimize_exact(candidates, budget=100, min_approval_count=1)
        self.assertEqual(result["status"], "infeasible")
        self.assertEqual(result["selectedIds"], [])

    def test_sorted_equal_exposure_frontier_is_direct_and_deterministic(self) -> None:
        candidates = [
            Candidate("a", 100, 5, 1, "x"),
            Candidate("b", 100, 8, 2, "x"),
            Candidate("c", 100, 7, 1, "x"),
        ]
        result = sorted_equal_exposure_frontier(candidates, budget=200)
        self.assertEqual(result["selectedIds"], ["b", "c"])
        with self.assertRaises(ValueError):
            sorted_equal_exposure_frontier([Candidate("a", 100, 5, 1, "x"), Candidate("b", 90, 8, 2, "x")], budget=200)


if __name__ == "__main__":
    unittest.main()
