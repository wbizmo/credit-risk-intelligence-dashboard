from __future__ import annotations

import unittest

import numpy as np

from governance import (
    POPULATION_CONDITIONING,
    PRIMARY_FEATURE_PROVENANCE,
    FeatureProvenance,
    assert_point_in_time,
    backtest_pd_policy,
    bootstrap_metric_interval,
    build_governance_evidence,
    calibration_by_bins,
    calibration_intercept_slope,
    population_stability_index,
    segment_diagnostics,
    validate_feature_provenance,
)


class ProvenanceTests(unittest.TestCase):
    def test_feature_provenance_fails_closed_for_missing_required_feature(self) -> None:
        provenance = [
            FeatureProvenance(
                feature="debtToIncome",
                sources=("dti_n",),
                availability="application-time",
                max_lookback_days=0,
            )
        ]

        with self.assertRaisesRegex(ValueError, "missing provenance"):
            validate_feature_provenance(provenance, ["debtToIncome", "creditScore"])

    def test_feature_provenance_rejects_outcome_derived_predictor(self) -> None:
        provenance = [
            FeatureProvenance(
                feature="futureRecovery",
                sources=("recoveries",),
                availability="post-default",
                max_lookback_days=None,
                outcome_derived=True,
            )
        ]

        with self.assertRaisesRegex(ValueError, "outcome-derived"):
            validate_feature_provenance(provenance, ["futureRecovery"])

    def test_feature_provenance_rejects_policy_derived_predictor(self) -> None:
        provenance = [
            FeatureProvenance(
                feature="historicalGrade",
                sources=("grade",),
                availability="application-time",
                max_lookback_days=0,
                policy_derived=True,
            )
        ]

        with self.assertRaisesRegex(ValueError, "policy-derived"):
            validate_feature_provenance(provenance, ["historicalGrade"])

    def test_primary_feature_provenance_matches_the_v3_champion_contract(self) -> None:
        names = [item.feature for item in PRIMARY_FEATURE_PROVENANCE]
        self.assertEqual(names, ["debtToIncome", "loanToIncome", "creditScore", "employmentYears"])
        validated = validate_feature_provenance(PRIMARY_FEATURE_PROVENANCE, names)
        self.assertEqual(set(validated), set(names))
        self.assertTrue(all(item.availability == "application-time" for item in validated.values()))

    def test_point_in_time_guard_rejects_future_availability_without_row_payloads(self) -> None:
        available_at = np.array(["2017-01-01", "2017-02-02"], dtype="datetime64[D]")
        as_of = np.array(["2017-01-01", "2017-02-01"], dtype="datetime64[D]")

        with self.assertRaisesRegex(ValueError, "1 feature value") as captured:
            assert_point_in_time(available_at, as_of, label="fixture")

        self.assertNotIn("2017-02-02", str(captured.exception))

    def test_point_in_time_guard_accepts_equal_or_past_dates(self) -> None:
        available_at = np.array(["2016-12-31", "2017-02-01"], dtype="datetime64[D]")
        as_of = np.array(["2017-01-01", "2017-02-01"], dtype="datetime64[D]")
        assert_point_in_time(available_at, as_of, label="fixture")


class DiagnosticsTests(unittest.TestCase):
    def test_binary_outcome_validation_rejects_fractional_labels_before_casting(self) -> None:
        y = np.array([0.0, 0.5, 1.0])
        p = np.array([0.1, 0.5, 0.9])
        with self.assertRaisesRegex(ValueError, "binary"):
            calibration_by_bins(y, p, bins=3, min_count=1, min_events=0)

    def test_calibration_bins_handle_duplicate_quantile_edges(self) -> None:
        y = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int8)
        p = np.array([0.25] * len(y), dtype=float)
        rows = calibration_by_bins(y, p, bins=10, min_count=2, min_events=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["count"], len(y))
        self.assertAlmostEqual(rows[0]["predicted"], 0.25)
        self.assertAlmostEqual(rows[0]["observed"], 0.5)

    def test_calibration_intercept_slope_recovers_positive_relationship(self) -> None:
        p = np.array([0.05, 0.10, 0.20, 0.35, 0.65, 0.80, 0.90, 0.95], dtype=float)
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
        result = calibration_intercept_slope(y, p)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(np.isfinite(result["intercept"]))
        self.assertTrue(np.isfinite(result["slope"]))
        self.assertGreater(result["slope"], 0)

    def test_bootstrap_interval_is_reproducible_from_seed(self) -> None:
        y = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1], dtype=np.int8)
        p = np.array([0.1, 0.8, 0.2, 0.7, 0.3, 0.4, 0.65, 0.9, 0.15, 0.6], dtype=float)
        first = bootstrap_metric_interval(y, p, metric="brier", samples=200, seed=42)
        second = bootstrap_metric_interval(y, p, metric="brier", samples=200, seed=42)
        self.assertEqual(first, second)
        self.assertLessEqual(first["lower"], first["point"])
        self.assertGreaterEqual(first["upper"], first["point"])

    def test_population_stability_detects_shift_with_zero_mass_bins(self) -> None:
        expected = np.array([0.2] * 100, dtype=float)
        actual = np.array([0.2] * 50 + [0.7] * 50, dtype=float)
        psi = population_stability_index(expected, actual, bins=10)
        self.assertTrue(np.isfinite(psi))
        self.assertGreater(psi, 0)

    def test_segment_diagnostics_keep_counts_and_mark_low_support(self) -> None:
        y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
        p = np.array([0.1, 0.2, 0.15, 0.4, 0.25, 0.5], dtype=float)
        segments = np.array(["A", "A", "A", "B", "B", "B"])
        rows = segment_diagnostics(y, p, segments, min_count=4, min_events=1)
        self.assertEqual({row["segment"] for row in rows}, {"A", "B"})
        self.assertTrue(all(row["count"] == 3 for row in rows))
        self.assertTrue(all(row["status"] == "insufficient-data" for row in rows))

    def test_build_governance_evidence_contains_auditable_population_and_policy_metadata(self) -> None:
        y_cal = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1], dtype=np.int8)
        p_cal = np.array([0.08, 0.12, 0.18, 0.22, 0.28, 0.31, 0.38, 0.44, 0.52, 0.61, 0.72, 0.83])
        y_test = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1], dtype=np.int8)
        p_test = np.array([0.10, 0.16, 0.20, 0.26, 0.33, 0.37, 0.43, 0.49, 0.58, 0.66, 0.73, 0.86])
        exposure = np.array([100.0] * len(y_test))
        segments = {
            "creditScoreBand": np.array(["700+", "700+", "700+", "700+", "660-699", "660-699", "660-699", "660-699", "<660", "<660", "<660", "<660"])
        }

        evidence = build_governance_evidence(
            y_cal,
            p_cal,
            y_test,
            p_test,
            exposure,
            feature_names=[item.feature for item in PRIMARY_FEATURE_PROVENANCE],
            segments=segments,
            bootstrap_samples=100,
            min_segment_count=4,
        )

        self.assertEqual(evidence["populationConditioning"], POPULATION_CONDITIONING)
        self.assertFalse(evidence["rejectInference"]["supported"])
        self.assertEqual(
            [item["feature"] for item in evidence["featureProvenance"]],
            [item.feature for item in PRIMARY_FEATURE_PROVENANCE],
        )
        self.assertEqual(set(evidence["uncertainty"]), {"auc", "brier", "logLoss", "ks"})
        self.assertGreaterEqual(evidence["stability"]["calibrationToTestPdPsi"], 0)
        self.assertEqual(evidence["policyBacktests"]["approveAll"]["selectedCount"], len(y_test))
        self.assertIn("creditScoreBand", evidence["segments"])


class PolicyBacktestTests(unittest.TestCase):
    def test_approve_all_reconciles_observed_default_rate(self) -> None:
        y = np.array([0, 1, 0, 1], dtype=np.int8)
        pd = np.array([0.10, 0.20, 0.30, 0.40], dtype=float)
        exposure = np.array([100.0, 100.0, 100.0, 100.0], dtype=float)

        result = backtest_pd_policy(y, pd, exposure, pd_threshold=1.0)

        self.assertEqual(result["populationConditioning"], POPULATION_CONDITIONING)
        self.assertEqual(result["selectedCount"], 4)
        self.assertAlmostEqual(result["selectionRate"], 1.0)
        self.assertAlmostEqual(result["observedDefaultRate"], 0.5)
        self.assertAlmostEqual(result["selectedExposure"], 400.0)

    def test_policy_threshold_is_inclusive_and_does_not_invent_rejected_outcomes(self) -> None:
        y = np.array([0, 1, 1], dtype=np.int8)
        pd = np.array([0.10, 0.20, 0.30], dtype=float)
        exposure = np.array([100.0, 200.0, 300.0], dtype=float)

        result = backtest_pd_policy(y, pd, exposure, pd_threshold=0.20)

        self.assertEqual(result["selectedCount"], 2)
        self.assertAlmostEqual(result["observedDefaultRate"], 0.5)
        self.assertEqual(result["counterfactualRejectedOutcomes"], "unsupported")


if __name__ == "__main__":
    unittest.main()
