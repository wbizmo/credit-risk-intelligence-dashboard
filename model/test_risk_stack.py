from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ead import amortizing_balance, ccf_proxy, observed_installment_ead
from lgd import economic_lgd
from registry import canonical_json_hash, sha256_file, update_registry_index, verify_manifest
from survival import cumulative_from_hazards, expand_person_period, validate_term_structure
from time_machine import SnapshotSpec, backward_asof_join, build_snapshot_masks, snapshot_id
from transitions import DEFAULT_STATES, map_repayment_status, propagate_distribution, transition_counts, transition_matrix


class RegistryTests(unittest.TestCase):
    def test_manifest_digest_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.json"
            artifact.write_text('{"model":"v1"}')
            manifest = {"artifactSha256": sha256_file(artifact)}
            self.assertTrue(verify_manifest(artifact, manifest))
            artifact.write_text('{"model":"tampered"}')
            self.assertFalse(verify_manifest(artifact, manifest))

    def test_canonical_config_hash_ignores_mapping_order(self) -> None:
        self.assertEqual(
            canonical_json_hash({"b": 2, "a": {"y": 2, "x": 1}}),
            canonical_json_hash({"a": {"x": 1, "y": 2}, "b": 2}),
        )

    def test_registry_rejects_same_model_version_with_different_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            first = {
                "modelId": "CRIX-MonoBoost@2.0.0",
                "modelName": "CRIX-MonoBoost",
                "version": "2.0.0",
                "artifactSha256": "a" * 64,
                "status": "approved-demo-champion",
            }
            second = {**first, "artifactSha256": "b" * 64}
            update_registry_index(path, first)
            with self.assertRaisesRegex(ValueError, "different artifact digest"):
                update_registry_index(path, second)


class TimeMachineTests(unittest.TestCase):
    def test_backward_asof_join_never_selects_future_value(self) -> None:
        left = np.array(["2020-01-15", "2020-02-15", "2020-03-01"], dtype="datetime64[D]")
        right = np.array(["2020-01-01", "2020-02-01", "2020-03-02"], dtype="datetime64[D]")
        values = np.array([10.0, 20.0, 30.0])
        joined = backward_asof_join(left, right, values)
        np.testing.assert_allclose(joined[:2], [10.0, 20.0])
        self.assertEqual(joined[2], 20.0)

    def test_snapshot_id_is_deterministic_and_contract_sensitive(self) -> None:
        spec = SnapshotSpec(
            train_end="2015-12-31",
            calibration_end="2016-12-31",
            decision_end="2017-12-31",
            outcome_cutoff="2020-12-31",
            source_sha256="a" * 64,
            mode="point-in-time",
        )
        self.assertEqual(snapshot_id(spec, "feature-contract-v1"), snapshot_id(spec, "feature-contract-v1"))
        self.assertNotEqual(snapshot_id(spec, "feature-contract-v1"), snapshot_id(spec, "feature-contract-v2"))

    def test_point_in_time_masks_require_label_availability_by_each_development_cutoff(self) -> None:
        issue_dates = np.array(["2015-01-01", "2015-02-01", "2016-06-01", "2017-06-01"], dtype="datetime64[D]")
        outcome_available = np.array(["2015-08-01", "2017-01-01", "2016-11-01", "2018-01-01"], dtype="datetime64[D]")
        spec = SnapshotSpec(
            train_end="2015-12-31",
            calibration_end="2016-12-31",
            decision_end="2017-12-31",
            outcome_cutoff="2018-12-31",
            source_sha256="a" * 64,
            mode="point-in-time",
        )
        masks = build_snapshot_masks(issue_dates, outcome_available, spec)
        np.testing.assert_array_equal(masks.train, [True, False, False, False])
        np.testing.assert_array_equal(masks.calibration, [False, False, True, False])
        np.testing.assert_array_equal(masks.decision, [False, False, False, True])
        np.testing.assert_array_equal(masks.evaluable, [True, True, True, True])


class SurvivalTests(unittest.TestCase):
    def test_known_hazards_produce_exact_survival_and_cumulative_pd(self) -> None:
        hazards = np.array([0.10, 0.20, 0.25])
        result = cumulative_from_hazards(hazards)
        expected_survival = np.array([0.90, 0.72, 0.54])
        np.testing.assert_allclose(result["survival"], expected_survival)
        np.testing.assert_allclose(result["cumulativePd"], 1 - expected_survival)
        np.testing.assert_allclose(result["marginalPd"], [0.10, 0.18, 0.18])
        validate_term_structure(result)

    def test_person_period_expansion_respects_censoring_and_event_month(self) -> None:
        durations = np.array([2, 3], dtype=np.int16)
        events = np.array([1, 0], dtype=np.int8)
        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        expanded = expand_person_period(features, durations, events, max_horizon=4)
        np.testing.assert_array_equal(expanded["month"], [1, 2, 1, 2, 3])
        np.testing.assert_array_equal(expanded["event"], [0, 1, 0, 0, 0])
        self.assertEqual(expanded["features"].shape, (5, 2))

    def test_term_structure_validation_rejects_non_monotone_cumulative_pd(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            validate_term_structure({
                "hazard": np.array([0.2, 0.2]),
                "survival": np.array([0.8, 0.7]),
                "marginalPd": np.array([0.2, 0.1]),
                "cumulativePd": np.array([0.2, 0.15]),
            })


class TransitionTests(unittest.TestCase):
    def test_repayment_status_mapping_preserves_current_and_delinquency_buckets(self) -> None:
        self.assertEqual(map_repayment_status(-1), "CURRENT")
        self.assertEqual(map_repayment_status(0), "CURRENT")
        self.assertEqual(map_repayment_status(1), "DPD_1_29")
        self.assertEqual(map_repayment_status(2), "DPD_30_59")
        self.assertEqual(map_repayment_status(3), "DPD_60_89")
        self.assertEqual(map_repayment_status(4), "DPD_90_PLUS")
        self.assertIsNone(map_repayment_status(np.nan))

    def test_transition_matrix_keeps_backward_cure_and_rows_sum_to_one(self) -> None:
        states = ["CURRENT", "DPD_30_59", "DEFAULT"]
        sequences = [
            ["CURRENT", "DPD_30_59", "CURRENT"],
            ["CURRENT", "DEFAULT", "DEFAULT"],
        ]
        counts = transition_counts(sequences, states)
        matrix = transition_matrix(counts, states, absorbing={"DEFAULT"})
        np.testing.assert_allclose(matrix.sum(axis=1), np.ones(3))
        self.assertGreater(matrix[1, 0], 0)
        self.assertEqual(matrix[2, 2], 1.0)

    def test_missing_observation_does_not_create_hidden_transition(self) -> None:
        states = ["CURRENT", "DPD_30_59"]
        counts = transition_counts([["CURRENT", None, "DPD_30_59"]], states)
        self.assertEqual(int(counts.sum()), 0)

    def test_propagation_matches_two_step_matrix_multiplication(self) -> None:
        matrix = np.array([[0.8, 0.2], [0.1, 0.9]])
        initial = np.array([1.0, 0.0])
        propagated = propagate_distribution(matrix, initial, 2)
        np.testing.assert_allclose(propagated, initial @ matrix @ matrix)


class EadTests(unittest.TestCase):
    def test_zero_rate_amortization_is_linear_and_off_by_one_safe(self) -> None:
        self.assertAlmostEqual(amortizing_balance(1200.0, 0.0, 12, 0), 1200.0)
        self.assertAlmostEqual(amortizing_balance(1200.0, 0.0, 12, 1), 1100.0)
        self.assertAlmostEqual(amortizing_balance(1200.0, 0.0, 12, 12), 0.0)

    def test_observed_installment_ead_never_goes_negative(self) -> None:
        result = observed_installment_ead(np.array([1000.0, 1000.0]), np.array([250.0, 1200.0]))
        np.testing.assert_allclose(result, [750.0, 0.0])

    def test_ccf_proxy_is_unclamped_and_zero_undrawn_is_missing(self) -> None:
        result = ccf_proxy(
            np.array([1000.0, 1000.0, 1000.0]),
            np.array([500.0, 900.0, 1000.0]),
            np.array([400.0, 1200.0, 1100.0]),
        )
        self.assertLess(result[0], 0)
        self.assertGreater(result[1], 1)
        self.assertTrue(np.isnan(result[2]))


class LgdTests(unittest.TestCase):
    def test_economic_lgd_uses_net_recovery_exactly(self) -> None:
        result = economic_lgd(
            np.array([100.0]),
            np.array([30.0]),
            np.array([5.0]),
            discount_factor=1.0,
        )
        np.testing.assert_allclose(result, [0.75])

    def test_collection_cost_can_produce_lgd_above_one_without_silent_clamp(self) -> None:
        result = economic_lgd(
            np.array([100.0]),
            np.array([0.0]),
            np.array([20.0]),
            discount_factor=1.0,
        )
        self.assertGreater(result[0], 1.0)

    def test_zero_ead_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "EAD"):
            economic_lgd(np.array([0.0]), np.array([0.0]), np.array([0.0]))


if __name__ == "__main__":
    unittest.main()
