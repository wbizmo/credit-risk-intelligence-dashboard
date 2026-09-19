from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from datasets import LENDINGCLUB_FILENAME, _extract_lendingclub_archive
from data_contracts import (
    ContractViolation,
    validate_chronological_splits,
    validate_external_dataset,
    validate_primary_harmonized,
    validate_primary_source,
)
from drift import adversarial_validation, build_distribution_shift_evidence, jensen_shannon_divergence
from explanation_validation import (
    _segment_rows,
    _small_valid_perturbation,
    deterministic_sample_indices,
    sign_agreement,
    spearman_explanation_rank,
    top_k_overlap,
    validate_champion_metadata,
)
from governance import FeatureProvenance, PRIMARY_FEATURE_PROVENANCE
from lineage import (
    LineageViolation,
    build_training_manifest,
    sha256_file,
    verify_training_manifest,
)


def _harmonized_fixture() -> pd.DataFrame:
    issue = pd.to_datetime(["2015-01-01", "2015-02-01", "2015-03-01"])
    return pd.DataFrame(
        {
            "sourceRowId": [1, 2, 3],
            "issueDate": issue,
            "featureAsOf": issue,
            "annualIncome": [50_000.0, 60_000.0, 70_000.0],
            "loanAmount": [10_000.0, 12_000.0, 14_000.0],
            "debtToIncome": [0.2, 0.3, 0.4],
            "loanToIncome": [0.2, 0.2, 0.2],
            "creditScore": [680.0, 700.0, 720.0],
            "employmentYears": [2.0, 5.0, 8.0],
            "target": [0, 1, 0],
        }
    )


class LendingClubDownloadTests(unittest.TestCase):
    def test_archive_fallback_extracts_only_expected_dataset_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "source.zip"
            output_path = root / LENDINGCLUB_FILENAME
            payload = b"issue_d,revenue,Default\nJan-15,50000,0\n"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(LENDINGCLUB_FILENAME, payload)
                archive.writestr("unrelated.txt", b"ignored")

            _extract_lendingclub_archive(archive_path, output_path)
            self.assertEqual(output_path.read_bytes(), payload)

    def test_archive_fallback_rejects_missing_or_invalid_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "missing.zip"
            output_path = root / LENDINGCLUB_FILENAME
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("other.csv", b"x\n")

            with self.assertRaisesRegex(RuntimeError, "does not contain expected file"):
                _extract_lendingclub_archive(archive_path, output_path)

            archive_path.write_bytes(b"not-a-zip")
            with self.assertRaisesRegex(RuntimeError, "not a valid ZIP"):
                _extract_lendingclub_archive(archive_path, output_path)


class DataContractTests(unittest.TestCase):
    def test_primary_source_missing_column_and_invalid_target_fail_safely(self) -> None:
        source = pd.DataFrame(
            {
                "issue_d": ["Jan-15"],
                "revenue": [50_000],
                "dti_n": [20],
                "loan_amnt": [10_000],
                "fico_n": [700],
                "emp_length": ["5 years"],
                "Default": [2],
            }
        )
        with self.assertRaisesRegex(ContractViolation, "target-domain") as captured:
            validate_primary_source(source)
        self.assertNotIn("50000", str(captured.exception))

        with self.assertRaisesRegex(ContractViolation, "required-columns"):
            validate_primary_source(source.drop(columns=["fico_n"]))

    def test_harmonized_contract_rejects_bad_ranges_nonfinite_duplicate_and_future_asof(self) -> None:
        for column, value, pattern in (
            ("annualIncome", 0.0, "annualIncome-positive"),
            ("loanAmount", -1.0, "loanAmount-positive"),
            ("debtToIncome", np.inf, "debtToIncome-finite"),
            ("creditScore", 999.0, "creditScore-upper-bound"),
            ("employmentYears", 11.0, "employmentYears-upper-bound"),
            ("target", 3, "target-domain"),
        ):
            frame = _harmonized_fixture()
            frame.loc[1, column] = value
            with self.assertRaisesRegex(ContractViolation, pattern):
                validate_primary_harmonized(
                    frame,
                    provenance=PRIMARY_FEATURE_PROVENANCE,
                    required_features=[item.feature for item in PRIMARY_FEATURE_PROVENANCE],
                )

        duplicate = _harmonized_fixture()
        duplicate.loc[2, "sourceRowId"] = 2
        with self.assertRaisesRegex(ContractViolation, "source-row-id-unique"):
            validate_primary_harmonized(
                duplicate,
                provenance=PRIMARY_FEATURE_PROVENANCE,
                required_features=[item.feature for item in PRIMARY_FEATURE_PROVENANCE],
            )

        future = _harmonized_fixture()
        future.loc[1, "featureAsOf"] = pd.Timestamp("2015-03-01")
        with self.assertRaisesRegex(ContractViolation, "feature-asof-not-after-origination"):
            validate_primary_harmonized(
                future,
                provenance=PRIMARY_FEATURE_PROVENANCE,
                required_features=[item.feature for item in PRIMARY_FEATURE_PROVENANCE],
            )

    def test_harmonized_contract_preserves_independent_provenance_gate(self) -> None:
        bad = list(PRIMARY_FEATURE_PROVENANCE)
        bad[0] = FeatureProvenance(
            feature="debtToIncome",
            sources=("future_status",),
            availability="post-outcome",
            max_lookback_days=None,
            outcome_derived=True,
        )
        with self.assertRaisesRegex(ValueError, "outcome-derived"):
            validate_primary_harmonized(
                _harmonized_fixture(),
                provenance=bad,
                required_features=[item.feature for item in PRIMARY_FEATURE_PROVENANCE],
            )

    def test_external_contracts_bind_product_specific_source_semantics(self) -> None:
        taiwan = pd.DataFrame(
            {
                "creditUtilization6mMean": np.zeros(30_000),
                "onTimePaymentRate6m": np.ones(30_000),
                "paymentDelayMonths6m": np.zeros(30_000),
                "recentCreditGrowth6m": np.zeros(30_000),
            }
        )
        target = np.zeros(30_000, dtype=np.int8)
        evidence = validate_external_dataset(
            "uci-taiwan-credit-card-default",
            taiwan,
            target,
            source_features=list(taiwan.columns),
        )
        self.assertEqual(evidence["product"], "revolving-credit-card")
        with self.assertRaisesRegex(ContractViolation, "product-specific-source-feature-contract"):
            validate_external_dataset(
                "uci-taiwan-credit-card-default",
                taiwan,
                target,
                source_features=["wrong", "feature", "contract", "set"],
            )

    def test_chronological_split_rejects_impossible_date_ordering_without_row_overlap(self) -> None:
        frame = _harmonized_fixture()
        train = frame.iloc[[1]].copy()
        calibration = frame.iloc[[0]].copy()
        oot = frame.iloc[[2]].copy()
        with self.assertRaisesRegex(ContractViolation, "chronological-boundaries"):
            validate_chronological_splits(train, calibration, oot)

    def test_chronological_split_rejects_overlap(self) -> None:
        frame = _harmonized_fixture()
        train = frame.iloc[[0]].copy()
        calibration = frame.iloc[[1]].copy()
        oot = frame.iloc[[2]].copy()
        validate_chronological_splits(train, calibration, oot)

        overlapping = oot.copy()
        overlapping.loc[:, "sourceRowId"] = train.iloc[0]["sourceRowId"]
        with self.assertRaisesRegex(ContractViolation, "split-row-overlap"):
            validate_chronological_splits(train, calibration, overlapping)


class DriftTests(unittest.TestCase):
    def test_matched_distributions_are_near_chance_and_shift_is_stronger(self) -> None:
        rng = np.random.default_rng(8)
        expected = rng.normal(size=(2_000, 4))
        matched = expected.copy()
        shifted = expected.copy()
        shifted[:, 0] += 2.5
        matched_result = adversarial_validation(
            expected,
            matched,
            seed=9,
            max_rows_per_population=2_000,
            minimum_rows_per_population=500,
        )
        shifted_result = adversarial_validation(
            expected,
            shifted,
            seed=9,
            max_rows_per_population=2_000,
            minimum_rows_per_population=500,
        )
        self.assertAlmostEqual(matched_result["auc"], 0.5, delta=0.06)
        self.assertGreater(shifted_result["auc"], matched_result["auc"] + 0.20)
        self.assertGreater(
            jensen_shannon_divergence(expected[:, 0], shifted[:, 0]),
            jensen_shannon_divergence(expected[:, 0], matched[:, 0]),
        )

    def test_insufficient_and_chronology_mixing_are_explicit(self) -> None:
        small = np.ones((20, 4))
        insufficient = adversarial_validation(small, small, minimum_rows_per_population=100)
        self.assertEqual(insufficient["status"], "insufficient-data")

        with self.assertRaisesRegex(ValueError, "chronological OOT"):
            build_distribution_shift_evidence(
                np.ones((600, 4)),
                np.ones((600, 4)),
                feature_names=["a", "b", "c", "d"],
                expected_dates=np.array(["2017-01-01"] * 600, dtype="datetime64[D]"),
                actual_dates=np.array(["2017-01-01"] * 600, dtype="datetime64[D]"),
            )


class ExplanationMetricTests(unittest.TestCase):
    def test_top_k_sign_and_rank_metrics_handle_ties_deterministically(self) -> None:
        left = np.array([3.0, 2.0, 2.0, 0.1])
        right = np.array([2.5, 2.0, 2.0, -0.1])
        self.assertEqual(top_k_overlap(left, right, k=3), 1.0)
        correlation = spearman_explanation_rank(left, right)
        self.assertIsNotNone(correlation)
        assert correlation is not None
        self.assertGreater(correlation, 0.8)
        self.assertAlmostEqual(sign_agreement(left, right), 0.75)

    def test_explanation_perturbations_are_small_and_domain_valid_for_ood_rows(self) -> None:
        features = ["debtToIncome", "loanToIncome", "creditScore", "employmentYears"]
        sample = np.array([[1.8, 2.8, 820.0, 9.5]], dtype=np.float32)
        changed = _small_valid_perturbation(sample, features)
        self.assertTrue(np.all(np.isfinite(changed)))
        self.assertGreaterEqual(changed[0, 0], 0.0)
        self.assertLessEqual(changed[0, 0], 2.0)
        self.assertGreaterEqual(changed[0, 1], 0.001)
        self.assertLessEqual(changed[0, 1], 3.0)
        self.assertGreaterEqual(changed[0, 2], 300.0)
        self.assertLessEqual(changed[0, 2], 850.0)
        self.assertGreaterEqual(changed[0, 3], 0.0)
        self.assertLessEqual(changed[0, 3], 10.0)
        self.assertLessEqual(abs(float(changed[0, 0] - sample[0, 0])), 0.021)
        self.assertLessEqual(abs(float(changed[0, 2] - sample[0, 2])), 5.51)

    def test_fixed_seed_sample_indices_are_deterministic(self) -> None:
        first = deterministic_sample_indices(1000, 100, 42)
        second = deterministic_sample_indices(1000, 100, 42)
        third = deterministic_sample_indices(1000, 100, 43)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, third))

    def test_weak_explanation_segments_return_insufficient_data(self) -> None:
        shap_values = np.array([[1.0, 0.5, 0.1], [0.9, 0.4, 0.2]])
        local_values = np.array([[0.8, 0.6, 0.1], [0.7, 0.5, 0.2]])
        perturbed = local_values.copy()
        rows = _segment_rows(
            shap_values,
            local_values,
            perturbed,
            np.array(["thin", "thin"]),
            k=2,
            minimum_count=3,
        )
        self.assertEqual(rows, [
            {"segment": "thin", "status": "insufficient-data", "count": 2}
        ])

    def test_metadata_mismatch_fails_before_explanation_report(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_champion_metadata(
                model_name="CRIX-MonoBoost",
                model_version="2.0.0",
                expected_name="CRIX-MonoBoost",
                expected_version="3.0.0",
            )


class LineageTests(unittest.TestCase):
    def _fixture(self, root: Path, previous: Path | None = None) -> tuple[Path, Path]:
        artifact = root / "artifact.json"
        dataset = root / "dataset.csv"
        lock = root / "requirements.txt"
        report = root / "report.md"
        artifact.write_text('{"model":"x"}\n')
        dataset.write_text("x,y\n1,0\n")
        lock.write_text("numpy==2.0\n")
        report.write_text("# aggregate report\n")
        runtime = {
            "modelId": "CRIX-Test@1.0.0",
            "artifactSha256": sha256_file(artifact),
            "trainingRunId": "run-1",
            "gitCommit": "1" * 40,
            "featureContractVersion": "features-v1",
            "datasets": [
                {
                    "id": "dataset-v1",
                    "doi": "10.test/example",
                    "version": "1",
                    "sourceMd5": "0" * 32,
                    "localSha256": sha256_file(dataset),
                }
            ],
            "split": {
                "train": {"samples": 10},
                "calibration": {"samples": 4},
                "test": {"samples": 5},
            },
        }
        manifest = build_training_manifest(
            root=root,
            runtime_manifest=runtime,
            artifact_path=artifact,
            dependency_lock_path=lock,
            evidence_paths=[report],
            dataset_path=dataset,
            run_id="lineage-test",
            git_commit="2" * 40,
            data_contract_version="data-v1",
            seeds={"training": 42},
            created_at="2026-09-19T00:00:00+00:00",
            previous_manifest_path=previous,
        )
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest_path, dataset

    def test_manifest_verifies_and_safe_output_contains_no_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, dataset = self._fixture(root)
            result = verify_training_manifest(
                manifest_path,
                root=root,
                dataset_path=dataset,
                expected_git_commit="2" * 40,
                expected_data_contract_version="data-v1",
            )
            self.assertEqual(result["status"], "pass")
            self.assertNotIn("x,y", json.dumps(result))

    def test_artifact_report_and_prior_manifest_tampering_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_manifest, _ = self._fixture(root)
            previous = root / "previous.json"
            first_manifest.replace(previous)
            manifest_path, dataset = self._fixture(root, previous=previous)

            verified = verify_training_manifest(manifest_path, root=root, dataset_path=dataset)
            self.assertEqual(verified["manifestChainDepth"], 1)

            (root / "artifact.json").write_text('{"model":"tampered"}\n')
            with self.assertRaisesRegex(LineageViolation, "artifact hash"):
                verify_training_manifest(manifest_path, root=root, dataset_path=dataset)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, dataset = self._fixture(root)
            (root / "report.md").write_text("tampered\n")
            with self.assertRaisesRegex(LineageViolation, "evidence hash"):
                verify_training_manifest(manifest_path, root=root, dataset_path=dataset)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_manifest, _ = self._fixture(root)
            previous = root / "previous.json"
            first_manifest.replace(previous)
            manifest_path, dataset = self._fixture(root, previous=previous)
            previous.write_text('{"schemaVersion":1,"tampered":true}\n')
            with self.assertRaisesRegex(LineageViolation, "hash-chain"):
                verify_training_manifest(manifest_path, root=root, dataset_path=dataset)

    def test_malformed_and_wrong_expected_metadata_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            broken = root / "broken.json"
            broken.write_text('{"schemaVersion":1}\n')
            with self.assertRaises(LineageViolation):
                verify_training_manifest(broken, root=root)

            manifest_path, dataset = self._fixture(root)
            with self.assertRaisesRegex(LineageViolation, "gitCommit"):
                verify_training_manifest(
                    manifest_path,
                    root=root,
                    dataset_path=dataset,
                    expected_git_commit="3" * 40,
                )
            with self.assertRaisesRegex(LineageViolation, "dataContractVersion"):
                verify_training_manifest(
                    manifest_path,
                    root=root,
                    dataset_path=dataset,
                    expected_data_contract_version="data-v2",
                )


if __name__ == "__main__":
    unittest.main()
