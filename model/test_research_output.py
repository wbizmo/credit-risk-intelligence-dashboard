from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research_validation import compare_artifacts, training_paths


class ResearchOutputIsolationTests(unittest.TestCase):
    def test_validation_output_uses_isolated_registry_but_deployed_champion(self) -> None:
        root = Path("/repo")
        output_dir = root / "model" / ".validation-artifacts"
        paths = training_paths(root, output_dir)

        self.assertEqual(paths["artifact_dir"], output_dir)
        self.assertEqual(paths["registry_path"], output_dir / "registry.json")
        self.assertEqual(paths["report_path"], output_dir / "RISK_STACK_REPORT.md")
        self.assertEqual(paths["champion_path"], root / "model" / "artifacts" / "crix-monoboost-v2.json")
        self.assertNotEqual(paths["registry_path"], root / "model" / "artifacts" / "registry.json")

    def test_retrain_comparison_accepts_tiny_metric_drift_but_rejects_contract_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            committed = root / "committed"
            generated = root / "generated"
            committed.mkdir()
            generated.mkdir()
            minimal = {
                "schemaVersion": 1,
                "name": "CRIX-LifetimePD",
                "version": "1.0.0",
                "product": "unsecured-personal-loan",
                "target": {"maxHorizonMonths": 36},
                "metrics": {"12m": {"auc": 0.6, "samples": 1000}},
                "model": {"method": "discrete-time-logistic-hazard", "featureNames": ["x"], "maxHorizonMonths": 36, "seed": 42},
            }
            for base in (committed, generated):
                for name in (
                    "crix-lifetime-pd-v1.json",
                    "crix-ead-installment-v1.json",
                    "crix-lgd-v1.json",
                    "crix-transitions-taiwan-v1.json",
                    "crix-ccf-taiwan-v1.json",
                ):
                    payload = dict(minimal)
                    payload["name"] = name.removesuffix(".json")
                    (base / name).write_text(json.dumps(payload))
                for name in ("snapshot-lifecycle-v1.json", "snapshot-v3-retrospective.json"):
                    (base / name).write_text(json.dumps({"snapshotId": name}))

            drifted = json.loads((generated / "crix-lifetime-pd-v1.json").read_text())
            drifted["metrics"]["12m"]["auc"] += 1e-6
            (generated / "crix-lifetime-pd-v1.json").write_text(json.dumps(drifted))
            self.assertEqual(compare_artifacts(committed, generated)["status"], "passed")

            drifted["target"]["maxHorizonMonths"] = 24
            (generated / "crix-lifetime-pd-v1.json").write_text(json.dumps(drifted))
            with self.assertRaises(AssertionError):
                compare_artifacts(committed, generated)


if __name__ == "__main__":
    unittest.main()
