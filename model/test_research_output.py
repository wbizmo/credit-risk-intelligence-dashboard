from __future__ import annotations

import unittest
from pathlib import Path

import train_risk_stack


class ResearchOutputIsolationTests(unittest.TestCase):
    def test_validation_output_uses_isolated_registry_but_deployed_champion(self) -> None:
        self.assertTrue(
            hasattr(train_risk_stack, "training_paths"),
            "research retrains need an isolated output/registry path once model versions are published",
        )
        root = Path("/repo")
        output_dir = root / "model" / ".validation-artifacts"
        paths = train_risk_stack.training_paths(root, output_dir)

        self.assertEqual(paths["artifact_dir"], output_dir)
        self.assertEqual(paths["registry_path"], output_dir / "registry.json")
        self.assertEqual(paths["report_path"], output_dir / "RISK_STACK_REPORT.md")
        self.assertEqual(paths["champion_path"], root / "model" / "artifacts" / "crix-monoboost-v2.json")
        self.assertNotEqual(paths["registry_path"], root / "model" / "artifacts" / "registry.json")


if __name__ == "__main__":
    unittest.main()
