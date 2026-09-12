from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from datasets import DATASET_REGISTRY, LENDINGCLUB_MD5
from registry import build_manifest, sha256_file, update_registry_index

FEATURE_CONTRACT_VERSION = "crix-granting-features-v1"
MODEL_ID = "CRIX-MonoBoost@2.0.0"


def git_commit(root: Path) -> str:
    supplied = os.environ.get("CRIX_TRAINING_GIT_SHA") or os.environ.get("GITHUB_SHA")
    if supplied:
        return supplied
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact_path = root / "model" / "artifacts" / "crix-monoboost-v2.json"
    manifest_path = root / "model" / "artifacts" / "crix-monoboost-v2.manifest.json"
    registry_path = root / "model" / "artifacts" / "registry.json"
    data_path = root / "model" / "data" / "LC_loans_granting_model_dataset.csv"
    requirements_path = root / "model" / "requirements.txt"
    artifact = json.loads(artifact_path.read_text())
    lendingclub = DATASET_REGISTRY["lendingclub"]

    datasets = [{
        "id": "lendingclub-granting-0.1",
        "product": lendingclub["product"],
        "version": lendingclub["version"],
        "doi": lendingclub["doi"],
        "license": lendingclub["license"],
        "sourceMd5": LENDINGCLUB_MD5,
        "localSha256": sha256_file(data_path),
    }]
    training_config = {
        "seed": 42,
        "featureNames": artifact["featureNames"],
        "monotoneConstraints": artifact["monotoneConstraints"],
        "treeCount": len(artifact["trees"]),
        "calibrationMethod": artifact["calibration"]["method"],
    }
    manifest = build_manifest(
        model_id=MODEL_ID,
        model_name=artifact["name"],
        version=artifact["version"],
        artifact_path=artifact_path,
        artifact_schema_version=artifact["schemaVersion"],
        product=lendingclub["product"],
        target=artifact["target"],
        git_commit=git_commit(root),
        datasets=datasets,
        feature_contract_version=FEATURE_CONTRACT_VERSION,
        split=artifact["training"]["split"],
        random_seed=42,
        training_config=training_config,
        environment_hash=sha256_file(requirements_path),
        metrics=artifact["metrics"],
        status="approved-demo-champion",
        model_card="MODEL_CARD.md",
        report="model/TRAINING_REPORT.md",
        parent=None,
    )
    atomic_json(manifest_path, manifest)
    update_registry_index(registry_path, {
        "modelId": MODEL_ID,
        "modelName": artifact["name"],
        "version": artifact["version"],
        "artifactSha256": manifest["artifactSha256"],
        "manifest": "model/artifacts/crix-monoboost-v2.manifest.json",
        "artifact": "model/artifacts/crix-monoboost-v2.json",
        "status": manifest["status"],
        "featureContractVersion": FEATURE_CONTRACT_VERSION,
    })
    print(json.dumps({"manifest": manifest, "registry": json.loads(registry_path.read_text())}, sort_keys=True))


if __name__ == "__main__":
    main()
