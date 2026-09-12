from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

REGISTRY_SCHEMA_VERSION = 1


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_manifest(artifact_path: Path, manifest: Mapping[str, object]) -> bool:
    expected = manifest.get("artifactSha256")
    return isinstance(expected, str) and len(expected) == 64 and artifact_path.is_file() and sha256_file(artifact_path) == expected


def build_manifest(
    *,
    model_id: str,
    model_name: str,
    version: str,
    artifact_path: Path,
    artifact_schema_version: int,
    product: str,
    target: Mapping[str, object],
    git_commit: str,
    datasets: list[Mapping[str, object]],
    feature_contract_version: str,
    split: Mapping[str, object],
    random_seed: int,
    training_config: Mapping[str, object],
    environment_hash: str,
    metrics: Mapping[str, object],
    status: str,
    model_card: str,
    report: str,
    parent: str | None = None,
) -> dict[str, object]:
    if not model_id or not model_name or not version:
        raise ValueError("model identity is required")
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    if status not in {"research", "challenger", "approved-demo-champion", "retired"}:
        raise ValueError(f"unsupported model status: {status}")
    return {
        "manifestSchemaVersion": 1,
        "modelId": model_id,
        "modelName": model_name,
        "version": version,
        "artifactSchemaVersion": artifact_schema_version,
        "product": product,
        "target": dict(target),
        "gitCommit": git_commit,
        "datasets": [dict(item) for item in datasets],
        "featureContractVersion": feature_contract_version,
        "split": dict(split),
        "randomSeed": random_seed,
        "trainingConfigHash": canonical_json_hash(training_config),
        "environmentHash": environment_hash,
        "artifactSha256": sha256_file(artifact_path),
        "metrics": dict(metrics),
        "status": status,
        "modelCard": model_card,
        "report": report,
        "parent": parent,
    }


def _atomic_json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def update_registry_index(index_path: Path, entry: Mapping[str, object]) -> None:
    model_id = entry.get("modelId")
    model_name = entry.get("modelName")
    version = entry.get("version")
    digest = entry.get("artifactSha256")
    if not all(isinstance(value, str) and value for value in (model_id, model_name, version, digest)):
        raise ValueError("registry entry requires modelId, modelName, version and artifactSha256")

    if index_path.exists():
        current = json.loads(index_path.read_text())
    else:
        current = {"schemaVersion": REGISTRY_SCHEMA_VERSION, "models": {}}
    if current.get("schemaVersion") != REGISTRY_SCHEMA_VERSION or not isinstance(current.get("models"), dict):
        raise ValueError("invalid registry index schema")

    models: dict[str, object] = current["models"]
    for existing in models.values():
        if not isinstance(existing, dict):
            raise ValueError("invalid registry entry")
        if existing.get("modelName") == model_name and existing.get("version") == version:
            if existing.get("artifactSha256") != digest:
                raise ValueError("same model name/version points to a different artifact digest")
    models[str(model_id)] = dict(entry)
    _atomic_json_write(index_path, current)
