from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


LINEAGE_SCHEMA_VERSION = 1
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


class LineageViolation(ValueError):
    pass


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(root: Path, path: Path) -> str:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise LineageViolation("lineage paths must remain inside the repository root") from exc
    if relative.is_absolute() or ".." in relative.parts:
        raise LineageViolation("lineage paths must be repository-relative")
    return relative.as_posix()


def _validate_relative_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise LineageViolation(f"{field} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise LineageViolation(f"{field} must be repository-relative")
    return value


def _resolve_under_root(root: Path, relative: object, field: str) -> Path:
    text = _validate_relative_text(relative, field)
    resolved_root = root.resolve()
    candidate = (resolved_root / text).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise LineageViolation(f"{field} resolves outside the repository root") from exc
    return candidate


def _validate_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise LineageViolation(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_git_commit(value: object) -> str:
    if not isinstance(value, str) or not _HEX40.fullmatch(value):
        raise LineageViolation("gitCommit must be a full lowercase 40-character commit SHA")
    return value


def _safe_evidence(paths: Sequence[Path], root: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append(
            {
                "path": _relative_path(root, path),
                "sha256": sha256_file(path),
            }
        )
    return entries


def build_training_manifest(
    *,
    root: Path,
    runtime_manifest: Mapping[str, Any],
    artifact_path: Path,
    dependency_lock_path: Path,
    evidence_paths: Sequence[Path],
    dataset_path: Path | None,
    run_id: str,
    git_commit: str,
    data_contract_version: str,
    seeds: Mapping[str, int],
    created_at: str | None = None,
    previous_manifest_path: Path | None = None,
) -> dict[str, object]:
    if not run_id or not data_contract_version:
        raise LineageViolation("runId and dataContractVersion are required")
    _validate_git_commit(git_commit)
    if not artifact_path.is_file() or not dependency_lock_path.is_file():
        raise FileNotFoundError("artifact and dependency lock must exist")

    datasets = runtime_manifest.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1 or not isinstance(datasets[0], dict):
        raise LineageViolation("runtime manifest must contain exactly one primary dataset entry")
    source = dict(datasets[0])
    source_sha = source.get("localSha256")
    _validate_sha256(source_sha, "dataset.sourceSha256")
    if dataset_path is not None and dataset_path.exists() and sha256_file(dataset_path) != source_sha:
        raise LineageViolation("dataset checksum does not match the approved runtime manifest")

    split = runtime_manifest.get("split")
    if not isinstance(split, dict) or set(split) != {"train", "calibration", "test"}:
        raise LineageViolation("runtime manifest split metadata is incomplete")

    previous: dict[str, str] | None = None
    if previous_manifest_path is not None:
        if not previous_manifest_path.is_file():
            raise FileNotFoundError(previous_manifest_path)
        previous = {
            "path": _relative_path(root, previous_manifest_path),
            "sha256": sha256_file(previous_manifest_path),
        }

    manifest = {
        "schemaVersion": LINEAGE_SCHEMA_VERSION,
        "runId": str(run_id),
        "createdAt": created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gitCommit": git_commit,
        "gitCommitScope": "approved-artifact-origin-code-revision",
        "pythonVersion": platform.python_version(),
        "pythonVersionScope": "v3.2-lineage-validation-environment",
        "dependencyLock": {
            "path": _relative_path(root, dependency_lock_path),
            "sha256": sha256_file(dependency_lock_path),
            "scope": "v3.2-lineage-validation-environment",
            "kind": "declared-requirements-specification",
        },
        "dataset": {
            "sourceId": source.get("id"),
            "doi": source.get("doi"),
            "version": source.get("version"),
            "sourceMd5": source.get("sourceMd5"),
            "sourceSha256": source_sha,
            "rows": {
                name: int(details["samples"])
                for name, details in split.items()
                if isinstance(details, dict) and "samples" in details
            },
        },
        "split": split,
        "seeds": {str(key): int(value) for key, value in seeds.items()},
        "featureContractVersion": runtime_manifest.get("featureContractVersion"),
        "dataContractVersion": data_contract_version,
        "artifact": {
            "path": _relative_path(root, artifact_path),
            "sha256": sha256_file(artifact_path),
        },
        "evidence": _safe_evidence(evidence_paths, root),
        "previousManifest": previous,
        "runtimeManifest": {
            "modelId": runtime_manifest.get("modelId"),
            "artifactSha256": runtime_manifest.get("artifactSha256"),
            "trainingRunId": runtime_manifest.get("trainingRunId"),
            "trainingGitCommit": runtime_manifest.get("gitCommit"),
            "trainingEnvironmentHash": runtime_manifest.get("environmentHash"),
        },
        "reproducibilityClass": (
            "tamper-evident retrospective lineage for the approved artifact plus v3.2 validation evidence; "
            "the original manifest environment hash is preserved separately, and deterministic seeds/hashes "
            "do not guarantee bitwise retraining across numerical-library/platform changes"
        ),
        "privacy": "aggregate counts, versions and hashes only; no borrower rows, IDs, secrets or absolute paths",
    }
    validate_manifest_schema(manifest)
    if manifest["artifact"]["sha256"] != runtime_manifest.get("artifactSha256"):
        raise LineageViolation("artifact hash does not match approved runtime manifest")
    return manifest


def validate_manifest_schema(manifest: Mapping[str, Any]) -> None:
    required = {
        "schemaVersion",
        "runId",
        "createdAt",
        "gitCommit",
        "gitCommitScope",
        "pythonVersion",
        "pythonVersionScope",
        "dependencyLock",
        "dataset",
        "split",
        "seeds",
        "featureContractVersion",
        "dataContractVersion",
        "artifact",
        "evidence",
        "previousManifest",
        "runtimeManifest",
        "reproducibilityClass",
        "privacy",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise LineageViolation(f"lineage manifest missing required field(s): {', '.join(missing)}")
    if manifest.get("schemaVersion") != LINEAGE_SCHEMA_VERSION:
        raise LineageViolation("unsupported lineage schema version")
    _validate_git_commit(manifest.get("gitCommit"))
    if manifest.get("gitCommitScope") != "approved-artifact-origin-code-revision":
        raise LineageViolation("unsupported gitCommitScope")
    if manifest.get("pythonVersionScope") != "v3.2-lineage-validation-environment":
        raise LineageViolation("unsupported pythonVersionScope")

    lock = manifest.get("dependencyLock")
    artifact = manifest.get("artifact")
    dataset = manifest.get("dataset")
    if not isinstance(lock, dict) or not isinstance(artifact, dict) or not isinstance(dataset, dict):
        raise LineageViolation("lineage dependencyLock, artifact and dataset must be objects")
    _validate_relative_text(lock.get("path"), "dependencyLock.path")
    _validate_sha256(lock.get("sha256"), "dependencyLock.sha256")
    if lock.get("scope") != "v3.2-lineage-validation-environment":
        raise LineageViolation("unsupported dependencyLock.scope")
    if lock.get("kind") != "declared-requirements-specification":
        raise LineageViolation("unsupported dependencyLock.kind")
    _validate_relative_text(artifact.get("path"), "artifact.path")
    _validate_sha256(artifact.get("sha256"), "artifact.sha256")
    _validate_sha256(dataset.get("sourceSha256"), "dataset.sourceSha256")

    evidence = manifest.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise LineageViolation("lineage evidence must contain at least one hashed report")
    for index, entry in enumerate(evidence):
        if not isinstance(entry, dict):
            raise LineageViolation(f"evidence[{index}] must be an object")
        _validate_relative_text(entry.get("path"), f"evidence[{index}].path")
        _validate_sha256(entry.get("sha256"), f"evidence[{index}].sha256")

    previous = manifest.get("previousManifest")
    if previous is not None:
        if not isinstance(previous, dict):
            raise LineageViolation("previousManifest must be null or an object")
        _validate_relative_text(previous.get("path"), "previousManifest.path")
        _validate_sha256(previous.get("sha256"), "previousManifest.sha256")

    seeds = manifest.get("seeds")
    if not isinstance(seeds, dict) or not seeds or any(not isinstance(value, int) for value in seeds.values()):
        raise LineageViolation("lineage seeds must be a non-empty integer mapping")


def _verify_previous_manifest_chain(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    seen: set[Path],
) -> int:
    depth = 0
    current = manifest
    while current.get("previousManifest") is not None:
        previous = current["previousManifest"]
        if not isinstance(previous, dict):
            raise LineageViolation("previousManifest must be null or an object")
        previous_path = _resolve_under_root(root, previous.get("path"), "previousManifest.path")
        if previous_path in seen:
            raise LineageViolation("lineage manifest chain contains a cycle")
        seen.add(previous_path)
        if not previous_path.is_file() or sha256_file(previous_path) != _validate_sha256(previous.get("sha256"), "previousManifest.sha256"):
            raise LineageViolation("previous manifest hash-chain verification failed")
        try:
            previous_manifest = json.loads(previous_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LineageViolation("unable to load previous lineage manifest") from exc
        if not isinstance(previous_manifest, dict):
            raise LineageViolation("previous lineage manifest root must be an object")
        validate_manifest_schema(previous_manifest)
        current = previous_manifest
        depth += 1
    return depth


def verify_training_manifest(
    manifest_path: Path,
    *,
    root: Path,
    dataset_path: Path | None = None,
    expected_git_commit: str | None = None,
    expected_data_contract_version: str | None = None,
) -> dict[str, object]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LineageViolation("unable to load lineage manifest") from exc
    if not isinstance(manifest, dict):
        raise LineageViolation("lineage manifest root must be an object")
    validate_manifest_schema(manifest)

    if expected_git_commit is not None and manifest["gitCommit"] != expected_git_commit:
        raise LineageViolation("lineage gitCommit does not match the expected approved revision")
    if (
        expected_data_contract_version is not None
        and manifest["dataContractVersion"] != expected_data_contract_version
    ):
        raise LineageViolation("lineage dataContractVersion does not match the expected contract")

    lock = manifest["dependencyLock"]
    dependency_path = _resolve_under_root(root, lock["path"], "dependencyLock.path")
    if not dependency_path.is_file() or sha256_file(dependency_path) != lock["sha256"]:
        raise LineageViolation("dependency lock hash verification failed")

    artifact = manifest["artifact"]
    artifact_path = _resolve_under_root(root, artifact["path"], "artifact.path")
    if not artifact_path.is_file() or sha256_file(artifact_path) != artifact["sha256"]:
        raise LineageViolation("artifact hash verification failed")

    for entry in manifest["evidence"]:
        evidence_path = _resolve_under_root(root, entry["path"], "evidence.path")
        if not evidence_path.is_file() or sha256_file(evidence_path) != entry["sha256"]:
            raise LineageViolation(f"evidence hash verification failed for {entry['path']}")

    previous = manifest["previousManifest"]
    chain_depth = _verify_previous_manifest_chain(
        manifest,
        root=root,
        seen={manifest_path.resolve()},
    )

    if dataset_path is not None and dataset_path.exists():
        if sha256_file(dataset_path) != manifest["dataset"]["sourceSha256"]:
            raise LineageViolation("dataset checksum verification failed")

    runtime = manifest["runtimeManifest"]
    if runtime.get("artifactSha256") != manifest["artifact"]["sha256"]:
        raise LineageViolation("runtime artifact digest and lineage artifact digest disagree")

    return {
        "status": "pass",
        "schemaVersion": manifest["schemaVersion"],
        "runId": manifest["runId"],
        "gitCommit": manifest["gitCommit"],
        "modelId": runtime.get("modelId"),
        "dataContractVersion": manifest["dataContractVersion"],
        "evidenceFiles": len(manifest["evidence"]),
        "previousManifestLinked": previous is not None,
        "manifestChainDepth": chain_depth,
        "privacy": "safe aggregate verification output only",
    }


def _load_runtime_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LineageViolation("runtime manifest must be an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or verify CRIX tamper-evident training lineage")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate")
    generate.add_argument("--root", type=Path, default=Path("."))
    generate.add_argument("--runtime-manifest", type=Path, required=True)
    generate.add_argument("--artifact", type=Path, required=True)
    generate.add_argument("--dependency-lock", type=Path, required=True)
    generate.add_argument("--dataset", type=Path)
    generate.add_argument("--evidence", type=Path, nargs="+", required=True)
    generate.add_argument("--run-id", required=True)
    generate.add_argument("--git-commit", required=True)
    generate.add_argument("--data-contract-version", required=True)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--created-at")
    generate.add_argument("--previous-manifest", type=Path)
    generate.add_argument("--output", type=Path, required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--root", type=Path, default=Path("."))
    verify.add_argument("--dataset", type=Path)
    verify.add_argument("--expected-git-commit")
    verify.add_argument("--expected-data-contract-version")

    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "generate":
        manifest = build_training_manifest(
            root=root,
            runtime_manifest=_load_runtime_manifest(args.runtime_manifest),
            artifact_path=args.artifact,
            dependency_lock_path=args.dependency_lock,
            evidence_paths=args.evidence,
            dataset_path=args.dataset,
            run_id=args.run_id,
            git_commit=args.git_commit,
            data_contract_version=args.data_contract_version,
            seeds={"training": args.seed, "governance": args.seed},
            created_at=args.created_at,
            previous_manifest_path=args.previous_manifest,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(manifest, sort_keys=True))
        return

    result = verify_training_manifest(
        args.manifest,
        root=root,
        dataset_path=args.dataset,
        expected_git_commit=args.expected_git_commit,
        expected_data_contract_version=args.expected_data_contract_version,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
