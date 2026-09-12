from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

RESEARCH_ARTIFACTS = (
    "crix-lifetime-pd-v1.json",
    "crix-ead-installment-v1.json",
    "crix-lgd-v1.json",
    "crix-transitions-taiwan-v1.json",
    "crix-ccf-taiwan-v1.json",
)
SNAPSHOTS = ("snapshot-lifecycle-v1.json", "snapshot-v3-retrospective.json")


def training_paths(root: Path, artifact_dir: Path) -> dict[str, Path]:
    """Resolve retrain outputs without redirecting the deployed champion input."""
    return {
        "artifact_dir": artifact_dir,
        "registry_path": artifact_dir / "registry.json",
        "report_path": artifact_dir / "RISK_STACK_REPORT.md",
        "champion_path": root / "model" / "artifacts" / "crix-monoboost-v2.json",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_lifecycle_source(source_model: Path) -> Path:
    """Hydrate the checksum-verified public lifecycle source when CI cache is cold."""
    from research_data import download_lifecycle_lendingclub

    lifecycle = source_model / "data" / "loan_data_2007_2014.csv"
    if not lifecycle.exists():
        download_lifecycle_lendingclub(lifecycle)
    return lifecycle


def prepare_workspace(root: Path, workspace: Path) -> Path:
    """Create an isolated model workspace for validation-only retraining."""
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    source_model = root / "model"
    _ensure_lifecycle_source(source_model)
    target_model = workspace / "model"
    shutil.copytree(
        source_model,
        target_model,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "data", ".validation-artifacts"),
    )
    os.symlink(source_model.joinpath("data").resolve(), target_model / "data", target_is_directory=True)
    model_card = root / "MODEL_CARD.md"
    if model_card.exists():
        shutil.copy2(model_card, workspace / "MODEL_CARD.md")

    artifact_dir = target_model / "artifacts"
    for name in RESEARCH_ARTIFACTS:
        (artifact_dir / name).unlink(missing_ok=True)
        (artifact_dir / name.replace(".json", ".manifest.json")).unlink(missing_ok=True)
    for name in SNAPSHOTS:
        (artifact_dir / name).unlink(missing_ok=True)
    (artifact_dir / "registry.json").unlink(missing_ok=True)
    (target_model / "RISK_STACK_REPORT.md").unlink(missing_ok=True)
    return target_model


def _assert_close_tree(expected: Any, actual: Any, path: str, tolerance: float = 1e-4) -> None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual:
            raise AssertionError(f"{path}: {actual!r} != {expected!r}")
        return
    if isinstance(expected, int) and isinstance(actual, int):
        if expected != actual:
            raise AssertionError(f"{path}: {actual} != {expected}")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        delta = abs(float(actual) - float(expected))
        allowed = tolerance * max(1.0, abs(float(expected)))
        if delta > allowed:
            raise AssertionError(f"{path}: delta {delta} exceeds {allowed}")
        return
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            raise AssertionError(f"{path}: key mismatch {set(actual) ^ set(expected)}")
        for key in expected:
            _assert_close_tree(expected[key], actual[key], f"{path}.{key}", tolerance)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise AssertionError(f"{path}: length {len(actual)} != {len(expected)}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _assert_close_tree(left, right, f"{path}[{index}]", tolerance)
        return
    if expected != actual:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def compare_artifacts(committed_dir: Path, generated_dir: Path, tolerance: float = 1e-4) -> dict[str, object]:
    """Validate fresh research fits against frozen published contracts and OOT metrics."""
    results: list[dict[str, object]] = []
    common_contract_keys = (
        "schemaVersion",
        "name",
        "version",
        "product",
        "target",
        "featureNames",
        "populationConditioning",
        "targetBounds",
        "targetScale",
        "v3RuntimeBaseline",
        "discountRate",
        "recoveryTimingSupport",
        "targetPolicy",
        "warning",
        "method",
        "stateTaxonomy",
        "sourceChronology",
        "nonDefaultOctoberState",
        "rows",
        "observedTransitions",
        "counts",
        "matrix",
    )
    for name in RESEARCH_ARTIFACTS:
        committed_path = committed_dir / name
        generated_path = generated_dir / name
        committed = json.loads(committed_path.read_text())
        generated = json.loads(generated_path.read_text())
        for key in common_contract_keys:
            if key in committed or key in generated:
                _assert_close_tree(committed.get(key), generated.get(key), f"{name}.{key}", tolerance)
        _assert_close_tree(committed.get("metrics"), generated.get("metrics"), f"{name}.metrics", tolerance)

        if isinstance(committed.get("model"), dict):
            for key in ("method", "featureNames", "maxHorizonMonths", "seed"):
                if key in committed["model"] or key in generated.get("model", {}):
                    _assert_close_tree(
                        committed["model"].get(key), generated.get("model", {}).get(key), f"{name}.model.{key}", tolerance
                    )
        if isinstance(committed.get("models"), dict):
            _assert_close_tree(
                committed["models"].get("recommended"),
                generated.get("models", {}).get("recommended"),
                f"{name}.models.recommended",
                tolerance,
            )

        results.append(
            {
                "artifact": name,
                "publishedSha256": _sha256(committed_path),
                "validationSha256": _sha256(generated_path),
                "status": "validated-within-tolerance",
            }
        )

    for name in SNAPSHOTS:
        committed = json.loads((committed_dir / name).read_text())
        generated = json.loads((generated_dir / name).read_text())
        _assert_close_tree(committed, generated, name, tolerance=0.0)

    return {"status": "passed", "metricTolerance": tolerance, "artifacts": results, "snapshots": list(SNAPSHOTS)}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate immutable CRIX research models with non-destructive fresh retrains")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--workspace", type=Path, required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--generated", type=Path, required=True)
    compare.add_argument("--committed", type=Path, default=root / "model" / "artifacts")
    compare.add_argument("--report", type=Path)
    compare.add_argument("--tolerance", type=float, default=1e-4)

    args = parser.parse_args()
    if args.command == "prepare":
        model_dir = prepare_workspace(root, args.workspace)
        print(model_dir)
        return

    result = compare_artifacts(args.committed, args.generated, args.tolerance)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
