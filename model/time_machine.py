from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from registry import canonical_json_hash

SnapshotMode = Literal["point-in-time", "retrospective-resolved"]


@dataclass(frozen=True)
class SnapshotSpec:
    train_end: str
    calibration_end: str
    decision_end: str
    outcome_cutoff: str
    source_sha256: str
    mode: SnapshotMode = "point-in-time"

    def validate(self) -> None:
        dates = np.array(
            [self.train_end, self.calibration_end, self.decision_end, self.outcome_cutoff],
            dtype="datetime64[D]",
        )
        if np.isnat(dates).any():
            raise ValueError("snapshot dates must be valid")
        if not (dates[0] < dates[1] < dates[2] <= dates[3]):
            raise ValueError("snapshot windows must be strictly chronological")
        if self.mode not in {"point-in-time", "retrospective-resolved"}:
            raise ValueError("unsupported snapshot mode")
        if len(self.source_sha256) != 64:
            raise ValueError("snapshot source checksum must be SHA-256")


@dataclass(frozen=True)
class SnapshotMasks:
    train: np.ndarray
    calibration: np.ndarray
    decision: np.ndarray
    evaluable: np.ndarray


def snapshot_id(spec: SnapshotSpec, feature_contract_version: str) -> str:
    spec.validate()
    if not feature_contract_version:
        raise ValueError("feature contract version is required")
    return canonical_json_hash({"spec": asdict(spec), "featureContractVersion": feature_contract_version})[:24]


def backward_asof_join(left_times: np.ndarray, right_times: np.ndarray, right_values: np.ndarray) -> np.ndarray:
    left = np.asarray(left_times, dtype="datetime64[ns]").reshape(-1)
    right = np.asarray(right_times, dtype="datetime64[ns]").reshape(-1)
    values = np.asarray(right_values).reshape(-1)
    if len(right) == 0 or len(right) != len(values):
        raise ValueError("right-side timestamps and values must be non-empty and aligned")
    if np.isnat(left).any() or np.isnat(right).any():
        raise ValueError("as-of joins do not accept missing timestamps")
    order = np.argsort(right, kind="stable")
    sorted_right = right[order]
    sorted_values = values[order]
    positions = np.searchsorted(sorted_right, left, side="right") - 1

    if np.issubdtype(sorted_values.dtype, np.number):
        output = np.full(len(left), np.nan, dtype=float)
    else:
        output = np.full(len(left), None, dtype=object)
    valid = positions >= 0
    output[valid] = sorted_values[positions[valid]]
    return output


def build_snapshot_masks(
    issue_dates: np.ndarray,
    outcome_available_at: np.ndarray,
    spec: SnapshotSpec,
) -> SnapshotMasks:
    spec.validate()
    issue = np.asarray(issue_dates, dtype="datetime64[D]").reshape(-1)
    outcome = np.asarray(outcome_available_at, dtype="datetime64[D]").reshape(-1)
    if len(issue) == 0 or len(issue) != len(outcome):
        raise ValueError("issue/outcome availability arrays must be non-empty and aligned")
    if np.isnat(issue).any() or np.isnat(outcome).any():
        raise ValueError("snapshot inputs contain missing dates")

    train_end = np.datetime64(spec.train_end, "D")
    calibration_end = np.datetime64(spec.calibration_end, "D")
    decision_end = np.datetime64(spec.decision_end, "D")
    outcome_cutoff = np.datetime64(spec.outcome_cutoff, "D")

    train_issue = issue <= train_end
    calibration_issue = (issue > train_end) & (issue <= calibration_end)
    decision = (issue > calibration_end) & (issue <= decision_end)
    evaluable = outcome <= outcome_cutoff

    if spec.mode == "point-in-time":
        train = train_issue & (outcome <= train_end)
        calibration = calibration_issue & (outcome <= calibration_end)
    else:
        train = train_issue & evaluable
        calibration = calibration_issue & evaluable

    return SnapshotMasks(train=train, calibration=calibration, decision=decision, evaluable=evaluable)


def snapshot_manifest(spec: SnapshotSpec, feature_contract_version: str, masks: SnapshotMasks) -> dict[str, object]:
    return {
        "snapshotId": snapshot_id(spec, feature_contract_version),
        "mode": spec.mode,
        "featureContractVersion": feature_contract_version,
        "sourceSha256": spec.source_sha256,
        "windows": {
            "trainEnd": spec.train_end,
            "calibrationEnd": spec.calibration_end,
            "decisionEnd": spec.decision_end,
            "outcomeCutoff": spec.outcome_cutoff,
        },
        "counts": {
            "train": int(masks.train.sum()),
            "calibration": int(masks.calibration.sum()),
            "decision": int(masks.decision.sum()),
            "evaluable": int(masks.evaluable.sum()),
        },
    }
