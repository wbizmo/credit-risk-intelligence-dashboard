from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from governance import FeatureProvenance, validate_feature_provenance


PRIMARY_SOURCE_CONTRACT_VERSION = "lendingclub-source-v1"
PRIMARY_HARMONIZED_CONTRACT_VERSION = "crix-lendingclub-harmonized-v1"
EXTERNAL_CONTRACT_VERSIONS = {
    "uci-taiwan-credit-card-default": "uci-taiwan-credit-card-v1",
    "uci-statlog-german-credit": "uci-statlog-german-v1",
    "uci-south-german-credit": "uci-south-german-v1",
}
EXTERNAL_CONTRACTS = {
    "uci-taiwan-credit-card-default": {
        "product": "revolving-credit-card",
        "target": "default-payment-next-month",
        "requiredSourceFeatures": (
            "creditUtilization6mMean",
            "onTimePaymentRate6m",
            "paymentDelayMonths6m",
            "recentCreditGrowth6m",
        ),
    },
    "uci-statlog-german-credit": {
        "product": "consumer-instalment-credit",
        "target": "bad-credit-risk",
        "requiredSourceFeatures": (
            "Attribute2",
            "Attribute5",
            "Attribute7",
            "Attribute8",
            "Attribute16",
        ),
    },
    "uci-south-german-credit": {
        "product": "consumer-instalment-credit",
        "target": "bad-credit-risk",
        "requiredSourceFeatures": (
            "duration",
            "amount",
            "employment_duration",
            "installment_rate",
            "number_credits",
        ),
    },
}

_SOURCE_COLUMNS = (
    "issue_d",
    "revenue",
    "dti_n",
    "loan_amnt",
    "fico_n",
    "emp_length",
    "Default",
)
_HARMONIZED_COLUMNS = (
    "sourceRowId",
    "issueDate",
    "featureAsOf",
    "annualIncome",
    "loanAmount",
    "debtToIncome",
    "loanToIncome",
    "creditScore",
    "employmentYears",
    "target",
)


@dataclass(frozen=True)
class ContractEvidence:
    contract: str
    version: str
    rows: int
    columns: int
    status: str = "pass"

    def to_dict(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "version": self.version,
            "rows": self.rows,
            "columns": self.columns,
            "status": self.status,
        }


class ContractViolation(ValueError):
    def __init__(
        self,
        *,
        contract: str,
        version: str,
        invariant: str,
        invalid_count: int,
        summary: Mapping[str, object] | None = None,
    ) -> None:
        self.contract = contract
        self.version = version
        self.invariant = invariant
        self.invalid_count = int(invalid_count)
        self.summary = dict(summary or {})
        safe_summary = ", ".join(f"{key}={value}" for key, value in sorted(self.summary.items()))
        suffix = f" ({safe_summary})" if safe_summary else ""
        super().__init__(
            f"{contract}@{version} invariant={invariant} invalid_count={self.invalid_count}{suffix}"
        )


def _missing_columns(frame: pd.DataFrame, required: Sequence[str]) -> list[str]:
    return [column for column in required if column not in frame.columns]


def _raise_missing(contract: str, version: str, missing: Sequence[str]) -> None:
    if missing:
        raise ContractViolation(
            contract=contract,
            version=version,
            invariant="required-columns",
            invalid_count=len(missing),
            summary={"missingColumns": ",".join(sorted(missing))},
        )


def _finite_values(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    contract: str,
    version: str,
) -> None:
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        invalid = ~np.isfinite(numeric)
        if invalid.any():
            raise ContractViolation(
                contract=contract,
                version=version,
                invariant=f"{column}-finite",
                invalid_count=int(invalid.sum()),
            )


def validate_primary_source(frame: pd.DataFrame) -> dict[str, object]:
    contract = "lendingclub-source"
    version = PRIMARY_SOURCE_CONTRACT_VERSION
    _raise_missing(contract, version, _missing_columns(frame, _SOURCE_COLUMNS))
    if not frame.index.is_unique:
        raise ContractViolation(
            contract=contract,
            version=version,
            invariant="source-row-index-unique",
            invalid_count=int(frame.index.duplicated().sum()),
        )

    target = pd.to_numeric(frame["Default"], errors="coerce")
    invalid_target = target.notna() & ~target.isin([0, 1])
    if invalid_target.any():
        raise ContractViolation(
            contract=contract,
            version=version,
            invariant="target-domain",
            invalid_count=int(invalid_target.sum()),
        )

    dates = pd.to_datetime(frame["issue_d"], errors="coerce", format="mixed")
    parseable = int(dates.notna().sum())
    if len(frame) and parseable / len(frame) < 0.95:
        raise ContractViolation(
            contract=contract,
            version=version,
            invariant="issue-date-parse-rate",
            invalid_count=int(len(frame) - parseable),
            summary={"minimumParseRate": 0.95},
        )

    return ContractEvidence(contract, version, len(frame), len(frame.columns)).to_dict()


def validate_primary_harmonized(
    frame: pd.DataFrame,
    *,
    provenance: Sequence[FeatureProvenance],
    required_features: Sequence[str],
) -> dict[str, object]:
    contract = "crix-primary-harmonized"
    version = PRIMARY_HARMONIZED_CONTRACT_VERSION
    _raise_missing(contract, version, _missing_columns(frame, _HARMONIZED_COLUMNS))
    if frame.empty:
        raise ContractViolation(
            contract=contract,
            version=version,
            invariant="non-empty",
            invalid_count=1,
        )

    _finite_values(
        frame,
        (
            "annualIncome",
            "loanAmount",
            "debtToIncome",
            "loanToIncome",
            "creditScore",
            "employmentYears",
            "target",
        ),
        contract=contract,
        version=version,
    )

    checks = {
        "annualIncome-positive": frame["annualIncome"].to_numpy(dtype=float) > 0,
        "loanAmount-positive": frame["loanAmount"].to_numpy(dtype=float) > 0,
        "debtToIncome-range": frame["debtToIncome"].to_numpy(dtype=float) >= 0,
        "debtToIncome-upper-bound": frame["debtToIncome"].to_numpy(dtype=float) <= 2.0,
        "loanToIncome-range": frame["loanToIncome"].to_numpy(dtype=float) >= 0.001,
        "loanToIncome-upper-bound": frame["loanToIncome"].to_numpy(dtype=float) <= 3.0,
        "creditScore-range": frame["creditScore"].to_numpy(dtype=float) >= 300,
        "creditScore-upper-bound": frame["creditScore"].to_numpy(dtype=float) <= 850,
        "employmentYears-range": frame["employmentYears"].to_numpy(dtype=float) >= 0,
        "employmentYears-upper-bound": frame["employmentYears"].to_numpy(dtype=float) <= 10,
        "target-domain": frame["target"].isin([0, 1]).to_numpy(),
    }
    for invariant, valid in checks.items():
        invalid_count = int((~np.asarray(valid, dtype=bool)).sum())
        if invalid_count:
            raise ContractViolation(
                contract=contract,
                version=version,
                invariant=invariant,
                invalid_count=invalid_count,
            )

    issue_date = pd.to_datetime(frame["issueDate"], errors="coerce")
    feature_as_of = pd.to_datetime(frame["featureAsOf"], errors="coerce")
    missing_dates = issue_date.isna() | feature_as_of.isna()
    if missing_dates.any():
        raise ContractViolation(
            contract=contract,
            version=version,
            invariant="dates-parse",
            invalid_count=int(missing_dates.sum()),
        )
    future = feature_as_of > issue_date
    if future.any():
        raise ContractViolation(
            contract=contract,
            version=version,
            invariant="feature-asof-not-after-origination",
            invalid_count=int(future.sum()),
        )

    source_ids = frame["sourceRowId"]
    if source_ids.isna().any() or source_ids.duplicated().any():
        raise ContractViolation(
            contract=contract,
            version=version,
            invariant="source-row-id-unique",
            invalid_count=int(source_ids.isna().sum() + source_ids.duplicated().sum()),
        )

    validate_feature_provenance(provenance, required_features)
    return ContractEvidence(contract, version, len(frame), len(frame.columns)).to_dict()


def validate_chronological_splits(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    oot: pd.DataFrame,
) -> dict[str, object]:
    contract = "crix-chronological-split"
    version = "crix-chronological-split-v1"
    for name, frame in (("train", train), ("calibration", calibration), ("oot", oot)):
        _raise_missing(contract, version, _missing_columns(frame, ("sourceRowId", "issueDate")))
        if frame.empty:
            raise ContractViolation(
                contract=contract,
                version=version,
                invariant=f"{name}-non-empty",
                invalid_count=1,
            )

    ids = {
        "train": set(train["sourceRowId"].tolist()),
        "calibration": set(calibration["sourceRowId"].tolist()),
        "oot": set(oot["sourceRowId"].tolist()),
    }
    overlaps = (
        len(ids["train"] & ids["calibration"])
        + len(ids["train"] & ids["oot"])
        + len(ids["calibration"] & ids["oot"])
    )
    if overlaps:
        raise ContractViolation(
            contract=contract,
            version=version,
            invariant="split-row-overlap",
            invalid_count=overlaps,
        )

    train_max = pd.Timestamp(train["issueDate"].max())
    calibration_min = pd.Timestamp(calibration["issueDate"].min())
    calibration_max = pd.Timestamp(calibration["issueDate"].max())
    oot_min = pd.Timestamp(oot["issueDate"].min())
    if not (train_max < calibration_min and calibration_max < oot_min):
        raise ContractViolation(
            contract=contract,
            version=version,
            invariant="chronological-boundaries",
            invalid_count=1,
            summary={
                "trainMax": train_max.date().isoformat(),
                "calibrationMin": calibration_min.date().isoformat(),
                "calibrationMax": calibration_max.date().isoformat(),
                "ootMin": oot_min.date().isoformat(),
            },
        )

    return {
        "contract": contract,
        "version": version,
        "status": "pass",
        "counts": {
            "train": int(len(train)),
            "calibration": int(len(calibration)),
            "oot": int(len(oot)),
        },
    }


def validate_external_dataset(
    dataset_id: str,
    features: pd.DataFrame,
    target: np.ndarray,
    *,
    source_features: Sequence[str] | None = None,
) -> dict[str, object]:
    if dataset_id not in EXTERNAL_CONTRACT_VERSIONS:
        raise ValueError(f"unsupported external data contract: {dataset_id}")
    version = EXTERNAL_CONTRACT_VERSIONS[dataset_id]
    contract = f"external-{dataset_id}"
    specification = EXTERNAL_CONTRACTS[dataset_id]
    observed_source = tuple(str(value) for value in (source_features or features.columns))
    expected_source = tuple(str(value) for value in specification["requiredSourceFeatures"])
    if observed_source != expected_source:
        raise ContractViolation(
            contract=contract,
            version=version,
            invariant="product-specific-source-feature-contract",
            invalid_count=len(set(observed_source).symmetric_difference(expected_source)) or 1,
            summary={"expectedFeatureCount": len(expected_source), "actualFeatureCount": len(observed_source)},
        )
    y = np.asarray(target).reshape(-1)
    if len(features) != len(y) or len(y) == 0:
        raise ContractViolation(
            contract=contract,
            version=version,
            invariant="aligned-non-empty",
            invalid_count=abs(len(features) - len(y)) or 1,
        )
    if not np.isin(y, [0, 1]).all():
        raise ContractViolation(
            contract=contract,
            version=version,
            invariant="binary-target",
            invalid_count=int((~np.isin(y, [0, 1])).sum()),
        )
    numeric = features.to_numpy(dtype=float, copy=False)
    if numeric.ndim != 2 or not np.isfinite(numeric).all():
        invalid = int((~np.isfinite(numeric)).sum()) if numeric.ndim == 2 else 1
        raise ContractViolation(
            contract=contract,
            version=version,
            invariant="finite-feature-matrix",
            invalid_count=invalid,
        )

    expected_rows = {
        "uci-taiwan-credit-card-default": 30_000,
        "uci-statlog-german-credit": 1_000,
        "uci-south-german-credit": 1_000,
    }[dataset_id]
    if len(y) != expected_rows:
        raise ContractViolation(
            contract=contract,
            version=version,
            invariant="documented-row-count",
            invalid_count=abs(len(y) - expected_rows),
            summary={"expectedRows": expected_rows, "actualRows": len(y)},
        )

    evidence = ContractEvidence(contract, version, len(features), features.shape[1]).to_dict()
    evidence["product"] = specification["product"]
    evidence["target"] = specification["target"]
    evidence["sourceFeatureCount"] = len(expected_source)
    return evidence
