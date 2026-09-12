from __future__ import annotations

import hashlib
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from registry import sha256_file

LIFECYCLE_LENDINGCLUB_URL = "https://huggingface.co/datasets/zafar0171/loan_default_dataset/resolve/main/loan_data_2007_2014.csv?download=true"
LIFECYCLE_LENDINGCLUB_SHA256 = "43ecf2dcf074c87215078f33019c1282d9edb0d3569ce052d5ad6a597004c545"
LIFECYCLE_SOURCE_ID = "lendingclub-lifecycle-2007-2014-mirror"
LIFECYCLE_LICENSE = "MIT (mirror metadata; underlying LendingClub provenance retained as research limitation)"
TAIWAN_DOI = "10.24432/C55S3H"
TAIWAN_LICENSE = "CC-BY-4.0"

DEFAULT_STATUSES = {
    "charged off",
    "default",
    "does not meet the credit policy. status:charged off",
}
FULLY_PAID_STATUSES = {
    "fully paid",
    "does not meet the credit policy. status:fully paid",
}

LIFECYCLE_COLUMNS = [
    "id",
    "loan_amnt",
    "funded_amnt",
    "term",
    "int_rate",
    "installment",
    "emp_length",
    "annual_inc",
    "issue_d",
    "loan_status",
    "dti",
    "total_rec_prncp",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_credit_pull_d",
]


@dataclass(frozen=True)
class LifecycleDataset:
    frame: pd.DataFrame
    source_sha256: str
    source_rows: int
    usable_rows: int


@dataclass(frozen=True)
class TaiwanCreditData:
    limit: np.ndarray
    repayment_history: np.ndarray
    bill_history: np.ndarray
    default_next_month: np.ndarray
    fingerprint_sha256: str
    rows: int


def _download_once(url: str, temporary: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "CRIX-risk-stack-research/1.0"})
    with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as output:
        while chunk := response.read(1 << 20):
            output.write(chunk)


def download_lifecycle_lendingclub(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256_file(destination) == LIFECYCLE_LENDINGCLUB_SHA256:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, 5):
        temporary.unlink(missing_ok=True)
        try:
            _download_once(LIFECYCLE_LENDINGCLUB_URL, temporary)
            actual = sha256_file(temporary)
            if actual != LIFECYCLE_LENDINGCLUB_SHA256:
                raise RuntimeError(
                    f"Lifecycle source checksum mismatch: expected {LIFECYCLE_LENDINGCLUB_SHA256}, got {actual}"
                )
            temporary.replace(destination)
            return destination
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < 4:
                time.sleep(attempt * 5)
    raise RuntimeError(f"Unable to download verified lifecycle source: {last_error}")


def _employment_years(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().lower()
    if text in {"", "n/a", "none", "nan"}:
        return np.nan
    if text.startswith("<"):
        return 0.5
    match = re.search(r"(\d+)", text)
    return float(min(int(match.group(1)), 10)) if match else np.nan


def _term_months(value: object) -> float:
    match = re.search(r"(\d+)", str(value))
    return float(int(match.group(1))) if match else np.nan


def _parse_dates(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    parsed = pd.to_datetime(text, errors="coerce", format="%b-%y")
    fallback = parsed.isna() & text.notna()
    if fallback.any():
        parsed.loc[fallback] = pd.to_datetime(text.loc[fallback], errors="coerce", format="mixed")
    return parsed


def _months_between(start: pd.Series, end: pd.Series) -> np.ndarray:
    start_month = start.values.astype("datetime64[M]").astype(np.int64)
    end_month = end.values.astype("datetime64[M]").astype(np.int64)
    return (end_month - start_month + 1).astype(np.int32)


def load_lifecycle_lendingclub(path: Path) -> LifecycleDataset:
    actual_sha = sha256_file(path)
    if actual_sha != LIFECYCLE_LENDINGCLUB_SHA256:
        raise RuntimeError(f"Unverified lifecycle source: {actual_sha}")
    raw = pd.read_csv(path, usecols=LIFECYCLE_COLUMNS, low_memory=False)
    source_rows = len(raw)

    frame = pd.DataFrame({
        "loanId": pd.to_numeric(raw["id"], errors="coerce"),
        "loanAmount": pd.to_numeric(raw["loan_amnt"], errors="coerce"),
        "fundedAmount": pd.to_numeric(raw["funded_amnt"], errors="coerce"),
        "termMonths": raw["term"].map(_term_months),
        "annualRate": pd.to_numeric(raw["int_rate"], errors="coerce"),
        "installment": pd.to_numeric(raw["installment"], errors="coerce"),
        "employmentYears": raw["emp_length"].map(_employment_years),
        "annualIncome": pd.to_numeric(raw["annual_inc"], errors="coerce"),
        "issueDate": _parse_dates(raw["issue_d"]),
        "status": raw["loan_status"].astype(str).str.strip().str.lower(),
        "debtToIncome": pd.to_numeric(raw["dti"], errors="coerce"),
        "principalReceived": pd.to_numeric(raw["total_rec_prncp"], errors="coerce"),
        "recoveries": pd.to_numeric(raw["recoveries"], errors="coerce"),
        "recoveryCosts": pd.to_numeric(raw["collection_recovery_fee"], errors="coerce"),
        "lastPaymentDate": _parse_dates(raw["last_pymnt_d"]),
        "lastCreditPullDate": _parse_dates(raw["last_credit_pull_d"]),
    })
    if frame["annualRate"].dropna().median() > 1:
        frame["annualRate"] /= 100.0
    if frame["debtToIncome"].dropna().median() > 2:
        frame["debtToIncome"] /= 100.0
    frame["loanToIncome"] = frame["loanAmount"] / frame["annualIncome"]
    frame["event"] = frame["status"].isin(DEFAULT_STATUSES).astype(np.int8)
    frame["resolved"] = frame["status"].isin(DEFAULT_STATUSES | FULLY_PAID_STATUSES)
    endpoint = frame["lastPaymentDate"].where(frame["lastPaymentDate"].notna(), frame["lastCreditPullDate"])
    unresolved_endpoint = frame["lastCreditPullDate"].where(frame["lastCreditPullDate"].notna(), endpoint)
    frame["outcomeAvailableAt"] = endpoint.where(frame["resolved"], unresolved_endpoint)
    frame["durationMonths"] = _months_between(frame["issueDate"], frame["outcomeAvailableAt"])
    frame["observedEad"] = np.maximum(frame["fundedAmount"] - frame["principalReceived"], 0.0)

    frame = frame.replace([np.inf, -np.inf], np.nan)
    required = [
        "issueDate", "outcomeAvailableAt", "loanAmount", "fundedAmount", "termMonths", "annualRate",
        "employmentYears", "annualIncome", "debtToIncome", "loanToIncome", "principalReceived",
        "recoveries", "recoveryCosts", "durationMonths",
    ]
    frame = frame.dropna(subset=required)
    frame = frame[
        (frame["annualIncome"] > 0)
        & (frame["loanAmount"] > 0)
        & (frame["fundedAmount"] > 0)
        & (frame["termMonths"].isin([36, 60]))
        & (frame["annualRate"].between(0, 1))
        & (frame["debtToIncome"].between(0, 2))
        & (frame["loanToIncome"].between(0.001, 3))
        & (frame["employmentYears"].between(0, 10))
        & (frame["durationMonths"] >= 1)
        & (frame["outcomeAvailableAt"] >= frame["issueDate"])
    ].copy()
    frame["durationMonths"] = np.minimum(frame["durationMonths"].to_numpy(), frame["termMonths"].to_numpy()).astype(np.int16)
    frame = frame.sort_values(["issueDate", "loanId"], kind="stable").reset_index(drop=True)
    return LifecycleDataset(frame=frame, source_sha256=actual_sha, source_rows=source_rows, usable_rows=len(frame))


def _fingerprint_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def load_taiwan_credit() -> TaiwanCreditData:
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=350)
    features = dataset.data.features
    target_frame = dataset.data.targets
    if features.shape[1] < 23 or len(features) != 30_000:
        raise RuntimeError(f"Unexpected UCI Taiwan shape: {features.shape}")

    numeric = features.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    target = pd.to_numeric(target_frame.iloc[:, 0], errors="raise").to_numpy(dtype=np.int8)
    if not np.isin(target, [0, 1]).all():
        raise RuntimeError("UCI Taiwan target is not binary")

    # UCI X6..X11 are Sep..Apr repayment status. Reverse into Apr..Sep chronology.
    repayment = numeric[:, [10, 9, 8, 7, 6, 5]]
    # UCI X12..X17 are Sep..Apr statement balances. Reverse into Apr..Sep chronology.
    bills = numeric[:, [16, 15, 14, 13, 12, 11]]
    limit = numeric[:, 0]
    fingerprint = _fingerprint_arrays(limit, repayment, bills, target)
    return TaiwanCreditData(
        limit=limit,
        repayment_history=repayment,
        bill_history=bills,
        default_next_month=target,
        fingerprint_sha256=fingerprint,
        rows=len(target),
    )
