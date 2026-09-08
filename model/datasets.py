from __future__ import annotations

import hashlib
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LENDINGCLUB_URL = "https://zenodo.org/records/11295916/files/LC_loans_granting_model_dataset.csv?download=1"
LENDINGCLUB_MD5 = "b019384d6bc65bf2a3e839362e4ff502"
LENDINGCLUB_DOI = "10.5281/zenodo.11295916"

# Registry is deliberately explicit about what is and is not used for training.
# Product-mismatched or access-gated sources are never silently concatenated into the champion cohort.
DATASET_REGISTRY = {
    "lendingclub": {
        "product": "unsecured-personal-loan",
        "role": "primary-training-calibration-oot-test",
        "access": "open-download",
        "source": "Lending Club loan dataset for granting models",
        "url": "https://zenodo.org/records/11295916",
        "upstream": "https://www.kaggle.com/datasets/wordsforthewise/lending-club",
        "doi": LENDINGCLUB_DOI,
        "version": "0.1",
        "license": "CC-BY-4.0",
        "target": "final resolved status: charged-off/default=1, fully-paid=0",
    },
    "uci-statlog-german-credit": {
        "product": "consumer-instalment-credit",
        "role": "source-specific-external-benchmark",
        "access": "open-download",
        "source": "UCI Statlog (German Credit Data)",
        "url": "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
        "doi": "10.24432/C5NC77",
        "license": "CC-BY-4.0",
        "rows": 1000,
        "target": "good=1, bad=2; CRIX benchmark remaps bad to 1",
        "note": "Used only as a dataset-specific benchmark. UCI documents known coding-table problems in the legacy Statlog representation; no values are mapped into the LendingClub champion feature contract.",
    },
    "uci-taiwan-credit-card-default": {
        "product": "revolving-credit-card",
        "role": "source-specific-external-benchmark",
        "access": "open-download",
        "source": "UCI Default of Credit Card Clients",
        "url": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
        "doi": "10.24432/C55S3H",
        "license": "CC-BY-4.0",
        "rows": 30000,
        "target": "default payment next month: yes=1, no=0",
        "note": "Real Taiwanese bank credit-card data. Used as a source-specific benchmark rather than pooled into the personal-loan champion because product, features, geography and target horizon differ.",
    },
    "home-credit": {
        "product": "consumer-credit",
        "role": "future-external-validation-or-separate-champion",
        "access": "kaggle-competition-rules-required",
        "url": "https://www.kaggle.com/c/home-credit-default-risk",
    },
    "give-me-some-credit": {
        "product": "consumer-credit",
        "role": "future-external-validation",
        "access": "kaggle-competition-rules-required",
        "url": "https://www.kaggle.com/c/GiveMeSomeCredit",
    },
    "fico-heloc": {
        "product": "heloc",
        "role": "future-product-specific-external-validation",
        "access": "request-form-required",
        "url": "https://community.fico.com/s/explainable-machine-learning-challenge",
    },
    "freddie-mac-single-family": {
        "product": "mortgage",
        "role": "separate-product-validation-only",
        "access": "registration-required",
        "url": "https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset",
    },
    "fannie-mae-single-family": {
        "product": "mortgage",
        "role": "separate-product-validation-only",
        "access": "registration-and-terms-required",
        "url": "https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data",
    },
}

FEATURES = ["debtToIncome", "loanToIncome", "creditScore", "employmentYears"]
MONOTONE = [1, 1, -1, -1]


@dataclass(frozen=True)
class HarmonizedDataset:
    frame: pd.DataFrame
    source_rows: int
    usable_rows: int
    feature_names: list[str]
    monotone_constraints: list[int]


def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_lendingclub(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and md5sum(destination) == LENDINGCLUB_MD5:
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        LENDINGCLUB_URL,
        headers={"User-Agent": "CRIX-model-training/3.0"},
    )
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
        while chunk := response.read(1 << 20):
            output.write(chunk)

    actual = md5sum(temporary)
    if actual != LENDINGCLUB_MD5:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"LendingClub checksum mismatch: expected {LENDINGCLUB_MD5}, got {actual}")

    temporary.replace(destination)
    return destination


def _parse_employment_years(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().lower()
    if not text or text in {"n/a", "na", "none", "nan", "unknown"}:
        return np.nan
    if text.startswith("<"):
        return 0.5
    match = re.search(r"(\d+)", text)
    if not match:
        return np.nan
    return float(min(int(match.group(1)), 10))


def _parse_issue_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    if parsed.notna().mean() >= 0.95:
        return parsed
    fallback = pd.to_datetime(series, errors="coerce")
    return parsed.fillna(fallback)


def harmonize_lendingclub(path: Path) -> HarmonizedDataset:
    required = ["issue_d", "revenue", "dti_n", "loan_amnt", "fico_n", "emp_length", "Default"]
    raw = pd.read_csv(path, usecols=required, low_memory=False)
    source_rows = len(raw)

    frame = pd.DataFrame(
        {
            "issueDate": _parse_issue_dates(raw["issue_d"]),
            "annualIncome": pd.to_numeric(raw["revenue"], errors="coerce"),
            "debtToIncome": pd.to_numeric(raw["dti_n"], errors="coerce"),
            "loanAmount": pd.to_numeric(raw["loan_amnt"], errors="coerce"),
            "creditScore": pd.to_numeric(raw["fico_n"], errors="coerce"),
            "employmentYears": raw["emp_length"].map(_parse_employment_years),
            "target": pd.to_numeric(raw["Default"], errors="coerce"),
        }
    )

    finite_dti = frame.loc[np.isfinite(frame["debtToIncome"]), "debtToIncome"]
    if not finite_dti.empty and finite_dti.median() > 2:
        frame["debtToIncome"] = frame["debtToIncome"] / 100.0

    frame["loanToIncome"] = frame["loanAmount"] / frame["annualIncome"]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["issueDate", "annualIncome", "loanAmount", "target", *FEATURES])

    frame = frame[
        (frame["annualIncome"] > 0)
        & (frame["loanAmount"] > 0)
        & (frame["debtToIncome"].between(0, 2.0))
        & (frame["loanToIncome"].between(0.001, 3.0))
        & (frame["creditScore"].between(300, 850))
        & (frame["employmentYears"].between(0, 10))
        & (frame["target"].isin([0, 1]))
    ].copy()

    frame["target"] = frame["target"].astype(np.int8)
    frame = frame.sort_values(["issueDate"], kind="stable").reset_index(drop=True)
    return HarmonizedDataset(
        frame=frame,
        source_rows=source_rows,
        usable_rows=len(frame),
        feature_names=FEATURES.copy(),
        monotone_constraints=MONOTONE.copy(),
    )
