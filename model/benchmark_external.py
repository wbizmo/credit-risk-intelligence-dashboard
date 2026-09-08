from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from ucimlrepo import fetch_ucirepo

SEED = 42


def _sanitize_feature_names(frame: pd.DataFrame) -> pd.DataFrame:
    sanitized: list[str] = []
    seen: dict[str, int] = {}
    for column in frame.columns:
        base = re.sub(r"[\[\]<>]", "_", str(column))
        count = seen.get(base, 0)
        seen[base] = count + 1
        sanitized.append(base if count == 0 else f"{base}__{count}")
    result = frame.copy()
    result.columns = sanitized
    return result


def _metrics_from_cv(X: pd.DataFrame, y: np.ndarray, *, folds: int = 10) -> dict[str, float | int]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    out = np.zeros(len(y), dtype=np.float64)
    fold_auc: list[float] = []

    for train_idx, test_idx in splitter.split(X, y):
        model = xgb.XGBClassifier(
            n_estimators=160,
            max_depth=3,
            learning_rate=0.04,
            min_child_weight=8,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=4.0,
            reg_alpha=0.05,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=SEED,
            n_jobs=2,
        )
        model.fit(X.iloc[train_idx], y[train_idx])
        probability = model.predict_proba(X.iloc[test_idx])[:, 1]
        out[test_idx] = probability
        fold_auc.append(float(roc_auc_score(y[test_idx], probability)))

    return {
        "samples": int(len(y)),
        "features": int(X.shape[1]),
        "defaultRate": round(float(y.mean()), 6),
        "auc": round(float(roc_auc_score(y, out)), 6),
        "meanFoldAuc": round(float(np.mean(fold_auc)), 6),
        "stdFoldAuc": round(float(np.std(fold_auc, ddof=1)), 6),
        "brier": round(float(brier_score_loss(y, out)), 6),
        "logLoss": round(float(log_loss(y, out)), 6),
    }


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def taiwan_behavioral() -> tuple[dict[str, object], pd.DataFrame, np.ndarray]:
    dataset = fetch_ucirepo(id=350)
    raw = dataset.data.features.copy()
    target = pd.to_numeric(dataset.data.targets.iloc[:, 0], errors="raise").astype(np.int8).to_numpy()

    # UCI fields are X1..X23. X2/X3/X4/X5 are sex/education/marriage/age and are
    # deliberately excluded. CRIX only derives behavior from limit, repayment status and bill history.
    limit_balance = pd.to_numeric(raw["X1"], errors="coerce")
    pay_status = raw[[f"X{i}" for i in range(6, 12)]].apply(pd.to_numeric, errors="coerce")
    bills = raw[[f"X{i}" for i in range(12, 18)]].apply(pd.to_numeric, errors="coerce")

    monthly_utilization = bills.div(limit_balance.replace(0, np.nan), axis=0)
    behavior = pd.DataFrame(
        {
            "creditUtilization6mMean": monthly_utilization.mean(axis=1),
            "onTimePaymentRate6m": (pay_status <= 0).mean(axis=1),
            "paymentDelayMonths6m": (pay_status > 0).sum(axis=1).astype(float),
            # X12 is Sep 2005 (most recent bill), X17 is Apr 2005 (oldest bill).
            # Normalize six-month balance change by the credit limit, not by a potentially tiny bill.
            "recentCreditGrowth6m": _safe_divide(bills["X12"] - bills["X17"], limit_balance),
        }
    ).replace([np.inf, -np.inf], np.nan)
    behavior = behavior.fillna(behavior.median(numeric_only=True)).astype(np.float32)

    metrics = _metrics_from_cv(behavior, target)
    return (
        {
            "dataset": "uci-taiwan-credit-card-default",
            "doi": "10.24432/C55S3H",
            "license": "CC-BY-4.0",
            "product": "revolving-credit-card",
            "target": "default payment next month",
            "role": "reduced-feature behavioral adaptation benchmark",
            "featureContract": list(behavior.columns),
            "protectedFieldsExcluded": ["SEX", "EDUCATION", "MARRIAGE", "AGE"],
            "mappingNotes": {
                "creditUtilization": "mean(BILL_AMT1..6 / LIMIT_BAL)",
                "onTimePaymentRate": "fraction(PAY_0,PAY_2..6 <= 0)",
                "delinquencyProxy": "count of delayed-payment months across six months; not represented as 24-month delinquencies",
                "recentCreditGrowth": "(BILL_AMT1 - BILL_AMT6) / LIMIT_BAL",
            },
            "metrics": metrics,
        },
        behavior,
        target,
    )


def _encode_german(dataset_id: int, corrected: bool) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    dataset = fetch_ucirepo(id=dataset_id)
    raw = dataset.data.features.copy()
    target_raw = pd.to_numeric(dataset.data.targets.iloc[:, 0], errors="raise").astype(int)

    if corrected:
        target = (target_raw.to_numpy() == 0).astype(np.int8)
        keep = ["duration", "amount", "employment_duration", "installment_rate", "number_credits"]
        if not set(keep).issubset(raw.columns):
            keep = [raw.columns[i] for i in [1, 4, 6, 7, 15]]
    else:
        target = (target_raw.to_numpy() == 2).astype(np.int8)
        keep = ["Attribute2", "Attribute5", "Attribute7", "Attribute8", "Attribute16"]

    frame = raw[keep].copy()
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = frame[column].astype("string")
    frame = pd.get_dummies(frame, dummy_na=True, dtype=np.float32)
    frame = _sanitize_feature_names(frame)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    for column in frame.columns:
        if frame[column].isna().any():
            frame[column] = frame[column].fillna(frame[column].median())
    return frame.astype(np.float32), target, [str(value) for value in keep]


def german_structural(dataset_id: int, corrected: bool) -> dict[str, object]:
    X, y, source_features = _encode_german(dataset_id, corrected)
    metrics = _metrics_from_cv(X, y)
    return {
        "dataset": "uci-south-german-credit" if corrected else "uci-statlog-german-credit",
        "doi": "10.24432/C5QG88" if corrected else "10.24432/C5NC77",
        "license": "CC-BY-4.0",
        "product": "consumer-instalment-credit",
        "target": "bad credit risk",
        "role": "structural credit benchmark only",
        "sourceFeatures": source_features,
        "protectedFieldsExcluded": ["personal_status_sex", "age", "telephone", "foreign_worker"],
        "mappingNotes": {
            "termMonths": "duration",
            "loanAmount": "credit amount",
            "employmentYears": "employment-duration band; not exact years",
            "debtToIncome": "installment rate as percent of disposable income; only a rough per-loan proxy",
            "openAccounts": "credits at the same bank only; not bureau-wide open accounts",
        },
        "metrics": metrics,
        **({"warning": "Legacy Statlog coding table has documented issues; corrected South German companion is preferred."} if not corrected else {}),
    }


def _compact_booster(model: xgb.XGBClassifier, temp_path: Path) -> dict[str, object]:
    model.get_booster().save_model(temp_path)
    raw = json.loads(temp_path.read_text())
    learner = raw["learner"]
    booster = learner["gradient_booster"]["model"]
    trees = []
    for tree in booster["trees"]:
        trees.append(
            {
                "left": tree["left_children"],
                "right": tree["right_children"],
                "feature": tree["split_indices"],
                "threshold": tree["split_conditions"],
                "defaultLeft": tree["default_left"],
            }
        )
    base_score = float(str(learner["learner_model_param"]["base_score"]).strip("[]"))
    temp_path.unlink(missing_ok=True)
    return {"baseScore": base_score, "trees": trees}


def train_taiwan_research_artifact(X: pd.DataFrame, y: np.ndarray, root: Path, benchmark: dict[str, object]) -> None:
    # This is intentionally a separate research model. Its next-month revolving-credit target is
    # not interchangeable with the LendingClub final-resolution personal-loan PD.
    model = xgb.XGBClassifier(
        n_estimators=160,
        max_depth=3,
        learning_rate=0.04,
        min_child_weight=8,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=4.0,
        reg_alpha=0.05,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        monotone_constraints=(1, -1, 1, 1),
        random_state=SEED,
        n_jobs=2,
    )
    model.fit(X, y)
    compact = _compact_booster(model, root / "model" / "raw-taiwan.json")
    artifact = {
        "schemaVersion": 1,
        "name": "CRIX-Behavior-TW",
        "version": "1.0.0",
        "trainedAt": date.today().isoformat(),
        "status": "research-only-not-runtime-champion",
        "target": "default payment next month",
        "product": "revolving-credit-card",
        "source": "UCI Default of Credit Card Clients",
        "doi": "10.24432/C55S3H",
        "license": "CC-BY-4.0",
        "featureNames": list(X.columns),
        "monotoneConstraints": [1, -1, 1, 1],
        "crossValidation": benchmark["metrics"],
        **compact,
    }
    path = root / "model" / "artifacts" / "crix-behavior-tw-v1.json"
    path.write_text(json.dumps(artifact, separators=(",", ":")))


def write_report(path: Path, payload: dict[str, object]) -> None:
    rows = []
    for benchmark in payload["benchmarks"]:
        metrics = benchmark["metrics"]
        rows.append(
            f"| {benchmark['dataset']} | {benchmark['product']} | {metrics['samples']:,} | {metrics['defaultRate']:.2%} | {metrics['auc']:.4f} | {metrics['brier']:.4f} | {metrics['logLoss']:.4f} |"
        )
    report = f"""# CRIX external real-world benchmarks

Generated {payload['generatedAt']}. These are **dataset-specific adaptation/benchmark models**, not evidence that the LendingClub champion transfers unchanged across products or geographies.

| Dataset | Product | Rows | Default rate | CV ROC-AUC | Brier | Log loss |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Taiwan credit-card behavioral adaptation

`CRIX-Behavior-TW 1.0.0` is trained separately on four behaviorally defensible features derived from the 30,000-client UCI Taiwan bank dataset:

- six-month mean revolving utilization;
- six-month on-time-payment rate;
- count of delayed-payment months in the six-month window;
- six-month bill-balance growth normalized by credit limit.

`SEX`, `EDUCATION`, `MARRIAGE`, and `AGE` are explicitly excluded. Missing personal-loan/bureau fields are **not imputed**. The model targets **next-month credit-card default**, so it is retained as a research adaptation artifact and is not blended into CRIX's personal-loan PD or policy.

## German structural benchmarks

The German datasets are used only for loan-structure benchmarking because their overlap with CRIX is limited to duration, amount, employment band, installment burden and same-bank credit count. The legacy Statlog version is retained for comparability, but the corrected South German Credit representation is preferred because UCI documents coding issues in the old representation.

## Governance rule

CRIX never concatenates these cohorts with LendingClub merely to increase row count. Product definition, target horizon, geography and feature semantics stay explicit in every artifact.
"""
    path.write_text(report)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    taiwan, taiwan_X, taiwan_y = taiwan_behavioral()
    benchmarks = [
        taiwan,
        german_structural(144, corrected=False),
        german_structural(573, corrected=True),
    ]
    payload = {
        "generatedAt": date.today().isoformat(),
        "benchmarks": benchmarks,
        "policy": "dataset-specific-only-no-cross-product-probability-blending",
    }
    artifact_dir = root / "model" / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "external-benchmarks.json").write_text(json.dumps(payload, separators=(",", ":")))
    train_taiwan_research_artifact(taiwan_X, taiwan_y, root, taiwan)
    write_report(root / "model" / "EXTERNAL_BENCHMARKS.md", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
