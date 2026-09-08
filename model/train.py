from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from datasets import (
    DATASET_REGISTRY,
    LENDINGCLUB_DOI,
    LENDINGCLUB_MD5,
    FEATURES,
    MONOTONE,
    download_lendingclub,
    harmonize_lendingclub,
)

SEED = 42
TRAIN_END = np.datetime64("2015-12-31")
CALIBRATION_END = np.datetime64("2016-12-31")
TEST_END = np.datetime64("2017-12-31")


def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1 / (1 + np.exp(-value))


def ks_statistic(y_true: np.ndarray, probability: np.ndarray) -> float:
    order = np.argsort(probability)
    observed = y_true[order]
    positive = (observed == 1).astype(float)
    negative = (observed == 0).astype(float)
    return float(
        np.max(
            np.abs(
                np.cumsum(positive) / max(positive.sum(), 1)
                - np.cumsum(negative) / max(negative.sum(), 1)
            )
        )
    )


def cohort_summary(frame) -> dict[str, object]:
    return {
        "samples": int(len(frame)),
        "defaultRate": round(float(frame["target"].mean()), 6),
        "start": frame["issueDate"].min().date().isoformat(),
        "end": frame["issueDate"].max().date().isoformat(),
    }


def split_chronologically(frame):
    eligible = frame[frame["issueDate"].values.astype("datetime64[D]") <= TEST_END].copy()
    train = eligible[eligible["issueDate"].values.astype("datetime64[D]") <= TRAIN_END]
    calibration = eligible[
        (eligible["issueDate"].values.astype("datetime64[D]") > TRAIN_END)
        & (eligible["issueDate"].values.astype("datetime64[D]") <= CALIBRATION_END)
    ]
    test = eligible[
        (eligible["issueDate"].values.astype("datetime64[D]") > CALIBRATION_END)
        & (eligible["issueDate"].values.astype("datetime64[D]") <= TEST_END)
    ]

    for name, cohort in (("train", train), ("calibration", calibration), ("test", test)):
        if len(cohort) < 10_000 or cohort["target"].nunique() != 2:
            raise RuntimeError(f"{name} cohort is too small or lacks both target classes: {len(cohort)} rows")
    return eligible, train, calibration, test


def make_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=96,
        max_depth=3,
        learning_rate=0.055,
        subsample=0.90,
        colsample_bytree=1.0,
        min_child_weight=120,
        reg_lambda=5.0,
        reg_alpha=0.08,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        max_bin=256,
        monotone_constraints=tuple(MONOTONE),
        random_state=SEED,
        n_jobs=4,
    )


def compact_xgboost_model(model: xgb.XGBClassifier, raw_path: Path) -> tuple[float, list[dict[str, list]]]:
    model.get_booster().save_model(raw_path)
    raw = json.loads(raw_path.read_text())
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
    base = float(str(learner["learner_model_param"]["base_score"]).strip("[]"))
    return base, trees


def calibration_diagnostics(y_true: np.ndarray, probability: np.ndarray) -> list[dict[str, float | int]]:
    quantiles = np.quantile(probability, np.linspace(0, 1, 11))
    rows = []
    for index in range(10):
        lower, upper = quantiles[index], quantiles[index + 1]
        mask = (probability >= lower) & ((probability <= upper) if index == 9 else (probability < upper))
        if not mask.any():
            continue
        rows.append(
            {
                "predicted": round(float(probability[mask].mean()), 6),
                "observed": round(float(y_true[mask].mean()), 6),
                "count": int(mask.sum()),
            }
        )
    return rows


def write_training_report(path: Path, artifact: dict) -> None:
    metrics = artifact["metrics"]
    split = artifact["training"]["split"]
    report = f"""# CRIX real-world training report

Generated from the immutable model artifact. This report is model-development evidence, not a production validation approval.

## Source

- Dataset: Lending Club loan dataset for granting models, version 0.1
- DOI: {LENDINGCLUB_DOI}
- Source checksum (MD5): `{LENDINGCLUB_MD5}`
- Upstream period: 2007–2018
- Rows in source: {artifact['training']['sourceRows']:,}
- Rows surviving CRIX harmonization: {artifact['training']['usableRows']:,}
- Target: final resolved loan status, charged-off/default = 1 and fully-paid = 0

## Temporal design

2018 is intentionally excluded from primary reported evaluation to reduce final-status maturation bias. No random train/test shuffle is used.

| Cohort | Origination window | Rows | Default rate |
|---|---|---:|---:|
| Train | {split['train']['start']} → {split['train']['end']} | {split['train']['samples']:,} | {split['train']['defaultRate']:.2%} |
| Calibration | {split['calibration']['start']} → {split['calibration']['end']} | {split['calibration']['samples']:,} | {split['calibration']['defaultRate']:.2%} |
| OOT test | {split['test']['start']} → {split['test']['end']} | {split['test']['samples']:,} | {split['test']['defaultRate']:.2%} |

## Out-of-time results

| Metric | Value |
|---|---:|
| ROC-AUC | {metrics['auc']:.4f} |
| KS | {metrics['ks']:.4f} |
| Brier score | {metrics['brier']:.4f} |
| Log loss | {metrics['logLoss']:.4f} |
| OOT observations | {metrics['testSamples']:,} |
| OOT default rate | {metrics['defaultRate']:.2%} |
| Logistic challenger ROC-AUC | {metrics['challengerAuc']:.4f} |

## Canonical CRIX features used by the champion

- `debtToIncome` ← LendingClub `dti_n` (normalized from percentage points when detected)
- `loanToIncome` ← `loan_amnt / revenue`
- `creditScore` ← LendingClub `fico_n`
- `employmentYears` ← normalized `emp_length`

The remaining CRIX application context is not falsely presented as part of the trained champion when it is absent from this public cohort.

## Dataset separation

Home Credit, Give Me Some Credit, FICO HELOC, Freddie Mac and Fannie Mae are registered as external/future product validation sources. They are not pooled into this model because their access terms, product definitions, features and/or default horizons differ.
"""
    path.write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CRIX-MonoBoost on the real LendingClub granting cohort")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "model" / "data" / "LC_loans_granting_model_dataset.csv",
    )
    parser.add_argument("--no-download", action="store_true", help="Fail if --data does not exist")
    args = parser.parse_args()

    if not args.data.exists():
        if args.no_download:
            raise FileNotFoundError(args.data)
        download_lendingclub(args.data)

    dataset = harmonize_lendingclub(args.data)
    eligible, train, calibration, test = split_chronologically(dataset.frame)

    X_train = train[FEATURES].to_numpy(dtype=np.float32)
    y_train = train["target"].to_numpy(dtype=np.int8)
    X_cal = calibration[FEATURES].to_numpy(dtype=np.float32)
    y_cal = calibration["target"].to_numpy(dtype=np.int8)
    X_test = test[FEATURES].to_numpy(dtype=np.float32)
    y_test = test["target"].to_numpy(dtype=np.int8)

    champion = make_model()
    champion.fit(X_train, y_train)

    # Calibrate the boosted margin on a later origination cohort.
    margin_cal = champion.predict(X_cal, output_margin=True).reshape(-1, 1)
    calibrator = LogisticRegression(C=1000, solver="lbfgs", max_iter=1000).fit(margin_cal, y_cal)
    margin_test = champion.predict(X_test, output_margin=True)
    p_test = calibrator.predict_proba(margin_test.reshape(-1, 1))[:, 1]

    # Independent transparent challenger trained on the same point-in-time real features.
    scaler = StandardScaler().fit(X_train)
    challenger = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(scaler.transform(X_train), y_train)
    challenger_test = challenger.predict_proba(scaler.transform(X_test))[:, 1]

    auc = roc_auc_score(y_test, p_test)
    brier = brier_score_loss(y_test, p_test)
    ll = log_loss(y_test, p_test)
    ks = ks_statistic(y_test, p_test)
    challenger_auc = roc_auc_score(y_test, challenger_test)

    artifact_dir = root / "model" / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "crix-monoboost-v2.json"
    raw_path = root / "model" / "raw-v2.json"
    base, trees = compact_xgboost_model(champion, raw_path)

    fpr, tpr, _ = roc_curve(y_test, p_test)
    indices = np.unique(np.linspace(0, len(fpr) - 1, 32, dtype=int))
    roc = [{"fpr": round(float(fpr[i]), 6), "tpr": round(float(tpr[i]), 6)} for i in indices]

    gain = champion.get_booster().get_score(importance_type="gain")
    importance = [{"feature": name, "gain": float(gain.get(f"f{i}", 0.0))} for i, name in enumerate(FEATURES)]
    total_gain = sum(item["gain"] for item in importance) or 1.0
    for item in importance:
        item["gain"] = round(item["gain"] / total_gain, 6)
    importance = sorted(importance, key=lambda item: item["gain"], reverse=True)

    training_bounds = {
        name: {
            "p01": round(float(train[name].quantile(0.01)), 6),
            "p99": round(float(train[name].quantile(0.99)), 6),
        }
        for name in FEATURES
    }
    reference = {name: round(float(train[name].median()), 6) for name in FEATURES}

    artifact = {
        "schemaVersion": 2,
        "name": "CRIX-MonoBoost",
        "version": "2.0.0",
        "trainedAt": date.today().isoformat(),
        "target": {
            "name": "originationDefaultRisk",
            "definition": "Probability that a granted LendingClub loan resolves as charged-off/default rather than fully paid.",
            "horizon": "final-loan-resolution (mixed contractual terms; not a 12-month PD)",
        },
        "featureNames": FEATURES,
        "monotoneConstraints": MONOTONE,
        "baseScore": base,
        "calibration": {
            "method": "platt-on-later-origination-cohort",
            "slope": float(calibrator.coef_[0, 0]),
            "intercept": float(calibrator.intercept_[0]),
        },
        "challenger": {
            "name": "real-data-logistic",
            "intercept": float(challenger.intercept_[0]),
            "coefficients": [float(value) for value in challenger.coef_[0]],
            "means": [float(value) for value in scaler.mean_],
            "scales": [float(value) for value in scaler.scale_],
        },
        "trees": trees,
        "metrics": {
            "auc": round(float(auc), 6),
            "brier": round(float(brier), 6),
            "logLoss": round(float(ll), 6),
            "ks": round(float(ks), 6),
            "testSamples": int(len(y_test)),
            "defaultRate": round(float(y_test.mean()), 6),
            "challengerAuc": round(float(challenger_auc), 6),
        },
        "diagnostics": {
            "calibration": calibration_diagnostics(y_test, p_test),
            "roc": roc,
            "featureImportance": importance,
        },
        "reference": reference,
        "trainingBounds": training_bounds,
        "training": {
            "dataset": "lendingclub",
            "sourceRows": dataset.source_rows,
            "usableRows": dataset.usable_rows,
            "eligibleThrough2017Rows": int(len(eligible)),
            "sourceLock": {"doi": LENDINGCLUB_DOI, "md5": LENDINGCLUB_MD5, "version": "0.1"},
            "splitPolicy": "chronological-origination",
            "split": {
                "train": cohort_summary(train),
                "calibration": cohort_summary(calibration),
                "test": cohort_summary(test),
            },
            "featureCoverage": {
                "champion": FEATURES,
                "notAvailableInPrimaryCohort": [
                    "creditUtilization",
                    "delinquencies24m",
                    "inquiries6m",
                    "oldestTradeMonths",
                    "openAccounts",
                    "cashBufferMonths",
                    "onTimePaymentRate",
                    "incomeStability",
                    "recentCreditGrowth",
                ],
            },
            "datasetRegistry": DATASET_REGISTRY,
        },
    }

    artifact_path.write_text(json.dumps(artifact, separators=(",", ":")))
    write_training_report(root / "model" / "TRAINING_REPORT.md", artifact)
    raw_path.unlink(missing_ok=True)

    print(json.dumps(artifact["metrics"], indent=2))
    print(f"artifact {artifact_path}")


if __name__ == "__main__":
    main()
