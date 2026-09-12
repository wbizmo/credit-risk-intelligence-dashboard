from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score

from datasets import harmonize_lendingclub
from ead import ccf_proxy, observed_installment_ead
from lgd import economic_lgd
from publish_champion_manifest import atomic_json, git_commit, main as publish_champion_manifest
from registry import build_manifest, sha256_file, update_registry_index
from research_data import (
    LIFECYCLE_LICENSE,
    LIFECYCLE_SOURCE_ID,
    LIFECYCLE_LENDINGCLUB_SHA256,
    TAIWAN_DOI,
    TAIWAN_LICENSE,
    download_lifecycle_lendingclub,
    load_lifecycle_lendingclub,
    load_taiwan_credit,
)
from survival import (
    fit_discrete_time_hazard,
    predict_cumulative_pd_batch,
    predict_term_structure,
    term_structure_at_horizons,
)
from time_machine import SnapshotSpec, build_snapshot_masks, snapshot_manifest
from transitions import DEFAULT_STATES, map_repayment_status, transition_counts, transition_matrix

SEED = 42
SURVIVAL_FEATURES = ["debtToIncome", "loanToIncome", "employmentYears"]
DEFAULT_HORIZONS = (3, 6, 12, 24, 36)


def _sample(frame, maximum: int, seed: int = SEED):
    if len(frame) <= maximum:
        return frame.copy()
    return frame.sample(n=maximum, random_state=seed).sort_index(kind="stable").copy()


def _ridge_fit(features: np.ndarray, target: np.ndarray, alpha: float = 10.0) -> tuple[dict[str, object], np.ndarray]:
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardized = (x - means) / scales
    model = Ridge(alpha=alpha).fit(standardized, y)
    artifact = {
        "method": "ridge",
        "alpha": alpha,
        "means": means.tolist(),
        "scales": scales.tolist(),
        "intercept": float(model.intercept_),
        "coefficients": model.coef_.tolist(),
    }
    return artifact, model.predict(standardized)


def _ridge_predict(model: Mapping[str, object], features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    means = np.asarray(model["means"], dtype=float)
    scales = np.asarray(model["scales"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    return float(model["intercept"]) + ((x - means) / scales) @ coefficients


def _regression_comparison(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    prediction_bounds: tuple[float | None, float | None] = (None, None),
) -> dict[str, object]:
    ridge, _ = _ridge_fit(train_x, train_y)
    ridge_pred = _ridge_predict(ridge, test_x)
    challenger = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=120,
        max_depth=3,
        min_samples_leaf=80,
        l2_regularization=2.0,
        random_state=SEED,
    ).fit(train_x, train_y)
    challenger_pred = challenger.predict(test_x)
    lower, upper = prediction_bounds
    if lower is not None:
        ridge_pred = np.maximum(ridge_pred, lower)
        challenger_pred = np.maximum(challenger_pred, lower)
    if upper is not None:
        ridge_pred = np.minimum(ridge_pred, upper)
        challenger_pred = np.minimum(challenger_pred, upper)

    def metrics(prediction: np.ndarray) -> dict[str, float]:
        return {
            "mae": float(mean_absolute_error(test_y, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(test_y, prediction))),
        }

    ridge_metrics = metrics(ridge_pred)
    challenger_metrics = metrics(challenger_pred)
    return {
        "baseline": {"model": ridge, "metrics": ridge_metrics},
        "challenger": {"name": "hist-gradient-boosting", "metrics": challenger_metrics},
        "recommended": "challenger" if challenger_metrics["mae"] < ridge_metrics["mae"] else "baseline",
        "baselinePrediction": ridge_pred,
        "challengerPrediction": challenger_pred,
    }


def _horizon_evaluation(frame, model: Mapping[str, object]) -> dict[str, object]:
    x = frame[SURVIVAL_FEATURES].to_numpy(dtype=float)
    duration = frame["durationMonths"].to_numpy(dtype=int)
    event = frame["event"].to_numpy(dtype=np.int8)
    resolved = frame["resolved"].to_numpy(dtype=bool)
    output: dict[str, object] = {}
    for horizon in DEFAULT_HORIZONS:
        known_default = (event == 1) & (duration <= horizon)
        known_nondefault = ((event == 0) & resolved) | (duration >= horizon)
        eligible = known_default | known_nondefault
        y = known_default[eligible].astype(np.int8)
        if int(eligible.sum()) < 500 or len(np.unique(y)) != 2:
            output[f"{horizon}m"] = {"status": "insufficient-data", "samples": int(eligible.sum())}
            continue
        prediction = predict_cumulative_pd_batch(model, x[eligible], horizon)
        output[f"{horizon}m"] = {
            "status": "ok",
            "samples": int(len(y)),
            "events": int(y.sum()),
            "observedRate": float(y.mean()),
            "averagePd": float(prediction.mean()),
            "auc": float(roc_auc_score(y, prediction)),
            "brier": float(brier_score_loss(y, prediction)),
        }
    return output


def _research_manifest(
    *,
    root: Path,
    artifact_path: Path,
    model_id: str,
    model_name: str,
    version: str,
    product: str,
    target: Mapping[str, object],
    datasets: list[Mapping[str, object]],
    feature_contract_version: str,
    split: Mapping[str, object],
    training_config: Mapping[str, object],
    metrics: Mapping[str, object],
) -> dict[str, object]:
    manifest = build_manifest(
        model_id=model_id,
        model_name=model_name,
        version=version,
        artifact_path=artifact_path,
        artifact_schema_version=1,
        product=product,
        target=target,
        git_commit=git_commit(root),
        datasets=datasets,
        feature_contract_version=feature_contract_version,
        split=split,
        random_seed=SEED,
        training_config=training_config,
        environment_hash=sha256_file(root / "model" / "requirements.txt"),
        metrics=metrics,
        status="research",
        model_card="MODEL_CARD.md",
        report="model/RISK_STACK_REPORT.md",
    )
    manifest_path = artifact_path.with_suffix(".manifest.json")
    atomic_json(manifest_path, manifest)
    update_registry_index(root / "model" / "artifacts" / "registry.json", {
        "modelId": model_id,
        "modelName": model_name,
        "version": version,
        "artifactSha256": manifest["artifactSha256"],
        "manifest": str(manifest_path.relative_to(root)),
        "artifact": str(artifact_path.relative_to(root)),
        "status": "research",
        "featureContractVersion": feature_contract_version,
    })
    return manifest


def _transition_artifact(taiwan) -> dict[str, object]:
    sequences: list[list[str | None]] = []
    for repayment, default in zip(taiwan.repayment_history, taiwan.default_next_month):
        sequence = [map_repayment_status(value) for value in repayment]
        if default == 1:
            sequence.append("DEFAULT")
        sequences.append(sequence)
    counts = transition_counts(sequences, DEFAULT_STATES)
    matrix = transition_matrix(counts, DEFAULT_STATES, absorbing={"DEFAULT"})
    return {
        "schemaVersion": 1,
        "name": "CRIX-Transitions-Taiwan",
        "version": "1.0.0",
        "product": "revolving-credit-card",
        "method": "empirical-monthly-transition-matrix",
        "stateTaxonomy": list(DEFAULT_STATES),
        "sourceChronology": ["Apr-2005", "May-2005", "Jun-2005", "Jul-2005", "Aug-2005", "Sep-2005", "Oct-2005-default-only"],
        "nonDefaultOctoberState": "not-fabricated",
        "counts": counts.tolist(),
        "matrix": matrix.tolist(),
        "rows": taiwan.rows,
        "observedTransitions": int(counts.sum()),
    }


def _ccf_artifact(taiwan) -> dict[str, object]:
    prior = taiwan.bill_history[:, :-1]
    following = taiwan.bill_history[:, 1:]
    limit = np.repeat(taiwan.limit[:, None], 5, axis=1)
    proxy = ccf_proxy(limit, prior, following)
    month = np.broadcast_to(np.arange(1, 6, dtype=float), proxy.shape)
    repayment_prior = taiwan.repayment_history[:, :-1]
    utilization = np.divide(prior, limit, out=np.zeros_like(prior, dtype=float), where=limit != 0)
    finite = np.isfinite(proxy) & np.isfinite(utilization) & np.isfinite(repayment_prior)
    supported = finite & (np.abs(proxy) <= 20)
    train = supported & (month < 5)
    test = supported & (month == 5)
    train_x = np.column_stack([utilization[train], repayment_prior[train], month[train] / 5.0])
    test_x = np.column_stack([utilization[test], repayment_prior[test], month[test] / 5.0])
    train_y = proxy[train]
    test_y = proxy[test]
    baseline, _ = _ridge_fit(train_x, train_y, alpha=25.0)
    prediction = _ridge_predict(baseline, test_x)
    return {
        "schemaVersion": 1,
        "name": "CRIX-CCF-Taiwan",
        "version": "1.0.0",
        "product": "revolving-credit-card",
        "target": "net-balance-change CCF proxy = (next statement balance - prior statement balance) / prior undrawn limit",
        "warning": "This is not pure draw CCF: statement balance changes also reflect payments, interest, fees and purchases.",
        "targetPolicy": "raw proxy is not clamped; model fitting uses finite observations with |proxy| <= 20 and reports that support rule explicitly",
        "featureNames": ["priorUtilization", "priorRepaymentStatusCode", "monthIndexNormalized"],
        "baseline": baseline,
        "metrics": {
            "testSamples": int(len(test_y)),
            "mae": float(mean_absolute_error(test_y, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(test_y, prediction))),
            "rawFiniteSamples": int(finite.sum()),
            "trainingSupportSamples": int(supported.sum()),
            "rawProxyMin": float(np.nanmin(proxy)),
            "rawProxyMax": float(np.nanmax(proxy)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CRIX research risk-stack models without changing v3 runtime semantics")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--lifecycle-data", type=Path, default=root / "model" / "data" / "loan_data_2007_2014.csv")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    if not args.lifecycle_data.exists():
        if args.no_download:
            raise FileNotFoundError(args.lifecycle_data)
        download_lifecycle_lendingclub(args.lifecycle_data)

    # Ensure registry starts with the exact currently deployed champion.
    registry_path = root / "model" / "artifacts" / "registry.json"
    if not registry_path.exists():
        publish_champion_manifest()

    lifecycle = load_lifecycle_lendingclub(args.lifecycle_data)
    frame = lifecycle.frame
    snapshot_spec = SnapshotSpec(
        train_end="2010-12-31",
        calibration_end="2011-12-31",
        decision_end="2012-12-31",
        outcome_cutoff="2016-12-31",
        source_sha256=lifecycle.source_sha256,
        mode="point-in-time",
        train_label_cutoff="2014-12-31",
        calibration_label_cutoff="2015-12-31",
    )
    masks = build_snapshot_masks(
        frame["issueDate"].to_numpy(dtype="datetime64[D]"),
        frame["outcomeAvailableAt"].to_numpy(dtype="datetime64[D]"),
        snapshot_spec,
    )
    snapshot = snapshot_manifest(snapshot_spec, "lifecycle-origination-v1", masks)
    if min(snapshot["counts"][name] for name in ("train", "calibration", "decision")) < 500:
        raise RuntimeError(f"Lifecycle point-in-time cohorts are too small: {snapshot['counts']}")
    atomic_json(root / "model" / "artifacts" / "snapshot-lifecycle-v1.json", snapshot)

    survival_train = _sample(frame[masks.train], 40_000)
    survival_test = frame[masks.decision & masks.evaluable].copy()
    hazard = fit_discrete_time_hazard(
        survival_train[SURVIVAL_FEATURES].to_numpy(dtype=float),
        survival_train["durationMonths"].to_numpy(dtype=int),
        survival_train["event"].to_numpy(dtype=np.int8),
        feature_names=SURVIVAL_FEATURES,
        max_horizon=36,
        seed=SEED,
    )
    horizon_metrics = _horizon_evaluation(survival_test, hazard)
    median_features = np.median(survival_train[SURVIVAL_FEATURES].to_numpy(dtype=float), axis=0)
    median_curve = predict_term_structure(hazard, median_features, horizon=36)
    survival_artifact = {
        "schemaVersion": 1,
        "name": "CRIX-LifetimePD",
        "version": "1.0.0",
        "product": "unsecured-personal-loan",
        "populationConditioning": "historically-granted-LendingClub-loans",
        "target": {
            "name": "timeToResolvedDefaultProxy",
            "definition": "Discrete-time hazard of charged-off/default resolution; last_pymnt_d is used as an endpoint proxy, not claimed to be an exact charge-off date.",
            "censoring": "fully-paid loans are competing terminal non-default outcomes; unresolved statuses are right-censored at last_credit_pull_d",
            "maxHorizonMonths": 36,
        },
        "featureProvenance": [
            {"feature": "debtToIncome", "source": "dti", "availability": "origination"},
            {"feature": "loanToIncome", "source": "loan_amnt / annual_inc", "availability": "origination"},
            {"feature": "employmentYears", "source": "emp_length", "availability": "origination"},
        ],
        "model": hazard,
        "snapshot": snapshot,
        "metrics": horizon_metrics,
        "exampleMedianBorrower": {
            "features": dict(zip(SURVIVAL_FEATURES, median_features.tolist())),
            "termStructure": term_structure_at_horizons(median_curve),
        },
    }
    artifact_dir = root / "model" / "artifacts"
    survival_path = artifact_dir / "crix-lifetime-pd-v1.json"
    atomic_json(survival_path, survival_artifact)
    lifecycle_dataset_meta = [{
        "id": LIFECYCLE_SOURCE_ID,
        "product": "unsecured-personal-loan",
        "sourceSha256": LIFECYCLE_LENDINGCLUB_SHA256,
        "license": LIFECYCLE_LICENSE,
        "sourceRows": lifecycle.source_rows,
        "usableRows": lifecycle.usable_rows,
    }]
    _research_manifest(
        root=root,
        artifact_path=survival_path,
        model_id="CRIX-LifetimePD@1.0.0",
        model_name="CRIX-LifetimePD",
        version="1.0.0",
        product="unsecured-personal-loan",
        target=survival_artifact["target"],
        datasets=lifecycle_dataset_meta,
        feature_contract_version="lifecycle-survival-v1",
        split=snapshot,
        training_config={"features": SURVIVAL_FEATURES, "maxHorizonMonths": 36, "sampleLimit": 40_000},
        metrics={"decisionSamples": int(len(survival_test)), "trainingLoans": hazard["trainingLoans"], "events": hazard["events"]},
    )

    # Installment EAD and LGD are conditional-at-default models and remain research-only.
    defaults = frame[(frame["event"] == 1) & (frame["observedEad"] > 0)].copy()
    train_defaults = defaults[(defaults["issueDate"] <= "2012-12-31") & (defaults["outcomeAvailableAt"] <= "2015-12-31")]
    test_defaults = defaults[(defaults["issueDate"] > "2012-12-31") & (defaults["outcomeAvailableAt"] <= "2016-12-31")]
    train_defaults = _sample(train_defaults, 60_000, seed=SEED)
    test_defaults = _sample(test_defaults, 40_000, seed=SEED + 1)
    if len(train_defaults) < 500 or len(test_defaults) < 250:
        raise RuntimeError(f"Default cohorts are too small for EAD/LGD research: {len(train_defaults)}, {len(test_defaults)}")
    conditional_features = ["debtToIncome", "loanToIncome", "employmentYears", "termMonths", "durationMonths"]
    train_x = train_defaults[conditional_features].to_numpy(dtype=float)
    test_x = test_defaults[conditional_features].to_numpy(dtype=float)

    train_ead_rate = observed_installment_ead(
        train_defaults["fundedAmount"].to_numpy(dtype=float),
        train_defaults["principalReceived"].to_numpy(dtype=float),
    ) / train_defaults["fundedAmount"].to_numpy(dtype=float)
    test_funded = test_defaults["fundedAmount"].to_numpy(dtype=float)
    test_ead = test_defaults["observedEad"].to_numpy(dtype=float)
    ead_models = _regression_comparison(train_x, train_ead_rate, test_x, test_ead / test_funded, prediction_bounds=(0.0, 1.0))
    baseline_ead = np.clip(ead_models.pop("baselinePrediction"), 0, 1) * test_funded
    challenger_ead = np.clip(ead_models.pop("challengerPrediction"), 0, 1) * test_funded
    ead_artifact = {
        "schemaVersion": 1,
        "name": "CRIX-EAD-Installment",
        "version": "1.0.0",
        "product": "unsecured-personal-loan",
        "target": "principal exposure proxy at default = max(funded_amnt - total_rec_prncp, 0)",
        "targetBounds": [0, 1],
        "targetScale": "EAD / funded amount; predictions explicitly bounded to [0,1] for closed-end principal exposure",
        "featureNames": conditional_features,
        "models": ead_models,
        "metrics": {
            "trainDefaults": int(len(train_defaults)),
            "testDefaults": int(len(test_defaults)),
            "baselineDollarMae": float(mean_absolute_error(test_ead, baseline_ead)),
            "challengerDollarMae": float(mean_absolute_error(test_ead, challenger_ead)),
        },
        "v3RuntimeBaseline": "ead = requested loanAmount remains unchanged in /api/v3",
    }
    ead_path = artifact_dir / "crix-ead-installment-v1.json"
    atomic_json(ead_path, ead_artifact)
    _research_manifest(
        root=root,
        artifact_path=ead_path,
        model_id="CRIX-EAD-Installment@1.0.0",
        model_name="CRIX-EAD-Installment",
        version="1.0.0",
        product="unsecured-personal-loan",
        target={"name": "exposureAtDefaultProxy", "definition": ead_artifact["target"], "horizon": "conditional on default month proxy"},
        datasets=lifecycle_dataset_meta,
        feature_contract_version="lifecycle-ead-v1",
        split={"trainOriginationsThrough": "2012-12-31", "trainLabelsAsOf": "2015-12-31", "testOriginationsAfter": "2012-12-31", "testLabelsAsOf": "2016-12-31"},
        training_config={"features": conditional_features, "targetBounds": [0, 1]},
        metrics=ead_artifact["metrics"],
    )

    train_lgd = economic_lgd(
        train_defaults["observedEad"].to_numpy(dtype=float),
        train_defaults["recoveries"].to_numpy(dtype=float),
        train_defaults["recoveryCosts"].to_numpy(dtype=float),
    )
    test_lgd = economic_lgd(
        test_ead,
        test_defaults["recoveries"].to_numpy(dtype=float),
        test_defaults["recoveryCosts"].to_numpy(dtype=float),
    )
    lgd_train_x = np.column_stack([train_x, train_defaults["observedEad"].to_numpy(dtype=float) / train_defaults["fundedAmount"].to_numpy(dtype=float)])
    lgd_test_x = np.column_stack([test_x, test_ead / test_funded])
    lgd_models = _regression_comparison(lgd_train_x, train_lgd, lgd_test_x, test_lgd)
    baseline_lgd = lgd_models.pop("baselinePrediction")
    challenger_lgd = lgd_models.pop("challengerPrediction")
    weights = test_ead / test_ead.sum()
    lgd_artifact = {
        "schemaVersion": 1,
        "name": "CRIX-LGD",
        "version": "1.0.0",
        "product": "unsecured-personal-loan",
        "target": "LGD = (EAD - (recoveries - collection_recovery_fee)) / EAD",
        "discountRate": 0.0,
        "recoveryTimingSupport": "aggregate-only; source does not provide dated recovery cashflows, so recovery dates are not fabricated",
        "featureNames": conditional_features + ["observedEadRate"],
        "models": lgd_models,
        "metrics": {
            "trainDefaults": int(len(train_defaults)),
            "testDefaults": int(len(test_defaults)),
            "baselineMae": float(mean_absolute_error(test_lgd, baseline_lgd)),
            "challengerMae": float(mean_absolute_error(test_lgd, challenger_lgd)),
            "baselineExposureWeightedMae": float(np.sum(np.abs(test_lgd - baseline_lgd) * weights)),
            "challengerExposureWeightedMae": float(np.sum(np.abs(test_lgd - challenger_lgd) * weights)),
            "observedMin": float(test_lgd.min()),
            "observedMax": float(test_lgd.max()),
        },
        "v3RuntimeBaseline": "CRIX v3 engineering LGD remains unchanged and separately versioned",
    }
    lgd_path = artifact_dir / "crix-lgd-v1.json"
    atomic_json(lgd_path, lgd_artifact)
    _research_manifest(
        root=root,
        artifact_path=lgd_path,
        model_id="CRIX-LGD@1.0.0",
        model_name="CRIX-LGD",
        version="1.0.0",
        product="unsecured-personal-loan",
        target={"name": "economicLossSeverityProxy", "definition": lgd_artifact["target"], "recoveryTimingSupport": lgd_artifact["recoveryTimingSupport"]},
        datasets=lifecycle_dataset_meta,
        feature_contract_version="lifecycle-lgd-v1",
        split={"trainOriginationsThrough": "2012-12-31", "trainLabelsAsOf": "2015-12-31", "testOriginationsAfter": "2012-12-31", "testLabelsAsOf": "2016-12-31"},
        training_config={"features": lgd_artifact["featureNames"], "discountRate": 0.0},
        metrics=lgd_artifact["metrics"],
    )

    taiwan = load_taiwan_credit()
    transitions_artifact = _transition_artifact(taiwan)
    transition_path = artifact_dir / "crix-transitions-taiwan-v1.json"
    atomic_json(transition_path, transitions_artifact)
    taiwan_meta = [{
        "id": "uci-taiwan-credit-card-default",
        "product": "revolving-credit-card",
        "doi": TAIWAN_DOI,
        "license": TAIWAN_LICENSE,
        "rows": taiwan.rows,
        "retrievedDataFingerprintSha256": taiwan.fingerprint_sha256,
    }]
    _research_manifest(
        root=root,
        artifact_path=transition_path,
        model_id="CRIX-Transitions-Taiwan@1.0.0",
        model_name="CRIX-Transitions-Taiwan",
        version="1.0.0",
        product="revolving-credit-card",
        target={"name": "monthlyDelinquencyMigration", "definition": "Observed Apr-Sep 2005 repayment-state transitions plus observed next-month default only"},
        datasets=taiwan_meta,
        feature_contract_version="uci-taiwan-transitions-v1",
        split={"history": "Apr-Sep-2005", "outcome": "Oct-2005 default indicator"},
        training_config={"states": list(DEFAULT_STATES), "absorbing": ["DEFAULT"]},
        metrics={"rows": taiwan.rows, "observedTransitions": transitions_artifact["observedTransitions"]},
    )

    ccf_artifact = _ccf_artifact(taiwan)
    ccf_path = artifact_dir / "crix-ccf-taiwan-v1.json"
    atomic_json(ccf_path, ccf_artifact)
    _research_manifest(
        root=root,
        artifact_path=ccf_path,
        model_id="CRIX-CCF-Taiwan@1.0.0",
        model_name="CRIX-CCF-Taiwan",
        version="1.0.0",
        product="revolving-credit-card",
        target={"name": "netBalanceChangeCcfProxy", "definition": ccf_artifact["target"], "warning": ccf_artifact["warning"]},
        datasets=taiwan_meta,
        feature_contract_version="uci-taiwan-ccf-v1",
        split={"trainTransitions": "Apr-Aug 2005", "testTransition": "Aug-Sep 2005"},
        training_config={"features": ccf_artifact["featureNames"], "targetPolicy": ccf_artifact["targetPolicy"]},
        metrics=ccf_artifact["metrics"],
    )

    # Prove the legacy v3 chronological split is reproducible only in an explicitly retrospective-resolved mode.
    granting_path = root / "model" / "data" / "LC_loans_granting_model_dataset.csv"
    granting = harmonize_lendingclub(granting_path)
    v3_spec = SnapshotSpec(
        train_end="2015-12-31",
        calibration_end="2016-12-31",
        decision_end="2017-12-31",
        outcome_cutoff="2018-12-31",
        source_sha256=sha256_file(granting_path),
        mode="retrospective-resolved",
    )
    v3_masks = build_snapshot_masks(granting.frame["issueDate"].to_numpy(dtype="datetime64[D]"), None, v3_spec)
    v3_snapshot = snapshot_manifest(v3_spec, "crix-granting-features-v1", v3_masks)
    champion = json.loads((artifact_dir / "crix-monoboost-v2.json").read_text())
    expected = champion["training"]["split"]
    actual = v3_snapshot["counts"]
    if actual["train"] != expected["train"]["samples"] or actual["calibration"] != expected["calibration"]["samples"] or actual["decision"] != expected["test"]["samples"]:
        raise RuntimeError(f"V3 retrospective snapshot does not reproduce committed split: {actual} vs {expected}")
    atomic_json(artifact_dir / "snapshot-v3-retrospective.json", v3_snapshot)

    report = f"""# CRIX research risk-stack report\n\nGenerated by `model/train_risk_stack.py`. These are research models, not bank/regulatory validation approvals. `/api/v3` semantics remain unchanged.\n\n## Lifetime PD\n\n- Source: {LIFECYCLE_SOURCE_ID}, SHA-256 `{LIFECYCLE_LENDINGCLUB_SHA256}`\n- Endpoint semantics: `last_pymnt_d` is an event/terminal-date proxy; no exact charge-off date is claimed.\n- Point-in-time snapshot: `{snapshot['snapshotId']}`\n- Features: {', '.join(SURVIVAL_FEATURES)}\n- Horizon metrics: `{json.dumps(horizon_metrics, sort_keys=True)}`\n\n## Delinquency transitions\n\n- Source: UCI Default of Credit Card Clients, DOI {TAIWAN_DOI}\n- Product: revolving credit card; not pooled with LendingClub personal loans.\n- States: {', '.join(DEFAULT_STATES)}\n- October non-default state is not fabricated.\n\n## EAD\n\n- Installment target: funded principal less principal received at default-resolution proxy.\n- Current `/api/v3` `EAD = loanAmount` remains the runtime baseline.\n- Taiwan CCF research is explicitly a net-balance-change proxy, not pure draw CCF.\n\n## LGD\n\n- Target: `(EAD - (recoveries - collection_recovery_fee)) / EAD`.\n- Recovery timing: aggregate-only; discount rate is 0 because dated recovery cashflows are unavailable.\n- Current `/api/v3` engineering LGD remains unchanged.\n\n## Historical time machine\n\n- Strict lifecycle snapshot separates origination windows from label-availability cutoffs.\n- The legacy v3 split is reproduced only under `retrospective-resolved` mode and is not mislabeled as a historical point-in-time training snapshot.\n"""
    (root / "model" / "RISK_STACK_REPORT.md").write_text(report)
    print(json.dumps({
        "survival": survival_artifact["metrics"],
        "ead": ead_artifact["metrics"],
        "lgd": lgd_artifact["metrics"],
        "transitions": {"rows": taiwan.rows, "observedTransitions": transitions_artifact["observedTransitions"]},
        "ccf": ccf_artifact["metrics"],
        "registryModels": sorted(json.loads(registry_path.read_text())["models"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
