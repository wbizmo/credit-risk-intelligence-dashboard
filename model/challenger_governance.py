from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from datasets import FEATURES, harmonize_lendingclub
from decisioning import ChallengerMetrics, compare_challengers
from governance import build_governance_evidence
from publish_champion_manifest import atomic_json
from train import fixed_segments, ks_statistic, split_chronologically


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _embedded_challenger(artifact: dict, matrix: np.ndarray) -> np.ndarray:
    challenger = artifact["challenger"]
    means = np.asarray(challenger["means"], dtype=float)
    scales = np.asarray(challenger["scales"], dtype=float)
    coefficients = np.asarray(challenger["coefficients"], dtype=float)
    if not (matrix.shape[1] == len(means) == len(scales) == len(coefficients)):
        raise ValueError("embedded challenger dimensions do not match the champion feature contract")
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("embedded challenger contains invalid scales")
    margin = float(challenger["intercept"]) + ((matrix - means) / scales) @ coefficients
    return np.clip(_sigmoid(margin), 1e-6, 1.0 - 1e-6)


def _summary(y: np.ndarray, p: np.ndarray, governance: dict) -> dict:
    calibration = governance["calibration"]["interceptSlope"]
    if calibration is None:
        raise RuntimeError("challenger calibration intercept/slope is unavailable")
    return {
        "auc": float(roc_auc_score(y, p)),
        "ks": float(ks_statistic(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "logLoss": float(log_loss(y, p)),
        "calibrationIntercept": float(calibration["intercept"]),
        "calibrationSlope": float(calibration["slope"]),
        "psi": float(governance["stability"]["calibrationToTestPdPsi"]),
        "uncertainty": governance["uncertainty"],
        "segments": governance["segments"],
    }


def _report(artifact: dict) -> str:
    champion = artifact["models"]["CRIX-MonoBoost@2.0.0"]
    challenger = artifact["models"]["CRIX-MonoBoost@2.0.0::real-data-logistic"]
    recommendation = artifact["recommendation"]
    return f"""# CRIX challenger governance report

This comparison uses the **same 2017 LendingClub out-of-time cohort**, target, feature contract and historically granted population for both the deployed champion and its embedded real-data logistic challenger. No OOT observations are used to refit either model in this comparison.

| Metric | CRIX-MonoBoost 2.0 | Embedded logistic challenger |
|---|---:|---:|
| ROC-AUC | {champion['auc']:.4f} | {challenger['auc']:.4f} |
| KS | {champion['ks']:.4f} | {challenger['ks']:.4f} |
| Brier | {champion['brier']:.4f} | {challenger['brier']:.4f} |
| Log loss | {champion['logLoss']:.4f} | {challenger['logLoss']:.4f} |
| Calibration intercept | {champion['calibrationIntercept']:.4f} | {challenger['calibrationIntercept']:.4f} |
| Calibration slope | {champion['calibrationSlope']:.4f} | {challenger['calibrationSlope']:.4f} |
| Calibration→OOT PSI | {champion['psi']:.4f} | {challenger['psi']:.4f} |

Bootstrap 95% intervals and fixed-segment diagnostics are stored in the JSON artifact for both models. Promotion is not “highest AUC wins”: compatibility, calibration and stability gates apply before discrimination ranks a challenger.

**Governance recommendation:** `{recommendation['recommendedModelId']}`.  
**Rule:** {recommendation['promotionRule']}

The challenger remains embedded in the approved demo artifact; this report does not independently promote it. Historical evidence remains conditional on granted LendingClub loans and does not infer rejected-applicant outcomes.
"""


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "model" / "data" / "LC_loans_granting_model_dataset.csv"
    artifact_path = root / "model" / "artifacts" / "crix-monoboost-v2.json"
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    deployed = json.loads(artifact_path.read_text())
    dataset = harmonize_lendingclub(data_path)
    _, _, calibration, test = split_chronologically(dataset.frame)

    X_cal = calibration[FEATURES].to_numpy(dtype=float)
    y_cal = calibration["target"].to_numpy(dtype=np.int8)
    X_test = test[FEATURES].to_numpy(dtype=float)
    y_test = test["target"].to_numpy(dtype=np.int8)
    challenger_cal = _embedded_challenger(deployed, X_cal)
    challenger_test = _embedded_challenger(deployed, X_test)

    challenger_governance = build_governance_evidence(
        y_cal,
        challenger_cal,
        y_test,
        challenger_test,
        test["loanAmount"].to_numpy(dtype=float),
        feature_names=FEATURES,
        segments=fixed_segments(test),
        bootstrap_samples=100,
        min_segment_count=500,
    )
    challenger = _summary(y_test, challenger_test, challenger_governance)

    champion_governance = deployed["training"]["governance"]
    champion_calibration = champion_governance["calibration"]["interceptSlope"]
    if champion_calibration is None:
        raise RuntimeError("champion governance lacks calibration intercept/slope")
    champion = {
        "auc": float(deployed["metrics"]["auc"]),
        "ks": float(deployed["metrics"]["ks"]),
        "brier": float(deployed["metrics"]["brier"]),
        "logLoss": float(deployed["metrics"]["logLoss"]),
        "calibrationIntercept": float(champion_calibration["intercept"]),
        "calibrationSlope": float(champion_calibration["slope"]),
        "psi": float(champion_governance["stability"]["calibrationToTestPdPsi"]),
        "uncertainty": champion_governance["uncertainty"],
        "segments": champion_governance["segments"],
    }

    compatibility_key = "lendingclub-granted|originationDefaultRisk|final-resolution|2017-oot|crix-granting-features-v1"
    recommendation = compare_challengers(
        [
            ChallengerMetrics(
                "CRIX-MonoBoost@2.0.0", compatibility_key,
                champion["auc"], champion["brier"], champion["calibrationSlope"],
                champion["calibrationIntercept"], champion["psi"], champion["logLoss"],
            ),
            ChallengerMetrics(
                "CRIX-MonoBoost@2.0.0::real-data-logistic", compatibility_key,
                challenger["auc"], challenger["brier"], challenger["calibrationSlope"],
                challenger["calibrationIntercept"], challenger["psi"], challenger["logLoss"],
            ),
        ],
        incumbent_id="CRIX-MonoBoost@2.0.0",
    )

    output = {
        "schemaVersion": 1,
        "name": "CRIX-Challenger-Governance",
        "version": "1.0.0",
        "status": "research-governance",
        "comparabilityKey": compatibility_key,
        "populationConditioning": "granted-loans-only",
        "target": deployed["target"],
        "split": deployed["training"]["split"],
        "featureNames": deployed["featureNames"],
        "models": {
            "CRIX-MonoBoost@2.0.0": champion,
            "CRIX-MonoBoost@2.0.0::real-data-logistic": challenger,
        },
        "recommendation": recommendation,
        "runtimeComplexity": {
            "champion": f"O({len(deployed['trees'])} trees × bounded depth <= 64)",
            "challenger": f"O({len(deployed['featureNames'])} features)",
            "runtimeBenchmark": "npm run benchmark:risk remains the serving hot-path performance gate",
            "artifactBytes": artifact_path.stat().st_size,
        },
        "limitations": [
            "OOT evidence is comparison evidence, not an unbounded tuning set.",
            "Outcomes exist only for historically granted LendingClub loans.",
            "This artifact records governance evidence and does not itself change runtime promotion status.",
        ],
    }

    output_path = root / "model" / "artifacts" / "crix-challenger-governance-v1.json"
    atomic_json(output_path, output)
    (root / "model" / "CHALLENGER_REPORT.md").write_text(_report(output), encoding="utf-8")
    print(json.dumps({
        "artifact": str(output_path.relative_to(root)),
        "recommendedModelId": recommendation["recommendedModelId"],
        "challengerAuc": challenger["auc"],
        "challengerBrier": challenger["brier"],
        "challengerPsi": challenger["psi"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
