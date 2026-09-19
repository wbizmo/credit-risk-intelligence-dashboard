from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata


EXPLANATION_VALIDATION_VERSION = "crix-explanation-fidelity-v1"


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _calibrated_probability(calibrator, margin: np.ndarray) -> np.ndarray:
    values = np.asarray(margin, dtype=float).reshape(-1, 1)
    probability = calibrator.predict_proba(values)[:, 1]
    return np.clip(probability, 0.0001, 0.9999)


def _ordered_features(values: np.ndarray) -> np.ndarray:
    magnitude = np.abs(np.asarray(values, dtype=float).reshape(-1))
    return np.lexsort((np.arange(len(magnitude)), -magnitude))


def top_k_overlap(left: np.ndarray, right: np.ndarray, *, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    a = _ordered_features(left)[:k]
    b = _ordered_features(right)[:k]
    denominator = min(k, len(a), len(b))
    if denominator == 0:
        return 0.0
    return len(set(a.tolist()) & set(b.tolist())) / denominator


def spearman_explanation_rank(left: np.ndarray, right: np.ndarray) -> float | None:
    a = np.abs(np.asarray(left, dtype=float).reshape(-1))
    b = np.abs(np.asarray(right, dtype=float).reshape(-1))
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("rank comparison requires aligned vectors with at least two features")
    a_rank = rankdata(a, method="average")
    b_rank = rankdata(b, method="average")
    if np.allclose(a_rank, a_rank[0]) or np.allclose(b_rank, b_rank[0]):
        return None
    value = float(np.corrcoef(a_rank, b_rank)[0, 1])
    return value if np.isfinite(value) else None


def sign_agreement(left: np.ndarray, right: np.ndarray, *, tolerance: float = 1e-12) -> float | None:
    a = np.asarray(left, dtype=float).reshape(-1)
    b = np.asarray(right, dtype=float).reshape(-1)
    if len(a) != len(b):
        raise ValueError("sign comparison requires aligned vectors")
    mask = (np.abs(a) > tolerance) & (np.abs(b) > tolerance)
    if not mask.any():
        return None
    return float(np.mean(np.sign(a[mask]) == np.sign(b[mask])))


def validate_champion_metadata(
    *,
    model_name: str,
    model_version: str,
    expected_name: str,
    expected_version: str,
) -> None:
    if model_name != expected_name or model_version != expected_version:
        raise ValueError(
            "explanation validation model metadata does not match the governed champion identity"
        )


def _local_sensitivity(
    model,
    calibrator,
    X: np.ndarray,
    *,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    baseline_margin = np.asarray(model.predict(X, output_margin=True), dtype=float)
    baseline_pd = _calibrated_probability(calibrator, baseline_margin)
    impacts = np.zeros((len(X), X.shape[1]), dtype=np.float64)
    for index in range(X.shape[1]):
        changed = np.array(X, dtype=np.float32, copy=True)
        changed[:, index] = reference[index]
        changed_margin = np.asarray(model.predict(changed, output_margin=True), dtype=float)
        changed_pd = _calibrated_probability(calibrator, changed_margin)
        impacts[:, index] = baseline_pd - changed_pd
    return baseline_pd, impacts


def _small_valid_perturbation(
    X: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    width = np.maximum(upper - lower, 1e-9)
    direction = np.where(np.arange(X.shape[1]) % 2 == 0, 1.0, -1.0)
    delta = width * 0.01 * direction
    return np.clip(X + delta[None, :], lower[None, :], upper[None, :]).astype(np.float32)


def _metrics(
    shap_values: np.ndarray,
    local_values: np.ndarray,
    stability_values: np.ndarray,
    *,
    k: int,
) -> dict[str, object]:
    if len(shap_values) == 0:
        return {"status": "insufficient-data", "count": 0}
    overlaps = [
        top_k_overlap(shap_values[index], local_values[index], k=k)
        for index in range(len(shap_values))
    ]
    ranks = [
        value
        for index in range(len(shap_values))
        if (value := spearman_explanation_rank(shap_values[index], local_values[index])) is not None
    ]
    signs = [
        value
        for index in range(len(shap_values))
        if (value := sign_agreement(shap_values[index], local_values[index])) is not None
    ]
    stability = [
        top_k_overlap(local_values[index], stability_values[index], k=k)
        for index in range(len(local_values))
    ]
    return {
        "status": "pass",
        "count": int(len(shap_values)),
        "topK": int(k),
        "meanTopKOverlap": float(np.mean(overlaps)),
        "meanAbsoluteRankCorrelation": float(np.mean(ranks)) if ranks else None,
        "meanSignAgreement": float(np.mean(signs)) if signs else None,
        "meanPerturbationTopKStability": float(np.mean(stability)),
    }


def _segment_rows(
    shap_values: np.ndarray,
    local_values: np.ndarray,
    stability_values: np.ndarray,
    labels: np.ndarray,
    *,
    k: int,
    minimum_count: int,
) -> list[dict[str, object]]:
    raw = np.asarray(labels).astype(str)
    rows: list[dict[str, object]] = []
    for label in sorted(set(raw.tolist())):
        mask = raw == label
        count = int(mask.sum())
        if count < minimum_count:
            rows.append({"segment": label, "status": "insufficient-data", "count": count})
            continue
        row = _metrics(
            shap_values[mask],
            local_values[mask],
            stability_values[mask],
            k=k,
        )
        rows.append({"segment": label, **row})
    return rows


def _bands(
    X: np.ndarray,
    pd: np.ndarray,
    feature_names: Sequence[str],
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, np.ndarray]:
    by_name = {name: X[:, index] for index, name in enumerate(feature_names)}
    credit = by_name["creditScore"]
    dti = by_name["debtToIncome"]
    lti = by_name["loanToIncome"]
    ood = np.any((X < lower[None, :]) | (X > upper[None, :]), axis=1)
    return {
        "creditScoreBand": np.select(
            [credit < 620, credit < 660, credit < 700, credit < 740],
            ["<620", "620-659", "660-699", "700-739"],
            default="740+",
        ),
        "debtToIncomeBand": np.select(
            [dti < 0.20, dti < 0.35, dti < 0.50, dti < 0.70],
            ["<0.20", "0.20-0.34", "0.35-0.49", "0.50-0.69"],
            default="0.70+",
        ),
        "loanToIncomeBand": np.select(
            [lti < 0.15, lti < 0.30, lti < 0.50, lti < 0.80],
            ["<0.15", "0.15-0.29", "0.30-0.49", "0.50-0.79"],
            default="0.80+",
        ),
        "pdBand": np.select(
            [pd < 0.10, pd < 0.20, pd < 0.35],
            ["<0.10", "0.10-0.19", "0.20-0.34"],
            default="0.35+",
        ),
        "distributionStatus": np.where(ood, "OOD", "in-distribution"),
    }


def build_explanation_fidelity_report(
    model,
    calibrator,
    validation_X: np.ndarray,
    *,
    feature_names: Sequence[str],
    reference: Mapping[str, float],
    training_bounds: Mapping[str, Mapping[str, float]],
    model_name: str,
    model_version: str,
    expected_model_name: str,
    expected_model_version: str,
    seed: int = 42,
    sample_size: int = 512,
    top_k: int = 3,
    minimum_segment_count: int = 30,
) -> dict[str, object]:
    validate_champion_metadata(
        model_name=model_name,
        model_version=model_version,
        expected_name=expected_model_name,
        expected_version=expected_model_version,
    )
    X = np.asarray(validation_X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != len(feature_names):
        raise ValueError("validation feature matrix does not match champion feature contract")
    if not np.isfinite(X).all():
        raise ValueError("explanation validation requires finite governed features")

    rng = np.random.default_rng(seed)
    size = min(sample_size, len(X))
    if size < max(50, minimum_segment_count):
        return {
            "schemaVersion": 1,
            "version": EXPLANATION_VALIDATION_VERSION,
            "status": "insufficient-data",
            "sampleSize": size,
            "minimumRequired": max(50, minimum_segment_count),
        }
    indices = np.sort(rng.choice(len(X), size=size, replace=False))
    sample = X[indices]

    reference_vector = np.asarray([reference[name] for name in feature_names], dtype=np.float32)
    lower = np.asarray([training_bounds[name]["p01"] for name in feature_names], dtype=np.float32)
    upper = np.asarray([training_bounds[name]["p99"] for name in feature_names], dtype=np.float32)
    reference_vector = np.clip(reference_vector, lower, upper)

    baseline_pd, local = _local_sensitivity(
        model,
        calibrator,
        sample,
        reference=reference_vector,
    )
    perturbed = _small_valid_perturbation(sample, lower, upper)
    _, perturbed_local = _local_sensitivity(
        model,
        calibrator,
        perturbed,
        reference=reference_vector,
    )

    try:
        import shap
    except Exception as exc:
        raise RuntimeError("offline SHAP validation requires model/requirements.txt") from exc

    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(sample)
    if isinstance(raw_shap, list):
        raw_shap = raw_shap[-1]
    shap_values = np.asarray(raw_shap, dtype=np.float64)
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]
    if shap_values.shape != local.shape:
        raise RuntimeError(
            f"SHAP output shape {shap_values.shape} does not match local sensitivity shape {local.shape}"
        )

    calibration_slope = float(calibrator.coef_[0, 0])
    shap_direction = shap_values * (1.0 if calibration_slope >= 0 else -1.0)
    overall = _metrics(shap_direction, local, perturbed_local, k=top_k)
    segments = {
        name: _segment_rows(
            shap_direction,
            local,
            perturbed_local,
            labels,
            k=top_k,
            minimum_count=minimum_segment_count,
        )
        for name, labels in _bands(sample, baseline_pd, feature_names, lower, upper).items()
    }

    return {
        "schemaVersion": 1,
        "version": EXPLANATION_VALIDATION_VERSION,
        "status": "research-governance",
        "model": {
            "name": model_name,
            "version": model_version,
            "featureNames": list(feature_names),
        },
        "sample": {
            "method": "fixed-seed-without-replacement",
            "seed": seed,
            "size": int(size),
        },
        "methods": {
            "reference": "offline SHAP TreeExplainer on champion tree model",
            "liveComparable": "deterministic local champion sensitivity to governed training reference",
            "policyReasons": "separate deterministic policy layer",
            "legalAdverseAction": "not established by this report",
        },
        "overall": overall,
        "segments": segments,
        "privacy": "aggregate-only; no borrower IDs, rows or raw feature vectors",
        "runtimeImpact": "none; SHAP remains offline and is not imported by the Fastify request path",
        "limitations": [
            "SHAP agreement validates model-explanation consistency only.",
            "SHAP values are not automatically legal adverse-action reasons.",
            "Local sensitivity and SHAP answer related but non-identical counterfactual questions.",
        ],
    }
