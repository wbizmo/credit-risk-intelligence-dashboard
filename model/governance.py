from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np

POPULATION_CONDITIONING = "granted-loans-only"


@dataclass(frozen=True)
class FeatureProvenance:
    feature: str
    sources: tuple[str, ...]
    availability: str
    max_lookback_days: int | None
    outcome_derived: bool = False
    policy_derived: bool = False
    allowed_targets: tuple[str, ...] = ("originationDefaultRisk",)

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["sources"] = list(self.sources)
        value["allowed_targets"] = list(self.allowed_targets)
        return value


PRIMARY_FEATURE_PROVENANCE = (
    FeatureProvenance("debtToIncome", ("dti_n",), "application-time", 0),
    FeatureProvenance("loanToIncome", ("loan_amnt", "revenue"), "application-time", 0),
    FeatureProvenance("creditScore", ("fico_n",), "application-time", 0),
    FeatureProvenance("employmentYears", ("emp_length",), "application-time", 0),
)


def validate_feature_provenance(
    provenance: Sequence[FeatureProvenance],
    required_features: Sequence[str],
) -> dict[str, FeatureProvenance]:
    by_name: dict[str, FeatureProvenance] = {}
    for item in provenance:
        if not item.feature or not item.sources or not item.availability:
            raise ValueError("feature provenance requires feature, source and availability metadata")
        if item.feature in by_name:
            raise ValueError(f"duplicate provenance for feature: {item.feature}")
        if item.outcome_derived:
            raise ValueError(f"outcome-derived feature cannot enter the predictor contract: {item.feature}")
        if item.max_lookback_days is not None and item.max_lookback_days < 0:
            raise ValueError(f"negative feature lookback is invalid: {item.feature}")
        by_name[item.feature] = item

    missing = [feature for feature in required_features if feature not in by_name]
    if missing:
        raise ValueError(f"missing provenance for required feature(s): {', '.join(missing)}")
    return by_name


def assert_point_in_time(
    available_at: np.ndarray,
    as_of: np.ndarray | np.datetime64,
    *,
    label: str,
) -> None:
    available = np.asarray(available_at, dtype="datetime64[ns]")
    cutoff = np.asarray(as_of, dtype="datetime64[ns]")
    if cutoff.ndim == 0:
        cutoff = np.full(available.shape, cutoff, dtype="datetime64[ns]")
    if available.shape != cutoff.shape:
        raise ValueError(f"{label} availability/as-of shapes do not match")
    missing = np.isnat(available) | np.isnat(cutoff)
    if missing.any():
        raise ValueError(f"{label} contains {int(missing.sum())} missing availability/as-of value(s)")
    future = available > cutoff
    if future.any():
        count = int(future.sum())
        noun = "value" if count == 1 else "values"
        raise ValueError(f"{label} contains {count} feature {noun} unavailable at the declared as-of time")


def _validated_binary_arrays(y_true: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.int8).reshape(-1)
    p = np.asarray(probability, dtype=float).reshape(-1)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("outcome and probability arrays must be non-empty and equal length")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite and within [0, 1]")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("outcomes must be binary")
    return y, p


def calibration_by_bins(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    bins: int = 10,
    min_count: int = 50,
    min_events: int = 1,
) -> list[dict[str, float | int | str]]:
    if bins < 1 or min_count < 1 or min_events < 0:
        raise ValueError("invalid calibration bin configuration")
    y, p = _validated_binary_arrays(y_true, probability)
    quantiles = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    if len(quantiles) == 1:
        bucket = np.zeros(len(p), dtype=np.int16)
        bucket_count = 1
    else:
        bucket = np.searchsorted(quantiles[1:-1], p, side="right")
        bucket_count = len(quantiles) - 1

    rows: list[dict[str, float | int | str]] = []
    for index in range(bucket_count):
        mask = bucket == index
        count = int(mask.sum())
        if count == 0:
            continue
        events = int(y[mask].sum())
        non_events = count - events
        rows.append(
            {
                "bin": index + 1,
                "predicted": float(p[mask].mean()),
                "observed": float(y[mask].mean()),
                "count": count,
                "events": events,
                "status": "ok" if count >= min_count and events >= min_events and non_events >= min_events else "insufficient-data",
            }
        )
    return rows


def calibration_intercept_slope(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    max_iterations: int = 50,
    tolerance: float = 1e-9,
) -> dict[str, float] | None:
    y, p = _validated_binary_arrays(y_true, probability)
    if len(y) < 4 or len(np.unique(y)) != 2:
        return None
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    design = np.column_stack([np.ones(len(logits)), logits])
    beta = np.array([0.0, 1.0], dtype=float)
    ridge = np.diag([1e-8, 1e-6])

    for _ in range(max_iterations):
        linear = np.clip(design @ beta, -35, 35)
        fitted = 1 / (1 + np.exp(-linear))
        weight = np.maximum(fitted * (1 - fitted), 1e-9)
        gradient = design.T @ (y - fitted) - ridge @ beta
        information = design.T @ (weight[:, None] * design) + ridge
        step = np.linalg.solve(information, gradient)
        step_norm = float(np.linalg.norm(step))
        if step_norm > 5:
            step *= 5 / step_norm
        beta += step
        if float(np.linalg.norm(step)) < tolerance:
            break

    if not np.isfinite(beta).all():
        return None
    return {"intercept": float(beta[0]), "slope": float(beta[1])}


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("auc requires both outcome classes")
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    ranks = np.empty(len(p), dtype=float)
    start = 0
    while start < len(p):
        end = start + 1
        while end < len(p) and sorted_p[end] == sorted_p[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    positive_rank_sum = float(ranks[y == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _ks(y: np.ndarray, p: np.ndarray) -> float:
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ks requires both outcome classes")
    order = np.argsort(p, kind="mergesort")
    observed = y[order]
    positive_cdf = np.cumsum(observed == 1) / positives
    negative_cdf = np.cumsum(observed == 0) / negatives
    return float(np.max(np.abs(positive_cdf - negative_cdf)))


def _metric(y: np.ndarray, p: np.ndarray, metric: str) -> float:
    if metric == "brier":
        return float(np.mean((p - y) ** 2))
    if metric == "logLoss":
        clipped = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))
    if metric == "auc":
        return _auc(y, p)
    if metric == "ks":
        return _ks(y, p)
    raise ValueError(f"unsupported metric: {metric}")


def bootstrap_metric_interval(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    metric: Literal["brier", "logLoss", "auc", "ks"] = "brier",
    samples: int = 500,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    y, p = _validated_binary_arrays(y_true, probability)
    if samples < 20 or not 0 < confidence < 1:
        raise ValueError("bootstrap requires at least 20 samples and confidence within (0, 1)")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        index = rng.integers(0, len(y), len(y))
        sample_y = y[index]
        if metric in {"auc", "ks"} and len(np.unique(sample_y)) != 2:
            continue
        values.append(_metric(sample_y, p[index], metric))
    if len(values) < max(10, samples // 2):
        raise ValueError("too few valid bootstrap resamples")
    alpha = (1 - confidence) / 2
    return {
        "metric": metric,
        "point": _metric(y, p, metric),
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1 - alpha)),
        "confidence": confidence,
        "samples": len(values),
        "seed": seed,
    }


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    baseline = np.asarray(expected, dtype=float).reshape(-1)
    current = np.asarray(actual, dtype=float).reshape(-1)
    if len(baseline) == 0 or len(current) == 0 or bins < 2:
        raise ValueError("psi requires non-empty arrays and at least two bins")
    if not np.isfinite(baseline).all() or not np.isfinite(current).all():
        raise ValueError("psi inputs must be finite")
    points = np.unique(np.quantile(baseline, np.linspace(0, 1, bins + 1)))
    if len(points) == 1:
        pivot = points[0]
        edges = np.array([-np.inf, pivot, np.nextafter(pivot, np.inf), np.inf])
    else:
        edges = np.concatenate(([-np.inf], points[1:-1], [np.inf]))
    expected_count, _ = np.histogram(baseline, bins=edges)
    actual_count, _ = np.histogram(current, bins=edges)
    expected_share = np.maximum(expected_count / len(baseline), epsilon)
    actual_share = np.maximum(actual_count / len(current), epsilon)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def backtest_pd_policy(
    y_true: np.ndarray,
    probability: np.ndarray,
    exposure: np.ndarray,
    *,
    pd_threshold: float,
) -> dict[str, float | int | str | None]:
    y, p = _validated_binary_arrays(y_true, probability)
    ead = np.asarray(exposure, dtype=float).reshape(-1)
    if len(ead) != len(y) or not np.isfinite(ead).all() or (ead < 0).any():
        raise ValueError("exposure must be finite, non-negative and aligned to outcomes")
    if not 0 <= pd_threshold <= 1:
        raise ValueError("pd threshold must be within [0, 1]")

    selected = p <= pd_threshold
    selected_count = int(selected.sum())
    total_exposure = float(ead.sum())
    selected_exposure = float(ead[selected].sum())
    return {
        "populationConditioning": POPULATION_CONDITIONING,
        "counterfactualRejectedOutcomes": "unsupported",
        "count": len(y),
        "selectedCount": selected_count,
        "selectionRate": selected_count / len(y),
        "observedDefaultRate": float(y[selected].mean()) if selected_count else None,
        "averagePd": float(p[selected].mean()) if selected_count else None,
        "totalExposure": total_exposure,
        "selectedExposure": selected_exposure,
        "predictedDefaultExposure": float(np.sum(p[selected] * ead[selected])),
        "observedDefaultExposure": float(np.sum(y[selected] * ead[selected])),
        "pdThreshold": pd_threshold,
    }
