from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

DEFAULT_AGE_BUCKET_ENDS = (3, 6, 12, 18, 24, 36)


def cumulative_from_hazards(hazards: np.ndarray) -> dict[str, np.ndarray]:
    hazard = np.asarray(hazards, dtype=float).reshape(-1)
    if len(hazard) == 0 or not np.isfinite(hazard).all() or ((hazard < 0) | (hazard > 1)).any():
        raise ValueError("hazards must be finite probabilities")
    prior_survival = np.concatenate(([1.0], np.cumprod(1.0 - hazard)[:-1]))
    marginal = prior_survival * hazard
    survival = np.cumprod(1.0 - hazard)
    cumulative = 1.0 - survival
    return {"hazard": hazard, "survival": survival, "marginalPd": marginal, "cumulativePd": cumulative}


def validate_term_structure(term_structure: Mapping[str, np.ndarray], tolerance: float = 1e-10) -> None:
    required = ("hazard", "survival", "marginalPd", "cumulativePd")
    arrays = {name: np.asarray(term_structure[name], dtype=float).reshape(-1) for name in required}
    lengths = {len(value) for value in arrays.values()}
    if lengths == {0} or len(lengths) != 1:
        raise ValueError("term-structure arrays must be non-empty and aligned")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("term structure contains non-finite values")
    if ((arrays["hazard"] < -tolerance) | (arrays["hazard"] > 1 + tolerance)).any():
        raise ValueError("hazards must remain within [0, 1]")
    if ((arrays["survival"] < -tolerance) | (arrays["survival"] > 1 + tolerance)).any():
        raise ValueError("survival must remain within [0, 1]")
    if np.diff(arrays["survival"]).max(initial=0) > tolerance:
        raise ValueError("survival must be non-increasing")
    if np.diff(arrays["cumulativePd"]).min(initial=0) < -tolerance:
        raise ValueError("cumulative PD must be non-decreasing")
    if not np.allclose(arrays["cumulativePd"], 1 - arrays["survival"], atol=tolerance, rtol=0):
        raise ValueError("cumulative PD must equal 1 - survival")
    if not np.allclose(arrays["marginalPd"].sum(), arrays["cumulativePd"][-1], atol=tolerance, rtol=0):
        raise ValueError("marginal PD must reconcile to cumulative PD")


def expand_person_period(
    features: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    *,
    max_horizon: int,
) -> dict[str, np.ndarray]:
    x = np.asarray(features, dtype=float)
    duration = np.asarray(durations, dtype=np.int32).reshape(-1)
    event = np.asarray(events).reshape(-1)
    if x.ndim != 2 or len(x) != len(duration) or len(duration) != len(event):
        raise ValueError("features, durations and events must align")
    if max_horizon < 1 or (duration < 1).any() or not np.isin(event, [0, 1]).all():
        raise ValueError("invalid survival expansion inputs")

    observed = np.minimum(duration, max_horizon)
    row_index = np.repeat(np.arange(len(x)), observed)
    starts = np.repeat(np.cumsum(observed) - observed, observed)
    month = np.arange(int(observed.sum())) - starts + 1
    expanded_event = np.zeros(int(observed.sum()), dtype=np.int8)
    event_observed = (event == 1) & (duration <= max_horizon)
    terminal = np.cumsum(observed)[event_observed] - 1
    expanded_event[terminal] = 1
    return {
        "features": x[row_index],
        "rowIndex": row_index,
        "month": month.astype(np.int16),
        "event": expanded_event,
    }


def _age_basis(months: np.ndarray, bucket_ends: Sequence[int]) -> np.ndarray:
    month = np.asarray(months, dtype=np.int32).reshape(-1)
    ends = np.asarray(bucket_ends, dtype=np.int32)
    if len(ends) == 0 or (np.diff(ends) <= 0).any() or (month < 1).any() or month.max(initial=1) > ends[-1]:
        raise ValueError("invalid discrete-time age basis")
    bucket = np.searchsorted(ends, month, side="left")
    return np.eye(len(ends), dtype=float)[bucket]


def _design_matrix(
    features: np.ndarray,
    months: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    bucket_ends: Sequence[int],
) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(means) or len(means) != len(scales):
        raise ValueError("survival feature dimensions do not match model contract")
    standardized = (x - means) / scales
    return np.column_stack([standardized, _age_basis(months, bucket_ends)])


def fit_discrete_time_hazard(
    features: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    *,
    feature_names: Sequence[str],
    max_horizon: int = 36,
    age_bucket_ends: Sequence[int] = DEFAULT_AGE_BUCKET_ENDS,
    seed: int = 42,
) -> dict[str, object]:
    from sklearn.linear_model import LogisticRegression

    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(feature_names) or not np.isfinite(x).all():
        raise ValueError("survival training features are invalid")
    if max_horizon != age_bucket_ends[-1]:
        raise ValueError("final age bucket must equal max horizon")
    expanded = expand_person_period(x, durations, events, max_horizon=max_horizon)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    design = _design_matrix(expanded["features"], expanded["month"], means, scales, age_bucket_ends)
    target = expanded["event"]
    if len(np.unique(target)) != 2:
        raise ValueError("survival training requires observed defaults and non-events")
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=750, random_state=seed)
    model.fit(design, target)
    return {
        "method": "discrete-time-logistic-hazard",
        "featureNames": list(feature_names),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "ageBucketEnds": list(age_bucket_ends),
        "maxHorizonMonths": max_horizon,
        "intercept": float(model.intercept_[0]),
        "coefficients": model.coef_[0].tolist(),
        "trainingLoans": int(len(x)),
        "personPeriods": int(len(target)),
        "events": int(np.asarray(events).sum()),
        "seed": seed,
    }


def _hazard_logits(model: Mapping[str, object], features: np.ndarray, horizon: int) -> np.ndarray:
    names = list(model["featureNames"])
    means = np.asarray(model["means"], dtype=float)
    scales = np.asarray(model["scales"], dtype=float)
    bucket_ends = tuple(int(value) for value in model["ageBucketEnds"])
    coefficients = np.asarray(model["coefficients"], dtype=float)
    intercept = float(model["intercept"])
    maximum = int(model["maxHorizonMonths"])
    x = np.asarray(features, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != len(names) or horizon < 1 or horizon > maximum:
        raise ValueError("invalid lifetime PD inference request")
    repeated = np.repeat(x, horizon, axis=0)
    months = np.tile(np.arange(1, horizon + 1, dtype=np.int16), len(x))
    design = _design_matrix(repeated, months, means, scales, bucket_ends)
    if design.shape[1] != len(coefficients):
        raise ValueError("survival artifact coefficient dimensions are invalid")
    return (intercept + design @ coefficients).reshape(len(x), horizon)


def predict_term_structure(model: Mapping[str, object], features: np.ndarray, horizon: int = 36) -> dict[str, np.ndarray]:
    logits = _hazard_logits(model, np.asarray(features, dtype=float), horizon)
    if logits.shape[0] != 1:
        raise ValueError("predict_term_structure scores one loan; use predict_cumulative_pd_batch for cohorts")
    hazards = 1.0 / (1.0 + np.exp(-np.clip(logits[0], -35, 35)))
    result = cumulative_from_hazards(hazards)
    validate_term_structure(result)
    return result


def predict_cumulative_pd_batch(model: Mapping[str, object], features: np.ndarray, horizon: int) -> np.ndarray:
    logits = _hazard_logits(model, features, horizon)
    hazards = 1.0 / (1.0 + np.exp(-np.clip(logits, -35, 35)))
    return 1.0 - np.prod(1.0 - hazards, axis=1)


def term_structure_at_horizons(term_structure: Mapping[str, np.ndarray], horizons: tuple[int, ...] = (3, 6, 12, 24, 36)) -> dict[str, float | None]:
    validate_term_structure(term_structure)
    cumulative = np.asarray(term_structure["cumulativePd"], dtype=float)
    return {f"pd{h}m": float(cumulative[h - 1]) if h <= len(cumulative) else None for h in horizons}
