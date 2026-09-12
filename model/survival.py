from __future__ import annotations

from typing import Mapping

import numpy as np


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


def term_structure_at_horizons(term_structure: Mapping[str, np.ndarray], horizons: tuple[int, ...] = (3, 6, 12, 24, 36)) -> dict[str, float | None]:
    validate_term_structure(term_structure)
    cumulative = np.asarray(term_structure["cumulativePd"], dtype=float)
    return {f"pd{h}m": float(cumulative[h - 1]) if h <= len(cumulative) else None for h in horizons}
