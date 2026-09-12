from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MacroObservation:
    reference_date: str
    published_at: str
    value: float

    def __post_init__(self) -> None:
        date.fromisoformat(self.reference_date)
        date.fromisoformat(self.published_at)
        if not math.isfinite(float(self.value)):
            raise ValueError("macro observation value must be finite")


def asof_join_macro(as_of_dates: Iterable[str], observations: Iterable[MacroObservation]) -> np.ndarray:
    """Backward as-of join by publication availability, never reference date alone."""
    obs = sorted(observations, key=lambda item: (item.published_at, item.reference_date))
    if not obs:
        raise ValueError("at least one macro observation is required")
    result: list[float] = []
    for raw in as_of_dates:
        as_of = date.fromisoformat(raw)
        eligible = [item for item in obs if date.fromisoformat(item.published_at) <= as_of]
        if not eligible:
            result.append(float("nan"))
            continue
        latest = max(eligible, key=lambda item: (item.published_at, item.reference_date))
        result.append(float(latest.value))
    return np.asarray(result, dtype=float)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def fit_univariate_logit(x: Iterable[float], y: Iterable[int], *, max_iter: int = 100, tolerance: float = 1e-10) -> dict:
    """Fit an auditable intercept + one-driver logistic model with Wald CI."""
    x_v = np.asarray(list(x), dtype=float)
    y_v = np.asarray(list(y), dtype=float)
    if x_v.ndim != 1 or y_v.ndim != 1 or x_v.size != y_v.size or x_v.size < 3:
        raise ValueError("x and y must be equal-length one-dimensional vectors with at least three rows")
    if not np.all(np.isfinite(x_v)) or not np.all(np.isfinite(y_v)):
        raise ValueError("x and y must be finite")
    if not np.all(np.isin(y_v, [0.0, 1.0])) or np.unique(y_v).size < 2:
        raise ValueError("y must contain both binary classes")
    if float(np.std(x_v)) == 0.0:
        raise ValueError("x must vary")

    design = np.column_stack([np.ones(x_v.size), x_v])
    beta = np.zeros(2, dtype=float)
    for _ in range(max_iter):
        p = np.clip(_sigmoid(design @ beta), 1e-9, 1.0 - 1e-9)
        w = p * (1.0 - p)
        hessian = design.T @ (w[:, None] * design)
        gradient = design.T @ (y_v - p)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise ValueError("macro logit Hessian is singular") from exc
        beta_next = beta + step
        if float(np.max(np.abs(step))) < tolerance:
            beta = beta_next
            break
        beta = beta_next
    else:
        raise ValueError("macro logit did not converge")

    p = np.clip(_sigmoid(design @ beta), 1e-9, 1.0 - 1e-9)
    w = p * (1.0 - p)
    information = design.T @ (w[:, None] * design)
    covariance = np.linalg.inv(information)
    standard_error = math.sqrt(float(covariance[1, 1]))
    coefficient = float(beta[1])
    return {
        "method": "univariate-logistic-macro-overlay",
        "intercept": float(beta[0]),
        "coefficient": coefficient,
        "coefficientStdError": standard_error,
        "coefficientCi95": [coefficient - 1.96 * standard_error, coefficient + 1.96 * standard_error],
        "samples": int(x_v.size),
        "events": int(np.sum(y_v)),
        "support": [float(np.min(x_v)), float(np.max(x_v))],
    }


def stress_pd_path(
    base_pd: Iterable[float],
    baseline_path: Iterable[float],
    scenario_path: Iterable[float],
    *,
    coefficient: float,
    scale: float = 1.0,
    support: tuple[float, float] | None = None,
) -> dict:
    """Apply an estimated macro log-odds overlay to a baseline PD path."""
    pd_v = np.asarray(list(base_pd), dtype=float)
    baseline = np.asarray(list(baseline_path), dtype=float)
    scenario = np.asarray(list(scenario_path), dtype=float)
    if pd_v.ndim != 1 or not (pd_v.size == baseline.size == scenario.size) or pd_v.size == 0:
        raise ValueError("PD and macro paths must be non-empty equal-length vectors")
    if np.any(~np.isfinite(pd_v)) or np.any((pd_v <= 0.0) | (pd_v >= 1.0)):
        raise ValueError("base PD must be finite and strictly inside (0, 1)")
    if np.any(~np.isfinite(baseline)) or np.any(~np.isfinite(scenario)):
        raise ValueError("macro paths must be finite")
    if not math.isfinite(coefficient) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("coefficient must be finite and scale positive")

    stressed = _sigmoid(_logit(pd_v) + coefficient * ((scenario - baseline) / scale))
    extrapolative = False
    if support is not None:
        low, high = map(float, support)
        if not (math.isfinite(low) and math.isfinite(high) and low <= high):
            raise ValueError("support must be a finite ordered pair")
        extrapolative = bool(np.any((scenario < low) | (scenario > high)))
    return {
        "pd": stressed,
        "baselinePd": pd_v,
        "macroDelta": scenario - baseline,
        "coefficient": float(coefficient),
        "scale": float(scale),
        "extrapolative": extrapolative,
        "status": "empirical relationship overlay; correlation is not a causal claim",
    }


def scenario_registry_entry(
    *,
    scenario_id: str,
    driver: str,
    source: str,
    frequency: str,
    geography: str,
    publication_semantics: str,
    baseline_path: Iterable[float],
    stressed_path: Iterable[float],
) -> dict:
    baseline = [float(v) for v in baseline_path]
    stressed = [float(v) for v in stressed_path]
    if not scenario_id or not driver or not source or len(baseline) != len(stressed) or not baseline:
        raise ValueError("scenario metadata and equal non-empty paths are required")
    return {
        "scenarioId": scenario_id,
        "driver": driver,
        "source": source,
        "frequency": frequency,
        "geography": geography,
        "publicationSemantics": publication_semantics,
        "baselinePath": baseline,
        "stressedPath": stressed,
        "version": 1,
    }
