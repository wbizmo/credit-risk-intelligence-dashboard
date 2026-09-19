from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.stats import genpareto


RESEARCH_DISCLAIMER = (
    "research-only EVT tail extrapolation; not regulatory capital, bank validation, "
    "or a replacement for empirical Monte Carlo VaR/ES"
)
_MAX_BOOTSTRAP_SAMPLES = 500
_DEFAULT_THRESHOLD_CANDIDATES = (0.95, 0.975, 0.99)


def _loss_vector(losses: Iterable[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("losses must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("losses must contain only finite values")
    if np.any(values < 0.0):
        raise ValueError("losses must be non-negative")
    return values


def _probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0 or value >= 1.0:
        raise ValueError(f"{name} must lie strictly inside (0, 1)")
    return value


def _fit_excesses(excesses: np.ndarray, minimum_exceedances: int) -> dict:
    if excesses.size < minimum_exceedances:
        return {
            "status": "insufficient-data",
            "shape": None,
            "scale": None,
            "reason": f"requires at least {minimum_exceedances} strict exceedances",
        }
    if float(np.ptp(excesses)) <= np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(excesses)))):
        return {
            "status": "unstable-fit",
            "shape": None,
            "scale": None,
            "reason": "tail excesses are degenerate",
        }

    try:
        shape, location, scale = genpareto.fit(excesses, floc=0.0)
    except Exception as exc:
        return {
            "status": "unstable-fit",
            "shape": None,
            "scale": None,
            "reason": f"GPD optimizer failed: {type(exc).__name__}",
        }

    if (
        not math.isfinite(float(shape))
        or not math.isfinite(float(scale))
        or abs(float(location)) > 1e-12
        or float(scale) <= 0.0
    ):
        return {
            "status": "unstable-fit",
            "shape": None,
            "scale": None,
            "reason": "GPD fit returned non-finite or invalid parameters",
        }

    return {
        "status": "ok",
        "shape": float(shape),
        "scale": float(scale),
        "reason": None,
    }


def _tail_quantile(
    *,
    threshold: float,
    exceedance_fraction: float,
    shape: float,
    scale: float,
    target_quantile: float,
) -> float | None:
    survival = (1.0 - target_quantile) / exceedance_fraction
    if survival <= 0.0 or survival >= 1.0:
        return None
    if abs(shape) < 1e-10:
        excess = -scale * math.log(survival)
    else:
        base = survival ** (-shape)
        excess = scale * (base - 1.0) / shape
    value = threshold + excess
    if not math.isfinite(value):
        return None
    return float(value)


def _expected_shortfall(
    *,
    threshold: float,
    quantile_value: float,
    shape: float,
    scale: float,
) -> float | None:
    if shape >= 1.0:
        return None
    value = (quantile_value + scale - shape * threshold) / (1.0 - shape)
    return float(value) if math.isfinite(value) else None


def _fit_at_threshold(
    losses: np.ndarray,
    threshold_quantile: float,
    minimum_exceedances: int,
) -> dict:
    threshold = float(np.quantile(losses, threshold_quantile, method="higher"))
    excesses = losses[losses > threshold] - threshold
    fit = _fit_excesses(excesses, minimum_exceedances)
    return {
        "thresholdQuantile": float(threshold_quantile),
        "thresholdValue": threshold,
        "exceedanceCount": int(excesses.size),
        "exceedanceFraction": float(excesses.size / losses.size),
        "status": fit["status"],
        "shape": fit["shape"],
        "scale": fit["scale"],
        "reason": fit["reason"],
        "_excesses": excesses,
    }


def _bootstrap_intervals(
    *,
    excesses: np.ndarray,
    threshold: float,
    exceedance_fraction: float,
    target_quantiles: tuple[float, ...],
    bootstrap_samples: int,
    seed: int,
    minimum_exceedances: int,
) -> dict[str, dict[str, list[float] | None]]:
    if bootstrap_samples == 0:
        return {}

    rng = np.random.default_rng(seed)
    samples: dict[str, dict[str, list[float]]] = {
        str(q): {"var": [], "expectedShortfall": []}
        for q in target_quantiles
    }
    for _ in range(bootstrap_samples):
        resampled = rng.choice(excesses, size=excesses.size, replace=True)
        fit = _fit_excesses(resampled, minimum_exceedances)
        if fit["status"] != "ok":
            continue
        shape = float(fit["shape"])
        scale = float(fit["scale"])
        for q in target_quantiles:
            quantile_value = _tail_quantile(
                threshold=threshold,
                exceedance_fraction=exceedance_fraction,
                shape=shape,
                scale=scale,
                target_quantile=q,
            )
            if quantile_value is None:
                continue
            samples[str(q)]["var"].append(quantile_value)
            es = _expected_shortfall(
                threshold=threshold,
                quantile_value=quantile_value,
                shape=shape,
                scale=scale,
            )
            if es is not None:
                samples[str(q)]["expectedShortfall"].append(es)

    intervals: dict[str, dict[str, list[float] | None]] = {}
    for q in target_quantiles:
        key = str(q)
        var_samples = np.asarray(samples[key]["var"], dtype=np.float64)
        es_samples = np.asarray(samples[key]["expectedShortfall"], dtype=np.float64)
        intervals[key] = {
            "varCi95": (
                [float(v) for v in np.quantile(var_samples, [0.025, 0.975])]
                if var_samples.size >= max(10, bootstrap_samples // 2)
                else None
            ),
            "expectedShortfallCi95": (
                [float(v) for v in np.quantile(es_samples, [0.025, 0.975])]
                if es_samples.size >= max(10, bootstrap_samples // 2)
                else None
            ),
        }
    return intervals


def fit_pot_tail(
    losses: Iterable[float] | np.ndarray,
    *,
    threshold_quantile: float = 0.975,
    target_quantiles: tuple[float, ...] = (0.99, 0.999),
    threshold_candidates: tuple[float, ...] = _DEFAULT_THRESHOLD_CANDIDATES,
    bootstrap_samples: int = 0,
    seed: int = 42,
    minimum_exceedances: int = 50,
) -> dict:
    """Fit research-only Peaks-Over-Threshold GPD tail models.

    This output is intentionally separate from empirical Monte Carlo VaR/ES.
    Fit failures are explicit and never replaced by fallback tail numbers.
    """
    values = _loss_vector(losses)
    threshold_quantile = _probability(threshold_quantile, "threshold_quantile")
    targets = tuple(_probability(q, "target_quantile") for q in target_quantiles)
    if not targets or any(q <= threshold_quantile for q in targets):
        raise ValueError("target_quantiles must all be greater than threshold_quantile")
    if len(set(targets)) != len(targets):
        raise ValueError("target_quantiles must be unique")
    if not isinstance(minimum_exceedances, int) or minimum_exceedances < 20:
        raise ValueError("minimum_exceedances must be an integer >= 20")
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 0 or bootstrap_samples > _MAX_BOOTSTRAP_SAMPLES:
        raise ValueError(f"bootstrap_samples must lie in [0, {_MAX_BOOTSTRAP_SAMPLES}]")

    candidates = {
        _probability(q, "threshold_candidate")
        for q in threshold_candidates
    }
    candidates.add(threshold_quantile)
    ordered_candidates = tuple(sorted(candidates))
    stability: list[dict] = []
    selected: dict | None = None

    for candidate in ordered_candidates:
        fit = _fit_at_threshold(values, candidate, minimum_exceedances)
        public_fit = {key: value for key, value in fit.items() if key != "_excesses"}
        stability.append(public_fit)
        if candidate == threshold_quantile:
            selected = fit

    assert selected is not None
    valid_shapes = [
        float(item["shape"])
        for item in stability
        if item["status"] == "ok" and item["shape"] is not None
    ]
    shape_span = max(valid_shapes) - min(valid_shapes) if len(valid_shapes) >= 2 else None
    stability_status = "stable"
    if shape_span is not None and shape_span > 0.35:
        stability_status = "unstable-fit"

    overall_status = str(selected["status"])
    if overall_status == "ok" and stability_status == "unstable-fit":
        overall_status = "unstable-fit"

    result = {
        "method": "peaks-over-threshold-gpd-v1",
        "fitMethod": "scipy.stats.genpareto.fit with location fixed at zero",
        "status": overall_status,
        "thresholdQuantile": selected["thresholdQuantile"],
        "thresholdValue": selected["thresholdValue"],
        "exceedanceCount": selected["exceedanceCount"],
        "exceedanceFraction": selected["exceedanceFraction"],
        "shape": selected["shape"],
        "scale": selected["scale"],
        "targets": {},
        "thresholdStability": {
            "status": stability_status,
            "shapeSpan": shape_span,
            "candidates": stability,
        },
        "bootstrap": {
            "samples": int(bootstrap_samples),
            "seed": int(seed),
            "maximumSamples": _MAX_BOOTSTRAP_SAMPLES,
        },
        "statusDetail": selected["reason"],
        "researchOnly": True,
        "disclaimer": RESEARCH_DISCLAIMER,
    }

    if selected["status"] != "ok":
        return result

    shape = float(selected["shape"])
    scale = float(selected["scale"])
    exceedance_fraction = float(selected["exceedanceFraction"])
    threshold = float(selected["thresholdValue"])
    intervals = _bootstrap_intervals(
        excesses=selected["_excesses"],
        threshold=threshold,
        exceedance_fraction=exceedance_fraction,
        target_quantiles=targets,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        minimum_exceedances=minimum_exceedances,
    )

    fitted_targets: dict[str, dict] = {}
    for q in targets:
        key = str(q)
        quantile_value = _tail_quantile(
            threshold=threshold,
            exceedance_fraction=exceedance_fraction,
            shape=shape,
            scale=scale,
            target_quantile=q,
        )
        if quantile_value is None:
            fitted_targets[key] = {
                "status": "unsupported",
                "var": None,
                "expectedShortfall": None,
                "finiteExpectedShortfall": shape < 1.0,
                **intervals.get(key, {}),
            }
            continue

        es = _expected_shortfall(
            threshold=threshold,
            quantile_value=quantile_value,
            shape=shape,
            scale=scale,
        )
        fitted_targets[key] = {
            "status": "ok" if overall_status == "ok" else overall_status,
            "var": quantile_value,
            "expectedShortfall": es,
            "finiteExpectedShortfall": es is not None,
            **intervals.get(key, {}),
        }

    result["targets"] = fitted_targets
    return result
