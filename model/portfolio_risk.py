from __future__ import annotations

import hashlib
import math
from typing import Iterable

import numpy as np


def _as_float_vector(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Vectorized SplitMix64 finalizer used as a deterministic counter RNG."""
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    z = values.astype(np.uint64, copy=False)
    z = (z + np.uint64(0x9E3779B97F4A7C15)) & mask
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & mask
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & mask
    return z ^ (z >> np.uint64(31))


def _uniform_from_counter(counter: np.ndarray, seed: int) -> np.ndarray:
    mixed = _splitmix64(counter ^ np.uint64(seed & 0xFFFFFFFFFFFFFFFF))
    # Use the top 53 bits so conversion to float64 is exact; avoid 0/1 for inverse CDF.
    u = ((mixed >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / (1 << 53))
    return np.clip(u, np.finfo(float).eps, 1.0 - np.finfo(float).eps)


def _norm_ppf(p: np.ndarray | Iterable[float]) -> np.ndarray:
    """Acklam inverse-normal approximation, vectorized and dependency-free."""
    p = np.asarray(p, dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("normal inverse CDF input must lie strictly inside (0, 1)")

    a = np.array([-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
                  1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00])
    b = np.array([-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
                  6.680131188771972e01, -1.328068155288572e01])
    c = np.array([-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
                  -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00])
    d = np.array([7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
                  3.754408661907416e00])
    plow = 0.02425
    phigh = 1.0 - plow
    out = np.empty_like(p)

    low = p < plow
    if np.any(low):
        q = np.sqrt(-2.0 * np.log(p[low]))
        out[low] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)

    high = p > phigh
    if np.any(high):
        q = np.sqrt(-2.0 * np.log(1.0 - p[high]))
        out[high] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)

    mid = ~(low | high)
    if np.any(mid):
        q = p[mid] - 0.5
        r = q * q
        out[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
                   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    return out


def _validate_quantiles(quantiles: Iterable[float]) -> tuple[float, ...]:
    result = tuple(float(q) for q in quantiles)
    if not result or any((not math.isfinite(q) or q <= 0.0 or q >= 1.0) for q in result):
        raise ValueError("quantiles must be finite probabilities strictly between 0 and 1")
    return result


def simulate_portfolio(
    pd: Iterable[float],
    lgd: Iterable[float],
    ead: Iterable[float],
    *,
    rho: float = 0.15,
    scenarios: int = 100_000,
    seed: int = 42,
    chunk_size: int = 2048,
    quantiles: Iterable[float] = (0.95, 0.99, 0.999),
    tail_contributions: bool = False,
) -> dict:
    """Simulate one-factor Gaussian latent credit losses with bounded memory.

    RNG values are indexed by (scenario, obligor), so changing chunk size does not
    change a seeded run. This is research analytics, not a regulatory capital model.
    """
    pd_v = _as_float_vector(pd, "pd")
    lgd_v = _as_float_vector(lgd, "lgd")
    ead_v = _as_float_vector(ead, "ead")
    if not (pd_v.size == lgd_v.size == ead_v.size):
        raise ValueError("pd, lgd and ead must have identical lengths")
    if np.any((pd_v <= 0.0) | (pd_v >= 1.0)):
        raise ValueError("pd values must lie strictly inside (0, 1)")
    if np.any(lgd_v < 0.0):
        raise ValueError("lgd must be non-negative")
    if np.any(ead_v < 0.0):
        raise ValueError("ead must be non-negative")
    if not math.isfinite(rho) or rho < 0.0 or rho >= 1.0:
        raise ValueError("rho must lie in [0, 1)")
    if scenarios <= 0 or scenarios > 2_000_000:
        raise ValueError("scenarios must lie in [1, 2,000,000]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    qs = _validate_quantiles(quantiles)
    n = pd_v.size
    thresholds = _norm_ppf(pd_v)
    loss_given_default = lgd_v * ead_v
    losses = np.empty(scenarios, dtype=np.float64)
    sqrt_rho = math.sqrt(rho)
    sqrt_idio = math.sqrt(1.0 - rho)

    for start in range(0, scenarios, chunk_size):
        stop = min(scenarios, start + chunk_size)
        scenario_index = np.arange(start, stop, dtype=np.uint64)
        systematic = _norm_ppf(_uniform_from_counter(scenario_index, seed ^ 0xA5A5A5A5))
        counters = scenario_index[:, None] * np.uint64(n) + np.arange(n, dtype=np.uint64)[None, :]
        idiosyncratic = _norm_ppf(_uniform_from_counter(counters, seed ^ 0x5A5A5A5A))
        latent = sqrt_rho * systematic[:, None] + sqrt_idio * idiosyncratic
        defaults = latent < thresholds[None, :]
        losses[start:stop] = defaults @ loss_given_default

    expected = float(np.mean(losses))
    unexpected = float(np.std(losses, ddof=0))
    stderr = unexpected / math.sqrt(scenarios)
    independent_expected = float(np.sum(pd_v * loss_given_default))
    independent_unexpected = float(np.sqrt(np.sum(pd_v * (1.0 - pd_v) * loss_given_default**2)))

    var: dict[str, float | None] = {}
    es: dict[str, float | None] = {}
    support: dict[str, dict] = {}
    quantile_values: dict[float, float] = {}
    for q in qs:
        key = str(q)
        minimum = 100_000 if q >= 0.999 else max(1_000, int(math.ceil(100.0 / (1.0 - q))))
        supported = scenarios >= minimum
        reason = None if supported else f"requires at least {minimum} scenarios for this tail precision policy"
        support[key] = {"supported": supported, "minimumScenarios": minimum, "reason": reason}
        if supported:
            threshold = float(np.quantile(losses, q, method="higher"))
            quantile_values[q] = threshold
            tail = losses[losses >= threshold]
            var[key] = threshold
            es[key] = float(np.mean(tail)) if tail.size else threshold
        else:
            var[key] = None
            es[key] = None

    contributions: dict[str, list[float]] = {}
    if tail_contributions:
        for q, threshold in quantile_values.items():
            mask = losses >= threshold
            tail_count = int(np.sum(mask))
            totals = np.zeros(n, dtype=float)
            if tail_count:
                for start in range(0, scenarios, chunk_size):
                    stop = min(scenarios, start + chunk_size)
                    local_mask = mask[start:stop]
                    if not np.any(local_mask):
                        continue
                    scenario_index = np.arange(start, stop, dtype=np.uint64)
                    systematic = _norm_ppf(_uniform_from_counter(scenario_index, seed ^ 0xA5A5A5A5))
                    counters = scenario_index[:, None] * np.uint64(n) + np.arange(n, dtype=np.uint64)[None, :]
                    idio = _norm_ppf(_uniform_from_counter(counters, seed ^ 0x5A5A5A5A))
                    defaults = (sqrt_rho * systematic[:, None] + sqrt_idio * idio) < thresholds[None, :]
                    totals += np.sum(defaults[local_mask] * loss_given_default[None, :], axis=0)
                contributions[str(q)] = (totals / tail_count).tolist()

    digest = hashlib.sha256(losses.astype("<f8", copy=False).tobytes()).hexdigest()
    return {
        "method": "one-factor-gaussian-latent-default",
        "rho": float(rho),
        "scenarios": int(scenarios),
        "seed": int(seed),
        "portfolioSize": int(n),
        "expectedLoss": expected,
        "unexpectedLoss": unexpected,
        "var": var,
        "expectedShortfall": es,
        "tailSupport": support,
        "tailContributions": contributions,
        "independentBaseline": {
            "expectedLoss": independent_expected,
            "unexpectedLoss": independent_unexpected,
        },
        "monteCarlo": {"expectedLossStdError": stderr},
        "lossDigest": digest,
        "status": "research-only; not regulatory capital or bank validation",
    }
