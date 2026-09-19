from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


RESEARCH_STATUS = "research-only; not regulatory capital or bank validation"
_MAX_SCENARIOS = 2_000_000
_MAX_FACTORS = 32
_MAX_QUANTILES = 128
_LOADING_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ArrayBackend:
    name: str
    xp: Any
    device: str
    reproducibility_class: str

    def to_numpy(self, values: Any) -> np.ndarray:
        if self.name == "numpy":
            return np.asarray(values)
        return self.xp.asnumpy(values)


@dataclass(frozen=True)
class DependencySpec:
    name: str
    version: str
    factor_loadings: np.ndarray
    residual_scale: np.ndarray
    rho: float | None
    degrees_of_freedom: float | None
    loading_kind: str

    @property
    def factor_count(self) -> int:
        return int(self.factor_loadings.shape[1])


def _resolve_backend(name: str) -> ArrayBackend:
    normalized = str(name).strip().lower()
    if normalized == "numpy":
        return ArrayBackend(
            name="numpy",
            xp=np,
            device="cpu",
            reproducibility_class="canonical-counter-indexed-chunk-invariant",
        )
    if normalized != "cupy":
        raise ValueError(f"unsupported backend: {name}")

    try:
        import cupy as cp  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "CuPy backend requested but CuPy/CUDA is unavailable. "
            "Install a CUDA-compatible CuPy build and use backend='cupy'."
        ) from exc

    try:
        device_id = int(cp.cuda.runtime.getDevice())
        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        raw_name = properties.get("name", "unknown-gpu")
        device_name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        # Force a tiny allocation so driver/runtime incompatibilities fail here.
        cp.asarray([0.0], dtype=cp.float64)
    except Exception as exc:
        raise RuntimeError(
            "CuPy backend requested but no usable CUDA device/runtime is available."
        ) from exc

    return ArrayBackend(
        name="cupy",
        xp=cp,
        device=device_name,
        reproducibility_class="same-backend-device-repeatable; statistical-equivalence-to-numpy",
    )


def _as_float_vector(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _splitmix64_backend(values: Any, xp: Any) -> Any:
    """Vectorized SplitMix64 finalizer used as a deterministic counter RNG."""
    mask = xp.uint64(0xFFFFFFFFFFFFFFFF)
    z = values.astype(xp.uint64, copy=False)
    z = (z + xp.uint64(0x9E3779B97F4A7C15)) & mask
    z = ((z ^ (z >> xp.uint64(30))) * xp.uint64(0xBF58476D1CE4E5B9)) & mask
    z = ((z ^ (z >> xp.uint64(27))) * xp.uint64(0x94D049BB133111EB)) & mask
    return z ^ (z >> xp.uint64(31))


def _uniform_from_counter_backend(counter: Any, seed: int, xp: Any) -> Any:
    mixed = _splitmix64_backend(counter ^ xp.uint64(seed & 0xFFFFFFFFFFFFFFFF), xp)
    u = ((mixed >> xp.uint64(11)).astype(xp.float64) + 0.5) * (1.0 / (1 << 53))
    eps = np.finfo(float).eps
    return xp.clip(u, eps, 1.0 - eps)


def _uniform_from_counter(counter: np.ndarray, seed: int) -> np.ndarray:
    return _uniform_from_counter_backend(counter, seed, np)


def _xp_any(value: Any, backend: ArrayBackend) -> bool:
    if backend.name == "numpy":
        return bool(np.any(value))
    return bool(backend.xp.asnumpy(backend.xp.any(value)))


def _norm_ppf_backend(p: Any, backend: ArrayBackend) -> Any:
    """Acklam inverse-normal approximation on either NumPy or CuPy."""
    xp = backend.xp
    p = xp.asarray(p, dtype=xp.float64)
    if _xp_any(~xp.isfinite(p), backend) or _xp_any((p <= 0.0) | (p >= 1.0), backend):
        raise ValueError("normal inverse CDF input must lie strictly inside (0, 1)")

    a = xp.asarray([
        -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
        1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
    ], dtype=xp.float64)
    b = xp.asarray([
        -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
        6.680131188771972e01, -1.328068155288572e01,
    ], dtype=xp.float64)
    c = xp.asarray([
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
        -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
    ], dtype=xp.float64)
    d = xp.asarray([
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e00, 3.754408661907416e00,
    ], dtype=xp.float64)
    plow = 0.02425
    phigh = 1.0 - plow
    out = xp.empty_like(p)

    low = p < plow
    if _xp_any(low, backend):
        q = xp.sqrt(-2.0 * xp.log(p[low]))
        out[low] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)

    high = p > phigh
    if _xp_any(high, backend):
        q = xp.sqrt(-2.0 * xp.log(1.0 - p[high]))
        out[high] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)

    mid = ~(low | high)
    if _xp_any(mid, backend):
        q = p[mid] - 0.5
        r = q * q
        out[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    return out


def _norm_ppf(p: np.ndarray | Iterable[float]) -> np.ndarray:
    return np.asarray(_norm_ppf_backend(p, _resolve_backend("numpy")), dtype=float)


def _validate_quantiles(quantiles: Iterable[float]) -> tuple[float, ...]:
    result = tuple(float(q) for q in quantiles)
    if not result or any((not math.isfinite(q) or q <= 0.0 or q >= 1.0) for q in result):
        raise ValueError("quantiles must be finite probabilities strictly between 0 and 1")
    if len(set(result)) != len(result):
        raise ValueError("quantiles must be unique")
    if len(result) > _MAX_QUANTILES:
        raise ValueError(f"quantiles must contain at most {_MAX_QUANTILES} values")
    return result


def _scenario_losses(defaults: Any, loss_given_default: Any, backend: ArrayBackend) -> Any:
    xp = backend.xp
    return xp.sum(defaults.astype(xp.float64) * loss_given_default[None, :], axis=1, dtype=xp.float64)


def _resolve_dependency(
    obligor_count: int,
    rho: float,
    dependency_model: dict[str, Any] | None,
) -> DependencySpec:
    config = dependency_model or {"name": "gaussian", "rho": rho}
    name = str(config.get("name", "gaussian")).strip().lower()
    aliases = {
        "one-factor-gaussian": "gaussian",
        "gaussian": "gaussian",
        "student-t": "student-t",
        "t": "student-t",
        "one-factor-student-t": "student-t",
    }
    if name not in aliases:
        raise ValueError(f"unsupported dependency model: {name}")
    name = aliases[name]

    df: float | None = None
    if name == "student-t":
        raw_df = config.get("degrees_of_freedom", config.get("df", 5.0))
        df = float(raw_df)
        if not math.isfinite(df) or df <= 2.0:
            raise ValueError("Student-t degrees_of_freedom must be finite and greater than 2")

    if "loadings" in config:
        if "rho" in config:
            raise ValueError("dependency_model must specify either rho or loadings, not both")
        loadings = np.asarray(config["loadings"], dtype=float)
        if loadings.ndim != 2 or loadings.shape[0] != obligor_count or loadings.shape[1] == 0:
            raise ValueError("loadings must have shape (obligors, factors)")
        if loadings.shape[1] > _MAX_FACTORS:
            raise ValueError(f"factor count must not exceed {_MAX_FACTORS}")
        if not np.all(np.isfinite(loadings)):
            raise ValueError("loadings must contain only finite values")
        squared = np.sum(loadings * loadings, axis=1)
        if np.any(squared > 1.0 + _LOADING_TOLERANCE):
            raise ValueError("sum of squared factor loadings per obligor must not exceed 1")
        residual = np.sqrt(np.maximum(0.0, 1.0 - squared))
        return DependencySpec(
            name=name,
            version=f"{name}-low-rank-v1",
            factor_loadings=loadings,
            residual_scale=residual,
            rho=None,
            degrees_of_freedom=df,
            loading_kind="explicit-low-rank",
        )

    selected_rho = float(config.get("rho", rho))
    if not math.isfinite(selected_rho) or selected_rho < 0.0 or selected_rho >= 1.0:
        raise ValueError("rho must lie in [0, 1)")
    loadings = np.full((obligor_count, 1), math.sqrt(selected_rho), dtype=np.float64)
    residual = np.full(obligor_count, math.sqrt(1.0 - selected_rho), dtype=np.float64)
    return DependencySpec(
        name=name,
        version=f"{name}-one-factor-v1",
        factor_loadings=loadings,
        residual_scale=residual,
        rho=selected_rho,
        degrees_of_freedom=df,
        loading_kind="homogeneous-one-factor",
    )


def _student_t_thresholds(pd_v: np.ndarray, df: float) -> np.ndarray:
    try:
        from scipy.stats import t as student_t
    except Exception as exc:
        raise RuntimeError("Student-t dependency requires scipy; install model/requirements.txt") from exc
    thresholds = np.asarray(student_t.ppf(pd_v, df=df), dtype=np.float64)
    if not np.all(np.isfinite(thresholds)):
        raise ValueError("Student-t thresholds are non-finite")
    return thresholds


def _student_t_scale(start: int, stop: int, seed: int, df: float) -> np.ndarray:
    try:
        from scipy.stats import chi2
    except Exception as exc:
        raise RuntimeError("Student-t dependency requires scipy; install model/requirements.txt") from exc
    scenario_index = np.arange(start, stop, dtype=np.uint64)
    uniforms = _uniform_from_counter(scenario_index, seed ^ 0xC3C3C3C3)
    draws = np.asarray(chi2.ppf(uniforms, df=df), dtype=np.float64)
    if not np.all(np.isfinite(draws)) or np.any(draws <= 0.0):
        raise RuntimeError("Student-t scale-mixture draw failed")
    return np.sqrt(draws / df)


def _dependency_thresholds(pd_v: np.ndarray, spec: DependencySpec) -> np.ndarray:
    if spec.name == "student-t":
        assert spec.degrees_of_freedom is not None
        return _student_t_thresholds(pd_v, spec.degrees_of_freedom)
    return _norm_ppf(pd_v)


def _defaults_chunk(
    *,
    start: int,
    stop: int,
    obligor_count: int,
    seed: int,
    thresholds: np.ndarray,
    spec: DependencySpec,
    backend: ArrayBackend,
) -> Any:
    xp = backend.xp
    scenario_index = xp.arange(start, stop, dtype=xp.uint64)
    idio_counters = scenario_index[:, None] * xp.uint64(obligor_count) + xp.arange(obligor_count, dtype=xp.uint64)[None, :]
    idiosyncratic = _norm_ppf_backend(
        _uniform_from_counter_backend(idio_counters, seed ^ 0x5A5A5A5A, xp),
        backend,
    )

    if spec.loading_kind == "homogeneous-one-factor":
        systematic = _norm_ppf_backend(
            _uniform_from_counter_backend(scenario_index, seed ^ 0xA5A5A5A5, xp),
            backend,
        )
        assert spec.rho is not None
        gaussian = math.sqrt(spec.rho) * systematic[:, None] + math.sqrt(1.0 - spec.rho) * idiosyncratic
    else:
        factor_count = spec.factor_count
        factor_counters = scenario_index[:, None] * xp.uint64(factor_count) + xp.arange(factor_count, dtype=xp.uint64)[None, :]
        systematic = _norm_ppf_backend(
            _uniform_from_counter_backend(factor_counters, seed ^ 0xA5A5A5A5, xp),
            backend,
        )
        loadings = xp.asarray(spec.factor_loadings, dtype=xp.float64)
        residual = xp.asarray(spec.residual_scale, dtype=xp.float64)
        gaussian = systematic @ loadings.T + idiosyncratic * residual[None, :]

    if spec.name == "student-t":
        assert spec.degrees_of_freedom is not None
        scale = xp.asarray(
            _student_t_scale(start, stop, seed, spec.degrees_of_freedom),
            dtype=xp.float64,
        )
        latent = gaussian / scale[:, None]
    else:
        latent = gaussian

    threshold_values = xp.asarray(thresholds, dtype=xp.float64)
    return latent < threshold_values[None, :]


def _simulate_loss_vector(
    *,
    loss_given_default: np.ndarray,
    thresholds: np.ndarray,
    spec: DependencySpec,
    scenarios: int,
    chunk_size: int,
    seed: int,
    backend: ArrayBackend,
) -> np.ndarray:
    losses = np.empty(scenarios, dtype=np.float64)
    loss_backend = backend.xp.asarray(loss_given_default, dtype=backend.xp.float64)
    n = loss_given_default.size
    for start in range(0, scenarios, chunk_size):
        stop = min(scenarios, start + chunk_size)
        defaults = _defaults_chunk(
            start=start,
            stop=stop,
            obligor_count=n,
            seed=seed,
            thresholds=thresholds,
            spec=spec,
            backend=backend,
        )
        chunk_losses = _scenario_losses(defaults, loss_backend, backend)
        losses[start:stop] = backend.to_numpy(chunk_losses).astype(np.float64, copy=False)
    return losses


def _tail_contributions_shared_pass(
    *,
    losses: np.ndarray,
    quantile_values: dict[float, float],
    loss_given_default: np.ndarray,
    thresholds: np.ndarray,
    spec: DependencySpec,
    scenarios: int,
    chunk_size: int,
    seed: int,
    backend: ArrayBackend,
) -> dict[str, list[float]]:
    """Replay every scenario once and aggregate disjoint loss buckets.

    Tail sets are nested by threshold. Each scenario's obligor-loss vector is
    summed into exactly one bucket, then bucket totals are accumulated from the
    most severe bucket down. This avoids Q complete S x N replays.
    """
    if not quantile_values:
        return {}

    xp = backend.xp
    n = loss_given_default.size
    ordered = sorted(quantile_values.items(), key=lambda item: (item[1], item[0]))
    ordered_thresholds = np.asarray([threshold for _, threshold in ordered], dtype=np.float64)
    bucket_totals = np.zeros((len(ordered) + 1, n), dtype=np.float64)
    loss_backend = xp.asarray(loss_given_default, dtype=xp.float64)

    for start in range(0, scenarios, chunk_size):
        stop = min(scenarios, start + chunk_size)
        defaults = _defaults_chunk(
            start=start,
            stop=stop,
            obligor_count=n,
            seed=seed,
            thresholds=thresholds,
            spec=spec,
            backend=backend,
        )
        weighted = defaults.astype(xp.float64) * loss_backend[None, :]
        bucket_index = np.searchsorted(ordered_thresholds, losses[start:stop], side="right")

        for bucket in range(1, len(ordered) + 1):
            local_mask = bucket_index == bucket
            if not np.any(local_mask):
                continue
            backend_mask = xp.asarray(local_mask)
            subtotal = xp.sum(weighted[backend_mask], axis=0, dtype=xp.float64)
            bucket_totals[bucket] += backend.to_numpy(subtotal).astype(np.float64, copy=False)

    tail_counts = {
        q: int(np.sum(losses >= threshold))
        for q, threshold in quantile_values.items()
    }
    totals_by_quantile: dict[float, np.ndarray] = {}
    running = np.zeros(n, dtype=np.float64)
    for ordered_index in range(len(ordered) - 1, -1, -1):
        running = running + bucket_totals[ordered_index + 1]
        q = ordered[ordered_index][0]
        totals_by_quantile[q] = running.copy()

    return {
        str(q): (totals_by_quantile[q] / tail_counts[q]).tolist()
        for q in quantile_values
        if tail_counts[q] > 0
    }


def _tail_contributions_reference(
    *,
    losses: np.ndarray,
    quantile_values: dict[float, float],
    loss_given_default: np.ndarray,
    thresholds: np.ndarray,
    spec: DependencySpec,
    scenarios: int,
    chunk_size: int,
    seed: int,
) -> dict[str, list[float]]:
    """Pre-v3.2 replay algorithm retained only for equivalence benchmarks/tests."""
    n = loss_given_default.size
    backend = _resolve_backend("numpy")
    contributions: dict[str, list[float]] = {}
    for q, threshold in quantile_values.items():
        mask = losses >= threshold
        tail_count = int(np.sum(mask))
        totals = np.zeros(n, dtype=np.float64)
        if not tail_count:
            continue
        for start in range(0, scenarios, chunk_size):
            stop = min(scenarios, start + chunk_size)
            local_mask = mask[start:stop]
            if not np.any(local_mask):
                continue
            defaults = _defaults_chunk(
                start=start,
                stop=stop,
                obligor_count=n,
                seed=seed,
                thresholds=thresholds,
                spec=spec,
                backend=backend,
            )
            totals += np.sum(
                defaults[local_mask].astype(np.float64) * loss_given_default[None, :],
                axis=0,
                dtype=np.float64,
            )
        contributions[str(q)] = (totals / tail_count).tolist()
    return contributions


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
    backend: str = "numpy",
    dependency_model: dict[str, Any] | None = None,
) -> dict:
    """Simulate bounded-memory latent credit losses for offline research.

    NumPy plus one-factor Gaussian dependence remains the canonical deterministic
    evidence path. RNG values are counter-indexed, so seeded NumPy runs are
    invariant to chunk size. GPU acceleration is opt-in and never used by the
    live TypeScript API.
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
    if not isinstance(scenarios, int) or scenarios <= 0 or scenarios > _MAX_SCENARIOS:
        raise ValueError(f"scenarios must lie in [1, {_MAX_SCENARIOS:,}]")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    qs = _validate_quantiles(quantiles)
    n = pd_v.size
    backend_impl = _resolve_backend(backend)
    spec = _resolve_dependency(n, rho, dependency_model)
    thresholds = _dependency_thresholds(pd_v, spec)
    loss_given_default = lgd_v * ead_v
    losses = _simulate_loss_vector(
        loss_given_default=loss_given_default,
        thresholds=thresholds,
        spec=spec,
        scenarios=scenarios,
        chunk_size=chunk_size,
        seed=seed,
        backend=backend_impl,
    )

    expected = float(np.mean(losses))
    unexpected = float(np.std(losses, ddof=0))
    stderr = unexpected / math.sqrt(scenarios)
    independent_expected = float(np.sum(pd_v * loss_given_default))
    independent_unexpected = float(np.sqrt(np.sum(pd_v * (1.0 - pd_v) * loss_given_default**2)))

    var: dict[str, float | None] = {}
    es: dict[str, float | None] = {}
    support: dict[str, dict[str, Any]] = {}
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

    contributions = (
        _tail_contributions_shared_pass(
            losses=losses,
            quantile_values=quantile_values,
            loss_given_default=loss_given_default,
            thresholds=thresholds,
            spec=spec,
            scenarios=scenarios,
            chunk_size=chunk_size,
            seed=seed,
            backend=backend_impl,
        )
        if tail_contributions
        else {}
    )

    digest = hashlib.sha256(losses.astype("<f8", copy=False).tobytes()).hexdigest()
    dependency_metadata: dict[str, Any] = {
        "name": spec.name,
        "version": spec.version,
        "factorCount": spec.factor_count,
        "loadingSpecification": spec.loading_kind,
        "loadingShape": [int(spec.factor_loadings.shape[0]), int(spec.factor_loadings.shape[1])],
        "maxSquaredLoadingSum": float(np.max(np.sum(spec.factor_loadings**2, axis=1))),
    }
    if spec.rho is not None:
        dependency_metadata["rho"] = float(spec.rho)
    if spec.degrees_of_freedom is not None:
        dependency_metadata["degreesOfFreedom"] = float(spec.degrees_of_freedom)

    if spec.name == "gaussian" and spec.loading_kind == "homogeneous-one-factor":
        method = "one-factor-gaussian-latent-default"
    elif spec.name == "student-t" and spec.loading_kind == "homogeneous-one-factor":
        method = "one-factor-student-t-latent-default"
    else:
        method = f"low-rank-{spec.name}-latent-default"

    return {
        "method": method,
        "rho": float(spec.rho) if spec.rho is not None else None,
        "dependencyModel": dependency_metadata,
        "backend": {
            "name": backend_impl.name,
            "device": backend_impl.device,
            "dtype": "float64",
            "seed": int(seed),
            "chunkSize": int(chunk_size),
            "reproducibilityClass": backend_impl.reproducibility_class,
        },
        "scenarios": int(scenarios),
        "seed": int(seed),
        "portfolioSize": int(n),
        "expectedLoss": expected,
        "unexpectedLoss": unexpected,
        "var": var,
        "expectedShortfall": es,
        "tailSupport": support,
        "tailContributions": contributions,
        "tailAttribution": {
            "algorithm": "single-replay-disjoint-buckets-v1",
            "boundedMemory": True,
            "fullScenarioObligorMatrixRetained": False,
        },
        "independentBaseline": {"expectedLoss": independent_expected, "unexpectedLoss": independent_unexpected},
        "monteCarlo": {"expectedLossStdError": stderr},
        "lossDigest": digest,
        "status": RESEARCH_STATUS,
    }
