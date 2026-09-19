from __future__ import annotations

import argparse
import json
import resource
import time
from typing import Callable

import numpy as np

from portfolio_risk import (
    _as_float_vector,
    _dependency_thresholds,
    _resolve_backend,
    _resolve_dependency,
    _simulate_loss_vector,
    _tail_contributions_reference,
    _tail_contributions_shared_pass,
    simulate_portfolio,
)


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * 1024 if value < 10_000_000 else value)


def _measure(operation: Callable[[], object]) -> dict:
    rss_before = _rss_bytes()
    started = time.perf_counter()
    operation()
    elapsed = time.perf_counter() - started
    return {
        "wallSeconds": elapsed,
        "peakRssBytes": _rss_bytes(),
        "rssPeakDeltaBytes": max(0, _rss_bytes() - rss_before),
    }


def _tail_case(*, n: int, scenarios: int, quantiles: tuple[float, ...], seed: int = 42) -> dict:
    pd = np.linspace(0.015, 0.18, n)
    lgd = np.linspace(0.25, 0.75, n)
    ead = np.geomspace(500.0, 50_000.0, n)
    pd_v = _as_float_vector(pd, "pd")
    lgd_v = _as_float_vector(lgd, "lgd")
    ead_v = _as_float_vector(ead, "ead")
    spec = _resolve_dependency(n, 0.20, None)
    thresholds = _dependency_thresholds(pd_v, spec)
    loss_given_default = lgd_v * ead_v
    backend = _resolve_backend("numpy")
    chunk_size = min(1024, scenarios)
    losses = _simulate_loss_vector(
        loss_given_default=loss_given_default,
        thresholds=thresholds,
        spec=spec,
        scenarios=scenarios,
        chunk_size=chunk_size,
        seed=seed,
        backend=backend,
    )
    quantile_values = {
        q: float(np.quantile(losses, q, method="higher"))
        for q in quantiles
    }

    shared_result: dict = {}
    reference_result: dict = {}

    def run_shared() -> None:
        nonlocal shared_result
        shared_result = _tail_contributions_shared_pass(
            losses=losses,
            quantile_values=quantile_values,
            loss_given_default=loss_given_default,
            thresholds=thresholds,
            spec=spec,
            scenarios=scenarios,
            chunk_size=chunk_size,
            seed=seed,
            backend=backend,
        )

    def run_reference() -> None:
        nonlocal reference_result
        reference_result = _tail_contributions_reference(
            losses=losses,
            quantile_values=quantile_values,
            loss_given_default=loss_given_default,
            thresholds=thresholds,
            spec=spec,
            scenarios=scenarios,
            chunk_size=chunk_size,
            seed=seed,
        )

    shared = _measure(run_shared)
    reference = _measure(run_reference)
    for key in shared_result:
        np.testing.assert_allclose(shared_result[key], reference_result[key], rtol=0, atol=1e-10)

    return {
        "portfolioSize": n,
        "scenarios": scenarios,
        "quantiles": list(quantiles),
        "quantileCount": len(quantiles),
        "shared": shared,
        "reference": reference,
        "speedup": reference["wallSeconds"] / shared["wallSeconds"],
        "equivalent": True,
        "complexity": {
            "shared": "O(S*N + S*log(Q) + Q*N)",
            "reference": "O(Q*S*N)",
            "fullScenarioObligorMatrixRetained": False,
        },
    }


def _gpu_case(n: int, scenarios: int, seed: int) -> dict:
    pd = np.linspace(0.01, 0.18, n)
    lgd = np.linspace(0.25, 0.70, n)
    ead = np.geomspace(500.0, 20_000.0, n)

    cpu_measurement = _measure(
        lambda: simulate_portfolio(
            pd, lgd, ead,
            rho=0.20,
            scenarios=scenarios,
            seed=seed,
            chunk_size=min(2048, scenarios),
            quantiles=(0.95,),
            backend="numpy",
        )
    )

    try:
        cupy_backend = _resolve_backend("cupy")
    except RuntimeError as exc:
        return {
            "portfolioSize": n,
            "scenarios": scenarios,
            "cpu": cpu_measurement,
            "gpu": {"status": "unavailable", "reason": str(exc)},
            "speedup": None,
            "beneficialBackend": "numpy-by-availability",
        }

    cp = cupy_backend.xp
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    gpu_before = int(pool.used_bytes())
    transfer_started = time.perf_counter()
    probe = cp.asarray(ead, dtype=cp.float64)
    cp.asnumpy(probe)
    cp.cuda.Stream.null.synchronize()
    transfer_seconds = time.perf_counter() - transfer_started

    gpu_measurement = _measure(
        lambda: simulate_portfolio(
            pd, lgd, ead,
            rho=0.20,
            scenarios=scenarios,
            seed=seed,
            chunk_size=min(2048, scenarios),
            quantiles=(0.95,),
            backend="cupy",
        )
    )
    cp.cuda.Stream.null.synchronize()
    gpu_after = int(pool.used_bytes())
    speedup = cpu_measurement["wallSeconds"] / gpu_measurement["wallSeconds"]
    return {
        "portfolioSize": n,
        "scenarios": scenarios,
        "cpu": cpu_measurement,
        "gpu": {
            **gpu_measurement,
            "status": "executed",
            "device": cupy_backend.device,
            "memoryUsedBeforeBytes": gpu_before,
            "memoryUsedAfterBytes": gpu_after,
            "hostTransferProbeSeconds": transfer_seconds,
        },
        "speedup": speedup,
        "beneficialBackend": "cupy" if speedup > 1.0 else "numpy",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    scenarios = 12_000 if args.quick else 50_000
    n = 40 if args.quick else 150
    tail_cases = [
        _tail_case(n=n, scenarios=scenarios, quantiles=(0.95,)),
        _tail_case(n=n, scenarios=scenarios, quantiles=(0.90, 0.95, 0.99)),
        _tail_case(n=n, scenarios=scenarios, quantiles=(0.80, 0.90, 0.95, 0.975, 0.99)),
    ]

    gpu_sizes = (
        [(20, 2_000), (80, 8_000), (200, 20_000)]
        if args.quick
        else [(25, 10_000), (200, 100_000), (1000, 250_000)]
    )
    gpu_cases = [_gpu_case(size, count, 700 + index) for index, (size, count) in enumerate(gpu_sizes)]

    print(json.dumps({
        "mode": "quick" if args.quick else "full",
        "numpyVersion": np.__version__,
        "researchOnly": True,
        "tailAttribution": tail_cases,
        "gpuBackend": {
            "canonicalEvidenceBackend": "numpy",
            "cases": gpu_cases,
            "note": (
                "CuPy is optional. CPU-only environments report unavailability rather than "
                "installing CUDA dependencies or fabricating a speedup."
            ),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
