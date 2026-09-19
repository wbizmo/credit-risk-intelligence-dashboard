from __future__ import annotations

import json

import numpy as np
from scipy.stats import genpareto

from evt import fit_pot_tail
from portfolio_risk import (
    _as_float_vector,
    _dependency_thresholds,
    _resolve_backend,
    _resolve_dependency,
    _simulate_loss_vector,
    simulate_portfolio,
)


def _synthetic_evt(shape: float, scale: float) -> dict:
    count = 1999
    body = np.linspace(0.0, 10.0, 10_000 - count)
    u = (np.arange(count, dtype=float) + 0.5) / count
    losses = np.concatenate((body, 10.0 + genpareto.ppf(u, c=shape, scale=scale)))
    fit = fit_pot_tail(
        losses,
        threshold_quantile=0.80,
        target_quantiles=(0.95, 0.99),
        threshold_candidates=(0.80, 0.85, 0.90, 0.95),
    )
    return {
        "trueShape": shape,
        "trueScale": scale,
        "estimatedShape": fit["shape"],
        "estimatedScale": fit["scale"],
        "status": fit["status"],
        "thresholdStability": fit["thresholdStability"],
    }


def main() -> None:
    n = 120
    pd = np.full(n, 0.04)
    lgd = np.full(n, 0.5)
    ead = np.linspace(500.0, 1500.0, n)
    common = dict(scenarios=30_000, seed=2026, quantiles=(0.95, 0.99))

    independent = simulate_portfolio(pd, lgd, ead, rho=0.0, **common)
    low = simulate_portfolio(pd, lgd, ead, rho=0.05, **common)
    high = simulate_portfolio(pd, lgd, ead, rho=0.40, **common)
    student = simulate_portfolio(
        pd,
        lgd,
        ead,
        dependency_model={"name": "student-t", "rho": 0.05, "df": 4},
        **common,
    )

    target_expected = float(np.sum(pd * lgd * ead))

    pd_v = _as_float_vector(pd, "pd")
    lgd_v = _as_float_vector(lgd, "lgd")
    ead_v = _as_float_vector(ead, "ead")
    spec = _resolve_dependency(n, 0.20, None)
    thresholds = _dependency_thresholds(pd_v, spec)
    loss_vector = _simulate_loss_vector(
        loss_given_default=lgd_v * ead_v,
        thresholds=thresholds,
        spec=spec,
        scenarios=100_000,
        chunk_size=2048,
        seed=3030,
        backend=_resolve_backend("numpy"),
    )
    empirical_var99 = float(np.quantile(loss_vector, 0.99, method="higher"))
    empirical_tail99 = loss_vector[loss_vector >= empirical_var99]
    empirical_es99 = float(np.mean(empirical_tail99))
    evt_comparison = fit_pot_tail(
        loss_vector,
        threshold_quantile=0.95,
        target_quantiles=(0.99,),
        threshold_candidates=(0.90, 0.95, 0.975),
    )

    report = {
        "status": "research-only; synthetic validation, not Basel/IRB validation",
        "marginalExpectedLoss": {
            "analyticIndependent": target_expected,
            "gaussianIndependent": independent["expectedLoss"],
            "gaussianHighDependence": high["expectedLoss"],
            "studentT": student["expectedLoss"],
        },
        "tailComparison": {
            "gaussianLowRhoEs99": low["expectedShortfall"]["0.99"],
            "gaussianHighRhoEs99": high["expectedShortfall"]["0.99"],
            "studentTLowRhoDf4Es99": student["expectedShortfall"]["0.99"],
            "highVsLowRatio": high["expectedShortfall"]["0.99"] / low["expectedShortfall"]["0.99"],
            "studentTVsGaussianLowRatio": student["expectedShortfall"]["0.99"] / low["expectedShortfall"]["0.99"],
        },
        "empiricalVsEvt": {
            "scenarios": 100_000,
            "empiricalVar99": empirical_var99,
            "empiricalExpectedShortfall99": empirical_es99,
            "evtStatus": evt_comparison["status"],
            "evtVar99": evt_comparison["targets"].get("0.99", {}).get("var"),
            "evtExpectedShortfall99": evt_comparison["targets"].get("0.99", {}).get("expectedShortfall"),
            "evtThresholdQuantile": evt_comparison["thresholdQuantile"],
            "note": "Separate estimates are reported side-by-side; EVT never overwrites empirical Monte Carlo.",
        },
        "evtRecovery": [
            _synthetic_evt(-0.10, 2.0),
            _synthetic_evt(0.20, 3.0),
            _synthetic_evt(0.45, 2.0),
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
