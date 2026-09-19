from __future__ import annotations

import json

import numpy as np
from scipy.stats import genpareto

from evt import fit_pot_tail
from portfolio_risk import simulate_portfolio


def _synthetic_evt(shape: float, scale: float) -> dict:
    body = np.linspace(0.0, 10.0, 9001)
    count = 1999
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
        "evtRecovery": [
            _synthetic_evt(-0.10, 2.0),
            _synthetic_evt(0.20, 3.0),
            _synthetic_evt(0.45, 2.0),
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
