from __future__ import annotations

import math
from statistics import NormalDist
from typing import Iterable

_NORMAL = NormalDist()


def irb_corporate_capital(pd: float, lgd: float, ead: float, *, maturity_years: float = 2.5, capital_ratio: float = 0.08) -> dict:
    """Basel IRB-inspired corporate capital formula for research comparison only."""
    pd = float(pd)
    lgd = float(lgd)
    ead = float(ead)
    maturity_years = float(maturity_years)
    capital_ratio = float(capital_ratio)
    if not (0.0 < pd < 1.0):
        raise ValueError("pd must lie strictly inside (0, 1)")
    if not (0.0 <= lgd <= 1.0):
        raise ValueError("IRB-style lgd must lie in [0, 1]")
    if not math.isfinite(ead) or ead < 0.0:
        raise ValueError("ead must be finite and non-negative")
    if not math.isfinite(maturity_years) or maturity_years <= 0.0:
        raise ValueError("maturity_years must be positive")
    if not (0.0 < capital_ratio <= 1.0):
        raise ValueError("capital_ratio must lie in (0, 1]")
    if ead == 0.0:
        return {
            "formulaVersion": "basel-irb-corporate-research-v1",
            "capitalRate": 0.0,
            "capital": 0.0,
            "rwaStyle": 0.0,
            "expectedLoss": 0.0,
            "status": "research only; not regulatory compliance or supervisory approval",
        }

    exp_term = math.exp(-50.0 * pd)
    base = 1.0 - math.exp(-50.0)
    correlation = 0.12 * (1.0 - exp_term) / base + 0.24 * (1.0 - (1.0 - exp_term) / base)
    b = (0.11852 - 0.05478 * math.log(pd)) ** 2
    maturity_adjustment = (1.0 + (maturity_years - 2.5) * b) / (1.0 - 1.5 * b)
    conditional_pd = _NORMAL.cdf((_NORMAL.inv_cdf(pd) + math.sqrt(correlation) * _NORMAL.inv_cdf(0.999)) / math.sqrt(1.0 - correlation))
    capital_rate = max(0.0, lgd * conditional_pd - pd * lgd) * maturity_adjustment
    capital = capital_rate * ead
    rwa_style = capital / capital_ratio
    return {
        "formulaVersion": "basel-irb-corporate-research-v1",
        "pd": pd,
        "lgd": lgd,
        "ead": ead,
        "maturityYears": maturity_years,
        "correlation": correlation,
        "maturityAdjustment": maturity_adjustment,
        "capitalRate": capital_rate,
        "capital": capital,
        "rwaStyle": rwa_style,
        "expectedLoss": pd * lgd * ead,
        "capitalRatioAssumption": capital_ratio,
        "status": "research only; not regulatory compliance or supervisory approval",
    }


def economic_capital(*, expected_loss: float, tail_loss: float, confidence: float, tail_measure: str = "VaR") -> dict:
    expected_loss = float(expected_loss)
    tail_loss = float(tail_loss)
    confidence = float(confidence)
    if any(not math.isfinite(v) for v in (expected_loss, tail_loss, confidence)):
        raise ValueError("capital inputs must be finite")
    if expected_loss < 0.0 or tail_loss < 0.0 or not (0.0 < confidence < 1.0):
        raise ValueError("losses must be non-negative and confidence inside (0, 1)")
    if tail_measure not in {"VaR", "ExpectedShortfall"}:
        raise ValueError("tail_measure must be VaR or ExpectedShortfall")
    return {
        "expectedLoss": expected_loss,
        "tailLoss": tail_loss,
        "economicCapital": max(0.0, tail_loss - expected_loss),
        "confidence": confidence,
        "tailMeasure": tail_measure,
        "status": "simulation-derived economic-capital research; separate from accounting ECL",
    }


def reconcile_tail_capital(tail_contributions: Iterable[float], expected_loss_contributions: Iterable[float]) -> dict:
    tail = [float(v) for v in tail_contributions]
    expected = [float(v) for v in expected_loss_contributions]
    if len(tail) != len(expected) or not tail:
        raise ValueError("tail and expected-loss contribution vectors must be equal and non-empty")
    if any(not math.isfinite(v) for v in tail + expected):
        raise ValueError("contributions must be finite")
    contributions = [t - e for t, e in zip(tail, expected)]
    return {
        "tailLoss": sum(tail),
        "expectedLoss": sum(expected),
        "economicCapital": sum(contributions),
        "economicCapitalContributions": contributions,
        "reconciles": abs(sum(contributions) - (sum(tail) - sum(expected))) < 1e-10,
    }
