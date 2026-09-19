from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class EclPolicy:
    relative_pd_multiplier: float = 2.0
    absolute_pd_increase: float = 0.05
    stage2_dpd: int = 30
    stage3_dpd: int = 90
    cure_probation_months: int = 3

    def __post_init__(self) -> None:
        if not math.isfinite(self.relative_pd_multiplier) or self.relative_pd_multiplier <= 1.0:
            raise ValueError("relative_pd_multiplier must be finite and greater than 1")
        if not math.isfinite(self.absolute_pd_increase) or not (0.0 <= self.absolute_pd_increase < 1.0):
            raise ValueError("absolute_pd_increase must lie in [0, 1)")
        if not isinstance(self.stage2_dpd, int) or not isinstance(self.stage3_dpd, int):
            raise ValueError("DPD backstops must be integers")
        if self.stage2_dpd < 0 or self.stage3_dpd <= self.stage2_dpd:
            raise ValueError("stage3_dpd must be greater than stage2_dpd >= 0")
        if not isinstance(self.cure_probation_months, int) or self.cure_probation_months < 0:
            raise ValueError("cure_probation_months must be a non-negative integer")


def cumulative_to_marginal(cumulative_pd: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(cumulative_pd), dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("cumulative PD must be a finite non-empty vector")
    if np.any((values < 0.0) | (values > 1.0)) or np.any(np.diff(values) < -1e-12):
        raise ValueError("cumulative PD must be bounded and non-decreasing")
    return np.diff(np.concatenate(([0.0], values)))


def determine_stage(
    origination_pd: float,
    current_pd: float,
    dpd: int,
    is_default: bool,
    *,
    previous_stage: int | None = None,
    months_since_cure: int | None = None,
    policy: EclPolicy = EclPolicy(),
) -> dict:
    if not all(math.isfinite(value) for value in (origination_pd, current_pd)):
        raise ValueError("PDs must be finite")
    if origination_pd <= 0 or current_pd <= 0 or origination_pd >= 1 or current_pd >= 1:
        raise ValueError("PDs must lie strictly inside (0, 1)")
    if not isinstance(dpd, int) or dpd < 0:
        raise ValueError("dpd must be a non-negative integer")
    if previous_stage is not None and previous_stage not in (1, 2, 3):
        raise ValueError("previous_stage must be 1, 2 or 3")
    if months_since_cure is not None and (
        not isinstance(months_since_cure, int) or months_since_cure < 0
    ):
        raise ValueError("months_since_cure must be a non-negative integer")
    reasons: list[str] = []
    if is_default or dpd >= policy.stage3_dpd:
        return {"stage": 3, "reasons": ["credit-impaired/default backstop"]}
    if previous_stage in (2, 3) and months_since_cure is not None and months_since_cure < policy.cure_probation_months:
        return {"stage": 2, "reasons": ["cure probation"]}
    relative = current_pd / origination_pd
    if dpd >= policy.stage2_dpd:
        reasons.append("days-past-due backstop")
    if relative >= policy.relative_pd_multiplier:
        reasons.append("relative PD increase")
    if current_pd - origination_pd >= policy.absolute_pd_increase:
        reasons.append("absolute PD increase")
    return {"stage": 2 if reasons else 1, "reasons": reasons or ["no SICR trigger"]}


def _scenario_ecl(cumulative_pd: list[float], lgd: list[float], ead: list[float], stage: int, annual_eir: float, interval_months: int) -> float:
    marginal = cumulative_to_marginal(cumulative_pd)
    if not (len(marginal) == len(lgd) == len(ead)):
        raise ValueError("PD, LGD and EAD curves must align")
    if any(not math.isfinite(v) or v < 0 for v in lgd + ead):
        raise ValueError("LGD/EAD curves must be finite and non-negative")
    if annual_eir <= -1.0 or not math.isfinite(annual_eir):
        raise ValueError("annual EIR must be finite and greater than -1")
    if interval_months <= 0:
        raise ValueError("interval_months must be positive")
    limit = min(len(marginal), max(1, math.ceil(12 / interval_months))) if stage == 1 else len(marginal)
    total = 0.0
    for i in range(limit):
        years = ((i + 1) * interval_months) / 12.0
        discount = (1.0 + annual_eir) ** (-years)
        total += float(marginal[i]) * float(lgd[i]) * float(ead[i]) * discount
    return total


def scenario_weighted_ecl(
    scenarios: Mapping[str, Mapping[str, object]],
    *,
    stage: int,
    annual_eir: float = 0.0,
    interval_months: int = 12,
    policy_version: str = "ifrs9-research-v1",
) -> dict:
    if stage not in (1, 2, 3) or not scenarios:
        raise ValueError("stage must be 1/2/3 and scenarios cannot be empty")
    weights = [float(item["weight"]) for item in scenarios.values()]
    if any((not math.isfinite(w) or w < 0.0) for w in weights) or abs(sum(weights) - 1.0) > 1e-10:
        raise ValueError("scenario weights must be non-negative and sum to 1")
    contributions: dict[str, float] = {}
    scenario_ecl: dict[str, float] = {}
    for name, item in scenarios.items():
        weight = float(item["weight"])
        ecl = _scenario_ecl(list(item["cumulativePd"]), list(item["lgd"]), list(item["ead"]), stage, annual_eir, interval_months)
        scenario_ecl[name] = ecl
        contributions[name] = weight * ecl
    total = float(sum(contributions.values()))
    return {
        "stage": stage,
        "ecl": total,
        "scenarioEcl": scenario_ecl,
        "scenarioContributions": contributions,
        "policyVersion": policy_version,
        "discountConvention": "effective-interest-rate",
        "status": "IFRS 9-style research output; not production accounting advice or compliance approval",
    }
