from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Iterable


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    exposure: float
    expected_return: float
    expected_loss: float
    segment: str
    eligible: bool = True

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.segment:
            raise ValueError("candidate_id and segment are required")
        values = (self.exposure, self.expected_return, self.expected_loss)
        if any(not math.isfinite(float(v)) for v in values):
            raise ValueError("candidate economics must be finite")
        if self.exposure <= 0.0 or self.expected_loss < 0.0:
            raise ValueError("exposure must be positive and expected loss non-negative")


@dataclass(frozen=True)
class ChallengerMetrics:
    model_id: str
    comparability_key: str
    auc: float
    brier: float
    calibration_slope: float
    calibration_intercept: float
    psi: float
    log_loss: float


def compare_challengers(models: Iterable[ChallengerMetrics], *, incumbent_id: str) -> dict:
    models = list(models)
    if not models:
        raise ValueError("at least one model is required")
    by_id = {m.model_id: m for m in models}
    if incumbent_id not in by_id:
        raise ValueError("incumbent model not found")
    incumbent = by_id[incumbent_id]
    output: dict[str, dict] = {}
    eligible: list[ChallengerMetrics] = []
    for model in models:
        failed: list[str] = []
        if model.comparability_key != incumbent.comparability_key:
            failed.append("population/target/horizon comparability")
        if not (0.80 <= model.calibration_slope <= 1.20) or abs(model.calibration_intercept) > 0.10:
            failed.append("calibration gate")
        if model.psi > 0.25:
            failed.append("stability/PSI gate")
        if not all(math.isfinite(float(v)) for v in (model.auc, model.brier, model.calibration_slope, model.calibration_intercept, model.psi, model.log_loss)):
            failed.append("finite-metric gate")
        output[model.model_id] = {
            "auc": model.auc,
            "brier": model.brier,
            "calibrationSlope": model.calibration_slope,
            "calibrationIntercept": model.calibration_intercept,
            "psi": model.psi,
            "logLoss": model.log_loss,
            "failedGates": failed,
        }
        if not failed:
            eligible.append(model)

    # Governance is calibration/stability-first; discrimination only ranks models that clear gates.
    recommended = max(eligible, key=lambda m: (m.auc, -m.brier, -m.log_loss)).model_id if eligible else incumbent_id
    if incumbent_id in by_id and output[recommended]["auc"] - incumbent.auc < 0.01:
        recommended = incumbent_id
    return {
        "incumbentModelId": incumbent_id,
        "recommendedModelId": recommended,
        "models": output,
        "promotionRule": "compatible cohort + calibration/stability gates first; AUC ranks only gated models",
    }


def _evaluate_subset(selected: list[Candidate], budget: float, max_expected_loss: float | None, max_segment_share: float | None) -> tuple[bool, dict]:
    exposure = sum(c.exposure for c in selected)
    loss = sum(c.expected_loss for c in selected)
    if exposure > budget + 1e-12:
        return False, {}
    if max_expected_loss is not None and loss > max_expected_loss + 1e-12:
        return False, {}
    segment_exposure: dict[str, float] = {}
    for candidate in selected:
        segment_exposure[candidate.segment] = segment_exposure.get(candidate.segment, 0.0) + candidate.exposure
    if max_segment_share is not None and exposure > 0.0:
        if any(value / exposure > max_segment_share + 1e-12 for value in segment_exposure.values()):
            return False, {}
    return True, {"exposure": exposure, "expectedLoss": loss, "segmentExposure": segment_exposure}


def optimize_exact(
    candidates: Iterable[Candidate],
    *,
    budget: float,
    max_expected_loss: float | None = None,
    max_segment_share: float | None = None,
    min_approval_count: int = 0,
) -> dict:
    candidates = list(candidates)
    if not math.isfinite(float(budget)) or budget < 0.0:
        raise ValueError("budget must be finite and non-negative")
    if max_expected_loss is not None and max_expected_loss < 0.0:
        raise ValueError("max_expected_loss cannot be negative")
    if max_segment_share is not None and not (0.0 < max_segment_share <= 1.0):
        raise ValueError("max_segment_share must lie in (0, 1]")
    if min_approval_count < 0:
        raise ValueError("min_approval_count cannot be negative")
    eligible = [c for c in candidates if c.eligible]
    if len(eligible) > 24:
        raise ValueError("exact optimizer is intentionally bounded to 24 eligible candidates; use a governed solver for larger coupled problems")

    best: tuple[float, float, tuple[str, ...], dict] | None = None
    for mask in range(1 << len(eligible)):
        selected = [eligible[i] for i in range(len(eligible)) if mask & (1 << i)]
        if len(selected) < min_approval_count:
            continue
        feasible, metrics = _evaluate_subset(selected, budget, max_expected_loss, max_segment_share)
        if not feasible:
            continue
        expected_return = sum(c.expected_return for c in selected)
        ids = tuple(sorted(c.candidate_id for c in selected))
        candidate_score = (expected_return, metrics["exposure"], tuple(reversed(ids)))
        if best is None or candidate_score > (best[0], best[1], tuple(reversed(best[2]))):
            best = (expected_return, metrics["exposure"], ids, metrics)

    if best is None:
        return {
            "status": "infeasible",
            "selectedIds": [],
            "expectedReturn": 0.0,
            "expectedLoss": 0.0,
            "exposure": 0.0,
            "reason": "no eligible portfolio satisfies all hard constraints",
        }
    expected_return, exposure, ids, metrics = best
    return {
        "status": "optimal",
        "selectedIds": list(ids),
        "expectedReturn": expected_return,
        "expectedLoss": metrics["expectedLoss"],
        "exposure": exposure,
        "segmentExposure": metrics["segmentExposure"],
        "bindingConstraints": {
            "budget": abs(exposure - budget) < 1e-10,
            "expectedLoss": max_expected_loss is not None and abs(metrics["expectedLoss"] - max_expected_loss) < 1e-10,
        },
        "method": "bounded exhaustive 0/1 search",
    }


def sorted_equal_exposure_frontier(candidates: Iterable[Candidate], *, budget: float) -> dict:
    candidates = [c for c in candidates if c.eligible]
    if not candidates:
        return {"status": "optimal", "selectedIds": [], "expectedReturn": 0.0, "exposure": 0.0, "method": "sorted-prefix"}
    exposures = {round(c.exposure, 12) for c in candidates}
    if len(exposures) != 1:
        raise ValueError("sorted equal-exposure frontier requires identical candidate exposures")
    exposure = candidates[0].exposure
    if budget < 0 or not math.isfinite(float(budget)):
        raise ValueError("budget must be finite and non-negative")
    count = min(len(candidates), int(math.floor(budget / exposure + 1e-12)))
    ranked = sorted(candidates, key=lambda c: (-c.expected_return, c.expected_loss, c.candidate_id))
    selected = ranked[:count]
    return {
        "status": "optimal",
        "selectedIds": [c.candidate_id for c in selected],
        "expectedReturn": sum(c.expected_return for c in selected),
        "expectedLoss": sum(c.expected_loss for c in selected),
        "exposure": sum(c.exposure for c in selected),
        "method": "O(n log n) sorted-prefix for equal exposures",
    }
