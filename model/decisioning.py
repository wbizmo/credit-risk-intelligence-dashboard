from __future__ import annotations

from dataclasses import dataclass
import heapq
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
    if max_expected_loss is not None and (
        not math.isfinite(float(max_expected_loss)) or max_expected_loss < 0.0
    ):
        raise ValueError("max_expected_loss must be finite and non-negative")
    if max_segment_share is not None and not (0.0 < max_segment_share <= 1.0):
        raise ValueError("max_segment_share must lie in (0, 1]")
    if not isinstance(min_approval_count, int) or min_approval_count < 0:
        raise ValueError("min_approval_count must be a non-negative integer")
    ids = [candidate.candidate_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate_id values must be unique")

    eligible = [candidate for candidate in candidates if candidate.eligible]
    if len(eligible) > 24:
        raise ValueError("exact optimizer is intentionally bounded to 24 eligible candidates; use a governed solver for larger coupled problems")

    n = len(eligible)
    lex_rank = {
        candidate_id: rank
        for rank, candidate_id in enumerate(sorted(candidate.candidate_id for candidate in eligible))
    }
    selected = [False] * n
    segment_exposure: dict[str, float] = {}
    segment_heap: list[tuple[float, str]] = []
    selected_count = 0
    exposure = 0.0
    expected_loss = 0.0
    expected_return = 0.0
    tie_mask = 0
    previous_gray = 0
    best_score: tuple[float, float, int] | None = None
    best_selection = 0

    def max_segment_exposure() -> float:
        while segment_heap:
            negative_value, segment = segment_heap[0]
            current = segment_exposure.get(segment, 0.0)
            if -negative_value == current:
                return current
            heapq.heappop(segment_heap)
        return 0.0

    # Gray-code enumeration flips exactly one candidate per subset. This keeps
    # exposure/loss/return/segment state incremental rather than rebuilding an
    # O(n) selected list for every one of 2^n subsets.
    for step in range(1 << n):
        gray = step ^ (step >> 1)
        if step:
            changed = gray ^ previous_gray
            index = changed.bit_length() - 1
            candidate = eligible[index]
            adding = bool(gray & changed)
            sign = 1.0 if adding else -1.0

            selected[index] = adding
            selected_count += 1 if adding else -1
            exposure += sign * candidate.exposure
            expected_loss += sign * candidate.expected_loss
            expected_return += sign * candidate.expected_return
            tie_mask ^= 1 << lex_rank[candidate.candidate_id]

            updated_segment = segment_exposure.get(candidate.segment, 0.0) + sign * candidate.exposure
            if abs(updated_segment) <= 1e-12:
                updated_segment = 0.0
            segment_exposure[candidate.segment] = updated_segment
            heapq.heappush(segment_heap, (-updated_segment, candidate.segment))
            previous_gray = gray

        if selected_count < min_approval_count:
            continue
        if exposure > budget + 1e-12:
            continue
        if max_expected_loss is not None and expected_loss > max_expected_loss + 1e-12:
            continue
        if (
            max_segment_share is not None
            and exposure > 0.0
            and max_segment_exposure() > max_segment_share * exposure + 1e-12
        ):
            continue

        score = (expected_return, exposure, tie_mask)
        if best_score is None or score > best_score:
            best_score = score
            best_selection = gray

    if best_score is None:
        return {
            "status": "infeasible",
            "selectedIds": [],
            "expectedReturn": 0.0,
            "expectedLoss": 0.0,
            "exposure": 0.0,
            "reason": "no eligible portfolio satisfies all hard constraints",
        }

    selected_candidates = [
        eligible[index]
        for index in range(n)
        if best_selection & (1 << index)
    ]
    feasible, metrics = _evaluate_subset(
        selected_candidates,
        budget,
        max_expected_loss,
        max_segment_share,
    )
    if not feasible:
        raise RuntimeError("incremental optimizer state diverged from final constraint revalidation")

    final_return = sum(candidate.expected_return for candidate in selected_candidates)
    selected_ids = sorted(candidate.candidate_id for candidate in selected_candidates)
    final_exposure = metrics["exposure"]
    return {
        "status": "optimal",
        "selectedIds": selected_ids,
        "expectedReturn": final_return,
        "expectedLoss": metrics["expectedLoss"],
        "exposure": final_exposure,
        "segmentExposure": metrics["segmentExposure"],
        "bindingConstraints": {
            "budget": abs(final_exposure - budget) < 1e-10,
            "expectedLoss": max_expected_loss is not None and abs(metrics["expectedLoss"] - max_expected_loss) < 1e-10,
        },
        "method": "bounded Gray-code exhaustive 0/1 search; O(2^n log G) with segment constraints",
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
