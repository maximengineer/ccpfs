"""POST /api/schedule - live re-scheduling with greedy heuristic."""

import time

import numpy as np
from fastapi import APIRouter, HTTPException

from api.dependencies import store
from api.schemas import ScheduleMetrics, ScheduleRequest, ScheduleResponse
from config import C_EVENT, C_VISIT, SPECIALTY_NAMES
from evaluation.metrics import event_before_followup_rate
from policy.specialty_scheduler import schedule_greedy_specialty

router = APIRouter()

SPECIALTY_INDEX = {name: idx for idx, name in enumerate(SPECIALTY_NAMES)}
VALID_SPECIALTIES = set(SPECIALTY_NAMES)


def _scale_capacity(
    base_capacity: dict[str, int],
    specialty_pools: np.ndarray,
    horizon: int,
) -> dict[int, int]:
    """Scale each specialty's capacity so its 30-day slots match its patient count.

    Each specialty is scaled independently based on how many patients it has,
    not the total cohort. This prevents overflow when one specialty dominates.

    Returns dict keyed by specialty index (matching config convention).
    """
    result = {}
    for name, base_val in base_capacity.items():
        idx = SPECIALTY_INDEX[name]
        n_in_pool = int(np.sum(specialty_pools == idx))
        if n_in_pool == 0:
            result[idx] = base_val
            continue
        needed_per_day = int(np.ceil(n_in_pool / horizon))
        # Scale so this specialty has enough slots, using base_val as the minimum
        scale = max(1.0, needed_per_day / base_val)
        result[idx] = int(np.ceil(base_val * scale))
    return result


DEFAULT_BASE_CAPACITY = {
    "cardiology": 15,
    "neurology": 10,
    "surgery": 15,
    "general_medicine": 25,
}


@router.post("/schedule", response_model=ScheduleResponse)
def run_schedule(req: ScheduleRequest):
    # Validate specialty names
    invalid = set(req.capacity.keys()) - VALID_SPECIALTIES
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown specialties: {invalid}. Valid: {sorted(VALID_SPECIALTIES)}",
        )

    # Merge with defaults so missing specialties keep their base capacity
    merged_capacity = {**DEFAULT_BASE_CAPACITY, **req.capacity}

    t0 = time.perf_counter()

    curves = store.curves
    n = curves.shape[0]
    horizon = curves.shape[1] - 1
    specialty_pools = store.cohort_test["specialty_pool"].to_numpy()

    scaled_cap = _scale_capacity(merged_capacity, specialty_pools, horizon)
    result = schedule_greedy_specialty(
        curves, specialty_pools, capacity_per_specialty_day=scaled_cap,
    )
    assignments = result["assignments"]

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Day histogram (1-indexed days)
    day_hist = [0] * horizon
    for day in assignments.values():
        if 1 <= day <= horizon:
            day_hist[day - 1] += 1

    # Per-specialty histograms
    by_specialty = {name: [0] * horizon for name in SPECIALTY_NAMES}
    for idx, day in assignments.items():
        sp = int(specialty_pools[idx])
        sp_name = SPECIALTY_NAMES[sp] if sp < len(SPECIALTY_NAMES) else "unknown"
        if 1 <= day <= horizon and sp_name in by_specialty:
            by_specialty[sp_name][day - 1] += 1

    # Metrics from scheduler's own total (uses config C_EVENT/C_VISIT)
    avg_cost = result["total_expected_cost"] / n if n > 0 else 0

    # EBF metrics
    ebf = event_before_followup_rate(assignments, store.t_test, store.e_test)

    # Compare to pre-computed uniform day 14
    uniform_total = store.pipeline_results["scheduling_results"]["uniform_d14"]["total_expected_cost"]
    uniform_cost = uniform_total / n
    vs_uniform = ((avg_cost - uniform_cost) / uniform_cost) * 100

    metrics = ScheduleMetrics(
        avg_cost=round(avg_cost, 0),
        catch_rate=round(ebf["catch_rate"] * 100, 1),
        ebf_rate=round(ebf["ebf_rate"] * 100, 1),
        vs_uniform_pct=round(vs_uniform, 1),
        elapsed_ms=round(elapsed_ms, 1),
    )

    return ScheduleResponse(
        day_histogram=day_hist,
        by_specialty=by_specialty,
        metrics=metrics,
    )
