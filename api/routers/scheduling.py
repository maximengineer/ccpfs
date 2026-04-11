"""POST /api/schedule - live scheduling simulation with greedy heuristic."""

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

DEFAULT_BASE_CAPACITY = {
    "cardiology": 15,
    "neurology": 10,
    "surgery": 15,
    "general_medicine": 25,
}

DEFAULT_PATIENTS_PER_DAY = {
    "cardiology": 10,
    "neurology": 7,
    "surgery": 5,
    "general_medicine": 20,
}


def _sample_patients(
    patients_per_day: dict[str, int],
    specialty_pools: np.ndarray,
    horizon: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Sample patient indices from the database based on daily discharge volume.

    For each specialty, randomly samples (patients_per_day * horizon) patients
    from the available pool to simulate realistic demand. Uses actual S(t)
    curves from the database for realistic risk profiles.

    Returns array of selected patient indices.
    """
    selected = []
    for name, daily_vol in patients_per_day.items():
        idx = SPECIALTY_INDEX[name]
        pool_indices = np.where(specialty_pools == idx)[0]
        n_needed = daily_vol * horizon
        if n_needed >= len(pool_indices):
            # Use all available patients from this specialty
            selected.append(pool_indices)
        else:
            chosen = rng.choice(pool_indices, size=n_needed, replace=False)
            selected.append(chosen)
    return np.concatenate(selected)


@router.post("/schedule", response_model=ScheduleResponse)
def run_schedule(req: ScheduleRequest):
    # Validate specialty names
    for field_name, field_val in [("capacity", req.capacity), ("patients_per_day", req.patients_per_day)]:
        invalid = set(field_val.keys()) - VALID_SPECIALTIES
        if invalid:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown specialties in {field_name}: {invalid}. Valid: {sorted(VALID_SPECIALTIES)}",
            )

    # Merge with defaults
    merged_capacity = {**DEFAULT_BASE_CAPACITY, **req.capacity}
    merged_demand = {**DEFAULT_PATIENTS_PER_DAY, **req.patients_per_day}

    t0 = time.perf_counter()

    all_curves = store.curves
    horizon = all_curves.shape[1] - 1
    all_pools = store.cohort_test["specialty_pool"].to_numpy()
    all_e = store.e_test
    all_t = store.t_test

    # Sample patients based on demand sliders
    rng = np.random.RandomState(42)  # fixed seed for reproducibility
    sample_idx = _sample_patients(merged_demand, all_pools, horizon, rng)

    # Extract sampled data
    curves = all_curves[sample_idx]
    specialty_pools = all_pools[sample_idx]
    e_test = all_e[sample_idx]
    t_test = all_t[sample_idx]
    n = len(sample_idx)

    # Convert capacity to index-keyed
    capacity = {SPECIALTY_INDEX[name]: val for name, val in merged_capacity.items()}

    result = schedule_greedy_specialty(
        curves, specialty_pools, capacity_per_specialty_day=capacity,
    )
    assignments = result["assignments"]

    elapsed_ms = (time.perf_counter() - t0) * 1000

    overflow_set = result.get("overflow_patients", set())

    # Day histogram (1-indexed days) - scheduled within capacity only
    day_hist = [0] * horizon
    by_specialty = {name: [0] * horizon for name in SPECIALTY_NAMES}
    for idx, day in assignments.items():
        if 1 <= day <= horizon and idx not in overflow_set:
            day_hist[day - 1] += 1
            sp = int(specialty_pools[idx])
            sp_name = SPECIALTY_NAMES[sp] if sp < len(SPECIALTY_NAMES) else "unknown"
            if sp_name in by_specialty:
                by_specialty[sp_name][day - 1] += 1

    # Metrics
    avg_cost = result["total_expected_cost"] / n if n > 0 else 0

    ebf = event_before_followup_rate(assignments, t_test, e_test)

    # Compare to fixed-interval baselines for the same sample
    uniform14_total = sum(C_EVENT * (1 - curves[i, min(14, horizon)]) + C_VISIT for i in range(n))
    uniform14_cost = uniform14_total / n if n > 0 else 0
    vs_uniform14 = ((avg_cost - uniform14_cost) / uniform14_cost) * 100 if uniform14_cost > 0 else 0

    uniform30_total = sum(C_EVENT * (1 - curves[i, horizon]) + C_VISIT for i in range(n))
    uniform30_cost = uniform30_total / n if n > 0 else 0
    vs_uniform30 = ((avg_cost - uniform30_cost) / uniform30_cost) * 100 if uniform30_cost > 0 else 0

    metrics = ScheduleMetrics(
        avg_cost=round(avg_cost, 0),
        catch_rate=round(ebf["catch_rate"] * 100, 1),
        ebf_rate=round(ebf["ebf_rate"] * 100, 1),
        uniform14_cost=round(uniform14_cost, 0),
        uniform30_cost=round(uniform30_cost, 0),
        vs_uniform14_pct=round(vs_uniform14, 1),
        vs_uniform30_pct=round(vs_uniform30, 1),
        elapsed_ms=round(elapsed_ms, 1),
    )

    n_overflow = len(overflow_set)
    return ScheduleResponse(
        day_histogram=day_hist,
        day_histogram_overflow=[0] * horizon,  # not shown in chart
        by_specialty=by_specialty,
        overflow_count=n_overflow,
        total_patients=n,
        scheduled_within_capacity=n - n_overflow,
        metrics=metrics,
    )
