"""
Min-Cost Assignment Solver
---------------------------
Solves the patient-to-day assignment as a min-cost problem using
scipy.optimize.linear_sum_assignment. Handles the full cohort in one
shot — no batching needed.

For specialty-constrained scheduling, each (specialty, day) combination
becomes a set of slots. Patients can only be assigned to slots in their
specialty pool.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import C_EVENT, C_VISIT, HORIZON_DAYS, N_SPECIALTIES


def schedule_mincost_specialty(
    survival_curves: np.ndarray,
    specialty_pools: np.ndarray,
    capacity_per_specialty_day: dict = None,
    c_event: float = C_EVENT,
    c_visit: float = C_VISIT,
    horizon: int = HORIZON_DAYS,
) -> dict:
    """Solve specialty-constrained scheduling via min-cost assignment.

    Reformulates as a rectangular cost matrix where columns represent
    individual slots within (specialty, day) buckets. Patients are
    assigned infinite cost for slots outside their specialty.

    Parameters
    ----------
    survival_curves : np.ndarray, shape (N, horizon+1)
    specialty_pools : np.ndarray, shape (N,) — pool index per patient
    capacity_per_specialty_day : dict — {pool_idx: slots_per_day}
    c_event : float
    horizon : int

    Returns
    -------
    dict with assignments, status, total_expected_cost
    """
    from config import DEFAULT_SPECIALTY_CAPACITY

    if capacity_per_specialty_day is None:
        capacity_per_specialty_day = DEFAULT_SPECIALTY_CAPACITY

    n_patients = len(survival_curves)
    risks = c_event * (1.0 - survival_curves[:, 1 : horizon + 1])  # (N, H)

    # Build slot structure: list of (specialty, day) for each slot column
    slots = []
    for k in range(N_SPECIALTIES):
        cap = capacity_per_specialty_day.get(k, 0)
        for d in range(horizon):
            for _ in range(cap):
                slots.append((k, d))

    n_slots = len(slots)

    if n_slots < n_patients:
        # Not enough total capacity — add overflow slots (high penalty)
        overflow_needed = n_patients - n_slots
        for d in range(horizon):
            for _ in range((overflow_needed // horizon) + 1):
                slots.append((-1, d))  # -1 = any specialty
        n_slots = len(slots)

    # Build cost matrix (N_patients x N_slots)
    INF_COST = 1e12
    cost_matrix = np.full((n_patients, n_slots), INF_COST, dtype=np.float64)

    for j, (slot_k, slot_d) in enumerate(slots):
        if slot_k == -1:
            # Overflow slot: any patient can use it, but with a penalty
            cost_matrix[:, j] = risks[:, slot_d] + c_event * 0.1
        else:
            # Only patients in this specialty can use this slot
            mask = specialty_pools == slot_k
            cost_matrix[mask, j] = risks[mask, slot_d]

    # For very large problems, use chunked approach
    if n_patients > 15000:
        return _solve_mincost_by_pool(
            survival_curves, specialty_pools, capacity_per_specialty_day,
            c_event, c_visit, horizon
        )

    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Extract assignments
    assignments = {}
    total_cost = 0.0
    overflow_count = 0

    for i, j in zip(row_ind, col_ind):
        slot_k, slot_d = slots[j]
        day = slot_d + 1  # 1-indexed
        assignments[int(i)] = day
        total_cost += c_event * (1.0 - survival_curves[i, day])
        if slot_k == -1:
            overflow_count += 1

    total_cost += len(assignments) * c_visit

    status = "Optimal" if overflow_count == 0 else f"Optimal (overflow={overflow_count})"

    return {
        "assignments": assignments,
        "status": status,
        "total_expected_cost": total_cost,
    }


def _solve_mincost_by_pool(
    survival_curves: np.ndarray,
    specialty_pools: np.ndarray,
    capacity_per_specialty_day: dict,
    c_event: float,
    c_visit: float,
    horizon: int,
) -> dict:
    """Solve per-specialty using linear_sum_assignment on each pool.

    For large cohorts where the full cost matrix would be too large,
    decompose by specialty pool. Each pool is solved independently
    (exact within pool, no cross-pool capacity sharing).
    """
    assignments = {}
    total_cost = 0.0
    overflow_count = 0

    for k in range(N_SPECIALTIES):
        members = np.where(specialty_pools == k)[0]
        if len(members) == 0:
            continue

        cap = capacity_per_specialty_day.get(k, 0)
        n_pool = len(members)
        n_pool_slots = cap * horizon

        risks = c_event * (1.0 - survival_curves[members, 1 : horizon + 1])

        # Build cost matrix: patients x slots
        cost_matrix = np.empty((n_pool, n_pool_slots), dtype=np.float64)
        for d in range(horizon):
            for s in range(cap):
                col = d * cap + s
                cost_matrix[:, col] = risks[:, d]

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for ri, ci in zip(row_ind, col_ind):
            patient_idx = int(members[ri])
            day = (ci // cap) + 1
            assignments[patient_idx] = day
            total_cost += c_event * (1.0 - survival_curves[patient_idx, day])

        # Handle overflow patients (not assigned due to insufficient capacity)
        if n_pool_slots < n_pool:
            assigned_set = set(row_ind)
            for ri in range(n_pool):
                if ri not in assigned_set:
                    patient_idx = int(members[ri])
                    risk_row = risks[ri]
                    if np.all(np.isnan(risk_row)):
                        risk_row = np.zeros_like(risk_row)
                    best_day = int(np.nanargmin(risk_row)) + 1
                    assignments[patient_idx] = best_day
                    total_cost += c_event * (1.0 - survival_curves[patient_idx, best_day])
                    overflow_count += 1

    total_cost += len(assignments) * c_visit

    status = "Optimal" if overflow_count == 0 else f"Optimal (overflow={overflow_count})"

    return {
        "assignments": assignments,
        "status": status,
        "total_expected_cost": total_cost,
    }


def schedule_mincost_global(
    survival_curves: np.ndarray,
    capacity_per_day: np.ndarray = None,
    c_event: float = C_EVENT,
    c_visit: float = C_VISIT,
    horizon: int = HORIZON_DAYS,
) -> dict:
    """Solve global (no specialty) scheduling via min-cost assignment.

    Each day has capacity_per_day[d] slots. Patients are assigned to
    minimize total expected risk cost.
    """
    n_patients = len(survival_curves)

    if capacity_per_day is None:
        cap_val = max(1, int(np.ceil(n_patients / horizon)))
        capacity_per_day = np.full(horizon, cap_val)

    risks = c_event * (1.0 - survival_curves[:, 1 : horizon + 1])

    # Build cost matrix
    total_slots = int(capacity_per_day.sum())
    if total_slots < n_patients:
        # Ensure enough slots
        extra = n_patients - total_slots
        capacity_per_day[-1] += extra
        total_slots = n_patients

    cost_matrix = np.empty((n_patients, total_slots), dtype=np.float64)
    col = 0
    for d in range(horizon):
        cap = int(capacity_per_day[d])
        for _ in range(cap):
            cost_matrix[:, col] = risks[:, d]
            col += 1

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Map column index back to day
    col_to_day = []
    for d in range(horizon):
        cap = int(capacity_per_day[d])
        col_to_day.extend([d + 1] * cap)

    assignments = {}
    total_cost = 0.0
    for i, j in zip(row_ind, col_ind):
        day = col_to_day[j]
        assignments[int(i)] = day
        total_cost += c_event * (1.0 - survival_curves[i, day])

    total_cost += len(assignments) * c_visit

    return {
        "assignments": assignments,
        "status": "Optimal",
        "total_expected_cost": total_cost,
    }
