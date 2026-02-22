"""
Specialty-Pool Capacity-Constrained Scheduler
----------------------------------------------
Extends the base ILP/greedy schedulers with per-specialty daily capacity.

Each patient belongs to one specialty pool (cardiology, neurology, surgery,
general_medicine). The scheduler ensures that:
  - Each patient is assigned to exactly one follow-up day
  - The number of patients from pool k scheduled on day d does not exceed C_k(d)
  - Total expected cost (harm + visit) is minimised

Two implementations:
  - ILP (exact): Uses PuLP/CBC for guaranteed optimality
  - Greedy (heuristic): Fast O(N×H) per-specialty assignment
"""

from typing import Optional

import numpy as np
from pulp import (
    LpBinary,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    lpSum,
    PULP_CBC_CMD,
)

from config import (
    C_EVENT,
    C_VISIT,
    DEFAULT_SPECIALTY_CAPACITY,
    HORIZON_DAYS,
    N_SPECIALTIES,
    SPECIALTY_NAMES,
)


def schedule_ilp_specialty(
    survival_curves: np.ndarray,
    specialty_pools: np.ndarray,
    capacity_per_specialty_day: Optional[dict] = None,
    c_event: float = C_EVENT,
    c_visit: float = C_VISIT,
    horizon: int = HORIZON_DAYS,
    time_limit: int = 300,
    verbose: bool = False,
) -> dict:
    """Solve specialty-constrained scheduling via ILP.

    Parameters
    ----------
    survival_curves : np.ndarray, shape (N, horizon+1)
        Patient survival curves.
    specialty_pools : np.ndarray of int, shape (N,)
        Pool index for each patient (0=cardiology, 1=neurology, etc.).
    capacity_per_specialty_day : dict, optional
        Maps pool_index -> daily capacity (int). Defaults to
        DEFAULT_SPECIALTY_CAPACITY.
    c_event, c_visit, horizon, time_limit, verbose : as in ilp_scheduler.

    Returns
    -------
    dict with keys:
        "assignments" : dict[int, int]  — patient index -> day (1-indexed)
        "status" : str
        "total_expected_cost" : float
        "utilisation" : dict  — per-specialty utilisation stats
    """
    n_patients = survival_curves.shape[0]
    days = range(1, horizon + 1)
    patients = range(n_patients)

    if capacity_per_specialty_day is None:
        capacity_per_specialty_day = DEFAULT_SPECIALTY_CAPACITY

    # Group patients by specialty pool
    pool_members = {}
    for k in range(N_SPECIALTIES):
        pool_members[k] = np.where(specialty_pools == k)[0]

    # Pre-compute risk costs
    risk_costs = c_event * (1.0 - survival_curves[:, 1 : horizon + 1])

    prob = LpProblem("CCPFS_Specialty", LpMinimize)

    # Decision variables
    x = {}
    for i in patients:
        for d in days:
            x[i, d] = LpVariable(f"x_{i}_{d}", cat=LpBinary)

    # Objective: minimise total expected harm
    prob += lpSum(
        x[i, d] * risk_costs[i, d - 1]
        for i in patients
        for d in days
    )

    # Constraint 1: each patient assigned to exactly one day
    for i in patients:
        prob += lpSum(x[i, d] for d in days) == 1

    # Constraint 2: per-specialty daily capacity
    for k in range(N_SPECIALTIES):
        cap = capacity_per_specialty_day.get(k, 0)
        members = pool_members[k]
        if len(members) == 0:
            continue
        for d in days:
            prob += lpSum(x[int(i), d] for i in members) <= cap

    # Solve
    solver = PULP_CBC_CMD(timeLimit=time_limit, msg=verbose)
    prob.solve(solver)

    status = LpStatus[prob.status]

    if status != "Optimal":
        return {
            "assignments": {},
            "status": status,
            "total_expected_cost": float("inf"),
            "utilisation": {},
        }

    # Extract assignments
    assignments = {}
    total_risk_cost = 0.0
    for i in patients:
        for d in days:
            if x[i, d].varValue is not None and x[i, d].varValue > 0.5:
                assignments[i] = d
                total_risk_cost += risk_costs[i, d - 1]
                break

    total_cost = total_risk_cost + n_patients * c_visit

    # Compute per-specialty utilisation
    utilisation = _compute_utilisation(
        assignments, specialty_pools, capacity_per_specialty_day, horizon
    )

    return {
        "assignments": assignments,
        "status": status,
        "total_expected_cost": total_cost,
        "utilisation": utilisation,
    }


def schedule_greedy_specialty(
    survival_curves: np.ndarray,
    specialty_pools: np.ndarray,
    capacity_per_specialty_day: Optional[dict] = None,
    c_event: float = C_EVENT,
    horizon: int = HORIZON_DAYS,
) -> dict:
    """Greedy heuristic with per-specialty capacity constraints.

    Within each specialty pool, patients are sorted by urgency (steepest
    near-term hazard) and assigned to their cost-minimising available day.

    Parameters & returns match schedule_ilp_specialty.
    """
    n_patients = survival_curves.shape[0]

    if capacity_per_specialty_day is None:
        capacity_per_specialty_day = DEFAULT_SPECIALTY_CAPACITY

    # Track remaining capacity per (specialty, day)
    remaining = {}
    for k in range(N_SPECIALTIES):
        cap = capacity_per_specialty_day.get(k, 0)
        remaining[k] = np.full(horizon, cap, dtype=float)

    risks = 1.0 - survival_curves[:, 1 : horizon + 1]
    hazards = survival_curves[:, :-1] - survival_curves[:, 1:]
    early_window = min(14, horizon)
    urgencies = np.max(hazards[:, :early_window], axis=1)

    # Process each specialty pool independently
    assignments = {}
    total_risk_cost = 0.0

    for k in range(N_SPECIALTIES):
        members = np.where(specialty_pools == k)[0]
        if len(members) == 0:
            continue

        # Sort by urgency within pool
        order = members[np.argsort(-urgencies[members])]

        for i in order:
            best_day = None
            best_cost = float("inf")

            for d in range(horizon):
                if remaining[k][d] > 0:
                    cost = c_event * risks[i, d]
                    if cost < best_cost:
                        best_cost = cost
                        best_day = d

            if best_day is not None:
                assignments[int(i)] = best_day + 1
                remaining[k][best_day] -= 1
                total_risk_cost += best_cost

    feasible = len(assignments) == n_patients
    status = "Feasible" if feasible else "Infeasible"

    utilisation = _compute_utilisation(
        assignments, specialty_pools, capacity_per_specialty_day, horizon
    )

    total_cost = total_risk_cost + len(assignments) * C_VISIT

    return {
        "assignments": assignments,
        "status": status,
        "total_expected_cost": total_cost,
        "utilisation": utilisation,
    }


def _compute_utilisation(
    assignments: dict,
    specialty_pools: np.ndarray,
    capacity_per_specialty_day: dict,
    horizon: int,
) -> dict:
    """Compute per-specialty utilisation statistics."""
    util = {}
    for k in range(N_SPECIALTIES):
        cap = capacity_per_specialty_day.get(k, 0)
        if cap == 0:
            continue

        # Count assignments per day for this pool
        daily_counts = np.zeros(horizon)
        for patient_idx, day in assignments.items():
            if specialty_pools[patient_idx] == k:
                daily_counts[day - 1] += 1

        total_capacity = cap * horizon
        total_assigned = daily_counts.sum()

        util[SPECIALTY_NAMES[k]] = {
            "daily_capacity": cap,
            "total_capacity": int(total_capacity),
            "total_assigned": int(total_assigned),
            "utilisation_pct": 100 * total_assigned / total_capacity if total_capacity > 0 else 0,
            "peak_day_count": int(daily_counts.max()),
            "mean_day_count": float(daily_counts.mean()),
        }

    return util
