"""
Greedy Capacity-Constrained Scheduler
--------------------------------------
Fast heuristic alternative to the ILP. Runs in O(N * horizon) vs the ILP's
combinatorial solve. Useful for:
  - Large cohorts where ILP is slow
  - Real-time scheduling (single patient at a time)
  - Sanity-checking ILP results

Algorithm:
  1. Compute urgency for each patient (steepest near-term hazard)
  2. Sort patients by urgency (descending)
  3. Assign each patient to their cost-minimising available day
"""

from typing import Optional

import numpy as np

from config import C_EVENT, C_VISIT, DEFAULT_DAILY_CAPACITY, HORIZON_DAYS


def schedule_greedy(
    survival_curves: np.ndarray,
    capacity_per_day: Optional[np.ndarray] = None,
    c_event: float = C_EVENT,
    c_visit: float = C_VISIT,
    horizon: int = HORIZON_DAYS,
) -> dict:
    """Greedy marginal-benefit capacity-constrained scheduling.

    Parameters
    ----------
    survival_curves : np.ndarray
        Shape (N, horizon+1). S_i(t) for t = 0..horizon.
    capacity_per_day : np.ndarray, optional
        Shape (horizon,). Slots available per day. Defaults to
        DEFAULT_DAILY_CAPACITY for all days.
    c_event : float
        Cost of adverse event.
    horizon : int
        Scheduling horizon.

    Returns
    -------
    dict with keys:
        "assignments" : dict[int, int]
            Patient index -> assigned day (1-indexed).
        "status" : str
            "Feasible" or "Infeasible" (if total capacity < N).
        "total_expected_cost" : float
    """
    n_patients = survival_curves.shape[0]

    if capacity_per_day is None:
        capacity_per_day = np.full(horizon, DEFAULT_DAILY_CAPACITY)

    remaining_capacity = capacity_per_day.copy().astype(float)

    # Risk at each day for each patient: R_i(d) = 1 - S_i(d)
    risks = 1.0 - survival_curves[:, 1 : horizon + 1]  # shape (N, horizon)

    # Urgency: maximum daily hazard increment in the first 14 days
    # h_i(d) = S_i(d-1) - S_i(d)  (probability of event on day d)
    hazards = survival_curves[:, :-1] - survival_curves[:, 1:]  # (N, horizon)
    early_window = min(14, horizon)
    urgencies = np.max(hazards[:, :early_window], axis=1)  # (N,)

    # Sort patients by urgency (most urgent first)
    patient_order = np.argsort(-urgencies)

    assignments = {}
    total_risk_cost = 0.0

    for i in patient_order:
        best_day = None
        best_cost = float("inf")

        for d in range(horizon):
            if remaining_capacity[d] > 0:
                cost = c_event * risks[i, d]
                if cost < best_cost:
                    best_cost = cost
                    best_day = d

        if best_day is not None:
            assignments[int(i)] = best_day + 1  # 1-indexed
            remaining_capacity[best_day] -= 1
            total_risk_cost += best_cost

    feasible = len(assignments) == n_patients
    status = "Feasible" if feasible else "Infeasible"

    total_cost = total_risk_cost + len(assignments) * c_visit

    return {
        "assignments": assignments,
        "status": status,
        "total_expected_cost": total_cost,
    }
