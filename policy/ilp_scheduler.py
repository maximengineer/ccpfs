"""
ILP Capacity-Constrained Scheduler
-----------------------------------
Exact optimal assignment of patients to follow-up days via Integer Linear
Programming (PuLP + CBC solver).

Formulation:
  Minimise  sum_i sum_d  x_{i,d} * C_EVENT * (1 - S_i(d))
  s.t.      sum_d x_{i,d} = 1          for all patients i
            sum_i x_{i,d} <= C(d)      for all days d
            x_{i,d} in {0, 1}

This is a standard assignment problem. CBC handles 10K+ patients in seconds.
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

from config import C_EVENT, C_VISIT, DEFAULT_DAILY_CAPACITY, HORIZON_DAYS


def schedule_ilp(
    survival_curves: np.ndarray,
    capacity_per_day: Optional[np.ndarray] = None,
    c_event: float = C_EVENT,
    c_visit: float = C_VISIT,
    horizon: int = HORIZON_DAYS,
    time_limit: int = 120,
    verbose: bool = False,
) -> dict:
    """Solve the capacity-constrained scheduling problem exactly.

    Parameters
    ----------
    survival_curves : np.ndarray
        Shape (N, horizon+1). Row i is patient i's survival curve S_i(t)
        for t = 0, 1, ..., horizon. S_i(0) = 1.0.
    capacity_per_day : np.ndarray, optional
        Shape (horizon,). capacity_per_day[d-1] = slots on day d.
        If None, uses DEFAULT_DAILY_CAPACITY for all days.
    c_event : float
        Cost of adverse event.
    c_visit : float
        Cost of follow-up visit (constant across days, does not affect
        optimal assignment but included for total cost reporting).
    horizon : int
        Scheduling horizon in days.
    time_limit : int
        Solver time limit in seconds.
    verbose : bool
        If True, print solver output.

    Returns
    -------
    dict with keys:
        "assignments" : dict[int, int]
            Patient index -> assigned day (1-indexed).
        "status" : str
            Solver status ("Optimal", "Infeasible", etc.).
        "total_expected_cost" : float
            Objective value + visit costs.
    """
    n_patients = survival_curves.shape[0]
    days = range(1, horizon + 1)
    patients = range(n_patients)

    if capacity_per_day is None:
        capacity_per_day = np.full(horizon, DEFAULT_DAILY_CAPACITY)

    # Pre-compute risk costs: cost[i][d] = C_EVENT * (1 - S_i(d))
    risk_costs = c_event * (1.0 - survival_curves[:, 1 : horizon + 1])

    prob = LpProblem("CCPFS", LpMinimize)

    # Decision variables
    x = [
        [LpVariable(f"x_{i}_{d}", cat=LpBinary) for d in days]
        for i in patients
    ]

    # Objective: minimise total expected harm
    prob += lpSum(
        x[i][d - 1] * risk_costs[i, d - 1]
        for i in patients
        for d in days
    )

    # Constraint 1: each patient gets exactly one slot
    for i in patients:
        prob += lpSum(x[i][d - 1] for d in days) == 1

    # Constraint 2: respect daily capacity
    for d in days:
        prob += lpSum(x[i][d - 1] for i in patients) <= capacity_per_day[d - 1]

    # Solve
    solver = PULP_CBC_CMD(timeLimit=time_limit, msg=verbose)
    prob.solve(solver)

    status = LpStatus[prob.status]

    if status != "Optimal":
        return {
            "assignments": {},
            "status": status,
            "total_expected_cost": float("inf"),
        }

    # Extract assignments
    assignments = {}
    total_risk_cost = 0.0
    for i in patients:
        for d in days:
            if x[i][d - 1].varValue is not None and x[i][d - 1].varValue > 0.5:
                assignments[i] = d
                total_risk_cost += risk_costs[i, d - 1]
                break

    total_cost = total_risk_cost + n_patients * c_visit

    return {
        "assignments": assignments,
        "status": status,
        "total_expected_cost": total_cost,
    }
