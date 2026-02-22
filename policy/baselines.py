"""
Baseline Scheduling Policies
-----------------------------
Reference policies for comparison against the CCPFS optimised scheduler.

Each baseline takes the same inputs and returns the same output format
as the ILP/greedy schedulers: dict with "assignments", "status", and
"total_expected_cost".
"""

from typing import Optional

import numpy as np

from config import C_EVENT, C_VISIT, HORIZON_DAYS


def _compute_total_cost(
    assignments: dict[int, int],
    survival_curves: np.ndarray,
    c_event: float,
    c_visit: float,
) -> float:
    """Total expected cost given assignments."""
    total = 0.0
    for i, d in assignments.items():
        total += c_event * (1.0 - survival_curves[i, d]) + c_visit
    return total


def uniform_policy(
    survival_curves: np.ndarray,
    day: int = 14,
    c_event: float = C_EVENT,
    c_visit: float = C_VISIT,
) -> dict:
    """Everyone gets the same fixed follow-up day.

    Parameters
    ----------
    survival_curves : np.ndarray
        Shape (N, horizon+1).
    day : int
        Fixed follow-up day for all patients.
    """
    n = survival_curves.shape[0]
    assignments = {i: day for i in range(n)}
    return {
        "assignments": assignments,
        "status": "Feasible",
        "total_expected_cost": _compute_total_cost(
            assignments, survival_curves, c_event, c_visit
        ),
    }


def risk_bucket_policy(
    survival_curves: np.ndarray,
    high_threshold: float = 0.30,
    mid_threshold: float = 0.15,
    high_day: int = 7,
    mid_day: int = 14,
    low_day: int = 30,
    c_event: float = C_EVENT,
    c_visit: float = C_VISIT,
    horizon: int = HORIZON_DAYS,
) -> dict:
    """Stratify patients into risk buckets with fixed days.

    Buckets based on 30-day cumulative risk R(horizon) = 1 - S(horizon):
      - High risk (R >= high_threshold)   -> high_day
      - Medium risk (R >= mid_threshold)  -> mid_day
      - Low risk (R < mid_threshold)      -> low_day
    """
    n = survival_curves.shape[0]
    risk_30 = 1.0 - survival_curves[:, horizon]

    assignments = {}
    for i in range(n):
        if risk_30[i] >= high_threshold:
            assignments[i] = high_day
        elif risk_30[i] >= mid_threshold:
            assignments[i] = mid_day
        else:
            assignments[i] = low_day

    return {
        "assignments": assignments,
        "status": "Feasible",
        "total_expected_cost": _compute_total_cost(
            assignments, survival_curves, c_event, c_visit
        ),
    }


def guideline_policy(
    survival_curves: np.ndarray,
    is_heart_failure: Optional[np.ndarray] = None,
    hf_day: int = 14,
    default_day: int = 28,
    c_event: float = C_EVENT,
    c_visit: float = C_VISIT,
) -> dict:
    """Condition-specific fixed intervals based on clinical guidelines.

    ACC 2024 HF guidelines: 2-4 weeks for HFrEF. We use 14 days (midpoint).
    All other conditions: 28 days (standard 4-week follow-up).

    Parameters
    ----------
    is_heart_failure : np.ndarray, optional
        Boolean array of shape (N,). If None, all patients get default_day.
    """
    n = survival_curves.shape[0]

    if is_heart_failure is None:
        is_heart_failure = np.zeros(n, dtype=bool)

    assignments = {}
    for i in range(n):
        assignments[i] = hf_day if is_heart_failure[i] else default_day

    return {
        "assignments": assignments,
        "status": "Feasible",
        "total_expected_cost": _compute_total_cost(
            assignments, survival_curves, c_event, c_visit
        ),
    }


def unconstrained_optimal_policy(
    survival_curves: np.ndarray,
    c_event: float = C_EVENT,
    c_visit: float = C_VISIT,
    horizon: int = HORIZON_DAYS,
) -> dict:
    """Each patient gets their individually optimal day (no capacity limit).

    This is the theoretical lower bound on cost. In practice unrealisable
    because everyone clusters on day 1 (earliest possible).
    """
    n = survival_curves.shape[0]
    risks = 1.0 - survival_curves[:, 1 : horizon + 1]
    costs = c_event * risks  # shape (N, horizon)

    assignments = {}
    for i in range(n):
        best_day = int(np.argmin(costs[i])) + 1  # 1-indexed
        assignments[i] = best_day

    return {
        "assignments": assignments,
        "status": "Feasible",
        "total_expected_cost": _compute_total_cost(
            assignments, survival_curves, c_event, c_visit
        ),
    }
