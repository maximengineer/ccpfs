"""
Cost Function for Follow-Up Scheduling
---------------------------------------
Converts survival curves S(t) into expected costs for scheduling decisions.

The core trade-off:
  - Schedule too late  -> patient deteriorates before follow-up (cost: C_EVENT)
  - Schedule too early -> consumes a scarce slot (opportunity cost for others)

Expected cost of scheduling patient i on day d:
  Cost_i(d) = C_EVENT * R_i(d) + C_VISIT
  where R_i(d) = 1 - S_i(d) = P(adverse event before day d)
"""

import numpy as np

from config import C_EVENT, C_VISIT, HORIZON_DAYS


def expected_cost(
    survival_curve: np.ndarray,
    day: int,
    c_event: float = C_EVENT,
    c_visit: float = C_VISIT,
) -> float:
    """Expected cost of scheduling follow-up on a given day.

    Parameters
    ----------
    survival_curve : np.ndarray
        S(t) for t = 0, 1, ..., horizon. S(0) = 1.0 by definition.
    day : int
        Candidate follow-up day (1-indexed: day 1 = tomorrow).
    c_event : float
        Cost of an adverse event (readmission).
    c_visit : float
        Cost of a follow-up visit.

    Returns
    -------
    float
        Expected cost.
    """
    risk = 1.0 - survival_curve[day]
    return c_event * risk + c_visit


def expected_cost_curve(
    survival_curve: np.ndarray,
    c_event: float = C_EVENT,
    c_visit: float = C_VISIT,
    horizon: int = HORIZON_DAYS,
) -> np.ndarray:
    """Expected cost for every candidate day 1..horizon.

    Returns
    -------
    np.ndarray
        Array of length horizon. Index 0 = day 1, index horizon-1 = day horizon.
    """
    days = np.arange(1, horizon + 1)
    risks = 1.0 - survival_curve[days]
    return c_event * risks + c_visit


def unconstrained_optimal_day(
    survival_curve: np.ndarray,
    c_event: float = C_EVENT,
    c_visit: float = C_VISIT,
    horizon: int = HORIZON_DAYS,
) -> int:
    """Optimal follow-up day ignoring capacity constraints.

    Without capacity constraints, this is always day 1 for any patient with
    non-zero risk - which is why capacity constraints are essential.

    Returns
    -------
    int
        Optimal day (1-indexed).
    """
    costs = expected_cost_curve(survival_curve, c_event, c_visit, horizon)
    return int(np.argmin(costs)) + 1


def marginal_benefit(
    survival_curve: np.ndarray,
    from_day: int,
    to_day: int,
    c_event: float = C_EVENT,
) -> float:
    """Marginal benefit of moving a patient from `from_day` to `to_day`.

    Positive value means moving to `to_day` saves expected cost.

    Parameters
    ----------
    survival_curve : np.ndarray
        S(t) for t = 0..horizon.
    from_day : int
        Current assigned day.
    to_day : int
        Candidate reassignment day.
    c_event : float
        Cost of adverse event.

    Returns
    -------
    float
        Cost reduction (positive = beneficial to move).
    """
    risk_from = 1.0 - survival_curve[from_day]
    risk_to = 1.0 - survival_curve[to_day]
    return c_event * (risk_from - risk_to)
