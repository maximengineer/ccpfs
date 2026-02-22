"""
Evaluation Metrics for Scheduling Policies
-------------------------------------------
Measures how well a scheduling policy performs given ground-truth outcomes.
"""

import numpy as np


def event_before_followup_rate(
    assignments: dict[int, int],
    event_times: np.ndarray,
    event_indicators: np.ndarray,
) -> dict:
    """Primary metric: what fraction of events occurred before follow-up?

    An event_before_followup means the scheduling policy was "too late" -
    the patient deteriorated before they were seen.

    Parameters
    ----------
    assignments : dict[int, int]
        Patient index -> assigned follow-up day (1-indexed).
    event_times : np.ndarray
        Shape (N,). Observed time-to-event (days from discharge).
        For censored patients: time = horizon.
    event_indicators : np.ndarray
        Shape (N,). 1 = event occurred, 0 = censored (no event within horizon).

    Returns
    -------
    dict with:
        "ebf_rate" : float  - events before follow-up / total patients
        "ebf_count" : int   - number of events before follow-up
        "total_events" : int - total events in cohort
        "events_caught" : int - events that would have had prior follow-up
    """
    n = len(assignments)
    ebf_count = 0
    total_events = 0

    for i in range(n):
        if event_indicators[i] == 1:
            total_events += 1
            if event_times[i] < assignments[i]:
                ebf_count += 1

    events_caught = total_events - ebf_count

    return {
        "ebf_rate": ebf_count / n if n > 0 else 0.0,
        "ebf_count": ebf_count,
        "total_events": total_events,
        "events_caught": events_caught,
        "catch_rate": events_caught / total_events if total_events > 0 else 1.0,
    }


def expected_cost_metric(
    assignments: dict[int, int],
    survival_curves: np.ndarray,
    c_event: float,
    c_visit: float,
) -> dict:
    """Total and mean expected cost under a policy.

    Parameters
    ----------
    assignments : dict[int, int]
        Patient index -> assigned day.
    survival_curves : np.ndarray
        Shape (N, horizon+1).
    c_event, c_visit : float
        Cost parameters.

    Returns
    -------
    dict with "total_cost", "mean_cost", "total_risk_cost", "total_visit_cost".
    """
    total_risk = 0.0
    n = len(assignments)

    for i, d in assignments.items():
        total_risk += c_event * (1.0 - survival_curves[i, d])

    total_visit = n * c_visit
    total = total_risk + total_visit

    return {
        "total_cost": total,
        "mean_cost": total / n if n > 0 else 0.0,
        "total_risk_cost": total_risk,
        "total_visit_cost": total_visit,
    }


def capacity_utilisation(
    assignments: dict[int, int],
    capacity_per_day: np.ndarray,
    horizon: int,
) -> dict:
    """How well does the policy use available capacity?

    Returns
    -------
    dict with "utilisation_by_day", "mean_utilisation", "overflow_days".
    """
    used = np.zeros(horizon)
    for d in assignments.values():
        if 1 <= d <= horizon:
            used[d - 1] += 1

    utilisation = np.where(
        capacity_per_day > 0, used / capacity_per_day, 0.0
    )

    overflow_days = int(np.sum(used > capacity_per_day))

    return {
        "utilisation_by_day": utilisation,
        "mean_utilisation": float(np.mean(utilisation)),
        "overflow_days": overflow_days,
        "patients_per_day": used,
    }


def scheduling_distribution(
    assignments: dict[int, int],
    horizon: int,
) -> dict:
    """Descriptive statistics of the schedule.

    Returns
    -------
    dict with "mean_day", "median_day", "std_day", "day_histogram".
    """
    days = np.array(list(assignments.values()), dtype=float)

    histogram = np.zeros(horizon)
    for d in assignments.values():
        if 1 <= d <= horizon:
            histogram[d - 1] += 1

    return {
        "mean_day": float(np.mean(days)),
        "median_day": float(np.median(days)),
        "std_day": float(np.std(days)),
        "min_day": int(np.min(days)),
        "max_day": int(np.max(days)),
        "day_histogram": histogram,
    }


def evaluate_policy(
    assignments: dict[int, int],
    survival_curves: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    capacity_per_day: np.ndarray,
    c_event: float,
    c_visit: float,
    horizon: int,
) -> dict:
    """Run all metrics on a policy. Single entry point for evaluation."""
    ebf = event_before_followup_rate(assignments, event_times, event_indicators)
    cost = expected_cost_metric(assignments, survival_curves, c_event, c_visit)
    cap = capacity_utilisation(assignments, capacity_per_day, horizon)
    dist = scheduling_distribution(assignments, horizon)

    return {
        "ebf": ebf,
        "cost": cost,
        "capacity": cap,
        "distribution": dist,
    }
