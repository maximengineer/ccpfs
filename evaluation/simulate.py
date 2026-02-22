"""
Offline Policy Simulation
--------------------------
Runs all scheduling policies on a cohort and compares them.

The key insight: MIMIC-IV contains actual readmission dates, so we can
retrospectively evaluate any scheduling policy by checking whether the
assigned follow-up day would have been before or after the actual event.
"""

from typing import Optional

import numpy as np

from config import C_EVENT, C_VISIT, DEFAULT_DAILY_CAPACITY, HORIZON_DAYS
from evaluation.metrics import evaluate_policy
from policy.ilp_scheduler import schedule_ilp
from policy.greedy_scheduler import schedule_greedy
from policy.baselines import (
    uniform_policy,
    risk_bucket_policy,
    guideline_policy,
    unconstrained_optimal_policy,
)
from policy.uncertainty_adjustment import apply_uncertainty_adjustment


def run_all_policies(
    survival_curves: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    capacity_per_day: Optional[np.ndarray] = None,
    survival_stds: Optional[np.ndarray] = None,
    is_heart_failure: Optional[np.ndarray] = None,
    c_event: float = C_EVENT,
    c_visit: float = C_VISIT,
    horizon: int = HORIZON_DAYS,
    uncertainty_alpha: float = 0.8,
) -> dict:
    """Run all scheduling policies and return comparative results.

    Parameters
    ----------
    survival_curves : np.ndarray
        Shape (N, horizon+1). Mean survival curves.
    event_times : np.ndarray
        Shape (N,). Observed time-to-event.
    event_indicators : np.ndarray
        Shape (N,). 1 = event, 0 = censored.
    capacity_per_day : np.ndarray, optional
        Shape (horizon,). Daily capacity. Defaults to DEFAULT_DAILY_CAPACITY.
    survival_stds : np.ndarray, optional
        Shape (N, horizon+1). For uncertainty-adjusted scheduling.
    is_heart_failure : np.ndarray, optional
        Shape (N,). Boolean, for guideline policy.
    c_event, c_visit : float
        Cost parameters.
    horizon : int
        Scheduling horizon.
    uncertainty_alpha : float
        Conservatism for uncertainty-adjusted policy.

    Returns
    -------
    dict[str, dict]
        Policy name -> evaluation results.
    """
    if capacity_per_day is None:
        capacity_per_day = np.full(horizon, DEFAULT_DAILY_CAPACITY)

    results = {}

    # --- Baselines ---
    baseline_configs = {
        "Uniform-7": lambda: uniform_policy(survival_curves, day=7, c_event=c_event, c_visit=c_visit),
        "Uniform-14": lambda: uniform_policy(survival_curves, day=14, c_event=c_event, c_visit=c_visit),
        "Risk-Bucket": lambda: risk_bucket_policy(survival_curves, c_event=c_event, c_visit=c_visit, horizon=horizon),
        "Guideline": lambda: guideline_policy(survival_curves, is_heart_failure=is_heart_failure, c_event=c_event, c_visit=c_visit),
        "Unconstrained": lambda: unconstrained_optimal_policy(survival_curves, c_event=c_event, c_visit=c_visit, horizon=horizon),
    }

    for name, policy_fn in baseline_configs.items():
        policy_result = policy_fn()
        results[name] = evaluate_policy(
            assignments=policy_result["assignments"],
            survival_curves=survival_curves,
            event_times=event_times,
            event_indicators=event_indicators,
            capacity_per_day=capacity_per_day,
            c_event=c_event,
            c_visit=c_visit,
            horizon=horizon,
        )

    # --- ILP (point estimate) ---
    ilp_result = schedule_ilp(
        survival_curves, capacity_per_day, c_event, c_visit, horizon
    )
    if ilp_result["status"] == "Optimal":
        results["CCPFS-ILP"] = evaluate_policy(
            assignments=ilp_result["assignments"],
            survival_curves=survival_curves,
            event_times=event_times,
            event_indicators=event_indicators,
            capacity_per_day=capacity_per_day,
            c_event=c_event,
            c_visit=c_visit,
            horizon=horizon,
        )

    # --- Greedy ---
    greedy_result = schedule_greedy(
        survival_curves, capacity_per_day, c_event, horizon
    )
    results["CCPFS-Greedy"] = evaluate_policy(
        assignments=greedy_result["assignments"],
        survival_curves=survival_curves,
        event_times=event_times,
        event_indicators=event_indicators,
        capacity_per_day=capacity_per_day,
        c_event=c_event,
        c_visit=c_visit,
        horizon=horizon,
    )

    # --- ILP with uncertainty adjustment ---
    if survival_stds is not None:
        conservative_curves = apply_uncertainty_adjustment(
            survival_curves, survival_stds, alpha=uncertainty_alpha
        )
        ilp_unc_result = schedule_ilp(
            conservative_curves, capacity_per_day, c_event, c_visit, horizon
        )
        if ilp_unc_result["status"] == "Optimal":
            # Evaluate against original curves (not conservative ones)
            results["CCPFS-ILP-Uncertainty"] = evaluate_policy(
                assignments=ilp_unc_result["assignments"],
                survival_curves=survival_curves,
                event_times=event_times,
                event_indicators=event_indicators,
                capacity_per_day=capacity_per_day,
                c_event=c_event,
                c_visit=c_visit,
                horizon=horizon,
            )

    return results


def results_summary_table(results: dict) -> str:
    """Format results as a readable comparison table.

    Returns
    -------
    str
        Markdown-formatted table.
    """
    header = (
        "| Policy | EBF Rate | Catch Rate | Mean Cost | "
        "Mean Day | Capacity Util |\n"
        "|--------|----------|------------|-----------|"
        "---------|---------------|\n"
    )
    rows = []
    for name, r in results.items():
        ebf = r["ebf"]["ebf_rate"]
        catch = r["ebf"]["catch_rate"]
        cost = r["cost"]["mean_cost"]
        day = r["distribution"]["mean_day"]
        util = r["capacity"]["mean_utilisation"]
        rows.append(
            f"| {name} | {ebf:.4f} | {catch:.4f} | "
            f"€{cost:,.0f} | {day:.1f} | {util:.2%} |"
        )

    return header + "\n".join(rows)
