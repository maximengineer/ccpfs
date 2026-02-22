"""
Sensitivity Analysis
---------------------
Systematically varies key parameters to test robustness of the CCPFS
scheduling policy and identify conditions under which it degrades.

Produces data for the paper's sensitivity plots.
"""

import numpy as np

from config import C_EVENT, C_VISIT, HORIZON_DAYS, DEFAULT_DAILY_CAPACITY
from evaluation.simulate import run_all_policies


def vary_capacity(
    survival_curves: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    capacities: list[int] = None,
    horizon: int = HORIZON_DAYS,
    **kwargs,
) -> dict:
    """Run all policies across different daily capacity levels.

    Tests H3: advantage of optimised allocation increases as capacity shrinks.

    Parameters
    ----------
    capacities : list[int]
        Daily capacity values to test. Default: [5, 10, 20, 50, 999].
        999 = effectively unconstrained.

    Returns
    -------
    dict[int, dict]
        Capacity -> policy comparison results.
    """
    if capacities is None:
        capacities = [5, 10, 20, 50, 999]

    results = {}
    for cap in capacities:
        cap_array = np.full(horizon, cap)
        results[cap] = run_all_policies(
            survival_curves=survival_curves,
            event_times=event_times,
            event_indicators=event_indicators,
            capacity_per_day=cap_array,
            horizon=horizon,
            **kwargs,
        )
    return results


def vary_cost_ratio(
    survival_curves: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    ratios: list[float] = None,
    c_visit: float = C_VISIT,
    horizon: int = HORIZON_DAYS,
    **kwargs,
) -> dict:
    """Run all policies across different C_EVENT / C_VISIT ratios.

    Parameters
    ----------
    ratios : list[float]
        C_EVENT / C_VISIT ratios to test. Default: [10, 33, 67, 100, 200].

    Returns
    -------
    dict[float, dict]
        Ratio -> policy comparison results.
    """
    if ratios is None:
        ratios = [10, 33, 67, 100, 200]

    results = {}
    for ratio in ratios:
        c_event = c_visit * ratio
        results[ratio] = run_all_policies(
            survival_curves=survival_curves,
            event_times=event_times,
            event_indicators=event_indicators,
            c_event=c_event,
            c_visit=c_visit,
            horizon=horizon,
            **kwargs,
        )
    return results


def vary_uncertainty_alpha(
    survival_curves: np.ndarray,
    survival_stds: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    alphas: list[float] = None,
    horizon: int = HORIZON_DAYS,
    **kwargs,
) -> dict:
    """Run the uncertainty-adjusted policy across alpha values.

    Tests H2: incorporating uncertainty improves expected utility.

    Parameters
    ----------
    alphas : list[float]
        Uncertainty adjustment values. Default: [0, 0.4, 0.8, 1.2].
        0 = ignore uncertainty. Higher = more conservative.

    Returns
    -------
    dict[float, dict]
        Alpha -> policy comparison results.
    """
    if alphas is None:
        alphas = [0.0, 0.4, 0.8, 1.2]

    results = {}
    for alpha in alphas:
        results[alpha] = run_all_policies(
            survival_curves=survival_curves,
            event_times=event_times,
            event_indicators=event_indicators,
            survival_stds=survival_stds,
            uncertainty_alpha=alpha,
            horizon=horizon,
            **kwargs,
        )
    return results


def vary_horizon(
    survival_curves: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    horizons: list[int] = None,
    **kwargs,
) -> dict:
    """Run all policies across different prediction horizons.

    Parameters
    ----------
    horizons : list[int]
        Horizon values to test. Default: [14, 21, 30].

    Returns
    -------
    dict[int, dict]
        Horizon -> policy comparison results.
    """
    if horizons is None:
        horizons = [14, 21, 30]

    results = {}
    for h in horizons:
        # Truncate survival curves and event times to this horizon
        truncated_curves = survival_curves[:, : h + 1]
        truncated_events = np.minimum(event_times, h)
        truncated_indicators = event_indicators.copy()
        # Censor events beyond the horizon
        truncated_indicators[event_times > h] = 0

        results[h] = run_all_policies(
            survival_curves=truncated_curves,
            event_times=truncated_events,
            event_indicators=truncated_indicators,
            horizon=h,
            **kwargs,
        )
    return results
