"""
Survival Model Evaluation
--------------------------
Concordance index, Integrated Brier Score, and calibration data.
"""

import numpy as np
from sksurv.metrics import concordance_index_censored, integrated_brier_score

from config import HORIZON_DAYS


def compute_cindex(
    event_indicators: np.ndarray,
    time_to_event: np.ndarray,
    risk_scores: np.ndarray,
) -> float:
    """Compute Harrell's C-index.

    Parameters
    ----------
    event_indicators : np.ndarray of bool
    time_to_event : np.ndarray of float
    risk_scores : np.ndarray of float
        Higher = higher risk (e.g. 1 - S(median_t)).

    Returns
    -------
    float
        C-index in [0, 1].
    """
    ci, _, _, _, _ = concordance_index_censored(
        event_indicators.astype(bool),
        time_to_event,
        risk_scores,
    )
    return ci


def compute_ibs(
    y_train_structured: np.ndarray,
    y_test_structured: np.ndarray,
    survival_curves: np.ndarray,
    horizon: int = HORIZON_DAYS,
) -> float:
    """Compute Integrated Brier Score over [1, horizon].

    Parameters
    ----------
    y_train_structured : np.ndarray
        Structured array with (event, time) from training set (for KM censoring).
    y_test_structured : np.ndarray
        Structured array for test set.
    survival_curves : np.ndarray
        Shape (N_test, horizon+1).
    horizon : int
        Max time point.

    Returns
    -------
    float
        IBS (lower is better).
    """
    # Times must be strictly within follow-up range of test data: [min_time, max_time)
    min_test_time = y_test_structured["time"].min()
    max_test_time = y_test_structured["time"].max()

    # Use integer days that fall within the valid range
    t_start = max(1, int(np.ceil(min_test_time)))
    t_end = min(horizon, int(np.floor(max_test_time)) - 1)

    if t_end <= t_start:
        return float("nan")

    times = np.arange(t_start, t_end + 1, dtype=float)
    preds = survival_curves[:, t_start : t_end + 1]

    try:
        ibs = integrated_brier_score(
            y_train_structured,
            y_test_structured,
            preds,
            times,
        )
        return ibs
    except ValueError:
        return float("nan")


def compute_calibration_data(
    event_indicators: np.ndarray,
    time_to_event: np.ndarray,
    survival_curves: np.ndarray,
    horizons: list[int] = None,
    n_bins: int = 10,
) -> dict:
    """Compute calibration data for reliability diagrams.

    For each horizon t, bins patients by predicted P(event by t) = 1 - S(t),
    then compares to observed event rate via Kaplan-Meier in each bin.

    Parameters
    ----------
    event_indicators, time_to_event : np.ndarray
    survival_curves : np.ndarray, shape (N, horizon+1)
    horizons : list of int, e.g. [7, 14, 21, 30]
    n_bins : int

    Returns
    -------
    dict mapping horizon -> {"predicted": array, "observed": array, "counts": array}
    """
    if horizons is None:
        horizons = [7, 14, 21, 30]

    results = {}
    for t in horizons:
        predicted_risk = 1.0 - survival_curves[:, t]

        # Bin patients by predicted risk
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_pred = np.zeros(n_bins)
        bin_obs = np.zeros(n_bins)
        bin_count = np.zeros(n_bins, dtype=int)

        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            if b == n_bins - 1:
                mask = (predicted_risk >= lo) & (predicted_risk <= hi)
            else:
                mask = (predicted_risk >= lo) & (predicted_risk < hi)

            if mask.sum() == 0:
                bin_pred[b] = (lo + hi) / 2
                bin_obs[b] = float("nan")
                bin_count[b] = 0
                continue

            bin_pred[b] = predicted_risk[mask].mean()
            bin_count[b] = mask.sum()

            # Observed: fraction who had event by time t
            # (accounting for censoring: only count those with time <= t)
            events_by_t = event_indicators[mask] & (time_to_event[mask] <= t)
            censored_after_t = time_to_event[mask] > t
            # Denominator: events + those still at risk at t
            denom = events_by_t.sum() + censored_after_t.sum()
            if denom > 0:
                bin_obs[b] = events_by_t.sum() / denom
            else:
                bin_obs[b] = float("nan")

        results[t] = {
            "predicted": bin_pred,
            "observed": bin_obs,
            "counts": bin_count,
        }

    return results


def evaluate_survival_model(
    y_train_structured: np.ndarray,
    y_test_structured: np.ndarray,
    event_indicators: np.ndarray,
    time_to_event: np.ndarray,
    survival_curves: np.ndarray,
    model_name: str = "Model",
    verbose: bool = True,
) -> dict:
    """Full evaluation: C-index + IBS + calibration.

    Returns dict with all metrics.
    """
    # Risk score for C-index: 1 - S(median_horizon)
    median_t = survival_curves.shape[1] // 2
    risk_scores = 1.0 - survival_curves[:, median_t]

    ci = compute_cindex(event_indicators, time_to_event, risk_scores)
    ibs = compute_ibs(y_train_structured, y_test_structured, survival_curves)
    cal = compute_calibration_data(event_indicators, time_to_event, survival_curves)

    if verbose:
        print(f"\n  {model_name} Evaluation:")
        print(f"    C-index: {ci:.4f}")
        print(f"    IBS:     {ibs:.4f}")
        for t, data in cal.items():
            valid = ~np.isnan(data["observed"])
            if valid.any():
                ece = np.abs(data["predicted"][valid] - data["observed"][valid]).mean()
                print(f"    ECE@{t}d: {ece:.4f}")

    return {
        "c_index": ci,
        "ibs": ibs,
        "calibration": cal,
    }
