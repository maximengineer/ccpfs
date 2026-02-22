"""
Survival Curve Calibration
----------------------------
Isotonic regression at selected horizons, interpolated to full curve.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression

from config import HORIZON_DAYS


def calibrate_curves(
    survival_curves: np.ndarray,
    event_indicators: np.ndarray,
    time_to_event: np.ndarray,
    calibration_horizons: list[int] = None,
    horizon: int = HORIZON_DAYS,
) -> tuple[np.ndarray, dict]:
    """Calibrate survival curves using isotonic regression at key horizons.

    Parameters
    ----------
    survival_curves : np.ndarray, shape (N, horizon+1)
        Uncalibrated survival curves.
    event_indicators : np.ndarray
        Boolean, True if event observed.
    time_to_event : np.ndarray
        Observed time.
    calibration_horizons : list[int]
        Time points at which to fit isotonic regression.
    horizon : int

    Returns
    -------
    calibrated_curves : np.ndarray, shape (N, horizon+1)
    calibrators : dict
        Horizon -> fitted IsotonicRegression (for applying to new data).
    """
    if calibration_horizons is None:
        calibration_horizons = [7, 14, 21, 30]

    calibrators = {}
    calibrated_at = {}

    for t in calibration_horizons:
        # Predicted risk at horizon t
        predicted_risk = 1.0 - survival_curves[:, t]

        # Binary label: did event occur by time t?
        # Subjects censored before t are excluded from calibration
        observable = (event_indicators) | (time_to_event > t)
        label = (event_indicators & (time_to_event <= t)).astype(float)

        if observable.sum() < 20:
            continue

        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso.fit(predicted_risk[observable], label[observable])
        calibrators[t] = iso

        # Calibrate all patients at this horizon
        calibrated_risk = iso.predict(predicted_risk)
        calibrated_at[t] = 1.0 - calibrated_risk  # Back to survival probability

    # Interpolate calibrated values to full curve
    calibrated_curves = survival_curves.copy()
    calibrated_curves[:, 0] = 1.0

    # Sort horizons
    cal_times = sorted(calibrated_at.keys())
    if not cal_times:
        return calibrated_curves, calibrators

    for day in range(1, horizon + 1):
        if day in calibrated_at:
            calibrated_curves[:, day] = calibrated_at[day]
        elif day < cal_times[0]:
            # Before first calibration point: linear interpolation from 1.0
            alpha = day / cal_times[0]
            calibrated_curves[:, day] = (
                (1 - alpha) * 1.0 + alpha * calibrated_at[cal_times[0]]
            )
        elif day > cal_times[-1]:
            # After last calibration point: use last calibrated value ratio
            ratio = survival_curves[:, day] / np.clip(
                survival_curves[:, cal_times[-1]], 1e-10, 1.0
            )
            calibrated_curves[:, day] = calibrated_at[cal_times[-1]] * ratio
        else:
            # Between two calibration points: linear interpolation
            t_lo = max(t for t in cal_times if t <= day)
            t_hi = min(t for t in cal_times if t >= day)
            if t_lo == t_hi:
                calibrated_curves[:, day] = calibrated_at[t_lo]
            else:
                alpha = (day - t_lo) / (t_hi - t_lo)
                calibrated_curves[:, day] = (
                    (1 - alpha) * calibrated_at[t_lo]
                    + alpha * calibrated_at[t_hi]
                )

    # Enforce monotonicity and bounds
    calibrated_curves = np.minimum.accumulate(calibrated_curves, axis=1)
    calibrated_curves = np.clip(calibrated_curves, 0.0, 1.0)

    return calibrated_curves, calibrators


def apply_calibration(
    survival_curves: np.ndarray,
    calibrators: dict,
    calibration_horizons: list[int] = None,
    horizon: int = HORIZON_DAYS,
) -> np.ndarray:
    """Apply previously fitted calibrators to new survival curves."""
    if calibration_horizons is None:
        calibration_horizons = sorted(calibrators.keys())

    calibrated_at = {}
    for t in calibration_horizons:
        if t not in calibrators:
            continue
        predicted_risk = 1.0 - survival_curves[:, t]
        calibrated_risk = calibrators[t].predict(predicted_risk)
        calibrated_at[t] = 1.0 - calibrated_risk

    calibrated_curves = survival_curves.copy()
    calibrated_curves[:, 0] = 1.0

    cal_times = sorted(calibrated_at.keys())
    if not cal_times:
        return calibrated_curves

    for day in range(1, horizon + 1):
        if day in calibrated_at:
            calibrated_curves[:, day] = calibrated_at[day]
        elif day < cal_times[0]:
            alpha = day / cal_times[0]
            calibrated_curves[:, day] = (
                (1 - alpha) * 1.0 + alpha * calibrated_at[cal_times[0]]
            )
        elif day > cal_times[-1]:
            ratio = survival_curves[:, day] / np.clip(
                survival_curves[:, cal_times[-1]], 1e-10, 1.0
            )
            calibrated_curves[:, day] = calibrated_at[cal_times[-1]] * ratio
        else:
            t_lo = max(t for t in cal_times if t <= day)
            t_hi = min(t for t in cal_times if t >= day)
            if t_lo == t_hi:
                calibrated_curves[:, day] = calibrated_at[t_lo]
            else:
                alpha = (day - t_lo) / (t_hi - t_lo)
                calibrated_curves[:, day] = (
                    (1 - alpha) * calibrated_at[t_lo]
                    + alpha * calibrated_at[t_hi]
                )

    calibrated_curves = np.minimum.accumulate(calibrated_curves, axis=1)
    calibrated_curves = np.clip(calibrated_curves, 0.0, 1.0)

    return calibrated_curves
