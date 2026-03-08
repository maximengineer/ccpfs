"""Survival model utilities."""

import numpy as np

from config import HORIZON_DAYS


def extract_curves_from_step_functions(
    surv_fns: list,
    horizon: int = HORIZON_DAYS,
) -> np.ndarray:
    """Convert scikit-survival step functions to a dense survival matrix.

    Works with both GradientBoostingSurvivalAnalysis and
    RandomSurvivalForest predictions.

    Parameters
    ----------
    surv_fns : list of StepFunction
        Output of model.predict_survival_function(X).
    horizon : int
        Max time point.

    Returns
    -------
    np.ndarray of shape (N, horizon+1), monotone non-increasing, clipped to [0, 1].
    """
    curves = np.ones((len(surv_fns), horizon + 1))

    for i, fn in enumerate(surv_fns):
        for t in range(1, horizon + 1):
            curves[i, t] = fn(float(t))

    curves[:, 0] = 1.0
    curves = np.minimum.accumulate(curves, axis=1)
    curves = np.clip(curves, 0.0, 1.0)
    return curves
