"""
Uncertainty-Aware Scheduling Adjustment
----------------------------------------
When prediction uncertainty is high, schedule conservatively (earlier).

The idea: two patients with identical point-estimate risk but different
uncertainty should be treated differently. The uncertain patient should
get an earlier slot as a safety margin.

Implementation: replace S_mean(t) with a conservative lower bound
  S_conservative(t) = S_mean(t) - alpha * S_std(t)

This increases the apparent risk for uncertain patients, causing the
scheduler to assign them earlier days.

When alpha = 0: ignore uncertainty (point estimate only)
When alpha > 0: schedule uncertain patients earlier
"""

import numpy as np

from config import UNCERTAINTY_ALPHA


def apply_uncertainty_adjustment(
    survival_means: np.ndarray,
    survival_stds: np.ndarray,
    alpha: float = UNCERTAINTY_ALPHA,
) -> np.ndarray:
    """Produce conservative survival curves for scheduling.

    Parameters
    ----------
    survival_means : np.ndarray
        Shape (N, T). Mean survival curves from bootstrap/MC dropout.
    survival_stds : np.ndarray
        Shape (N, T). Standard deviation of survival curves.
    alpha : float
        Conservatism parameter. Higher = more conservative (earlier scheduling
        for uncertain patients).

    Returns
    -------
    np.ndarray
        Shape (N, T). Conservative survival curves, clipped to [0, 1].
    """
    conservative = survival_means - alpha * survival_stds
    return np.clip(conservative, 0.0, 1.0)


def uncertainty_score(survival_stds: np.ndarray) -> np.ndarray:
    """Per-patient uncertainty score (mean CI width across horizon).

    Parameters
    ----------
    survival_stds : np.ndarray
        Shape (N, T). Standard deviation of survival curves.

    Returns
    -------
    np.ndarray
        Shape (N,). Mean uncertainty per patient.
    """
    return np.mean(survival_stds, axis=1)
