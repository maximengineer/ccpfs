"""
Bootstrap Uncertainty Estimation
---------------------------------
Train K GBM models on bootstrapped training data to estimate
prediction uncertainty (epistemic uncertainty).
"""

import numpy as np
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

from config import HORIZON_DAYS, N_BOOTSTRAP, RANDOM_SEED
from models.classical.train_gbm import extract_survival_curves, make_structured_target


def bootstrap_survival_curves(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_predict: np.ndarray,
    best_params: dict,
    n_bootstrap: int = N_BOOTSTRAP,
    horizon: int = HORIZON_DAYS,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Train K GBMs on bootstrap resamples, return mean and std curves.

    Parameters
    ----------
    X_train : np.ndarray, shape (N_train, F)
    y_train : np.ndarray
        Structured array (event, time).
    X_predict : np.ndarray, shape (N_pred, F)
        Patients for whom to generate curves.
    best_params : dict
        GBM hyperparameters (from grid search).
    n_bootstrap : int
        Number of bootstrap iterations.
    horizon : int
        Prediction horizon.
    verbose : bool

    Returns
    -------
    means : np.ndarray, shape (N_pred, horizon+1)
    stds : np.ndarray, shape (N_pred, horizon+1)
    """
    n_train = X_train.shape[0]
    n_pred = X_predict.shape[0]
    all_curves = np.zeros((n_bootstrap, n_pred, horizon + 1))

    rng = np.random.RandomState(RANDOM_SEED)

    for k in range(n_bootstrap):
        if verbose:
            print(f"  Bootstrap {k+1}/{n_bootstrap}...")

        # Resample training data with replacement
        idx = rng.choice(n_train, size=n_train, replace=True)
        X_boot = X_train[idx]
        y_boot = y_train[idx]

        # Train model
        model = GradientBoostingSurvivalAnalysis(
            random_state=RANDOM_SEED + k, **best_params
        )
        model.fit(X_boot, y_boot)

        # Extract curves (sequential to control memory)
        curves = extract_survival_curves(model, X_predict, horizon=horizon)
        all_curves[k] = curves

        # Free model memory
        del model

    means = all_curves.mean(axis=0)
    stds = all_curves.std(axis=0)

    if verbose:
        avg_std = stds[:, 1:].mean()
        max_std = stds[:, 1:].max()
        print(f"  Bootstrap uncertainty: mean_std={avg_std:.4f}, max_std={max_std:.4f}")

    return means, stds
