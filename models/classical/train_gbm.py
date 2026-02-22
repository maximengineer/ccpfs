"""
Gradient Boosting Survival Model
----------------------------------
Train GradientBoostingSurvivalAnalysis from scikit-survival.
Generates patient-level survival curves S(t) for t=0..30.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import ParameterGrid
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from config import GBM_PARAM_GRID, HORIZON_DAYS, MODEL_DIR, RANDOM_SEED


def make_structured_target(
    event_indicators: np.ndarray,
    time_to_event: np.ndarray,
) -> np.ndarray:
    """Convert to scikit-survival structured array format."""
    y = np.zeros(len(event_indicators), dtype=[("event", bool), ("time", float)])
    y["event"] = event_indicators.astype(bool)
    y["time"] = time_to_event
    return y


def train_gbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_grid: dict = None,
    verbose: bool = True,
) -> tuple[dict, GradientBoostingSurvivalAnalysis]:
    """Train GBM with manual grid search on validation C-index.

    Returns
    -------
    best_params : dict
    best_model : GradientBoostingSurvivalAnalysis
    """
    if param_grid is None:
        param_grid = GBM_PARAM_GRID

    grid = list(ParameterGrid(param_grid))
    if verbose:
        print(f"  Grid search: {len(grid)} parameter combinations")

    best_cindex = -1.0
    best_params = None
    best_model = None

    for i, params in enumerate(grid):
        model = GradientBoostingSurvivalAnalysis(
            random_state=RANDOM_SEED, **params
        )
        model.fit(X_train, y_train)

        pred = model.predict(X_val)
        ci = concordance_index_censored(y_val["event"], y_val["time"], pred)[0]

        if verbose:
            print(f"    [{i+1}/{len(grid)}] {params} -> C-index: {ci:.4f}")

        if ci > best_cindex:
            best_cindex = ci
            best_params = params
            best_model = model

    if verbose:
        print(f"  Best: C-index={best_cindex:.4f}, params={best_params}")

    return best_params, best_model


def extract_survival_curves(
    model: GradientBoostingSurvivalAnalysis,
    X: np.ndarray,
    horizon: int = HORIZON_DAYS,
) -> np.ndarray:
    """Extract S(t) for t=0..horizon from the fitted model.

    Returns
    -------
    np.ndarray of shape (N, horizon+1), monotone non-increasing.
    """
    surv_fns = model.predict_survival_function(X)
    times = np.arange(0, horizon + 1, dtype=float)
    curves = np.ones((len(surv_fns), horizon + 1))

    for i, fn in enumerate(surv_fns):
        # StepFunction may not cover all time points; evaluate carefully
        for t in range(1, horizon + 1):
            curves[i, t] = fn(float(t))

    # Enforce S(0) = 1.0 and monotonicity
    curves[:, 0] = 1.0
    curves = np.minimum.accumulate(curves, axis=1)
    curves = np.clip(curves, 0.0, 1.0)
    return curves


def save_model(model: GradientBoostingSurvivalAnalysis, path: Path = None):
    """Serialize model to disk."""
    if path is None:
        path = MODEL_DIR / "gbm_survival.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path = None) -> GradientBoostingSurvivalAnalysis:
    """Load serialized model."""
    if path is None:
        path = MODEL_DIR / "gbm_survival.joblib"
    return joblib.load(path)
