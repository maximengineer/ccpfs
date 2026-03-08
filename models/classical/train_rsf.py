"""
Random Survival Forest
-----------------------
Train RandomSurvivalForest from scikit-survival.
Generates patient-level survival curves S(t) for t=0..30.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import ParameterGrid
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

from config import HORIZON_DAYS, MODEL_DIR, RANDOM_SEED


def train_rsf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_grid: dict = None,
    verbose: bool = True,
) -> tuple[dict, RandomSurvivalForest]:
    """Train RSF with manual grid search on validation C-index.

    Returns
    -------
    best_params : dict
    best_model : RandomSurvivalForest
    """
    if param_grid is None:
        param_grid = {
            "n_estimators": [200, 500],
            "max_depth": [5, 10, None],
            "min_samples_leaf": [10, 20],
            "min_samples_split": [10],
            "max_features": ["sqrt"],
        }

    grid = list(ParameterGrid(param_grid))
    if verbose:
        print(f"  RSF grid search: {len(grid)} parameter combinations")

    best_cindex = -1.0
    best_params = None
    best_model = None

    for i, params in enumerate(grid):
        model = RandomSurvivalForest(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            **params,
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
        print(f"  RSF Best: C-index={best_cindex:.4f}, params={best_params}")

    return best_params, best_model


def extract_survival_curves_rsf(
    model: RandomSurvivalForest,
    X: np.ndarray,
    horizon: int = HORIZON_DAYS,
) -> np.ndarray:
    """Extract S(t) for t=0..horizon from the fitted RSF.

    Returns
    -------
    np.ndarray of shape (N, horizon+1), monotone non-increasing.
    """
    from models import extract_curves_from_step_functions

    surv_fns = model.predict_survival_function(X)
    return extract_curves_from_step_functions(surv_fns, horizon)


def save_model(model: RandomSurvivalForest, path: Path = None):
    """Serialize model to disk."""
    if path is None:
        path = MODEL_DIR / "rsf_survival.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path = None) -> RandomSurvivalForest:
    """Load serialized model."""
    if path is None:
        path = MODEL_DIR / "rsf_survival.joblib"
    return joblib.load(path)
