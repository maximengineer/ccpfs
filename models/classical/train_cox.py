"""
Cox Proportional Hazards Baseline
----------------------------------
Train CoxPHFitter from lifelines as a simpler survival baseline.
Generates patient-level survival curves S(t) for t=0..30.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from config import HORIZON_DAYS, MODEL_DIR


def train_cox(
    X_train: np.ndarray,
    y_event_train: np.ndarray,
    y_time_train: np.ndarray,
    feature_names: list[str],
    penalizer: float = 0.01,
    verbose: bool = True,
) -> CoxPHFitter:
    """Train Cox PH model on training data.

    Returns
    -------
    CoxPHFitter
        Fitted model.
    """
    df = pd.DataFrame(X_train, columns=feature_names)
    df["T"] = y_time_train
    df["E"] = y_event_train.astype(int)

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df, duration_col="T", event_col="E")

    if verbose:
        ci = cph.concordance_index_
        print(f"  Cox PH: C-index (train) = {ci:.4f}")

    return cph


def extract_survival_curves_cox(
    model: CoxPHFitter,
    X: np.ndarray,
    feature_names: list[str],
    horizon: int = HORIZON_DAYS,
) -> np.ndarray:
    """Extract S(t) for t=0..horizon from the fitted Cox model.

    Returns
    -------
    np.ndarray of shape (N, horizon+1), monotone non-increasing.
    """
    df = pd.DataFrame(X, columns=feature_names)
    times = np.arange(0, horizon + 1, dtype=float)

    # predict_survival_function returns DataFrame: rows=times, cols=subjects
    sf = model.predict_survival_function(df, times=times)

    # Transpose to (N, horizon+1)
    curves = sf.T.values.copy()

    # Enforce S(0) = 1.0 and monotonicity
    curves[:, 0] = 1.0
    curves = np.minimum.accumulate(curves, axis=1)
    curves = np.clip(curves, 0.0, 1.0)
    return curves


def save_model(model: CoxPHFitter, path: Path = None):
    """Serialize model to disk via pickle."""
    import pickle

    if path is None:
        path = MODEL_DIR / "cox_ph.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path = None) -> CoxPHFitter:
    """Load serialized Cox PH model."""
    import pickle

    if path is None:
        path = MODEL_DIR / "cox_ph.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
