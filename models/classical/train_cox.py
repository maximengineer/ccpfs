"""
Cox Proportional Hazards Baseline
----------------------------------
Train CoxPHFitter from lifelines as a simpler survival baseline.
Generates patient-level survival curves S(t) for t=0..30.

Features are standardised (zero mean, unit variance) before fitting
to prevent large-range features from dominating the exp(βx) link.
The scaler is saved alongside the model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler

from config import HORIZON_DAYS, MODEL_DIR


def train_cox(
    X_train: np.ndarray,
    y_event_train: np.ndarray,
    y_time_train: np.ndarray,
    feature_names: list[str],
    penalizer: float = 0.1,
    verbose: bool = True,
) -> tuple[CoxPHFitter, StandardScaler]:
    """Train Cox PH model on training data with feature scaling.

    Returns
    -------
    tuple of (CoxPHFitter, StandardScaler)
        Fitted model and the scaler used for transforming features.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Drop near-zero-variance features (cause NaN gradients in Cox)
    variances = np.var(X_scaled, axis=0)
    keep_mask = variances > 1e-6
    kept_names = [n for n, k in zip(feature_names, keep_mask) if k]
    dropped = [n for n, k in zip(feature_names, keep_mask) if not k]
    if dropped and verbose:
        print(f"  Cox PH: dropping {len(dropped)} near-zero-variance features: {dropped[:10]}{'...' if len(dropped) > 10 else ''}")

    X_filtered = X_scaled[:, keep_mask]
    df = pd.DataFrame(X_filtered, columns=kept_names)
    df["T"] = y_time_train
    df["E"] = y_event_train.astype(int)

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df, duration_col="T", event_col="E")

    if verbose:
        ci = cph.concordance_index_
        print(f"  Cox PH: C-index (train) = {ci:.4f}")

    # Store which features were kept so prediction uses the same subset
    cph._kept_feature_mask = keep_mask
    cph._kept_feature_names = kept_names

    return cph, scaler


def extract_survival_curves_cox(
    model: CoxPHFitter,
    X: np.ndarray,
    feature_names: list[str],
    scaler: StandardScaler = None,
    horizon: int = HORIZON_DAYS,
) -> np.ndarray:
    """Extract S(t) for t=0..horizon from the fitted Cox model.

    Parameters
    ----------
    scaler : StandardScaler, optional
        If provided, applies the same scaling used during training.

    Returns
    -------
    np.ndarray of shape (N, horizon+1), monotone non-increasing.
    """
    X_input = scaler.transform(X) if scaler is not None else X
    # Apply same feature mask used during training
    if hasattr(model, '_kept_feature_mask'):
        X_input = X_input[:, model._kept_feature_mask]
        feature_names = model._kept_feature_names
    df = pd.DataFrame(X_input, columns=feature_names)
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


def save_model(model: CoxPHFitter, path: Path = None, scaler: StandardScaler = None):
    """Serialize model and scaler to disk via pickle."""
    import pickle

    if path is None:
        path = MODEL_DIR / "cox_ph.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)


def load_model(path: Path = None) -> tuple[CoxPHFitter, StandardScaler]:
    """Load serialized Cox PH model and scaler."""
    import pickle

    if path is None:
        path = MODEL_DIR / "cox_ph.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    # Backward compatibility: old saves stored just the model
    if isinstance(data, dict):
        return data["model"], data.get("scaler")
    return data, None
