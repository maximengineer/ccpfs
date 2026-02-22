"""
Synthetic Cohort Generator
---------------------------
Generates realistic synthetic survival curves and outcomes for testing
the scheduling pipeline without real data.

Uses Weibull distributions with three risk profiles to mimic a realistic
heart failure discharge cohort:
  - High risk (~25%):  steep early hazard, 30-day event rate ~35%
  - Medium risk (~50%): moderate hazard, 30-day event rate ~15%
  - Low risk (~25%):   flat hazard, 30-day event rate ~5%
"""

import numpy as np

from config import HORIZON_DAYS, RANDOM_SEED


def generate_weibull_survival(
    scale: float,
    shape: float,
    horizon: int = HORIZON_DAYS,
) -> np.ndarray:
    """Generate a Weibull survival curve S(t) = exp(-(t/scale)^shape).

    Parameters
    ----------
    scale : float
        Weibull scale (higher = slower deterioration).
    shape : float
        Weibull shape (>1 = increasing hazard, <1 = decreasing).
    horizon : int
        Number of days.

    Returns
    -------
    np.ndarray
        Shape (horizon+1,). S(0) = 1.0, S(t) for t=1..horizon.
    """
    t = np.arange(0, horizon + 1, dtype=float)
    return np.exp(-((t / scale) ** shape))


def generate_synthetic_cohort(
    n_patients: int = 500,
    horizon: int = HORIZON_DAYS,
    seed: int = RANDOM_SEED,
    high_risk_frac: float = 0.25,
    med_risk_frac: float = 0.50,
    noise_std: float = 0.02,
) -> dict:
    """Generate a synthetic cohort with survival curves and outcomes.

    Parameters
    ----------
    n_patients : int
        Number of patients.
    horizon : int
        Scheduling horizon in days.
    seed : int
        Random seed for reproducibility.
    high_risk_frac : float
        Fraction of high-risk patients.
    med_risk_frac : float
        Fraction of medium-risk patients.
    noise_std : float
        Per-patient noise added to survival curves (simulates prediction
        uncertainty / individual variation).

    Returns
    -------
    dict with keys:
        "survival_curves" : np.ndarray, shape (N, horizon+1)
        "survival_stds" : np.ndarray, shape (N, horizon+1) - simulated uncertainty
        "event_times" : np.ndarray, shape (N,)
        "event_indicators" : np.ndarray, shape (N,)
        "risk_groups" : np.ndarray, shape (N,) - 0=low, 1=med, 2=high
        "is_heart_failure" : np.ndarray, shape (N,) - boolean
    """
    rng = np.random.default_rng(seed)

    n_high = int(n_patients * high_risk_frac)
    n_med = int(n_patients * med_risk_frac)
    n_low = n_patients - n_high - n_med

    # Weibull parameters for each risk group
    # (scale, shape) - calibrated so 30-day event rates are realistic
    profiles = {
        "high": {"scale": 40, "shape": 1.5, "n": n_high, "label": 2},
        "med": {"scale": 80, "shape": 1.3, "n": n_med, "label": 1},
        "low": {"scale": 200, "shape": 1.1, "n": n_low, "label": 0},
    }

    all_curves = []
    all_stds = []
    all_events = []
    all_indicators = []
    all_groups = []
    all_hf = []

    for group_name, params in profiles.items():
        base_curve = generate_weibull_survival(
            params["scale"], params["shape"], horizon
        )

        for _ in range(params["n"]):
            # Add per-patient variation
            patient_scale = params["scale"] * rng.lognormal(0, 0.15)
            patient_shape = params["shape"] * rng.lognormal(0, 0.05)
            curve = generate_weibull_survival(patient_scale, patient_shape, horizon)

            # Add small noise and ensure monotonicity
            noise = rng.normal(0, noise_std, horizon + 1)
            noisy_curve = curve + noise
            noisy_curve[0] = 1.0
            noisy_curve = np.clip(noisy_curve, 0.001, 1.0)
            # Enforce non-increasing (survival curves must be monotone)
            noisy_curve = np.minimum.accumulate(noisy_curve)

            # Simulate uncertainty (higher for high-risk patients)
            base_std = 0.03 if group_name == "low" else 0.06 if group_name == "med" else 0.10
            patient_std = np.full(horizon + 1, base_std)
            patient_std[0] = 0.0  # No uncertainty at t=0
            # Uncertainty grows with time
            patient_std *= np.linspace(0.5, 1.5, horizon + 1)

            # Sample actual event time from the patient's true curve
            # h(d) = S(d-1) - S(d) for d = 1..horizon
            hazards = curve[:-1] - curve[1:]  # length = horizon
            hazards = np.maximum(hazards, 0.0)  # ensure non-negative
            hazards_norm = hazards / (hazards.sum() + 1e-10)

            # Decide if event happens within horizon
            event_prob = 1.0 - curve[horizon]
            if rng.random() < event_prob:
                # Sample event day from hazard distribution
                event_day = rng.choice(np.arange(1, horizon + 1), p=hazards_norm)
                all_events.append(int(event_day))
                all_indicators.append(1)
            else:
                all_events.append(horizon)
                all_indicators.append(0)

            all_curves.append(noisy_curve)
            all_stds.append(patient_std)
            all_groups.append(params["label"])
            # 80% of high-risk are HF, 50% of med, 20% of low
            hf_prob = {"high": 0.8, "med": 0.5, "low": 0.2}[group_name]
            all_hf.append(rng.random() < hf_prob)

    # Shuffle to avoid ordering by risk group
    idx = rng.permutation(n_patients)

    return {
        "survival_curves": np.array(all_curves)[idx],
        "survival_stds": np.array(all_stds)[idx],
        "event_times": np.array(all_events)[idx],
        "event_indicators": np.array(all_indicators)[idx],
        "risk_groups": np.array(all_groups)[idx],
        "is_heart_failure": np.array(all_hf)[idx],
    }
