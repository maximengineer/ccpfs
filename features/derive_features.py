"""
Derived Feature Engineering
-----------------------------
Create derived features from the base 47 extracted features.
Runs as a post-processing step — no need to re-extract from MEDS shards.

New features:
  - Lab instability ranges (max - min) for each lab
  - Lab ratios (creatinine/BUN for kidney function)
  - Log-transforms for heavily skewed count features
  - Comorbidity burden score (sum of binary flags)
  - Clinically meaningful interactions
  - Discharge acuity composite
"""

import numpy as np


# Labs that have last/max/min columns — ranges computed for all
_LABS = [
    "creatinine", "bun", "sodium", "potassium", "hemoglobin", "wbc", "bnp",
    "albumin", "lactate", "glucose", "troponin_t", "ast", "alt", "bilirubin",
    "inr", "bicarbonate", "calcium", "magnesium", "phosphate",
    "heart_rate", "sbp_arterial", "dbp_arterial", "sbp_noninvasive",
    "dbp_noninvasive", "respiratory_rate", "spo2", "temperature_f", "temperature_c",
]

# Comorbidity flag columns (original 6 + new 6)
_COMORBIDITY_FLAGS = [
    "has_heart_failure", "has_diabetes", "has_ckd",
    "has_copd", "has_hypertension", "has_afib",
    "has_liver_disease", "has_malignancy", "has_depression",
    "has_obesity", "has_stroke", "has_acs",
]


def derive_features(
    X: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Create derived features from base feature matrix.

    Parameters
    ----------
    X : np.ndarray, shape (N, 47)
        Base feature matrix (already imputed).
    feature_names : list[str]
        Column names for X.

    Returns
    -------
    X_out : np.ndarray, shape (N, F_new)
        Extended feature matrix.
    names_out : list[str]
        Column names for X_out.
    """
    idx = {name: i for i, name in enumerate(feature_names)}
    new_cols = []
    new_names = []

    # --- 1. Lab instability ranges: max - min ---
    for lab in _LABS:
        max_col = f"{lab}_max"
        min_col = f"{lab}_min"
        if max_col in idx and min_col in idx:
            lab_range = X[:, idx[max_col]] - X[:, idx[min_col]]
            new_cols.append(lab_range)
            new_names.append(f"{lab}_range")

    # --- 2. Creatinine/BUN ratio (renal function indicator) ---
    if "creatinine_last" in idx and "bun_last" in idx:
        bun = X[:, idx["bun_last"]]
        creat = X[:, idx["creatinine_last"]]
        # BUN/Creatinine ratio (normal ~10-20:1)
        ratio = np.where(creat > 0.01, bun / creat, 0.0)
        new_cols.append(ratio)
        new_names.append("bun_creatinine_ratio")

    # --- 3. Sodium range (instability indicator) ---
    # Already captured by sodium_range above, but add a binary flag for
    # clinically significant sodium swings (>5 mEq/L)
    if "sodium_max" in idx and "sodium_min" in idx:
        na_range = X[:, idx["sodium_max"]] - X[:, idx["sodium_min"]]
        new_cols.append((na_range > 5).astype(float))
        new_names.append("sodium_instability")

    # --- 4. Anemia severity (hemoglobin < 10) ---
    if "hemoglobin_last" in idx:
        new_cols.append((X[:, idx["hemoglobin_last"]] < 10.0).astype(float))
        new_names.append("anemia_flag")

    # --- 5. Renal impairment (creatinine > 1.5) ---
    if "creatinine_last" in idx:
        new_cols.append((X[:, idx["creatinine_last"]] > 1.5).astype(float))
        new_names.append("elevated_creatinine")

    # --- 6. Comorbidity burden score ---
    comorbidity_cols = [idx[f] for f in _COMORBIDITY_FLAGS if f in idx]
    if comorbidity_cols:
        burden = X[:, comorbidity_cols].sum(axis=1)
        new_cols.append(burden)
        new_names.append("comorbidity_burden")

    # --- 7. Log-transforms for skewed features ---
    for feat in ["n_prior_admissions", "n_prior_admissions_365d",
                 "days_since_last_discharge", "n_medications",
                 "los_days", "n_diagnoses"]:
        if feat in idx:
            new_cols.append(np.log1p(X[:, idx[feat]]))
            new_names.append(f"{feat}_log1p")

    # --- 8. Interaction features ---
    # Age × heart failure (older HF patients at highest risk)
    if "age" in idx and "has_heart_failure" in idx:
        new_cols.append(X[:, idx["age"]] * X[:, idx["has_heart_failure"]])
        new_names.append("age_x_heart_failure")

    # Prior admissions × LOS (frequent fliers with long stays)
    if "n_prior_admissions" in idx and "los_days" in idx:
        new_cols.append(
            np.log1p(X[:, idx["n_prior_admissions"]]) * X[:, idx["los_days"]]
        )
        new_names.append("prior_admissions_x_los")

    # Emergency admission × ICU stay (high acuity)
    if "admission_type_emergency" in idx and "has_icu_stay" in idx:
        new_cols.append(
            X[:, idx["admission_type_emergency"]] * X[:, idx["has_icu_stay"]]
        )
        new_names.append("emergency_x_icu")

    # Comorbidity burden × age (multi-morbid elderly)
    if comorbidity_cols and "age" in idx:
        new_cols.append(burden * X[:, idx["age"]])
        new_names.append("comorbidity_x_age")

    # Recent readmission flag (prior admission within 30 days)
    if "days_since_last_discharge" in idx:
        days = X[:, idx["days_since_last_discharge"]]
        new_cols.append(((days > 0) & (days <= 30)).astype(float))
        new_names.append("recent_readmission_30d")

    # --- 9. Vital sign flags (clinically meaningful thresholds) ---
    # Tachycardia at discharge (HR > 100)
    if "heart_rate_last" in idx:
        new_cols.append((X[:, idx["heart_rate_last"]] > 100).astype(float))
        new_names.append("tachycardia_flag")

    # Hypotension (SBP < 90)
    for sbp_col in ["sbp_noninvasive_last", "sbp_arterial_last"]:
        if sbp_col in idx:
            new_cols.append((X[:, idx[sbp_col]] < 90).astype(float))
            new_names.append(f"{sbp_col.replace('_last','')}_hypotension")
            break

    # Tachypnea (RR > 22)
    if "respiratory_rate_last" in idx:
        new_cols.append((X[:, idx["respiratory_rate_last"]] > 22).astype(float))
        new_names.append("tachypnea_flag")

    # Hypoxemia (SpO2 < 94%)
    if "spo2_last" in idx:
        new_cols.append((X[:, idx["spo2_last"]] < 94).astype(float))
        new_names.append("hypoxemia_flag")

    # --- 10. Liver dysfunction flag (AST or ALT > 3x upper limit) ---
    if "ast_last" in idx:
        new_cols.append((X[:, idx["ast_last"]] > 120).astype(float))
        new_names.append("elevated_ast")
    if "alt_last" in idx:
        new_cols.append((X[:, idx["alt_last"]] > 120).astype(float))
        new_names.append("elevated_alt")

    # Hypoalbuminemia (albumin < 3.5 g/dL)
    if "albumin_last" in idx:
        new_cols.append((X[:, idx["albumin_last"]] < 3.5).astype(float))
        new_names.append("hypoalbuminemia_flag")

    # Elevated lactate (> 2 mmol/L)
    if "lactate_last" in idx:
        new_cols.append((X[:, idx["lactate_last"]] > 2.0).astype(float))
        new_names.append("elevated_lactate")

    # Elevated troponin (> 0.04 ng/mL)
    if "troponin_t_last" in idx:
        new_cols.append((X[:, idx["troponin_t_last"]] > 0.04).astype(float))
        new_names.append("elevated_troponin")

    # Elevated INR (> 1.5 — coagulopathy)
    if "inr_last" in idx:
        new_cols.append((X[:, idx["inr_last"]] > 1.5).astype(float))
        new_names.append("elevated_inr")

    # --- 11. Discharge acuity composite (fixed: no global min/max) ---
    acuity_components = []
    if "admission_type_emergency" in idx:
        acuity_components.append(X[:, idx["admission_type_emergency"]])
    if "has_icu_stay" in idx:
        acuity_components.append(X[:, idx["has_icu_stay"]])
    if acuity_components:
        acuity = np.mean(acuity_components, axis=0)
        new_cols.append(acuity)
        new_names.append("discharge_acuity")

    # --- Assemble output ---
    if new_cols:
        X_derived = np.column_stack(new_cols)
        # Replace any NaN/inf in derived features
        X_derived = np.nan_to_num(X_derived, nan=0.0, posinf=0.0, neginf=0.0)
        X_out = np.hstack([X, X_derived])
        names_out = list(feature_names) + new_names
    else:
        X_out = X
        names_out = list(feature_names)

    return X_out, names_out
