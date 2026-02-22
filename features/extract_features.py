"""
Feature Extraction Pipeline
-----------------------------
Extracts tabular features from MEDS data for each cohort discharge episode.

Streaming architecture: processes one shard at a time, extracting features
only for subjects that appear in the cohort.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from config import MEDS_DATA_DIR, PROCESSED_DIR
from features.comorbidity_flags import compute_comorbidity_flags
from features.lab_aggregates import compute_lab_features

# Columns needed from each shard for feature extraction
_FEATURE_COLUMNS = ["subject_id", "time", "code", "numeric_value", "hadm_id"]

# ICU-related transfer locations
_ICU_KEYWORDS = {
    "Intensive Care",
    "ICU",
    "MICU",
    "SICU",
    "CCU",
    "CSICU",
    "TSICU",
    "CVICU",
    "Neuro Stepdown",
    "Neuro Intermediate",
}


def extract_features(
    cohort_path: Path = None,
    cohort_df: pl.DataFrame = None,
    meds_data_dir: Path = MEDS_DATA_DIR,
    output_dir: Path = None,
    verbose: bool = True,
) -> tuple[np.ndarray, list[str], pl.DataFrame]:
    """Extract features for all cohort episodes.

    Parameters
    ----------
    cohort_path : Path
        Path to cohort.parquet.
    cohort_df : pl.DataFrame, optional
        Pre-loaded cohort DataFrame (overrides cohort_path).
    meds_data_dir : Path
        MEDS data directory.
    output_dir : Path
        Where to save feature files.
    verbose : bool
        Print progress.

    Returns
    -------
    X : np.ndarray
        Feature matrix (N, F).
    feature_names : list[str]
        Feature column names.
    cohort : pl.DataFrame
        Cohort with features aligned by row order.
    """
    if cohort_path is None:
        cohort_path = PROCESSED_DIR / "cohort.parquet"
    if output_dir is None:
        output_dir = PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if cohort_df is not None:
        cohort = cohort_df
    else:
        cohort = pl.read_parquet(cohort_path)

    # Build index: shard_path -> list of (subject_id, hadm_id, admission_time, discharge_time)
    shard_episodes = {}
    for row in cohort.iter_rows(named=True):
        sp = row["shard_path"]
        if sp not in shard_episodes:
            shard_episodes[sp] = []
        shard_episodes[sp].append(row)

    all_feature_rows = []
    n_shards = len(shard_episodes)

    for idx, (shard_path, episodes) in enumerate(shard_episodes.items()):
        if verbose and idx % 50 == 0:
            print(f"  Extracting features: shard {idx}/{n_shards}...")

        shard_features = _extract_shard_features(Path(shard_path), episodes)
        all_feature_rows.extend(shard_features)

    # Convert to DataFrame, align with cohort order
    feature_df = pl.DataFrame(all_feature_rows)

    # Join back to cohort on hadm_id to preserve order
    cohort_with_features = cohort.join(
        feature_df, on="hadm_id", how="left"
    )

    # Extract feature columns (everything added by feature extraction)
    feature_names = [c for c in feature_df.columns if c != "hadm_id"]
    X = cohort_with_features.select(feature_names).to_numpy().astype(np.float64)

    if verbose:
        n_features = len(feature_names)
        nan_pct = 100 * np.isnan(X).mean()
        print(f"\n  Features extracted: {X.shape[0]:,} episodes x {n_features} features")
        print(f"  NaN rate before imputation: {nan_pct:.1f}%")

    return X, feature_names, cohort_with_features


def _extract_shard_features(
    shard_path: Path,
    episodes: list[dict],
) -> list[dict]:
    """Extract features for all cohort episodes from one shard."""
    df = pl.scan_parquet(shard_path).select(_FEATURE_COLUMNS).collect()

    # Build subject-level indices
    subject_ids = {e["subject_id"] for e in episodes}
    df = df.filter(pl.col("subject_id").is_in(subject_ids))

    # Group events by hadm_id for efficient lookup
    events_by_hadm = {}
    for row in df.iter_rows(named=True):
        hid = row["hadm_id"]
        if hid is not None:
            if hid not in events_by_hadm:
                events_by_hadm[hid] = []
            events_by_hadm[hid].append(row)

    # Events without hadm_id (need time-range matching)
    null_hadm_events = df.filter(pl.col("hadm_id").is_null())

    # Count prior admissions per subject
    admissions_by_subject = {}
    for row in df.filter(
        pl.col("code").str.starts_with("HOSPITAL_ADMISSION//")
    ).iter_rows(named=True):
        sid = row["subject_id"]
        if sid not in admissions_by_subject:
            admissions_by_subject[sid] = []
        admissions_by_subject[sid].append(row["time"])

    # Count prior discharges per subject (for days_since_last_discharge)
    discharges_by_subject = {}
    for row in df.filter(
        pl.col("code").str.starts_with("HOSPITAL_DISCHARGE//")
    ).iter_rows(named=True):
        sid = row["subject_id"]
        if sid not in discharges_by_subject:
            discharges_by_subject[sid] = []
        discharges_by_subject[sid].append(row["time"])

    results = []
    for ep in episodes:
        features = _extract_episode_features(
            ep, events_by_hadm, null_hadm_events,
            admissions_by_subject, discharges_by_subject,
        )
        features["hadm_id"] = ep["hadm_id"]
        results.append(features)

    return results


def _extract_episode_features(
    episode: dict,
    events_by_hadm: dict,
    null_hadm_events: pl.DataFrame,
    admissions_by_subject: dict,
    discharges_by_subject: dict,
) -> dict[str, float]:
    """Extract all features for a single discharge episode."""
    hadm_id = episode["hadm_id"]
    subject_id = episode["subject_id"]
    admission_time = episode["admission_time"]
    discharge_time = episode["discharge_time"]

    features = {}

    # --- Static features ---
    features["age"] = float(episode["age"])
    features["gender_male"] = 1.0 if episode["gender"] == "M" else 0.0
    features["los_days"] = float(episode["los_days"])

    admit_type = episode.get("admission_type", "")
    features["admission_type_emergency"] = 1.0 if admit_type in (
        "EW EMER.", "DIRECT EMER."
    ) else 0.0
    features["ed_origin"] = 1.0 if admit_type.startswith(("EW", "EU")) else 0.0

    # --- Get events for this admission ---
    hadm_events = events_by_hadm.get(hadm_id, [])

    # Also get null-hadm_id events in the admission time window for this subject
    if len(null_hadm_events) > 0:
        windowed = null_hadm_events.filter(
            (pl.col("subject_id") == subject_id)
            & (pl.col("time") >= admission_time)
            & (pl.col("time") <= discharge_time)
        )
        if len(windowed) > 0:
            hadm_events = hadm_events + windowed.to_dicts()

    # Separate event types
    diagnosis_codes = []
    lab_rows = []
    med_codes = set()
    has_icu = False
    n_procedures = 0

    for evt in hadm_events:
        code = evt["code"]
        if code is None:
            continue
        if code.startswith("DIAGNOSIS//ICD//"):
            diagnosis_codes.append(code)
        elif code.startswith("LAB//"):
            lab_rows.append(evt)
        elif code.startswith("INFUSION_START//") or code.startswith("INFUSION_END//"):
            med_codes.add(code.split("//")[1] if "//" in code else code)
        elif code.startswith("PROCEDURE//ICD//"):
            n_procedures += 1
        elif code.startswith("TRANSFER_TO//"):
            service = code.replace("TRANSFER_TO//", "")
            if any(kw in service for kw in _ICU_KEYWORDS):
                has_icu = True

    # --- Comorbidity flags ---
    comorbidity = compute_comorbidity_flags(diagnosis_codes)
    features.update(comorbidity)

    # --- Lab features ---
    if lab_rows:
        lab_df = pl.DataFrame(lab_rows).select("code", "numeric_value", "time")
        lab_features = compute_lab_features(lab_df)
    else:
        lab_features = compute_lab_features(pl.DataFrame())
    features.update(lab_features)

    # --- Prior utilisation ---
    prior_admissions = [
        t for t in admissions_by_subject.get(subject_id, [])
        if t < admission_time
    ]
    features["n_prior_admissions"] = float(len(prior_admissions))

    # Prior admissions within 365 days
    cutoff_365 = admission_time
    n_365 = sum(
        1 for t in prior_admissions
        if (cutoff_365 - t).total_seconds() <= 365 * 86400
    )
    features["n_prior_admissions_365d"] = float(n_365)

    # Days since last discharge
    prior_discharges = [
        t for t in discharges_by_subject.get(subject_id, [])
        if t < admission_time
    ]
    if prior_discharges:
        last_disch = max(prior_discharges)
        features["days_since_last_discharge"] = (
            (admission_time - last_disch).total_seconds() / 86400.0
        )
    else:
        features["days_since_last_discharge"] = float("nan")

    # --- Medications and procedures ---
    features["n_medications"] = float(len(med_codes))
    features["n_icd_procedures"] = float(n_procedures)
    features["has_icu_stay"] = 1.0 if has_icu else 0.0

    return features


def impute_features(
    X: np.ndarray,
    feature_names: list[str],
    split_mask_train: np.ndarray = None,
    output_dir: Path = None,
) -> tuple[np.ndarray, list[str]]:
    """Impute NaN values with training-set medians.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix with possible NaN values.
    feature_names : list[str]
        Feature names.
    split_mask_train : np.ndarray, optional
        Boolean mask for training rows. If None, uses all rows.
    output_dir : Path, optional
        Where to save imputer medians.

    Returns
    -------
    X_imputed : np.ndarray
        No NaN values.
    feature_names : list[str]
        Same as input (no extra columns needed since _missing indicators
        are already computed during extraction).
    """
    if output_dir is None:
        output_dir = PROCESSED_DIR

    X_out = X.copy()

    # Compute medians from training set only
    if split_mask_train is not None:
        X_train = X[split_mask_train]
    else:
        X_train = X

    medians = {}
    for j, name in enumerate(feature_names):
        col = X_train[:, j]
        valid = col[~np.isnan(col)]
        median_val = float(np.median(valid)) if len(valid) > 0 else 0.0
        medians[name] = median_val

        # Impute NaNs
        nan_mask = np.isnan(X_out[:, j])
        X_out[nan_mask, j] = median_val

    # Save medians
    medians_path = output_dir / "imputer_medians.json"
    with open(medians_path, "w") as f:
        json.dump(medians, f, indent=2)

    return X_out, feature_names
