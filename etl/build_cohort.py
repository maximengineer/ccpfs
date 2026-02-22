"""
Cohort Builder
--------------
Extracts eligible discharge episodes from MEDS-format MIMIC-IV data.

Streaming architecture: processes one parquet shard at a time (~50 MB each
after column projection) to stay within 16 GB RAM.

Inclusion criteria:
  - Adult patients (age >= 18)
  - Discharged alive to home or home health care
  - Length of stay >= 1 day

Target variable:
  - time_to_readmission: days from discharge to next unplanned admission
  - event_indicator: 1 if readmitted within 30 days, 0 if censored
"""

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from config import (
    CARDIOLOGY_ICD10,
    ELIGIBLE_DISCHARGE_LOCATIONS,
    HF_ICD10_PREFIX,
    HF_ICD9_PREFIX,
    HORIZON_DAYS,
    MEDS_DATA_DIR,
    MEDS_METADATA_DIR,
    MIN_AGE,
    MIN_LOS_DAYS,
    NEUROLOGY_ICD10,
    PROCESSED_DIR,
    SPECIALTY_CARDIOLOGY,
    SPECIALTY_GENERAL_MEDICINE,
    SPECIALTY_NAMES,
    SPECIALTY_NEUROLOGY,
    SPECIALTY_SURGERY,
)

# Columns we actually need from each shard (skip the 19+ unused ones)
_SHARD_COLUMNS = ["subject_id", "time", "code", "numeric_value", "hadm_id"]


def build_cohort(
    meds_data_dir: Path = MEDS_DATA_DIR,
    output_path: Optional[Path] = None,
    horizon_days: int = HORIZON_DAYS,
    verbose: bool = True,
) -> pl.DataFrame:
    """Build the full cohort from all MEDS shards.

    Parameters
    ----------
    meds_data_dir : Path
        Directory containing train/, tuning/, held_out/ subdirectories.
    output_path : Path, optional
        Where to save cohort.parquet. Defaults to PROCESSED_DIR / cohort.parquet.
    horizon_days : int
        Readmission horizon for censoring.
    verbose : bool
        Print progress.

    Returns
    -------
    pl.DataFrame
        One row per eligible discharge episode.
    """
    if output_path is None:
        output_path = PROCESSED_DIR / "cohort.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load subject splits
    splits_df = pl.read_parquet(MEDS_METADATA_DIR / "subject_splits.parquet")

    cohort_fragments = []
    for split_name in ["train", "tuning", "held_out"]:
        split_dir = meds_data_dir / split_name
        if not split_dir.exists():
            continue

        shard_files = sorted(split_dir.glob("*.parquet"))
        for idx, shard_path in enumerate(shard_files):
            if verbose and idx % 50 == 0:
                print(f"  [{split_name}] Processing shard {idx}/{len(shard_files)}...")

            fragment = _process_shard(shard_path, horizon_days, split_name)
            if len(fragment) > 0:
                cohort_fragments.append(fragment)

    cohort = pl.concat(cohort_fragments)

    # Join subject splits for train/val/test labels
    cohort = cohort.join(
        splits_df.rename({"split": "subject_split"}),
        on="subject_id",
        how="left",
    )

    if verbose:
        n = len(cohort)
        n_events = cohort["event_indicator"].sum()
        print(f"\nCohort built: {n:,} discharge episodes")
        print(f"  Readmission rate: {n_events / n:.1%}")
        print(f"  Specialty distribution:")
        spec_counts = cohort["specialty_pool"].value_counts().sort("specialty_pool")
        for row in spec_counts.iter_rows():
            pool_id, count = row
            print(f"    {SPECIALTY_NAMES[pool_id]}: {count:,}")

    cohort.write_parquet(output_path)
    if verbose:
        print(f"  Saved to {output_path}")

    return cohort


def _process_shard(
    shard_path: Path,
    horizon_days: int,
    split_name: str,
) -> pl.DataFrame:
    """Process a single MEDS shard and return eligible cohort rows."""
    df = pl.scan_parquet(shard_path).select(_SHARD_COLUMNS).collect()

    if len(df) == 0:
        return pl.DataFrame()

    # --- Extract demographics ---
    births = (
        df.filter(pl.col("code") == "MEDS_BIRTH")
        .select("subject_id", pl.col("time").dt.year().alias("birth_year"))
        .unique(subset=["subject_id"])
    )

    genders = (
        df.filter(pl.col("code").str.starts_with("GENDER//"))
        .select(
            "subject_id",
            pl.col("code").str.replace("GENDER//", "").alias("gender"),
        )
        .unique(subset=["subject_id"])
    )

    # --- Extract admissions ---
    admissions = (
        df.filter(pl.col("code").str.starts_with("HOSPITAL_ADMISSION//"))
        .select(
            "subject_id",
            "hadm_id",
            pl.col("time").alias("admission_time"),
            pl.col("code")
            .str.split("//")
            .list.get(1)
            .alias("admission_type"),
        )
        .drop_nulls(subset=["hadm_id"])
    )

    # --- Extract discharges ---
    discharges = (
        df.filter(pl.col("code").str.starts_with("HOSPITAL_DISCHARGE//"))
        .select(
            "subject_id",
            "hadm_id",
            pl.col("time").alias("discharge_time"),
            pl.col("code")
            .str.split("//")
            .list.get(1)
            .alias("discharge_location"),
        )
        .drop_nulls(subset=["hadm_id"])
    )

    if len(admissions) == 0 or len(discharges) == 0:
        return pl.DataFrame()

    # --- Build episodes: join admission to discharge ---
    episodes = admissions.join(discharges, on=["subject_id", "hadm_id"], how="inner")

    # --- Filter eligible ---
    episodes = episodes.filter(
        pl.col("discharge_location").is_in(ELIGIBLE_DISCHARGE_LOCATIONS)
    )

    if len(episodes) == 0:
        return pl.DataFrame()

    # Add demographics
    episodes = episodes.join(births, on="subject_id", how="left")
    episodes = episodes.join(genders, on="subject_id", how="left")

    # Compute age and LOS
    episodes = episodes.with_columns(
        (pl.col("discharge_time").dt.year() - pl.col("birth_year")).alias("age"),
        (
            (pl.col("discharge_time") - pl.col("admission_time")).dt.total_hours()
            / 24.0
        ).alias("los_days"),
    )

    # Apply age and LOS filters
    episodes = episodes.filter(
        (pl.col("age") >= MIN_AGE) & (pl.col("los_days") >= MIN_LOS_DAYS)
    )

    if len(episodes) == 0:
        return pl.DataFrame()

    # --- Calculate time to readmission ---
    # For each discharge, find the next admission for the same subject
    # Sort all admissions by subject and time
    all_admissions_sorted = (
        admissions.select("subject_id", "admission_time")
        .sort("subject_id", "admission_time")
    )

    episodes = _compute_readmission_times(
        episodes, all_admissions_sorted, horizon_days
    )

    # --- Assign specialty pool and HF flag ---
    diagnoses = df.filter(
        pl.col("code").str.starts_with("DIAGNOSIS//ICD//")
    ).select("hadm_id", "code").drop_nulls(subset=["hadm_id"])

    # Extract admitting service (e.g., TRANSFER_TO//admit//Medicine)
    admit_services = (
        df.filter(pl.col("code").str.starts_with("TRANSFER_TO//admit//"))
        .select(
            "hadm_id",
            pl.col("code")
            .str.replace("TRANSFER_TO//admit//", "")
            .alias("admit_service"),
        )
        .drop_nulls(subset=["hadm_id"])
        .unique(subset=["hadm_id"], keep="first")
    )

    episodes = _assign_specialty_and_hf(episodes, diagnoses, admit_services)

    # Add split and shard info
    episodes = episodes.with_columns(
        pl.lit(split_name).alias("data_split"),
        pl.lit(str(shard_path)).alias("shard_path"),
    )

    return episodes.select(
        "subject_id",
        "hadm_id",
        "admission_time",
        "discharge_time",
        "admission_type",
        "discharge_location",
        "gender",
        "age",
        "los_days",
        "time_to_readmission",
        "event_indicator",
        "specialty_pool",
        "is_heart_failure",
        "data_split",
        "shard_path",
    )


def _compute_readmission_times(
    episodes: pl.DataFrame,
    all_admissions: pl.DataFrame,
    horizon_days: int,
) -> pl.DataFrame:
    """For each discharge episode, find time to next admission."""
    results = []

    # Group admissions by subject for fast lookup
    admission_times_by_subject = {}
    for row in all_admissions.iter_rows(named=True):
        sid = row["subject_id"]
        if sid not in admission_times_by_subject:
            admission_times_by_subject[sid] = []
        admission_times_by_subject[sid].append(row["admission_time"])

    for row in episodes.iter_rows(named=True):
        sid = row["subject_id"]
        disch_time = row["discharge_time"]

        # Find next admission after this discharge
        future_admissions = [
            t for t in admission_times_by_subject.get(sid, [])
            if t > disch_time
        ]

        if future_admissions:
            next_admit = min(future_admissions)
            days_to_readmit = (next_admit - disch_time).total_seconds() / 86400.0
            if days_to_readmit <= horizon_days:
                results.append(
                    {"_idx": row["hadm_id"], "time_to_readmission": days_to_readmit, "event_indicator": 1}
                )
            else:
                results.append(
                    {"_idx": row["hadm_id"], "time_to_readmission": float(horizon_days), "event_indicator": 0}
                )
        else:
            results.append(
                {"_idx": row["hadm_id"], "time_to_readmission": float(horizon_days), "event_indicator": 0}
            )

    readmit_df = pl.DataFrame(results).rename({"_idx": "hadm_id"})
    return episodes.join(readmit_df, on="hadm_id", how="left")


def _assign_specialty_and_hf(
    episodes: pl.DataFrame,
    diagnoses: pl.DataFrame,
    admit_services: pl.DataFrame,
) -> pl.DataFrame:
    """Assign specialty pool (from admitting service + ICD) and HF flag."""
    hadm_ids = episodes["hadm_id"].unique().to_list()

    # Build lookup dicts
    diag_by_hadm = {}
    for row in diagnoses.iter_rows(named=True):
        hid = row["hadm_id"]
        if hid not in diag_by_hadm:
            diag_by_hadm[hid] = []
        diag_by_hadm[hid].append(row["code"])

    service_by_hadm = {}
    for row in admit_services.iter_rows(named=True):
        service_by_hadm[row["hadm_id"]] = row["admit_service"]

    specialty_map = {}
    hf_map = {}

    for hid in hadm_ids:
        codes = diag_by_hadm.get(hid, [])
        service = service_by_hadm.get(hid, "")
        specialty_map[hid] = _classify_specialty(codes, service)
        hf_map[hid] = _has_heart_failure(codes)

    spec_df = pl.DataFrame(
        {
            "hadm_id": list(specialty_map.keys()),
            "specialty_pool": list(specialty_map.values()),
            "is_heart_failure": [hf_map[k] for k in specialty_map.keys()],
        }
    )

    return episodes.join(spec_df, on="hadm_id", how="left").with_columns(
        pl.col("specialty_pool").fill_null(SPECIALTY_GENERAL_MEDICINE),
        pl.col("is_heart_failure").fill_null(False),
    )


# Admitting services that indicate surgical care
_SURGICAL_SERVICES = {
    "Surgery",
    "Surgery/Trauma",
    "Surgery/Pancreatic/Biliary/Bariatric",
    "Surgery/Vascular/Intermediate",
    "Cardiac Surgery",
    "Thoracic Surgery",
    "Surgical Intensive Care Unit (SICU)",
    "Trauma SICU (TSICU)",
    "Surgical Intermediate",
    "Med/Surg/Trauma",
    "PACU",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)",
}

# Admitting services that indicate cardiology care
_CARDIOLOGY_SERVICES = {
    "Medicine/Cardiology",
    "Medicine/Cardiology Intermediate",
    "Coronary Care Unit (CCU)",
    "Cardiac Vascular Intensive Care Unit (CVICU)",
    "Cardiology",
    "Cardiology Surgery Intermediate",
}

# Admitting services that indicate neurology care
_NEUROLOGY_SERVICES = {
    "Neurology",
    "Neuro Intermediate",
    "Neuro Stepdown",
}


def _classify_specialty(diagnosis_codes: list[str], admit_service: str) -> int:
    """Assign specialty pool using admitting service + ICD diagnosis.

    Strategy:
      1. If admitting service is explicitly surgical → Surgery
      2. If admitting service is explicitly cardiology → Cardiology
      3. If admitting service is explicitly neurology → Neurology
      4. For general services (Medicine, Med/Surg, ED), use ICD diagnosis:
         - Cardiovascular ICD (I20-I52) → Cardiology
         - Neurological ICD (G*, I60-I69) → Neurology
      5. Default → General Medicine
    """
    # Service-based classification (most reliable)
    if admit_service in _SURGICAL_SERVICES:
        return SPECIALTY_SURGERY
    if admit_service in _CARDIOLOGY_SERVICES:
        return SPECIALTY_CARDIOLOGY
    if admit_service in _NEUROLOGY_SERVICES:
        return SPECIALTY_NEUROLOGY

    # For general/ambiguous services, fall back to ICD diagnosis
    for code in diagnosis_codes:
        icd_root = _extract_icd_root(code)
        if icd_root is None:
            continue
        if icd_root in CARDIOLOGY_ICD10:
            return SPECIALTY_CARDIOLOGY
        if icd_root in NEUROLOGY_ICD10 or (len(icd_root) >= 1 and icd_root[0] == "G"):
            return SPECIALTY_NEUROLOGY

    return SPECIALTY_GENERAL_MEDICINE


def _has_heart_failure(diagnosis_codes: list[str]) -> bool:
    """Check if any diagnosis is heart failure."""
    for code in diagnosis_codes:
        # DIAGNOSIS//ICD//10//I50xxx or DIAGNOSIS//ICD//9//428xx
        parts = code.split("//")
        if len(parts) >= 4:
            icd_code = parts[3]
            if icd_code.startswith(HF_ICD10_PREFIX) or icd_code.startswith(
                HF_ICD9_PREFIX
            ):
                return True
    return False


def _extract_icd_root(code: str) -> Optional[str]:
    """Extract 3-character ICD root from MEDS diagnosis code.

    'DIAGNOSIS//ICD//10//I5032' -> 'I50'
    'DIAGNOSIS//ICD//9//42821'  -> '428'
    """
    parts = code.split("//")
    if len(parts) >= 4 and parts[0] == "DIAGNOSIS":
        icd_code = parts[3]
        return icd_code[:3] if len(icd_code) >= 3 else None
    return None


if __name__ == "__main__":
    import sys
    import time

    start = time.time()
    print("Building cohort from MEDS data...")
    cohort = build_cohort(verbose=True)
    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.0f} seconds")
    sys.exit(0)
