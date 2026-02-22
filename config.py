"""
CCPFS Configuration
-------------------
Central configuration for paths, cost parameters, and defaults.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MEDS_DIR = DATA_DIR / "meds"
SYNTHEA_DIR = DATA_DIR / "synthea"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"

# MEDS dataset (at project root, produced by MIMIC-IV MEDS ETL)
MEDS_BASE_DIR = PROJECT_ROOT.parent / "data" / "meds" / "MEDS_cohort"
MEDS_DATA_DIR = MEDS_BASE_DIR / "data"
MEDS_METADATA_DIR = MEDS_BASE_DIR / "metadata"

# ---------------------------------------------------------------------------
# Clinical cost parameters (EUR, literature-derived)
# ---------------------------------------------------------------------------
C_EVENT = 10_000    # Cost of adverse event (HF readmission)
C_VISIT = 150       # Cost of outpatient follow-up visit

# Derived: unconstrained cost-optimal threshold
# tau* = C_VISIT / C_EVENT = 0.015 (1.5%)
COST_THRESHOLD = C_VISIT / C_EVENT

# ---------------------------------------------------------------------------
# Scheduling defaults
# ---------------------------------------------------------------------------
HORIZON_DAYS = 30           # Prediction and scheduling horizon
DEFAULT_DAILY_CAPACITY = 10 # Slots per day (adjustable in sensitivity analysis)
UNCERTAINTY_ALPHA = 0.8     # Conservative scheduling parameter (0 = ignore uncertainty)

# ---------------------------------------------------------------------------
# Cohort
# ---------------------------------------------------------------------------
MIN_AGE = 18
MIN_LOS_DAYS = 1            # Exclude observation stays < 1 day
HF_ICD10_PREFIX = "I50"     # Heart failure ICD-10 codes
HF_ICD9_PREFIX = "428"      # Heart failure ICD-9 codes

ELIGIBLE_DISCHARGE_LOCATIONS = {"HOME", "HOME HEALTH CARE"}

# ---------------------------------------------------------------------------
# Specialty pools (diagnosis-based follow-up clinic assignment)
# ---------------------------------------------------------------------------
SPECIALTY_NAMES = ["cardiology", "neurology", "surgery", "general_medicine"]
N_SPECIALTIES = len(SPECIALTY_NAMES)
SPECIALTY_CARDIOLOGY = 0
SPECIALTY_NEUROLOGY = 1
SPECIALTY_SURGERY = 2
SPECIALTY_GENERAL_MEDICINE = 3

# ICD-10 prefix sets for specialty assignment
CARDIOLOGY_ICD10 = {f"I{i}" for i in range(20, 53)}  # I20-I52
NEUROLOGY_ICD10 = {f"I{i}" for i in range(60, 70)} | {"G"}  # I60-I69 + G*

# Default per-specialty daily capacity (realistic for ~500-bed hospital)
DEFAULT_SPECIALTY_CAPACITY = {
    SPECIALTY_CARDIOLOGY: 15,
    SPECIALTY_NEUROLOGY: 10,
    SPECIALTY_SURGERY: 15,
    SPECIALTY_GENERAL_MEDICINE: 25,
}

# ---------------------------------------------------------------------------
# Target labs (MIMIC-IV labevents itemids)
# ---------------------------------------------------------------------------
TARGET_LABS = {
    "creatinine": "50912",
    "bun": "51006",
    "sodium": "50983",
    "potassium": "50971",
    "hemoglobin": "51222",
    "wbc": "51301",
    "bnp": "50963",
}

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
RANDOM_SEED = 42
N_BOOTSTRAP = 20            # For uncertainty estimation

GBM_PARAM_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "min_samples_leaf": [20, 50],
    "subsample": [0.8],
}
