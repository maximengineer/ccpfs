"""Singleton data loader - loads all pre-computed artifacts at startup."""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

# Ensure project root is importable (for config, policy, evaluation)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SPECIALTY_NAMES

logger = logging.getLogger(__name__)

# Policy display metadata: (key_prefix, display_name, capacity_aware, feasible)
POLICY_META = [
    ("guideline", "Guideline (ACC/AHA)", False, True),
    ("uniform_d14", "Uniform day 14", False, True),
    ("risk_bucket", "Risk bucket", False, True),
    ("uniform_d14_cap", "Uniform-14 (capacity)", True, False),
    ("guideline_cap", "Guideline (capacity)", True, False),
    ("greedy_global", "Greedy (global)", True, True),
    ("greedy_specialty", "Greedy (specialty)", True, True),
    ("mincost_global", "MinCost (global)", True, True),
    ("mincost_specialty", "MinCost (specialty)", True, True),
    ("unconstrained", "Unconstrained (oracle)", False, True),
]


class DataStore:
    """Holds all pre-computed data in memory for fast serving."""

    def __init__(self):
        self.curves: np.ndarray | None = None          # (N, 31)
        self.e_test: np.ndarray | None = None           # (N,)
        self.t_test: np.ndarray | None = None           # (N,)
        self.cohort_test: pl.DataFrame | None = None    # test split only
        self.pipeline_results: dict | None = None
        self.models_info: dict | None = None
        self.scheduling: dict | None = None             # all policy assignments
        self.loaded: bool = False
        self.models_loaded: list[str] = []

    def load(self):
        data_dir = Path(os.environ.get("CCPFS_DATA_DIR", "data/processed"))
        model_dir = Path(os.environ.get("CCPFS_MODEL_DIR", "models/saved"))

        try:
            # Survival curves (N x 31)
            curves_data = np.load(data_dir / "curves_test.npz")
            self.curves = curves_data["curves_test"]
            self.e_test = curves_data["e_test"]
            self.t_test = curves_data["t_test"]

            # Cohort metadata - test split only (preserve original row order
            # to match positional indices in curves/scheduling arrays)
            cohort_full = pl.read_parquet(data_dir / "cohort.parquet")
            self.cohort_test = cohort_full.filter(
                pl.col("data_split") == "held_out"
            )

            # Verify alignment: cohort rows == curve rows
            if len(self.cohort_test) != self.curves.shape[0]:
                logger.warning(
                    f"Cohort/curve size mismatch: {len(self.cohort_test)} vs {self.curves.shape[0]}"
                )

            # Pipeline results
            with open(data_dir / "pipeline_results.json") as f:
                self.pipeline_results = json.load(f)

            # Models info
            models_info_path = data_dir / "models_info.json"
            if models_info_path.exists():
                with open(models_info_path) as f:
                    self.models_info = json.load(f)

            # Scheduling results (all policy assignment arrays)
            sched_path = data_dir / "scheduling_results.npz"
            if sched_path.exists():
                sched = np.load(sched_path, allow_pickle=True)
                self.scheduling = {}
                for key in POLICY_META:
                    prefix = key[0]
                    days_key = f"{prefix}_days"
                    if days_key in sched:
                        self.scheduling[prefix] = sched[days_key]

            # Check which models are available
            if (model_dir / "gbm_survival.joblib").exists():
                self.models_loaded.append("gbm")
            if (model_dir / "cox_ph.pkl").exists():
                self.models_loaded.append("cox")
            if (model_dir / "rsf_survival.joblib").exists():
                self.models_loaded.append("rsf")

            self.loaded = True
            n = self.curves.shape[0]
            logger.info(f"DataStore loaded: {n} patients, {len(self.scheduling)} policies, models: {self.models_loaded}")

        except FileNotFoundError as e:
            logger.error(f"Required data file missing: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise


# Global singleton
store = DataStore()
