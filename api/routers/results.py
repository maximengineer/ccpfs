"""GET /api/results - serve pre-computed pipeline results."""

import json
import os
from pathlib import Path

from fastapi import APIRouter

from api.dependencies import POLICY_META, store
from config import SPECIALTY_NAMES
from api.schemas import (
    CohortSummary,
    ModelInfo,
    PipelineResults,
    PolicyResult,
)

router = APIRouter()


@router.get("/results", response_model=PipelineResults)
def get_results():
    data_dir = Path(os.environ.get("CCPFS_DATA_DIR", "data/processed"))
    pr = store.pipeline_results
    sr = pr["scheduling_results"]
    mp = pr["model_performance"]

    # Cohort summary
    n_test = store.curves.shape[0]
    n_events = int(store.e_test.sum())
    specialty_pools = store.cohort_test["specialty_pool"].to_numpy()
    specialty_pcts = {}
    for idx, name in enumerate(SPECIALTY_NAMES):
        specialty_pcts[name] = round(float((specialty_pools == idx).mean()) * 100, 1)

    cohort = CohortSummary(
        total_episodes=pr["cohort_size"],
        test_episodes=n_test,
        event_rate=round(n_events / n_test, 3),
        specialties=specialty_pcts,
    )

    # Models
    models = []
    if "gbm_metrics" in mp:
        models.append(ModelInfo(
            name="GBM",
            c_index=round(mp["gbm_metrics"]["c_index"], 4),
            ibs=round(mp["gbm_metrics"]["ibs"], 4),
            params=mp.get("gbm_params"),
        ))
    if "cox_metrics" in mp:
        models.append(ModelInfo(
            name="Cox PH",
            c_index=round(mp["cox_metrics"]["c_index"], 4),
            ibs=round(mp["cox_metrics"]["ibs"], 4),
        ))
    if "rsf_metrics" in mp:
        models.append(ModelInfo(
            name="RSF",
            c_index=round(mp["rsf_metrics"]["c_index"], 4),
            ibs=round(mp["rsf_metrics"]["ibs"], 4),
            params=mp.get("rsf_params"),
        ))
    # MOTOR results from motor_result.json
    motor_path = data_dir / "motor_output" / "motor_result.json"
    if motor_path.exists():
        with open(motor_path) as f:
            motor = json.load(f)
        models.append(ModelInfo(
            name="MOTOR+GBM",
            c_index=round(motor["c_index"], 4),
            ibs=round(motor["ibs"], 4),
            params=motor.get("params"),
        ))

    # Policies
    uniform_cost = sr["uniform_d14"]["total_expected_cost"] / n_test
    policies = []
    for prefix, display_name, cap_aware, feasible in POLICY_META:
        if prefix not in sr:
            continue
        policy_data = sr[prefix]
        avg_cost = policy_data["total_expected_cost"] / n_test
        vs_uniform = ((avg_cost - uniform_cost) / uniform_cost) * 100
        ebf = policy_data.get("ebf", {})

        policies.append(PolicyResult(
            name=display_name,
            capacity_aware=cap_aware,
            feasible=feasible,
            avg_cost=round(avg_cost, 0),
            vs_uniform_pct=round(vs_uniform, 1),
            catch_rate=round(ebf.get("catch_rate", 0) * 100, 1),
            ebf_rate=round(ebf.get("ebf_rate", 0) * 100, 1),
        ))

    return PipelineResults(cohort=cohort, models=models, policies=policies)
