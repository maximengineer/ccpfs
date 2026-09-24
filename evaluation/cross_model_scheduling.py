#!/usr/bin/env python3
"""
Cross-Model Scheduling Comparison
----------------------------------
Runs MinCost (specialty) on every survival model's calibrated test curves
(Cox PH, GBM, RSF, MOTOR+GBM) under the same proportional capacity as the
main pipeline. Produces the paper's cross-model table (Table III).

Reads (never overwrites) the main pipeline outputs:
  data/processed/parallel_tmp/{cox,gbm,rsf}_curves.npz  uncalibrated test curves
  models/saved/*                                        fitted classical models
  data/processed/motor_output/*                         MOTOR embeddings + GBM

Each model is calibrated on its own validation-split curves, exactly as
run_pipeline.py::step_calibrate does for the best model. Calibrated curves
are cached in data/processed/cross_model/ so re-runs skip curve extraction.
Each model writes its own {model}_result.json; models can run in parallel.

Usage:
    PYTHONPATH=. python evaluation/cross_model_scheduling.py
    PYTHONPATH=. python evaluation/cross_model_scheduling.py --models gbm,cox
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import C_EVENT, C_VISIT, HORIZON_DAYS, MODEL_DIR, PROCESSED_DIR
from evaluation.metrics import event_before_followup_rate
from models.calibrate import apply_calibration, calibrate_curves
from policy.baselines import uniform_policy
from policy.mincost_solver import schedule_mincost_specialty
from policy.specialty_scheduler import proportional_specialty_capacity

ALL_MODELS = ["cox", "gbm", "rsf", "motor"]
OUT_DIR = PROCESSED_DIR / "cross_model"
MOTOR_DIR = PROCESSED_DIR / "motor_output"


def _val_curves_classical(name, X_val, feature_names):
    """Validation-split curves from a saved classical model."""
    if name == "gbm":
        from models.classical.train_gbm import extract_survival_curves, load_model
        return extract_survival_curves(load_model(), X_val)
    if name == "cox":
        from models.classical.train_cox import extract_survival_curves_cox, load_model
        model, scaler = load_model()
        return extract_survival_curves_cox(model, X_val, feature_names, scaler=scaler)
    if name == "rsf":
        from models.classical.train_rsf import extract_survival_curves_rsf, load_model
        return extract_survival_curves_rsf(load_model(), X_val)
    raise ValueError(name)


def _motor_curves():
    """MOTOR+GBM (val, test) curves: embeddings -> scaler -> PCA -> GBM."""
    import joblib
    from models.motor.train_on_embeddings import extract_survival_curves

    emb = np.load(MOTOR_DIR / "aligned_embeddings.npz")
    scaler = joblib.load(MOTOR_DIR / "motor_scaler.joblib")
    pca = joblib.load(MOTOR_DIR / "motor_pca.joblib")
    gbm = joblib.load(MOTOR_DIR / "motor_gbm.joblib")

    def predict(X):
        return extract_survival_curves(gbm, pca.transform(scaler.transform(X)))

    test = predict(emb["X_test"])
    ref = np.load(MOTOR_DIR / "motor_curves.npz")["curves"]
    if not np.allclose(test, ref, atol=1e-9):
        raise RuntimeError("Reconstructed MOTOR test curves differ from motor_curves.npz")
    return predict(emb["X_val"]), test, emb["e_val"], emb["t_val"]


def calibrated_test_curves(name, cohort, X_val, feature_names):
    """Calibrated test curves for one model, cached on disk."""
    cache = OUT_DIR / f"{name}_calibrated.npz"
    if cache.exists():
        return np.load(cache)["curves_test"]

    t0 = time.time()
    if name == "motor":
        val_curves, test_curves, e_val, t_val = _motor_curves()
    else:
        val_mask = (cohort["data_split"] == "tuning").to_numpy()
        e_val = cohort["event_indicator"].to_numpy()[val_mask]
        t_val = cohort["time_to_readmission"].to_numpy()[val_mask]
        test_curves = np.load(PROCESSED_DIR / "parallel_tmp" / f"{name}_curves.npz")["curves"]
        val_curves = _val_curves_classical(name, X_val, feature_names)

    _, calibrators = calibrate_curves(val_curves, e_val, t_val)
    cal = apply_calibration(test_curves, calibrators)
    np.savez_compressed(cache, curves_test=cal)
    print(f"  {name}: calibrated test curves in {time.time() - t0:.0f}s")
    return cal


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--models", default=",".join(ALL_MODELS))
    args = parser.parse_args()
    models = [m.strip().lower() for m in args.models.split(",")]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cohort = pl.read_parquet(PROCESSED_DIR / "cohort.parquet")
    test_mask = (cohort["data_split"] == "held_out").to_numpy()
    test = cohort.filter(pl.Series(test_mask))
    pools = test["specialty_pool"].to_numpy()
    e_test = test["event_indicator"].to_numpy()
    t_test = test["time_to_readmission"].to_numpy()
    n = len(test)

    X_val, feature_names = None, None
    if {"cox", "gbm", "rsf"} & set(models):
        feats = np.load(PROCESSED_DIR / "features.npz", allow_pickle=True)
        X_val = feats["X"][(cohort["data_split"] == "tuning").to_numpy()]
        feature_names = list(feats["feature_names"])

    info = json.load(open(PROCESSED_DIR / "models_info.json"))
    motor_info = json.load(open(MOTOR_DIR / "motor_result.json"))
    cindex = {m: info[f"{m}_metrics"]["c_index"] for m in ("cox", "gbm", "rsf")}
    cindex["motor"] = motor_info["c_index"]

    capacity = proportional_specialty_capacity(pools, HORIZON_DAYS)
    print(f"Capacity per day: {capacity}")

    for name in models:
        curves = calibrated_test_curves(name, cohort, X_val, feature_names)
        if name == "gbm":
            main_curves = np.load(PROCESSED_DIR / "curves_test.npz")["curves_test"]
            diff = float(np.abs(curves - main_curves).max())
            print(f"  gbm: max |calibrated - curves_test.npz| = {diff:.2e}")

        t0 = time.time()
        r = schedule_mincost_specialty(curves, pools, capacity_per_specialty_day=capacity)
        elapsed = time.time() - t0
        u14 = uniform_policy(curves, day=14)
        ebf = event_before_followup_rate(r["assignments"], t_test, e_test)
        ebf_u14 = event_before_followup_rate(u14["assignments"], t_test, e_test)

        result = {
            "c_index": cindex[name],
            "status": r["status"],
            "avg_cost": r["total_expected_cost"] / n,
            "avg_cost_uniform_d14": u14["total_expected_cost"] / n,
            "catch_rate": ebf["catch_rate"],
            "catch_rate_uniform_d14": ebf_u14["catch_rate"],
            "solve_seconds": elapsed,
        }
        print(f"  {name}: {r['status']}, EUR {result['avg_cost']:.0f} "
              f"(U14 EUR {result['avg_cost_uniform_d14']:.0f}), "
              f"catch {ebf['catch_rate']:.1%}, {elapsed:.0f}s")

        # One file per model so several models can run in parallel processes
        with open(OUT_DIR / f"{name}_result.json", "w") as f:
            json.dump(result, f, indent=2)

    # Merge every model finished so far into one table
    results = {m: json.load(open(OUT_DIR / f"{m}_result.json"))
               for m in ALL_MODELS if (OUT_DIR / f"{m}_result.json").exists()}
    results_path = OUT_DIR / "cross_model_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {results_path} ({', '.join(results)})")


if __name__ == "__main__":
    main()
