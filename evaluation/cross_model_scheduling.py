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
run_pipeline.py::step_calibrate does for the best model. Before calibrating,
the script checks that the inputs belong together: each saved classical model
must reproduce its parallel_tmp test curves, MOTOR's aligned embeddings must
follow the cohort's test order, and the re-calibrated GBM curves must equal
curves_test.npz. Any mismatch stops the run.

Calibrated curves are cached in data/processed/cross_model/ together with a
fingerprint (size + mtime) of their inputs, and are recomputed when an input
changes. Each model writes {model}_result.json and {model}_days.npy (the
assigned day per test patient, used by evaluation/bootstrap_ci.py); models can
run in parallel processes.

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

MODEL_FILES = {
    "gbm": MODEL_DIR / "gbm_survival.joblib",
    "cox": MODEL_DIR / "cox_ph.pkl",
    "rsf": MODEL_DIR / "rsf_survival.joblib",
}
N_REPRO_CHECK = 200  # test rows re-predicted to confirm model and curves match
REPRO_ATOL = 1e-6


def _classical_predictor(name, feature_names):
    """Return X -> S(t) curves for a saved classical model."""
    if name == "gbm":
        from models.classical.train_gbm import extract_survival_curves, load_model
        model = load_model()
        return lambda X: extract_survival_curves(model, X)
    if name == "cox":
        from models.classical.train_cox import extract_survival_curves_cox, load_model
        model, scaler = load_model()
        return lambda X: extract_survival_curves_cox(model, X, feature_names, scaler=scaler)
    if name == "rsf":
        from models.classical.train_rsf import extract_survival_curves_rsf, load_model
        model = load_model()
        return lambda X: extract_survival_curves_rsf(model, X)
    raise ValueError(name)


def _classical_curves(name, ctx):
    """(val, test) curves for a classical model, after checking that the saved
    model reproduces the parallel_tmp test curves it is paired with."""
    test = np.load(PROCESSED_DIR / "parallel_tmp" / f"{name}_curves.npz")["curves"]
    if len(test) != ctx["n_test"]:
        raise RuntimeError(f"{name}: parallel_tmp has {len(test)} test curves, cohort has {ctx['n_test']}")

    predict = _classical_predictor(name, ctx["feature_names"])
    k = min(N_REPRO_CHECK, len(test))
    diff = float(np.abs(predict(ctx["X_test"][:k]) - test[:k]).max())
    if diff > REPRO_ATOL:
        raise RuntimeError(
            f"{name}: saved model does not reproduce parallel_tmp/{name}_curves.npz "
            f"(max diff {diff:.2e} on {k} test rows). The model was probably retrained after "
            f"parallel_train.py. Do not rerun parallel_train.py (it retrains every model and "
            f"overwrites curves_test.npz); see paper/mimic_followup.md section 5."
        )
    print(f"  {name}: saved model reproduces parallel_tmp curves (max diff {diff:.1e})")
    return predict(ctx["X_val"]), test


def _motor_curves(ctx):
    """MOTOR+GBM (val, test) curves: embeddings -> scaler -> PCA -> GBM."""
    import joblib
    from models.motor.train_on_embeddings import extract_survival_curves

    emb = np.load(MOTOR_DIR / "aligned_embeddings.npz")
    # Aligned rows must be the cohort's test rows in the same order, or the
    # catch rate and pools below would be matched to the wrong patients
    if len(emb["e_test"]) != ctx["n_test"] or not (
        np.array_equal(emb["e_test"], ctx["e_test"]) and np.allclose(emb["t_test"], ctx["t_test"])
    ):
        raise RuntimeError("MOTOR aligned test rows do not match the cohort's test split order")

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


def _input_files(name):
    """Files a model's calibrated curves are derived from."""
    if name == "motor":
        return [MOTOR_DIR / f for f in ("aligned_embeddings.npz", "motor_scaler.joblib",
                                        "motor_pca.joblib", "motor_gbm.joblib", "motor_curves.npz")]
    return [PROCESSED_DIR / "parallel_tmp" / f"{name}_curves.npz", MODEL_FILES[name],
            PROCESSED_DIR / "features.npz", PROCESSED_DIR / "cohort.parquet"]


def _fingerprint(paths):
    """Size and mtime of each input, so a stale cache is detected."""
    out = {}
    for p in paths:
        st = Path(p).stat()
        out[str(p)] = [st.st_size, st.st_mtime_ns]
    return json.dumps(out, sort_keys=True)


def calibrated_test_curves(name, ctx):
    """Calibrated test curves for one model, cached on disk."""
    cache = OUT_DIR / f"{name}_calibrated.npz"
    fingerprint = _fingerprint(_input_files(name))
    if cache.exists():
        cached = np.load(cache)
        if "inputs" in cached and str(cached["inputs"]) == fingerprint:
            return cached["curves_test"]
        print(f"  {name}: cache is stale or has no input fingerprint, recomputing")

    t0 = time.time()
    if name == "motor":
        val_curves, test_curves, e_val, t_val = _motor_curves(ctx)
    else:
        val_curves, test_curves = _classical_curves(name, ctx)
        e_val, t_val = ctx["e_val"], ctx["t_val"]

    _, calibrators = calibrate_curves(val_curves, e_val, t_val)
    cal = apply_calibration(test_curves, calibrators)
    np.savez_compressed(cache, curves_test=cal, inputs=np.array(fingerprint))
    print(f"  {name}: calibrated test curves in {time.time() - t0:.0f}s")
    return cal


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--models", default=",".join(ALL_MODELS))
    args = parser.parse_args()
    models = [m.strip().lower() for m in args.models.split(",")]
    unknown = set(models) - set(ALL_MODELS)
    if unknown:
        parser.error(f"unknown models {sorted(unknown)}; choose from {ALL_MODELS}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cohort = pl.read_parquet(PROCESSED_DIR / "cohort.parquet")
    split = cohort["data_split"].to_numpy()
    test_mask, val_mask = split == "held_out", split == "tuning"
    test = cohort.filter(pl.Series(test_mask))
    pools = test["specialty_pool"].to_numpy()
    e_test = test["event_indicator"].to_numpy()
    t_test = test["time_to_readmission"].to_numpy()
    n = len(test)

    ctx = {
        "n_test": n, "e_test": e_test, "t_test": t_test,
        "e_val": cohort["event_indicator"].to_numpy()[val_mask],
        "t_val": cohort["time_to_readmission"].to_numpy()[val_mask],
        "X_val": None, "X_test": None, "feature_names": None,
    }
    if {"cox", "gbm", "rsf"} & set(models):
        feats = np.load(PROCESSED_DIR / "features.npz", allow_pickle=True)
        ctx["X_val"] = feats["X"][val_mask]
        ctx["X_test"] = feats["X"][test_mask]
        ctx["feature_names"] = list(feats["feature_names"])

    info = json.load(open(PROCESSED_DIR / "models_info.json"))
    motor_info = json.load(open(MOTOR_DIR / "motor_result.json"))
    cindex = {m: info[f"{m}_metrics"]["c_index"] for m in ("cox", "gbm", "rsf")}
    cindex["motor"] = motor_info["c_index"]

    capacity = proportional_specialty_capacity(pools, HORIZON_DAYS)
    print(f"Capacity per day: {capacity}")

    for name in models:
        curves = calibrated_test_curves(name, ctx)
        if name == "gbm":
            main_curves = np.load(PROCESSED_DIR / "curves_test.npz")["curves_test"]
            diff = float(np.abs(curves - main_curves).max())
            if diff > 1e-9:
                raise RuntimeError(
                    f"gbm: re-calibrated curves differ from curves_test.npz (max diff {diff:.2e}); "
                    f"Table III would not match Table II"
                )
            print(f"  gbm: calibrated curves match curves_test.npz (max diff {diff:.1e})")

        t0 = time.time()
        r = schedule_mincost_specialty(curves, pools, capacity_per_specialty_day=capacity)
        elapsed = time.time() - t0
        u14 = uniform_policy(curves, day=14)
        ebf = event_before_followup_rate(r["assignments"], t_test, e_test)
        ebf_u14 = event_before_followup_rate(u14["assignments"], t_test, e_test)

        days = np.full(n, -1, dtype=np.int64)
        for i, d in r["assignments"].items():
            days[i] = d
        np.save(OUT_DIR / f"{name}_days.npy", days)

        result = {
            "c_index": cindex[name],
            "n": n,
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
