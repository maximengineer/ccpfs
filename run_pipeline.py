#!/usr/bin/env python3
"""
CCPFS End-to-End Pipeline
--------------------------
Orchestrates: cohort → features → train → calibrate → schedule → report.

Each step saves its output to data/processed/ so steps can run independently.

Usage:
    python run_pipeline.py                           # Full pipeline (all steps)
    python run_pipeline.py --step cohort             # Just build cohort
    python run_pipeline.py --step features           # Just extract features
    python run_pipeline.py --step train --model gbm  # Train only GBM
    python run_pipeline.py --step train --model cox  # Train only Cox PH
    python run_pipeline.py --step calibrate          # Calibrate best model
    python run_pipeline.py --step schedule           # Run scheduling policies
    python run_pipeline.py --step report             # Generate report from saved results
    python run_pipeline.py --step train,calibrate    # Multiple steps
    python run_pipeline.py --model gbm               # Full pipeline, GBM only
    python run_pipeline.py --survival-curves m.npz   # External model curves
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

# Ensure scheduling_follow_up is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    C_EVENT,
    C_VISIT,
    DEFAULT_SPECIALTY_CAPACITY,
    HORIZON_DAYS,
    MODEL_DIR,
    N_SPECIALTIES,
    PROCESSED_DIR,
    SPECIALTY_NAMES,
)

ALL_STEPS = ["cohort", "features", "train", "calibrate", "schedule", "report"]
VALID_MODELS = ["gbm", "cox", "rsf"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="CCPFS Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps (run individually or comma-separated):
  cohort      Build cohort from MEDS data → cohort.parquet
  features    Extract & derive features → features.npz
  train       Train survival model(s) → saved models + curves_test.npz
  calibrate   Isotonic calibration on validation set → curves_test.npz (updated)
  schedule    Run all scheduling policies → scheduling_results.npz
  report      Print results summary → pipeline_results.json

Examples:
  python run_pipeline.py --step train --model gbm --fast-grid
  python run_pipeline.py --step schedule,report
  python run_pipeline.py --survival-curves motor_curves.npz --step schedule,report
        """,
    )
    parser.add_argument("--step", type=str, default="all",
                        help="Step(s) to run: cohort,features,train,calibrate,schedule,report,all (default: all)")
    parser.add_argument("--model", type=str, default="all",
                        help="Model(s) to train: gbm,cox,rsf,all (default: all)")
    parser.add_argument("--max-patients", type=int, default=None,
                        help="Limit cohort size for testing")
    parser.add_argument("--scheduler-batch", type=int, default=2000,
                        help="Batch size for ILP scheduler (patients per batch)")
    parser.add_argument("--fast-grid", action="store_true",
                        help="Use smaller GBM parameter grid for faster training")
    parser.add_argument("--train-subsample", type=int, default=None,
                        help="Subsample training data (e.g., 50000)")
    parser.add_argument("--survival-curves", type=str, default=None,
                        help="Path to .npz with pre-computed survival curves (N, 31). "
                             "Bypasses training — plug in MOTOR or any external model.")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="Skip bootstrap uncertainty estimation")
    # Backward compatibility
    parser.add_argument("--skip-cohort", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-features", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-training", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _parse_steps(args):
    """Parse --step flag into a set of step names."""
    if args.step == "all":
        return set(ALL_STEPS)
    steps = set()
    for s in args.step.split(","):
        s = s.strip().lower()
        if s not in ALL_STEPS:
            print(f"ERROR: Unknown step '{s}'. Valid: {', '.join(ALL_STEPS)}")
            sys.exit(1)
        steps.add(s)
    return steps


def _parse_models(args):
    """Parse --model flag into a list of model names."""
    if args.model == "all":
        return list(VALID_MODELS)
    models = []
    for m in args.model.split(","):
        m = m.strip().lower()
        if m not in VALID_MODELS:
            print(f"ERROR: Unknown model '{m}'. Valid: {', '.join(VALID_MODELS)}")
            sys.exit(1)
        models.append(m)
    return models


# ---------------------------------------------------------------------------
# Intermediate file paths
# ---------------------------------------------------------------------------
def _cohort_path():
    return PROCESSED_DIR / "cohort.parquet"

def _features_path():
    return PROCESSED_DIR / "features.npz"

def _curves_path():
    return PROCESSED_DIR / "curves_test.npz"

def _scheduling_path():
    return PROCESSED_DIR / "scheduling_results.npz"


# ---------------------------------------------------------------------------
# STEP 1: Cohort
# ---------------------------------------------------------------------------
def step_cohort(args):
    """Build cohort from MEDS data → cohort.parquet."""
    print("\n" + "=" * 60)
    print("STEP 1: Build Cohort")
    print("=" * 60)

    from etl.build_cohort import build_cohort
    cohort = build_cohort(verbose=True)
    return cohort


def load_cohort():
    """Load existing cohort from disk."""
    path = _cohort_path()
    if not path.exists():
        print(f"ERROR: {path} not found. Run --step cohort first.")
        sys.exit(1)
    cohort = pl.read_parquet(path)
    print(f"  Loaded cohort: {len(cohort):,} episodes")
    return cohort


# ---------------------------------------------------------------------------
# STEP 2: Features
# ---------------------------------------------------------------------------
def step_features(args, cohort):
    """Extract & derive features → features.npz."""
    print("\n" + "=" * 60)
    print("STEP 2: Extract Features")
    print("=" * 60)

    from features.extract_features import extract_features, impute_features
    from features.derive_features import derive_features

    t0 = time.time()
    X, feature_names, _ = extract_features(cohort_df=cohort, verbose=True)

    # Train-set median imputation
    train_mask = (cohort["data_split"] == "train").to_numpy()
    X_imputed, feature_names = impute_features(X, feature_names, split_mask_train=train_mask)

    # Derive additional features
    X_derived, feature_names = derive_features(X_imputed, feature_names)
    print(f"  Derived features: {X_imputed.shape[1]} -> {X_derived.shape[1]} columns")

    elapsed = time.time() - t0
    print(f"  Feature extraction complete in {elapsed:.0f}s")

    # Save
    features_path = _features_path()
    np.savez_compressed(features_path, X=X_derived, feature_names=np.array(feature_names))
    print(f"  Saved features to {features_path}")

    return X_derived, feature_names


def load_features():
    """Load existing features from disk."""
    path = _features_path()
    if not path.exists():
        print(f"ERROR: {path} not found. Run --step features first.")
        sys.exit(1)
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    feature_names = list(data["feature_names"])
    print(f"  Loaded features: {X.shape}")
    return X, feature_names


# ---------------------------------------------------------------------------
# STEP 3: Train
# ---------------------------------------------------------------------------
def step_train(args, X, feature_names, cohort, model_names):
    """Train selected survival model(s), extract test curves, evaluate."""
    print("\n" + "=" * 60)
    print(f"STEP 3: Train Survival Models ({', '.join(m.upper() for m in model_names)})")
    print("=" * 60)

    from models.classical.train_gbm import (
        extract_survival_curves, make_structured_target,
        save_model as save_gbm, train_gbm, load_model as load_gbm,
    )
    from models.classical.train_cox import (
        extract_survival_curves_cox, save_model as save_cox,
        train_cox, load_model as load_cox,
    )
    from models.classical.train_rsf import (
        extract_survival_curves_rsf, save_model as save_rsf,
        train_rsf, load_model as load_rsf,
    )
    from models.evaluate_model import evaluate_survival_model

    # Split data
    split_col = cohort["data_split"].to_numpy()
    train_mask = split_col == "train"
    val_mask = split_col == "tuning"
    test_mask = split_col == "held_out"

    event_indicators = cohort["event_indicator"].to_numpy()
    time_to_event = cohort["time_to_readmission"].to_numpy()

    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    e_train = event_indicators[train_mask]
    t_train = time_to_event[train_mask]
    e_val = event_indicators[val_mask]
    t_val = time_to_event[val_mask]
    e_test = event_indicators[test_mask]
    t_test = time_to_event[test_mask]

    print(f"\n  Split sizes: train={train_mask.sum():,}, val={val_mask.sum():,}, test={test_mask.sum():,}")

    # Subsample training data
    if args.train_subsample and len(X_train) > args.train_subsample:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train), size=args.train_subsample, replace=False)
        X_train_fit, e_train_fit, t_train_fit = X_train[idx], e_train[idx], t_train[idx]
        print(f"  Subsampled training data: {args.train_subsample:,} / {len(X_train):,}")
    else:
        X_train_fit, e_train_fit, t_train_fit = X_train, e_train, t_train

    y_train = make_structured_target(e_train, t_train)
    y_train_fit = make_structured_target(e_train_fit, t_train_fit)
    y_val = make_structured_target(e_val, t_val)
    y_test = make_structured_target(e_test, t_test)

    models_info = {}
    trained = {}  # name -> (model, curves_test, scaler_or_none)

    # --- Train each requested model ---
    if "gbm" in model_names:
        print("\n  Training GBM...")
        t0 = time.time()
        param_grid = None
        if args.fast_grid:
            param_grid = {
                "n_estimators": [300],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
                "min_samples_leaf": [20],
                "subsample": [0.8],
            }
        best_params, gbm_model = train_gbm(X_train_fit, y_train_fit, X_val, y_val, param_grid=param_grid)
        print(f"  GBM trained in {time.time() - t0:.0f}s")
        save_gbm(gbm_model)
        models_info["gbm_params"] = best_params

        curves = extract_survival_curves(gbm_model, X_test)
        metrics = evaluate_survival_model(y_train_fit, y_test, e_test, t_test, curves, "GBM")
        models_info["gbm_metrics"] = {"c_index": metrics["c_index"], "ibs": metrics["ibs"]}
        trained["gbm"] = {"model": gbm_model, "curves_test": curves, "scaler": None}

    if "cox" in model_names:
        print("\n  Training Cox PH...")
        t0 = time.time()
        cox_model, cox_scaler = train_cox(X_train_fit, e_train_fit, t_train_fit, feature_names)
        print(f"  Cox PH trained in {time.time() - t0:.0f}s")
        save_cox(cox_model, scaler=cox_scaler)

        curves = extract_survival_curves_cox(cox_model, X_test, feature_names, scaler=cox_scaler)
        metrics = evaluate_survival_model(y_train_fit, y_test, e_test, t_test, curves, "Cox PH")
        models_info["cox_metrics"] = {"c_index": metrics["c_index"], "ibs": metrics["ibs"]}
        trained["cox"] = {"model": cox_model, "curves_test": curves, "scaler": cox_scaler}

    if "rsf" in model_names:
        print("\n  Training Random Survival Forest...")
        t0 = time.time()
        rsf_params, rsf_model = train_rsf(X_train_fit, y_train_fit, X_val, y_val)
        print(f"  RSF trained in {time.time() - t0:.0f}s")
        save_rsf(rsf_model)
        models_info["rsf_params"] = rsf_params

        curves = extract_survival_curves_rsf(rsf_model, X_test)
        metrics = evaluate_survival_model(y_train_fit, y_test, e_test, t_test, curves, "RSF")
        models_info["rsf_metrics"] = {"c_index": metrics["c_index"], "ibs": metrics["ibs"]}
        trained["rsf"] = {"model": rsf_model, "curves_test": curves, "scaler": None}

    # Select best model
    best_name = None
    best_cindex = -1.0
    for name, info in trained.items():
        ci = models_info[f"{name}_metrics"]["c_index"]
        if ci > best_cindex:
            best_cindex = ci
            best_name = name

    if best_name is None:
        print("ERROR: No models were trained.")
        sys.exit(1)

    print(f"\n  Best model: {best_name.upper()} (C-index={best_cindex:.4f})")
    models_info["best_model"] = best_name

    best_curves_test = trained[best_name]["curves_test"]

    # Save curves + metadata for downstream steps
    save_data = {
        "curves_test": best_curves_test,
        "model_name": best_name,
        "e_test": e_test,
        "t_test": t_test,
    }
    # Save per-model curves too (for calibrate step to re-extract val curves)
    for name, info in trained.items():
        save_data[f"{name}_curves_test"] = info["curves_test"]

    np.savez_compressed(_curves_path(), **save_data)
    print(f"  Saved test curves to {_curves_path()}")

    # Save models_info as JSON
    info_path = PROCESSED_DIR / "models_info.json"
    with open(info_path, "w") as f:
        json.dump(models_info, f, indent=2, default=str)
    print(f"  Saved model info to {info_path}")

    return best_curves_test, models_info, e_test, t_test


# ---------------------------------------------------------------------------
# STEP 4: Calibrate
# ---------------------------------------------------------------------------
def step_calibrate(args, X, feature_names, cohort):
    """Calibrate the best model's survival curves using isotonic regression."""
    print("\n" + "=" * 60)
    print("STEP 4: Calibrate Survival Curves")
    print("=" * 60)

    from models.calibrate import calibrate_curves, apply_calibration

    # Load saved curves
    curves_path = _curves_path()
    if not curves_path.exists():
        print(f"ERROR: {curves_path} not found. Run --step train first.")
        sys.exit(1)

    data = np.load(curves_path, allow_pickle=True)
    test_curves = data["curves_test"]
    best_name = str(data["model_name"])
    e_test = data["e_test"]
    t_test = data["t_test"]

    print(f"  Best model: {best_name.upper()}")
    print(f"  Test curves shape: {test_curves.shape}")

    # Extract validation curves for the best model
    split_col = cohort["data_split"].to_numpy()
    val_mask = split_col == "tuning"
    event_indicators = cohort["event_indicator"].to_numpy()
    time_to_event = cohort["time_to_readmission"].to_numpy()

    if best_name == "gbm":
        from models.classical.train_gbm import extract_survival_curves, load_model
        model = load_model()
        val_curves = extract_survival_curves(model, X[val_mask])
    elif best_name == "cox":
        from models.classical.train_cox import extract_survival_curves_cox, load_model
        model, scaler = load_model()
        val_curves = extract_survival_curves_cox(model, X[val_mask], feature_names, scaler=scaler)
    elif best_name == "rsf":
        from models.classical.train_rsf import extract_survival_curves_rsf, load_model
        model = load_model()
        val_curves = extract_survival_curves_rsf(model, X[val_mask])
    else:
        print(f"  WARNING: Unknown model '{best_name}', skipping calibration.")
        return test_curves, e_test, t_test

    # Fit calibrators on validation set
    _, calibrators = calibrate_curves(
        val_curves, event_indicators[val_mask], time_to_event[val_mask],
    )

    if not calibrators:
        print("  WARNING: No calibrators fitted (insufficient data). Skipping.")
        return test_curves, e_test, t_test

    # Apply calibration to test curves
    test_curves_cal = apply_calibration(test_curves, calibrators)

    print(f"  Mean S(15) before cal: {test_curves[:, 15].mean():.4f}")
    print(f"  Mean S(15) after cal:  {test_curves_cal[:, 15].mean():.4f}")

    # Overwrite curves file with calibrated version
    save_data = {
        "curves_test": test_curves_cal,
        "model_name": best_name,
        "e_test": e_test,
        "t_test": t_test,
        "calibrated": True,
    }
    np.savez_compressed(_curves_path(), **save_data)
    print(f"  Saved calibrated curves to {_curves_path()}")

    return test_curves_cal, e_test, t_test


# ---------------------------------------------------------------------------
# STEP 5: Schedule
# ---------------------------------------------------------------------------
def step_schedule(args, survival_curves, cohort):
    """Run all scheduling policies and compare."""
    print("\n" + "=" * 60)
    print("STEP 5: Run Scheduling Policies")
    print("=" * 60)

    from policy.specialty_scheduler import schedule_ilp_specialty, schedule_greedy_specialty
    from policy.ilp_scheduler import schedule_ilp
    from policy.greedy_scheduler import schedule_greedy
    from policy.mincost_solver import schedule_mincost_specialty, schedule_mincost_global
    from policy.baselines import (
        uniform_policy, risk_bucket_policy, guideline_policy,
        unconstrained_optimal_policy, uniform_capacity_policy, guideline_capacity_policy,
    )

    specialty_pools = cohort["specialty_pool"].to_numpy()
    n_patients = len(cohort)
    batch_size = args.scheduler_batch
    results = {}

    # Scale capacity
    total_default_cap = sum(DEFAULT_SPECIALTY_CAPACITY.values())
    needed_per_day = int(np.ceil(n_patients / HORIZON_DAYS))
    capacity_scale = max(1.0, needed_per_day / total_default_cap)

    scaled_specialty_cap = {
        k: int(np.ceil(v * capacity_scale))
        for k, v in DEFAULT_SPECIALTY_CAPACITY.items()
    }
    total_scaled_cap = sum(scaled_specialty_cap.values())
    global_capacity = np.full(HORIZON_DAYS, total_scaled_cap)

    print(f"\n  Capacity scaling: {capacity_scale:.1f}x "
          f"({total_default_cap}/day -> {total_scaled_cap}/day, "
          f"total={total_scaled_cap * HORIZON_DAYS:,} slots for {n_patients:,} patients)")
    for k, name in enumerate(SPECIALTY_NAMES):
        print(f"    {name}: {DEFAULT_SPECIALTY_CAPACITY[k]}/day -> {scaled_specialty_cap[k]}/day")

    # --- Baselines ---
    print(f"\n  Running baselines on {n_patients:,} patients...")

    results["uniform_d14"] = uniform_policy(survival_curves, day=14)
    print(f"    Uniform (d14): cost={results['uniform_d14']['total_expected_cost']:,.0f}")

    results["risk_bucket"] = risk_bucket_policy(survival_curves)
    print(f"    Risk bucket: cost={results['risk_bucket']['total_expected_cost']:,.0f}")

    is_hf = cohort["is_heart_failure"].to_numpy().astype(bool)
    results["guideline"] = guideline_policy(survival_curves, is_heart_failure=is_hf)
    print(f"    Guideline: cost={results['guideline']['total_expected_cost']:,.0f}")

    results["unconstrained"] = unconstrained_optimal_policy(survival_curves)
    print(f"    Unconstrained optimal: cost={results['unconstrained']['total_expected_cost']:,.0f}")

    # --- Capacity-aware baselines ---
    print(f"\n  Running capacity-aware baselines...")
    results["uniform_d14_cap"] = uniform_capacity_policy(
        survival_curves, specialty_pools, day=14,
        capacity_per_specialty_day=scaled_specialty_cap,
    )
    print(f"    Uniform d14 (capacity): cost={results['uniform_d14_cap']['total_expected_cost']:,.0f}"
          f"  {results['uniform_d14_cap']['status']}")

    results["guideline_cap"] = guideline_capacity_policy(
        survival_curves, specialty_pools, is_heart_failure=is_hf,
        capacity_per_specialty_day=scaled_specialty_cap,
    )
    print(f"    Guideline (capacity): cost={results['guideline_cap']['total_expected_cost']:,.0f}"
          f"  {results['guideline_cap']['status']}")

    # --- Greedy ---
    print("\n  Running greedy (global capacity)...")
    results["greedy_global"] = schedule_greedy(survival_curves, capacity_per_day=global_capacity)
    print(f"    Greedy (global): cost={results['greedy_global']['total_expected_cost']:,.0f}")

    print("  Running greedy (specialty pools)...")
    results["greedy_specialty"] = schedule_greedy_specialty(
        survival_curves, specialty_pools, capacity_per_specialty_day=scaled_specialty_cap,
    )
    print(f"    Greedy (specialty): cost={results['greedy_specialty']['total_expected_cost']:,.0f}"
          f"  {results['greedy_specialty']['status']}")

    # --- Min-cost assignment ---
    print(f"\n  Running min-cost assignment (specialty, {n_patients:,} patients)...")
    results["mincost_specialty"] = schedule_mincost_specialty(
        survival_curves, specialty_pools, capacity_per_specialty_day=scaled_specialty_cap,
    )
    print(f"    MinCost (specialty): cost={results['mincost_specialty']['total_expected_cost']:,.0f}"
          f"  {results['mincost_specialty']['status']}")

    print(f"  Running min-cost assignment (global, {n_patients:,} patients)...")
    results["mincost_global"] = schedule_mincost_global(
        survival_curves, capacity_per_day=global_capacity,
    )
    print(f"    MinCost (global): cost={results['mincost_global']['total_expected_cost']:,.0f}"
          f"  {results['mincost_global']['status']}")

    # --- ILP (small cohorts only) ---
    if n_patients <= batch_size:
        print(f"\n  Running ILP (specialty, {n_patients:,} patients)...")
        results["ilp_specialty"] = schedule_ilp_specialty(
            survival_curves, specialty_pools,
            capacity_per_specialty_day=scaled_specialty_cap, time_limit=300,
        )
        print(f"    ILP (specialty): cost={results['ilp_specialty']['total_expected_cost']:,.0f}")

        print(f"  Running ILP (global capacity)...")
        results["ilp_global"] = schedule_ilp(
            survival_curves, capacity_per_day=global_capacity, time_limit=300,
        )
        print(f"    ILP (global): cost={results['ilp_global']['total_expected_cost']:,.0f}")
    else:
        print(f"\n  Skipping ILP — cohort ({n_patients:,}) > batch limit ({batch_size:,}), using min-cost.")
        results["ilp_specialty"] = results["mincost_specialty"]
        results["ilp_global"] = results["mincost_global"]

    # Save scheduling results
    save_data = {}
    for name, result in results.items():
        assignments = result.get("assignments", {})
        days = np.full(n_patients, -1)
        for i, d in assignments.items():
            days[i] = d
        save_data[f"{name}_days"] = days
        save_data[f"{name}_cost"] = result.get("total_expected_cost", float("nan"))
        save_data[f"{name}_status"] = result.get("status", "N/A")

    np.savez_compressed(_scheduling_path(), **save_data, policy_names=list(results.keys()))
    print(f"\n  Saved scheduling results to {_scheduling_path()}")

    return results


# ---------------------------------------------------------------------------
# STEP 6: Report
# ---------------------------------------------------------------------------
def step_report(scheduling_results, models_info, cohort, e_test=None, t_test=None):
    """Generate summary report."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    n = len(cohort)

    # Model performance
    print("\n--- Model Performance ---")
    for name in VALID_MODELS:
        key = f"{name}_metrics"
        if key in models_info:
            m = models_info[key]
            print(f"  {name.upper():>6}: C-index={m['c_index']:.4f}, IBS={m['ibs']:.4f}")
    if "best_model" in models_info:
        print(f"  Best model for scheduling: {models_info['best_model'].upper()}")

    # Scheduling comparison
    print(f"\n--- Scheduling Policy Comparison ({n:,} patients) ---")
    print(f"  {'Policy':<25} {'Status':<30} {'Total Cost':>12} {'Avg/Pt':>10} {'EBF Rate':>10}")
    print(f"  {'-'*25} {'-'*30} {'-'*12} {'-'*10} {'-'*10}")

    # Compute EBF rates
    ebf_results = {}
    if e_test is not None and t_test is not None:
        from evaluation.metrics import event_before_followup_rate
        for name, result in scheduling_results.items():
            assignments = result.get("assignments", {})
            if len(assignments) == n:
                ebf_results[name] = event_before_followup_rate(assignments, t_test, e_test)

    for name, result in scheduling_results.items():
        status = result.get("status", "N/A")
        cost = result.get("total_expected_cost", float("inf"))
        avg = cost / n if cost < float("inf") else float("inf")
        ebf_str = f"{ebf_results[name]['ebf_rate']:.1%}" if name in ebf_results else "N/A"
        print(f"  {name:<25} {status:<30} {cost:>12,.0f} {avg:>10,.0f} {ebf_str:>10}")

    # EBF detail
    if ebf_results:
        print("\n--- Event-Before-Followup Analysis ---")
        print(f"  {'Policy':<25} {'EBF Count':>10} {'Total Events':>14} {'Catch Rate':>12}")
        print(f"  {'-'*25} {'-'*10} {'-'*14} {'-'*12}")
        for name, ebf in ebf_results.items():
            print(f"  {name:<25} {ebf['ebf_count']:>10,} {ebf['total_events']:>14,} {ebf['catch_rate']:>12.1%}")

    # Save results JSON
    output = {
        "cohort_size": n,
        "model_performance": models_info,
        "scheduling_results": {
            name: {
                "status": r.get("status", "N/A"),
                "total_expected_cost": r.get("total_expected_cost", None),
                "n_assigned": len(r.get("assignments", {})),
                **({"ebf": ebf_results[name]} if name in ebf_results else {}),
            }
            for name, r in scheduling_results.items()
        },
    }

    results_path = PROCESSED_DIR / "pipeline_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")


# ---------------------------------------------------------------------------
# External curves loader
# ---------------------------------------------------------------------------
def _load_external_curves(args, cohort):
    """Load pre-computed survival curves from an external model."""
    print("\n" + "=" * 60)
    print("EXTERNAL SURVIVAL CURVES")
    print("=" * 60)

    data = np.load(args.survival_curves, allow_pickle=True)
    # Accept both "curves" and "curves_test" keys for flexibility
    if "curves" in data:
        curves = data["curves"]
    elif "curves_test" in data:
        curves = data["curves_test"]
    else:
        raise KeyError(f"Expected 'curves' or 'curves_test' key in {args.survival_curves}, "
                       f"found: {list(data.keys())}")
    model_name = str(data["model_name"]) if "model_name" in data else "External"

    print(f"  Loaded {curves.shape[0]:,} curves from {args.survival_curves}")
    print(f"  Model: {model_name}, Shape: {curves.shape}")

    if curves.shape[1] != HORIZON_DAYS + 1:
        raise ValueError(f"Curves must have {HORIZON_DAYS + 1} columns, got {curves.shape[1]}")

    split_col = cohort["data_split"].to_numpy()
    test_mask = split_col == "held_out"
    n_test = test_mask.sum()

    if curves.shape[0] == len(cohort):
        print(f"  Full cohort ({len(cohort):,}), extracting test set...")
        test_curves = curves[test_mask]
    elif curves.shape[0] == n_test:
        print(f"  Matches test set ({n_test:,}), using directly.")
        test_curves = curves
    else:
        raise ValueError(
            f"Curve count ({curves.shape[0]}) doesn't match cohort ({len(cohort)}) "
            f"or test set ({n_test})."
        )

    # Sanity checks
    if not np.allclose(test_curves[:, 0], 1.0, atol=0.01):
        print("  WARNING: S(0) != 1.0 for some patients. Forcing S(0) = 1.0.")
        test_curves[:, 0] = 1.0
    test_curves = np.clip(test_curves, 0.0, 1.0)

    event_indicators = cohort["event_indicator"].to_numpy()
    time_to_event = cohort["time_to_readmission"].to_numpy()
    e_test = event_indicators[test_mask]
    t_test = time_to_event[test_mask]

    models_info = {"best_model": model_name}

    if "c_index" in data:
        ci = float(data["c_index"])
    else:
        from models.evaluate_model import compute_cindex
        risk_scores = -np.trapezoid(test_curves, np.arange(HORIZON_DAYS + 1, dtype=float), axis=1)
        ci = compute_cindex(e_test, t_test, risk_scores)

    models_info[f"{model_name.lower()}_metrics"] = {
        "c_index": ci, "ibs": float(data.get("ibs", float("nan")))
    }
    print(f"  C-index: {ci:.4f}")

    # Save to standard curves file so schedule step can find it
    np.savez_compressed(_curves_path(),
        curves_test=test_curves, model_name=model_name,
        e_test=e_test, t_test=t_test,
    )

    info_path = PROCESSED_DIR / "models_info.json"
    with open(info_path, "w") as f:
        json.dump(models_info, f, indent=2, default=str)

    test_cohort = cohort.filter(pl.Series(test_mask))
    return test_curves, test_cohort, models_info, e_test, t_test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    t_start = time.time()

    steps = _parse_steps(args)
    model_names = _parse_models(args)

    # Handle backward-compat flags
    if args.skip_cohort and "cohort" in steps:
        steps.discard("cohort")
    if args.skip_features and "features" in steps:
        steps.discard("features")
    if args.skip_training and "train" in steps:
        steps.discard("train")

    print("=" * 60)
    print("CCPFS Pipeline — MIMIC-IV Real Data")
    print(f"  Steps: {', '.join(s for s in ALL_STEPS if s in steps)}")
    print(f"  Models: {', '.join(m.upper() for m in model_names)}")
    print("=" * 60)

    # --- Step 1: Cohort ---
    if "cohort" in steps:
        cohort = step_cohort(args)
    else:
        cohort = load_cohort()

    # Optional subsample — store indices for consistent subsampling of features
    _subsample_idx = None
    if args.max_patients and len(cohort) > args.max_patients:
        print(f"\n  Subsampling to {args.max_patients:,} patients...")
        rng = np.random.RandomState(42)
        _subsample_idx = rng.choice(len(cohort), size=args.max_patients, replace=False)
        _subsample_idx.sort()
        cohort = cohort[_subsample_idx.tolist()]

    # --- External curves path ---
    if args.survival_curves:
        test_curves, test_cohort, models_info, e_test, t_test = (
            _load_external_curves(args, cohort)
        )
    else:
        # Check which steps need features / curves
        needs_features = bool({"features", "train", "calibrate"} & steps)
        needs_curves = bool({"calibrate", "schedule", "report"} & steps)

        X, feature_names = None, None
        test_curves, models_info, e_test, t_test = None, {}, None, None
        test_cohort = None

        # --- Step 2: Features ---
        if needs_features:
            if "features" in steps:
                X, feature_names = step_features(args, cohort)
            else:
                X, feature_names = load_features()
                # Subsample features using same indices as cohort
                if _subsample_idx is not None and X.shape[0] != len(cohort):
                    X = X[_subsample_idx]

        # --- Step 3: Train ---
        if "train" in steps:
            test_curves, models_info, e_test, t_test = step_train(
                args, X, feature_names, cohort, model_names
            )
        elif needs_curves:
            # Load from saved curves
            cdata = np.load(_curves_path(), allow_pickle=True)
            test_curves = cdata["curves_test"]
            e_test = cdata["e_test"]
            t_test = cdata["t_test"]
            info_path = PROCESSED_DIR / "models_info.json"
            with open(info_path) as f:
                models_info = json.load(f)
            print(f"  Loaded saved curves: {test_curves.shape}, model={models_info.get('best_model', '?')}")

        # --- Step 4: Calibrate ---
        if "calibrate" in steps:
            test_curves, e_test, t_test = step_calibrate(args, X, feature_names, cohort)

        # Build test cohort if needed for downstream steps
        if {"schedule", "report"} & steps:
            split_col = cohort["data_split"].to_numpy()
            test_mask = split_col == "held_out"
            test_cohort = cohort.filter(pl.Series(test_mask))

            print(f"\n  Scheduling with {models_info.get('best_model', '?').upper()} on "
                  f"{len(test_cohort):,} test patients")

    # --- Step 5: Schedule ---
    if "schedule" in steps:
        scheduling_results = step_schedule(args, test_curves, test_cohort)
    elif "report" in steps and _scheduling_path().exists():
        # Reconstruct scheduling_results dict from saved arrays
        sdata = np.load(_scheduling_path(), allow_pickle=True)
        policy_names = list(sdata["policy_names"])
        scheduling_results = {}
        for name in policy_names:
            days = sdata[f"{name}_days"]
            assignments = {int(i): int(d) for i, d in enumerate(days) if d >= 0}
            scheduling_results[name] = {
                "assignments": assignments,
                "total_expected_cost": float(sdata[f"{name}_cost"]),
                "status": str(sdata[f"{name}_status"]),
            }
        print(f"  Loaded saved scheduling results ({len(policy_names)} policies)")
    else:
        scheduling_results = {}
        if "report" in steps:
            print("  WARNING: No scheduling results found. Run --step schedule first.")

    # --- Step 6: Report ---
    if "report" in steps and scheduling_results:
        step_report(scheduling_results, models_info, test_cohort, e_test=e_test, t_test=t_test)

    elapsed = time.time() - t_start
    print(f"\n  Total pipeline time: {elapsed / 60:.1f} minutes")
    print("  Done.")


if __name__ == "__main__":
    main()
