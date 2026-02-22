#!/usr/bin/env python3
"""
CCPFS End-to-End Pipeline
--------------------------
Orchestrates: cohort → features → train models → generate curves → schedule → evaluate.

Usage:
    python run_pipeline.py                        # Full pipeline
    python run_pipeline.py --skip-cohort          # Re-use existing cohort
    python run_pipeline.py --skip-features        # Re-use existing features
    python run_pipeline.py --skip-training        # Re-use saved models
    python run_pipeline.py --max-patients 5000    # Subset for quick testing
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


def parse_args():
    parser = argparse.ArgumentParser(description="CCPFS Pipeline")
    parser.add_argument("--skip-cohort", action="store_true",
                        help="Skip cohort building (use existing cohort.parquet)")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip feature extraction (use existing features.npz)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip model training (use saved models)")
    parser.add_argument("--max-patients", type=int, default=None,
                        help="Limit cohort size for testing")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="Skip bootstrap uncertainty estimation")
    parser.add_argument("--scheduler-batch", type=int, default=2000,
                        help="Batch size for ILP scheduler (patients per batch)")
    parser.add_argument("--fast-grid", action="store_true",
                        help="Use smaller GBM parameter grid for faster training")
    parser.add_argument("--train-subsample", type=int, default=None,
                        help="Subsample training data for GBM (e.g., 50000)")
    return parser.parse_args()


def step_cohort(args):
    """Phase B: Build cohort from MEDS data."""
    print("\n" + "=" * 60)
    print("STEP 1: Build Cohort")
    print("=" * 60)

    if args.skip_cohort:
        cohort_path = PROCESSED_DIR / "cohort.parquet"
        if not cohort_path.exists():
            print("ERROR: --skip-cohort but cohort.parquet not found")
            sys.exit(1)
        cohort = pl.read_parquet(cohort_path)
        print(f"  Loaded existing cohort: {len(cohort):,} episodes")
        return cohort

    from etl.build_cohort import build_cohort
    cohort = build_cohort(verbose=True)
    return cohort


def step_features(args, cohort):
    """Phase C: Extract features."""
    print("\n" + "=" * 60)
    print("STEP 2: Extract Features")
    print("=" * 60)

    features_path = PROCESSED_DIR / "features.npz"

    if args.skip_features:
        if not features_path.exists():
            print("ERROR: --skip-features but features.npz not found")
            sys.exit(1)
        data = np.load(features_path, allow_pickle=True)
        X = data["X"]
        feature_names = list(data["feature_names"])
        print(f"  Loaded existing features: {X.shape}")
        return X, feature_names

    from features.extract_features import extract_features, impute_features

    t0 = time.time()
    X, feature_names, cohort_with_features = extract_features(
        cohort_df=cohort,
        verbose=True,
    )

    # Get split assignments for train-set median imputation
    train_mask = (cohort["data_split"] == "train").to_numpy()

    X_imputed, feature_names = impute_features(
        X, feature_names, split_mask_train=train_mask
    )

    elapsed = time.time() - t0
    print(f"  Feature extraction complete in {elapsed:.0f}s")

    # Save
    np.savez_compressed(
        features_path,
        X=X_imputed,
        feature_names=np.array(feature_names),
    )
    print(f"  Saved features to {features_path}")

    return X_imputed, feature_names


def step_train(args, X, feature_names, cohort):
    """Phase D: Train survival models."""
    print("\n" + "=" * 60)
    print("STEP 3: Train Survival Models")
    print("=" * 60)

    from models.classical.train_gbm import (
        extract_survival_curves,
        make_structured_target,
        save_model as save_gbm,
        train_gbm,
    )
    from models.classical.train_cox import (
        extract_survival_curves_cox,
        save_model as save_cox,
        train_cox,
    )
    from models.evaluate_model import evaluate_survival_model

    # Split data using cohort's pre-assigned splits
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

    # Subsample training data for faster GBM training
    if args.train_subsample and len(X_train) > args.train_subsample:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train), size=args.train_subsample, replace=False)
        X_train_fit = X_train[idx]
        e_train_fit = e_train[idx]
        t_train_fit = t_train[idx]
        print(f"  Subsampled training data: {args.train_subsample:,} / {len(X_train):,}")
    else:
        X_train_fit = X_train
        e_train_fit = e_train
        t_train_fit = t_train

    y_train = make_structured_target(e_train, t_train)  # Full (for IBS baseline)
    y_train_fit = make_structured_target(e_train_fit, t_train_fit)
    y_val = make_structured_target(e_val, t_val)
    y_test = make_structured_target(e_test, t_test)

    models_info = {}

    if args.skip_training:
        from models.classical.train_gbm import load_model as load_gbm
        from models.classical.train_cox import load_model as load_cox

        gbm_model = load_gbm()
        cox_model = load_cox()
        print("  Loaded saved models")
    else:
        # --- Train GBM ---
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
        best_params, gbm_model = train_gbm(
            X_train_fit, y_train_fit, X_val, y_val, param_grid=param_grid
        )
        print(f"  GBM trained in {time.time() - t0:.0f}s")
        save_gbm(gbm_model)
        models_info["gbm_params"] = best_params

        # --- Train Cox PH ---
        print("\n  Training Cox PH...")
        t0 = time.time()
        cox_model = train_cox(X_train_fit, e_train_fit, t_train_fit, feature_names)
        print(f"  Cox PH trained in {time.time() - t0:.0f}s")
        save_cox(cox_model)

    # --- Extract survival curves ---
    print("\n  Extracting survival curves (test set)...")
    gbm_curves_test = extract_survival_curves(gbm_model, X_test)
    cox_curves_test = extract_survival_curves_cox(cox_model, X_test, feature_names)

    # --- Evaluate ---
    gbm_metrics = evaluate_survival_model(
        y_train, y_test, e_test, t_test, gbm_curves_test, "GBM"
    )
    cox_metrics = evaluate_survival_model(
        y_train, y_test, e_test, t_test, cox_curves_test, "Cox PH"
    )

    models_info["gbm_metrics"] = {
        "c_index": gbm_metrics["c_index"],
        "ibs": gbm_metrics["ibs"],
    }
    models_info["cox_metrics"] = {
        "c_index": cox_metrics["c_index"],
        "ibs": cox_metrics["ibs"],
    }

    # Also extract full curves for scheduling (all patients)
    print("\n  Extracting survival curves (all patients)...")
    gbm_curves_all = extract_survival_curves(gbm_model, X)
    cox_curves_all = extract_survival_curves_cox(cox_model, X, feature_names)

    return {
        "gbm_model": gbm_model,
        "cox_model": cox_model,
        "gbm_curves_all": gbm_curves_all,
        "cox_curves_all": cox_curves_all,
        "gbm_curves_test": gbm_curves_test,
        "cox_curves_test": cox_curves_test,
        "models_info": models_info,
        "masks": {"train": train_mask, "val": val_mask, "test": test_mask},
        "y_train": y_train,
        "y_test": y_test,
        "e_test": e_test,
        "t_test": t_test,
    }


def step_calibrate(model_results, X, feature_names, cohort):
    """Phase D (cont): Calibrate survival curves on validation set."""
    print("\n" + "=" * 60)
    print("STEP 4: Calibrate Survival Curves")
    print("=" * 60)

    from models.calibrate import calibrate_curves, apply_calibration
    from models.classical.train_gbm import extract_survival_curves
    from models.classical.train_cox import extract_survival_curves_cox

    event_indicators = cohort["event_indicator"].to_numpy()
    time_to_event = cohort["time_to_readmission"].to_numpy()
    val_mask = model_results["masks"]["val"]

    # Calibrate GBM on validation set
    gbm_curves_val = extract_survival_curves(
        model_results["gbm_model"], X[val_mask]
    )
    _, gbm_calibrators = calibrate_curves(
        gbm_curves_val,
        event_indicators[val_mask],
        time_to_event[val_mask],
    )

    # Apply calibration to all curves
    gbm_curves_cal = apply_calibration(
        model_results["gbm_curves_all"], gbm_calibrators
    )

    print(f"  Calibrated GBM curves: {gbm_curves_cal.shape}")
    print(f"  Mean S(15) before cal: {model_results['gbm_curves_all'][:, 15].mean():.4f}")
    print(f"  Mean S(15) after cal:  {gbm_curves_cal[:, 15].mean():.4f}")

    return gbm_curves_cal, gbm_calibrators


def step_schedule(args, survival_curves, cohort):
    """Phase E+F: Run scheduling policies and compare."""
    print("\n" + "=" * 60)
    print("STEP 5: Run Scheduling Policies")
    print("=" * 60)

    from policy.specialty_scheduler import (
        schedule_ilp_specialty,
        schedule_greedy_specialty,
    )
    from policy.ilp_scheduler import schedule_ilp
    from policy.greedy_scheduler import schedule_greedy
    from policy.baselines import (
        uniform_policy,
        risk_bucket_policy,
        guideline_policy,
        unconstrained_optimal_policy,
    )

    specialty_pools = cohort["specialty_pool"].to_numpy()
    n_patients = len(cohort)

    batch_size = args.scheduler_batch
    results = {}

    # Scale capacity so that total capacity = ~1.3x patients (slight over-capacity)
    # This represents feasible scheduling with realistic utilisation ~75-80%
    total_default_cap = sum(DEFAULT_SPECIALTY_CAPACITY.values())  # 65/day
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

    # --- Baselines (fast, no capacity constraints) ---
    print(f"\n  Running baselines on {n_patients:,} patients...")

    results["uniform_d14"] = uniform_policy(survival_curves, day=14)
    print(f"    Uniform (d14): cost={results['uniform_d14']['total_expected_cost']:,.0f}")

    results["risk_bucket"] = risk_bucket_policy(survival_curves)
    print(f"    Risk bucket: cost={results['risk_bucket']['total_expected_cost']:,.0f}")

    # Clinical guideline policy needs HF flags
    is_hf = cohort["is_heart_failure"].to_numpy().astype(bool)
    results["guideline"] = guideline_policy(survival_curves, is_heart_failure=is_hf)
    print(f"    Guideline: cost={results['guideline']['total_expected_cost']:,.0f}")

    results["unconstrained"] = unconstrained_optimal_policy(survival_curves)
    print(f"    Unconstrained optimal: cost={results['unconstrained']['total_expected_cost']:,.0f}")

    # --- Greedy (no specialty) ---
    print("\n  Running greedy (global capacity)...")
    results["greedy_global"] = schedule_greedy(
        survival_curves, capacity_per_day=global_capacity
    )
    print(f"    Greedy (global): status={results['greedy_global']['status']}, "
          f"cost={results['greedy_global']['total_expected_cost']:,.0f}")

    # --- Greedy with specialty pools ---
    print("\n  Running greedy (specialty pools)...")
    results["greedy_specialty"] = schedule_greedy_specialty(
        survival_curves, specialty_pools,
        capacity_per_specialty_day=scaled_specialty_cap,
    )
    print(f"    Greedy (specialty): status={results['greedy_specialty']['status']}, "
          f"cost={results['greedy_specialty']['total_expected_cost']:,.0f}")

    # --- ILP with specialty pools (batched for large cohorts) ---
    if n_patients <= batch_size:
        print(f"\n  Running ILP (specialty, {n_patients:,} patients)...")
        results["ilp_specialty"] = schedule_ilp_specialty(
            survival_curves, specialty_pools,
            capacity_per_specialty_day=scaled_specialty_cap,
            time_limit=300,
        )
        print(f"    ILP (specialty): status={results['ilp_specialty']['status']}, "
              f"cost={results['ilp_specialty']['total_expected_cost']:,.0f}")
    else:
        print(f"\n  Running ILP (specialty, batched {batch_size}/batch)...")
        results["ilp_specialty"] = _batched_ilp_specialty(
            survival_curves, specialty_pools, batch_size, scaled_specialty_cap
        )
        print(f"    ILP (specialty): status={results['ilp_specialty']['status']}, "
              f"cost={results['ilp_specialty']['total_expected_cost']:,.0f}")

    # --- ILP global (batched) ---
    if n_patients <= batch_size:
        print(f"\n  Running ILP (global capacity)...")
        results["ilp_global"] = schedule_ilp(
            survival_curves, capacity_per_day=global_capacity, time_limit=300
        )
        print(f"    ILP (global): status={results['ilp_global']['status']}, "
              f"cost={results['ilp_global']['total_expected_cost']:,.0f}")
    else:
        print(f"\n  Running ILP (global, batched {batch_size}/batch)...")
        results["ilp_global"] = _batched_ilp_global(
            survival_curves, global_capacity, batch_size, global_capacity[0]
        )
        print(f"    ILP (global): status={results['ilp_global']['status']}, "
              f"cost={results['ilp_global']['total_expected_cost']:,.0f}")

    return results


def _batched_ilp_specialty(survival_curves, specialty_pools, batch_size, scaled_specialty_cap):
    """Run ILP in batches, scaling capacity proportionally."""
    from policy.specialty_scheduler import schedule_ilp_specialty

    n = len(survival_curves)
    all_assignments = {}
    total_cost = 0.0
    status = "Optimal"

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_curves = survival_curves[start:end]
        batch_pools = specialty_pools[start:end]
        batch_n = end - start

        # Scale capacity proportionally to batch size
        scale = batch_n / n
        batch_cap = {
            k: max(1, int(np.ceil(v * scale)))
            for k, v in scaled_specialty_cap.items()
        }
        # Ensure enough total capacity per pool
        for k in batch_cap:
            pool_count = (batch_pools == k).sum()
            min_cap = int(np.ceil(pool_count / HORIZON_DAYS))
            batch_cap[k] = max(batch_cap[k], min_cap)

        result = schedule_ilp_specialty(
            batch_curves, batch_pools,
            capacity_per_specialty_day=batch_cap,
            time_limit=120,
        )

        if result["status"] != "Optimal":
            status = result["status"]

        for patient_idx, day in result["assignments"].items():
            all_assignments[patient_idx + start] = day
        total_cost += result["total_expected_cost"]

    return {
        "assignments": all_assignments,
        "status": status,
        "total_expected_cost": total_cost,
    }


def _batched_ilp_global(survival_curves, global_capacity, batch_size, total_cap_per_day):
    """Run ILP (global) in batches."""
    from policy.ilp_scheduler import schedule_ilp

    n = len(survival_curves)
    all_assignments = {}
    total_cost = 0.0
    status = "Optimal"

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_curves = survival_curves[start:end]
        batch_n = end - start

        # Scale capacity proportionally to batch size
        batch_cap_val = max(
            int(np.ceil(batch_n / HORIZON_DAYS)),
            int(np.ceil(total_cap_per_day * batch_n / n)),
        )
        batch_cap = np.full(HORIZON_DAYS, batch_cap_val)

        result = schedule_ilp(
            batch_curves, capacity_per_day=batch_cap, time_limit=120
        )

        if result["status"] != "Optimal":
            status = result["status"]

        for patient_idx, day in result["assignments"].items():
            all_assignments[patient_idx + start] = day
        total_cost += result["total_expected_cost"]

    return {
        "assignments": all_assignments,
        "status": status,
        "total_expected_cost": total_cost,
    }


def step_report(scheduling_results, model_info, cohort):
    """Generate summary report."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    n = len(cohort)

    # Model performance
    print("\n--- Model Performance ---")
    for name in ["gbm", "cox"]:
        key = f"{name}_metrics"
        if key in model_info:
            m = model_info[key]
            print(f"  {name.upper():>6}: C-index={m['c_index']:.4f}, IBS={m['ibs']:.4f}")

    # Scheduling comparison
    print("\n--- Scheduling Policy Comparison ---")
    print(f"  {'Policy':<25} {'Status':<12} {'Total Cost':>15} {'Avg Cost/Patient':>18}")
    print(f"  {'-'*25} {'-'*12} {'-'*15} {'-'*18}")

    for name, result in scheduling_results.items():
        status = result.get("status", "N/A")
        cost = result.get("total_expected_cost", float("inf"))
        avg = cost / n if cost < float("inf") else float("inf")
        print(f"  {name:<25} {status:<12} {cost:>15,.0f} {avg:>18,.2f}")

    # Specialty utilisation (if available)
    if "ilp_specialty" in scheduling_results and "utilisation" in scheduling_results["ilp_specialty"]:
        print("\n--- Specialty Pool Utilisation (ILP) ---")
        util = scheduling_results["ilp_specialty"]["utilisation"]
        for pool_name, stats in util.items():
            print(f"  {pool_name:<20}: {stats['total_assigned']:>6,} / {stats['total_capacity']:>6,} "
                  f"({stats['utilisation_pct']:.1f}%), peak={stats['peak_day_count']}/day")

    # Save results
    output = {
        "cohort_size": n,
        "model_performance": model_info,
        "scheduling_results": {
            name: {
                "status": r.get("status", "N/A"),
                "total_expected_cost": r.get("total_expected_cost", None),
                "n_assigned": len(r.get("assignments", {})),
            }
            for name, r in scheduling_results.items()
        },
    }

    results_path = PROCESSED_DIR / "pipeline_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")


def main():
    args = parse_args()
    t_start = time.time()

    print("=" * 60)
    print("CCPFS Pipeline — MIMIC-IV Real Data")
    print("=" * 60)

    # Step 1: Cohort
    cohort = step_cohort(args)

    # Step 2: Features
    X, feature_names = step_features(args, cohort)

    # Optional: subsample for testing (after features so indices match)
    if args.max_patients and len(cohort) > args.max_patients:
        print(f"\n  Subsampling to {args.max_patients:,} patients...")
        rng = np.random.RandomState(42)
        idx = rng.choice(len(cohort), size=args.max_patients, replace=False)
        idx.sort()
        cohort = cohort[idx.tolist()]
        X = X[idx]

    # Step 3: Train models
    model_results = step_train(args, X, feature_names, cohort)

    # Step 4: Schedule on TEST SET with properly scaled capacity
    # The full cohort spans 12+ years. For scheduling evaluation, we use the
    # test set (27K patients) with capacity scaled to simulate batches of
    # patients being scheduled over realistic time windows.
    test_mask = model_results["masks"]["test"]
    test_curves = model_results["gbm_curves_test"]
    test_cohort = cohort.filter(pl.Series(test_mask))

    scheduling_results = step_schedule(args, test_curves, test_cohort)

    # Step 5: Report
    step_report(scheduling_results, model_results["models_info"], test_cohort)

    elapsed = time.time() - t_start
    print(f"\n  Total pipeline time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
