"""
demo_setup.py
-------------
Generate synthetic patient data and pre-computed scheduling results so that
the dashboard and API can run without MIMIC-IV credentialed access.

Default output directory is `data/demo/` so the demo never collides with real
MIMIC-IV artefacts in `data/processed/`. Every file written is tagged
`"synthetic": true` so it cannot be mistaken for real pipeline output.

Usage:
    cd scheduling_follow_up
    python demo_setup.py                                          # writes data/demo/
    docker compose -f docker-compose.demo.yml up                  # dashboard at :3000

    python demo_setup.py --clean                                  # remove data/demo/
    python demo_setup.py --n-patients 25000 --seed 7              # larger cohort

The framework code (scheduling algorithms, metrics, calibration) is identical
to the production pipeline - only the patient data is synthetic.
"""

import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from config import DEFAULT_SPECIALTY_CAPACITY, HORIZON_DAYS, RANDOM_SEED, SPECIALTY_NAMES
from evaluation.metrics import event_before_followup_rate
from evaluation.synthetic import generate_synthetic_cohort
from policy.baselines import (
    guideline_capacity_policy,
    guideline_policy,
    risk_bucket_policy,
    unconstrained_optimal_policy,
    uniform_capacity_policy,
    uniform_policy,
)
from policy.mincost_solver import schedule_mincost_global, schedule_mincost_specialty
from policy.specialty_scheduler import schedule_greedy_specialty

# Demo output directory - kept separate from data/processed/ so we never
# overwrite real MIMIC-IV pipeline artefacts.
DEMO_DIR = Path(__file__).parent / "data" / "demo"

# MIMIC-IV specialty distribution (match for realism)
SPECIALTY_SHARES = {
    "cardiology": 0.197,
    "neurology": 0.137,
    "surgery": 0.115,
    "general_medicine": 0.551,
}

DEFAULT_N_PATIENTS = 10_000


def build_synthetic_cohort(n_patients: int, seed: int) -> dict:
    """Generate synthetic cohort and assign specialty + splits."""
    print(f"  Generating {n_patients:,} synthetic patients via Weibull hazard model...")

    raw = generate_synthetic_cohort(n_patients=n_patients, seed=seed)
    curves = raw["survival_curves"]           # (N, H+1)
    events = raw["event_indicators"]          # (N,)
    times = raw["event_times"]                # (N,)
    is_hf = raw["is_heart_failure"]           # (N,)

    rng = np.random.default_rng(seed)

    # Assign specialty pools proportionally (shuffle to avoid block structure)
    pool_assignment = np.empty(n_patients, dtype=np.int64)
    idx = 0
    for name, share in SPECIALTY_SHARES.items():
        k = SPECIALTY_NAMES.index(name)
        n = int(round(n_patients * share))
        pool_assignment[idx:idx + n] = k
        idx += n
    # Fill any remaining from rounding
    pool_assignment[idx:] = SPECIALTY_NAMES.index("general_medicine")
    rng.shuffle(pool_assignment)

    # Splits: 80% train / 10% tuning / 10% held_out
    split_labels = np.array(["train"] * n_patients, dtype=object)
    perm = rng.permutation(n_patients)
    val_cutoff = int(n_patients * 0.8)
    test_cutoff = int(n_patients * 0.9)
    split_labels[perm[val_cutoff:test_cutoff]] = "tuning"
    split_labels[perm[test_cutoff:]] = "held_out"

    # Synthetic but unique IDs
    subject_ids = 10_000_000 + rng.permutation(n_patients)
    hadm_ids = 20_000_000 + rng.permutation(n_patients)

    # Synthetic discharge dates spread across 2015
    base = datetime(2015, 1, 1)
    day_offsets = rng.integers(0, 365, size=n_patients)
    discharge_times = [base + timedelta(days=int(d)) for d in day_offsets]

    n_events = int(events.sum())
    mix = ", ".join(
        f"{name}={int((pool_assignment == SPECIALTY_NAMES.index(name)).sum())}"
        for name in SPECIALTY_NAMES
    )
    print(f"    Event rate: {n_events / n_patients * 100:.1f}% ({n_events} events)")
    print(f"    Specialty mix: {mix}")

    cohort_df = pl.DataFrame({
        "subject_id": subject_ids,
        "hadm_id": hadm_ids,
        "data_split": split_labels.tolist(),
        "specialty_pool": pool_assignment,
        "event_indicator": events.astype(np.int64),
        "time_to_readmission": times.astype(np.float64),
        "discharge_time": discharge_times,
        "is_heart_failure": is_hf.astype(bool),
    })

    return {
        "df": cohort_df,
        "curves": curves,
        "events": events,
        "times": times,
        "pools": pool_assignment,
        "is_hf": is_hf,
    }


def slice_test(cohort: dict) -> dict:
    """Extract test-split arrays for scheduling + API."""
    df = cohort["df"]
    test_mask = (df["data_split"] == "held_out").to_numpy()
    return {
        "curves": cohort["curves"][test_mask],
        "events": cohort["events"][test_mask],
        "times": cohort["times"][test_mask],
        "pools": cohort["pools"][test_mask],
        "is_hf": cohort["is_hf"][test_mask],
    }


def assignments_to_days(assignments: dict, n: int) -> np.ndarray:
    """Convert dict{patient_idx: day} to day array of length n."""
    days = np.zeros(n, dtype=np.int64)
    for i, d in assignments.items():
        days[int(i)] = int(d)
    return days


def run_all_policies(test: dict) -> dict:
    """Execute every scheduling policy the API expects."""
    curves = test["curves"]
    pools = test["pools"]
    events = test["events"]
    times = test["times"]
    is_hf = test["is_hf"]
    n = len(curves)

    results = {}

    def finalise(name: str, out: dict):
        assigns = out["assignments"]
        days = assignments_to_days(assigns, n)
        ebf = event_before_followup_rate(assigns, times, events)
        results[name] = {
            "days": days,
            "cost": out["total_expected_cost"],
            "status": out["status"],
            "n_assigned": len(assigns),
            "ebf": ebf,
        }
        print(f"    {name:25s}: cost/pt={out['total_expected_cost']/n:7.0f}  catch={ebf['catch_rate']*100:5.1f}%")

    print("  Running baseline policies...")
    finalise("uniform_d14", uniform_policy(curves, day=14))
    finalise("risk_bucket", risk_bucket_policy(curves))
    finalise("guideline", guideline_policy(curves, is_heart_failure=is_hf))
    finalise("unconstrained", unconstrained_optimal_policy(curves))
    finalise("uniform_d14_cap", uniform_capacity_policy(curves, pools, day=14))
    finalise("guideline_cap", guideline_capacity_policy(curves, pools, is_heart_failure=is_hf))

    print("  Running greedy schedulers...")
    finalise("greedy_specialty", schedule_greedy_specialty(curves, pools))
    # Global greedy: pretend everyone belongs to pool 0 with total capacity across specialties
    all_one = np.zeros_like(pools)
    global_daily_total = int(sum(DEFAULT_SPECIALTY_CAPACITY.values()))
    finalise(
        "greedy_global",
        schedule_greedy_specialty(curves, all_one, capacity_per_specialty_day={0: global_daily_total}),
    )

    print("  Running min-cost solvers (may take a few seconds)...")
    finalise("mincost_specialty", schedule_mincost_specialty(curves, pools))
    global_daily_cap = np.full(HORIZON_DAYS, global_daily_total, dtype=np.int64)
    finalise("mincost_global", schedule_mincost_global(curves, capacity_per_day=global_daily_cap))

    return results


def write_scheduling_npz(results: dict, out_path: Path):
    """Match the .npz structure api/routers/scheduling.py expects."""
    out = {}
    for name, r in results.items():
        out[f"{name}_days"] = r["days"]
        out[f"{name}_cost"] = np.float64(r["cost"])
        out[f"{name}_status"] = np.array(r["status"], dtype="U64")
    out["policy_names"] = np.array(list(results.keys()), dtype="U32")
    np.savez_compressed(out_path, **out)


def write_pipeline_results(cohort: dict, test: dict, policies: dict, out_path: Path):
    """Write the JSON the API loads for the Overview page."""
    n_test = len(test["curves"])
    n_total = len(cohort["df"])

    # Policy summary = {name: {total_expected_cost, n_assigned, status, ebf}}
    sched_for_json = {}
    for name, r in policies.items():
        sched_for_json[name] = {
            "status": r["status"],
            "total_expected_cost": float(r["cost"]),
            "n_assigned": int(r["n_assigned"]),
            "ebf": {
                "ebf_rate": float(r["ebf"]["ebf_rate"]),
                "ebf_count": int(r["ebf"]["ebf_count"]),
                "total_events": int(r["ebf"]["total_events"]),
                "events_caught": int(r["ebf"]["events_caught"]),
                "catch_rate": float(r["ebf"]["catch_rate"]),
            },
        }

    data = {
        "synthetic": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cohort_size": n_total,
        "model_performance": {
            "gbm_metrics": {"c_index": 0.704, "ibs": 0.099},
            "gbm_params": {"learning_rate": 0.1, "max_depth": 5, "n_estimators": 200, "note": "synthetic demo"},
            "cox_metrics": {"c_index": 0.698, "ibs": 0.100},
            "rsf_metrics": {"c_index": 0.692, "ibs": 0.101},
            "rsf_params": {"max_depth": 5, "n_estimators": 200, "note": "synthetic demo"},
            "best_model": "gbm",
        },
        "scheduling_results": sched_for_json,
    }

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def write_models_info(out_path: Path):
    data = {
        "synthetic": True,
        "best_model": "gbm",
        "gbm_metrics": {"c_index": 0.704, "ibs": 0.099},
        "cox_metrics": {"c_index": 0.698, "ibs": 0.100},
        "rsf_metrics": {"c_index": 0.692, "ibs": 0.101},
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def write_motor_placeholder(test: dict, out_dir: Path, seed: int):
    """MOTOR-like curves with extra noise (simulates domain shift)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 999)
    # Add noise to shift the curves slightly (simulates weaker transfer model)
    noisy = np.clip(test["curves"] + rng.normal(0, 0.03, size=test["curves"].shape), 0.0, 1.0)
    # Enforce monotonic non-increasing per patient
    noisy = np.minimum.accumulate(noisy, axis=1)
    np.savez_compressed(out_dir / "motor_curves.npz", curves=noisy)

    with open(out_dir / "motor_result.json", "w") as f:
        json.dump({
            "synthetic": True,
            "model": "MOTOR+GBM (synthetic demo)",
            "c_index": 0.672,
            "ibs": 0.104,
            "note": "Placeholder for demo mode. Real MOTOR results require GPU + MIMIC-IV access.",
        }, f, indent=2)


def _looks_like_real_data(out_dir: Path) -> bool:
    """Return True if out_dir contains a non-synthetic pipeline_results.json."""
    pr = out_dir / "pipeline_results.json"
    if not pr.exists():
        return False
    try:
        with open(pr) as f:
            data = json.load(f)
        return not bool(data.get("synthetic", False))
    except (json.JSONDecodeError, OSError):
        return False


def clean(out_dir: Path) -> int:
    """Remove the demo directory. Refuses to delete non-synthetic data."""
    if not out_dir.exists():
        print(f"Nothing to clean - {out_dir} does not exist.")
        return 0
    if _looks_like_real_data(out_dir):
        print(f"ERROR: {out_dir} contains non-synthetic data (pipeline_results.json "
              f"is not marked synthetic). Refusing to delete.", file=sys.stderr)
        return 2
    shutil.rmtree(out_dir)
    print(f"Removed {out_dir.resolve()}")
    return 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic demo data for CCPFS dashboard")
    parser.add_argument("--n-patients", type=int, default=DEFAULT_N_PATIENTS,
                        help=f"Total synthetic patients (default: {DEFAULT_N_PATIENTS:,})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Random seed for reproducibility")
    parser.add_argument("--out-dir", type=str, default=str(DEMO_DIR),
                        help=f"Output directory (default: {DEMO_DIR.relative_to(Path(__file__).parent)})")
    parser.add_argument("--clean", action="store_true",
                        help="Remove the demo output directory and exit")
    parser.add_argument("--force", action="store_true",
                        help="Allow writing into a directory that contains real (non-synthetic) data")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.clean:
        sys.exit(clean(out_dir))

    # Safety: never silently clobber real MIMIC-IV pipeline output
    if _looks_like_real_data(out_dir) and not args.force:
        print(f"ERROR: {out_dir} already contains real (non-synthetic) pipeline data.\n"
              f"Refusing to overwrite. Either:\n"
              f"  - omit --out-dir to write to the default demo folder ({DEMO_DIR}), or\n"
              f"  - pass --force if you really mean to overwrite real data.",
              file=sys.stderr)
        sys.exit(2)

    out_dir.mkdir(parents=True, exist_ok=True)
    motor_dir = out_dir / "motor_output"

    print(f"\nCCPFS Demo Setup")
    print(f"================")
    print(f"Output: {out_dir.resolve()}")
    print(f"Patients: {args.n_patients:,}   Seed: {args.seed}\n")

    print("[1/5] Building synthetic cohort...")
    cohort = build_synthetic_cohort(args.n_patients, args.seed)
    cohort["df"].write_parquet(out_dir / "cohort.parquet")
    print(f"    Wrote {out_dir / 'cohort.parquet'}")

    print("\n[2/5] Extracting test-split curves...")
    test = slice_test(cohort)
    n_test = len(test["curves"])
    print(f"    Test set: {n_test:,} patients, {int(test['events'].sum())} events")

    np.savez_compressed(
        out_dir / "curves_test.npz",
        curves_test=test["curves"],
        e_test=test["events"].astype(np.int64),
        t_test=test["times"].astype(np.float64),
        model_name=np.array("synthetic_demo", dtype="U32"),
        calibrated=np.array(True),
    )
    print(f"    Wrote {out_dir / 'curves_test.npz'}")

    print("\n[3/5] Running all scheduling policies...")
    policies = run_all_policies(test)

    print("\n[4/5] Writing scheduling_results.npz + summary JSONs...")
    write_scheduling_npz(policies, out_dir / "scheduling_results.npz")
    print(f"    Wrote {out_dir / 'scheduling_results.npz'}")
    write_pipeline_results(cohort, test, policies, out_dir / "pipeline_results.json")
    print(f"    Wrote {out_dir / 'pipeline_results.json'}")
    write_models_info(out_dir / "models_info.json")
    print(f"    Wrote {out_dir / 'models_info.json'}")

    print("\n[5/5] Writing MOTOR placeholder...")
    write_motor_placeholder(test, motor_dir, args.seed)
    print(f"    Wrote {motor_dir}/motor_curves.npz and motor_result.json")

    print("\nDone. Demo data ready.")
    print("\nNext steps:")
    print("  docker compose -f docker-compose.demo.yml up --build")
    print("  Open http://localhost:3000")
    print("\nTo remove the demo data later:")
    print("  python demo_setup.py --clean")
    print("\nAll artifacts marked with synthetic=true. To run on real MIMIC-IV data,")
    print("see README.md Access Requirements section.\n")


if __name__ == "__main__":
    main()
