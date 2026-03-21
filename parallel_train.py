"""
Parallel model training — runs GBM, Cox PH, and RSF concurrently.

GBM and Cox are single-threaded; RSF uses n_jobs=-1.
Running all 3 in parallel utilises all CPU cores and cuts wall time
from (GBM + Cox + RSF) to max(GBM, Cox, RSF).

Usage:
    python parallel_train.py                # Full 220K training set
    python parallel_train.py --fast-grid    # Reduced GBM grid
    python parallel_train.py --subsample 50000  # Quick test
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from config import PROCESSED_DIR, RANDOM_SEED


def load_data(subsample: int = None):
    """Load cohort + features and split into train/val/test."""
    from models.classical.train_gbm import make_structured_target

    cohort = pl.read_parquet(PROCESSED_DIR / "cohort.parquet")
    data = np.load(PROCESSED_DIR / "features.npz", allow_pickle=True)
    X = data["X"]
    feature_names = list(data["feature_names"])

    split_col = cohort["data_split"].to_numpy()
    event_indicators = cohort["event_indicator"].to_numpy()
    time_to_event = cohort["time_to_readmission"].to_numpy()

    train_mask = split_col == "train"
    val_mask = split_col == "tuning"
    test_mask = split_col == "held_out"

    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    e_train = event_indicators[train_mask]
    t_train = time_to_event[train_mask]
    e_val = event_indicators[val_mask]
    t_val = time_to_event[val_mask]
    e_test = event_indicators[test_mask]
    t_test = time_to_event[test_mask]

    if subsample and len(X_train) > subsample:
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(len(X_train), size=subsample, replace=False)
        X_train_fit = X_train[idx]
        e_train_fit, t_train_fit = e_train[idx], t_train[idx]
    else:
        X_train_fit, e_train_fit, t_train_fit = X_train, e_train, t_train

    y_train_fit = make_structured_target(e_train_fit, t_train_fit)
    y_val = make_structured_target(e_val, t_val)
    y_test = make_structured_target(e_test, t_test)

    return {
        "X_train_fit": X_train_fit, "e_train_fit": e_train_fit, "t_train_fit": t_train_fit,
        "X_val": X_val, "X_test": X_test,
        "e_test": e_test, "t_test": t_test,
        "y_train_fit": y_train_fit, "y_val": y_val, "y_test": y_test,
        "feature_names": feature_names,
    }


def train_gbm_worker(data, fast_grid, result_dir):
    """Train GBM in a subprocess."""
    try:
        from models.classical.train_gbm import train_gbm, extract_survival_curves, save_model
        from models.evaluate_model import evaluate_survival_model

        param_grid = None
        if fast_grid:
            param_grid = {
                "n_estimators": [300],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
                "min_samples_leaf": [20],
                "subsample": [0.8],
            }

        t0 = time.time()
        best_params, model = train_gbm(
            data["X_train_fit"], data["y_train_fit"],
            data["X_val"], data["y_val"],
            param_grid=param_grid,
        )
        elapsed = time.time() - t0
        print(f"  [GBM] Training done in {elapsed:.0f}s", flush=True)

        save_model(model)
        curves = extract_survival_curves(model, data["X_test"])
        metrics = evaluate_survival_model(
            data["y_train_fit"], data["y_test"],
            data["e_test"], data["t_test"], curves, "GBM",
        )

        result = {
            "model": "gbm",
            "params": best_params,
            "c_index": metrics["c_index"],
            "ibs": metrics["ibs"],
            "elapsed": elapsed,
        }
        np.savez_compressed(result_dir / "gbm_curves.npz", curves=curves)

        with open(result_dir / "gbm_result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"  [GBM] C-index={metrics['c_index']:.4f}, IBS={metrics['ibs']:.4f}", flush=True)
        return result
    except Exception:
        traceback.print_exc()
        return {"model": "gbm", "error": traceback.format_exc()}


def train_cox_worker(data, result_dir):
    """Train Cox PH in a subprocess."""
    try:
        from models.classical.train_cox import train_cox, extract_survival_curves_cox, save_model
        from models.evaluate_model import evaluate_survival_model

        t0 = time.time()
        model, scaler = train_cox(
            data["X_train_fit"], data["e_train_fit"], data["t_train_fit"],
            data["feature_names"],
        )
        elapsed = time.time() - t0
        print(f"  [Cox] Training done in {elapsed:.0f}s", flush=True)

        save_model(model, scaler=scaler)
        curves = extract_survival_curves_cox(
            model, data["X_test"], data["feature_names"], scaler=scaler,
        )
        metrics = evaluate_survival_model(
            data["y_train_fit"], data["y_test"],
            data["e_test"], data["t_test"], curves, "Cox PH",
        )

        result = {
            "model": "cox",
            "c_index": metrics["c_index"],
            "ibs": metrics["ibs"],
            "elapsed": elapsed,
        }
        np.savez_compressed(result_dir / "cox_curves.npz", curves=curves)

        with open(result_dir / "cox_result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"  [Cox] C-index={metrics['c_index']:.4f}, IBS={metrics['ibs']:.4f}", flush=True)
        return result
    except Exception:
        traceback.print_exc()
        return {"model": "cox", "error": traceback.format_exc()}


def train_rsf_worker(data, result_dir):
    """Train RSF in a subprocess."""
    try:
        from models.classical.train_rsf import train_rsf, extract_survival_curves_rsf, save_model
        from models.evaluate_model import evaluate_survival_model

        t0 = time.time()
        best_params, model = train_rsf(
            data["X_train_fit"], data["y_train_fit"],
            data["X_val"], data["y_val"],
        )
        elapsed = time.time() - t0
        print(f"  [RSF] Training done in {elapsed:.0f}s", flush=True)

        save_model(model)
        curves = extract_survival_curves_rsf(model, data["X_test"])
        metrics = evaluate_survival_model(
            data["y_train_fit"], data["y_test"],
            data["e_test"], data["t_test"], curves, "RSF",
        )

        result = {
            "model": "rsf",
            "params": best_params,
            "c_index": metrics["c_index"],
            "ibs": metrics["ibs"],
            "elapsed": elapsed,
        }
        np.savez_compressed(result_dir / "rsf_curves.npz", curves=curves)

        with open(result_dir / "rsf_result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"  [RSF] C-index={metrics['c_index']:.4f}, IBS={metrics['ibs']:.4f}", flush=True)
        return result
    except Exception:
        traceback.print_exc()
        return {"model": "rsf", "error": traceback.format_exc()}


def merge_results(result_dir, e_test, t_test):
    """Merge per-model results into the format step_train produces."""
    models_info = {}
    all_curves = {}
    best_name = None
    best_cindex = -1.0

    for name in ["gbm", "cox", "rsf"]:
        result_path = result_dir / f"{name}_result.json"
        curves_path = result_dir / f"{name}_curves.npz"

        if not result_path.exists():
            print(f"  [merge] {name} — no results found, skipping")
            continue

        with open(result_path) as f:
            result = json.load(f)

        if "error" in result:
            print(f"  [merge] {name} — FAILED: {result['error'][:200]}")
            continue

        models_info[f"{name}_metrics"] = {
            "c_index": result["c_index"],
            "ibs": result["ibs"],
        }
        if "params" in result:
            models_info[f"{name}_params"] = result["params"]

        curves_data = np.load(curves_path)
        all_curves[f"{name}_curves_test"] = curves_data["curves"]

        ci = result["c_index"]
        print(f"  [merge] {name}: C-index={ci:.4f}, IBS={result['ibs']:.4f}, time={result['elapsed']:.0f}s")

        if ci > best_cindex:
            best_cindex = ci
            best_name = name

    if best_name is None:
        print("ERROR: No models completed successfully.")
        sys.exit(1)

    models_info["best_model"] = best_name
    print(f"\n  Best model: {best_name.upper()} (C-index={best_cindex:.4f})")

    # Save merged curves_test.npz (same format as step_train)
    save_data = {
        "curves_test": all_curves[f"{best_name}_curves_test"],
        "model_name": best_name,
        "e_test": e_test,
        "t_test": t_test,
    }
    save_data.update(all_curves)
    np.savez_compressed(PROCESSED_DIR / "curves_test.npz", **save_data)
    print(f"  Saved merged curves to {PROCESSED_DIR / 'curves_test.npz'}")

    # Save models_info.json
    info_path = PROCESSED_DIR / "models_info.json"
    with open(info_path, "w") as f:
        json.dump(models_info, f, indent=2, default=str)
    print(f"  Saved model info to {info_path}")


def main():
    parser = argparse.ArgumentParser(description="Parallel model training")
    parser.add_argument("--fast-grid", action="store_true", help="Reduced GBM grid")
    parser.add_argument("--subsample", type=int, default=None, help="Subsample training data")
    args = parser.parse_args()

    print("=" * 60)
    print("CCPFS Parallel Training — GBM | Cox PH | RSF")
    print("=" * 60)

    # Load data once in the main process
    print("\n  Loading data...")
    t0 = time.time()
    data = load_data(args.subsample)
    print(f"  Data loaded in {time.time() - t0:.1f}s")
    print(f"  Train: {data['X_train_fit'].shape[0]:,}, Test: {data['X_test'].shape[0]:,}")
    print(f"  Features: {data['X_train_fit'].shape[1]}")

    # Temp directory for per-model results
    result_dir = PROCESSED_DIR / "parallel_tmp"
    result_dir.mkdir(exist_ok=True)

    # Launch 3 processes
    print(f"\n  Launching 3 parallel training processes...")
    t_start = time.time()

    # Use spawn to avoid fork issues with numpy/sklearn
    ctx = mp.get_context("fork")

    p_gbm = ctx.Process(target=train_gbm_worker, args=(data, args.fast_grid, result_dir))
    p_cox = ctx.Process(target=train_cox_worker, args=(data, result_dir))
    p_rsf = ctx.Process(target=train_rsf_worker, args=(data, result_dir))

    p_gbm.start()
    p_cox.start()
    p_rsf.start()

    print(f"  PIDs: GBM={p_gbm.pid}, Cox={p_cox.pid}, RSF={p_rsf.pid}")
    print(f"  Waiting for all models to complete...\n")

    p_gbm.join()
    print(f"  GBM process finished (exit code {p_gbm.exitcode})")
    p_cox.join()
    print(f"  Cox process finished (exit code {p_cox.exitcode})")
    p_rsf.join()
    print(f"  RSF process finished (exit code {p_rsf.exitcode})")

    total_time = time.time() - t_start
    print(f"\n  All models done in {total_time:.0f}s ({total_time/3600:.1f}h)")

    # Merge results
    print("\n  Merging results...")
    merge_results(result_dir, data["e_test"], data["t_test"])

    print(f"\n{'=' * 60}")
    print("  Training complete. Next: python run_pipeline.py --step calibrate")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
