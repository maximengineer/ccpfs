"""
MOTOR Foundation Model — Embedding Extraction & Survival Prediction
--------------------------------------------------------------------
Uses the pretrained MOTOR-T-Base (143M params) from Stanford Shah Lab
to extract patient representations at discharge time, then fits a
GBM survival model on those embeddings.

This gives us a 4th model (MOTOR+GBM) alongside the classical
GBM/Cox/RSF trained on hand-crafted features, demonstrating
the framework's model-agnostic design.

Requirements:
    pip install femr xformers meds_reader torch transformers
    GPU with >= 16GB VRAM (NVIDIA L4 recommended)

Usage:
    python -m models.motor.train_motor \
        --meds-db data/meds/MEDS_reader_db \
        --model-path motor-t-base \
        --output-dir data/processed/motor_output
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    HORIZON_DAYS,
    MEDS_BASE_DIR,
    MODEL_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
)


def convert_meds_for_reader(meds_base_dir: Path, output_dir: Path, num_threads: int = 4):
    """Convert MEDS parquet data to meds_reader indexed format.

    meds_reader needs a specific directory layout with an index.
    This runs `meds_reader_convert` to create it.
    """
    import subprocess

    if (output_dir / "metadata").exists():
        print(f"  meds_reader DB already exists at {output_dir}, skipping conversion.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "meds_reader",
        str(meds_base_dir),
        str(output_dir),
        "--num_threads", str(num_threads),
    ]
    print(f"  Converting MEDS data for meds_reader: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # Try alternative: meds_reader_convert CLI
        cmd2 = [
            "meds_reader_convert",
            str(meds_base_dir),
            str(output_dir),
            "--num_threads", str(num_threads),
        ]
        print(f"  Trying CLI: {' '.join(cmd2)}")
        result = subprocess.run(cmd2, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[:500]}")
        raise RuntimeError(f"meds_reader conversion failed (exit {result.returncode})")

    print(f"  Conversion done → {output_dir}")


def load_cohort_discharge_times(processed_dir: Path):
    """Load cohort and extract discharge times for label creation.

    Returns dict mapping subject_id → discharge_datetime for the test set.
    """
    cohort = pl.read_parquet(processed_dir / "cohort.parquet")

    # We need discharge times for creating MOTOR prediction labels.
    # The cohort has discharge_time and time_to_readmission.
    test = cohort.filter(pl.col("data_split") == "held_out")
    train = cohort.filter(pl.col("data_split") == "train")
    val = cohort.filter(pl.col("data_split") == "tuning")

    return {
        "train": train,
        "val": val,
        "test": test,
    }


def create_labels(cohort_df: pl.DataFrame):
    """Create meds.Label objects from cohort dataframe.

    Each label represents a prediction point at discharge time.
    """
    import meds

    labels = []
    for row in cohort_df.iter_rows(named=True):
        pred_time = row["discharge_time"]
        labels.append(
            meds.Label(
                subject_id=row["subject_id"],
                prediction_time=pred_time,
                boolean_value=bool(row["event_indicator"]),
            )
        )
    return labels


def extract_motor_embeddings(
    db_path: Path,
    model_path: Path,
    labels: list,
    batch_size: int = 512,
    device: str = "cuda",
) -> np.ndarray:
    """Extract MOTOR representations at each label's prediction time.

    Returns array of shape (n_labels, hidden_size).
    """
    import femr.models.transformer

    print(f"  Extracting MOTOR embeddings for {len(labels)} patients...")
    print(f"  Model: {model_path}")
    print(f"  Device: {device}")

    t0 = time.time()
    result = femr.models.transformer.compute_features(
        db=str(db_path),
        model_path=str(model_path),
        labels=labels,
        device=torch.device(device),
        tokens_per_batch=batch_size,
    )
    elapsed = time.time() - t0
    print(f"  Extracted {result['features'].shape} embeddings in {elapsed:.1f}s")

    return result["features"]  # shape: (N, hidden_size)


def train_survival_on_embeddings(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
):
    """Train GBM survival model on MOTOR embeddings.

    Returns (best_params, model).
    """
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from sklearn.model_selection import ParameterGrid

    param_grid = {
        "n_estimators": [300],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "min_samples_leaf": [20],
        "subsample": [0.8],
    }

    grid = list(ParameterGrid(param_grid))
    print(f"  GBM grid search on MOTOR embeddings: {len(grid)} combos")

    best_ci = -1.0
    best_params = None
    best_model = None

    for i, params in enumerate(grid):
        model = GradientBoostingSurvivalAnalysis(random_state=RANDOM_SEED, **params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        ci = concordance_index_censored(y_val["event"], y_val["time"], pred)[0]
        print(f"    [{i+1}/{len(grid)}] {params} -> C-index: {ci:.4f}")

        if ci > best_ci:
            best_ci = ci
            best_params = params
            best_model = model

    print(f"  Best MOTOR+GBM: C-index={best_ci:.4f}, params={best_params}")
    return best_params, best_model


def extract_survival_curves(model, X: np.ndarray, horizon: int = HORIZON_DAYS):
    """Extract S(t) curves from GBM model trained on MOTOR embeddings."""
    from models import extract_curves_from_step_functions

    surv_fns = model.predict_survival_function(X)
    return extract_curves_from_step_functions(surv_fns, horizon)


def make_structured_target(event_indicators, time_to_event):
    """Convert to scikit-survival structured array."""
    y = np.zeros(len(event_indicators), dtype=[("event", bool), ("time", float)])
    y["event"] = event_indicators.astype(bool)
    y["time"] = time_to_event
    return y


def run_motor_pipeline(
    meds_db_path: Path,
    motor_model_path: Path,
    output_dir: Path,
    device: str = "cuda",
    subsample: int = None,
):
    """End-to-end MOTOR pipeline: embeddings → survival model → curves.

    Saves results compatible with merge_results() in parallel_train.py.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load cohort splits
    print("\n[1/5] Loading cohort...")
    splits = load_cohort_discharge_times(PROCESSED_DIR)
    print(f"  Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    # 2. Create labels for each split
    print("\n[2/5] Creating prediction labels...")
    train_labels = create_labels(splits["train"])
    val_labels = create_labels(splits["val"])
    test_labels = create_labels(splits["test"])

    # 3. Extract MOTOR embeddings
    print("\n[3/5] Extracting MOTOR embeddings...")
    X_train = extract_motor_embeddings(meds_db_path, motor_model_path, train_labels, device=device)
    X_val = extract_motor_embeddings(meds_db_path, motor_model_path, val_labels, device=device)
    X_test = extract_motor_embeddings(meds_db_path, motor_model_path, test_labels, device=device)

    # Build structured targets
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    e_train = train_df["event_indicator"].to_numpy()
    t_train = train_df["time_to_readmission"].to_numpy()
    e_val = val_df["event_indicator"].to_numpy()
    t_val = val_df["time_to_readmission"].to_numpy()
    e_test = test_df["event_indicator"].to_numpy()
    t_test = test_df["time_to_readmission"].to_numpy()

    y_train = make_structured_target(e_train, t_train)
    y_val = make_structured_target(e_val, t_val)
    y_test = make_structured_target(e_test, t_test)

    # Subsample training if requested
    if subsample and len(X_train) > subsample:
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(len(X_train), size=subsample, replace=False)
        X_train_fit = X_train[idx]
        y_train_fit = y_train[idx]
    else:
        X_train_fit = X_train
        y_train_fit = y_train

    # 4. Train survival model on embeddings
    print(f"\n[4/5] Training GBM on MOTOR embeddings ({X_train_fit.shape})...")
    t0 = time.time()
    best_params, model = train_survival_on_embeddings(X_train_fit, y_train_fit, X_val, y_val)
    train_time = time.time() - t0

    # 5. Extract curves and evaluate
    print("\n[5/5] Extracting survival curves and evaluating...")
    curves = extract_survival_curves(model, X_test)

    from models.evaluate_model import evaluate_survival_model
    metrics = evaluate_survival_model(y_train_fit, y_test, e_test, t_test, curves, "MOTOR+GBM")

    # Save results in same format as parallel_train.py
    result = {
        "model": "motor",
        "params": best_params,
        "c_index": metrics["c_index"],
        "ibs": metrics["ibs"],
        "elapsed": train_time,
        "embedding_dim": int(X_train.shape[1]),
    }

    np.savez_compressed(output_dir / "motor_curves.npz", curves=curves)
    with open(output_dir / "motor_result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Also save embeddings for potential reuse
    np.savez_compressed(
        output_dir / "motor_embeddings.npz",
        X_train=X_train, X_val=X_val, X_test=X_test,
        e_test=e_test, t_test=t_test,
    )

    print(f"\n{'='*60}")
    print(f"  MOTOR+GBM Results:")
    print(f"    C-index: {metrics['c_index']:.4f}")
    print(f"    IBS:     {metrics['ibs']:.4f}")
    print(f"    Embedding dim: {X_train.shape[1]}")
    print(f"    Train time: {train_time:.0f}s")
    print(f"  Saved to: {output_dir}")
    print(f"{'='*60}")

    return result


def main():
    parser = argparse.ArgumentParser(description="MOTOR pipeline for CCPFS")
    parser.add_argument("--meds-db", type=Path, required=True,
                        help="Path to meds_reader converted database")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Path to MOTOR pretrained model (HuggingFace dir)")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR / "motor_output",
                        help="Output directory for results")
    parser.add_argument("--device", default="cuda", help="torch device")
    parser.add_argument("--subsample", type=int, default=50000,
                        help="Subsample training data (0 = no subsample)")
    args = parser.parse_args()

    run_motor_pipeline(
        meds_db_path=args.meds_db,
        motor_model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        subsample=args.subsample if args.subsample > 0 else None,
    )


if __name__ == "__main__":
    main()
