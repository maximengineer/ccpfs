"""
Train GBM survival model on pre-extracted MOTOR embeddings.

Takes the representations pickle from femr_compute_representations
and trains a GBM survival model, outputting curves compatible with
merge_results() in parallel_train.py.

Usage:
    python -m models.motor.train_on_embeddings \
        --embeddings-path data/meds/motor_representations.pkl \
        --cohort-path data/processed/cohort.parquet \
        --output-dir data/processed/motor_output
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import HORIZON_DAYS, RANDOM_SEED


def load_embeddings(embeddings_path: Path):
    """Load MOTOR representations from pickle file."""
    with open(embeddings_path, "rb") as f:
        data = pickle.load(f)

    print(f"  Loaded embeddings: {data['representations'].shape}")
    print(f"  Patient IDs: {len(data['patient_ids'])}")
    print(f"  Prediction times: {len(data['prediction_times'])}")

    return data


def align_embeddings_with_cohort(embeddings: dict, cohort: pl.DataFrame):
    """Align MOTOR embeddings with cohort data to get train/val/test splits.

    MOTOR representations are ordered by (patient_id, prediction_time).
    We need to match these with our cohort's split assignments and outcomes.
    """
    reprs = embeddings["representations"]
    pids = embeddings["patient_ids"]
    pred_times = embeddings["prediction_times"]

    # Create lookup: (subject_id, discharge_time_str) -> embedding index
    emb_lookup = {}
    for i in range(len(pids)):
        pid = int(pids[i])
        # Handle different prediction_time formats
        pt = pred_times[i]
        if hasattr(pt, "strftime"):
            pt_str = pt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            pt_str = str(pt)
        emb_lookup[(pid, pt_str)] = i

    # Match cohort rows to embeddings
    train_idx, val_idx, test_idx = [], [], []
    train_emb, val_emb, test_emb = [], [], []

    matched = 0
    for row in cohort.iter_rows(named=True):
        pid = row["subject_id"]
        dt = row["discharge_time"]
        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S") if hasattr(dt, "strftime") else str(dt)

        emb_i = emb_lookup.get((pid, dt_str))
        if emb_i is None:
            continue

        matched += 1
        split = row["data_split"]

        if split == "train":
            train_idx.append(row)
            train_emb.append(reprs[emb_i])
        elif split == "tuning":
            val_idx.append(row)
            val_emb.append(reprs[emb_i])
        elif split == "held_out":
            test_idx.append(row)
            test_emb.append(reprs[emb_i])

    print(f"  Matched {matched}/{len(cohort)} cohort entries to embeddings")
    print(f"  Train: {len(train_emb)}, Val: {len(val_emb)}, Test: {len(test_emb)}")

    def to_arrays(rows, embs):
        X = np.array(embs, dtype=np.float32)
        events = np.array([r["event_indicator"] for r in rows])
        times = np.array([r["time_to_readmission"] for r in rows])
        return X, events, times

    X_train, e_train, t_train = to_arrays(train_idx, train_emb)
    X_val, e_val, t_val = to_arrays(val_idx, val_emb)
    X_test, e_test, t_test = to_arrays(test_idx, test_emb)

    return (X_train, e_train, t_train), (X_val, e_val, t_val), (X_test, e_test, t_test)


def make_structured_target(event_indicators, time_to_event):
    """Convert to scikit-survival structured array."""
    y = np.zeros(len(event_indicators), dtype=[("event", bool), ("time", float)])
    y["event"] = event_indicators.astype(bool)
    y["time"] = time_to_event
    return y


def train_survival_model(X_train, y_train, X_val, y_val, subsample=50000):
    """Train GBM survival model on MOTOR embeddings."""
    from sklearn.model_selection import ParameterGrid
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored

    # Subsample training data if needed
    if subsample and len(X_train) > subsample:
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(len(X_train), size=subsample, replace=False)
        X_fit = X_train[idx]
        y_fit = y_train[idx]
    else:
        X_fit = X_train
        y_fit = y_train

    param_grid = {
        "n_estimators": [300],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "min_samples_leaf": [20],
        "subsample": [0.8],
    }

    grid = list(ParameterGrid(param_grid))
    print(f"  GBM grid search: {len(grid)} combos on {X_fit.shape}")

    best_ci = -1.0
    best_params = None
    best_model = None

    for i, params in enumerate(grid):
        model = GradientBoostingSurvivalAnalysis(random_state=RANDOM_SEED, **params)
        model.fit(X_fit, y_fit)
        pred = model.predict(X_val)
        ci = concordance_index_censored(y_val["event"], y_val["time"], pred)[0]
        print(f"    [{i+1}/{len(grid)}] {params} -> C-index: {ci:.4f}")

        if ci > best_ci:
            best_ci = ci
            best_params = params
            best_model = model

    print(f"  Best MOTOR+GBM: C-index={best_ci:.4f}")
    return best_params, best_model, y_fit


def extract_survival_curves(model, X, horizon=HORIZON_DAYS):
    """Extract S(t) curves from GBM model."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from models import extract_curves_from_step_functions

    surv_fns = model.predict_survival_function(X)
    return extract_curves_from_step_functions(surv_fns, horizon)


def main():
    parser = argparse.ArgumentParser(description="Train survival on MOTOR embeddings")
    parser.add_argument("--embeddings-path", type=Path, required=True)
    parser.add_argument("--cohort-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subsample", type=int, default=50000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading MOTOR embeddings...")
    embeddings = load_embeddings(args.embeddings_path)

    print("\n[2/4] Aligning with cohort...")
    cohort = pl.read_parquet(args.cohort_path)
    train_data, val_data, test_data = align_embeddings_with_cohort(embeddings, cohort)

    X_train, e_train, t_train = train_data
    X_val, e_val, t_val = val_data
    X_test, e_test, t_test = test_data

    y_train = make_structured_target(e_train, t_train)
    y_val = make_structured_target(e_val, t_val)
    y_test = make_structured_target(e_test, t_test)

    print(f"\n[3/4] Training GBM on MOTOR embeddings...")
    t0 = time.time()
    best_params, model, y_train_fit = train_survival_model(
        X_train, y_train, X_val, y_val, subsample=args.subsample
    )
    train_time = time.time() - t0

    print(f"\n[4/4] Extracting survival curves and evaluating...")
    curves = extract_survival_curves(model, X_test)

    from models.evaluate_model import evaluate_survival_model
    metrics = evaluate_survival_model(y_train_fit, y_test, e_test, t_test, curves, "MOTOR+GBM")

    # Save results
    result = {
        "model": "motor",
        "params": best_params,
        "c_index": metrics["c_index"],
        "ibs": metrics["ibs"],
        "elapsed": train_time,
        "embedding_dim": int(X_train.shape[1]),
    }

    np.savez_compressed(args.output_dir / "motor_curves.npz", curves=curves)
    with open(args.output_dir / "motor_result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  MOTOR+GBM Results:")
    print(f"    C-index: {metrics['c_index']:.4f}")
    print(f"    IBS:     {metrics['ibs']:.4f}")
    print(f"    Embedding dim: {X_train.shape[1]}")
    print(f"    Train time: {train_time:.0f}s")
    print(f"  Saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
