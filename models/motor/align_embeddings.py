"""Align MOTOR embeddings with cohort by temporal proximity."""
import pickle
import polars as pl
import numpy as np
from datetime import timedelta
from collections import defaultdict

with open("data/processed/motor_output/motor_representations.pkl", "rb") as f:
    data = pickle.load(f)

pids = data["patient_ids"]
ptimes = data["prediction_times"]
reprs = data["representations"]

print(f"Embeddings: {reprs.shape}")
print(f"Unique emb pids: {len(set(pids))}")

c = pl.read_parquet("data/processed/cohort.parquet")
print(f"Cohort: {len(c)}")

# Build lookup: patient_id -> list of (emb_time, emb_index)
pid_to_embs = defaultdict(list)
for i in range(len(pids)):
    pid_to_embs[int(pids[i])].append((ptimes[i], i))

# Sort each patient's embeddings by time
for pid in pid_to_embs:
    pid_to_embs[pid].sort(key=lambda x: x[0])

# Match: for each cohort row, find the embedding closest to discharge
# that falls within [admission - 1day, discharge + 1day]
matched_emb_idx = []
matched_cohort_idx = []
no_match = 0

for row_idx, row in enumerate(c.iter_rows(named=True)):
    pid = row["subject_id"]
    adm = row["admission_time"]
    dis = row["discharge_time"]

    candidates = pid_to_embs.get(pid, [])

    best_idx = None
    best_diff = timedelta(days=999)

    for emb_time, emb_i in candidates:
        if emb_time >= adm - timedelta(days=1) and emb_time <= dis + timedelta(days=1):
            diff = abs(dis - emb_time)
            if diff < best_diff:
                best_diff = diff
                best_idx = emb_i

    if best_idx is not None:
        matched_emb_idx.append(best_idx)
        matched_cohort_idx.append(row_idx)
    else:
        no_match += 1

print(f"\nMatched: {len(matched_emb_idx)}/{len(c)} ({100*len(matched_emb_idx)/len(c):.1f}%)")
print(f"No match: {no_match}")

# Build aligned arrays
X_all = reprs[matched_emb_idx]
cohort_matched = c[matched_cohort_idx]

# Split
splits = {}
for split_name in ["train", "tuning", "held_out"]:
    mask = (cohort_matched["data_split"] == split_name).to_numpy()
    X_split = X_all[mask]
    e_split = cohort_matched.filter(pl.col("data_split") == split_name)["event_indicator"].to_numpy()
    t_split = cohort_matched.filter(pl.col("data_split") == split_name)["time_to_readmission"].to_numpy()
    splits[split_name] = (X_split, e_split, t_split)
    print(f"  {split_name}: X={X_split.shape}, events={e_split.sum()}/{len(e_split)} ({100*e_split.mean():.1f}%)")

# Save aligned data for reuse
np.savez_compressed(
    "data/processed/motor_output/aligned_embeddings.npz",
    X_train=splits["train"][0], e_train=splits["train"][1], t_train=splits["train"][2],
    X_val=splits["tuning"][0], e_val=splits["tuning"][1], t_val=splits["tuning"][2],
    X_test=splits["held_out"][0], e_test=splits["held_out"][1], t_test=splits["held_out"][2],
)
print("\nSaved aligned_embeddings.npz")
