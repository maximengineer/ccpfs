"""
Convert MEDS parquet (OMOP-recoded) to simple femr CSV format.

The simple femr format is: CSV files with columns patient_id, start, code, value
where code has vocabulary prefix (e.g., SNOMED/12345, LOINC/1234-5).

Usage:
    python -m models.motor.meds_to_simple_femr \
        --meds-dir data/meds/MEDS_cohort_omop/ \
        --output-dir data/meds/simple_femr/ \
        --num-shards 8
"""

import argparse
import csv
import os
from pathlib import Path

import polars as pl


def convert_meds_to_simple_femr(meds_dir: Path, output_dir: Path, num_shards: int = 8):
    """Convert MEDS parquet files to simple femr CSV format.

    Streams file-by-file to avoid OOM. Each input parquet shard is written
    directly to a corresponding output CSV shard.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather all input parquet files
    input_files = []
    for split in ["train", "tuning", "held_out"]:
        split_dir = meds_dir / "data" / split
        if not split_dir.exists():
            continue
        input_files.extend(sorted(split_dir.glob("*.parquet")))

    print(f"  Found {len(input_files)} parquet files")

    # Process each parquet file → write to a CSV shard
    # Use one CSV per input parquet to avoid loading all data
    total_kept = 0
    total_skipped = 0
    shard_idx = 0

    for f in input_files:
        df = pl.read_parquet(f, columns=["subject_id", "time", "code", "numeric_value"])

        # Filter: keep only OMOP-standard codes (contain / but NOT //)
        standard = df.filter(
            pl.col("code").str.contains("/") & ~pl.col("code").str.contains("//")
        )

        n_skipped = len(df) - len(standard)
        total_skipped += n_skipped

        if len(standard) == 0:
            continue

        # Drop null times
        standard = standard.filter(pl.col("time").is_not_null())

        total_kept += len(standard)

        # Write this shard
        output_path = output_dir / f"shard_{shard_idx:04d}.csv"
        with open(output_path, "w", newline="") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["patient_id", "start", "code", "value"])
            for row in standard.iter_rows(named=True):
                t = row["time"]
                start_str = t.strftime("%Y-%m-%d %H:%M:%S") if hasattr(t, "strftime") else str(t)
                val = str(row["numeric_value"]) if row["numeric_value"] is not None else ""
                writer.writerow([row["subject_id"], start_str, row["code"], val])

        shard_idx += 1

        if shard_idx % 50 == 0:
            print(f"    Processed {shard_idx} files, {total_kept} events kept...")

    print(f"  Total events kept: {total_kept}, skipped: {total_skipped}")
    print(f"  Output: {shard_idx} CSV shards in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert MEDS to simple femr format")
    parser.add_argument("--meds-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8,
                        help="Ignored (one CSV per input parquet for memory efficiency)")
    args = parser.parse_args()

    print("\nConverting MEDS (OMOP-recoded) to simple femr format...")
    convert_meds_to_simple_femr(args.meds_dir, args.output_dir)


if __name__ == "__main__":
    main()
