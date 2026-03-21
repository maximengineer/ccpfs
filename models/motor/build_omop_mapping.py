"""
Build OMOP code mapping for MOTOR model integration.

Maps MIMIC-IV native codes (ICD-9/10, lab itemids, drug names) to OMOP standard
vocabulary codes (SNOMED, LOINC, RxNorm) that MOTOR's dictionary understands.

Usage:
    python -m models.motor.build_omop_mapping \
        --athena-dir vocab/ \
        --ohdsi-dir /tmp/ohdsi_mimic/custom_mapping_csv/ \
        --meds-dir data/meds/MEDS_cohort/ \
        --output-dir data/meds/MEDS_cohort_omop/

This script:
1. Loads Athena CONCEPT.csv and CONCEPT_RELATIONSHIP.csv
2. Loads OHDSI/MIMIC custom mapping CSVs (labs, drugs)
3. Scans all MEDS parquet files for unique codes
4. Builds mapping: MIMIC code -> OMOP standard code (vocabulary_id/concept_code)
5. Recodes MEDS parquet files with mapped codes
6. Outputs recoded data ready for meds_reader_convert + MOTOR
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl


def load_athena_concepts(athena_dir: Path) -> pl.DataFrame:
    """Load CONCEPT.csv from Athena vocabulary download."""
    concept_path = athena_dir / "CONCEPT.csv"
    if not concept_path.exists():
        raise FileNotFoundError(f"CONCEPT.csv not found in {athena_dir}")

    print(f"  Loading {concept_path}...")
    df = pl.read_csv(
        concept_path,
        separator="\t",
        infer_schema_length=10000,
        null_values=[""],
        quote_char=None,
        truncate_ragged_lines=True,
    )
    print(f"  Loaded {len(df)} concepts")
    return df


def load_athena_relationships(athena_dir: Path) -> pl.DataFrame:
    """Load CONCEPT_RELATIONSHIP.csv for 'Maps to' relationships."""
    rel_path = athena_dir / "CONCEPT_RELATIONSHIP.csv"
    if not rel_path.exists():
        raise FileNotFoundError(f"CONCEPT_RELATIONSHIP.csv not found in {athena_dir}")

    print(f"  Loading {rel_path}...")
    df = pl.read_csv(
        rel_path,
        separator="\t",
        infer_schema_length=10000,
        null_values=[""],
        quote_char=None,
        truncate_ragged_lines=True,
    )
    # Filter to "Maps to" relationships only
    maps_to = df.filter(pl.col("relationship_id") == "Maps to")
    print(f"  Loaded {len(df)} relationships, {len(maps_to)} 'Maps to' entries")
    return maps_to


def load_ohdsi_lab_mappings(ohdsi_dir: Path) -> dict:
    """Load OHDSI/MIMIC lab itemid -> OMOP concept_id mapping."""
    path = ohdsi_dir / "gcpt_meas_lab_loinc_mod.csv"
    if not path.exists():
        print(f"  Warning: {path} not found, skipping lab mappings")
        return {}

    df = pl.read_csv(path)
    mapping = {}
    for row in df.iter_rows(named=True):
        itemid = str(row["concept_code"])  # MIMIC itemid
        target_id = row["target_concept_id"]
        if target_id and target_id > 0:
            mapping[itemid] = int(target_id)

    print(f"  Loaded {len(mapping)} lab itemid -> concept_id mappings")
    return mapping


def load_ohdsi_drug_mappings(ohdsi_dir: Path) -> dict:
    """Load OHDSI/MIMIC drug name -> OMOP concept_id mapping."""
    path = ohdsi_dir / "gcpt_drug_ndc.csv"
    if not path.exists():
        print(f"  Warning: {path} not found, skipping drug mappings")
        return {}

    df = pl.read_csv(path)
    mapping = {}
    for row in df.iter_rows(named=True):
        drug_code = str(row["concept_code"])  # MIMIC drug name
        target_id = row["target_concept_id"]
        if target_id and target_id > 0:
            mapping[drug_code] = int(target_id)

    print(f"  Loaded {len(mapping)} drug name -> concept_id mappings")
    return mapping


def load_ohdsi_chartevents_mappings(ohdsi_dir: Path) -> dict:
    """Load OHDSI/MIMIC chartevents itemid -> concept_id mapping."""
    path = ohdsi_dir / "gcpt_meas_chartevents_main_mod.csv"
    if not path.exists():
        return {}
    df = pl.read_csv(path)
    mapping = {}
    for row in df.iter_rows(named=True):
        itemid = str(row["concept_code"])
        target_id = row["target_concept_id"]
        if target_id and target_id > 0:
            mapping[itemid] = int(target_id)
    print(f"  Loaded {len(mapping)} chartevents itemid -> concept_id mappings")
    return mapping


def build_icd_to_snomed_map(concepts: pl.DataFrame, relationships: pl.DataFrame) -> dict:
    """Build ICD-9/10 -> SNOMED mapping using Athena vocabularies.

    Mapping chain:
        ICD code -> concept_id (CONCEPT.csv) -> Maps to -> concept_id_2 -> SNOMED code
    """
    # Get ICD concepts (both ICD9CM and ICD10CM)
    icd_concepts = concepts.filter(
        pl.col("vocabulary_id").is_in(["ICD9CM", "ICD10CM", "ICD9Proc", "ICD10PCS"])
    ).select(["concept_id", "concept_code", "vocabulary_id"])

    print(f"  Found {len(icd_concepts)} ICD concepts in Athena")

    # Join with Maps to relationships
    mapped = icd_concepts.join(
        relationships.select(["concept_id_1", "concept_id_2"]),
        left_on="concept_id",
        right_on="concept_id_1",
        how="inner",
    )

    # Resolve target concept_id to vocabulary/code
    target_concepts = concepts.filter(
        pl.col("standard_concept") == "S"  # Standard concepts only
    ).select(
        pl.col("concept_id").alias("target_concept_id"),
        pl.col("vocabulary_id").alias("target_vocab"),
        pl.col("concept_code").alias("target_code"),
    )

    result = mapped.join(
        target_concepts,
        left_on="concept_id_2",
        right_on="target_concept_id",
        how="inner",
    )

    # Build mapping dict: (vocab, code) -> "target_vocab/target_code"
    mapping = {}
    for row in result.iter_rows(named=True):
        src_vocab = row["vocabulary_id"]
        src_code = row["concept_code"]
        target = f"{row['target_vocab']}/{row['target_code']}"
        mapping[(src_vocab, src_code)] = target

    print(f"  Built {len(mapping)} ICD -> standard concept mappings")
    return mapping


def build_concept_id_to_vocab_code(concepts: pl.DataFrame) -> dict:
    """Build concept_id -> 'vocabulary_id/concept_code' lookup for standard concepts."""
    standard = concepts.filter(
        pl.col("standard_concept") == "S"
    ).select(["concept_id", "vocabulary_id", "concept_code"])

    mapping = {}
    for row in standard.iter_rows(named=True):
        cid = row["concept_id"]
        mapping[cid] = f"{row['vocabulary_id']}/{row['concept_code']}"

    print(f"  Built {len(mapping)} concept_id -> vocab/code lookup")
    return mapping


def map_meds_code(code: str, icd_map: dict, lab_map: dict, drug_map: dict,
                  chart_map: dict, concept_lookup: dict) -> str:
    """Map a single MEDS code to OMOP standard vocabulary format.

    Returns the mapped code or the original if no mapping exists.
    """
    parts = code.split("//")
    prefix = parts[0]

    if prefix == "DIAGNOSIS" and len(parts) >= 4:
        # DIAGNOSIS//ICD//9//29690 or DIAGNOSIS//ICD//10//I10
        icd_version = parts[2]  # "9" or "10"
        icd_code = parts[3]

        # Map ICD version to Athena vocabulary_id
        if icd_version == "9":
            vocab = "ICD9CM"
        elif icd_version == "10":
            vocab = "ICD10CM"
        else:
            return code

        # Try exact match first
        mapped = icd_map.get((vocab, icd_code))
        if mapped:
            return mapped

        # Try with dot notation (ICD-10 codes often have dots)
        if icd_version == "10" and len(icd_code) > 3:
            dotted = icd_code[:3] + "." + icd_code[3:]
            mapped = icd_map.get((vocab, dotted))
            if mapped:
                return mapped

        # Try without dots
        undotted = icd_code.replace(".", "")
        mapped = icd_map.get((vocab, undotted))
        if mapped:
            return mapped

    elif prefix == "LAB" and len(parts) >= 2:
        # LAB//51429//%  -> extract itemid
        itemid = parts[1]
        concept_id = lab_map.get(itemid)
        if concept_id and concept_id in concept_lookup:
            return concept_lookup[concept_id]

    elif prefix == "MEDICATION" and len(parts) >= 2:
        # MEDICATION//TraZODone//UNK or MEDICATION//STOP//TraZODone
        if parts[1] == "STOP" and len(parts) >= 3:
            drug_name = parts[2]
        else:
            drug_name = parts[1]

        concept_id = drug_map.get(drug_name)
        if concept_id and concept_id in concept_lookup:
            return concept_lookup[concept_id]

    elif prefix == "PROCEDURE" and len(parts) >= 3:
        # PROCEDURE//ICD//9//8828 or PROCEDURE//ICD//10//0T768DZ
        # or PROCEDURE//START//224275
        if parts[1] == "ICD" and len(parts) >= 4:
            icd_version = parts[2]
            proc_code = parts[3]
            if icd_version == "9":
                vocab = "ICD9Proc"
            elif icd_version == "10":
                vocab = "ICD10PCS"
            else:
                return code
            mapped = icd_map.get((vocab, proc_code))
            if mapped:
                return mapped
        elif parts[1] in ("START", "END") and len(parts) >= 3:
            itemid = parts[2]
            concept_id = chart_map.get(itemid)
            if concept_id and concept_id in concept_lookup:
                return concept_lookup[concept_id]

    elif prefix in ("INFUSION_START", "INFUSION_END") and len(parts) >= 2:
        itemid = parts[1]
        concept_id = chart_map.get(itemid)
        if concept_id and concept_id in concept_lookup:
            return concept_lookup[concept_id]

    elif prefix == "SUBJECT_FLUID_OUTPUT" and len(parts) >= 2:
        itemid = parts[1]
        concept_id = chart_map.get(itemid)
        if concept_id and concept_id in concept_lookup:
            return concept_lookup[concept_id]

    # Return original if no mapping found
    return code


def collect_unique_codes(meds_dir: Path) -> set:
    """Scan all MEDS parquet files and collect unique codes."""
    codes = set()
    for split in ["train", "tuning", "held_out"]:
        split_dir = meds_dir / "data" / split
        if not split_dir.exists():
            continue
        for f in sorted(split_dir.glob("*.parquet")):
            df = pl.read_parquet(f, columns=["code"])
            codes.update(df["code"].unique().to_list())
    print(f"  Collected {len(codes)} unique codes from MEDS data")
    return codes


def recode_meds_data(meds_dir: Path, output_dir: Path, code_map: dict):
    """Recode all MEDS parquet files with mapped codes."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy metadata
    meta_src = meds_dir / "metadata"
    meta_dst = output_dir / "metadata"
    if meta_src.exists():
        import shutil
        if meta_dst.exists():
            shutil.rmtree(meta_dst)
        shutil.copytree(meta_src, meta_dst)
        print(f"  Copied metadata to {meta_dst}")

    total_recoded = 0
    total_events = 0

    for split in ["train", "tuning", "held_out"]:
        split_src = meds_dir / "data" / split
        split_dst = output_dir / "data" / split
        split_dst.mkdir(parents=True, exist_ok=True)

        if not split_src.exists():
            continue

        files = sorted(split_src.glob("*.parquet"))
        print(f"  Recoding {split}: {len(files)} files...")

        for f in files:
            df = pl.read_parquet(f)
            n_events = len(df)

            # Map codes
            mapped_codes = df["code"].map_elements(
                lambda c: code_map.get(c, c), return_dtype=pl.Utf8
            )
            n_changed = (mapped_codes != df["code"]).sum()

            df = df.with_columns(mapped_codes.alias("code"))
            df.write_parquet(split_dst / f.name)

            total_recoded += n_changed
            total_events += n_events

        print(f"    {split} done")

    pct = (total_recoded / total_events * 100) if total_events > 0 else 0
    print(f"  Recoded {total_recoded}/{total_events} events ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Build OMOP code mapping for MOTOR")
    parser.add_argument("--athena-dir", type=Path, required=True,
                        help="Directory containing Athena CONCEPT.csv etc.")
    parser.add_argument("--ohdsi-dir", type=Path, required=True,
                        help="Directory with OHDSI/MIMIC custom mapping CSVs")
    parser.add_argument("--meds-dir", type=Path, required=True,
                        help="Source MEDS cohort directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for recoded MEDS data")
    args = parser.parse_args()

    print("\n[1/6] Loading Athena vocabularies...")
    concepts = load_athena_concepts(args.athena_dir)
    relationships = load_athena_relationships(args.athena_dir)

    print("\n[2/6] Loading OHDSI/MIMIC custom mappings...")
    lab_map = load_ohdsi_lab_mappings(args.ohdsi_dir)
    drug_map = load_ohdsi_drug_mappings(args.ohdsi_dir)
    chart_map = load_ohdsi_chartevents_mappings(args.ohdsi_dir)

    print("\n[3/6] Building ICD -> SNOMED mapping...")
    icd_map = build_icd_to_snomed_map(concepts, relationships)

    print("\n[4/6] Building concept_id -> vocab/code lookup...")
    concept_lookup = build_concept_id_to_vocab_code(concepts)

    print("\n[5/6] Building full code mapping table...")
    all_codes = collect_unique_codes(args.meds_dir)

    code_map = {}
    mapped_count = 0
    for code in all_codes:
        mapped = map_meds_code(code, icd_map, lab_map, drug_map, chart_map, concept_lookup)
        if mapped != code:
            code_map[code] = mapped
            mapped_count += 1

    print(f"  Mapped {mapped_count}/{len(all_codes)} unique codes ({mapped_count/len(all_codes)*100:.1f}%)")

    # Show mapping stats by category
    categories = {}
    for code in all_codes:
        prefix = code.split("//")[0] if "//" in code else "OTHER"
        if prefix not in categories:
            categories[prefix] = [0, 0]
        categories[prefix][0] += 1
        if code in code_map:
            categories[prefix][1] += 1

    print("\n  Mapping coverage by category:")
    for cat in sorted(categories, key=lambda x: -categories[x][0]):
        total, mapped = categories[cat]
        pct = (mapped / total * 100) if total > 0 else 0
        print(f"    {cat}: {mapped}/{total} ({pct:.0f}%)")

    print(f"\n[6/6] Recoding MEDS parquet files...")
    recode_meds_data(args.meds_dir, args.output_dir, code_map)

    print(f"\n{'='*60}")
    print(f"  OMOP mapping complete!")
    print(f"  Output: {args.output_dir}")
    print(f"  Next: run meds_reader_convert on the output directory")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
