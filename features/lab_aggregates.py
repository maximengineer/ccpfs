"""
Lab Aggregate Features
-----------------------
Extract last/max/min values for target labs during a hospital admission.
"""

from datetime import datetime

import polars as pl

from config import TARGET_LABS

# Build code prefix lookup: lab_name -> MEDS code prefix
_LAB_CODE_PREFIXES = {
    name: f"LAB//{itemid}//" for name, itemid in TARGET_LABS.items()
}


def compute_lab_features(
    lab_events: pl.DataFrame,
) -> dict[str, float]:
    """Compute lab aggregate features from events during one admission.

    Parameters
    ----------
    lab_events : pl.DataFrame
        Events filtered to LAB// codes within one admission window.
        Must have columns: code, numeric_value, time.

    Returns
    -------
    dict[str, float]
        Feature name -> value. NaN for missing labs.
    """
    features = {}

    if len(lab_events) == 0:
        for lab_name in _LAB_CODE_PREFIXES:
            features[f"{lab_name}_last"] = float("nan")
            features[f"{lab_name}_max"] = float("nan")
            features[f"{lab_name}_min"] = float("nan")
            features[f"{lab_name}_missing"] = 1.0
        return features

    for lab_name, code_prefix in _LAB_CODE_PREFIXES.items():
        lab_rows = lab_events.filter(
            pl.col("code").str.starts_with(code_prefix)
            & pl.col("numeric_value").is_not_null()
        )

        if len(lab_rows) == 0:
            features[f"{lab_name}_last"] = float("nan")
            features[f"{lab_name}_max"] = float("nan")
            features[f"{lab_name}_min"] = float("nan")
            features[f"{lab_name}_missing"] = 1.0
        else:
            sorted_labs = lab_rows.sort("time")
            features[f"{lab_name}_last"] = float(
                sorted_labs["numeric_value"][-1]
            )
            features[f"{lab_name}_max"] = float(
                lab_rows["numeric_value"].max()
            )
            features[f"{lab_name}_min"] = float(
                lab_rows["numeric_value"].min()
            )
            features[f"{lab_name}_missing"] = 0.0

    return features
