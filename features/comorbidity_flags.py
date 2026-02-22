"""
Comorbidity Flag Extraction
----------------------------
Extract binary comorbidity indicators from ICD diagnosis codes.
"""

# (ICD-10 prefixes, ICD-9 prefixes) for each comorbidity
COMORBIDITY_PREFIXES = {
    "has_heart_failure": (["I50"], ["428"]),
    "has_diabetes": (["E10", "E11"], ["250"]),
    "has_ckd": (["N18"], ["585"]),
    "has_copd": (["J44"], ["496"]),
    "has_hypertension": (["I10"], ["401"]),
    "has_afib": (["I48"], ["427"]),
}


def compute_comorbidity_flags(diagnosis_codes: list[str]) -> dict[str, int]:
    """Compute binary comorbidity flags from MEDS diagnosis codes.

    Parameters
    ----------
    diagnosis_codes : list[str]
        MEDS-format codes, e.g. 'DIAGNOSIS//ICD//10//I5032'.

    Returns
    -------
    dict[str, int]
        Flag name -> 0/1.
    """
    # Extract ICD codes from MEDS format
    icd_codes = []
    for code in diagnosis_codes:
        parts = code.split("//")
        if len(parts) >= 4 and parts[0] == "DIAGNOSIS":
            icd_codes.append((parts[2], parts[3]))  # (version, code)

    flags = {}
    for flag_name, (icd10_prefixes, icd9_prefixes) in COMORBIDITY_PREFIXES.items():
        found = False
        for version, icd_code in icd_codes:
            if version == "10":
                if any(icd_code.startswith(p) for p in icd10_prefixes):
                    found = True
                    break
            elif version == "9":
                if any(icd_code.startswith(p) for p in icd9_prefixes):
                    found = True
                    break
        flags[flag_name] = int(found)

    # Count distinct 3-char ICD roots
    roots = set()
    for _, icd_code in icd_codes:
        if len(icd_code) >= 3:
            roots.add(icd_code[:3])
    flags["n_diagnosis_categories"] = len(roots)
    flags["n_diagnoses"] = len(icd_codes)

    return flags
