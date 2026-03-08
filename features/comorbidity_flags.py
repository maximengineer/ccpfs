"""
Comorbidity Flag Extraction
----------------------------
Extract binary comorbidity indicators from ICD diagnosis codes.
"""

# (ICD-10 prefixes, ICD-9 prefixes) for each comorbidity
COMORBIDITY_PREFIXES = {
    # Original 6
    "has_heart_failure": (["I50"], ["428"]),
    "has_diabetes": (["E10", "E11"], ["250"]),
    "has_ckd": (["N18"], ["585"]),
    "has_copd": (["J44"], ["496"]),
    "has_hypertension": (["I10"], ["401"]),
    "has_afib": (["I48"], ["427"]),
    # New 6 — high-impact for readmission
    "has_liver_disease": (["K70", "K71", "K72", "K73", "K74", "K75", "K76"], ["571"]),
    "has_malignancy": (["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"], ["14", "15", "16", "17", "18", "19", "20"]),
    "has_depression": (["F32", "F33"], ["296", "311"]),
    "has_obesity": (["E66"], ["278"]),
    "has_stroke": (["I63", "I64", "I65", "I66"], ["433", "434", "436"]),
    "has_acs": (["I21", "I22", "I24"], ["410"]),
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
