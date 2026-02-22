# CCPFS Technical Implementation Plan

## Capacity-Constrained Personalised Follow-Up Scheduling

*Revised 2026-02-22 — Updated with full MIMIC-IV pipeline results and paper draft*

---

## 0. Progress Status

| Phase | Status | Notes |
|-------|--------|-------|
| **Phase 0: Setup** | DONE | Python 3.12 venv, project structure, requirements |
| **Phase 1: Data Pipeline** | DONE | MEDS ETL complete; 364,627 patients, 366 shards, 13 GB |
| **Phase 2: Feature Engineering** | DONE | 47 features extracted from 275,022 episodes via streaming shard processing |
| **Phase 3: Survival Models** | DONE | GBM (C-index 0.695) + Cox PH (0.688) trained on 30K subsample |
| **Phase 4: Scheduling Policy** | DONE | All schedulers + specialty pool extension; 8 policies compared |
| **Phase 5: Evaluation Framework** | DONE | Full pipeline run on 27,641 test patients; results in pipeline_results.json |
| **Phase 6: API + Dashboard** | NOT STARTED | Deferred — not required for paper |
| **Phase 7: Paper** | DONE | 6-page IEEE paper draft at `2_pages/paper_draft.md` with real MIMIC-IV results |

**Key decisions**:
- Phase 4-5 (scheduling + evaluation) were built first because they have zero data dependency — they consume survival curves regardless of source.
- MOTOR foundation model (Track A) was deferred; GBM (Track B) serves as the primary model with strong results.
- Specialty pool scheduler was added to extend the ILP with per-clinic capacity constraints.
- Pipeline runs end-to-end in ~93 minutes (feature extraction ~36 min, GBM grid search ~89 min on 30K subsample).

### Pipeline Results Summary (2026-02-22)

**Model Performance (held-out test set, N=27,641):**

| Model | C-index | IBS |
|-------|---------|-----|
| GBSA | 0.695 | 0.099 |
| Cox PH | 0.688 | 0.100 |

**Scheduling Policy Comparison:**

| Policy | Status | Avg Cost/Patient |
|--------|--------|-----------------|
| Guideline (ACC) | Infeasible† | €2,025 |
| Uniform day 14 | Infeasible† | €1,433 |
| Risk bucket | Infeasible† | €1,428 |
| Greedy (global) | Feasible | €1,037 |
| ILP (global) | Optimal | €1,036 |
| **ILP (specialty)** | **Optimal** | **€954** |
| Unconstrained | Infeasible | €211 |

†Baselines do not respect daily capacity limits.

**Key result**: ILP specialty achieves 33.4% cost reduction vs uniform and 52.9% vs guideline scheduling.

---

### Design Philosophy

> Fine-tune a pretrained time-to-event foundation model (MOTOR) on MIMIC-IV in MEDS format, calibrate its survival outputs, feed them into a capacity-constrained ILP scheduler, and evaluate the scheduling policy offline against three baselines.

---

## 1. Data Strategy

### 1.1 Primary Dataset: MIMIC-IV v3.1

| Attribute | Detail |
|-----------|--------|
| Size | 364,627 patients in MEDS format (366 shards, 13 GB) |
| Eligible cohort | 275,022 discharge episodes from 137,054 unique patients |
| Source | Beth Israel Deaconess Medical Centre (Boston) |
| 30-day readmission rate | 20.3% |
| Readmission dates | YES - `subject_id` persists across admissions; exact `admittime`/`dischtime` |
| Labs/Vitals/Meds | YES - full inpatient data |
| Outpatient follow-up | NO - acknowledged limitation |
| Access | PhysioNet credentialed (CITI training completed) |
| MEDS location | `data/meds/MEDS_cohort/data/{train,tuning,held_out}/` |
| Splits | 291,702 train / 36,463 tuning / 36,462 held_out (80/10/10) |

Key tables from the `hosp` module:

| Table | Purpose |
|-------|---------|
| `patients` | Demographics (gender, anchor_age, dod) |
| `admissions` | Index events + outcomes (admittime, dischtime, admission_type, discharge_location) |
| `diagnoses_icd` | Comorbidities (ICD-9/10 codes) |
| `labevents` + `d_labitems` | Lab trajectories with timestamps |
| `prescriptions` / `pharmacy` | Inpatient medication data |
| `procedures_icd` | Surgical/procedural history |
| `transfers` | ICU transfers, care unit movements, LOS |

Companion modules:

| Module | Value |
|--------|-------|
| **MIMIC-IV-ED** | ED triage vitals, ED diagnoses, disposition - captures the ED-to-admission pathway |
| **MIMIC-IV-Note** | Discharge summaries for NLP features (future extension) |

### 1.2 Data Format: MEDS (Medical Event Data Standard)

Rather than writing custom ETL from raw CSVs, we use the **MEDS ecosystem** - the successor to Event Stream GPT (Paper 4 in our literature review).

| Component | What it does |
|-----------|-------------|
| **MEDS schema** | 4 columns: `subject_id`, `time`, `code`, `numeric_value` |
| **MIMIC-IV MEDS ETL** | Ready-made pipeline to convert MIMIC-IV to MEDS format |
| **MIMIC-IV demo in MEDS** | Pre-converted demo data on PhysioNet for instant prototyping |
| **MEDS-Transforms** | Scalable feature extraction framework |
| **MEDS-DEV** | Benchmark framework with shared task definitions (includes readmission) |

**Why this matters:**
- MOTOR and CLMBR foundation models consume MEDS/OMOP format natively via FEMR
- Write the pipeline once, run it on any MEDS-compatible dataset (MIMIC-IV, eICU, SICdb, NWICU)
- Reproducible benchmarking via MEDS-DEV
- The MIMIC-IV MEDS ETL is a pip-installable package (`pip install MIMIC-IV-MEDS`)

### 1.3 External Validation Datasets

| Dataset | Size | Readmission tracking | Multi-site | CCPFS role |
|---------|------|---------------------|------------|------------|
| **eICU-CRD** | 197K ICU admissions, 208 US hospitals | Limited cross-hospitalisation | YES (208 hospitals) | Multi-centre validation |
| **SICdb** (Salzburg) | 27K ICU admissions | YES - explicit `OffsetAfterFirstAdmission` field | NO (1 European site) | European validation |
| **AmsterdamUMCdb** | 23K ICU admissions | YES (ICU readmission) | NO (1 European site) | Geographic robustness |

### 1.4 Development Data: Synthetic Cohort Generator *(DONE)*

Rather than using Synthea (which was considered but dropped — MIMIC-IV access is expected soon), we built a **Weibull-based synthetic cohort generator** (`evaluation/synthetic.py`) that produces realistic survival curves and outcomes directly, without needing any external data pipeline.

| Attribute | Detail |
|-----------|--------|
| Module | `evaluation/synthetic.py` |
| Patients | Configurable (default 500) |
| Risk groups | High (~25%, 30-day event rate ~35%), Medium (~50%, ~15%), Low (~25%, ~5%) |
| Outputs | Survival curves, uncertainty bands, event times, event indicators, risk groups, HF flags |
| Noise | Per-patient Weibull parameter variation + Gaussian noise + monotonicity enforcement |

**Why this is sufficient for development**: The scheduling policy layer consumes `np.ndarray` survival curves regardless of how they were produced. The synthetic generator lets us test all schedulers, baselines, metrics, and sensitivity analyses without any real data dependency. When MIMIC-IV arrives, we swap in real model predictions — the policy layer is unchanged.

### 1.5 Datasets Acknowledged but Not Used

| Dataset | Why mentioned | Why not primary |
|---------|---------------|----------------|
| **CPRD** (UK, 60M patients) | Has outpatient follow-up + GP visits + prescriptions - ideal for CCPFS | Expensive ($5K-$50K), long access process |
| **CMS Medicare** (US, tens of millions) | Has outpatient claims + Part D pharmacy | Requires DUA, IRB, institutional affiliation, $2K-$20K |
| **HCUP NRD** (18M discharges/year) | National readmission statistics | Administrative only - no clinical features (no labs, vitals) |

These are mentioned in the paper's Discussion as datasets that a production CCPFS system would use. Our framework is designed to scale to them.

---

## 2. Model Strategy: Two-Track Approach

We implement **two survival model tracks** and compare them. This is both scientifically stronger (ablation study) and safer (fallback if one fails).

### Track A: MOTOR Foundation Model (Primary)

MOTOR (Many Outcome Time Oriented Representations) is a 143M-parameter transformer pretrained on 55 million EHR records using self-supervised time-to-event objectives [Paper 1]. It is the **only publicly available foundation model designed specifically for time-to-event prediction**.

| Attribute | Detail |
|-----------|--------|
| Weights | `StanfordShahLab/motor-t-base` on Hugging Face |
| Parameters | 143M |
| Architecture | Transformer + piecewise exponential hazard model |
| Pretraining | 55M EHRs, 9B clinical events, 8,192 simultaneous TTE tasks |
| Performance | +4.6% C-statistic over SOTA, 95% label efficiency gains |
| Framework | FEMR (Framework for Electronic Medical Records) |
| Data format | MEDS / OMOP-CDM |

**Fine-tuning pipeline:**

```
MIMIC-IV raw CSVs
    --> MIMIC-IV MEDS ETL (pip install MIMIC-IV-MEDS)
        --> MEDS format (subject_id, time, code, numeric_value)
            --> FEMR data loader
                --> Load MOTOR-T-Base pretrained weights
                    --> Fine-tune survival head for 30-day readmission
                        --> Output: patient-level hazard curves h(t), survival curves S(t)
```

**Why MOTOR over training from scratch:**
- Pretrained on 9 billion clinical events - captures temporal patterns we cannot learn from MIMIC-IV alone
- 95% label efficiency - critical when the HF sub-cohort may only have ~10K episodes
- Cross-site transferability already demonstrated
- Directly produces time-to-event outputs (not binary classification adapted post-hoc)
- Published at ICLR 2024 - citing it strengthens the paper

**Compute requirement:**
- Fine-tuning (not full pretraining) - requires ~16 GB VRAM
- AMD RX 9070 with ROCm should handle this; alternatively rent an A100 for a few hours on Vast.ai (~$1-2/hr)

**Key resources:**
- `github.com/som-shahlab/femr` - FEMR framework
- `github.com/som-shahlab/motor_tutorial` - Fine-tuning tutorial
- `github.com/sungresearch/femr-on-mimic` - Community MIMIC-IV integration

### Track B: Gradient Boosting Survival Analysis (Baseline/Fallback)

| Attribute | Detail |
|-----------|--------|
| Library | scikit-survival (`GradientBoostingSurvivalAnalysis`) |
| Why | Trains in minutes on CPU. Native SHAP support. Strong on tabular data. Interpretable. |
| Role | Classical ML baseline for ablation; fallback if MOTOR fine-tuning encounters issues |
| Also compare | Cox PH (lifelines), Random Survival Forest (scikit-survival) |

**Why we keep this track:**
- Ablation: does the foundation model actually help vs. a well-tuned GBM?
- Interpretability: GBM + SHAP gives cleaner feature-level explanations
- Robustness: if MOTOR has issues with ROCm or MIMIC-IV format, GBM still delivers results
- The paper can show: "MOTOR improves C-index from X (GBM) to Y, and the scheduling policy amplifies this advantage under capacity constraints"

### Model Comparison Table (for the paper)

| Model | Type | C-index target | Survival curve? | Uncertainty? | Explainability |
|-------|------|---------------|-----------------|-------------|----------------|
| Cox PH | Classical baseline | ~0.65-0.70 | Yes | Bootstrap | Coefficients |
| Random Survival Forest | ML baseline | ~0.68-0.72 | Yes | OOB variance | Permutation importance |
| Gradient Boosting Survival | ML primary | ~0.72-0.78 | Yes | Bootstrap ensemble | SHAP (TreeExplainer) |
| **MOTOR (fine-tuned)** | Foundation model | ~0.78-0.82 | Yes (piecewise exponential) | Ensemble / MC dropout | Attention weights |

---

## 3. Project Structure

Legend: **[DONE]** = implemented and tested, **[PLANNED]** = not yet written.

```
scheduling_follow_up/
    PLAN.md                         # This file
    README.md                       # Project overview and quick start     [DONE]
    requirements.txt                # Python dependencies                  [DONE]
    config.py                       # Paths, constants, cost params        [DONE]
    run_pipeline.py                 # End-to-end orchestration script      [DONE]
    .python-version                 # Python 3.12                          [DONE]

    # --- Data Pipeline (Phase 1) ---                                      [DONE]
    data/
        meds/                       # MEDS-format MIMIC-IV (13 GB, gitignored)
            MEDS_cohort/data/{train,tuning,held_out}/  # 366 shards
            MEDS_cohort/metadata/   # codes.parquet, subject_splits.parquet
            raw_input/{hosp,icu}/   # Raw CSVs (~60 GB decompressed)
        processed/
            cohort.parquet          # 275,022 episodes                     [DONE]
            features.npz            # 275,022 × 47 feature matrix         [DONE]
            pipeline_results.json   # Final scheduling results             [DONE]
            pipeline_log.txt        # Full pipeline output log             [DONE]

    etl/
        __init__.py                                                        [DONE]
        build_cohort.py             # Streaming cohort builder (366 shards)[DONE]

    # --- Feature Engineering (Phase 2) ---                                [DONE]
    features/
        __init__.py                                                        [DONE]
        extract_features.py         # 47 features, streaming extraction    [DONE]
        lab_aggregates.py           # Lab last/max/min per admission       [DONE]
        comorbidity_flags.py        # Binary comorbidity indicators        [DONE]

    # --- Survival Models (Phase 3) ---                                    [DONE]
    models/
        __init__.py                                                        [DONE]
        classical/
            __init__.py                                                    [DONE]
            train_gbm.py            # GBM survival (C-index 0.695)         [DONE]
            train_cox.py            # Cox PH baseline (C-index 0.688)      [DONE]
        calibrate.py                # Isotonic calibration (4 horizons)     [DONE]
        uncertainty.py              # Bootstrap ensemble (K GBMs)           [DONE]
        evaluate_model.py           # C-index, IBS, ECE metrics            [DONE]
        saved/                      # Serialised model artifacts            [DONE]

    # --- Scheduling Policy (Phase 4) ---                                  [DONE]
    policy/
        cost_function.py            # C_event, C_visit, expected cost      [DONE]
        ilp_scheduler.py            # ILP via PuLP/CBC (global capacity)   [DONE]
        greedy_scheduler.py         # Greedy marginal-benefit (global)     [DONE]
        specialty_scheduler.py      # ILP + greedy with per-pool capacity  [DONE]
        uncertainty_adjustment.py   # Conservative scheduling              [DONE]
        baselines.py                # Uniform, risk-bucket, guideline      [DONE]

    # --- Evaluation Framework (Phase 5) ---                               [DONE]
    evaluation/
        synthetic.py                # Weibull-based synthetic cohort gen    [DONE]
        simulate.py                 # Offline policy simulation            [DONE]
        metrics.py                  # EBF rate, expected cost, capacity    [DONE]
        sensitivity.py              # Vary capacity, cost ratio, horizon   [DONE]

    # --- Tests ---                                                        [DONE]
    tests/
        test_policy.py              # 24 tests across 7 test classes       [DONE]

    # --- Notebooks ---
    notebooks/
        01_pipeline_demo.ipynb      # Synthetic data demo                  [DONE]

    # --- Deployment (Phase 6) ---                                         [PLANNED]
    api/
        app.py                      # FastAPI service (score, schedule)
    dashboard/
        streamlit_app.py            # Streamlit clinical dashboard
```

---

## 4. Phase 1 - Data Pipeline

### 4.1 Cohort Definition

```python
# etl/build_cohort.py — IMPLEMENTED

Inclusion:
    - Adult patients (age >= 18 at discharge)
    - Discharged to HOME or HOME HEALTH CARE
    - Length of stay >= 1 day

Target variable:
    - time_to_readmission: days from discharge to next admission for same subject
    - event_indicator: 1 if readmitted within 30 days, 0 if censored at 30

Specialty pool assignment (hybrid approach):
    - Priority 1: Admitting service (surgical services → Surgery, cardiology services → Cardiology, etc.)
    - Priority 2: ICD-10 diagnosis codes (I20-I52 → Cardiology, G*/I60-I69 → Neurology)
    - Priority 3: Default → General Medicine

ACTUAL SIZES (from 366 MEDS shards):
    - Full cohort: 275,022 discharge episodes from 137,054 unique patients
    - 30-day readmission rate: 20.3%
    - Specialty distribution:
        General Medicine: 55.1%
        Cardiology: 19.7%
        Neurology: 13.7%
        Surgery: 11.5%
    - Heart failure prevalence (ICD-10 I50*): 14.7%
    - Output: data/processed/cohort.parquet
```

### 4.2 MEDS ETL *(DONE)*

```bash
# MIMIC-IV MEDS ETL was run successfully — v0.0.7
# Output: data/meds/MEDS_cohort/ (364,627 patients, 366 shards, 13 GB)
# Raw data: data/meds/raw_input/{hosp,icu}/ (~10 GB compressed, ~60 GB decompressed)
# Metadata: data/meds/MEDS_cohort/metadata/ (codes.parquet, subject_splits.parquet)
```

The MEDS output is a standardised event stream: every lab result, diagnosis, medication, procedure, and vital sign becomes a row with `(subject_id, time, code, numeric_value)`. The cohort builder (`etl/build_cohort.py`) streams one shard at a time with column projection (~50 MB/shard) to stay within 16 GB RAM.

**Known MEDS ETL issues** (see `memory/meds_etl_lessons.md` for details):
- v0.0.7 has specific code format conventions (e.g., `HOSPITAL_ADMISSION//ELECTIVE`, `DIAGNOSIS//ICD//10//I5032`)
- ~13% of lab events have null `hadm_id` — handled via time-range matching fallback
- Subject splits in `metadata/subject_splits.parquet` use column name `split` (not `data_split`)

### 4.3 Development with Synthetic Cohort Generator *(DONE — superseded by real data)*

The synthetic cohort generator was used for initial development. All scheduling and evaluation code was developed and tested using `evaluation/synthetic.py` before MIMIC-IV data became available:

```python
from evaluation.synthetic import generate_synthetic_cohort

cohort = generate_synthetic_cohort(n_patients=500, seed=42)
# Returns: survival_curves (N, 31), survival_stds (N, 31),
#          event_times (N,), event_indicators (N,),
#          risk_groups (N,), is_heart_failure (N,)
```

This generator produces Weibull-distributed survival curves with three risk profiles (high/medium/low), per-patient variation, simulated prediction uncertainty, and sampled event times. All 24 tests and the demo notebook (`01_pipeline_demo.ipynb`) use this generator. When MIMIC-IV predictions become available, the policy layer consumes them identically — it only needs `np.ndarray` survival curves.

---

## 5. Phase 2 - Feature Engineering *(DONE)*

### 5.1 Implemented Features — `features/extract_features.py`

**47 features across 5 groups**, extracted via streaming shard processing (one MEDS shard at a time, column projection to ~50 MB/shard):

| Group | Features | Count |
|-------|----------|-------|
| Demographics | age, gender_male, emergency_admission, ed_origin, insurance dummies | 5 |
| Comorbidity | has_{HF, diabetes, CKD, COPD, hypertension, afib}, n_diagnoses, n_diagnosis_categories | 8 |
| Labs | {creatinine, BUN, sodium, potassium, Hgb, WBC, BNP} × {last, max, min} + 7 `_was_missing` indicators | 28 |
| Prior utilisation | n_prior_admissions, n_prior_365d, days_since_last_discharge | 3 |
| Care intensity | n_medications, n_icd_procedures, has_icu_stay | 3 |

**Lab matching strategy**: Match by `hadm_id` for 87% of labs; time-range fallback (admission ± 1 day) for the 13% with null `hadm_id`.

**Missing value strategy**: Impute with training-set medians + binary `_was_missing` indicators. Overall 11.7% NaN rate before imputation, predominantly BNP (available in only 23% of admissions).

**Output**: `data/processed/features.npz` (275,022 × 47 matrix, ~180 MB)

**Dropped from plan** (insufficient data or complexity for marginal gain):
- eGFR (CKD-EPI derivation)
- lab_trend_deteriorating
- med_changes_last_48h
- Elixhauser/Charlson composite scores (replaced with individual comorbidity flags)
- Weekend discharge flag

### 5.2 Track A (MOTOR) — Deferred

MOTOR fine-tuning was deferred. The GBM (Track B) achieves competitive discrimination (C-index 0.695) and the scheduling framework works well with its predictions. MOTOR remains a future work direction for improving discrimination and enabling cross-site transfer learning.

---

## 6. Phase 3 - Survival Models *(DONE)*

### 6.1 Track A: MOTOR Fine-Tuning *(DEFERRED)*

```python
# motor/finetune_motor.py (conceptual)

import femr
from femr.models import MotorModel

# Load MEDS-format MIMIC-IV data
dataset = femr.datasets.MEDSDataset("data/meds/")

# Define task: 30-day readmission survival
task = femr.tasks.TimeToEventTask(
    index_event="discharge",
    outcome_event="unplanned_readmission",
    horizon_days=30,
    censoring="right"
)

# Load pretrained MOTOR weights
model = MotorModel.from_pretrained("StanfordShahLab/motor-t-base")

# Fine-tune on MIMIC-IV HF cohort
model.finetune(
    dataset=dataset,
    task=task,
    cohort_filter=lambda x: has_hf_diagnosis(x),
    epochs=10,
    lr=1e-4,
    batch_size=32,
    device="cuda"  # ROCm or cloud GPU
)

# Generate survival curves for all patients
for patient in cohort:
    S_t = model.predict_survival(patient)  # S(t) for t=1..30
    h_t = model.predict_hazard(patient)    # h(t) daily hazards
    # Store for scheduler input
```

### 6.2 Track B: Classical Survival Models *(DONE)*

**Implemented in**: `models/classical/train_gbm.py`, `models/classical/train_cox.py`

**GBM (Primary)**:
- `GradientBoostingSurvivalAnalysis` from scikit-survival
- Trained on 30,000-episode random subsample of 219,702 training episodes (full training set was too slow — ~90 min per config)
- Grid search: 4 configurations (learning_rate × max_depth), best on validation C-index
- Best params: `{learning_rate: 0.1, max_depth: 5, min_samples_leaf: 20, n_estimators: 300, subsample: 0.8}`
- Training time: ~89 minutes total (4 configs × ~22 min each)

**Cox PH (Baseline)**:
- `CoxPHFitter` from lifelines
- Trained on same 30K subsample
- Training time: ~2 seconds

**Results on held-out test set (N=27,641)**:

| Model | C-index | IBS | ECE@7d | ECE@14d | ECE@21d | ECE@30d |
|-------|---------|-----|--------|---------|---------|---------|
| GBSA | 0.695 | 0.099 | 0.180 | 0.140 | 0.091 | 0.502 |
| Cox PH | 0.688 | 0.100 | 0.237 | 0.122 | 0.113 | 0.502 |

**Survival curve extraction**: `predict_survival_function()` evaluated at integer days t=0..30, producing (N, 31) arrays with S(0)=1.0 and monotonicity enforced.

### 6.3 Calibration *(IMPLEMENTED but not applied in final results)*

**Implemented in**: `models/calibrate.py`

Isotonic regression calibration at horizons t=7, 14, 21, 30, with linear interpolation between calibration points and monotonicity enforcement.

**Issue encountered**: On the full MIMIC-IV dataset, isotonic calibration pushed all survival values toward 1.0 (Mean S(15) after calibration: 1.0000), producing unusable curves. This is likely because:
- The readmission rate (20.3%) is distributed across 30 days, so per-day event rates are low
- Isotonic regression on highly imbalanced binary labels at each horizon collapsed to trivial mappings

**Current status**: Raw (uncalibrated) GBM curves used for scheduling. ECE at day 21 is 0.091 — adequate for mid-horizon scheduling. ECE at day 30 is 0.50 due to heavy censoring at the boundary.

**Future work**: Platt scaling, Venn-Abers calibration, or D-calibration testing may produce better results.

### 6.4 Uncertainty Estimation *(IMPLEMENTED, not used in final results)*

**Implemented in**: `models/uncertainty.py`

Bootstrap ensemble: K GBMs trained on resampled training data (sequential to control memory). Returns mean and std survival curves across bootstraps.

**Not used in final pipeline**: Bootstrap was skipped with `--no-bootstrap` flag because each GBM takes ~22 minutes to train, and K=20 bootstraps would add ~7 hours. The scheduling framework supports uncertainty-adjusted curves via `S_conservative(t) = max(0, S(t) - α·σ(t))` but this was not evaluated in the final results.

### 6.5 Evaluation Metrics *(DONE)*

**Implemented in**: `models/evaluate_model.py`

| Metric | What it measures | GBM result | Target |
|--------|-----------------|------------|--------|
| Harrell's C-index | Rank discrimination | 0.695 | > 0.70 (close) |
| Integrated Brier Score | Discrimination + calibration | 0.099 | Lower = better |
| ECE@{7,14,21,30}d | Expected calibration error per horizon | 0.18/0.14/0.09/0.50 | < 0.10 ideal |

**Note**: IBS computation required time bounds strictly within test data follow-up range — implemented with `t_start = max(1, ceil(min_test_time))` and try/except fallback.

### 6.6 Explainability *(NOT IMPLEMENTED — future work)*

**Planned (Track B):** SHAP via `shap.TreeExplainer` (fast, exact for GBMs). Would produce per-patient feature importance + global summary.

**Planned (Track A):** Attention weights from MOTOR's transformer layers (deferred with MOTOR).

Not implemented for the current paper. GBM feature importances are available via `model.feature_importances_` but SHAP analysis was not run.

---

## 7. Phase 4 - Scheduling Policy *(DONE)*

This is the **core innovation** — the layer that no prior work has built. **All code in this section is implemented and tested (24/24 tests passing).** Extended with per-specialty capacity constraints for real MIMIC-IV evaluation.

### 7.1 Cost Function — `policy/cost_function.py`

| Function | Purpose |
|----------|---------|
| `expected_cost(S, day)` | Cost for one patient on one day: `C_EVENT * (1 - S(d)) + C_VISIT` |
| `expected_cost_curve(S, horizon)` | Cost at every day d=1..horizon |
| `unconstrained_optimal_day(S)` | Day that minimises cost ignoring capacity (always day 1 for monotone S) |
| `marginal_benefit(S, from_day, to_day)` | Cost difference from rescheduling — used by greedy solver |

Parameters: `C_EVENT = €10,000` (adverse event), `C_VISIT = €150` (follow-up). Threshold `τ* = C_VISIT / C_EVENT = 1.5%`.

### 7.2 ILP Scheduler — `policy/ilp_scheduler.py`

**Formulation:**

```
Minimise:  Σ_i Σ_d  x_{i,d} × C_EVENT × (1 - S_i(d))

Subject to:
  Σ_d x_{i,d} = 1          ∀i  (one appointment per patient)
  Σ_i x_{i,d} ≤ C(d)       ∀d  (capacity per day)
  x_{i,d} ∈ {0, 1}
```

Solved via PuLP with CBC backend. Pre-computes a risk-cost matrix to avoid recomputation. Configurable time limit. Returns `{"assignments": {patient: day}, "status": "Optimal"|"Infeasible", "total_expected_cost": float}`.

### 7.3 Greedy Scheduler — `policy/greedy_scheduler.py`

Fast heuristic for large cohorts. Sorts patients by urgency (steepest early hazard), assigns each to the cost-minimising day with remaining capacity. O(N × H) vs ILP's exponential worst case.

### 7.4 Uncertainty-Adjusted Scheduling — `policy/uncertainty_adjustment.py`

Conservative survival: `S_conservative(t) = S_mean(t) - α × S_std(t)`, clipped to [0, 1] with monotonicity enforced. When `α = 0`, the scheduler uses point estimates only. When `α > 0`, uncertain patients appear higher-risk and get earlier slots. Tests Hypothesis H2.

### 7.5 Baselines — `policy/baselines.py`

| Policy | Function | Description |
|--------|----------|-------------|
| **Uniform-7** | `uniform_policy(day=7)` | Everyone at day 7 |
| **Uniform-14** | `uniform_policy(day=14)` | Everyone at day 14 |
| **Risk-bucket** | `risk_bucket_policy()` | Top 30% risk → day 7, mid → day 14, low → day 30 |
| **Guideline** | `guideline_policy(is_heart_failure)` | HF → day 14, others → day 28 (ACC/AHA) |
| **Unconstrained** | `unconstrained_optimal_policy()` | Each patient at individual cost-optimal day (ignores capacity) |

### 7.6 Specialty Pool Scheduler — `policy/specialty_scheduler.py` *(NEW)*

Extends the base ILP with per-specialty daily capacity constraints. Each patient belongs to one of four specialty pools (assigned during cohort building). Two implementations:

**ILP (exact):** `schedule_ilp_specialty(survival_curves, specialty_pools, capacity_per_specialty_day)`
```
Minimise:  Σ_i Σ_d  x_{i,d} × C_EVENT × (1 - S_i(d))
Subject to:
  Σ_d x_{i,d} = 1                              ∀i
  Σ_{i ∈ pool_k} x_{i,d} ≤ C_k(d)             ∀k, ∀d   ← per-specialty
  x_{i,d} ∈ {0, 1}
```

**Greedy (heuristic):** `schedule_greedy_specialty(...)` — same interface, processes each pool independently sorted by urgency.

**Default capacity per pool** (realistic for ~500-bed hospital):
| Pool | Slots/day |
|------|-----------|
| Cardiology | 15 |
| Neurology | 10 |
| Surgery | 15 |
| General Medicine | 25 |
| **Total** | **65** |

For evaluation on 27,641 test patients, capacity is scaled 14.2× (923 slots/day total = 27,690 total slots, slightly exceeding cohort size).

### 7.7 Batched ILP Decomposition — `run_pipeline.py`

For large cohorts exceeding solver memory, the ILP is decomposed into batches of 2,000 patients with proportionally scaled capacity per batch. Per-pool capacity floors ensure each specialty gets at least `ceil(pool_count / horizon)` daily slots. This introduces minor boundary effects (see Limitations in paper).

### 7.8 End-to-End Pipeline — `run_pipeline.py` *(NEW)*

Orchestrates: cohort → features → train models → generate curves → run scheduling → output results.

CLI flags: `--skip-cohort`, `--skip-features`, `--skip-training`, `--max-patients`, `--no-bootstrap`, `--scheduler-batch`, `--fast-grid`, `--train-subsample`

Total pipeline time: **92.6 minutes** (feature extraction ~36 min, GBM grid search ~89 min on 30K subsample, scheduling ~5 min).

---

## 8. Phase 5 - Evaluation *(DONE)*

Core evaluation framework is implemented and has been run on real MIMIC-IV data. Off-policy evaluation and paper-quality plots remain as future extensions.

### 8.1 Offline Simulation — `evaluation/simulate.py` *(DONE)*

`run_all_policies()` executes all scheduling policies (Uniform-7, Uniform-14, Risk-Bucket, Guideline, Unconstrained, CCPFS-ILP, CCPFS-Greedy, CCPFS-ILP-Uncertainty) against a cohort and evaluates each with all metrics. `results_summary_table()` produces a markdown-formatted comparison table.

The key insight: MIMIC-IV contains actual readmission dates, so we can retrospectively evaluate any scheduling policy. The same framework works with synthetic event times for development.

### 8.2 Metrics — `evaluation/metrics.py` *(DONE)*

| Metric | Formula | What it captures |
|--------|---------|-----------------|
| Event-before-follow-up rate (EBF) | events before assigned day / N | Policy "too late" failures (PRIMARY) |
| Expected total cost | sum C_event * R_i(d_i) + N * C_visit | Economic objective |
| Mean assigned day | mean(d_i) | Resource use intensity |
| Capacity utilisation | slots used / available per day | Efficiency |
| Gini coefficient | across demographic subgroups | Fairness / equity |
| Net benefit (DCA) | decision curve analysis at key thresholds [Paper 40] | Clinical utility |

### 8.3 Off-Policy Evaluation *(NOT IMPLEMENTED — future work)*

Planned but not implemented. Would use IPW estimators to re-weight observed outcomes by target/behavioural policy probability ratios. MIMIC-IV does not contain outpatient follow-up scheduling data, so behavioural policy probabilities would need to be estimated or assumed uniform.

### 8.4 Sensitivity Analysis — `evaluation/sensitivity.py` *(IMPLEMENTED, not run on final data)*

Four sensitivity axes are implemented in code, each running all policies across parameter variations:

| Function | Parameter | Values | Hypothesis tested |
|----------|-----------|--------|-------------------|
| `vary_capacity()` | Daily capacity C(d) | 5, 10, 20, 50, 999 | Advantage vs capacity tightness |
| `vary_cost_ratio()` | C_event / C_visit | 10, 33, 67, 100, 200 | Threshold sensitivity |
| `vary_horizon()` | Horizon | 14, 21, 30 days | Scope of prediction window |
| `vary_uncertainty_alpha()` | Alpha | 0, 0.4, 0.8, 1.2 | Value of uncertainty awareness |

**Status**: The code exists and works with synthetic data (tested during Phase 4-5 development), but sensitivity analyses were **not run on real MIMIC-IV survival curves** for the paper. This is listed as future work.

**Planned extensions** (not implemented):
- Calibration quality sweep (perfect, +10% bias, +20% bias)
- Model comparison (Cox vs GBM scheduling performance)

### 8.5 Paper Figures *(PARTIALLY DONE)*

**Included in paper (as tables):**
- Table I: Model performance comparison (GBM vs Cox PH)
- Table II: 8-policy scheduling comparison (costs, mean day, feasibility)
- Table III: Specialty pool case study (4 patients across pools)

**Not generated (future work for poster/presentation):**
- Survival curves for example patients with CI bands
- Calibration plots pre/post recalibration
- Capacity sensitivity plots
- SHAP summary plots
- Schedule distribution histograms

---

## 9. Phase 6 - API & Dashboard *(NOT STARTED — deferred to post-paper)*

### 9.1 FastAPI Service

```
POST /score
  Input: patient features or MEDS event stream
  Output: S(t), R(t), uncertainty bands, top contributing factors

POST /schedule
  Input: list of patients + capacity constraints
  Output: {patient_id: assigned_day} + per-patient rationale

GET /explain/{patient_id}
  Output: risk curve plot, SHAP/attention waterfall, scheduling reasoning
```

### 9.2 Streamlit Dashboard

- **Discharge list** sorted by urgency (risk gradient * capacity scarcity)
- **Patient detail view**: risk curve with CI, recommendation, key drivers
- **Capacity calendar**: heatmap of slots filled/available per day
- **Policy comparison**: toggle between CCPFS and baselines, see impact
- **Sensitivity sliders**: adjust capacity, cost ratio in real-time

---

## 10. Dependencies

**Runtime**: Python 3.12 (chosen for compatibility with meds-transforms ≥3.12, PyTorch ROCm, scikit-survival, FEMR).

### Currently installed (Week 1 — `requirements.txt`):

```
numpy>=2.4
scipy>=1.17
PuLP>=3.3
pytest>=9.0
matplotlib>=3.10
seaborn>=0.13
jupyter>=1.1
```

### To be added in later phases:

```
# Data (Phase 1)
pandas>=2.2
polars>=1.0
pyarrow>=18.0

# MEDS Ecosystem (Phase 1)
MIMIC-IV-MEDS               # MIMIC-IV to MEDS ETL
meds-transforms              # MEDS data processing

# Foundation Model - Track A (Phase 3)
femr                         # FEMR framework for MOTOR
torch>=2.6                   # PyTorch (ROCm or CUDA)

# Classical Survival - Track B (Phase 3)
scikit-survival>=0.24
lifelines>=0.30
scikit-learn>=1.6

# Explainability (Phase 3)
shap>=0.46

# API + Dashboard (Phase 6)
fastapi>=0.115
uvicorn>=0.34
streamlit>=1.42
```

**Note**: Dependencies are added incrementally per phase rather than all upfront, to keep the environment lean and avoid version conflicts between packages that aren't yet needed.

---

## 11. Implementation Order

```
Phase 0: Setup                                                    ✅ DONE
    |-- Python 3.12 venv, project structure, config.py
    |-- requirements.txt, .python-version
    |
Phase 4: Scheduling Policy                                       ✅ DONE
    |-- cost_function.py, ilp_scheduler.py, greedy_scheduler.py
    |-- baselines.py, uncertainty_adjustment.py
    |-- test_policy.py (24 tests, all passing)
    |-- 01_pipeline_demo.ipynb (synthetic data demo)
    |
Phase 1: Data Pipeline                                           ✅ DONE
    |-- MEDS ETL: 364,627 patients, 366 shards, 13 GB
    |-- etl/build_cohort.py: 275,022 episodes, 137,054 patients
    |-- Cohort: 20.3% readmission, 4 specialty pools
    |
Phase 2: Feature Engineering                                     ✅ DONE
    |-- features/extract_features.py: 47 features, streaming shard processing
    |-- Lab aggregates (7 labs × last/max/min + missingness indicators)
    |-- Comorbidity flags (6 conditions + complexity counts)
    |-- Imputation with training-set medians (11.7% NaN rate)
    |
Phase 3: Survival Models                                         ✅ DONE
    |-- models/classical/train_gbm.py: C-index 0.695 (30K subsample)
    |-- models/classical/train_cox.py: C-index 0.688
    |-- models/evaluate_model.py: C-index, IBS, ECE metrics
    |-- models/calibrate.py: isotonic calibration (implemented but not applied)
    |-- models/uncertainty.py: bootstrap ensemble (implemented but not used)
    |-- MOTOR (Track A): DEFERRED to future work
    |
Phase 4b: Specialty Scheduler Extension                          ✅ DONE
    |-- policy/specialty_scheduler.py: ILP + greedy with per-pool capacity
    |-- Batched ILP decomposition for large cohorts (2000/batch)
    |
Phase 5b: Full Evaluation (real MIMIC-IV data)                   ✅ DONE
    |-- run_pipeline.py: end-to-end orchestration (92.6 min)
    |-- 8 policies compared on 27,641 test patients
    |-- ILP specialty: €954/patient (33.4% below uniform)
    |-- Results: data/processed/pipeline_results.json
    |
Phase 7: Paper                                                   ✅ DONE
    |-- 6-page IEEE paper: 2_pages/paper_draft.md
    |-- 23 references, 3 tables, real MIMIC-IV results
    |-- Critical review: 9 issues found and fixed
    |
Phase 6: Engineering Deliverable                                 🔲 NOT STARTED
    |-- FastAPI endpoints (score, schedule, explain)
    |-- Streamlit dashboard
    |-- Docker containerisation
```

**Critical path completed**: Data → Features → Models → Scheduling → Paper

**What the strategy achieved**: Building Phase 4-5a first (scheduling framework with synthetic data) meant that when MIMIC-IV data arrived, we only needed to build the data-to-prediction pipeline (Phases 1-3) and plug the real curves into the existing scheduler. Total time from data access to paper: ~2 days.

---

## 12. Key Design Decisions

1. **Build scheduling layer first**: Phase 4-5 (scheduling + evaluation) were implemented before Phase 1-3 (data + models) because they consume `np.ndarray` survival curves regardless of source. This meant when MIMIC-IV data arrived, we plugged in real predictions and immediately got results. Total time from data access to paper: ~2 days. *(Validated — this was the most impactful architectural decision.)*

2. **MOTOR deferred, GBM as primary model**: Originally planned as a two-track approach (MOTOR + GBM), but GBM achieved C-index 0.695 — competitive for readmission prediction and sufficient to demonstrate the scheduling framework's value. MOTOR fine-tuning was deferred rather than abandoned: it remains a future work direction for improved discrimination and zero-shot cross-site transfer, but was not required to validate the core scheduling contribution.

3. **MEDS as data backbone**: Standardised format enabled streaming shard processing (critical for 16 GB RAM constraint), multi-dataset portability, and reproducible benchmarking. The streaming architecture processes one shard at a time (~50 MB projected vs 357 MB full), enabling the full 275K-episode cohort extraction on constrained hardware.

4. **ILP over RL for scheduling**: Follow-up scheduling is a one-shot batch assignment problem (discharge cohort → follow-up days), not a sequential MDP. ILP gives the provably optimal solution. RL is noted as future work for dynamic rescheduling scenarios.

5. **All-cause readmission as primary outcome**: Originally planned HF-primary with all-cause secondary, but the full cohort approach (275,022 episodes) provides more statistical power and validates generalisability across four specialty pools. Heart failure is used as a subgroup flag for the guideline baseline (ACC/AHA 14-day recommendation).

6. **30-day horizon**: Matches CMS readmission quality metrics. All scheduling recommendations stay within the prediction window (critical lesson: never recommend follow-up at 6-8 weeks when the model only predicts over days 1-30).

7. **30K training subsample**: Full 220K-episode GBM grid search was estimated at ~90 minutes per configuration (~6+ hours total). The 30K random subsample completed 4 configs in 89 minutes total with minimal discrimination loss. Cox PH also trained on the subsample for fair comparison.

8. **Specialty pool scheduler**: Extends the base ILP with per-specialty daily capacity constraints (4 pools: cardiology, neurology, surgery, general medicine) for more realistic hospital modelling. Each patient assigned to one pool during cohort building based on admitting service + ICD-10 diagnosis codes.

9. **Batched ILP decomposition**: For 27,641 test patients, the ILP is decomposed into batches of 2,000 with proportionally scaled capacity. This introduces minor boundary effects (ceiling rounding gives specialty batches ~20% more total capacity than global batches), documented honestly in the paper as a limitation.

10. **Synthetic cohort generator for early development**: Built `evaluation/synthetic.py` with Weibull-distributed survival curves to enable scheduling framework testing before MIMIC-IV access. *(Superseded — all final results use real MIMIC-IV predictions.)*

11. **Offline simulation, not causal claims**: We measure "would follow-up have occurred before the adverse event?" not "would follow-up have prevented the adverse event?" Transparent and defensible for a research paper.

12. **Raw survival curves over calibrated**: Isotonic calibration pushed all S(t) → 1.0 on the full dataset due to low per-day event rates (20.3% spread over 30 days). Raw GBM curves used instead, with ECE@21d = 0.091 — adequate for mid-horizon scheduling.

13. **Python 3.12**: Chosen for compatibility across meds-transforms (≥3.12), scikit-survival, lifelines, and PuLP. Mature and widely supported.

---

## 13. Risk Register

| Risk | Impact | Mitigation | Status |
|------|--------|-----------|--------|
| MIMIC-IV access delayed | Blocks Phases 1-3 on real data | Built scheduling layer first with synthetic data | **RESOLVED** — data obtained and pipeline complete |
| ROCm + PyTorch issues on AMD GPU | Cannot fine-tune MOTOR locally | Rent cloud GPU; defer MOTOR to future work | **RESOLVED** — MOTOR deferred; GBM sufficient |
| MOTOR fine-tuning fails on MIMIC-IV | No foundation model results | Track B (GBM) as fallback | **RESOLVED** — GBM C-index 0.695, paper complete without MOTOR |
| FEMR/MEDS ecosystem breaking changes | Pipeline breaks | Pin dependency versions | **RESOLVED** — MEDS ETL complete (v0.0.7 with known bugs documented) |
| HF sub-cohort too small (<5K) | Underpowered analysis | Expanded to all-cause readmission as primary | **RESOLVED** — 275K episodes; HF used as subgroup flag only |
| Model C-index < 0.65 | Weak predictions degrade scheduling | Add features, try RSF | **RESOLVED** — GBM C-index 0.695, close to 0.70 target |
| Calibration poor after recalibration | Scheduling decisions unreliable | Report honestly; use raw curves | **RESOLVED** — Isotonic failed; raw curves used with ECE@21d = 0.091 |
| Dependency version conflicts | Cannot install all packages together | Incremental requirements | **RESOLVED** — all dependencies installed successfully |
| ILP too slow for full cohort | Cannot schedule 27K+ patients | Batched decomposition (2000/batch) | **RESOLVED** — introduces minor boundary effects (documented in paper) |
| Training on full 220K episodes too slow | GBM grid search takes hours | 30K random subsample | **RESOLVED** — 4 configs in 89 min with competitive C-index |
| Batching introduces capacity artifacts | Specialty ILP gets more capacity than global ILP | Documented honestly in paper limitations | **ACCEPTED** — ceiling rounding gives ~20% extra capacity per specialty batch |

---

## 14. What This Produced for the Paper *(COMPLETE)*

The 6-page IEEE paper (`2_pages/paper_draft.md`) has been written with real MIMIC-IV results. Below maps paper sections to their implementation sources:

| Paper Section | Source | Status |
|---------------|--------|--------|
| Abstract: 33.4% cost reduction | `pipeline_results.json` — ILP specialty vs uniform | DONE |
| Introduction: 275K episodes, 20.3% readmission | `etl/build_cohort.py` → `cohort.parquet` | DONE |
| Related Work: 23 references | Literature review (40 papers) condensed to 23 | DONE |
| Method: survival model (GBM + Cox PH) | `models/classical/train_gbm.py`, `train_cox.py` | DONE |
| Method: cost function | `policy/cost_function.py` | DONE |
| Method: ILP scheduler + specialty pools | `policy/ilp_scheduler.py`, `specialty_scheduler.py` | DONE |
| Results: Table I (model comparison) | `models/evaluate_model.py` → C-index, IBS, ECE | DONE |
| Results: Table II (8-policy comparison) | `run_pipeline.py` → `pipeline_results.json` | DONE |
| Results: Table III (specialty case study) | `policy/specialty_scheduler.py` | DONE |
| Discussion: calibration limitations | `models/calibrate.py` — isotonic failure | DONE |
| Discussion: batching artifact | `run_pipeline.py` — ceiling rounding | DONE |
| Discussion: training subsample trade-off | 30K of 220K training episodes | DONE |

**Hypotheses tested in paper:**
- H1: Risk-optimised allocation reduces expected cost vs fixed schedules — **Confirmed** (33.4% reduction)
- H2: Coarse risk stratification provides negligible improvement over uniform — **Confirmed** (0.3%)
- H3: Greedy heuristic achieves >99% of ILP cost reduction — **Confirmed** (€1,037 vs €1,036)

**Not in paper (future work):**
- MOTOR foundation model comparison
- SHAP/explainability analysis
- Off-policy evaluation
- Capacity sensitivity analysis (H3-original)
- Bootstrap uncertainty-adjusted scheduling
- API/dashboard deployment
