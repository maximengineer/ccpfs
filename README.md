# CCPFS - Capacity-Constrained Personalised Follow-Up Scheduling

An optimisation framework that uses patient-level survival predictions to schedule post-discharge follow-up appointments under clinic capacity constraints, minimising adverse events while respecting resource limits.

**Validated on MIMIC-IV v3.1** - 275,022 discharge episodes from 137,054 patients.

## Problem

After hospital discharge, patients need follow-up appointments. Current practice uses fixed schedules (e.g., "everyone at 14 days") that ignore individual risk profiles and clinic capacity. High-risk patients may wait too long; low-risk patients consume scarce slots unnecessarily.

CCPFS solves this by:
1. **Predicting** each patient's 30-day readmission risk curve S(t) — the framework is model-agnostic and accepts curves from any survival model (GBM, Cox PH, RSF, MOTOR, or custom)
2. **Optimising** appointment timing via capacity-constrained assignment to minimise total expected cost under per-specialty daily limits
3. **Evaluating** the policy against clinical baselines using retrospective simulation on real EHR data

## Project Structure

```
scheduling_follow_up/
├── config.py                       # Central configuration (costs, horizons, paths)
├── run_pipeline.py                 # Step-based pipeline orchestrator
├── parallel_train.py               # Parallel model training (GBM + Cox + RSF concurrently)
├── requirements.txt                # Python dependencies
│
├── etl/                            # Data pipeline
│   └── build_cohort.py             # MEDS → discharge cohort (275K episodes)
│
├── features/                       # Feature engineering
│   ├── extract_features.py         # Base feature extraction pipeline
│   ├── derive_features.py          # Derived features (ranges, ratios, clinical flags)
│   ├── lab_aggregates.py           # Lab/vital aggregation (28 measurements × last/max/min)
│   └── comorbidity_flags.py        # ICD-10 comorbidity flag extraction (12 conditions)
│
├── models/                         # Survival prediction
│   ├── classical/
│   │   ├── train_gbm.py            # Gradient-boosted survival analysis
│   │   ├── train_cox.py            # Cox PH baseline (with feature scaling)
│   │   └── train_rsf.py            # Random Survival Forest
│   ├── motor/
│   │   ├── extract_motor_float32.py # MOTOR embedding extraction (JAX, CUDA fallbacks)
│   │   ├── align_embeddings.py      # Align MOTOR embeddings with cohort splits
│   │   ├── train_on_embeddings.py   # Train GBM on MOTOR embeddings
│   │   ├── train_motor.py           # End-to-end MOTOR pipeline
│   │   ├── meds_to_simple_femr.py   # MEDS → simple femr CSV conversion
│   │   └── build_omop_mapping.py    # MEDS codes → OMOP standard concepts
│   ├── evaluate_model.py           # C-index, IBS, ECE evaluation
│   ├── calibrate.py                # Isotonic regression calibration
│   └── uncertainty.py              # Bootstrap uncertainty estimation
│
├── policy/                         # Scheduling algorithms
│   ├── cost_function.py            # Expected cost C(d) = C_EVENT·(1-S(d)) + C_VISIT
│   ├── ilp_scheduler.py            # Exact ILP solver (PuLP/CBC)
│   ├── mincost_solver.py           # Min-cost assignment (scipy, handles full cohort)
│   ├── greedy_scheduler.py         # Fast heuristic scheduler
│   ├── specialty_scheduler.py      # Per-specialty capacity ILP + greedy
│   ├── baselines.py                # Uniform, risk-bucket, guideline, unconstrained
│   └── uncertainty_adjustment.py   # Conservative scheduling under prediction uncertainty
│
├── evaluation/                     # Metrics and simulation
│   ├── synthetic.py                # Weibull-based synthetic cohort generator
│   ├── metrics.py                  # EBF rate, catch rate, cost, capacity utilisation
│   ├── simulate.py                 # Run all policies and produce comparison tables
│   └── sensitivity.py              # Vary capacity, cost ratio, uncertainty, horizon
│
├── tests/                          # pytest suite (24 tests)
│   └── test_policy.py
│
├── notebooks/                      # Jupyter demonstrations
│   └── 01_pipeline_demo.ipynb
│
└── data/                           # Gitignored - restricted MIMIC-IV data
    ├── meds/                       # MEDS-format MIMIC-IV (364K patients, 366 shards)
    └── processed/                  # Cohort, features, models, results
        ├── cohort.parquet           # Output of --step cohort
        ├── features.npz             # Output of --step features
        ├── curves_test.npz          # Output of --step train / calibrate
        ├── models_info.json         # Model metrics and params
        ├── scheduling_results.npz   # Output of --step schedule
        ├── pipeline_results.json    # Output of --step report
        └── motor_output/            # MOTOR foundation model results
            ├── aligned_embeddings.npz  # Train/val/test embeddings (768-dim)
            ├── motor_curves.npz        # S(t) curves from MOTOR+GBM
            ├── motor_gbm.joblib        # Trained GBM on PCA embeddings
            ├── motor_pca.joblib        # PCA transformer (768→64)
            ├── motor_scaler.joblib     # StandardScaler for embeddings
            └── motor_result.json       # Metrics and hyperparameters
```

## Quick Start

```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

### Running the Pipeline (Step by Step)

Each step saves its output to `data/processed/` so subsequent steps can run independently. This makes debugging and iteration much faster.

```bash
# Full pipeline (all steps, all models)
python run_pipeline.py

# Run individual steps
python run_pipeline.py --step cohort              # Build cohort → cohort.parquet
python run_pipeline.py --step features            # Extract features → features.npz
python run_pipeline.py --step train --model gbm   # Train GBM only → curves_test.npz
python run_pipeline.py --step calibrate           # Calibrate curves → curves_test.npz (updated)
python run_pipeline.py --step schedule            # Run policies → scheduling_results.npz
python run_pipeline.py --step report              # Print results → pipeline_results.json

# Combine steps
python run_pipeline.py --step train,calibrate,schedule,report --model gbm

# Speed options
python run_pipeline.py --step train --model gbm --fast-grid          # 4 combos vs 324
python run_pipeline.py --step train --model gbm --train-subsample 50000
python run_pipeline.py --max-patients 5000 --step train,schedule     # Small test run
```

### Model Selection

Use `--model` to choose which survival model(s) to train and evaluate:

```bash
python run_pipeline.py --step train --model gbm        # GBM only
python run_pipeline.py --step train --model cox         # Cox PH only
python run_pipeline.py --step train --model rsf         # Random Survival Forest only
python run_pipeline.py --step train --model gbm,cox     # GBM + Cox
python run_pipeline.py --step train --model all         # All three (default)
```

The best model (by C-index on validation set) is automatically selected for scheduling. Curves are saved to `curves_test.npz` so the calibrate/schedule/report steps can run independently later.

### Using Your Own Model (Model-Agnostic Interface)

The scheduling framework accepts survival curves from **any** model. Save your curves as a `.npz` file:

```bash
# Use pre-computed survival curves from any external model
python run_pipeline.py --survival-curves path/to/curves.npz --step schedule,report
```

The `.npz` file must contain:
- `curves`: np.ndarray of shape `(N, 31)` — survival probabilities S(t) for t=0..30
- `model_name`: str (optional) — name for reporting (e.g., "MOTOR")

```python
# Example: generating curves from MOTOR foundation model
import numpy as np

# ... fine-tune MOTOR on MIMIC-IV, then:
curves = motor_model.predict_survival(cohort, times=range(31))  # (N, 31)
np.savez("motor_curves.npz", curves=curves, model_name="MOTOR")

# Then schedule:
# python run_pipeline.py --survival-curves motor_curves.npz --step schedule,report
```

## How It Works

### 1. Cohort Building

Extracts eligible discharge episodes from MIMIC-IV in MEDS format via streaming shard processing. Filters for adults discharged home with LOS >= 1 day. Assigns each episode to one of four specialty pools based on ICD-10 diagnosis codes. Computes 30-day readmission as the outcome.

### 2. Feature Engineering

Extracts ~137 base features in 5 groups, then generates ~55 derived clinical features (192 total after imputation):

**Base features:**
- **Static** (5): age, gender, LOS, admission type, ED origin
- **Comorbidity** (14): 12 condition flags (HF, diabetes, CKD, COPD, hypertension, AFib, liver disease, malignancy, depression, obesity, stroke, ACS) + diagnosis counts
- **Labs & Vitals** (112): 28 measurements (7 core labs + 12 additional labs + 9 vital signs) × last/max/min + missingness indicators
- **Prior utilisation** (3): prior admissions count, 365-day count, days since last discharge
- **Procedures/ICU** (3): medication count, procedure count, ICU stay flag

**Derived features (~55):**
- Lab instability ranges, BUN/creatinine ratio
- Clinical threshold flags: tachycardia, hypotension, tachypnea, hypoxemia, elevated troponin/lactate/INR, hypoalbuminemia, anemia, renal impairment
- Comorbidity burden score, log-transforms for skewed counts
- Interaction features: age × HF, prior admissions × LOS, emergency × ICU
- Discharge acuity composite

### 3. Survival Prediction

Three survival models compared. Best model selected by C-index on validation set. Outputs patient-specific survival curves S_i(t) for t = 0..30 days. Cox PH uses StandardScaler (fitted on training data only). Isotonic regression calibration is applied on the validation set before scheduling.

### 4. Capacity-Constrained Scheduling

Each patient *i* has a predicted survival curve S_i(t). Scheduling follow-up on day *d* incurs:

```
Cost_i(d) = C_EVENT × (1 - S_i(d)) + C_VISIT
```

where C_EVENT = €10,000 (adverse event cost) and C_VISIT = €150 (appointment cost).

Two exact solvers: ILP (PuLP/CBC) for small cohorts, and min-cost assignment (scipy `linear_sum_assignment`) for the full cohort without batching.

**Specialty pool capacity** (base rates for ~500-bed hospital):

| Pool | Slots/day |
|------|-----------|
| Cardiology | 15 |
| Neurology | 10 |
| Surgery | 15 |
| General Medicine | 25 |
| **Total** | **65** |

### Baselines

| Policy | Description |
|--------|-------------|
| Uniform-14 | Everyone at day 14 |
| Risk-Bucket | High risk → day 7, Medium → day 14, Low → day 30 |
| Guideline (ACC/AHA) | Heart failure → day 14, Others → day 28 |
| Unconstrained | Each patient at individual cost-optimal day (ignores capacity) |
| Uniform-14 (capacity) | Day 14 with overflow to nearest available (capacity-aware) |
| Guideline (capacity) | Guideline with overflow handling (capacity-aware) |

## Results (MIMIC-IV, 27,641 Test Patients)

### Model Performance

| Model | C-index | IBS |
|-------|---------|-----|
| **GBM** | **0.706** | **0.097** |
| Cox PH | 0.702 | 0.098 |
| RSF | 0.698 | 0.098 |
| MOTOR+GBM | 0.669 | 0.101 |

The three classical models achieve near-identical discrimination (C-index spread: 0.008), confirming that the scheduling framework's value comes from the optimisation layer, not the specific risk model. GBM selected as best model for scheduling. Isotonic calibration applied at horizons 7, 14, 21, 30 days.

The MOTOR-T-Base foundation model (143M params, pretrained on 2.57M Stanford EHRs) was also evaluated using frozen embeddings reduced to 64 dimensions via PCA (93% variance retained). Its lower performance reflects domain shift (Stanford → MIMIC-IV) and the advantage of purpose-built features for this specific task, while validating the framework's model-agnostic design.

### Scheduling Policy Comparison

| Policy | Capacity | Avg Cost (€) | Catch Rate |
|--------|:--------:|-----------:|-----------:|
| Guideline (ACC/AHA) | No | 1,992 | 9.8% |
| Uniform day 14 | No | 1,422 | 37.3% |
| Uniform-14 (capacity) | Yes | 1,392 | 38.3% |
| Greedy (specialty) | Yes | 946 | 61.7% |
| **MinCost (specialty)** | **Yes** | **759** | **71.0%** |
| Unconstrained (oracle) | No | 254 | 96.8% |

The optimised specialty scheduler achieves **47% cost reduction** vs uniform-14 and catches **71% of adverse events** before follow-up (vs 37% for uniform). Per-specialty capacity pooling adds 24% improvement over global pooling.

## Technology

- **Python 3.12**
- **NumPy / SciPy** - numerical computation, min-cost assignment solver
- **PuLP** - ILP formulation and CBC solver
- **scikit-survival** - gradient-boosted survival analysis, Random Survival Forest
- **lifelines** - Cox proportional hazards
- **scikit-learn** - StandardScaler, grid search utilities, isotonic regression
- **Polars / PyArrow** - streaming data processing
- **matplotlib / seaborn** - visualisation
- **pytest** - test suite (24 tests passing)

## Data

**All data directories are gitignored** - the `data/` folder, trained models in `models/saved/`, and derived outputs are excluded from version control because they contain or are derived from restricted clinical data.

### Access Requirements

This project uses **MIMIC-IV v3.1** converted to **MEDS format**. To reproduce:

1. Complete the [CITI training](https://physionet.org/about/citi-course/) for human subjects research
2. Apply for credentialed access to [MIMIC-IV on PhysioNet](https://physionet.org/content/mimiciv/3.1/)
3. Download and convert to MEDS format using [MEDS-ETL](https://github.com/Medical-Event-Data-Standard/meds_etl)
4. Place output in `data/meds/MEDS_cohort/` (or update paths in `config.py`)

### Cohort Statistics

275,022 eligible discharge episodes from 137,054 unique patients, 20.5% 30-day readmission rate:

| Pool | Episodes | Share |
|------|----------|-------|
| General Medicine | 151,509 | 55.1% |
| Cardiology | 54,173 | 19.7% |
| Neurology | 37,724 | 13.7% |
| Surgery | 31,616 | 11.5% |
