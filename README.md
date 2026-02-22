# CCPFS - Capacity-Constrained Personalised Follow-Up Scheduling

An optimisation framework that uses patient-level survival predictions to schedule post-discharge follow-up appointments under clinic capacity constraints, minimising adverse events while respecting resource limits.

**Validated on MIMIC-IV v3.1** - 275,022 discharge episodes from 137,054 patients, achieving a 33.4% cost reduction over standard uniform scheduling.

## Problem

After hospital discharge, patients need follow-up appointments. Current practice uses fixed schedules (e.g., "everyone at 14 days") that ignore individual risk profiles and clinic capacity. High-risk patients may wait too long; low-risk patients consume scarce slots unnecessarily.

CCPFS solves this by:
1. **Predicting** each patient's 30-day readmission risk curve using gradient-boosted survival analysis
2. **Optimising** appointment timing via Integer Linear Programming (ILP) to minimise total expected cost under per-specialty daily capacity constraints
3. **Evaluating** the policy against clinical baselines using retrospective simulation on real EHR data

## Key Results

Evaluated on 27,641 held-out test patients from MIMIC-IV across 8 scheduling policies:

| Policy | Cost/patient | vs Uniform-14 |
|--------|-------------|----------------|
| Uniform-14 (baseline) | €1,433 | - |
| Guideline (ACC/AHA) | €2,025 | +41.3% |
| Risk-Bucket | €1,429 | -0.3% |
| **CCPFS ILP (specialty)** | **€954** | **-33.4%** |
| CCPFS ILP (global) | €1,036 | -27.7% |
| CCPFS Greedy (global) | €1,037 | -27.6% |

The ILP with per-specialty capacity constraints (cardiology, neurology, surgery, general medicine) achieves the lowest cost while respecting realistic clinic resource limits.

## Project Structure

```
scheduling_follow_up/
├── config.py                       # Central configuration (costs, horizons, paths)
├── run_pipeline.py                 # End-to-end pipeline orchestrator
├── requirements.txt                # Python dependencies
├── PLAN.md                         # Full implementation plan and design decisions
│
├── etl/                            # Data pipeline
│   └── build_cohort.py             # MEDS → discharge cohort (275K episodes)
│
├── features/                       # Feature engineering
│   ├── extract_features.py         # 47-feature extraction pipeline
│   ├── lab_aggregates.py           # Lab value aggregation (7 labs × last/max/min)
│   └── comorbidity_flags.py        # ICD-10 comorbidity flag extraction
│
├── models/                         # Survival prediction
│   ├── classical/
│   │   ├── train_gbm.py            # Gradient-boosted survival analysis (C-index 0.695)
│   │   └── train_cox.py            # Cox PH baseline (C-index 0.688)
│   ├── evaluate_model.py           # C-index, IBS, ECE evaluation
│   ├── calibrate.py                # Isotonic regression calibration
│   └── uncertainty.py              # Bootstrap uncertainty estimation
│
├── policy/                         # Scheduling algorithms
│   ├── cost_function.py            # Expected cost C(d) = C_EVENT·(1-S(d)) + C_VISIT
│   ├── ilp_scheduler.py            # Exact ILP solver (PuLP/CBC)
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
│   └── 01_pipeline_demo.ipynb      # Pipeline demo with visualisations
│
└── data/                           # Gitignored - restricted MIMIC-IV data
    ├── meds/                       # MEDS-format MIMIC-IV (364K patients, 366 shards)
    └── processed/                  # Cohort, features, models, results
```

## Quick Start

```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests (scheduling framework)
pytest tests/ -v

# Run full pipeline (requires MIMIC-IV MEDS data in data/meds/)
python run_pipeline.py

# Run with options
python run_pipeline.py --skip-cohort --skip-features    # reuse cached data
python run_pipeline.py --train-subsample 30000           # limit training size
python run_pipeline.py --fast-grid                       # reduced hyperparameter search
```

## How It Works

### 1. Cohort Building

Extracts eligible discharge episodes from MIMIC-IV in MEDS format via streaming shard processing (one shard at a time, ~50 MB projected). Filters for adults discharged home with LOS >= 1 day. Assigns each episode to one of four specialty pools based on admitting service and ICD-10 diagnosis codes. Computes 30-day readmission as the outcome.

### 2. Feature Engineering

Extracts 47 features in 5 groups:
- **Static** (8): age, gender, insurance, LOS, admission type, ED origin
- **Comorbidity** (8): HF, diabetes, CKD, COPD, hypertension, AFib flags + complexity counts
- **Labs** (28): 7 labs (creatinine, BUN, sodium, potassium, Hgb, WBC, BNP) × last/max/min + missingness indicators
- **Prior utilisation** (3): prior admissions count, 365-day count, days since last discharge
- **Procedures/ICU** (3): medication count, procedure count, ICU stay flag

### 3. Survival Prediction

Gradient-Boosted Survival Analysis (scikit-survival) trained on 30,000 episodes. Outputs patient-specific survival curves S_i(t) for t = 0..30 days.

| Model | C-index | IBS |
|-------|---------|-----|
| GBSA | 0.695 | 0.099 |
| Cox PH | 0.688 | 0.100 |

### 4. Capacity-Constrained Scheduling

Each patient *i* has a predicted survival curve S_i(t). Scheduling follow-up on day *d* incurs:

```
Cost_i(d) = C_EVENT × (1 - S_i(d)) + C_VISIT
```

where C_EVENT = €10,000 (adverse event cost) and C_VISIT = €150 (appointment cost).

The ILP assigns each patient to exactly one day, minimising total expected cost subject to:
- Each patient is seen exactly once within the 30-day horizon
- No specialty pool exceeds its daily capacity

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
| Uniform-7/14 | Everyone at day 7 or 14 |
| Risk-Bucket | High risk → day 7, Medium → day 14, Low → day 30 |
| Guideline (ACC/AHA) | Heart failure → day 14, Others → day 28 |
| Unconstrained | Each patient at their individual cost-optimal day (ignores capacity) |

## Hypotheses

- **H1**: Risk-optimised allocation under capacity constraints reduces expected cost compared to fixed-schedule and guideline policies - **Confirmed** (33.4% reduction vs uniform)
- **H2**: Coarse risk stratification provides negligible improvement over uniform scheduling - **Confirmed** (0.3% reduction)
- **H3**: The greedy heuristic achieves near-optimal cost reduction compared to the exact ILP - **Confirmed** (>99% of ILP's cost reduction)

## Technology

- **Python 3.12**
- **NumPy / SciPy** - numerical computation
- **PuLP** - ILP formulation and CBC solver
- **scikit-survival** - gradient-boosted survival analysis
- **lifelines** - Cox proportional hazards
- **Polars / PyArrow** - streaming data processing
- **matplotlib / seaborn** - visualisation
- **pytest** - test suite (24 tests passing)

## Data

**All data directories are gitignored** - the `data/` folder, trained models in `models/saved/`, and derived outputs are excluded from version control because they contain or are derived from restricted clinical data.

### Access Requirements

This project uses **MIMIC-IV v3.1** converted to **MEDS format** (Medical Event Data Standard). To reproduce the pipeline:

1. Complete the [CITI training](https://physionet.org/about/citi-course/) for human subjects research
2. Apply for credentialed access to [MIMIC-IV on PhysioNet](https://physionet.org/content/mimiciv/3.1/)
3. Download the data and convert to MEDS format using [MEDS-ETL](https://github.com/Medical-Event-Data-Standard/meds_etl)
4. Place the MEDS output in `data/meds/MEDS_cohort/` (or update paths in `config.py`)

The MEDS conversion produces 364,627 patients across 366 parquet shards (~13 GB). The full raw MIMIC-IV data is ~60 GB decompressed.

### Cohort Statistics

The cohort extraction yields 275,022 eligible discharge episodes from 137,054 unique patients, with a 20.3% 30-day readmission rate and the following specialty distribution:

| Pool | Episodes | Share |
|------|----------|-------|
| General Medicine | 151,509 | 55.1% |
| Cardiology | 54,173 | 19.7% |
| Neurology | 37,724 | 13.7% |
| Surgery | 31,616 | 11.5% |
