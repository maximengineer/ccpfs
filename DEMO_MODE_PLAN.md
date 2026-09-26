# Demo Mode - Design & Implementation

## Goal

Let anyone clone the repo and run the CCPFS framework end-to-end in minutes, without MIMIC-IV credentialed access. The dashboard shows meaningful results using synthetic patient data, and the setup never interferes with a real MIMIC-IV pipeline run on the same machine.

## User Experience

```bash
git clone https://github.com/maximengineer/ccpfs
cd ccpfs
pip install -r requirements.txt
python demo_setup.py                                       # ~30s - 10K synthetic patients
docker compose -f docker-compose.demo.yml up               # Dashboard at http://localhost:3000
```

When finished:

```bash
python demo_setup.py --clean                               # remove data/demo/
```

## Design Decisions

### 1. Separate output directory - never touch `data/processed/`

**Problem.** Many users will already have real MIMIC-IV output in `data/processed/` from running `run_pipeline.py`. Writing synthetic files there would silently overwrite real results.

**Decision.** Default output is `data/demo/` (a sibling of `data/processed/`). Both live under the gitignored `data/` folder, so neither is committed; they just never occupy the same namespace.

**Enforcement.** The script inspects `<out_dir>/pipeline_results.json` before writing. If it exists and does **not** have `"synthetic": true`, the script refuses to run. A `--force` flag exists as an escape hatch, but writing to the default `data/demo/` needs no flag.

### 2. Standalone compose file (`docker-compose.demo.yml`), not an override

**Problem.** Using `-f docker-compose.yml -f docker-compose.demo.yml` to override one volume is fragile: Compose *merges* arrays, so two `/data` mounts collide and behaviour depends on Compose version.

**Decision.** Ship a second, fully standalone compose file that points the backend at `./data/demo`. Run it with `docker compose -f docker-compose.demo.yml up`. Plain `docker compose up` remains the MIMIC-IV path pointing at `./data/processed`. Two commands, two data paths, zero ambiguity.

### 3. Single script, not a pipeline flag

**Problem.** Adding `--demo` to `run_pipeline.py` would entangle the demo path with the real ETL/feature/training stages, which assume MIMIC-IV is available.

**Decision.** `demo_setup.py` is self-contained. It bypasses `run_pipeline.py` entirely and writes directly the artefacts `api/dependencies.py` loads at startup. No training, no ETL, no feature engineering needed.

### 4. `--clean` removes the demo data safely

`python demo_setup.py --clean` deletes the output directory, but re-runs the same safety check first - it refuses to delete a directory whose `pipeline_results.json` is not marked synthetic. This means `--clean` is safe to run even if the user accidentally pointed `--out-dir` at real data.

### 5. Reuse the real framework; only the data is synthetic

Survival curves come from `evaluation/synthetic.py` (Weibull-based). All scheduling policies call the real solvers in `policy/`. Metrics use `evaluation/metrics.py`. Only the cohort generation is synthetic; the optimisation and evaluation stack is bit-identical to the MIMIC-IV path. This is what makes the dashboard's numbers meaningful.

### 6. Every artefact is marked synthetic

`pipeline_results.json`, `models_info.json`, and `motor_result.json` all carry a top-level `"synthetic": true` flag. Anyone inspecting the files (or the API response, via `/api/results`) knows immediately this is demo data, not MIMIC-IV.

### 7. Small cohort by default

10,000 patients by default (vs. 275,000 in MIMIC-IV). Fast to generate, fast to schedule, still large enough that the dashboard's test split (1,000 patients) feels real. `--n-patients` lets power users scale up.

## Data Structure

### Cohort composition

Matches MIMIC-IV specialty distribution:

| Pool | Share | Count (N=10,000) |
|------|------:|-----:|
| General Medicine | 55.1% | 5,510 |
| Cardiology | 19.7% | 1,970 |
| Neurology | 13.7% | 1,370 |
| Surgery | 11.5% | 1,150 |

### Risk mix per specialty

Weibull parameters from `evaluation/synthetic.py`:

- High-risk (25%): scale=40, shape=1.5 -> ~35% 30-day event rate
- Medium-risk (50%): scale=80, shape=1.3 -> ~15% 30-day event rate
- Low-risk (25%): scale=200, shape=1.1 -> ~5% 30-day event rate

Overall event rate ~20%, matching MIMIC-IV.

### Splits

- `train`: 80%
- `tuning`: 10%
- `held_out`: 10% (the only split the dashboard needs)

### Columns in `cohort.parquet`

Required by `api/dependencies.py` and `api/routers/scheduling.py`:

| Column | Type | Purpose |
|--------|------|---------|
| `subject_id` | int | Patient id |
| `hadm_id` | int | Admission id |
| `data_split` | str | "train" / "tuning" / "held_out" |
| `specialty_pool` | int (0-3) | Specialty index matching `SPECIALTY_NAMES` |
| `event_indicator` | int (0/1) | Readmission occurred within 30 days |
| `time_to_readmission` | float | Days to event (or horizon if censored) |
| `discharge_time` | datetime | Synthetic dates |
| `is_heart_failure` | bool | For the guideline policy |

## File Layout

```
ccpfs/
  demo_setup.py                 # the script
  docker-compose.demo.yml       # standalone compose for demo mode
  DEMO_MODE_PLAN.md             # this document
  data/
    processed/                  # REAL MIMIC-IV output (untouched by demo)
    demo/                       # NEW - demo output, gitignored
      cohort.parquet
      curves_test.npz           # curves_test, e_test, t_test
      models_info.json          # {synthetic: true, best_model: "gbm", ...}
      pipeline_results.json     # {synthetic: true, cohort_size, test_episodes, total_episodes, model_performance, scheduling_results}
      scheduling_results.npz    # {policy}_days, {policy}_cost, {policy}_status, policy_names
      motor_output/
        motor_curves.npz        # noisier curves (simulated domain shift)
        motor_result.json       # {synthetic: true, c_index, ibs, ...}
```

## Pipeline Inside `demo_setup.py`

1. **Build synthetic cohort** via `generate_synthetic_cohort()` (10K patients, Weibull curves).
2. **Assign specialties** proportionally to match MIMIC-IV's 55/20/14/11 mix, then shuffle.
3. **Assign 80/10/10 splits** via a fresh permutation.
4. **Slice the test split** (held_out) for scheduling.
5. **Run all policies** (uniform_d14, risk_bucket, guideline, unconstrained, uniform_d14_cap, guideline_cap, greedy_specialty, greedy_global, mincost_specialty, mincost_global; ILP rows aliased to mincost). Compute EBF/catch rate for each via `event_before_followup_rate`.
6. **Write every artefact** in the exact shape `api/dependencies.py` and `api/routers/results.py` expect.

## Policy Keys Produced

These match the `POLICY_META` list in `api/dependencies.py` exactly:

```
guideline, uniform_d14, risk_bucket, uniform_d14_cap, guideline_cap,
greedy_global, greedy_specialty, mincost_global, mincost_specialty, unconstrained
```

No ILP keys are produced because the API/dashboard never load them; the ILP solver is only used by `run_pipeline.py` on the real MIMIC-IV path.

## CLI

```
python demo_setup.py [--n-patients N] [--seed S] [--out-dir DIR] [--clean] [--force]
```

| Flag | Default | Effect |
|------|---------|--------|
| `--n-patients` | 10,000 | Total synthetic patients |
| `--seed` | `RANDOM_SEED` from `config.py` | Reproducibility |
| `--out-dir` | `data/demo` | Output directory |
| `--clean` | - | Delete `--out-dir` (safety-checked), then exit |
| `--force` | - | Allow writing into a directory containing non-synthetic data |

## Verification Checklist

After running the demo, confirm:

1. [ ] `data/demo/` exists and `data/processed/` is untouched.
2. [ ] `pipeline_results.json` has `"synthetic": true`.
3. [ ] `scheduling_results.npz` contains every `{policy}_days` key in `POLICY_META`.
4. [ ] `docker compose -f docker-compose.demo.yml up` starts both services healthy.
5. [ ] `GET /api/health` returns 200 OK with a positive patient count.
6. [ ] Overview dashboard shows a large cost reduction for MinCost (specialty) vs Uniform-14.
7. [ ] Patient Explorer renders S(t) curves for individual synthetic patients.
8. [ ] Interactive Scheduler responds to capacity sliders.
9. [ ] `python demo_setup.py --clean` removes `data/demo/`.
10. [ ] Running `python demo_setup.py --out-dir data/processed` **refuses** with an explicit error (if real data is present).

## README Changes

A "Try It Without MIMIC-IV (Demo Mode)" section is added near the top of `README.md`, above the Quick Start. It explains:

- The demo commands (`python demo_setup.py` then `docker compose -f docker-compose.demo.yml up`).
- That the output goes to `data/demo/` and never collides with `data/processed/`.
- The `--clean`, `--n-patients`, `--seed` flags.
- That `docker compose up` (no `-f`) is still the MIMIC-IV path.
- A link back to `Access Requirements` for reproducing the paper's results on real data.

## Success Criteria

A reviewer who has never seen this project clones the repo, runs two commands, and sees the full framework working with meaningful numbers within two minutes. A user with real MIMIC-IV data can run the demo without any risk of clobbering their pipeline output.
