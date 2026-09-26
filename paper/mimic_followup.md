# MIMIC-IV follow-up: confidence intervals, capacity-aware risk bucket, paper corrections

**Written:** 2026-09-26, after reviewing commit `572577e` (the capacity-fix re-run) on a machine without MIMIC-IV data.
**Purpose:** a self-contained list of what still has to be done on the MIMIC-IV machine, and the paper corrections that go with it. The background for the capacity fix itself is in `paper/capacity_fix_rerun.md`; you don't need it to follow this file.

Paper line numbers refer to `paper/research_paper.md` as committed in `572577e` (232 lines). The current text is quoted, so each spot can still be found after edits.

---

## 0. Checklist

- [ ] 1. Pull the review commits and confirm they're present (section 2).
- [ ] 2. Update the venv and run the tests: **30 should pass** (section 3).
- [ ] 3. Re-run `schedule,report` to add the capacity-aware risk bucket and the episode counts, and to check the 2026-09-24 numbers reproduce (section 4).
- [ ] 4. Re-run the cross-model script, which now checks its inputs and saves each model's schedule (section 5).
- [ ] 5. Run the bootstrap confidence intervals (section 6).
- [ ] 6. Optional: measure what calibration actually changes (section 7).
- [ ] 7. Apply the paper corrections that need no data: 3 items (section 8).
- [ ] 8. Add the intervals and the capacity-aware risk bucket to the paper (section 9).
- [ ] 9. Update the README results table (section 10).
- [ ] 10. Restart the dashboard/API (section 11).
- [ ] 11. Commit (section 12).

---

## 1. What the review found and why

| # | Finding | Where it's handled |
|---|---|---|
| R1 | Finding 5 (line 142) says the greedy heuristic is 38% above the optimum on the synthetic cohort because its trajectories are "steeper and more varied". Tested: the gap comes from the random daily noise the synthetic generator adds to every curve. Greedy ranks patients by their single largest daily drop in S(t), and noise decides which day that is. Without the noise the gap is 0.1–0.2%. | Section 8, item 1 |
| R2 | §III.C (line 76) says overestimating risk makes "too many patients compete for early slots", and that before calibration "the solver assigned nearly all patients to the first week". Under fixed capacity the second is impossible: at most 6,461 of 27,641 patients fit in days 1–7. The first contradicts §III.D (line 84): overestimating every patient's risk by the same factor is the same as scaling C_EVENT, which leaves the schedule unchanged. | Section 8, item 2; optional evidence in section 7 |
| R3 | Limitation 5 (line 179) says the catch rate "ranks the policies the same way" as expected cost. Among the four optimised policies it doesn't. By cost: MinCost (global) 999 < Greedy (global) 1,003 = MinCost (spec) 1,003 < Greedy (spec) 1,007. By catch rate: Greedy (global) 59.2 > MinCost (global) 59.1 > Greedy (spec) 59.0 > MinCost (spec) 58.9. | Section 8, item 3 |
| R4 | No confidence intervals. A catch rate over 5,677 readmissions has a standard error of about 0.65 points, so the 0.1–0.9-point differences in Findings 4–5 and Table III are within noise. | Sections 6 and 9 |
| R5 | The risk-bucket comparison isn't like-for-like. Risk bucket ignores capacity, but its 5% gain is set against the capacity-limited optimiser's 29% (contribution 3, Finding 2, conclusion). | Sections 4, 6 and 9 |

Code changes made in the review (30 tests pass; everything was verified on demo data on the review machine, and the cross-model script on synthetic stand-in models; see section 5):
- **`policy/baselines.py`:** new `risk_bucket_capacity_policy`. Same buckets as the risk bucket (30-day risk ≥ 30% → day 7, ≥ 15% → day 14, else day 30). Patients are placed highest-risk first, and anyone whose bucket day is full moves to the nearest free day in their pool. It shares one helper with the capacity-aware uniform and guideline baselines; their schedules were verified identical before and after.
- **`run_pipeline.py`, `demo_setup.py`, `api/dependencies.py`:** the new baseline is run and shown as "Risk bucket (capacity)" (`risk_bucket_cap`).
- **`run_pipeline.py`, `demo_setup.py`, `api/routers/results.py`:** `pipeline_results.json` gets explicit `total_episodes` and `test_episodes` keys. The dashboard's "total episodes" now shows 275,022 instead of 27,641. `cohort_size` is unchanged (test size) for existing readers.
- **`evaluation/cross_model_scheduling.py`:**
  - Stops on any input mismatch: each saved classical model must reproduce the first 200 rows of its `parallel_tmp` test curves (RSF was never checked before), MOTOR's aligned test rows must match the cohort's outcomes in order, and the re-calibrated GBM must equal `curves_test.npz`.
  - Refreshes cached calibrated curves when their inputs change.
  - Saves each model's schedule as `cross_model/{model}_days.npy`.
  - Run end to end on the review machine against synthetic stand-in models only; it has not yet seen the real MIMIC-IV files (section 5).
- **`evaluation/bootstrap_ci.py` (new):** paired bootstrap confidence intervals for Table II and Table III.
- **Dashboard:** the chart used the rounded readmission time to decide caught/missed, so 13.6 with follow-up on day 14 showed "Caught" while the metric says missed. It now uses the unrounded time, and the text no longer says the follow-up "would have occurred first" for same-day events.
- **`DEMO_MODE_PLAN.md`:** stale `scheduling_follow_up` paths and the `pipeline_results.json` key list fixed.
- **README:** test count 30; new baseline in the baselines table; the API table no longer claims "10 policies".

---

## 2. Get the review commits

The review changes were committed on the review machine as six commits on top of `572577e`:

1. `fix stale paths in DEMO_MODE_PLAN.md`
2. `use unrounded readmission time for caught/missed in explorer and chart`
3. `verify inputs, refresh stale caches and save schedules in cross-model script`
4. `add capacity-aware risk bucket baseline and explicit episode counts`
5. `add paired bootstrap confidence intervals for scheduling results`
6. `add MIMIC-IV follow-up guide and update re-run record`

```bash
git pull
git log --oneline -7
grep -n "def risk_bucket_capacity_policy" policy/baselines.py
grep -n "total_episodes" run_pipeline.py api/routers/results.py
ls evaluation/bootstrap_ci.py
```

---

## 3. Environment

The existing `.venv` on this machine has PuLP 3.3.0, which is why only 22 of 28 tests passed last time. The code needs 3.3.2 (`prob.add_variable`) and the `cbc` extra.

```bash
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q                          # expect: 30 passed
```

---

## 4. Re-run schedule and report (about 1 h 45 min, ~6.2 GB peak)

This adds `risk_bucket_cap` and the episode counts to `pipeline_results.json` and `scheduling_results.npz`. Every other policy's numbers must come out **identical** to the 2026-09-24 run, since nothing else changed. That makes the run a reproducibility check too.

```bash
mkdir -p data/processed/pre_followup
cp data/processed/{pipeline_results.json,scheduling_results.npz} data/processed/pre_followup/
python run_pipeline.py --step schedule,report 2>&1 | tee data/processed/rerun_followup.log
```

Check the log for `Risk bucket (capacity): cost=… Feasible (capacity-aware)`.

Then confirm nothing else moved:

```python
import json
old = json.load(open("data/processed/pre_followup/pipeline_results.json"))["scheduling_results"]
new = json.load(open("data/processed/pipeline_results.json"))
print("episodes:", new.get("total_episodes"), new.get("test_episodes"))   # expect 275022 27641
for k, v in old.items():
    a, b = v["total_expected_cost"], new["scheduling_results"][k]["total_expected_cost"]
    ca, cb = v.get("ebf", {}).get("catch_rate"), new["scheduling_results"][k].get("ebf", {}).get("catch_rate")
    print(f"{k:20s} {'same' if (a == b and ca == cb) else 'CHANGED'}")
rb = new["scheduling_results"]["risk_bucket_cap"]
print("risk_bucket_cap:", rb["status"], round(rb["total_expected_cost"] / new["test_episodes"]), rb["ebf"]["catch_rate"])
```

If any existing policy shows `CHANGED`, stop and investigate before touching the paper.

**If time is short:** skip this re-run. `evaluation/bootstrap_ci.py` (section 6) computes `risk_bucket_cap` itself when it's missing from `scheduling_results.npz`, using the same function and capacity, so the paper numbers are the same. The dashboard won't show the new row until the pipeline is re-run. For the episode label alone, `python run_pipeline.py --step report` takes seconds.

---

## 5. Re-run the cross-model script (Table III)

The script now verifies its inputs and saves each model's schedule, which the Table III intervals need. It recomputes the cached calibrated curves once, because the old caches have no input fingerprint. Each model's solve takes about as long as the specialty MinCost solve (~16 min). Models can run in parallel, but each process loads the full feature matrix and builds a ~1.8 GB cost matrix for the general-medicine pool, so allow several GB per process. If RAM is tight, run them one after another:

```bash
for m in cox gbm rsf motor; do
  PYTHONPATH=. python evaluation/cross_model_scheduling.py --models $m > data/processed/cross_model_$m.log 2>&1 &
done
wait
# Each process merges only the models finished before it; rebuild the merged table once all are done
python -c "import json; from pathlib import Path; d = Path('data/processed/cross_model'); \
json.dump({m: json.load(open(d / f'{m}_result.json')) for m in ['cox', 'gbm', 'rsf', 'motor']}, \
open(d / 'cross_model_results.json', 'w'), indent=2)"
cat data/processed/cross_model/cross_model_results.json
```

Expected:
- Each log shows `saved model reproduces parallel_tmp curves` (Cox, GBM, RSF) and, for GBM, `calibrated curves match curves_test.npz`.
- The results equal Table III exactly (Cox 1,011 / 58.3%, GBM 1,003 / 58.9%, RSF 1,011 / 58.0%, MOTOR 1,058 / 54.7%).
- `data/processed/cross_model/{cox,gbm,rsf,motor}_days.npy` exist.

**If a check fails** (most likely RSF, which was never checked): the saved model and the `parallel_tmp` curves come from different training runs. In that case Table III's row for that model was calibrated on one model and applied to another. Options:
- Drop that row from Table III (and the claims about that model in the text around it).
- Or keep the row using the saved model for both validation and test curves. That needs a code change: extract its test curves from the saved model instead of `parallel_tmp`, and recompute its C-index for Table I so both tables use the same model. Ask for that change rather than improvising it.

**Do not re-run `parallel_train.py` to fix this.** It has no per-model option: it retrains all three models and then overwrites `data/processed/curves_test.npz`, the calibrated GBM curves behind Table II.

**Tested on synthetic stand-ins, not on the real files.** On the review machine the script was run end to end against a 3,000-patient synthetic cohort, with GBM, Cox and RSF models trained by the repo's own training code and a MOTOR-style scaler → PCA → GBM stack:
- **Consistent inputs:** all checks passed and all four models produced results and `{model}_days.npy`. The GBM schedule was identical to the pipeline-style MinCost schedule, and `bootstrap_ci.py` picked up the saved schedules.
- **Cache:** a second run reused the cache, and an old-format cache without a fingerprint was recomputed.
- **Deliberately broken inputs:** each stopped the run with its error message.
  - RSF retrained with a different seed: max difference 1.2e-01.
  - MOTOR test rows shuffled out of cohort order.
  - Main GBM curves altered by 0.1%: max difference 1.0e-03.

What remains untested is the real MIMIC-IV files: their size, their library versions, and whether the saved models really match `parallel_tmp`.

A check can therefore still fail here for two different reasons. The error message prints the maximum difference, which tells them apart:
- **Real mismatch:** differences of about 1e-2 or larger, because curves from a different training run differ in their predictions, not just in rounding. The retrained RSF above gave 1.2e-01. Handle it as described above, and don't change the tolerance.
- **Numerical noise:** a tiny difference, below about 1e-4, with every other check passing. This can come from a library version that changed since the models were trained. Consistent inputs gave at most 4.4e-16 in the synthetic test. Raise the tolerance to just above the reported difference (`REPRO_ATOL` near the top of the script, or the `1e-9` in the GBM check in `main()`), re-run, and confirm the results still equal Table III exactly. Record the tolerance change and the reported difference in the results section of this file.

Anything between the two ranges, or any other error (crash, wrong shape, missing file), is unexpected: stop and investigate rather than working around it.

---

## 6. Bootstrap confidence intervals (seconds)

```bash
PYTHONPATH=. python evaluation/bootstrap_ci.py            # 2,000 paired replicates, writes data/processed/bootstrap_ci.json
```

It prints:
- Table II with a 95% interval for each policy's cost and catch rate.
- Paired differences for the comparisons the paper makes, as cost % and catch-rate points with 95% intervals.
- Table III catch rates with intervals, and each model vs GBM.

How the intervals work: every replicate resamples the 27,641 test patients with replacement and scores every policy's **fixed** schedule on the same resample, so comparisons are paired. The intervals cover test-set sampling only, not retraining the models or re-solving the schedule.

How to read them:
- **Interval excludes 0:** a difference you can report.
- **Interval includes 0:** say the policies are indistinguishable on this test set; don't rank them.
- **Specialty vs global MinCost:** the cost interval can include negative values even though global MinCost is a lower bound. The bound holds for the full test set, not for every resample of fixed schedules. Report the full-sample difference and note that its interval includes 0 if it does.

For reference, the same script on the synthetic demo (1,000 patients, 198 readmissions) gives catch-rate intervals about ±6–7 points wide. On MIMIC-IV, with 5,677 readmissions, expect roughly ±1.3 points for a single policy, and narrower for paired differences.

---

## 7. Optional: what calibration actually changes (~16 min, one solve)

This gives §III.C (section 8, item 2) real evidence in place of the removed anecdote. It schedules with the **uncalibrated** GBM curves under the same capacity, then scores that schedule with the calibrated curves (common yardstick) and with observed readmissions. Save it as `calibration_effect.py` in the repo root and run `PYTHONPATH=. python calibration_effect.py`:

```python
import numpy as np, polars as pl
from config import C_EVENT, C_VISIT, HORIZON_DAYS
from evaluation.metrics import event_before_followup_rate
from policy.mincost_solver import schedule_mincost_specialty
from policy.specialty_scheduler import proportional_specialty_capacity

P = "data/processed"
raw = np.load(f"{P}/parallel_tmp/gbm_curves.npz")["curves"]
cal = np.load(f"{P}/curves_test.npz")
curves, e, t = cal["curves_test"], cal["e_test"], cal["t_test"]
pools = pl.read_parquet(f"{P}/cohort.parquet").filter(pl.col("data_split") == "held_out")["specialty_pool"].to_numpy()
cap = proportional_specialty_capacity(pools, HORIZON_DAYS)

rows = np.arange(len(curves))
a = schedule_mincost_specialty(raw, pools, capacity_per_specialty_day=cap)["assignments"]
schedules = {
    "uncalibrated": np.array([a[i] for i in rows]),
    "calibrated": np.load(f"{P}/scheduling_results.npz")["mincost_specialty_days"],  # already solved by the pipeline
}
print(f"mean 30-day risk: raw {1 - raw[:, 30].mean():.1%}, calibrated {1 - curves[:, 30].mean():.1%}")
for name, d in schedules.items():
    cost = np.mean(C_EVENT * (1 - curves[rows, d]) + C_VISIT)   # scored with calibrated curves
    catch = event_before_followup_rate({int(i): int(x) for i, x in enumerate(d)}, t, e)["catch_rate"]
    changed = (d != schedules["calibrated"]).mean()
    print(f"{name:13s} EUR {cost:,.0f} (calibrated yardstick)  catch {catch:.1%}  "
          f"patients on a different day than the calibrated schedule: {changed:.1%}")
```

Report the catch-rate difference (and the cost under the calibrated yardstick) in §III.C. If the difference is small, say so plainly: it would mean GBM's miscalibration shifted risks for most patients in a similar way, which barely changes who gets the early slots. Don't present a small difference as evidence that calibration is critical.

---

## 8. Paper corrections that need no data

### Item 1 (R1): Finding 5, line 142

Current: *"On these calibrated curves the two orderings nearly coincide, but this is not guaranteed: on the synthetic Weibull cohort distributed with the code, whose risk trajectories are steeper and more varied, the same heuristic is 38% above the optimum (EUR 1,077 vs EUR 783 per patient)."*

Proposed:

> On these calibrated curves the two orderings nearly coincide, but this is not guaranteed. Urgency is read from a single day's drop in S(t), so it is sensitive to day-to-day jitter in the curves: on the synthetic Weibull cohort distributed with the code, which adds random daily noise to every curve, the heuristic is 35-38% above the optimum across random seeds (EUR 1,077 vs EUR 783 per patient in the demo), and 0.1-0.2% when that noise is removed.

Evidence, reproducible without MIMIC-IV (1,000-patient samples as in the demo, proportional capacity, three seeds):

| Daily noise added by the generator | Greedy above MinCost (specialty) |
|---|---|
| 0.02 (default) | +35.6%, +34.9%, +36.4% |
| 0.01 | +24.7%, +23.0%, +23.4% |
| 0.005 | +14.4%, +13.2%, +13.1% |
| 0 | +0.1%, +0.2%, +0.2% |

```bash
PYTHONPATH=. python - <<'EOF'
import numpy as np
from config import HORIZON_DAYS
from evaluation.synthetic import generate_synthetic_cohort
from policy.specialty_scheduler import proportional_specialty_capacity, schedule_greedy_specialty
from policy.mincost_solver import schedule_mincost_specialty

for noise in (0.02, 0.01, 0.005, 0.0):
    gaps = []
    for seed in (42, 7, 1):
        g = generate_synthetic_cohort(n_patients=10_000, seed=seed, high_risk_frac=0.10,
                                      med_risk_frac=0.40, noise_std=noise)
        c = g["survival_curves"]
        rng = np.random.default_rng(seed)
        pools = rng.choice(4, size=len(c), p=[.197, .137, .115, .551])
        idx = rng.permutation(len(c))[:1000]
        c, pools = c[idx], pools[idx]
        cap = proportional_specialty_capacity(pools, HORIZON_DAYS)
        gr = schedule_greedy_specialty(c, pools, capacity_per_specialty_day=cap)["total_expected_cost"]
        mc = schedule_mincost_specialty(c, pools, capacity_per_specialty_day=cap)["total_expected_cost"]
        gaps.append((gr - mc) / mc * 100)
    print(f"noise_std={noise:<6} greedy above MinCost: " + ", ".join(f"{x:+.1f}%" for x in gaps))
EOF
```

The €1,077 vs €783 figure (37.5%) is from `python demo_setup.py` (seed 42) and is unchanged by the review. The table above samples its 1,000 patients differently from the demo, which is why its default-noise gaps (35–36%) are slightly lower; hence "35-38%" in the proposed text.

### Item 2 (R2): §III.C, line 76

Current: *"Since survival curves feed directly into the cost function, calibration is critical. If a model systematically overestimates risk, too many patients compete for early slots; if it underestimates, high-risk patients are deferred when they should not be. We discovered this concretely during development: before calibration, the solver assigned nearly all patients to the first week because raw GBM probabilities were inflated."*

Proposed (keep the following sentences from "We apply isotonic regression calibration…" unchanged):

> Since survival curves feed directly into the cost function, calibration matters, though not in the way a uniform bias would suggest. Under fixed capacity, a model that overestimates every patient's risk by the same factor produces exactly the same schedule, because scaling all risks is equivalent to scaling C_EVENT (Section III.D). What changes the schedule is miscalibration that differs between patients or between days - for example, risk inflated for some patient groups but not others, or placed too early in the 30-day window - because it distorts the relative marginal benefits the optimiser compares. Calibration also determines whether the reported expected costs are meaningful in absolute terms.

If you ran section 7, add one sentence with its result, e.g. *"On our test cohort, scheduling with uncalibrated GBM curves changes the catch rate from X% to Y%."*

### Item 3 (R3): Limitation 5, line 179

Current: *"…which favours the optimised policies by construction; the catch rate, computed from observed readmissions, is the outcome-based check, and it ranks the policies the same way."*

Proposed:

> …which favours the optimised policies by construction; the catch rate, computed from observed readmissions, is the outcome-based check. It separates the optimised policies from every baseline in the same direction as expected cost, but it does not distinguish among the four optimised policies, whose catch rates lie within 0.3 percentage points of each other, inside the bootstrap intervals.

Adjust the last clause to whatever section 6 shows.

---

## 9. Paper additions from sections 4–6

### 9.1 Method sentence (§IV intro, after the catch-rate definition, line 105)

> We report 95% confidence intervals from a paired percentile bootstrap over test patients (2,000 replicates): each replicate resamples the 27,641 test episodes with replacement and re-scores every policy's fixed schedule on the same resample. The intervals capture test-set sampling variability, not variability from re-training the models or re-solving the schedule.

### 9.2 Where the intervals go

| Location | Current | Add |
|---|---|---|
| Table II (lines 111–122) | cost, vs Uniform-14, catch | 95% CIs for cost and catch rate, from the Table II part of the output. Either as extra columns or inline as "catch [lower, upper]". |
| Abstract (line 5), contribution 2 (line 20), conclusion (line 193) | "29%", "59% … versus 38%" | The CI of "MinCost (spec) vs Uniform-14 (capacity)" for cost % and catch points, if the venue allows. Otherwise at least in Finding 1. |
| Finding 1 (line 134) | "reduces cost by 28.9% … catching 58.9% … versus 37.7%" | Both CIs from the same comparison |
| Finding 4 (line 140) | "raises expected cost by 0.4% (EUR 1,003 vs EUR 999) and lowers the catch rate from 59.1% to 58.9%" | Keep the 0.4% cost figure, but describe the catch-rate change as within noise if its interval includes 0 ("MinCost (spec) vs MinCost (global)") |
| Finding 5 (line 142) | "0.4% above the exact optimum … catch rates within 0.1 percentage points" | CIs for "Greedy (spec) vs MinCost (spec)" and "Greedy (global) vs MinCost (global)" |
| Table III (lines 165–172) and text (line 174) | catch rates 58.3 / 58.9 / 58.0 / 54.7, "within 0.9 percentage points", "4.2 percentage points" | Catch-rate CIs and "vs GBM" intervals from the Table III part of the output. Say the classical models are indistinguishable if their intervals vs GBM include 0; keep the MOTOR claim only if its interval excludes 0. |

### 9.3 Capacity-aware risk bucket (R5)

**Table II (after "Guideline (capacity)", line 117):** add

`| Risk bucket (capacity) | Yes | [cost] | [vs U14 %] | [catch] |`

Cost and catch rate are in the `risk_bucket_cap` row of the bootstrap's Table II output. "vs U14" is the comparison line "Risk bucket (capacity) vs Uniform-14". The same numbers are in `data/processed/bootstrap_ci.json`.

**Line 124:** current *"The capacity-aware baselines start from their intended day (day 14, or the guideline day) and move each patient to the nearest day with a free slot in their own specialty pool."* Proposed:

> The capacity-aware baselines start from their intended day (day 14, the guideline day, or the risk-bucket day) and move each patient to the nearest day with a free slot in their own specialty pool; the capacity-aware risk bucket places the highest-risk patients first, so it is lower-risk patients who move when a bucket day is full.

**Finding 2 (line 136), contribution 3 (line 21), conclusion (line 193).** The current claim compares risk bucket against *uncapacitated* uniform-14 (5.1%). The like-for-like comparison under identical capacity comes from two bootstrap lines:
- "Risk bucket (capacity) vs Uniform-14 (capacity)": how much risk awareness alone gains under capacity.
- "MinCost (spec) vs Risk bucket (capacity)": how much patient-level optimisation adds on top.

Decide by the results:
- **Risk bucket (capacity) gains little over Uniform-14 (capacity), and MinCost is clearly better than it** (intervals exclude 0), which is the expected case. Keep the thesis and restate it with the capacity-aware numbers. For example, for Finding 2: *"Under the same capacity, the risk bucket reduces cost by [a]% [CI] relative to capacity-aware uniform scheduling, while MinCost reduces it by a further [b]% [CI] relative to the risk bucket and catches [c] points [CI] more readmissions."* Contribution 3 becomes *"Evidence that coarse risk stratification, even given the same capacity, reduces cost by only [a]% relative to capacity-aware uniform scheduling, against [29]% for patient-level optimisation…"*. Keep the uncapacitated 5.1% as a secondary sentence, or drop it.
- **Risk bucket (capacity) captures a large share of MinCost's gain:** the thesis "patient-level optimisation, not risk awareness, drives the gains" no longer holds as stated. Rewrite contribution 3, Finding 2 and the conclusion around the size of the remaining gap instead of the 5% figure.

---

## 10. README (results table, line ~305)

After the `Uniform-14 (capacity)` row, add

`| Risk bucket (capacity) | Yes | [cost] | [catch] |`

using the same values as Table II. Optionally mention the confidence intervals in the paragraph under the table.

---

## 11. Dashboard

The API loads `data/processed/` at startup: `docker compose up --build`. After section 4, the overview shows the "Risk bucket (capacity)" row and total episodes = 275,022.

---

## 12. Commit

Short one-line messages, for example:

```bash
git add paper/research_paper.md README.md
git commit -m "add bootstrap intervals and capacity-aware risk bucket to paper, correct greedy-gap, calibration and limitation text"
git add paper/mimic_followup.md
git commit -m "record MIMIC-IV follow-up results"
```

Before committing, record the results under a new heading at the end of this file: the section 4 comparison, the section 5 checks, the key bootstrap lines and any section 7 result. Keep the output files in `data/processed/`: `rerun_followup.log`, `bootstrap_ci.json` and `cross_model/*`.
