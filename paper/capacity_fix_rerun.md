# Capacity allocation fix: re-run guide and paper changes

**Written:** 2026-09-24, from a review session on a machine without MIMIC-IV data.
**Purpose:** a self-contained handover for re-running the pipeline on the machine that has the data, and for updating the paper and README. It records everything found in the review and everything that still needs doing.

**Status when written:**
- The code fixes are made and tested: 28 tests pass, and the demo and API were checked. They are committed as four commits on top of `71e8e1b` (section 3.2).
- The paper (`paper/research_paper.md`) has **not** been edited.
- No MIMIC-IV number has been regenerated. Every result that depends on capacity must come from the re-run in section 5.

Line numbers for the paper refer to `paper/research_paper.md` as it was on 2026-09-24 (232 lines). Line numbers for the README refer to `README.md` after the commits in section 3.2. Quoted text is included so each spot can still be found after edits.

---

## 0. Start here: checklist

Work through these in order. Each step points to the section with the details.

- [ ] 1. `git pull` and confirm the fix is present (section 3.3).
- [ ] 2. Set up the environment and run the tests: 28 should pass (section 5.1).
- [ ] 3. Point the repo at the MIMIC-IV data (section 5.2).
- [ ] 4. Confirm the diagnosis on the **current, pre-fix** results (section 4).
- [ ] 5. Back up the current results (section 5.3).
- [ ] 6. Re-run `python run_pipeline.py --step schedule,report` and check the statuses (section 5.4).
- [ ] 7. Extract every number the paper needs (section 5.5). Optionally measure timings (section 5.6).
- [ ] 8. Fix the paper text that is wrong regardless of the re-run: 13 items (section 6.1).
- [ ] 9. Replace the capacity-dependent numbers in the paper (section 6.2).
- [ ] 10. Update the README (section 6.3).
- [ ] 11. Restart the dashboard/API on the new results (section 5.7).
- [ ] 12. Decide on Table III: regenerate it, which needs code changes (section 7), or remove it.
- [ ] 13. Optionally fix the remaining code issues (section 8.3).
- [ ] 14. Commit the paper and README updates.

---

## 1. Background

CCPFS turns patient-level survival curves S(t) (30-day readmission risk) into follow-up appointment days under per-specialty daily clinic capacity. Scheduling patient *i* on day *d* costs `C_EVENT·(1−S_i(d)) + C_VISIT`, with C_EVENT = €10,000 and C_VISIT = €150. Each patient belongs to one of four specialty pools: 0 cardiology, 1 neurology, 2 surgery, 3 general medicine. Slots can't move between pools.

Files relevant to this issue:

| File | Role |
|---|---|
| `run_pipeline.py` | Pipeline. `step_schedule` sets capacity and runs every policy. `step_report` writes `data/processed/pipeline_results.json`. |
| `policy/specialty_scheduler.py` | `proportional_specialty_capacity` (new), `schedule_greedy_specialty`, `schedule_ilp_specialty` |
| `policy/mincost_solver.py` | Exact min-cost assignment. `schedule_mincost_specialty` switches to `_solve_mincost_by_pool` above 15,000 patients. `schedule_mincost_global` |
| `policy/greedy_scheduler.py` | `schedule_greedy` (global greedy) |
| `policy/baselines.py` | Uniform, risk bucket, guideline, unconstrained, and the capacity-aware uniform/guideline baselines |
| `evaluation/metrics.py` | `event_before_followup_rate` (catch rate) |
| `config.py` | Costs, `HORIZON_DAYS = 30`, `DEFAULT_SPECIALTY_CAPACITY` (15/10/15/25, used by the API/dashboard as defaults), `PROCESSED_DIR` |
| `demo_setup.py` | Synthetic demo that writes to `data/demo/` |
| `api/`, `dashboard/` | FastAPI + Dash; the API loads `data/processed/` (or `data/demo/`) at startup |

---

## 2. The issue

### 2.1 What the paper says

§III.D (line 91): *"We set C_k proportionally to each specialty's share of the cohort - cardiology 15, neurology 10, surgery 15, general medicine 25 slots per day - so that total 30-day capacity exactly equals the test cohort size (27,641 slots for 27,641 patients). This ensures that capacity is binding but not infeasible: every patient can be scheduled…"*

### 2.2 What the code did (before the fix)

`run_pipeline.py::step_schedule` scaled the default 15/10/15/25 split by a single factor instead: `ceil(n / 30) / 65 = 922 / 65 = 14.18` for n = 27,641.

| | Cardiology | Neurology | Surgery | General medicine | Total |
|---|---:|---:|---:|---:|---:|
| Default slots/day | 15 | 10 | 15 | 25 | 65 |
| Scaled ×14.18, rounded up | 213 | 142 | 213 | 355 | 923 |
| 30-day slots | 6,390 | 4,260 | 6,390 | 10,650 | 27,690 |
| Share of slots | 23% | 15% | 23% | **38%** | |
| Share of patients (full cohort) | 19.7% | 13.7% | 11.5% | **55.1%** | |

General medicine gets 38% of the slots but has about 55% of the patients, so that pool has roughly 4,450 more patients than slots. Surgery gets about twice the slots it needs.

The Table II footnote (line 124) reports **4,447 patients (16.1%) who could not be scheduled**. That is exactly the general-medicine shortfall if the test set has 15,097 general-medicine episodes (54.6%), since 15,097 − 10,650 = 4,447. Section 4 confirms this from the saved results.

### 2.3 How each policy handled the overflow patients

- **MinCost (specialty):** above 15,000 patients the solver decomposes by pool (`_solve_mincost_by_pool`). It placed every overflow patient on their lowest-cost day, which is always **day 1** because risk only rises over time, and it never checked capacity. About 4,447 general-medicine patients got day-1 appointments that did not exist (only 355 day-1 slots), and the status still read "Optimal (overflow=…)". The dashboard hard-coded this policy as feasible. These free early slots feed directly into the reported €759 / 71.0%.
- **Greedy (specialty):** overflow patients were also placed outside capacity. Before commit `dafe86a` (2026-04-11) they went to their lowest-cost day (day 1), with status `Feasible (overflow=N)`. From `dafe86a` on they were spread across days, with status `Overflow (N patients beyond capacity)`. Which applies depends on when the paper numbers were produced.
- **Uniform-14 / Guideline (capacity):** reported as infeasible (the Table II footnote).
- **Global policies:** one shared pool has no mismatch, so these got no free capacity. This is why the specialty solver appeared to beat the global one by 24% (Finding 4). With identical total capacity, that's impossible for a correct solver (section 6.1 item 3).

### 2.4 Size of the effect (synthetic reproduction, NOT MIMIC-IV)

This used a Weibull cohort of N = 5,000 with the MIMIC specialty mix and the pipeline's old scaling rule, calling the per-pool solver directly (the code path the 27,641-patient run uses). Two seeds were run:

| Per-pool solver | Cost/patient | vs uniform-14 | Catch rate | Patients on day 1 |
|---|---:|---:|---:|---:|
| Old (overflow → day 1) | €637–644 | −44 to −45% | 78–80% | ~920 (capacity ~170) |
| New (capacity respected) | €860–868 | −25% | ~69% | ~195 |

In this reproduction, phantom capacity accounts for almost half of the gain over uniform-14. The old result (−45%) is close to the paper's −46.6%. **The real MIMIC-IV effect is only known after the re-run.** The script is in section 9.2.

---

## 3. Code changes

### 3.1 What changed and why

**The capacity fix (changes results)**

| File | Change |
|---|---|
| `policy/specialty_scheduler.py` | New `proportional_specialty_capacity(specialty_pools, horizon)`: pool *k* gets `ceil(n_k / horizon)` slots per day. Every pool can then schedule all its patients with fewer than 30 spare slots. That is binding but feasible, which is the design §III.D describes. |
| `run_pipeline.py` | `step_schedule` uses `proportional_specialty_capacity` for every capacity-aware policy. Global capacity is the sum of the pool capacities, so global and specialty policies get the same total slots. It prints each pool's patient count, slots per day and 30-day slots. The unused `DEFAULT_SPECIALTY_CAPACITY` import was removed. |
| `policy/mincost_solver.py` | Per-pool path: if a pool still overflows, the extra patients go to the least-loaded day instead of day 1. The `n > 15000` check now runs *before* the full cost matrix is built, which avoids a ~6 GB allocation at 27,641 patients. `schedule_mincost_global` copies `capacity_per_day` instead of modifying the caller's array. |
| `demo_setup.py` | The demo uses the same proportional capacity for every capacity-aware policy, so it mirrors the paper's setup. |
| `api/dependencies.py`, `api/routers/results.py` | The dashboard's "feasible" flag is derived from each policy's saved status: any status containing "infeasible" or "overflow" counts as infeasible. Before, it was hard-coded per policy. |
| `tests/test_policy.py` | Four new tests (class `TestSpecialtyCapacity`): capacity covers every pool, no overflow under proportional capacity, overflow spread evenly on the per-pool path, capacity array not modified. 24 → 28 tests. |

**Dependencies (no effect on results)**

| File | Change |
|---|---|
| `requirements.txt` | Added `polars>=1.27` (the demo crashed without it). `PuLP>=3.3` → `PuLP[cbc]>=3.3.2,<4`. |
| `api/requirements.txt` | `PuLP==3.3.0` → `PuLP==3.3.2`, needed for `prob.add_variable`. |
| `policy/ilp_scheduler.py`, `policy/specialty_scheduler.py` | Deprecated `PULP_CBC_CMD` replaced by `cbc_solver()`: `COIN_CMD` with the CBC binary from the `cbcbox` package that `PuLP[cbc]` installs. `cbcbox` is imported lazily, so the API image, which never runs the ILP, doesn't need it. Deprecated `LpVariable(...)` replaced by `prob.add_variable(...)`. This removed ~18K deprecation warnings per test run. |

**Demo, dashboard and docs (no effect on results)**

| File | Change |
|---|---|
| `demo_setup.py` | Risk mix set to 10% high / 40% medium / 50% low (`HIGH_RISK_FRAC`, `MED_RISK_FRAC`), giving a 20.5% event rate (was 26.9%; the README claims ~20%). Docstring `cd scheduling_follow_up` → `cd ccpfs`. |
| `README.md` | `cd ccpfs/scheduling_follow_up` → `cd ccpfs`; `cd scheduling_follow_up` → `cd ccpfs` (3 places); layout root `scheduling_follow_up/` → `ccpfs/`; removed the missing `DEMO_MODE_PLAN.md` entry; documented demand resampling. |
| `api/routers/scheduling.py` | `/api/schedule` resamples patients with replacement when demand exceeds a pool's size, instead of silently capping (the demo sliders stopped having any effect). Handles zero demand. |
| `dashboard/pages/patient_explorer.py` | The patient range label and input bounds come from the API's patient count. Before: hard-coded "1 to 27,641" (the inputs are 0-based) and a max of 27,640. |
| `dashboard/pages/interactive_scheduler.py` | "uses real patient risk curves from MIMIC-IV" → samples from the loaded test cohort (MIMIC-IV or synthetic demo). |
| `run_pipeline.py` | Comment "Ensure scheduling_follow_up is on the path" → "Ensure project root is on the path". |

### 3.2 Commits

Run on the review machine from the repo root. Each code commit was checked to pass the tests on its own.

```bash
git add requirements.txt api/requirements.txt policy/ilp_scheduler.py
git commit -m "add polars and migrate ILP solver to COIN_CMD with pulp[cbc]"

git add policy/specialty_scheduler.py policy/mincost_solver.py run_pipeline.py demo_setup.py api/dependencies.py api/routers/results.py tests/test_policy.py
git commit -m "allocate specialty capacity proportionally to pool size"

git add README.md api/routers/scheduling.py dashboard/pages/interactive_scheduler.py dashboard/pages/patient_explorer.py
git commit -m "fix README paths, resample demand beyond pool size, cohort-agnostic dashboard text"

git add paper/research_paper.md paper/capacity_fix_rerun.md
git commit -m "add research paper and capacity fix re-run guide"

git push
```

The second commit is the capacity fix that changes the scheduling results. The paper is committed unchanged; sections 6.1–6.2 refer to its line numbers as committed.

### 3.3 Confirm the fix is present on the other machine

```bash
git log --oneline -5                                                    # the four commits above
grep -n "def proportional_specialty_capacity" policy/specialty_scheduler.py
grep -n "proportional_specialty_capacity(specialty_pools" run_pipeline.py
grep -n "PULP_CBC_CMD\|LpVariable(" policy/*.py                         # should print nothing
```

---

## 4. Before re-running: confirm the diagnosis on the current results

Run this from the repo root with `PYTHONPATH=.`, **before** the re-run overwrites `pipeline_results.json` and `scheduling_results.npz`:

```python
import json
import numpy as np
import polars as pl

r = json.load(open("data/processed/pipeline_results.json"))["scheduling_results"]
for k in ["mincost_specialty", "greedy_specialty", "mincost_global", "uniform_d14_cap", "guideline_cap"]:
    print(f"{k:20s} {r[k]['status']}")

test = pl.read_parquet("data/processed/cohort.parquet").filter(pl.col("data_split") == "held_out")
pools = test["specialty_pool"].to_numpy()
print("patients per pool (cardio, neuro, surgery, gen med):", np.bincount(pools, minlength=4).tolist())

days = np.load("data/processed/scheduling_results.npz")["mincost_specialty_days"]
print("general medicine patients on day 1:", int(((pools == 3) & (days == 1)).sum()), "(355 slots)")
```

Expected if the diagnosis is right:
- `mincost_specialty` shows `Optimal (overflow=4447)`.
- `greedy_specialty` shows `Feasible (overflow=4447)` (run before `dafe86a`) or `Overflow (4447 patients beyond capacity)` (run after it).
- The capacity baselines show `Infeasible (4447 overflow assignments)`.
- There are about 15,097 general-medicine patients.
- About 4,800 general-medicine patients are on day 1 (355 in capacity plus about 4,447 overflow).

If the output doesn't match, stop and re-check before changing the paper: the diagnosis would need revisiting.

---

## 5. Re-running the pipeline

Only the `schedule` and `report` steps depend on capacity. They reload the saved calibrated GBM curves (`curves_test.npz`) and `models_info.json`, so **no retraining or recalibration is needed**.

### 5.1 Environment

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, scipy, polars, PuLP[cbc]: enough for schedule + report
pytest tests/ -q                         # expect: 28 passed
```

### 5.2 Point the repo at the data

`config.py` hard-codes `PROCESSED_DIR = <repo>/data/processed`; there is no environment variable override. If the data lives elsewhere, symlink it (`data/` is gitignored):

```bash
ln -s /path/to/your/data data            # needs processed/{cohort.parquet,curves_test.npz,models_info.json}
```

### 5.3 Back up the current results

The re-run overwrites two files. Keep the old ones for the before/after comparison and for section 4:

```bash
mkdir -p data/processed/pre_capacity_fix
cp data/processed/{pipeline_results.json,scheduling_results.npz} data/processed/pre_capacity_fix/
```

### 5.4 Run

```bash
python run_pipeline.py --step schedule,report 2>&1 | tee data/processed/rerun_capacity_fix.log
```

Check in the log:
- A block headed `Capacity (proportional to pool size)` lists each pool's patient count, slots per day and 30-day slots. Each pool's 30-day slots must be at least its patient count.
- `greedy_specialty` → `Feasible`; `mincost_specialty` → `Optimal` with **no** `overflow=`; `uniform_d14_cap` and `guideline_cap` → `Feasible (capacity-aware)`.

Resources: expect the run to need **≥ 16 GB RAM** and several tens of minutes.
- `schedule_mincost_global` builds one dense ~27.6K × 27.7K float64 matrix (~6.1 GB) before solving. The fix doesn't change this.
- The exact per-pool solve for general medicine (~15.1K patients) builds a ~1.9 GB matrix and is slow. Measured on the review machine with synthetic curves at 27,641 patients (15 GB RAM): 55 s for a 5.4K pool, 18 s for 3.6K, 16 s for 3.3K. General medicine (15.3K) was still running after more than 11 minutes when this was written. Cubic scaling from the 5.4K pool suggests about 20 minutes.
- Greedy (specialty): 123 ms.

### 5.5 Extract every number the paper needs

Save as `paper_numbers.py` in the repo root and run `PYTHONPATH=. python paper_numbers.py`:

```python
import json
import math

import numpy as np
import polars as pl

P = "data/processed"
INCLUDE_VISIT = True  # False if the paper defines cost without the €150 visit (section 6.1 item 4)
VISIT = 150

r = json.load(open(f"{P}/pipeline_results.json"))
n, s = r["cohort_size"], r["scheduling_results"]


def avg(k):
    a = s[k]["total_expected_cost"] / n  # total_expected_cost includes the €150 visit per patient
    return a if INCLUDE_VISIT else a - VISIT


def catch(k):
    return s[k].get("ebf", {}).get("catch_rate", float("nan")) * 100


def pct(a, b):
    return (a - b) / b * 100


# Capacity (for §III.D and §V.D wording)
test = pl.read_parquet(f"{P}/cohort.parquet").filter(pl.col("data_split") == "held_out")
counts = np.bincount(test["specialty_pool"].to_numpy(), minlength=4)
caps = [math.ceil(c / 30) for c in counts]
print("Capacity:")
for name, c, k in zip(["cardiology", "neurology", "surgery", "general_medicine"], counts, caps):
    print(f"  {name:17s} {c:6,d} patients  {k:4d}/day  {k * 30:6,d} slots  spare {k * 30 - c}")
print(f"  total {sum(caps)}/day, {sum(caps) * 30:,} slots for {n:,} patients\n")

# Table II
rows = ["guideline", "uniform_d14", "risk_bucket", "uniform_d14_cap", "guideline_cap",
        "greedy_global", "greedy_specialty", "mincost_global", "mincost_specialty", "unconstrained"]
u14 = avg("uniform_d14")
print(f"Table II (cost {'incl.' if INCLUDE_VISIT else 'excl.'} visit):")
for k in rows:
    print(f"  {k:18s} {s[k]['status']:34s} €{avg(k):7,.0f}  vs U14 {pct(avg(k), u14):+6.1f}%  catch {catch(k):5.1f}%")

# Derived claims
print("\nDerived:")
print(f"  MinCost(spec) vs uniform-14                 {pct(avg('mincost_specialty'), u14):+.1f}%")
print(f"  MinCost(spec) vs uniform-14 (capacity)      {pct(avg('mincost_specialty'), avg('uniform_d14_cap')):+.1f}%")
print(f"  Greedy(spec) above MinCost(spec)            {pct(avg('greedy_specialty'), avg('mincost_specialty')):+.1f}%")
print(f"  Greedy(global) above MinCost(global)        {pct(avg('greedy_global'), avg('mincost_global')):+.1f}%")
print(f"  MinCost(spec) above MinCost(global)         {pct(avg('mincost_specialty'), avg('mincost_global')):+.1f}%  (cost of specialty separation; must be >= 0)")
print(f"  Oracle vs MinCost(spec) (price of capacity) {pct(avg('unconstrained'), avg('mincost_specialty')):+.1f}%")
print(f"  Capacity baseline overflow: {s['uniform_d14_cap']['status']} / {s['guideline_cap']['status']}")

# Line 126: early vs late slots under MinCost (specialty), risk cost only
c = np.load(f"{P}/curves_test.npz")["curves_test"]
d = np.load(f"{P}/scheduling_results.npz")["mincost_specialty_days"]
risk_cost = 10_000 * (1 - c[np.arange(len(d)), d])
print(f"\n  Early (days 1-7) / late (days 21-30) mean risk cost: {risk_cost[d <= 7].mean() / risk_cost[d >= 21].mean():.1f}x")
```

### 5.6 Optional: measure timings on your hardware (for section 6.1 item 7)

This re-solves MinCost (specialty) pool by pool, so it takes as long as the pipeline's own solve.

```python
import time

import numpy as np
import polars as pl

from config import C_EVENT, C_VISIT, HORIZON_DAYS
from policy.mincost_solver import _solve_mincost_by_pool
from policy.specialty_scheduler import proportional_specialty_capacity, schedule_greedy_specialty

c = np.load("data/processed/curves_test.npz")["curves_test"]
pools = pl.read_parquet("data/processed/cohort.parquet").filter(
    pl.col("data_split") == "held_out")["specialty_pool"].to_numpy()
cap = proportional_specialty_capacity(pools, HORIZON_DAYS)

t = time.perf_counter(); schedule_greedy_specialty(c, pools, capacity_per_specialty_day=cap)
print(f"greedy (specialty): {(time.perf_counter() - t) * 1000:.0f} ms")
for k in range(4):
    m = pools == k
    t = time.perf_counter()
    _solve_mincost_by_pool(c[m], np.full(m.sum(), k), cap, C_EVENT, C_VISIT, HORIZON_DAYS)
    print(f"mincost pool {k} ({m.sum():,} patients): {time.perf_counter() - t:.1f} s")
```

### 5.7 Dashboard

The API loads `data/processed/` at startup, so restart it after the re-run with `docker compose up --build`. The overview's feasibility colouring now comes from the saved statuses.

---

## 6. Paper and README changes

### 6.1 Paper text that is wrong regardless of the re-run

1. **§III.D, line 91: capacity description.** Current: *"We set C_k proportionally to each specialty's share of the cohort - cardiology 15, neurology 10, surgery 15, general medicine 25 slots per day - so that total 30-day capacity exactly equals the test cohort size (27,641 slots for 27,641 patients)."* The 15/10/15/25 split isn't proportional to the cohort, and the old run had 27,690 slots. Suggested replacement, filled from the "Capacity" output of section 5.5:

   > We set each pool's daily capacity proportionally to its share of the test cohort, C_k = ⌈n_k / 30⌉, giving [C_card] cardiology, [C_neuro] neurology, [C_surg] surgery and [C_gm] general-medicine slots per day ([total] slots over 30 days for 27,641 patients, fewer than 30 spare slots per pool). Capacity is therefore binding but feasible: every patient can be scheduled within their pool, but not every patient can be scheduled on their optimal day.

2. **§V.D, line 183.** Current: *"The per-specialty capacity limits (15/10/15/25 slots per day) were set proportionally to cohort composition"*. Use the same wording and values as item 1.

3. **§IV.B Fourth finding (line 140): remove or reverse it.** Current: *"The specialty MinCost (EUR 759) outperforms global MinCost (EUR 999) by 24%."* With the same total daily capacity, the specialty constraints only restrict the global problem. So MinCost (global) ≤ MinCost (specialty) always holds, and specialty pooling can't win; the 24% came from the phantom day-1 slots. With the fix, the synthetic demo gives global €779 vs specialty €783 (seed 42) and €807 vs €810 (seed 7). Replace it with the *cost of specialty separation* ("MinCost(spec) above MinCost(global)" in section 5.5): the price of keeping clinics separate. Its current explanation ("pooling them into one shared queue wastes the structure…") must go too.

4. **Line 105: cost definition.** Current: *"Each cost figure represents the average per-patient expected adverse-event cost: the readmission probability at the assigned follow-up day multiplied by C_EVENT."* The code's `total_expected_cost` includes the €150 visit cost for every policy, so all Table II figures are €150 higher than that definition. Either reword to "…plus the €150 visit cost", or subtract €150: run section 5.5 with `INCLUDE_VISIT = False`. The percentages change slightly.

5. **§III.E, line 99: greedy description.** Current: *"The algorithm computes each patient's marginal benefit - the cost difference between their worst and best feasible day - and iterates in descending order."* The code (`schedule_greedy`, `schedule_greedy_specialty`) ranks patients by **urgency**, the largest single-day hazard `S(d−1) − S(d)` in days 1–14, most urgent first. Each patient then gets the lowest-cost free day, which is the earliest free day because risk only rises. Change the text to match the code, or change the code (and re-run).

6. **Table III (lines 163–174) conflicts with Table II.** GBM MinCost (specialty) is €239 with a 95.6% catch rate in Table III. That's below Table II's unconstrained oracle (€254, 96.8%), and the oracle is a lower bound for the same curves. The two can't come from the same curves and setup. Nothing in the repo produces Table III (section 7). Regenerate it or remove it, along with the "less than 4%" claim (line 174) and the §V.B text that relies on it.

7. **Timing claims.**
   - *"approximately 3 seconds per pool, compared to under 50ms for the greedy heuristic"* (line 93) and *"under 50ms for the full 27,641-patient test cohort"* (line 156).
   - Measured on synthetic curves at 27,641 patients (review machine): exact per-pool 55 s for a 5.4K-patient pool and ≥ 11 min for the 15.3K general-medicine pool; greedy 123 ms.

   Re-measure with section 5.6 and update both places.

8. **Comparator mismatch (lines 5, 20, 134).** The abstract's 47% (Table II's −46.6%) is against the *uncapacitated* uniform-14. Contribution 2 (line 20) says *"reduce expected adverse-event cost by 47% relative to capacity-aware uniform scheduling"*; against capacity-aware uniform it was 45.5% (line 134). Pick one comparator and use it everywhere. Section 5.5 prints both.

9. **Risk-bucket description (line 136).** Current: *"assigns patients to three groups based on their predicted 30-day readmission probability (top tertile to day 7, middle to day 14, bottom to day 30)"*. The code (`risk_bucket_policy`, defaults) uses fixed thresholds on 30-day risk: ≥ 30% → day 7, ≥ 15% → day 14, otherwise day 30. Change the text to match the thresholds. If you change the code to tertiles instead, re-run and update the risk-bucket row and the "5%" claims (lines 21, 136, 193).

10. **"Cross-pool interactions" explanation (lines 99 and 142).** Current: *"because heterogeneous pool sizes create interactions that greedy ordering cannot optimise globally"* and *"heterogeneous pool sizes create cross-pool interactions that require exact optimisation"*. The model has none: each pool is solved independently and slots can't move between pools. Any greedy-vs-optimal gap comes from within each pool. For example, urgency ranking plus the earliest free day doesn't minimise total cost. Rewrite once the re-run gives the new gap.

11. **Catch-rate definition (line 105).** Current: *"whose readmission occurred after their assigned follow-up day"*. The code (`event_before_followup_rate`) counts a miss only when `event_time < assigned_day`, so a readmission on the follow-up day itself counts as caught. Change it to "on or after".

12. **"Hungarian algorithm" (line 93).** Current: *"computed exactly via the Hungarian algorithm [17] in O(n^3) time"*. SciPy's `linear_sum_assignment` implements a modified Jonker–Volgenant algorithm (per its documentation), not the Hungarian algorithm. Both are exact with O(n³) worst case, so only the wording changes, e.g. "solved exactly with a shortest-augmenting-path (Jonker–Volgenant) assignment algorithm [17]". The README says the same (section 6.3).

13. **Greedy complexity (lines 99 and 156).** The paper says O(N·H). The greedy sorts patients first, so it is O(N log N + N·H). Minor.

### 6.2 Paper numbers to replace after the re-run

All come from section 5.5.

| Location | Current | Replace with |
|---|---|---|
| Abstract (line 5) | "reduces expected adverse-event cost by 47%", "catches 71%" | MinCost(spec) vs chosen comparator; its catch rate (the 37% uniform catch rate is unchanged) |
| Contribution 2 (line 20) | 47%, 71% | same; fix the comparator (6.1 item 8) |
| §III.D (line 91) | 15/10/15/25, 27,641 slots | new capacity (6.1 item 1) |
| §III.D (line 93) | timings | 6.1 item 7 |
| §III.E (line 99) | "over 99% of the optimal solver's cost reduction"; "the gap widens to 25% (EUR 946 vs EUR 759)" | greedy vs MinCost, global and specialty; rewrite the explanation (6.1 item 10) |
| Table II (lines 116–121) | Uniform-14 (capacity), Guideline (capacity), Greedy (global), Greedy (specialty), MinCost (global), MinCost (specialty) | new rows |
| Table II `*` marks and footnote (lines 116–117, 124) | "4,447 patients (16.1%) cannot be scheduled" | should be 0 now: remove the `*` marks and footnote if both statuses read `Feasible (capacity-aware)` |
| Line 126 | "3-4x higher" | early/late ratio |
| Finding 1 (line 134) | €759, 45.5%, €1,392, 71.0% vs 38.3%, "a near-doubling" | new values; recheck the wording |
| Finding 4 (line 140) | 24% specialty advantage | remove or reverse (6.1 item 3) |
| Finding 5 (line 142) | "EUR 1,003 vs EUR 999, <1% gap", "25% suboptimal … (EUR 946 vs EUR 759)" | new values; recheck the claim and explanation |
| Oracle paragraph (line 144) | "The gap between the oracle and the specialty MinCost…" | optionally quote the new gap |
| §V.A (line 156) | "modest optimality (25% under specialty constraints)", "under 50ms" | new values |
| Table III + §V.B text (lines 163–174) | cross-model results | regenerate or remove (6.1 item 6, section 7) |
| Conclusion (line 193) | "47% cost reduction", "71%" | same as abstract |

**Unchanged** (no capacity involved): the Guideline, Uniform day 14, Risk bucket and Unconstrained rows (lines 113–115, 122); the 5% risk-bucket claims (lines 21, 136, 193) unless the code changes (6.1 item 9); Table I; the cohort statistics; §II.

**What to expect (synthetic demo after the fix, not MIMIC-IV, seed 42):**
- MinCost (specialty) €783, 73.7% catch.
- MinCost (global) €779, 72.7%.
- Greedy (specialty) €1,077, 66.2%.
- Uniform-14 €1,125, 65.2%; Uniform-14 (capacity) €1,198, 58.6%.

So MinCost stays clearly best among the feasible policies. The greedy gap grows to about 38% (paper: 25%). The capacity-aware uniform baseline now costs slightly more than uncapacitated uniform-14.

### 6.3 README changes

| README line | Current | Change |
|---|---|---|
| 118 | `tests/ … # pytest suite (24 tests)` | 28 tests |
| 259 | "solved exactly via the Hungarian algorithm"; greedy "under 50ms" | 6.1 items 7 and 12 |
| 261–270 | Capacity table "default daily slots, configurable per hospital" (15/10/15/25) | Keep as API/dashboard defaults, but add that the paper's evaluation uses capacity proportional to each pool's size (`proportional_specialty_capacity`) |
| 282–293 | Model performance table | unchanged |
| 297–306 | Scheduling policy table (€946, €759, 71.0%, …) | new values |
| 308 | "47% cost reduction", "71%", "Per-specialty capacity pooling adds 24% improvement over global pooling" | new values; remove the 24% claim |
| 377 | Dashboard description "Hero metrics (47% cost reduction, 71% catch rate)" | new values (the dashboard computes them from `pipeline_results.json`; only the README text changes) |
| 394 | "pytest - test suite (24 tests passing)" | 28 |

---

## 7. Table III: reproducing it needs code changes (not done)

The current code can't produce Table III safely:
- `step_calibrate` calibrates only the best model and saves only its curves. The per-model `{name}_curves_test` arrays written by `step_train` are dropped when `curves_test.npz` is overwritten.
- `--survival-curves` overwrites `data/processed/curves_test.npz` and `models_info.json`, which destroys the calibrated GBM curves the main results use.
- `PROCESSED_DIR` is hard-coded, so a second run can't be pointed at a separate directory.

Suggested changes:
1. Allow `PROCESSED_DIR` to be overridden with an environment variable (the API already uses `CCPFS_DATA_DIR`).
2. Have `step_calibrate` calibrate and save curves for every trained model (`{model}_curves_test`).
3. Add a step (e.g. `--step schedule_all_models`) that runs MinCost (specialty) with `proportional_specialty_capacity` on each model's calibrated curves, plus MOTOR from `data/processed/motor_output/motor_curves.npz`, and writes a separate results file.

A quicker alternative: remove Table III and the cross-model scheduling claim.

---

## 8. Every problem found (complete list)

### 8.1 Paper

| # | Problem | Where handled |
|---|---|---|
| P1 | Capacity described as proportional to cohort share, with 27,641 slots and every patient schedulable. The code used a non-proportional split, gave 27,690 slots, and left ~4,447 patients without a slot (line 91). | 6.1 item 1; code fixed |
| P2 | Same wrong capacity description in §V.D (line 183). | 6.1 item 2 |
| P3 | Headline results inflated by phantom day-1 slots: abstract, contribution 2, Table II capacity-aware rows and footnote, line 126, findings 1 and 5, §III.E, §V.A, conclusion. | 6.2; needs re-run |
| P4 | Finding 4 ("specialty beats global by 24%") can't hold: with equal total capacity, global MinCost is a lower bound for specialty MinCost. | 6.1 item 3 |
| P5 | Cost figures include the €150 visit cost; the paper defines them without it (line 105). | 6.1 item 4 |
| P6 | Greedy described as marginal-benefit ordering; the code ranks by maximum early daily hazard (line 99). | 6.1 item 5 |
| P7 | Table III's GBM cost (€239) is below Table II's oracle lower bound (€254); no code in the repo produces Table III. | 6.1 item 6; section 7 |
| P8 | Timing claims: "~3 s per pool", greedy "under 50ms". Measured: 55 s for a 5.4K pool, ≥ 11 min for general medicine, 123 ms for greedy (lines 93, 156). | 6.1 item 7 |
| P9 | The 47% uses different comparators in the abstract (uncapacitated uniform-14) and contribution 2 (capacity-aware uniform) (lines 5, 20). | 6.1 item 8 |
| P10 | Risk bucket described as tertiles; the code uses fixed 30% / 15% thresholds (line 136). | 6.1 item 9 |
| P11 | The "cross-pool interactions" explanation contradicts the model, where pools are independent (lines 99, 142). | 6.1 item 10 |
| P12 | Catch rate defined as a readmission "after" follow-up; the code counts same-day events as caught (line 105). | 6.1 item 11 |
| P13 | "Hungarian algorithm": SciPy uses a modified Jonker–Volgenant algorithm (line 93). | 6.1 item 12 |
| P14 | Greedy complexity stated as O(N·H); it is O(N log N + N·H) (lines 99, 156). | 6.1 item 13 |
| P15 | README repeats P3, P4, P8 and P13 and states the old test count. | 6.3 |

### 8.2 Code: fixed (commits in section 3.2)

| # | Problem | Fix |
|---|---|---|
| C1 | Pipeline capacity scaled the default 15/10/15/25 split, so general medicine had 38% of slots for about 55% of patients. | `proportional_specialty_capacity`, used in `run_pipeline.py` |
| C2 | The per-pool min-cost solver put every overflow patient on day 1, ignoring capacity, and still reported "Optimal". | Overflow spread over least-loaded days |
| C3 | The min-cost solver built the full cost matrix (~6 GB at 27,641 patients) before checking whether to decompose by pool. | Size check moved first |
| C4 | `schedule_mincost_global` modified the caller's `capacity_per_day` array. | Copies the array |
| C5 | The dashboard hard-coded the capacity baselines as infeasible and MinCost (specialty) as feasible, regardless of actual overflow. | Feasibility read from saved status |
| C6 | The demo used the unscaled default capacity, so it didn't mirror the paper's setup. | Same proportional rule |
| C7 | `polars` was missing from `requirements.txt`, so the README demo steps crashed on a clean install. | Added |
| C8 | PuLP: `PULP_CBC_CMD` and direct `LpVariable` construction are deprecated (~18K warnings per test run). The root file had `PuLP>=3.3` while the API pinned 3.3.0. | `COIN_CMD` + `PuLP[cbc]`, `prob.add_variable`, both files on 3.3.2 |
| C9 | README said `cd ccpfs/scheduling_follow_up` and `cd scheduling_follow_up` (4 places) and showed the old layout root. The `demo_setup.py` docstring and a `run_pipeline.py` comment had the same leftover. | Corrected |
| C10 | README listed `DEMO_MODE_PLAN.md`, which doesn't exist. | Removed |
| C11 | Demo event rate was 26.9%; the README claims about 20%. | Risk mix 10/40/50 → 20.5% |
| C12 | The patient explorer showed a hard-coded range "1 to 27,641" (inputs are 0-based; the demo has 1,000 patients), with hard-coded input bounds. | Range and bounds from the API |
| C13 | The scheduler page said curves come from MIMIC-IV, including in demo mode. | Cohort-agnostic text |
| C14 | `/api/schedule` silently capped demand at pool size, so the demo's demand sliders stopped having any effect. | Resampling with replacement |

### 8.3 Code: not fixed

| # | Problem | Notes |
|---|---|---|
| U1 | Table III can't be reproduced: calibration keeps only the best model's curves, `--survival-curves` overwrites `curves_test.npz` and `models_info.json`, and `PROCESSED_DIR` has no override. | Section 7 |
| U2 | `schedule_mincost_global` builds a dense ~6.1 GB matrix at 27,641 patients and has no decomposition. | Needs ≥ 16 GB RAM (5.4) |
| U3 | The exact per-pool solve is slow for large pools: 55 s at 5.4K patients, ≥ 11 min for general medicine (15.3K). | Relevant to P8 |
| U4 | The `evaluation/synthetic.py` docstring says the risk groups have ~35% / 15% / 5% event rates; the parameters actually give ~48% / 24% / 12%. | Docstring only |
| U5 | `demo_setup.py` writes fixed, made-up model metrics (C-index 0.704, etc.) that the demo dashboard shows. Tagged synthetic, not computed. | Worth a README note |
| U6 | The patient explorer fetches the patient count once, when the dashboard starts. If the API isn't up yet, it falls back to the MIMIC value (27,640). | Minor |
| U7 | `/api/patients/{i}/curve` falls back to assigned day 15 when no scheduling results are loaded, without saying so. | Minor |

---

## 9. Record of the review session

### 9.1 What was verified

- **Clean install:** from `requirements.txt` in a new Python 3.12 venv, `pytest tests/ -q -W error::DeprecationWarning` gives 28 passed. Each of the three code commits in section 3.2 also passes on its own on top of `71e8e1b`.
- **Demo:** `python demo_setup.py` runs in about 5 s, with a 20.5% event rate (seed 42) and capacity 8/4/4/19 per day for the 1,000-patient test split.
- **API** (with `api/requirements.txt`, on demo data):
  - `/api/health`, `/api/patients/count` and `/api/results` respond; results show 10 policies, all flagged feasible after the fix.
  - `/api/patients/999/curve` returns 200 and `/1000` returns 404.
  - `/api/schedule` honours demand beyond pool size (1,260 / 1,860 / 3,660 patients), handles zero demand, and rejects unknown specialties with 422.
- **ILP solvers:** `schedule_ilp` and `schedule_ilp_specialty` solve to Optimal through `COIN_CMD` + `cbcbox` with the venv's `bin/` not on `PATH`.
- **Full scale, fixed code** (synthetic curves, 27,641 patients, MIMIC mix):
  - Capacity 181/121/110/511 per day, 27,690 slots.
  - Uniform-14 (capacity) and Greedy (specialty) both `Feasible`.
  - Per-pool timings as in 5.4.
  - Peak memory of `schedule_mincost_specialty` with the old skewed capacity: 0.31 GB (was ~6 GB before C3).

### 9.2 Reproducing the synthetic evidence (no MIMIC-IV needed)

Old vs new per-pool solver under the old scaling rule (section 2.4). Run from the repo root:

```bash
git show 71e8e1b:policy/mincost_solver.py > /tmp/old_mincost.py
PYTHONPATH=. python - <<'EOF'
import importlib.util
import numpy as np
from config import C_EVENT, C_VISIT, DEFAULT_SPECIALTY_CAPACITY, HORIZON_DAYS
from evaluation.metrics import event_before_followup_rate
from evaluation.synthetic import generate_synthetic_cohort
import policy.mincost_solver as new

spec = importlib.util.spec_from_file_location("old", "/tmp/old_mincost.py")
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)

for seed in (42, 7):
    N = 5000
    g = generate_synthetic_cohort(n_patients=N, seed=seed, high_risk_frac=0.10, med_risk_frac=0.40)
    c, e, t = g["survival_curves"], g["event_indicators"], g["event_times"]
    pools = np.random.default_rng(seed).choice(4, size=N, p=[.197, .137, .115, .551])
    # the pipeline's OLD capacity rule
    scale = max(1.0, int(np.ceil(N / HORIZON_DAYS)) / sum(DEFAULT_SPECIALTY_CAPACITY.values()))
    cap = {k: int(np.ceil(v * scale)) for k, v in DEFAULT_SPECIALTY_CAPACITY.items()}
    u14 = np.mean(C_EVENT * (1 - c[:, 14]) + C_VISIT)
    for name, fn in [("old", old._solve_mincost_by_pool), ("new", new._solve_mincost_by_pool)]:
        r = fn(c, pools, cap, C_EVENT, C_VISIT, HORIZON_DAYS)
        cost = r["total_expected_cost"] / N
        catch = event_before_followup_rate(r["assignments"], t, e)["catch_rate"] * 100
        day1 = sum(1 for d in r["assignments"].values() if d == 1)
        print(f"seed {seed} {name}: €{cost:.0f} ({(cost - u14) / u14 * 100:+.0f}% vs U14) catch {catch:.1f}% day-1 {day1}")
EOF
```

Fixed pipeline behaviour on the synthetic demo (section 6.2 expectations):

```bash
python demo_setup.py --out-dir /tmp/ccpfs_demo --seed 42    # prints capacity and every policy's cost/catch
```
