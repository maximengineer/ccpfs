#!/usr/bin/env python3
"""
Bootstrap Confidence Intervals for Scheduling Results
------------------------------------------------------
Paired percentile bootstrap over test patients for every policy in
scheduling_results.npz (Table II) and, if present, every model schedule in
cross_model/ (Table III).

Each replicate resamples test patients with replacement and re-scores every
policy on the same resampled patients, so differences between policies are
paired. Schedules are held fixed: the intervals reflect which patients happen
to be in the test set, not re-training the model or re-solving the schedule.

Reported per policy: mean expected cost per patient (incl. the EUR 150 visit,
as in Table II) and catch rate (readmissions on or after the follow-up day /
all readmissions). Reported per comparison: cost difference in % and catch-rate
difference in percentage points, each with a 95% interval.

If scheduling_results.npz predates the capacity-aware risk bucket
(risk_bucket_cap), that policy is computed here with the same capacity rule
as run_pipeline.py.

Usage:
    PYTHONPATH=. python evaluation/bootstrap_ci.py                       # data/processed
    PYTHONPATH=. python evaluation/bootstrap_ci.py --data-dir data/demo  # synthetic demo
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import C_EVENT, C_VISIT, HORIZON_DAYS, PROCESSED_DIR
from policy.baselines import risk_bucket_capacity_policy
from policy.specialty_scheduler import proportional_specialty_capacity

POLICIES = [
    "guideline", "uniform_d14", "risk_bucket", "uniform_d14_cap", "guideline_cap",
    "risk_bucket_cap", "greedy_global", "greedy_specialty", "mincost_global",
    "mincost_specialty", "unconstrained",
]

# (label, policy_a, policy_b): reported as a relative to b
COMPARISONS = [
    ("MinCost (spec) vs Uniform-14 (capacity)", "mincost_specialty", "uniform_d14_cap"),
    ("MinCost (spec) vs Uniform-14", "mincost_specialty", "uniform_d14"),
    ("MinCost (spec) vs Risk bucket (capacity)", "mincost_specialty", "risk_bucket_cap"),
    ("Risk bucket (capacity) vs Uniform-14 (capacity)", "risk_bucket_cap", "uniform_d14_cap"),
    ("Risk bucket (capacity) vs Uniform-14", "risk_bucket_cap", "uniform_d14"),
    ("Risk bucket vs Uniform-14", "risk_bucket", "uniform_d14"),
    ("MinCost (spec) vs MinCost (global)", "mincost_specialty", "mincost_global"),
    ("Greedy (spec) vs MinCost (spec)", "greedy_specialty", "mincost_specialty"),
    ("Greedy (global) vs MinCost (global)", "greedy_global", "mincost_global"),
    ("Greedy (spec) vs Greedy (global)", "greedy_specialty", "greedy_global"),
]

CROSS_MODELS = ["cox", "gbm", "rsf", "motor"]


def load_schedules(data_dir: Path, curves: np.ndarray) -> dict:
    """Assigned day per test patient for each policy found on disk."""
    sched = np.load(data_dir / "scheduling_results.npz", allow_pickle=True)
    days = {p: sched[f"{p}_days"].astype(int) for p in POLICIES if f"{p}_days" in sched}

    if "risk_bucket_cap" not in days:
        cohort = pl.read_parquet(data_dir / "cohort.parquet")
        pools = cohort.filter(pl.col("data_split") == "held_out")["specialty_pool"].to_numpy()
        if len(pools) != len(curves):
            raise RuntimeError(
                f"cohort test split has {len(pools)} rows but curves_test.npz has {len(curves)} "
                f"(pipeline run with --max-patients?); re-run the schedule step instead"
            )
        cap = proportional_specialty_capacity(pools, HORIZON_DAYS)
        r = risk_bucket_capacity_policy(curves, pools, capacity_per_specialty_day=cap)
        days["risk_bucket_cap"] = np.array([r["assignments"][i] for i in range(len(curves))])
        print(f"risk_bucket_cap computed here (not in scheduling_results.npz): {r['status']}")

    for p, d in list(days.items()):
        if len(d) != len(curves) or (d < 1).any():
            print(f"skipping {p}: {int((d < 1).sum())} unassigned patients or wrong length")
            del days[p]
    return days


def bootstrap(per_patient_cost: np.ndarray, caught: np.ndarray, events: np.ndarray,
              n_boot: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Replicate mean cost (B, P) and catch rate (B, P) with paired resampling.

    per_patient_cost and caught are (n, P); events is (n,).
    """
    n = len(events)
    rng = np.random.default_rng(seed)
    caught_events = caught & events[:, None]
    costs = np.empty((n_boot, per_patient_cost.shape[1]))
    catch = np.empty_like(costs)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n)
        costs[b] = w @ per_patient_cost / n
        catch[b] = (w @ caught_events) / (w @ events)
    return costs, catch


def ci(x: np.ndarray) -> list[float]:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    cdata = np.load(data_dir / "curves_test.npz")
    curves = cdata["curves_test"]
    events = cdata["e_test"].astype(bool)
    times = cdata["t_test"].astype(float)
    n = len(curves)
    rows = np.arange(n)

    days = load_schedules(data_dir, curves)
    names = list(days)
    cost_pp = np.column_stack([C_EVENT * (1 - curves[rows, days[p]]) + C_VISIT for p in names])
    caught = np.column_stack([times >= days[p] for p in names])

    boot_cost, boot_catch = bootstrap(cost_pp, caught, events, args.n_boot, args.seed)
    point_cost = cost_pp.mean(axis=0)
    point_catch = (caught & events[:, None]).sum(axis=0) / events.sum()

    out = {"n_test": n, "n_events": int(events.sum()), "n_boot": args.n_boot,
           "policies": {}, "comparisons": {}, "cross_model": {}}

    print(f"\nTable II with 95% intervals ({n:,} patients, {int(events.sum()):,} readmissions, "
          f"{args.n_boot} paired replicates)")
    print(f"  {'policy':18s} {'EUR/patient':>11s} {'95% CI':>17s} {'catch':>7s} {'95% CI':>15s}")
    for j, p in enumerate(names):
        c_ci, k_ci = ci(boot_cost[:, j]), ci(boot_catch[:, j] * 100)
        out["policies"][p] = {"avg_cost": float(point_cost[j]), "avg_cost_ci": c_ci,
                              "catch_rate_pct": float(point_catch[j] * 100), "catch_rate_pct_ci": k_ci}
        print(f"  {p:18s} {point_cost[j]:11,.0f} [{c_ci[0]:7,.0f}, {c_ci[1]:7,.0f}] "
              f"{point_catch[j] * 100:6.1f}% [{k_ci[0]:5.1f}, {k_ci[1]:5.1f}]")

    print("\nPaired differences (a relative to b); an interval containing 0 means no clear difference")
    for label, a, b in COMPARISONS:
        if a not in days or b not in days:
            continue
        ia, ib = names.index(a), names.index(b)
        cost_pct = (boot_cost[:, ia] - boot_cost[:, ib]) / boot_cost[:, ib] * 100
        catch_pts = (boot_catch[:, ia] - boot_catch[:, ib]) * 100
        pc = (point_cost[ia] - point_cost[ib]) / point_cost[ib] * 100
        pk = (point_catch[ia] - point_catch[ib]) * 100
        out["comparisons"][label] = {"cost_pct": pc, "cost_pct_ci": ci(cost_pct),
                                     "catch_pts": pk, "catch_pts_ci": ci(catch_pts)}
        c_ci, k_ci = ci(cost_pct), ci(catch_pts)
        print(f"  {label:48s} cost {pc:+6.1f}% [{c_ci[0]:+6.1f}, {c_ci[1]:+6.1f}]   "
              f"catch {pk:+5.1f} pts [{k_ci[0]:+5.1f}, {k_ci[1]:+5.1f}]")

    # Table III: catch rate is comparable across models (observed outcomes);
    # expected cost is not (each model scores its own schedule), so only catch
    cross_dir = data_dir / "cross_model"
    model_days = {m: np.load(cross_dir / f"{m}_days.npy").astype(int)
                  for m in CROSS_MODELS if (cross_dir / f"{m}_days.npy").exists()}
    for m, d in list(model_days.items()):
        if len(d) != n or (d < 1).any():
            print(f"skipping cross-model {m}: wrong length or unassigned patients")
            del model_days[m]
    if model_days:
        models = list(model_days)
        m_caught = np.column_stack([times >= model_days[m] for m in models])
        _, m_boot = bootstrap(np.zeros((n, len(models))), m_caught, events, args.n_boot, args.seed)
        m_point = (m_caught & events[:, None]).sum(axis=0) / events.sum()
        print("\nTable III catch rates with 95% intervals")
        for j, m in enumerate(models):
            entry = {"catch_rate_pct": float(m_point[j] * 100), "catch_rate_pct_ci": ci(m_boot[:, j] * 100)}
            line = f"  {m:6s} {m_point[j] * 100:5.1f}% [{entry['catch_rate_pct_ci'][0]:5.1f}, {entry['catch_rate_pct_ci'][1]:5.1f}]"
            if "gbm" in models and m != "gbm":
                g = models.index("gbm")
                d = (m_boot[:, j] - m_boot[:, g]) * 100
                entry["vs_gbm_pts"] = float((m_point[j] - m_point[g]) * 100)
                entry["vs_gbm_pts_ci"] = ci(d)
                line += f"   vs GBM {entry['vs_gbm_pts']:+5.1f} pts [{ci(d)[0]:+5.1f}, {ci(d)[1]:+5.1f}]"
            out["cross_model"][m] = entry
            print(line)
    else:
        print(f"\nNo {cross_dir}/<model>_days.npy files; run evaluation/cross_model_scheduling.py for Table III intervals")

    out_path = data_dir / "bootstrap_ci.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
