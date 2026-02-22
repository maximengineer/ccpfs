"""
Tests for the scheduling policy layer.
Validates correctness of cost functions, schedulers, and baselines
using synthetic survival curves.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from policy.cost_function import (
    expected_cost,
    expected_cost_curve,
    unconstrained_optimal_day,
    marginal_benefit,
)
from policy.ilp_scheduler import schedule_ilp
from policy.greedy_scheduler import schedule_greedy
from policy.baselines import (
    uniform_policy,
    risk_bucket_policy,
    guideline_policy,
    unconstrained_optimal_policy,
)
from policy.uncertainty_adjustment import (
    apply_uncertainty_adjustment,
    uncertainty_score,
)
from evaluation.synthetic import generate_synthetic_cohort, generate_weibull_survival
from evaluation.metrics import (
    event_before_followup_rate,
    expected_cost_metric,
    capacity_utilisation,
    evaluate_policy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_survival():
    """A simple linearly decreasing survival curve for 30 days."""
    # S(t) = 1 - t/60, so R(30) = 0.5
    t = np.arange(0, 31, dtype=float)
    return 1.0 - t / 60.0


@pytest.fixture
def cohort():
    """Small synthetic cohort for integration tests."""
    return generate_synthetic_cohort(n_patients=100, seed=42)


# ---------------------------------------------------------------------------
# Cost function tests
# ---------------------------------------------------------------------------

class TestCostFunction:
    def test_expected_cost_day1(self, simple_survival):
        cost = expected_cost(simple_survival, day=1, c_event=10000, c_visit=150)
        risk = 1.0 - simple_survival[1]  # 1/60 ~ 0.0167
        assert abs(cost - (10000 * risk + 150)) < 0.01

    def test_expected_cost_increases_with_day(self, simple_survival):
        cost_early = expected_cost(simple_survival, day=5, c_event=10000, c_visit=150)
        cost_late = expected_cost(simple_survival, day=25, c_event=10000, c_visit=150)
        # Later day = higher risk = higher cost
        assert cost_late > cost_early

    def test_cost_curve_length(self, simple_survival):
        curve = expected_cost_curve(simple_survival, horizon=30)
        assert len(curve) == 30

    def test_unconstrained_optimal_is_day1(self, simple_survival):
        # Without constraints, day 1 always minimises cost (lowest risk)
        day = unconstrained_optimal_day(simple_survival)
        assert day == 1

    def test_marginal_benefit_positive_when_moving_earlier(self, simple_survival):
        benefit = marginal_benefit(simple_survival, from_day=20, to_day=5)
        assert benefit > 0  # Moving earlier reduces expected harm

    def test_marginal_benefit_negative_when_moving_later(self, simple_survival):
        benefit = marginal_benefit(simple_survival, from_day=5, to_day=20)
        assert benefit < 0  # Moving later increases expected harm


# ---------------------------------------------------------------------------
# ILP scheduler tests
# ---------------------------------------------------------------------------

class TestILPScheduler:
    def test_all_patients_assigned(self, cohort):
        curves = cohort["survival_curves"]
        n = curves.shape[0]
        cap = np.full(30, 10)  # 10 * 30 = 300 slots for 100 patients
        result = schedule_ilp(curves, cap, time_limit=30)
        assert result["status"] == "Optimal"
        assert len(result["assignments"]) == n

    def test_capacity_respected(self, cohort):
        curves = cohort["survival_curves"]
        cap = np.full(30, 10)
        result = schedule_ilp(curves, cap, time_limit=30)
        # Count patients per day
        day_counts = np.zeros(30)
        for d in result["assignments"].values():
            day_counts[d - 1] += 1
        assert np.all(day_counts <= cap)

    def test_each_patient_assigned_once(self, cohort):
        curves = cohort["survival_curves"]
        cap = np.full(30, 10)
        result = schedule_ilp(curves, cap, time_limit=30)
        days = list(result["assignments"].values())
        assert all(1 <= d <= 30 for d in days)

    def test_infeasible_when_capacity_too_low(self):
        # 10 patients but only 1 slot total
        curves = np.ones((10, 31))
        curves[:, 1:] = 0.95
        cap = np.zeros(30)
        cap[0] = 1  # Only 1 slot across all 30 days... need 10
        result = schedule_ilp(curves, cap, time_limit=10)
        assert result["status"] == "Infeasible"


# ---------------------------------------------------------------------------
# Greedy scheduler tests
# ---------------------------------------------------------------------------

class TestGreedyScheduler:
    def test_all_patients_assigned(self, cohort):
        curves = cohort["survival_curves"]
        cap = np.full(30, 10)
        result = schedule_greedy(curves, cap)
        assert result["status"] == "Feasible"
        assert len(result["assignments"]) == curves.shape[0]

    def test_high_risk_patients_get_earlier_days(self, cohort):
        curves = cohort["survival_curves"]
        groups = cohort["risk_groups"]
        cap = np.full(30, 10)
        result = schedule_greedy(curves, cap)

        high_days = [result["assignments"][i] for i in range(len(groups)) if groups[i] == 2]
        low_days = [result["assignments"][i] for i in range(len(groups)) if groups[i] == 0]

        # High-risk patients should on average have earlier follow-up
        assert np.mean(high_days) < np.mean(low_days)


# ---------------------------------------------------------------------------
# Baseline tests
# ---------------------------------------------------------------------------

class TestBaselines:
    def test_uniform_all_same_day(self, cohort):
        curves = cohort["survival_curves"]
        result = uniform_policy(curves, day=14)
        assert all(d == 14 for d in result["assignments"].values())

    def test_risk_bucket_three_groups(self, cohort):
        curves = cohort["survival_curves"]
        result = risk_bucket_policy(curves)
        days = set(result["assignments"].values())
        assert days.issubset({7, 14, 30})

    def test_guideline_hf_gets_14(self, cohort):
        curves = cohort["survival_curves"]
        hf = cohort["is_heart_failure"]
        result = guideline_policy(curves, is_heart_failure=hf)
        for i, is_hf in enumerate(hf):
            if is_hf:
                assert result["assignments"][i] == 14
            else:
                assert result["assignments"][i] == 28

    def test_unconstrained_assigns_day1(self):
        # For a steep survival curve, unconstrained should assign day 1
        curves = np.ones((5, 31))
        for t in range(1, 31):
            curves[:, t] = 1.0 - t * 0.02  # 60% survival at day 30
        result = unconstrained_optimal_policy(curves)
        assert all(d == 1 for d in result["assignments"].values())


# ---------------------------------------------------------------------------
# Uncertainty adjustment tests
# ---------------------------------------------------------------------------

class TestUncertainty:
    def test_conservative_lowers_survival(self):
        means = np.array([[1.0, 0.9, 0.8, 0.7]])
        stds = np.array([[0.0, 0.05, 0.05, 0.05]])
        conservative = apply_uncertainty_adjustment(means, stds, alpha=1.0)
        # Conservative survival should be <= mean (higher apparent risk)
        assert np.all(conservative <= means + 1e-10)

    def test_alpha_zero_returns_mean(self):
        means = np.array([[1.0, 0.9, 0.8, 0.7]])
        stds = np.array([[0.0, 0.05, 0.05, 0.05]])
        result = apply_uncertainty_adjustment(means, stds, alpha=0.0)
        np.testing.assert_array_almost_equal(result, means)

    def test_uncertainty_score_shape(self):
        stds = np.random.rand(10, 31) * 0.1
        scores = uncertainty_score(stds)
        assert scores.shape == (10,)


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_ebf_perfect_policy(self):
        # All patients have events on day 20; policy assigns day 10 (before event)
        assignments = {0: 10, 1: 10, 2: 10}
        event_times = np.array([20, 20, 20])
        indicators = np.array([1, 1, 1])
        result = event_before_followup_rate(assignments, event_times, indicators)
        assert result["ebf_count"] == 0
        assert result["catch_rate"] == 1.0

    def test_ebf_worst_policy(self):
        # All events on day 5; policy assigns day 25 (after all events)
        assignments = {0: 25, 1: 25, 2: 25}
        event_times = np.array([5, 5, 5])
        indicators = np.array([1, 1, 1])
        result = event_before_followup_rate(assignments, event_times, indicators)
        assert result["ebf_count"] == 3
        assert result["catch_rate"] == 0.0

    def test_capacity_overflow_detected(self):
        assignments = {0: 1, 1: 1, 2: 1}  # All on day 1
        cap = np.full(30, 2)  # Only 2 slots per day
        result = capacity_utilisation(assignments, cap, horizon=30)
        assert result["overflow_days"] == 1  # Day 1 overflows


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_ilp_beats_uniform_on_ebf(self, cohort):
        """The optimised policy should catch more events than uniform-14."""
        curves = cohort["survival_curves"]
        events = cohort["event_times"]
        indicators = cohort["event_indicators"]
        cap = np.full(30, 10)

        # Uniform-14
        uniform = uniform_policy(curves, day=14)
        ebf_uniform = event_before_followup_rate(
            uniform["assignments"], events, indicators
        )

        # ILP
        ilp = schedule_ilp(curves, cap)
        ebf_ilp = event_before_followup_rate(
            ilp["assignments"], events, indicators
        )

        # ILP should have fewer events before follow-up (or equal)
        assert ebf_ilp["ebf_count"] <= ebf_uniform["ebf_count"]

    def test_full_simulation_runs(self, cohort):
        """End-to-end: generate cohort, run all policies, get results table."""
        from evaluation.simulate import run_all_policies, results_summary_table

        results = run_all_policies(
            survival_curves=cohort["survival_curves"],
            event_times=cohort["event_times"],
            event_indicators=cohort["event_indicators"],
            survival_stds=cohort["survival_stds"],
            is_heart_failure=cohort["is_heart_failure"],
        )

        # Should have at least 5 policies
        assert len(results) >= 5

        # Print the comparison table
        table = results_summary_table(results)
        print("\n" + table)
