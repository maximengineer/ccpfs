"""Page 1: Overview Dashboard - key metrics + policy comparison."""

import os

import dash
import requests
from dash import html, dcc, callback, Input, Output

dash.register_page(__name__, path="/", name="Overview", order=0)

from dashboard.components.charts import (
    model_comparison_figure,
    policy_comparison_figure,
)
from dashboard.components.layout import metric_card, section_header

API_URL = os.environ.get("CCPFS_API_URL", "http://localhost:8000")

layout = html.Div([
    html.H2("CCPFS Dashboard", style={"textAlign": "center", "color": "#1f2937"}),
    html.P(
        "Capacity-Constrained Personalized Follow-Up Scheduling",
        style={"textAlign": "center", "color": "#6b7280", "marginBottom": "8px"},
    ),

    # Framework explainer
    html.Div(style={
        "backgroundColor": "#f0f9ff", "borderRadius": "8px", "padding": "16px",
        "borderLeft": "4px solid #2563eb", "marginBottom": "24px",
    }, children=[
        html.P([
            html.Strong("What is this? "),
            "After hospital discharge, patients need follow-up appointments, but clinic capacity is limited. "
            "This framework uses each patient's individual readmission risk trajectory to decide ",
            html.Em("when"),
            " they should be seen, so that high-benefit patients get earlier slots while respecting daily clinic limits.",
        ], style={"margin": "0 0 8px 0", "color": "#1e3a5f", "fontSize": "14px"}),
        html.P([
            html.Strong("How to read the results below: "),
            html.Strong("Cost Reduction", style={"color": "#16a34a"}),
            " shows how much the optimised scheduler saves compared to giving everyone a fixed day-14 appointment. ",
            html.Strong("Catch Rate", style={"color": "#2563eb"}),
            " is the percentage of patients who were readmitted ",
            html.Em("after"),
            " their scheduled follow-up, meaning a clinician could have intervened in time.",
        ], style={"margin": "0", "color": "#1e3a5f", "fontSize": "14px"}),
    ]),

    # Hero metrics (populated by callback)
    html.Div(id="hero-metrics", style={
        "display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "24px",
    }),

    # Policy comparison chart
    section_header("Scheduling Policy Comparison"),
    html.P(
        "Each bar represents a different scheduling strategy. "
        "The percentage shows cost change relative to the Uniform day 14 baseline. "
        "Negative values (green) mean the policy reduces adverse-event exposure.",
        style={"color": "#6b7280", "fontSize": "13px", "marginBottom": "8px"},
    ),

    # Policy legend
    html.Div(style={
        "backgroundColor": "#f9fafb", "borderRadius": "8px", "padding": "12px 16px",
        "marginBottom": "16px", "fontSize": "13px", "color": "#374151",
        "display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "6px 24px",
    }, children=[
        html.Div([html.Strong("Baselines (fixed rules):"),]),
        html.Div([html.Strong("Optimised (patient-level):"),]),
        html.Div([
            html.Span("Uniform day 14", style={"fontWeight": "600"}),
            " - every patient seen at day 14 regardless of risk",
        ]),
        html.Div([
            html.Span("Greedy (global/specialty)", style={"fontWeight": "600"}),
            " - fast heuristic that assigns patients by marginal benefit in descending order",
        ]),
        html.Div([
            html.Span("Risk bucket", style={"fontWeight": "600"}),
            " - top-third risk at day 7, middle at day 14, bottom at day 30",
        ]),
        html.Div([
            html.Span("MinCost (global/specialty)", style={"fontWeight": "600"}),
            " - exact optimal solver that minimises total expected cost across all patients",
        ]),
        html.Div([
            html.Span("Guideline (ACC/AHA)", style={"fontWeight": "600"}),
            " - heart failure at day 14, all other conditions at day 28",
        ]),
        html.Div([
            html.Span("Unconstrained (oracle)", style={"fontWeight": "600"}),
            " - theoretical best: every patient on their individual optimal day, ignoring capacity",
        ]),
        html.Div([
            html.Span("(capacity)", style={"fontWeight": "600"}),
            " - same rule but with daily clinic limits enforced; overflow patients are pushed later",
        ]),
        html.Div([
            html.Span("global vs specialty", style={"fontWeight": "600"}),
            " - global = one shared queue; specialty = separate pools per clinical service",
        ]),
    ]),

    dcc.Graph(id="policy-chart"),

    # Model comparison
    section_header("Survival Model Performance"),
    html.P([
        "The framework is model-agnostic: it works with any survival model that produces patient-level "
        "risk curves. Here we compare four architectures. ",
        html.Strong("C-index"),
        " measures how well the model ranks patients by risk (1.0 = perfect, 0.5 = random). ",
        html.Strong("IBS"),
        " (Integrated Brier Score) measures calibration and accuracy over 30 days (lower = better).",
    ], style={"color": "#6b7280", "fontSize": "13px", "marginBottom": "8px"}),
    dcc.Graph(id="model-chart"),

    # Hidden store for API data
    dcc.Store(id="pipeline-data"),
    dcc.Interval(id="load-trigger", interval=1000, max_intervals=1),
])


@callback(
    Output("pipeline-data", "data"),
    Input("load-trigger", "n_intervals"),
)
def load_data(_):
    try:
        resp = requests.get(f"{API_URL}/api/results", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


@callback(
    Output("hero-metrics", "children"),
    Output("policy-chart", "figure"),
    Output("model-chart", "figure"),
    Input("pipeline-data", "data"),
)
def update_overview(data):
    if not data or "error" in data:
        empty = {"data": [], "layout": {"title": "Loading..."}}
        return [], empty, empty

    cohort = data["cohort"]
    models = data["models"]
    policies = data["policies"]

    # Find best policy (MinCost specialty)
    best = next((p for p in policies if p["name"] == "MinCost (specialty)"), policies[-1])

    metrics = [
        metric_card("Cost Reduction", f"{abs(best['vs_uniform_pct']):.0f}%",
                     "vs uniform scheduling", "#16a34a"),
        metric_card("Catch Rate", f"{best['catch_rate']:.1f}%",
                     "events caught before follow-up", "#2563eb"),
        metric_card("Test Patients", f"{cohort['test_episodes']:,}",
                     f"{cohort['event_rate']*100:.1f}% readmission rate", "#7c3aed"),
        metric_card("Models Compared", str(len(models)),
                     "architectures validated", "#d97706"),
    ]

    policy_fig = policy_comparison_figure(policies)
    model_fig = model_comparison_figure(models)

    return metrics, policy_fig, model_fig
