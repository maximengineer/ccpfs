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
        style={"textAlign": "center", "color": "#6b7280", "marginBottom": "24px"},
    ),

    # Hero metrics (populated by callback)
    html.Div(id="hero-metrics", style={
        "display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "24px",
    }),

    # Policy comparison chart
    section_header("Scheduling Policy Comparison"),
    dcc.Graph(id="policy-chart"),

    # Model comparison
    section_header("Survival Model Performance"),
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
