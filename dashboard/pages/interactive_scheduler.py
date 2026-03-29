"""Page 3: Interactive Scheduler - adjust capacity and re-schedule in real time."""

import os

import dash
import requests
from dash import html, dcc, callback, Input, Output, State

dash.register_page(__name__, path="/scheduler", name="Interactive Scheduler", order=2)

from dashboard.components.charts import day_histogram_figure
from dashboard.components.layout import metric_card, section_header

API_URL = os.environ.get("CCPFS_API_URL", "http://localhost:8000")

layout = html.Div([
    html.H2("Interactive Scheduler", style={"textAlign": "center", "color": "#1f2937"}),

    # Explainer
    html.Div(style={
        "backgroundColor": "#f0f9ff", "borderRadius": "8px", "padding": "16px",
        "borderLeft": "4px solid #2563eb", "marginBottom": "24px",
    }, children=[
        html.P([
            html.Strong("What does this page do? "),
            "This is a live demonstration of the scheduling framework. Use the sliders below to "
            "simulate different hospital capacity scenarios, then click ",
            html.Strong("Run Scheduler"),
            " to re-optimise all 27,641 patient appointments in real time using the greedy heuristic.",
        ], style={"margin": "0 0 8px 0", "color": "#1e3a5f", "fontSize": "14px"}),
        html.P([
            html.Strong("What to try: "),
            "Reduce cardiology capacity to see high-risk cardiac patients pushed to later days, "
            "increasing cost and lowering catch rate. Or increase capacity across all specialties "
            "to watch costs drop as more patients can be scheduled earlier. The histogram below "
            "shows how patients are distributed across the 30-day window.",
        ], style={"margin": "0 0 8px 0", "color": "#1e3a5f", "fontSize": "14px"}),
        html.P([
            html.Strong("Greedy vs Optimal: "),
            "The greedy heuristic runs in under 200ms but achieves about 75% of the optimal "
            "MinCost solver's cost reduction under specialty constraints. The reference comparison "
            "at the bottom shows the gap.",
        ], style={"margin": "0", "color": "#1e3a5f", "fontSize": "14px"}),
    ]),

    # Capacity sliders
    section_header("Daily Follow-Up Slots Per Specialty"),
    html.P(
        "How many outpatient follow-up appointments can each specialty handle per day? "
        "These numbers are scaled proportionally so total 30-day capacity matches the cohort size (27,641 patients). "
        "Reducing slots forces the scheduler to defer more patients to later days.",
        style={"color": "#9ca3af", "fontSize": "13px", "marginBottom": "16px"},
    ),

    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginBottom": "24px"}, children=[
        html.Div([
            html.Label("Cardiology", style={"fontWeight": "bold", "color": "#ef4444"}),
            dcc.Slider(id="slider-cardiology", min=1, max=100, step=1, value=15,
                       marks={1: "1", 15: "15", 25: "25", 50: "50", 75: "75", 100: "100"},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ]),
        html.Div([
            html.Label("Neurology", style={"fontWeight": "bold", "color": "#8b5cf6"}),
            dcc.Slider(id="slider-neurology", min=1, max=100, step=1, value=10,
                       marks={1: "1", 10: "10", 25: "25", 50: "50", 75: "75", 100: "100"},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ]),
        html.Div([
            html.Label("Surgery", style={"fontWeight": "bold", "color": "#f59e0b"}),
            dcc.Slider(id="slider-surgery", min=1, max=100, step=1, value=15,
                       marks={1: "1", 15: "15", 25: "25", 50: "50", 75: "75", 100: "100"},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ]),
        html.Div([
            html.Label("General Medicine", style={"fontWeight": "bold", "color": "#3b82f6"}),
            dcc.Slider(id="slider-general", min=1, max=100, step=1, value=25,
                       marks={1: "1", 25: "25", 50: "50", 75: "75", 100: "100"},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ]),
    ]),

    html.Div(style={"textAlign": "center", "marginBottom": "24px"}, children=[
        html.Button(
            "Run Scheduler", id="run-btn",
            style={
                "padding": "12px 32px", "backgroundColor": "#2563eb", "color": "white",
                "border": "none", "borderRadius": "6px", "cursor": "pointer",
                "fontSize": "16px", "fontWeight": "bold",
            },
        ),
        html.Span(id="elapsed-text", style={"marginLeft": "16px", "color": "#9ca3af"}),
    ]),

    # Results metrics
    html.Div(id="schedule-metrics", style={
        "display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "24px",
    }),

    # Day histogram
    section_header("Assignment Distribution"),
    dcc.Graph(id="schedule-histogram", figure={
        "data": [],
        "layout": {
            "xaxis": {"title": "Day After Discharge", "range": [0.5, 30.5], "dtick": 5},
            "yaxis": {"title": "Number of Patients", "range": [0, 100]},
            "annotations": [{
                "text": "Click 'Run Scheduler' to see the assignment distribution",
                "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,
                "showarrow": False, "font": {"size": 16, "color": "#9ca3af"},
            }],
            "plot_bgcolor": "white",
            "margin": {"t": 20},
        },
    }),

    # Comparison note
    html.Div(id="comparison-note", style={
        "marginTop": "16px", "padding": "16px", "backgroundColor": "#f0fdf4",
        "borderRadius": "8px", "borderLeft": "4px solid #16a34a",
    }),
])


@callback(
    Output("schedule-metrics", "children"),
    Output("schedule-histogram", "figure"),
    Output("elapsed-text", "children"),
    Output("comparison-note", "children"),
    Input("run-btn", "n_clicks"),
    State("slider-cardiology", "value"),
    State("slider-neurology", "value"),
    State("slider-surgery", "value"),
    State("slider-general", "value"),
    prevent_initial_call=True,
)
def run_scheduling(n_clicks, card, neuro, surg, gen):
    empty = {"data": [], "layout": {"title": "Click 'Run Scheduler' to see results"}}

    try:
        resp = requests.post(f"{API_URL}/api/schedule", json={
            "capacity": {
                "cardiology": card,
                "neurology": neuro,
                "surgery": surg,
                "general_medicine": gen,
            },
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return [], empty, f"Error: {e}", ""

    m = data["metrics"]

    metrics = [
        metric_card("Avg Cost", f"EUR {m['avg_cost']:,.0f}", f"{m['vs_uniform_pct']:+.1f}% vs uniform",
                     "#16a34a" if m["vs_uniform_pct"] < 0 else "#dc2626"),
        metric_card("Catch Rate", f"{m['catch_rate']:.1f}%", "events caught before follow-up", "#2563eb"),
        metric_card("EBF Rate", f"{m['ebf_rate']:.1f}%",
                     "readmissions that occurred before the scheduled follow-up (lower is better)", "#d97706"),
    ]

    fig = day_histogram_figure(
        data["day_histogram"],
        by_specialty=data.get("by_specialty"),
        title=f"Assignment Distribution (Card: {card} / Neuro: {neuro} / Surg: {surg} / GenMed: {gen})",
    )

    elapsed = f"Computed in {m['elapsed_ms']:.0f}ms"

    # Fetch pre-computed MinCost reference from API
    ref_cost, ref_catch = "N/A", "N/A"
    try:
        ref_resp = requests.get(f"{API_URL}/api/results", timeout=5)
        if ref_resp.ok:
            policies = ref_resp.json().get("policies", [])
            mincost = next((p for p in policies if p["name"] == "MinCost (specialty)"), None)
            if mincost:
                ref_cost = f"EUR {mincost['avg_cost']:,.0f}"
                ref_catch = f"{mincost['catch_rate']:.1f}%"
    except Exception:
        pass

    note = html.Div([
        html.Strong("Reference: "),
        html.Span(
            f"The pre-computed optimal MinCost (specialty) achieves {ref_cost} avg cost "
            f"and {ref_catch} catch rate. The greedy heuristic here achieves "
            f"EUR {m['avg_cost']:,.0f} ({m['catch_rate']:.1f}% catch rate), "
            f"showing the trade-off between computation speed ({m['elapsed_ms']:.0f}ms) "
            f"and optimality."
        ),
    ])

    return metrics, fig, elapsed, note
