"""Page 3: Interactive Scheduler - simulate capacity vs demand scenarios."""

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
            "Simulate how the scheduling framework performs under different hospital scenarios. "
            "Set your clinic's daily follow-up capacity (supply) and daily patient discharge volume (demand), "
            "then click ",
            html.Strong("Run Scheduler"),
            " to see how the greedy heuristic allocates appointments in real time.",
        ], style={"margin": "0 0 8px 0", "color": "#1e3a5f", "fontSize": "14px"}),
        html.P([
            html.Strong("What to try: "),
            "Start with defaults (supply > demand = everyone gets scheduled). Then increase patient volume "
            "or reduce capacity to simulate pressure - watch costs rise and catch rate drop as patients "
            "compete for limited early slots. The framework uses real patient risk curves from MIMIC-IV "
            "to produce realistic scheduling outcomes.",
        ], style={"margin": "0 0 8px 0", "color": "#1e3a5f", "fontSize": "14px"}),
        html.P([
            html.Strong("Supply vs Demand: "),
            "When total 30-day slots exceed patient count, everyone is scheduled within capacity. "
            "When demand exceeds supply, overflow patients are deferred - the framework prioritises "
            "those with the steepest risk trajectories for the limited early slots.",
        ], style={"margin": "0", "color": "#1e3a5f", "fontSize": "14px"}),
    ]),

    # Two-column layout: Supply (left) | Demand (right)
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "24px", "marginBottom": "24px"}, children=[

        # LEFT: Supply (capacity)
        html.Div([
            section_header("Supply: Daily Follow-Up Slots"),
            html.P("How many follow-up appointments can each specialty handle per day?",
                   style={"color": "#9ca3af", "fontSize": "13px", "marginBottom": "12px"}),
            html.Div([
                html.Label("Cardiology (slots/day)", style={"fontWeight": "bold", "color": "#ef4444"}),
                dcc.Slider(id="slider-cap-cardiology", min=1, max=100, step=1, value=15,
                           marks={1: "1", 15: "15", 50: "50", 100: "100"},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"marginBottom": "8px"}),
            html.Div([
                html.Label("Neurology (slots/day)", style={"fontWeight": "bold", "color": "#8b5cf6"}),
                dcc.Slider(id="slider-cap-neurology", min=1, max=100, step=1, value=10,
                           marks={1: "1", 10: "10", 50: "50", 100: "100"},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"marginBottom": "8px"}),
            html.Div([
                html.Label("Surgery (slots/day)", style={"fontWeight": "bold", "color": "#f59e0b"}),
                dcc.Slider(id="slider-cap-surgery", min=1, max=100, step=1, value=15,
                           marks={1: "1", 15: "15", 50: "50", 100: "100"},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"marginBottom": "8px"}),
            html.Div([
                html.Label("General Medicine (slots/day)", style={"fontWeight": "bold", "color": "#3b82f6"}),
                dcc.Slider(id="slider-cap-general", min=1, max=100, step=1, value=25,
                           marks={1: "1", 25: "25", 50: "50", 100: "100"},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ]),
        ]),

        # RIGHT: Demand (patient volume)
        html.Div([
            section_header("Demand: Daily Patient Discharges"),
            html.P("How many patients are discharged per day needing follow-up?",
                   style={"color": "#9ca3af", "fontSize": "13px", "marginBottom": "12px"}),
            html.Div([
                html.Label("Cardiology (patients/day)", style={"fontWeight": "bold", "color": "#ef4444"}),
                dcc.Slider(id="slider-dem-cardiology", min=1, max=100, step=1, value=10,
                           marks={1: "1", 10: "10", 50: "50", 100: "100"},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"marginBottom": "8px"}),
            html.Div([
                html.Label("Neurology (patients/day)", style={"fontWeight": "bold", "color": "#8b5cf6"}),
                dcc.Slider(id="slider-dem-neurology", min=1, max=100, step=1, value=7,
                           marks={1: "1", 7: "7", 50: "50", 100: "100"},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"marginBottom": "8px"}),
            html.Div([
                html.Label("Surgery (patients/day)", style={"fontWeight": "bold", "color": "#f59e0b"}),
                dcc.Slider(id="slider-dem-surgery", min=1, max=100, step=1, value=5,
                           marks={1: "1", 5: "5", 50: "50", 100: "100"},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"marginBottom": "8px"}),
            html.Div([
                html.Label("General Medicine (patients/day)", style={"fontWeight": "bold", "color": "#3b82f6"}),
                dcc.Slider(id="slider-dem-general", min=1, max=100, step=1, value=20,
                           marks={1: "1", 20: "20", 50: "50", 100: "100"},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ]),
        ]),
    ]),

    # Supply vs Demand summary (updated live)
    html.Div(id="supply-demand-summary", style={
        "backgroundColor": "#f9fafb", "borderRadius": "8px", "padding": "12px 16px",
        "marginBottom": "16px", "fontSize": "13px", "color": "#374151",
    }),

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
    Output("supply-demand-summary", "children"),
    Input("slider-cap-cardiology", "value"),
    Input("slider-cap-neurology", "value"),
    Input("slider-cap-surgery", "value"),
    Input("slider-cap-general", "value"),
    Input("slider-dem-cardiology", "value"),
    Input("slider-dem-neurology", "value"),
    Input("slider-dem-surgery", "value"),
    Input("slider-dem-general", "value"),
)
def update_summary(cap_c, cap_n, cap_s, cap_g, dem_c, dem_n, dem_s, dem_g):
    supply_daily = cap_c + cap_n + cap_s + cap_g
    demand_daily = dem_c + dem_n + dem_s + dem_g
    supply_30 = supply_daily * 30
    demand_30 = demand_daily * 30

    specs = [
        ("Cardiology", cap_c, dem_c, "#ef4444"),
        ("Neurology", cap_n, dem_n, "#8b5cf6"),
        ("Surgery", cap_s, dem_s, "#f59e0b"),
        ("Gen. Medicine", cap_g, dem_g, "#3b82f6"),
    ]

    items = []
    for name, cap, dem, color in specs:
        if cap >= dem:
            status = "OK"
            status_color = "#16a34a"
        else:
            overflow_pct = (dem - cap) / dem * 100
            status = f"{overflow_pct:.0f}% overflow"
            status_color = "#dc2626"
        items.append(
            html.Div([
                html.Span(f"{name}: ", style={"color": color, "fontWeight": "bold"}),
                html.Span(f"{cap} slots vs {dem} patients/day "),
                html.Span(f"({status})", style={"color": status_color, "fontWeight": "bold"}),
            ]),
        )

    if supply_daily >= demand_daily:
        overall = f"Sufficient ({supply_daily} slots >= {demand_daily} patients/day)"
        overall_color = "#16a34a"
    else:
        overall = f"Under capacity ({supply_daily} slots < {demand_daily} patients/day, {demand_30 - supply_30:,} patients deferred over 30 days)"
        overall_color = "#dc2626"

    return html.Div([
        html.Div([
            html.Span("Supply vs Demand: ", style={"fontWeight": "bold"}),
            html.Span(overall, style={"color": overall_color, "fontWeight": "bold"}),
        ], style={"marginBottom": "8px"}),
        html.Div(items, style={
            "display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "4px 24px",
        }),
    ])


@callback(
    Output("schedule-metrics", "children"),
    Output("schedule-histogram", "figure"),
    Output("elapsed-text", "children"),
    Output("comparison-note", "children"),
    Input("run-btn", "n_clicks"),
    State("slider-cap-cardiology", "value"),
    State("slider-cap-neurology", "value"),
    State("slider-cap-surgery", "value"),
    State("slider-cap-general", "value"),
    State("slider-dem-cardiology", "value"),
    State("slider-dem-neurology", "value"),
    State("slider-dem-surgery", "value"),
    State("slider-dem-general", "value"),
    prevent_initial_call=True,
)
def run_scheduling(n_clicks, cap_c, cap_n, cap_s, cap_g, dem_c, dem_n, dem_s, dem_g):
    empty = {"data": [], "layout": {"title": "Click 'Run Scheduler' to see results"}}

    try:
        resp = requests.post(f"{API_URL}/api/schedule", json={
            "capacity": {
                "cardiology": cap_c,
                "neurology": cap_n,
                "surgery": cap_s,
                "general_medicine": cap_g,
            },
            "patients_per_day": {
                "cardiology": dem_c,
                "neurology": dem_n,
                "surgery": dem_s,
                "general_medicine": dem_g,
            },
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return [], empty, f"Error: {e}", ""

    m = data["metrics"]
    n_overflow = data.get("overflow_count", 0)
    n_scheduled = data.get("scheduled_within_capacity", 0)
    n_total = data.get("total_patients", 0)

    capacity_note = f"{n_scheduled:,} of {n_total:,} within capacity"
    if n_overflow > 0:
        capacity_note += f", {n_overflow:,} deferred ({n_overflow/n_total*100:.0f}%)"

    metrics = [
        metric_card("Avg Cost", f"EUR {m['avg_cost']:,.0f}",
                     f"{m['vs_uniform14_pct']:+.1f}% vs fixed day-14 for all",
                     "#16a34a" if m["vs_uniform14_pct"] < 0 else "#dc2626"),
        metric_card("vs Day-30", f"{m['vs_uniform30_pct']:+.1f}%",
                     "vs fixed day-30 for all",
                     "#16a34a" if m["vs_uniform30_pct"] < 0 else "#dc2626"),
        metric_card("Catch Rate", f"{m['catch_rate']:.1f}%", "events caught before follow-up", "#2563eb"),
        metric_card("Patients", f"{n_total:,}", capacity_note,
                     "#16a34a" if n_overflow == 0 else "#d97706"),
    ]

    total_daily_cap = cap_c + cap_n + cap_s + cap_g
    fig = day_histogram_figure(
        data["day_histogram"],
        by_specialty=data.get("by_specialty"),
        total_capacity_per_day=total_daily_cap,
        title=f"Scheduled Within Capacity ({n_scheduled:,} patients across 30 days)",
    )

    elapsed = f"Computed in {m['elapsed_ms']:.0f}ms"

    # Summary note
    demand_daily = dem_c + dem_n + dem_s + dem_g
    note = html.Div([
        html.Strong("Simulation summary: "),
        html.Span(
            f"{demand_daily} patients/day x 30 days = {n_total:,} patients scheduled against "
            f"{total_daily_cap} slots/day. "
        ),
        html.Span(
            f"The greedy heuristic achieves EUR {m['avg_cost']:,.0f} avg cost "
            f"({m['catch_rate']:.1f}% catch rate) in {m['elapsed_ms']:.0f}ms. "
        ),
        html.Span(
            f"{n_scheduled:,} patients scheduled within capacity"
            + (f", {n_overflow:,} deferred beyond 30-day window." if n_overflow > 0 else ".")
        ),
    ])

    return metrics, fig, elapsed, note
