"""Page 2: Patient Explorer - individual S(t) curves and assignments."""

import os
import random

import dash
import requests
from dash import html, dcc, callback, Input, Output, State

dash.register_page(__name__, path="/patients", name="Patient Explorer", order=1)

from dashboard.components.charts import survival_curve_figure
from dashboard.components.layout import section_header

API_URL = os.environ.get("CCPFS_API_URL", "http://localhost:8000")

MAX_PATIENT = 27640  # updated at runtime via _get_max_patient()


def _get_max_patient() -> int:
    try:
        resp = requests.get(f"{API_URL}/api/patients/count", timeout=5)
        resp.raise_for_status()
        return resp.json()["count"] - 1
    except Exception:
        return MAX_PATIENT


def _initial_pair():
    mx = _get_max_patient()
    a = random.randint(0, mx)
    b = random.randint(0, mx)
    while b == a:
        b = random.randint(0, mx)
    return a, b


_default_a, _default_b = _initial_pair()


layout = html.Div([
    html.H2("Patient Explorer", style={"textAlign": "center", "color": "#1f2937"}),

    # Explainer
    html.Div(style={
        "backgroundColor": "#f0f9ff", "borderRadius": "8px", "padding": "16px",
        "borderLeft": "4px solid #2563eb", "marginBottom": "24px",
    }, children=[
        html.P([
            html.Strong("What does this page show? "),
            "Each patient has a unique survival curve S(t) that tracks their probability of "
            "remaining readmission-free over 30 days. The curve starts at 1.0 (just discharged) "
            "and decreases as risk accumulates. The ",
            html.Strong("shaded area"),
            " represents risk exposure before the follow-up appointment.",
        ], style={"margin": "0 0 8px 0", "color": "#1e3a5f", "fontSize": "14px"}),
        html.P([
            html.Strong("Key insight: "),
            "The framework does not simply schedule the highest-risk patients first. Instead, "
            "it prioritises patients with the ",
            html.Em("steepest risk trajectories"),
            " (largest risk spread between early and late days), because these patients "
            "benefit most from being seen sooner. Try clicking ",
            html.Strong("Random Pair"),
            " to explore different cases.",
        ], style={"margin": "0", "color": "#1e3a5f", "fontSize": "14px"}),
    ]),

    section_header("Select Patients to Compare"),
    html.P(
        "Compare two patients side-by-side to see how the framework assigns "
        "earlier appointments to patients with steeper risk trajectories, "
        "even if their absolute risk is lower.",
        style={"color": "#6b7280", "marginBottom": "16px"},
    ),

    # Store the actual max patient index (fetched from API)
    dcc.Store(id="max-patient-store", data=MAX_PATIENT),

    html.Div(style={"display": "flex", "gap": "16px", "alignItems": "center", "marginBottom": "16px"}, children=[
        html.Div(style={"flex": "1"}, children=[
            html.Label("Patient A:", style={"fontWeight": "bold"}),
            html.Span(" (1 to 27,641)", style={"color": "#9ca3af", "fontSize": "12px"}),
            dcc.Input(id="patient-a-input", type="number", value=_default_a, min=0, max=MAX_PATIENT,
                      style={"width": "100%", "padding": "8px", "borderRadius": "4px", "border": "1px solid #d1d5db"}),
        ]),
        html.Div(style={"flex": "1"}, children=[
            html.Label("Patient B:", style={"fontWeight": "bold"}),
            html.Span(" (1 to 27,641)", style={"color": "#9ca3af", "fontSize": "12px"}),
            dcc.Input(id="patient-b-input", type="number", value=_default_b, min=0, max=MAX_PATIENT,
                      style={"width": "100%", "padding": "8px", "borderRadius": "4px", "border": "1px solid #d1d5db"}),
        ]),
        html.Div(style={"paddingTop": "20px"}, children=[
            html.Button("Random Pair", id="random-btn",
                        style={"padding": "8px 16px", "backgroundColor": "#2563eb", "color": "white",
                               "border": "none", "borderRadius": "4px", "cursor": "pointer"}),
        ]),
    ]),

    # Side-by-side curves
    html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
        html.Div(style={"flex": "1", "minWidth": "400px"}, children=[
            dcc.Graph(id="curve-a"),
            html.Div(id="info-a", style={"padding": "8px", "backgroundColor": "#f9fafb", "borderRadius": "4px"}),
        ]),
        html.Div(style={"flex": "1", "minWidth": "400px"}, children=[
            dcc.Graph(id="curve-b"),
            html.Div(id="info-b", style={"padding": "8px", "backgroundColor": "#f9fafb", "borderRadius": "4px"}),
        ]),
    ]),

    # Marginal benefit explanation
    html.Div(id="comparison-insight", style={
        "marginTop": "16px", "padding": "16px", "backgroundColor": "#eff6ff",
        "borderRadius": "8px", "borderLeft": "4px solid #2563eb",
    }),
])


def _fetch_patient(index: int) -> dict | None:
    try:
        resp = requests.get(f"{API_URL}/api/patients/{index}/curve", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


@callback(
    Output("patient-a-input", "value"),
    Output("patient-b-input", "value"),
    Input("random-btn", "n_clicks"),
    prevent_initial_call=True,
)
def random_pair(_):
    mx = _get_max_patient()
    a = random.randint(0, mx)
    b = random.randint(0, mx)
    while b == a:
        b = random.randint(0, mx)
    return a, b


@callback(
    Output("curve-a", "figure"),
    Output("curve-b", "figure"),
    Output("info-a", "children"),
    Output("info-b", "children"),
    Output("comparison-insight", "children"),
    Input("patient-a-input", "value"),
    Input("patient-b-input", "value"),
)
def update_curves(idx_a, idx_b):
    empty = {"data": [], "layout": {"title": "Enter a patient index"}}

    pa = _fetch_patient(idx_a) if idx_a is not None else None
    pb = _fetch_patient(idx_b) if idx_b is not None else None

    def make_fig(p):
        if not p:
            return empty
        return survival_curve_figure(
            p["survival_curve"], p["assigned_day"],
            p["event_indicator"], p["time_to_event"],
            p["patient_index"], p["specialty"],
        )

    def make_info(p):
        if not p:
            return "No data"
        if p["event_indicator"]:
            caught = p["time_to_event"] > p["assigned_day"]
            event_str = f"Readmitted on day {p['time_to_event']:.0f}"
            if caught:
                event_str += f" - follow-up on day {p['assigned_day']} would have occurred first"
            else:
                event_str += f" - readmitted before day {p['assigned_day']} follow-up"
            event_color = "#16a34a" if caught else "#dc2626"
        else:
            event_str = "No readmission within 30 days"
            event_color = "#6b7280"
        risk_30 = (1 - p["survival_curve"][30]) * 100 if len(p["survival_curve"]) > 30 else 0
        caught_icon = "✓" if p["event_indicator"] and p["time_to_event"] > p["assigned_day"] else "✗" if p["event_indicator"] else "✓"
        if not p["event_indicator"]:
            event_color = "#16a34a"
        return html.Div([
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "4px 16px", "lineHeight": "1.8"}, children=[
                html.Div([html.Strong("Specialty: "), p['specialty'].replace('_', ' ').title()]),
                html.Div([html.Strong("30-day risk: "), f"{risk_30:.1f}%"]),
                html.Div([html.Strong("Follow-up: "), f"day {p['assigned_day']}"]),
                html.Div([html.Strong("Cost: "), f"EUR {p['cost']:,.0f}"]),
            ]),
            html.Div(style={
                "marginTop": "8px", "paddingTop": "8px",
                "borderTop": "1px solid #e5e7eb",
                "display": "flex", "alignItems": "center", "gap": "8px",
            }, children=[
                html.Span(caught_icon, style={
                    "fontSize": "16px", "fontWeight": "bold", "color": event_color,
                }),
                html.Span([html.Strong("Outcome: "), event_str], style={"color": "#1f2937"}),
            ]),
        ])

    fig_a = make_fig(pa)
    fig_b = make_fig(pb)
    info_a = make_info(pa)
    info_b = make_info(pb)

    # Generate comparison insight
    insight = ""
    if pa and pb:
        # Marginal benefit: risk spread between day 1 and day 30 (as percentage points)
        def marginal_risk_spread(p):
            c = p["survival_curve"]
            if len(c) > 30:
                return ((1 - c[30]) - (1 - c[1])) * 100  # percentage points
            return 0

        mb_a = marginal_risk_spread(pa)
        mb_b = marginal_risk_spread(pb)
        risk_a = (1 - pa["survival_curve"][30]) * 100 if len(pa["survival_curve"]) > 30 else 0
        risk_b = (1 - pb["survival_curve"][30]) * 100 if len(pb["survival_curve"]) > 30 else 0

        insight = html.Div([
            html.Strong("Marginal Benefit Analysis: "),
            html.Span(
                f"Patient A has {risk_a:.1f}% 30-day risk (risk spread: {mb_a:.1f}pp), "
                f"assigned to day {pa['assigned_day']}. "
                f"Patient B has {risk_b:.1f}% 30-day risk (risk spread: {mb_b:.1f}pp), "
                f"assigned to day {pb['assigned_day']}. "
            ),
            html.Span(
                "The framework prioritises patients with larger risk spreads "
                "(steeper trajectories), not necessarily higher absolute risk.",
                style={"fontStyle": "italic"},
            ),
        ])

    return fig_a, fig_b, info_a, info_b, insight
