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


layout = html.Div([
    html.H2("Patient Explorer", style={"textAlign": "center", "color": "#1f2937"}),

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
            html.Label("Patient A (index):", style={"fontWeight": "bold"}),
            dcc.Input(id="patient-a-input", type="number", value=42, min=0, max=MAX_PATIENT,
                      style={"width": "100%", "padding": "8px", "borderRadius": "4px", "border": "1px solid #d1d5db"}),
        ]),
        html.Div(style={"flex": "1"}, children=[
            html.Label("Patient B (index):", style={"fontWeight": "bold"}),
            dcc.Input(id="patient-b-input", type="number", value=100, min=0, max=MAX_PATIENT,
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
        event_str = f"Readmitted day {p['time_to_event']:.0f}" if p["event_indicator"] else "No readmission"
        return html.Div([
            html.Span(f"Specialty: {p['specialty'].replace('_', ' ').title()}", style={"marginRight": "16px"}),
            html.Span(f"Assigned day: {p['assigned_day']}", style={"marginRight": "16px", "fontWeight": "bold"}),
            html.Span(f"Cost: EUR {p['cost']:,.0f}", style={"marginRight": "16px"}),
            html.Span(event_str),
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
