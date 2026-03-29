"""Reusable Plotly figure builders for the CCPFS dashboard."""

import plotly.graph_objects as go


# Consistent color palette
COLORS = {
    "primary": "#2563eb",      # blue-600
    "success": "#16a34a",      # green-600
    "warning": "#d97706",      # amber-600
    "danger": "#dc2626",       # red-600
    "muted": "#9ca3af",        # gray-400
    "cardiology": "#ef4444",   # red-500
    "neurology": "#8b5cf6",    # violet-500
    "surgery": "#f59e0b",      # amber-500
    "general_medicine": "#3b82f6",  # blue-500
}

SPECIALTY_COLORS = {
    "cardiology": COLORS["cardiology"],
    "neurology": COLORS["neurology"],
    "surgery": COLORS["surgery"],
    "general_medicine": COLORS["general_medicine"],
}


def policy_comparison_figure(policies: list[dict]) -> go.Figure:
    """Grouped bar chart: cost and catch rate for all policies."""
    names = [p["name"] for p in policies]
    costs = [p["avg_cost"] for p in policies]
    catch_rates = [p["catch_rate"] for p in policies]
    feasible = [p["feasible"] for p in policies]

    # Color bars by feasibility
    cost_colors = [COLORS["primary"] if f else COLORS["muted"] for f in feasible]
    catch_colors = [COLORS["success"] if f else COLORS["muted"] for f in feasible]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Avg Cost (EUR)",
        x=names, y=costs,
        marker_color=cost_colors,
        yaxis="y",
        text=[f"{c:,.0f}" for c in costs],
        textposition="outside",
        textfont_size=10,
    ))

    fig.add_trace(go.Scatter(
        name="Catch Rate (%)",
        x=names, y=catch_rates,
        mode="lines+markers",
        marker=dict(size=10, color=COLORS["success"]),
        line=dict(width=2, color=COLORS["success"]),
        yaxis="y2",
        text=[f"{c:.1f}%" for c in catch_rates],
        textposition="top center",
    ))

    fig.update_layout(
        title="Scheduling Policy Comparison",
        xaxis=dict(tickangle=-35),
        yaxis=dict(title="Avg Cost (EUR)", side="left"),
        yaxis2=dict(title="Catch Rate (%)", side="right", overlaying="y", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=450,
        margin=dict(b=120),
        bargap=0.3,
    )

    return fig


def survival_curve_figure(curve: list[float], assigned_day: int,
                          event_indicator: bool, time_to_event: float,
                          patient_index: int, specialty: str) -> go.Figure:
    """Individual patient S(t) curve with assignment marker."""
    days = list(range(len(curve)))

    fig = go.Figure()

    # S(t) curve
    fig.add_trace(go.Scatter(
        x=days, y=curve,
        mode="lines",
        name="S(t)",
        line=dict(width=3, color=COLORS["primary"]),
        fill="tozeroy",
        fillcolor="rgba(37, 99, 235, 0.1)",
    ))

    # Risk exposure area: shade region between S(t) and 1.0 up to assigned day
    risk_days = list(range(assigned_day + 1))
    fig.add_trace(go.Scatter(
        x=risk_days + risk_days[::-1],
        y=[1.0] * len(risk_days) + [curve[d] for d in risk_days][::-1],
        fill="toself",
        fillcolor="rgba(220, 38, 38, 0.1)",
        line=dict(width=0),
        name="Risk exposure",
        showlegend=True,
        hoverinfo="skip",
    ))

    # Assigned day vertical line
    fig.add_vline(
        x=assigned_day, line_dash="dash", line_color=COLORS["success"], line_width=2,
        annotation_text=f"Follow-up: day {assigned_day}",
        annotation_position="top right",
    )

    # Event marker
    if event_indicator and time_to_event <= 30:
        event_day = round(time_to_event)
        event_surv = curve[min(event_day, len(curve) - 1)]
        color = COLORS["danger"] if event_day < assigned_day else COLORS["success"]
        label = "Missed" if event_day < assigned_day else "Caught"
        fig.add_trace(go.Scatter(
            x=[event_day], y=[event_surv],
            mode="markers+text",
            marker=dict(size=14, color=color, symbol="diamond"),
            text=[f"Event ({label})"],
            textposition="bottom center",
            name=f"Readmission (day {event_day})",
        ))

    color = SPECIALTY_COLORS.get(specialty, COLORS["primary"])
    fig.update_layout(
        title=f"Patient {patient_index} - {specialty.replace('_', ' ').title()}",
        xaxis_title="Days since discharge",
        yaxis_title="Survival probability S(t)",
        yaxis_range=[0, 1.05],
        xaxis_range=[0, 30],
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


def day_histogram_figure(day_hist: list[int],
                         by_specialty: dict[str, list[int]] | None = None,
                         title: str = "Patient Assignment Distribution") -> go.Figure:
    """Stacked bar chart of assignments across 30 days, colored by specialty."""
    days = list(range(1, len(day_hist) + 1))

    fig = go.Figure()

    if by_specialty:
        for name in ["cardiology", "neurology", "surgery", "general_medicine"]:
            if name in by_specialty:
                fig.add_trace(go.Bar(
                    name=name.replace("_", " ").title(),
                    x=days,
                    y=by_specialty[name],
                    marker_color=SPECIALTY_COLORS.get(name, COLORS["muted"]),
                ))
        fig.update_layout(barmode="stack")
    else:
        fig.add_trace(go.Bar(
            x=days, y=day_hist,
            marker_color=COLORS["primary"],
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Follow-up day",
        yaxis_title="Number of patients",
        height=350,
        showlegend=bool(by_specialty),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


def model_comparison_figure(models: list[dict]) -> go.Figure:
    """Horizontal bar chart comparing model C-indices."""
    names = [m["name"] for m in models]
    c_indices = [m["c_index"] for m in models]

    colors = [COLORS["primary"] if c == max(c_indices) else COLORS["muted"] for c in c_indices]

    fig = go.Figure(go.Bar(
        x=c_indices, y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{c:.4f}" for c in c_indices],
        textposition="outside",
    ))

    fig.update_layout(
        title="Survival Model Comparison (C-index)",
        xaxis_title="C-index (higher is better)",
        xaxis_range=[0.6, 0.75],
        height=250,
        margin=dict(l=120),
    )

    return fig
