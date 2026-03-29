"""Common layout elements for the CCPFS dashboard."""

from dash import html


def metric_card(title: str, value: str, subtitle: str = "", color: str = "#2563eb"):
    """Render a hero metric card."""
    return html.Div(
        style={
            "background": "white",
            "borderRadius": "8px",
            "padding": "20px",
            "textAlign": "center",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.12)",
            "borderTop": f"4px solid {color}",
            "flex": "1",
            "minWidth": "180px",
        },
        children=[
            html.H4(title, style={"color": "#6b7280", "margin": "0 0 8px 0", "fontSize": "14px"}),
            html.H2(value, style={"color": color, "margin": "0 0 4px 0", "fontSize": "28px"}),
            html.P(subtitle, style={"color": "#9ca3af", "margin": "0", "fontSize": "12px"}),
        ],
    )


def section_header(text: str):
    """Render a section divider."""
    return html.H3(
        text,
        style={
            "borderBottom": "2px solid #e5e7eb",
            "paddingBottom": "8px",
            "marginTop": "24px",
            "color": "#1f2937",
        },
    )
