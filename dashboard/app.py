"""CCPFS Dashboard - Plotly Dash application entry point."""

import dash
from dash import Dash, html, dcc, page_container

app = Dash(
    __name__,
    use_pages=True,
    pages_folder="pages",
    suppress_callback_exceptions=True,
    title="CCPFS Dashboard",
)

app.layout = html.Div(
    style={"fontFamily": "system-ui, -apple-system, sans-serif", "backgroundColor": "#f3f4f6", "minHeight": "100vh"},
    children=[
        # Navigation bar
        html.Nav(
            style={
                "backgroundColor": "#1f2937", "padding": "12px 24px",
                "display": "flex", "alignItems": "center", "gap": "24px",
            },
            children=[
                html.Span("CCPFS", style={"color": "white", "fontSize": "20px", "fontWeight": "bold"}),
                html.Div(
                    style={"display": "flex", "gap": "8px"},
                    children=[
                        dcc.Link(
                            page["name"],
                            href=page["path"],
                            style={
                                "color": "#d1d5db", "textDecoration": "none",
                                "padding": "6px 12px", "borderRadius": "4px",
                                "fontSize": "14px",
                            },
                            className="nav-link",
                        )
                        for page in dash.page_registry.values()
                    ],
                ),
            ],
        ),
        # Page content
        html.Div(
            style={"maxWidth": "1200px", "margin": "0 auto", "padding": "24px"},
            children=[page_container],
        ),
    ],
)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 3000))
    debug = os.environ.get("DASH_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
