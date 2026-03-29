"""CCPFS Dashboard - Plotly Dash application entry point."""

import dash
from dash import Dash, html, dcc, page_container, callback, Input, Output

app = Dash(
    __name__,
    use_pages=True,
    pages_folder="pages",
    suppress_callback_exceptions=True,
    title="CCPFS Dashboard",
)

NAV_STYLE = {
    "color": "#9ca3af", "textDecoration": "none",
    "padding": "8px 16px", "borderRadius": "6px",
    "fontSize": "14px", "fontWeight": "500",
    "transition": "all 0.15s ease",
}

NAV_STYLE_ACTIVE = {
    **NAV_STYLE,
    "color": "white", "backgroundColor": "#374151",
}

app.layout = html.Div(
    style={"fontFamily": "system-ui, -apple-system, sans-serif", "backgroundColor": "#f3f4f6", "minHeight": "100vh"},
    children=[
        dcc.Location(id="url", refresh=False),

        # Navigation bar
        html.Nav(
            style={
                "backgroundColor": "#1f2937", "padding": "10px 24px",
                "display": "flex", "alignItems": "center", "gap": "24px",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.2)",
            },
            children=[
                # Logo
                dcc.Link(
                    href="/",
                    style={"textDecoration": "none", "display": "flex", "alignItems": "center", "gap": "0"},
                    children=[
                        html.Span("CC", style={
                            "color": "#60a5fa", "fontSize": "20px", "fontWeight": "800",
                            "letterSpacing": "1px", "fontFamily": "'Courier New', monospace",
                        }),
                        html.Span("PFS", style={
                            "color": "white", "fontSize": "20px", "fontWeight": "800",
                            "letterSpacing": "1px", "fontFamily": "'Courier New', monospace",
                        }),
                    ],
                ),

                # Separator
                html.Span("|", style={"color": "#4b5563", "fontSize": "20px", "fontWeight": "300"}),

                # Nav links (updated by callback)
                html.Div(
                    id="nav-links",
                    style={"display": "flex", "gap": "4px"},
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


@callback(
    Output("nav-links", "children"),
    Input("url", "pathname"),
)
def update_nav(pathname):
    if pathname is None:
        pathname = "/"
    links = []
    for page in dash.page_registry.values():
        is_active = pathname == page["path"]
        style = NAV_STYLE_ACTIVE if is_active else NAV_STYLE
        links.append(
            dcc.Link(page["name"], href=page["path"], style=style)
        )
    return links

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 3000))
    debug = os.environ.get("DASH_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
