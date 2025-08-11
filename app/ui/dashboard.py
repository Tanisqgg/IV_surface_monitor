from pathlib import Path
import os
import glob
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, dash_table, Input, Output
import plotly.express as px
import plotly.graph_objects as go

from app.utils.io import read_table
from app.surface import build_surface
from app.anomaly import calendar_violations, convexity_violations
from app.svi import fit_svi_smile, evaluate_svi_iv_on_grid

DATA_DIR = Path("app/data")


def latest_feat_iv():
    files = sorted(glob.glob(str(DATA_DIR / "*_feat_iv.parquet"))) + \
            sorted(glob.glob(str(DATA_DIR / "*_feat_iv.csv")))
    return Path(files[-1]) if files else None


def load_df(path: Path) -> pd.DataFrame:
    df = read_table(path)
    # keep reasonable rows
    if "iv_est" in df.columns:
        df = df[(df["T"] > 1e-6) & df["iv_est"].between(0.01, 5.0)].copy()
    return df


def make_smile_fig(df: pd.DataFrame, expiry: pd.Timestamp) -> go.Figure:
    sm = df[df["expiration"] == expiry].sort_values("k")
    if sm.empty:
        return go.Figure()
    hover = [c for c in ["strike", "right", "bid", "ask", "open_interest", "T"] if c in sm.columns]
    fig = px.scatter(sm, x="k", y="iv_est", hover_data=hover, title=f"Smile — {expiry}")
    fig.update_xaxes(title="log-moneyness k")
    fig.update_yaxes(title="IV")
    return fig


def make_surface_fig(k_grid: np.ndarray, T_grid: np.ndarray, IV: np.ndarray) -> go.Figure:
    K, TT = np.meshgrid(k_grid, T_grid)
    fig = go.Figure(data=[go.Surface(x=K, y=TT, z=IV)])
    fig.update_layout(
        title="IV Surface",
        scene=dict(xaxis_title="k", yaxis_title="T (years)", zaxis_title="IV"),
        height=650,
    )
    return fig


app = Dash(__name__)
app.title = "IV Surface Monitor"

initial = latest_feat_iv()
df0 = load_df(initial) if initial else pd.DataFrame()
expiries0 = sorted(df0["expiration"].unique()) if not df0.empty else []

app.layout = html.Div(
    [
        html.H2("IV Surface Monitor (Alpha Vantage – historical)"),

        html.Details([
            html.Summary("📖 How to Use", style={"cursor": "pointer", "fontWeight": "bold", "marginBottom": "8px"}),
            html.Div([
                html.P(
                    "1. Wait for the page to load. If you see 'Loading…', it may take 30–60 seconds for the server to start."),
                html.P("2. Use the 'File' dropdown to select a dataset (e.g., SPY_chain_YYYYMMDD_feat_iv.parquet)."),
                html.P("3. Pick an 'Expiry' date to view that day's smile."),
                html.P("4. Choose 'Surface Method':"),
                html.Ul([
                    html.Li("SVI (fitted) → Smooth, arbitrage-free curves."),
                    html.Li("Raw (interp) → Straight-line interpolation between quotes."),
                ]),
                html.P("5. Explore the tabs:"),
                html.Ul([
                    html.Li("Smile → 2D plot of IV vs log-moneyness for chosen expiry."),
                    html.Li("Surface → 3D IV surface across expiries."),
                    html.Li("Anomalies → Tables of calendar/convexity violations."),
                ]),
                html.P(
                    "💡 Tip: Hover over points to see strike, type (C/P), bid/ask, and open interest. Drag/zoom in 3D surface."),
            ], style={"fontSize": "14px", "lineHeight": "1.4em"})
        ], open=False),

        html.Div(
            [
                html.Div(
                    [
                        html.Label("File"),
                        dcc.Dropdown(
                            id="file-dd",
                            options=[
                                {"label": Path(p).name, "value": p}
                                for p in sorted(glob.glob(str(DATA_DIR / "*_feat_iv.*")))
                            ],
                            value=str(initial) if initial else None,
                            style={"minWidth": "420px"},
                        ),
                    ],
                    style={"marginRight": "20px"},
                ),
                html.Div(
                    [
                        html.Label("Expiry"),
                        dcc.Dropdown(
                            id="expiry-dd",
                            options=[{"label": str(e), "value": str(e)} for e in expiries0],
                            value=str(expiries0[0]) if expiries0 else None,
                            style={"minWidth": "240px"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Surface Method"),
                        dcc.RadioItems(
                            id="method-radio",
                            options=[
                                {"label": "Raw (interp)", "value": "interp"},
                                {"label": "SVI (fitted)", "value": "svi"},
                            ],
                            value="svi",
                            inline=True,
                        ),
                    ],
                    style={"marginLeft": "20px"},
                ),
                dcc.Interval(id="refresh", interval=60_000, n_intervals=0),
            ],
            style={"display": "flex", "gap": "16px", "alignItems": "end", "flexWrap": "wrap"},
        ),

        html.Hr(),

        dcc.Tabs(
            id="tabs",
            value="tab-smile",
            children=[
                dcc.Tab(label="Smile", value="tab-smile", children=[dcc.Graph(id="smile-fig")]),
                dcc.Tab(label="Surface", value="tab-surface", children=[dcc.Graph(id="surface-fig")]),
                dcc.Tab(
                    label="Anomalies",
                    value="tab-anoms",
                    children=[
                        html.Div(
                            [
                                html.H4("Calendar violations (w(T) should be non-decreasing in T)"),
                                dash_table.DataTable(id="cal-table", page_size=8),
                                html.Br(),
                                html.H4("Convexity violations (call price convex in strike)"),
                                dash_table.DataTable(id="conv-table", page_size=8),
                            ]
                        )
                    ],
                ),
            ],
        ),
    ],
    style={"padding": "14px 18px", "fontFamily": "Inter, system-ui, sans-serif"},
)


@app.callback(
    Output("file-dd", "options"),
    Output("file-dd", "value"),
    Input("refresh", "n_intervals"),
    prevent_initial_call=False,
)
def refresh_files(_):
    files = sorted(glob.glob(str(DATA_DIR / "*_feat_iv.*")))
    opts = [{"label": Path(p).name, "value": p} for p in files]
    val = files[-1] if files else None
    return opts, val


@app.callback(
    Output("expiry-dd", "options"),
    Output("expiry-dd", "value"),
    Input("file-dd", "value"),
)
def set_expiries(path):
    if not path:
        return [], None
    df = load_df(Path(path))
    exps = sorted(df["expiration"].unique())
    return (
        [{"label": str(e), "value": str(e)} for e in exps],
        (str(exps[0]) if exps else None),
    )


@app.callback(
    Output("smile-fig", "figure"),
    Output("surface-fig", "figure"),
    Output("cal-table", "data"),
    Output("cal-table", "columns"),
    Output("conv-table", "data"),
    Output("conv-table", "columns"),
    Input("file-dd", "value"),
    Input("expiry-dd", "value"),
    Input("method-radio", "value"),
)
def update_all(path, expiry, method):
    # Defaults
    empty_fig = go.Figure()
    empty_cols, empty_rows = [], []

    if not path:
        return empty_fig, empty_fig, empty_rows, empty_cols, empty_rows, empty_cols

    df = load_df(Path(path))
    if df.empty:
        return empty_fig, empty_fig, empty_rows, empty_cols, empty_rows, empty_cols

    # --- Smile ---
    try:
        exp_date = pd.to_datetime(expiry).date() if expiry else df["expiration"].min()
    except Exception:
        exp_date = df["expiration"].min()

    fig_smile = make_smile_fig(df, exp_date)

    # If SVI, overlay line
    if method == "svi":
        sm = df[df["expiration"] == exp_date].copy()
        if not sm.empty:
            T = float(sm["T"].median())
            sm = sm[(sm["iv_est"].between(0.01, 3.0))]
            k = sm["k"].to_numpy()
            w = (sm["iv_est"].to_numpy() ** 2) * T
            p = fit_svi_smile(k, w)
            if p.ok:
                k_line = np.linspace(float(np.nanmin(k)), float(np.nanmax(k)), 200)
                iv_line = evaluate_svi_iv_on_grid(k_line, T, p)
                fig_smile.add_scatter(x=k_line, y=iv_line, mode="lines", name="SVI fit")

    # --- Surface + anomalies ---
    k_grid, T_grid, IV, W, meta = build_surface(df, n_k=61, max_exps=8, method=method)
    fig_surf = make_surface_fig(k_grid, T_grid, IV)

    cal = calendar_violations(k_grid, T_grid, W, tol=1e-6)
    conv = convexity_violations(df, max_exps=8, tol=0.0)

    cal_cols = [{"name": c, "id": c} for c in cal.columns]
    conv_cols = [{"name": c, "id": c} for c in conv.columns]

    return (
        fig_smile,
        fig_surf,
        cal.to_dict("records"),
        cal_cols,
        conv.to_dict("records"),
        conv_cols,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8050"))  # Render sets PORT
    app.run(debug=False, host="0.0.0.0", port=port)
