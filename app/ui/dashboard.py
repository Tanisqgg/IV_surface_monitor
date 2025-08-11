from pathlib import Path
import os
import glob
import numpy as np
import pandas as pd
import dash
from dash import Dash, dcc, html, dash_table, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

from app.utils.io import read_table
from app.surface import build_surface
from app.anomaly import calendar_violations, convexity_violations
from app.svi import fit_svi_smile, evaluate_svi_iv_on_grid

DATA_DIR = Path("app/data")

# -----------------------------
# Data helpers
# -----------------------------
def latest_feat_iv():
    files = sorted(glob.glob(str(DATA_DIR / "*_feat_iv.parquet"))) + \
            sorted(glob.glob(str(DATA_DIR / "*_feat_iv.csv")))
    return Path(files[-1]) if files else None

def load_df(path: Path) -> pd.DataFrame:
    df = read_table(path)
    # keep reasonable rows
    if "iv_est" in df.columns:
        df = df[(df["T"] > 1e-6) & df["iv_est"].between(0.01, 5.0)].copy()
    if "expiration" in df.columns:
        df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
    # normalize helpers
    for c in ("open_interest", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# -----------------------------
# Figure helpers
# -----------------------------
def make_smile_fig(df: pd.DataFrame, expiry, show_svi: bool) -> go.Figure:
    sm = df[df["expiration"] == expiry].sort_values("k")
    if sm.empty:
        return go.Figure()
    hover = [c for c in ["strike","right","bid","ask","open_interest","volume","T",
                         "delta","gamma","theta","vega"] if c in sm.columns]
    fig = px.scatter(sm, x="k", y="iv_est", color=("right" if "right" in sm.columns else None),
                     hover_data=hover, title=f"Smile — {expiry}")
    fig.update_xaxes(title="log-moneyness k")
    fig.update_yaxes(title="IV")

    if show_svi and not sm.empty:
        T = float(sm["T"].median())
        k = sm["k"].to_numpy()
        w = (sm["iv_est"].to_numpy() ** 2) * T
        p = fit_svi_smile(k, w)
        if p.ok:
            k_line = np.linspace(float(np.nanmin(k)), float(np.nanmax(k)), 200)
            iv_line = evaluate_svi_iv_on_grid(k_line, T, p)
            fig.add_scatter(x=k_line, y=iv_line, mode="lines", name="SVI fit")
    return fig

def make_surface_fig_3d(k_grid: np.ndarray, T_grid: np.ndarray, IV: np.ndarray) -> go.Figure:
    K, TT = np.meshgrid(k_grid, T_grid)
    fig = go.Figure(data=[go.Surface(x=K, y=TT, z=IV)])
    fig.update_layout(
        title="IV Surface (3D)",
        scene=dict(xaxis_title="k", yaxis_title="T (years)", zaxis_title="IV"),
        height=650,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig

def make_surface_fig_heatmap(k_grid: np.ndarray, T_grid: np.ndarray, IV: np.ndarray) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(x=k_grid, y=T_grid, z=IV))
    fig.update_layout(
        title="IV Surface (heatmap)",
        xaxis_title="k",
        yaxis_title="T (years)",
        height=650,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig

def make_term_structure_fig(T_grid: np.ndarray, IV: np.ndarray, k_grid: np.ndarray, k_star: float) -> go.Figure:
    if len(k_grid) == 0 or IV.size == 0:
        return go.Figure()
    j = int(np.nanargmin(np.abs(k_grid - float(k_star))))
    iv_line = IV[:, j]
    fig = go.Figure()
    fig.add_scatter(x=T_grid, y=iv_line, mode="lines+markers", name=f"k≈{k_grid[j]:.3f}")
    fig.update_layout(title="Term structure (IV vs T) at fixed k",
                      xaxis_title="T (years)", yaxis_title="IV",
                      height=350, margin=dict(l=0, r=0, t=40, b=10))
    return fig

# -----------------------------
# App + Layout
# -----------------------------
app = Dash(__name__)
app.title = "IV Surface Monitor"

initial = latest_feat_iv()
df0 = load_df(initial) if initial else pd.DataFrame()
expiries0 = sorted(df0["expiration"].unique()) if not df0.empty else []

# Sidebar: How to Use
howto_panel = html.Div(
    [
        html.H3("How to Use", style={"marginTop": "0"}),
        html.Ul([
            html.Li("Pick a Data file (latest auto-selected)."),
            html.Li("Choose an Expiry to view its smile."),
            html.Li("Toggle Calls/Puts and adjust k/IV/LIQ filters."),
            html.Li("3D/Heatmap for surface; SVI fit optional."),
            html.Li("Set k* to see the IV term-structure line."),
            html.Li("Export anomalies or the current smile as CSV."),
        ], style={"lineHeight": "1.5", "margin": 0, "paddingLeft": "18px"})
    ],
    style={
        "backgroundColor": "#f9f9f9",
        "padding": "10px 12px",
        "border": "1px solid #ddd",
        "borderRadius": "6px",
        "marginBottom": "10px"
    }
)

# Sidebar controls (with trader QoL)
sidebar = html.Div(
    [
        howto_panel,

        html.H3("IV Surface Monitor", style={"marginBottom": "4px"}),
        html.Div("Alpha Vantage — historical", style={"opacity": 0.7, "fontSize": "12px"}),
        html.Hr(),

        html.Label("Data file"),
        dcc.Dropdown(
            id="file-dd",
            options=[{"label": Path(p).name, "value": p}
                     for p in sorted(glob.glob(str(DATA_DIR / "*_feat_iv.*")))],
            value=str(initial) if initial else None,
            style={"minWidth": "260px"},
        ),
        html.Br(),

        html.Label("Expiry"),
        dcc.Dropdown(
            id="expiry-dd",
            options=[{"label": str(e), "value": str(e)} for e in expiries0],
            value=str(expiries0[0]) if expiries0 else None,
            placeholder="Select expiry…",
            searchable=True,
        ),
        html.Br(),

        html.Label("Rights"),
        dcc.Checklist(
            id="rights-cb",
            options=[{"label": "Calls", "value": "C"}, {"label": "Puts", "value": "P"}],
            value=["C", "P"],
            inline=True,
        ),
        html.Br(),

        html.Label("k range (log-moneyness)"),
        dcc.RangeSlider(id="k-range", min=-1.0, max=1.0, step=0.01, value=[-0.5, 0.5],
                        tooltip={"placement": "bottom", "always_visible": False}),
        html.Br(),

        html.Label("IV range"),
        dcc.RangeSlider(id="iv-range", min=0.01, max=3.0, step=0.01, value=[0.05, 2.0],
                        tooltip={"placement": "bottom", "always_visible": False}),
        html.Br(),

        html.Label("Min Open Interest / Min Volume"),
        html.Div([
            dcc.Slider(id="oi-min", min=0, max=5000, step=10, value=0,
                       tooltip={"placement": "bottom", "always_visible": False}),
            dcc.Slider(id="vol-min", min=0, max=2000, step=10, value=0,
                       tooltip={"placement": "bottom", "always_visible": False}),
        ], style={"marginBottom": "8px"}),

        # Quick actions
        html.Div([
            html.Button("ATM (k=0)", id="atm-btn", n_clicks=0, style={"marginRight": "6px"}),
            html.Button("OTM wings", id="wings-btn", n_clicks=0, style={"marginRight": "6px"}),
            html.Button("Reset filters", id="reset-btn", n_clicks=0),
        ], style={"marginBottom": "8px"}),

        html.Hr(),

        html.Label("Surface method"),
        dcc.RadioItems(
            id="method-radio",
            options=[
                {"label": "Raw (interp)", "value": "interp"},
                {"label": "SVI (fitted)", "value": "svi"},
            ],
            value="svi",
        ),
        html.Br(),

        html.Label("Surface view"),
        dcc.RadioItems(
            id="surface-mode",
            options=[
                {"label": "3D", "value": "3d"},
                {"label": "Heatmap", "value": "heatmap"},
            ],
            value="3d",
        ),
        html.Br(),

        html.Label("Term-structure k (log-moneyness)"),
        dcc.Slider(id="k-slider", min=-0.8, max=0.8, step=0.01, value=0.0,
                   tooltip={"placement": "bottom", "always_visible": False}),
        html.Div(id="k-slider-readout", style={"marginTop": "4px", "fontSize": "12px", "opacity": 0.7}),
        html.Br(),

        dcc.Checklist(
            id="svi-toggle",
            options=[{"label": " Show SVI overlay", "value": "on"}],
            value=["on"],
            style={"marginBottom": "8px"},
        ),

        dcc.Interval(id="refresh", interval=60_000, n_intervals=0),
        html.Hr(),

        # Downloads
        html.Div([
            html.Button("Download calendar CSV", id="dl-cal-btn"),
            dcc.Download(id="dl-cal"),
        ], style={"marginBottom": "8px"}),
        html.Div([
            html.Button("Download convexity CSV", id="dl-conv-btn"),
            dcc.Download(id="dl-conv"),
        ], style={"marginBottom": "8px"}),
        html.Div([
            html.Button("Download smile CSV", id="dl-smile-btn"),
            dcc.Download(id="dl-smile"),
        ]),
    ],
    style={
        "width": "320px",
        "padding": "14px",
        "borderRight": "1px solid #eee",
        "position": "sticky",
        "top": 0,
        "height": "100vh",
        "overflowY": "auto",
        "background": "#fafafa",
        "fontFamily": "Inter, system-ui, sans-serif",
    },
)

# Main content
content = html.Div(
    [
        dcc.Tabs(
            id="tabs",
            value="tab-overview",
            children=[
                dcc.Tab(
                    label="Overview",
                    value="tab-overview",
                    children=[
                        html.Div([
                            dcc.Graph(id="smile-fig"),
                            dcc.Graph(id="surface-fig"),
                            dcc.Graph(id="term-fig"),
                        ])
                    ],
                ),
                dcc.Tab(
                    label="Anomalies",
                    value="tab-anoms",
                    children=[
                        html.Div([
                            html.H4("Calendar violations (w(T) should be non-decreasing in T)"),
                            dash_table.DataTable(
                                id="cal-table",
                                page_size=10,
                                sort_action="native",
                                filter_action="native",
                                style_table={"overflowX": "auto"},
                                style_data_conditional=[
                                    {"if": {"filter_query": "{gap} > 0.02"}, "backgroundColor": "#ffe5e5"},
                                    {"if": {"filter_query": "{gap} > 0.05"}, "backgroundColor": "#ffc2c2", "fontWeight": "600"},
                                ],
                            ),
                            html.Br(),
                            html.H4("Convexity violations (call price convex in strike)"),
                            dash_table.DataTable(
                                id="conv-table",
                                page_size=10,
                                sort_action="native",
                                filter_action="native",
                                style_table={"overflowX": "auto"},
                                style_data_conditional=[
                                    {"if": {"filter_query": "{violation} < -0.01"}, "backgroundColor": "#fff1cc"},
                                    {"if": {"filter_query": "{violation} < -0.05"}, "backgroundColor": "#ffe08a", "fontWeight": "600"},
                                ],
                            ),
                        ], style={"padding": "10px"}),
                    ],
                ),
            ],
        ),
    ],
    style={"padding": "12px", "flex": 1, "minWidth": 0},
)

app_layout = html.Div([sidebar, content], style={"display": "flex", "gap": "0"})
app.layout = app_layout

# -----------------------------
# Callbacks
# -----------------------------
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

# Quick actions -> update filter widgets
@app.callback(
    Output("k-slider", "value"),
    Output("k-range", "value"),
    Output("rights-cb", "value"),
    Output("iv-range", "value"),
    Output("oi-min", "value"),
    Output("vol-min", "value"),
    Input("atm-btn", "n_clicks"),
    Input("wings-btn", "n_clicks"),
    Input("reset-btn", "n_clicks"),
    State("k-slider", "value"),
    State("k-range", "value"),
    State("rights-cb", "value"),
    State("iv-range", "value"),
    State("oi-min", "value"),
    State("vol-min", "value"),
    prevent_initial_call=True,
)
def quick_actions(n_atm, n_wings, n_reset, k_val, k_rng, rights, iv_rng, oi_min, vol_min):
    trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0] if dash.callback_context.triggered else None
    if trig == "atm-btn":
        return 0.0, k_rng, rights, iv_rng, oi_min, vol_min
    if trig == "wings-btn":
        return k_val, [-1.0, -0.20] if k_val < 0 else [0.20, 1.0], rights, iv_rng, oi_min, vol_min
    # reset
    return 0.0, [-0.5, 0.5], ["C", "P"], [0.05, 2.0], 0, 0

@app.callback(
    Output("smile-fig", "figure"),
    Output("surface-fig", "figure"),
    Output("term-fig", "figure"),
    Output("cal-table", "data"),
    Output("cal-table", "columns"),
    Output("conv-table", "data"),
    Output("conv-table", "columns"),
    Output("k-slider-readout", "children"),
    Input("file-dd", "value"),
    Input("expiry-dd", "value"),
    Input("method-radio", "value"),
    Input("surface-mode", "value"),
    Input("k-slider", "value"),
    Input("rights-cb", "value"),
    Input("k-range", "value"),
    Input("iv-range", "value"),
    Input("oi-min", "value"),
    Input("vol-min", "value"),
    Input("svi-toggle", "value"),
)
def update_all(path, expiry, method, surface_mode, k_star,
               rights, k_rng, iv_rng, oi_min, vol_min, svi_toggle):
    # Defaults
    empty_fig = go.Figure()
    empty_cols, empty_rows = [], []
    k_readout = f"k = {float(k_star or 0.0):.3f}"

    if not path:
        return empty_fig, empty_fig, empty_fig, empty_rows, empty_cols, empty_rows, empty_cols, k_readout

    df = load_df(Path(path))
    if df.empty:
        return empty_fig, empty_fig, empty_fig, empty_rows, empty_cols, empty_rows, empty_cols, k_readout

    # Parse inputs
    rights = set(rights or ["C", "P"])
    k_lo, k_hi = (k_rng or [-1.0, 1.0])
    iv_lo, iv_hi = (iv_rng or [0.01, 3.0])
    oi_min = int(oi_min or 0)
    vol_min = int(vol_min or 0)
    show_svi = ("on" in (svi_toggle or []))

    # Apply filters for plotting
    df_plot = df.copy()
    if "right" in df_plot.columns:
        df_plot = df_plot[df_plot["right"].isin(list(rights))]
    df_plot = df_plot[df_plot["k"].between(k_lo, k_hi)]
    df_plot = df_plot[df_plot["iv_est"].between(iv_lo, iv_hi)]
    if "open_interest" in df_plot.columns:
        df_plot = df_plot[df_plot["open_interest"].fillna(0) >= oi_min]
    if "volume" in df_plot.columns:
        df_plot = df_plot[df_plot["volume"].fillna(0) >= vol_min]

    # --- Smile ---
    try:
        exp_date = pd.to_datetime(expiry).date() if expiry else df_plot["expiration"].min()
    except Exception:
        exp_date = df_plot["expiration"].min()
    fig_smile = make_smile_fig(df_plot, exp_date, show_svi)

    # --- Surface + anomalies ---
    # For surface stability, keep broader dataset but still respect liquidity filters
    df_surface = df.copy()
    if "open_interest" in df_surface.columns:
        df_surface = df_surface[df_surface["open_interest"].fillna(0) >= oi_min]
    if "volume" in df_surface.columns:
        df_surface = df_surface[df_surface["volume"].fillna(0) >= vol_min]
    if "right" in df_surface.columns:
        df_surface = df_surface[df_surface["right"].isin(list(rights))]
    df_surface = df_surface[df_surface["iv_est"].between(iv_lo, iv_hi)]

    if df_surface.empty:
        fig_surf = empty_fig
        cal = pd.DataFrame()
        conv = pd.DataFrame()
        cal_cols = []; conv_cols = []
        fig_term = empty_fig
    else:
        k_grid, T_grid, IV, W, meta = build_surface(df_surface, n_k=61, max_exps=8, method=method)
        fig_surf = make_surface_fig_heatmap(k_grid, T_grid, IV) if surface_mode == "heatmap" \
                   else make_surface_fig_3d(k_grid, T_grid, IV)
        fig_term = make_term_structure_fig(T_grid, IV, k_grid, float(k_star or 0.0))

        cal = calendar_violations(k_grid, T_grid, W, tol=1e-6)
        conv = convexity_violations(df_surface, max_exps=8, tol=0.0)
        cal_cols = [{"name": c, "id": c} for c in cal.columns]
        conv_cols = [{"name": c, "id": c} for c in conv.columns]

    return (
        fig_smile,
        fig_surf,
        fig_term,
        cal.to_dict("records"),
        cal_cols,
        conv.to_dict("records"),
        conv_cols,
        k_readout,
    )

# Downloads (CSV)
@app.callback(
    Output("dl-cal", "data"),
    Input("dl-cal-btn", "n_clicks"),
    State("file-dd", "value"),
    State("method-radio", "value"),
    State("rights-cb", "value"),
    State("iv-range", "value"),
    State("oi-min", "value"),
    State("vol-min", "value"),
    prevent_initial_call=True,
)
def download_calendar(n, path, method, rights, iv_rng, oi_min, vol_min):
    if not path:
        return None
    df = load_df(Path(path))
    # Match surface filters
    if "right" in df.columns and rights:
        df = df[df["right"].isin(list(rights))]
    if "open_interest" in df.columns:
        df = df[df["open_interest"].fillna(0) >= int(oi_min or 0)]
    if "volume" in df.columns:
        df = df[df["volume"].fillna(0) >= int(vol_min or 0)]
    if iv_rng:
        df = df[df["iv_est"].between(iv_rng[0], iv_rng[1])]
    if df.empty:
        return None
    k_grid, T_grid, IV, W, meta = build_surface(df, n_k=61, max_exps=8, method=method)
    cal = calendar_violations(k_grid, T_grid, W, tol=1e-6)
    return dcc.send_data_frame(cal.to_csv, "calendar_violations.csv", index=False)

@app.callback(
    Output("dl-conv", "data"),
    Input("dl-conv-btn", "n_clicks"),
    State("file-dd", "value"),
    State("rights-cb", "value"),
    State("iv-range", "value"),
    State("oi-min", "value"),
    State("vol-min", "value"),
    prevent_initial_call=True,
)
def download_convexity(n, path, rights, iv_rng, oi_min, vol_min):
    if not path:
        return None
    df = load_df(Path(path))
    if "right" in df.columns and rights:
        df = df[df["right"].isin(list(rights))]
    if "open_interest" in df.columns:
        df = df[df["open_interest"].fillna(0) >= int(oi_min or 0)]
    if "volume" in df.columns:
        df = df[df["volume"].fillna(0) >= int(vol_min or 0)]
    if iv_rng:
        df = df[df["iv_est"].between(iv_rng[0], iv_rng[1])]
    conv = convexity_violations(df, max_exps=8, tol=0.0)
    return dcc.send_data_frame(conv.to_csv, "convexity_violations.csv", index=False)

@app.callback(
    Output("dl-smile", "data"),
    Input("dl-smile-btn", "n_clicks"),
    State("file-dd", "value"),
    State("expiry-dd", "value"),
    State("rights-cb", "value"),
    State("k-range", "value"),
    State("iv-range", "value"),
    State("oi-min", "value"),
    State("vol-min", "value"),
    prevent_initial_call=True,
)
def download_smile(n, path, expiry, rights, k_rng, iv_rng, oi_min, vol_min):
    if not path:
        return None
    df = load_df(Path(path))
    if expiry:
        try:
            exp_date = pd.to_datetime(expiry).date()
        except Exception:
            exp_date = df["expiration"].min()
        df = df[df["expiration"] == exp_date]
    if "right" in df.columns and rights:
        df = df[df["right"].isin(list(rights))]
    df = df[df["k"].between(*(k_rng or [-1, 1]))]
    df = df[df["iv_est"].between(*(iv_rng or [0.01, 3.0]))]
    if "open_interest" in df.columns:
        df = df[df["open_interest"].fillna(0) >= int(oi_min or 0)]
    if "volume" in df.columns:
        df = df[df["volume"].fillna(0) >= int(vol_min or 0)]
    if df.empty:
        return None
    cols = [c for c in ["expiration","right","strike","k","T","iv_est","bid","ask","last","open_interest","volume","delta","gamma","theta","vega","F"] if c in df.columns]
    return dcc.send_data_frame(df[cols].to_csv, "smile.csv", index=False)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8050"))  # Render sets PORT
    app.run(debug=False, host="0.0.0.0", port=port)