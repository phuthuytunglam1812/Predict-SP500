from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from predict_helper import DATA_PATH, FEATURE_ORDER, directional_mse_loss, evaluate_models, load_data, load_models, predict


load_dotenv()
load_dotenv("app.env", override=False)
globals()["directional_mse_loss"] = directional_mse_loss
ALLOCATION_LOOKUP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "allocation_lookup.json")
ALLOCATION_RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "allocation_rules.json")

st.set_page_config(
    page_title="CPI x SPX Impact Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
    :root {
        --bg: #070b14;
        --panel: rgba(15, 23, 42, 0.78);
        --panel-strong: rgba(17, 24, 39, 0.95);
        --ink: #e5edf7;
        --muted: #93a4b8;
        --line: rgba(148, 163, 184, 0.28);
        --line-strong: rgba(45, 212, 191, 0.34);
        --accent: #2dd4bf;
        --accent-2: #38bdf8;
        --danger: #fb7185;
        --success: #34d399;
        --warn: #fbbf24;
        --mono: "Cascadia Mono", "SFMono-Regular", Consolas, monospace;
    }
    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(45, 212, 191, 0.14), transparent 28%),
            radial-gradient(circle at 88% 4%, rgba(56, 189, 248, 0.10), transparent 26%),
            linear-gradient(135deg, #060914 0%, #0b1120 52%, #111827 100%);
        color: var(--ink);
    }
    .block-container {
        max-width: 1480px;
        padding-top: 3.6rem;
        padding-bottom: 2.4rem;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(2, 6, 23, 0.96), rgba(15, 23, 42, 0.94));
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] * { color: var(--ink); }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    h1 { font-size: 2.05rem; margin-bottom: 0.2rem; }
    h2 { font-size: 1.05rem; margin-top: 0.8rem; margin-bottom: 0.65rem; }
    .subtle { color: var(--muted); font-size: 0.92rem; margin-bottom: 0.9rem; }
    .hero-panel, .flow-card, .metric-card, .context-item, div[data-testid="stPlotlyChart"] {
        border: 1px solid var(--line);
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.86), rgba(15, 23, 42, 0.56));
        box-shadow: 0 20px 55px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(14px);
    }
    .hero-panel {
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }
    .hero-kicker { color: var(--accent); font-family: var(--mono); font-size: 0.76rem; letter-spacing: 0.12em; text-transform: uppercase; }
    .flow-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.75rem; margin: 0.85rem 0 1rem; }
    .flow-card { border-radius: 14px; padding: 0.8rem 0.85rem; min-height: 98px; }
    .flow-step { color: var(--accent); font-family: var(--mono); font-size: 0.74rem; margin-bottom: 0.3rem; }
    .flow-title { color: var(--ink); font-weight: 800; margin-bottom: 0.25rem; }
    .flow-copy { color: var(--muted); font-size: 0.82rem; line-height: 1.35; }
    .flow-copy strong { color: var(--ink); font-weight: 900; }
    .term {
        border-bottom: 1px dotted var(--accent);
        color: var(--ink);
        cursor: help;
        display: inline-block;
        position: relative;
        text-decoration: none;
        z-index: 5;
    }
    .term .definition-box {
        background: #f8fafc;
        border: 1px solid rgba(45, 212, 191, 0.55);
        border-radius: 10px;
        box-shadow: 0 18px 44px rgba(0, 0, 0, 0.34);
        color: #0f172a;
        display: none;
        font-family: Inter, "Segoe UI", sans-serif;
        font-size: 0.78rem;
        font-weight: 650;
        left: 50%;
        letter-spacing: 0;
        line-height: 1.35;
        min-width: 300px;
        max-width: 360px;
        padding: 0.75rem 0.85rem;
        position: fixed;
        text-transform: none;
        top: 4.8rem;
        transform: translateX(-50%);
        white-space: normal;
        z-index: 100000;
    }
    .term .definition-box::before {
        display: none;
        background: #f8fafc;
        border-left: 1px solid rgba(45, 212, 191, 0.55);
        border-top: 1px solid rgba(45, 212, 191, 0.55);
        content: "";
        height: 10px;
        left: 16px;
        position: fixed;
        bottom: -6px;
        transform: rotate(45deg);
        width: 10px;
    }
    .term:hover { z-index: 100000; }
    .term:hover .definition-box {
        display: block;
        z-index: 100000;
    }
    .status-strip {
        border: 1px solid var(--line-strong);
        border-radius: 14px;
        color: var(--accent);
        font-family: var(--mono);
        padding: 0.75rem 0.9rem;
        background: rgba(6, 78, 59, 0.18);
        margin: 0.75rem 0 1rem;
    }
    .metric-card {
        border-radius: 14px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        height: 132px;
        justify-content: space-between;
        padding: 0.9rem 1rem;
    }
    .metric-label { color: var(--muted); font-family: var(--mono); font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; line-height: 1.2; }
    .metric-value { color: var(--ink); font-family: var(--mono); font-size: 1.55rem; line-height: 1.05; font-weight: 850; overflow-wrap: anywhere; text-shadow: 0 0 18px rgba(45, 212, 191, 0.10); }
    .metric-note { color: var(--muted); font-size: 0.74rem; line-height: 1.25; }
    .positive { color: var(--success); text-shadow: 0 0 18px rgba(52, 211, 153, 0.38); }
    .negative { color: var(--danger); text-shadow: 0 0 18px rgba(251, 113, 133, 0.32); }
    .neutral { color: var(--warn); text-shadow: 0 0 18px rgba(251, 191, 36, 0.26); }
    div[data-testid="stPlotlyChart"] { border-radius: 22px; padding: 0.75rem; overflow: hidden; }
    div[data-testid="stPlotlyChart"] > div { border-radius: 18px; overflow: hidden; }
    div[data-testid="stPlotlyChart"] .js-plotly-plot,
    div[data-testid="stPlotlyChart"] .plot-container,
    div[data-testid="stPlotlyChart"] .svg-container,
    div[data-testid="stPlotlyChart"] .main-svg { background: transparent !important; border-radius: 18px !important; overflow: hidden !important; }
    .context-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 0.7rem; overflow: visible; position: relative; z-index: 20; }
    .context-item { border-radius: 14px; padding: 0.72rem 0.78rem; min-height: 84px; border-left: 3px solid var(--accent); overflow: visible; position: relative; z-index: 1; }
    .context-item:hover { z-index: 10000; }
    .context-label { color: var(--muted); font-family: var(--mono); font-size: 0.7rem; margin-bottom: 0.28rem; text-transform: uppercase; letter-spacing: 0.06em; overflow: visible; position: relative; z-index: 10001; }
    .context-value { color: var(--ink); font-family: var(--mono); font-size: 1.1rem; font-weight: 800; }
    .reasoning-list { margin: 0; padding-left: 1.2rem; color: var(--ink); font-family: var(--mono); font-size: 0.92rem; line-height: 1.62; }
    .reasoning-list li { margin: 0.45rem 0; }
    .placeholder-box { border: 1px dashed var(--line-strong); background: rgba(2, 6, 23, 0.38); border-radius: 12px; padding: 0.8rem; color: var(--muted); font-size: 0.84rem; }
    .sidebar-market-board { border: 1px solid var(--line-strong); background: linear-gradient(145deg, rgba(15,23,42,0.92), rgba(2,6,23,0.72)); border-radius: 14px; padding: 0.85rem; box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 0 28px rgba(45,212,191,0.08); }
    .market-board-title { color: var(--ink); font-family: var(--mono); font-weight: 800; font-size: 0.82rem; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.18rem; }
    .market-board-date { color: var(--muted); font-size: 0.76rem; margin-bottom: 0.7rem; }
    .market-stat-row { display: flex; justify-content: space-between; gap: 0.75rem; align-items: baseline; padding: 0.52rem 0; border-top: 1px solid rgba(148,163,184,0.16); }
    .market-stat-row span { color: var(--muted); font-size: 0.78rem; }
    .market-stat-row strong { color: var(--ink); font-family: var(--mono); font-size: 0.94rem; text-align: right; }
    .market-stat-row strong.good { color: var(--success); }
    .market-stat-row strong.bad { color: var(--danger); }
    .allocation-guide { border: 1px solid var(--line); background: linear-gradient(145deg, rgba(15,23,42,0.92), rgba(2,6,23,0.72)); border-radius: 16px; padding: 1rem 1.1rem; margin: 1rem 0 1.15rem; box-shadow: 0 20px 55px rgba(0,0,0,0.26), inset 0 1px 0 rgba(255,255,255,0.05); }
    .allocation-head { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 1rem; align-items: end; border-bottom: 1px solid rgba(148,163,184,0.18); padding-bottom: 0.85rem; margin-bottom: 0.4rem; }
    .allocation-title { color: var(--ink); font-family: var(--mono); font-weight: 900; font-size: 1.05rem; letter-spacing: 0.04em; text-transform: uppercase; }
    .allocation-subtitle { color: var(--muted); font-size: 0.82rem; margin-top: 0.22rem; }
    .allocation-current { color: var(--success); font-family: var(--mono); font-size: 2rem; font-weight: 900; text-shadow: 0 0 20px rgba(52,211,153,0.32); text-align: right; }
    .allocation-money { color: var(--muted); font-size: 0.82rem; text-align: right; }
    .allocation-table { width: 100%; border-collapse: collapse; margin-top: 0.35rem; }
    .allocation-table th { color: var(--muted); font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; text-align: left; padding: 0.7rem 0.55rem; border-bottom: 1px solid rgba(148,163,184,0.22); }
    .allocation-table td { color: var(--ink); padding: 0.72rem 0.55rem; border-bottom: 1px solid rgba(148,163,184,0.12); vertical-align: top; font-size: 0.9rem; }
    .allocation-table td:last-child { color: var(--ink); font-family: var(--mono); font-weight: 850; text-align: right; white-space: nowrap; }
    .allocation-note { color: var(--muted); font-size: 0.78rem; line-height: 1.35; margin-top: 0.72rem; }
    .stButton > button { border-radius: 10px; border: 1px solid var(--accent); color: #061017; background: var(--accent); min-height: 2.55rem; font-weight: 800; box-shadow: 0 0 24px rgba(45, 212, 191, 0.18); }
    .stButton > button:hover { border-color: var(--accent-2); background: var(--accent-2); color: #020617; }
    div[data-testid="stSlider"] label,
    div[data-testid="stSlider"] label p,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stNumberInput"] label p { color: var(--ink) !important; font-size: 0.86rem !important; font-weight: 750 !important; opacity: 1 !important; }
    div[data-testid="stNumberInput"] input {
        background: rgba(2, 6, 23, 0.72) !important;
        color: var(--ink) !important;
        border-color: var(--line) !important;
        font-family: var(--mono) !important;
    }
    .stDataFrame { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
    @media (max-width: 1000px) {
        .flow-grid, .context-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .metric-value { font-size: 1.25rem; }
    }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def cached_data() -> pd.DataFrame:
    return load_data(DATA_PATH)


def format_pct(value: float | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:+.{decimals}f}%"


def format_plain(value: float | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"

@st.cache_data(ttl=900, show_spinner=False)
def fetch_live_market_quote(symbol: str) -> dict[str, float | str | None]:
    encoded_symbol = urllib.parse.quote(symbol, safe="")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}?range=5d&interval=1d"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8"))
        result = payload["chart"]["result"][0]
        meta = result.get("meta", {})
        price = meta.get("regularMarketPrice")
        previous = meta.get("chartPreviousClose") or meta.get("previousClose")
        change_pct = None
        if price is not None and previous not in (None, 0):
            change_pct = (float(price) - float(previous)) / float(previous) * 100
        return {
            "symbol": symbol,
            "price": float(price) if price is not None else None,
            "change_pct": change_pct,
            "source": "Yahoo Finance live",
        }
    except Exception:
        return {"symbol": symbol, "price": None, "change_pct": None, "source": "CSV fallback"}



@st.cache_data(ttl=21600, show_spinner=False)
def fetch_official_cpi_yoy() -> tuple[pd.DataFrame, str]:
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/CUUR0000SA0"
    payload = json.dumps({"seriesid": ["CUUR0000SA0"], "startyear": "2024", "endyear": "2026"}).encode("utf-8")
    try:
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=6) as response:
            data = json.loads(response.read().decode("utf-8"))
        observations = data["Results"]["series"][0]["data"]
        rows = []
        for item in observations:
            period = item.get("period", "")
            if not period.startswith("M"):
                continue
            rows.append(
                {
                    "Release_Date": pd.Timestamp(year=int(item["year"]), month=int(period[1:]), day=1),
                    "CPI_Index": float(item["value"]),
                }
            )
        cpi_df = pd.DataFrame(rows).sort_values("Release_Date")
        cpi_df["CPI_YoY"] = cpi_df["CPI_Index"].pct_change(12) * 100
        cpi_df = cpi_df.dropna(subset=["CPI_YoY"]).tail(12).reset_index(drop=True)
        if cpi_df.empty:
            raise ValueError("No official CPI observations returned")
        return cpi_df, "Official BLS CPI-U"
    except Exception:
        return pd.DataFrame(), "CSV fallback"


def cpi_trend_data(history: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    official_df, source = fetch_official_cpi_yoy()
    if not official_df.empty:
        return official_df, source
    chart_df = history.sort_values("Release_Date").copy().tail(12)
    return chart_df[["Release_Date", "CPI_YoY"]], source


def cpi_trend_summary(chart_df: pd.DataFrame, source: str) -> str:
    if chart_df.empty:
        return "CPI trend is unavailable right now."
    latest_value = float(chart_df["CPI_YoY"].iloc[-1])
    first_value = float(chart_df["CPI_YoY"].iloc[0])
    change = latest_value - first_value
    if abs(change) < 0.15:
        direction = "has been broadly stable"
    elif change > 0:
        direction = "has moved higher"
    else:
        direction = "has cooled"
    return f"Trend: CPI inflation {direction} over the past 12 months, moving from {first_value:.2f}% to {latest_value:.2f}% ({source})."


def cpi_trend_chart(chart_df: pd.DataFrame) -> go.Figure:
    chart_df = chart_df.sort_values("Release_Date").copy()
    latest_row = chart_df.iloc[-1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df["Release_Date"],
            y=chart_df["CPI_YoY"],
            mode="lines+markers",
            name="CPI YoY",
            line={"color": "#2dd4bf", "width": 3},
            marker={"color": "#99f6e4", "size": 6},
            hovertemplate="%{x|%Y-%m}<br>CPI YoY %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[latest_row["Release_Date"]],
            y=[latest_row["CPI_YoY"]],
            mode="markers+text",
            name="Latest CPI",
            marker={"color": "#fb7185", "size": 11, "line": {"color": "#ffe4e6", "width": 1.4}},
            text=[f"{latest_row['CPI_YoY']:.2f}%"],
            textposition="top center",
            textfont={"color": "#e5edf7", "size": 12, "family": "Cascadia Mono, Consolas, monospace"},
            hovertemplate="Latest %{x|%Y-%m}<br>CPI YoY %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=210,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(2,6,23,0.42)",
        font={"color": "#cbd5e1", "family": "Inter, Segoe UI, sans-serif"},
        showlegend=False,
        xaxis={"showgrid": False, "tickfont": {"color": "#93a4b8", "size": 10}, "title": None},
        yaxis={"title": None, "ticksuffix": "%", "gridcolor": "rgba(148,163,184,0.16)", "zeroline": False, "tickfont": {"color": "#93a4b8", "size": 10}},
    )
    return fig
def tone_class(value: float) -> str:
    if value > 0:
        return "negative"
    if value < 0:
        return "positive"
    return "neutral"


def metric_card(label: str, value: str, note: str = "", css_class: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {css_class}">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def gauge_chart(actual_surprise: float, history: pd.DataFrame) -> go.Figure:
    lower = min(-0.6, float(history["CPI_Surprise_Pct"].min()) - 0.05, actual_surprise - 0.15)
    upper = max(0.6, float(history["CPI_Surprise_Pct"].max()) + 0.05, actual_surprise + 0.15)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=actual_surprise,
            number={"suffix": " pts", "font": {"size": 36, "color": "#e5edf7", "family": "Cascadia Mono, Consolas, monospace"}},
            gauge={
                "axis": {
                    "range": [lower, upper],
                    "tickwidth": 2,
                    "tickcolor": "#94a3b8",
                    "tickfont": {"color": "#cbd5e1", "size": 13},
                },
                "bar": {"color": "#2dd4bf", "thickness": 0.34},
                "bgcolor": "rgba(2,6,23,0.35)",
                "borderwidth": 1,
                "bordercolor": "rgba(148,163,184,0.35)",
                "steps": [
                    {"range": [lower, -0.1], "color": "rgba(52,211,153,0.22)"},
                    {"range": [-0.1, 0.1], "color": "rgba(251,191,36,0.26)"},
                    {"range": [0.1, upper], "color": "rgba(251,113,133,0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#67e8f9", "width": 6},
                    "thickness": 0.75,
                    "value": actual_surprise,
                },
            },
            title={"text": "Actual CPI Surprise", "font": {"color": "#cbd5e1", "size": 14}},
            domain={"x": [0.02, 0.98], "y": [0.02, 0.98]},
        )
    )
    fig.update_layout(
        height=300,
        margin={"l": 42, "r": 42, "t": 48, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5edf7"},
    )
    return fig


def scatter_chart(history: pd.DataFrame, actual_surprise: float, result: dict) -> go.Figure:
    chart_df = history.copy()
    chart_df["Similar"] = "Other events"
    similar_dates = set(result["similar_events"]["Release_Date"]) if len(result["similar_events"]) else set()
    chart_df.loc[chart_df["Release_Date"].isin(similar_dates), "Similar"] = "Similar events"

    fig = go.Figure()
    base = chart_df[chart_df["Similar"] == "Other events"]
    similar = chart_df[chart_df["Similar"] == "Similar events"]

    fig.add_trace(
        go.Scatter(
            x=base["CPI_Surprise_Pct"],
            y=base["SP500_Daily_Return_Pct"],
            mode="markers",
            name="Other events",
            marker={"color": "#64748b", "size": 8, "opacity": 0.58},
            text=base["Release_Date"].dt.strftime("%Y-%m-%d"),
            hovertemplate="Date %{text}<br>CPI surprise %{x:.2f}<br>SPX 1D %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=similar["CPI_Surprise_Pct"],
            y=similar["SP500_Daily_Return_Pct"],
            mode="markers",
            name="Similar events",
            marker={"color": "#2dd4bf", "size": 13, "line": {"color": "#ecfeff", "width": 1.5}},
            text=similar["Release_Date"].dt.strftime("%Y-%m-%d"),
            hovertemplate="Date %{text}<br>CPI surprise %{x:.2f}<br>SPX 1D %{y:.2f}%<extra></extra>",
        )
    )
    if similar.empty:
        fig.add_annotation(
            text="No similar events in the historical data for this input",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.07,
            showarrow=False,
            font={"color": "#fbbf24", "size": 12},
        )
    fig.add_vline(x=actual_surprise, line_width=3, line_dash="dash", line_color="#fb7185")
    fig.add_hline(y=0, line_width=2, line_color="rgba(148,163,184,0.62)")
    fig.update_layout(
        height=370,
        margin={"l": 10, "r": 10, "t": 54, "b": 10},
        xaxis_title="CPI surprise",
        yaxis_title="SPX 1D return (%)",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.12,
            "xanchor": "left",
            "x": 0,
            "font": {"color": "#e5edf7", "size": 13},
            "bgcolor": "rgba(2,6,23,0.72)",
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(2,6,23,0.48)",
        font={"color": "#cbd5e1", "size": 13},
        xaxis={
            "gridcolor": "rgba(148,163,184,0.20)",
            "zerolinecolor": "rgba(148,163,184,0.60)",
            "linecolor": "#64748b",
            "tickfont": {"color": "#cbd5e1"},
            "title": {"font": {"color": "#cbd5e1"}},
        },
        yaxis={
            "gridcolor": "rgba(148,163,184,0.20)",
            "zerolinecolor": "rgba(148,163,184,0.60)",
            "linecolor": "#64748b",
            "tickfont": {"color": "#cbd5e1"},
            "title": {"font": {"color": "#cbd5e1"}},
        },
    )
    return fig


def probability_bar(result: dict) -> go.Figure:
    probability = float(result.get("spx_probability") or 0.0)
    direction = result.get("spx_direction", "N/A")
    color = "#34d399" if direction == "Up" else "#fb7185" if direction == "Down" else "#fbbf24"
    fig = go.Figure(
        go.Bar(
            x=[probability],
            y=[direction],
            orientation="h",
            marker={"color": color},
            text=[f"{probability:.1f}%"],
            textposition="inside",
            insidetextanchor="middle",
            textfont={"color": "#ffffff", "size": 14},
            hovertemplate="Probability %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=110,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={
            "range": [0, 100],
            "title": None,
            "showgrid": True,
            "gridcolor": "rgba(148,163,184,0.18)",
            "tickfont": {"color": "#cbd5e1"},
        },
        yaxis={"title": None, "tickfont": {"color": "#cbd5e1", "size": 13}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(2,6,23,0.48)",
        font={"color": "#cbd5e1"},
        showlegend=False,
    )
    return fig


@st.cache_data(show_spinner=False)
def load_allocation_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


@st.cache_data(show_spinner=False)
def load_allocation_lookup(path: str = ALLOCATION_LOOKUP_PATH) -> dict:
    return load_allocation_json(path)


@st.cache_data(show_spinner=False)
def load_allocation_rules(path: str = ALLOCATION_RULES_PATH) -> dict:
    return load_allocation_json(path)


def select_allocation_rule(result: dict, latest: pd.Series, actual_cpi: float) -> str:
    inflation = float(actual_cpi if actual_cpi is not None else latest.get("CPI_YoY", 0.0))
    fed_change = float(latest.get("FedFunds_Change_3M", 0.0) or 0.0)
    spx_direction = str(result.get("spx_direction", "Neutral")).lower()

    is_easing = fed_change < -0.05
    is_tightening = fed_change > 0.05

    if inflation < 1.0:
        return "deflation_tightening" if is_tightening else "deflation_recession"
    if inflation >= 7.0:
        return "hyper_inflation_tightening"
    if inflation >= 5.0:
        if is_easing:
            return "high_inflation_panic_cut"
        return "high_inflation_stable" if not is_tightening else "hyper_inflation_tightening"
    if inflation >= 3.0:
        if is_tightening:
            return "stable_growth_tightening"
        if is_easing:
            return "stable_growth_easing"
        return "reflation_base"
    if is_easing:
        return "stable_growth_easing"
    if is_tightening:
        return "stable_growth_tightening"
    return "goldilocks_bull" if spx_direction == "up" else "stable_growth_easing"


def normalize_allocation_row(row: dict, regime: str, source: str) -> dict[str, str | int | float]:
    stocks_pct = int(round(float(row.get("stock_pct", row.get("stocks_pct", 60)))))
    bonds_pct = int(round(float(row.get("bond_pct", row.get("bonds_pct", 100 - stocks_pct)))))
    total_pct = stocks_pct + bonds_pct
    if total_pct != 100 and total_pct > 0:
        stocks_pct = int(round(stocks_pct * 100 / total_pct))
        bonds_pct = 100 - stocks_pct

    tone = "positive" if stocks_pct >= 70 else "negative" if stocks_pct <= 40 else "neutral"
    confidence = str(row.get("confidence", source)).lower()
    avg_return = row.get("avg_return")
    note = f"{regime} allocation from {source}"
    if avg_return is not None:
        note = f"{note}; historical average return {float(avg_return):+.2f}%"
    if confidence and confidence not in {"lookup", "rules"}:
        note = f"{note}; {confidence} confidence"

    return {
        "allocation": f"{stocks_pct}/{bonds_pct}",
        "stocks_pct": stocks_pct,
        "bonds_pct": bonds_pct,
        "tone": tone,
        "note": note,
        "regime": regime,
        "avg_return": float(avg_return) if avg_return is not None else 0.0,
        "volatility": float(row.get("volatility", 0.0)),
        "risk_adjusted_score": float(row.get("risk_adjusted_score", 0.0)),
        "source": source,
    }


def allocation_signal(result: dict, latest: pd.Series, actual_cpi: float) -> dict[str, str | int | float]:
    rules = load_allocation_rules()
    if rules:
        regime = select_allocation_rule(result, latest, actual_cpi)
        row = rules.get(regime) or rules.get("reflation_base") or next(iter(rules.values()))
        return normalize_allocation_row(row, regime, "allocation_rules")

    lookup = load_allocation_lookup()
    direction = str(result.get("spx_direction", "Neutral")).lower()
    probability = float(result.get("spx_probability") or 0.0)
    if probability == 0.0 or direction not in {"up", "down"}:
        regime = "base"
    elif direction == "up":
        regime = "bull"
    else:
        regime = "bear"
    row = lookup.get(regime) or lookup.get("base") or {}
    return normalize_allocation_row(row, regime, "allocation_lookup")


def get_openai_api_key() -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", None)
        except Exception:
            api_key = None
    return api_key


def get_openai_model(default: str = "gpt-4.1-mini") -> str:
    model = os.getenv("OPENAI_MODEL", default)
    try:
        model = st.secrets.get("OPENAI_MODEL", model)
    except Exception:
        pass
    return model


def extract_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        data = json.loads(text[start : end + 1])
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def ai_allocation_signal(
    base_allocation: dict[str, str | int | float],
    result: dict,
    latest: pd.Series,
    actual_cpi: float,
    forecast_cpi: float,
) -> dict[str, str | int | float]:
    api_key = get_openai_api_key()
    if not api_key:
        return base_allocation

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        prompt = f"""
        You are adjusting an educational stock/bond allocation signal for a finance dashboard.
        Combine the file-based allocation rule with the model outputs and macro context.
        This is not personalized investment advice. Do not use personal risk tolerance or investor age.

        Baseline file allocation:
        - Source: {base_allocation.get('source')}
        - Regime: {base_allocation.get('regime')}
        - Stocks: {base_allocation.get('stocks_pct')}%
        - Bonds: {base_allocation.get('bonds_pct')}%
        - Baseline note: {base_allocation.get('note')}

        Model and macro context:
        - Actual CPI: {actual_cpi}
        - Forecast CPI: {forecast_cpi}
        - CPI surprise: {result['actual_surprise']}
        - Pre-release expected surprise: {result['predicted_surprise']}
        - Surprise direction model: {result['surprise_direction']} at {result['surprise_confidence']:.1f}%
        - SPX impact model: {result['spx_direction']} at {result['spx_probability']:.1f}%
        - Similar historical events: {result['similar_events_count']}
        - Similar up/down: {result['similar_events_up']} up, {result['similar_events_down']} down
        - VIX return: {latest['VIX_Return_Pct']}%
        - Fed Funds Rate: {latest['FedFunds_Rate']}%
        - Fed Funds 3M Change: {latest.get('FedFunds_Change_3M', 0.0)}%

        Rules:
        - Return JSON only.
        - stocks_pct and bonds_pct must be integers between 0 and 100 and sum to 100.
        - Prefer increments of 10.
        - Treat the baseline file allocation as the anchor; only adjust if the model context gives a clear reason.
        - Keep the explanation causal and short.

        JSON schema:
        {{"stocks_pct": 30, "bonds_pct": 70, "reason": "short causal reason"}}
        """
        response = client.responses.create(
            model=get_openai_model(),
            input=prompt,
            max_output_tokens=180,
        )
        data = extract_json_object(response.output_text)
        stocks_pct = int(round(float(data.get("stocks_pct"))))
        bonds_pct = int(round(float(data.get("bonds_pct"))))
        if stocks_pct < 0 or stocks_pct > 100 or bonds_pct < 0 or bonds_pct > 100:
            return base_allocation
        if stocks_pct + bonds_pct != 100:
            bonds_pct = 100 - stocks_pct
        reason = str(data.get("reason", "AI reviewed the file-based allocation and model context.")).strip()
        final_allocation = dict(base_allocation)
        final_allocation["stocks_pct"] = stocks_pct
        final_allocation["bonds_pct"] = bonds_pct
        final_allocation["allocation"] = f"{stocks_pct}/{bonds_pct}"
        final_allocation["tone"] = "positive" if stocks_pct >= 70 else "negative" if stocks_pct <= 40 else "neutral"
        final_allocation["note"] = f"AI blended with {base_allocation.get('source')} ({base_allocation.get('regime')}): {reason}"
        final_allocation["source"] = f"{base_allocation.get('source')} + AI"
        return final_allocation
    except Exception:
        return base_allocation

def money_split(amount: float, allocation: dict[str, str | int]) -> tuple[float, float]:
    stocks_amount = float(amount) * int(allocation["stocks_pct"]) / 100
    bonds_amount = float(amount) * int(allocation["bonds_pct"]) / 100
    return stocks_amount, bonds_amount


def fallback_reasoning(actual_cpi: float, forecast_cpi: float, result: dict, latest: pd.Series) -> list[str]:
    surprise = float(result["actual_surprise"])
    predicted_surprise = float(result["predicted_surprise"])
    similar_count = int(result["similar_events_count"])
    up = int(result["similar_events_up"])
    down = int(result["similar_events_down"])
    fed_rate = float(latest.get("FedFunds_Rate", 0.0))
    vix_return = float(latest.get("VIX_Return_Pct", 0.0))

    if surprise > 0.05:
        inflation_mechanism = "When <strong>actual CPI is higher than forecast</strong>, markets can read it as inflation pressure lasting longer than expected. That can push rate expectations and bond yields higher, which often pressures stock valuations and long-duration bonds."
    elif surprise < -0.05:
        inflation_mechanism = "When <strong>actual CPI is lower than forecast</strong>, markets can read it as inflation cooling faster than expected. That can reduce pressure on future interest rates, which often supports risk assets and can help bonds."
    else:
        inflation_mechanism = "When <strong>actual CPI is close to forecast</strong>, the release contains less new inflation information. Markets may then lean more on the existing rate backdrop, volatility, and positioning instead of the CPI print alone."

    if predicted_surprise * surprise < 0:
        model_mechanism = "The pre-release surprise model and the actual surprise point in different directions, so the setup is a <strong>model disagreement</strong>. That matters because the market may have been positioned for one inflation story while the released CPI tells another."
    else:
        model_mechanism = "The pre-release surprise model and the actual surprise tell a similar inflation story, so the market signal is cleaner. A cleaner signal can make the SPX reaction model more interpretable, even if it is still probabilistic."

    if vix_return > 0:
        volatility_mechanism = "Because the <strong>VIX is rising</strong>, investors are demanding more protection. That can amplify bad inflation news because risk appetite is already weaker."
    elif vix_return < 0:
        volatility_mechanism = "Because the <strong>VIX is falling</strong>, investors are demanding less protection. That can cushion inflation worries because risk appetite is already calmer."
    else:
        volatility_mechanism = "The <strong>VIX</strong> matters because it shows how much protection investors are demanding. Higher volatility can make the same CPI surprise feel more threatening to risk assets."

    if similar_count:
        final_mechanism = f"The historical comparison is useful because it asks whether similar inflation regimes led to broad risk-on or risk-off behavior before. Similar cases split {up} up versus {down} down sessions out of {similar_count}, so the past evidence should be read as context rather than certainty."
    elif fed_rate >= 4.5:
        final_mechanism = "A high <strong>Fed Funds Rate</strong> means policy is already restrictive, so another hot CPI print can reinforce the idea of rates staying high for longer. That is usually harder for stocks and bond prices."
    elif fed_rate <= 2.0:
        final_mechanism = "A low <strong>Fed Funds Rate</strong> leaves more policy flexibility, so markets may react less harshly to mild CPI noise and more strongly to growth expectations."
    else:
        final_mechanism = "A mid-range <strong>Fed Funds Rate</strong> makes the CPI surprise important because it can shift expectations about the next policy path rather than simply confirming an already extreme rate environment."

    return [inflation_mechanism, model_mechanism, volatility_mechanism, final_mechanism]

def openai_reasoning(actual_cpi: float, forecast_cpi: float, result: dict, latest: pd.Series) -> tuple[list[str], str]:
    api_key = get_openai_api_key()
    if not api_key:
        return fallback_reasoning(actual_cpi, forecast_cpi, result, latest), "Fallback reasoning"

    try:
        from openai import OpenAI

        model = get_openai_model()
        client = OpenAI(api_key=api_key)
        prompt = f"""
        Create exactly 4 concise English bullets for a finance dashboard reasoning layer.
        The user wants explanations of WHY the model signal could make sense, not a report of the existing numbers.

        Write each bullet as a causal explanation. Use beginner-friendly finance language.
        Do not simply restate the CPI, forecast, model probability, VIX, or Fed Funds values.
        You may mention a number only if it helps explain the mechanism.
        Use <strong>...</strong> around 1 important concept per bullet.

        Context:
        - Actual CPI: {actual_cpi}
        - Forecast CPI: {forecast_cpi}
        - Actual CPI surprise: {result['actual_surprise']}
        - Pre-release expected surprise: {result['predicted_surprise']}
        - Pre-release expected direction model: {result['surprise_direction']} at {result['surprise_confidence']:.1f}%
        - SPX impact model: {result['spx_direction']} at {result['spx_probability']:.1f}%
        - Similar historical events: {result['similar_events_count']}
        - Similar up/down: {result['similar_events_up']} up, {result['similar_events_down']} down
        - Latest VIX return: {latest['VIX_Return_Pct']}%
        - Latest Fed funds rate: {latest['FedFunds_Rate']}%

        Explain mechanisms such as inflation surprise, interest-rate expectations, bond-yield sensitivity, equity valuation pressure, volatility/risk appetite, and model disagreement.
        If actual surprise and pre-release expectation disagree, explain why that disagreement matters for interpretation.
        Avoid investment advice and do not add a disclaimer.
        Return bullets only, no heading.
        """
        response = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=260,
        )
        text = response.output_text
        bullets = [line.lstrip("-* ").strip() for line in text.splitlines() if line.strip()]
        return (bullets[:4], "AI reasoning") if bullets[:4] else (fallback_reasoning(actual_cpi, forecast_cpi, result, latest), "Fallback reasoning")
    except Exception:
        return fallback_reasoning(actual_cpi, forecast_cpi, result, latest), "Fallback reasoning"


history = cached_data()
latest = history.sort_values("Release_Date").iloc[-1]
default_actual_cpi = float(latest.get("CPI_YoY", 4.30))
default_forecast_cpi = default_actual_cpi - float(latest.get("CPI_Surprise_Pct", 0.0))

if "started" not in st.session_state:
    st.session_state.started = False
if "actual_cpi" not in st.session_state:
    st.session_state.actual_cpi = default_actual_cpi
if "forecast_cpi" not in st.session_state:
    st.session_state.forecast_cpi = default_forecast_cpi
if "portfolio_amount" not in st.session_state:
    st.session_state.portfolio_amount = 10000.0
if "actual_cpi_slider" not in st.session_state:
    st.session_state.actual_cpi_slider = st.session_state.actual_cpi
if "actual_cpi_input" not in st.session_state:
    st.session_state.actual_cpi_input = st.session_state.actual_cpi
if "forecast_cpi_slider" not in st.session_state:
    st.session_state.forecast_cpi_slider = st.session_state.forecast_cpi
if "forecast_cpi_input" not in st.session_state:
    st.session_state.forecast_cpi_input = st.session_state.forecast_cpi
if "analysis" not in st.session_state:
    st.session_state.analysis = None


def term(label: str, definition: str) -> str:
    return f'<span class="term">{label}<span class="definition-box">{definition}</span></span>'


def set_actual_cpi(value: float) -> None:
    st.session_state.actual_cpi = float(value)
    st.session_state.actual_cpi_slider = float(value)
    st.session_state.actual_cpi_input = float(value)
    clear_analysis()


def set_forecast_cpi(value: float) -> None:
    st.session_state.forecast_cpi = float(value)
    st.session_state.forecast_cpi_slider = float(value)
    st.session_state.forecast_cpi_input = float(value)
    clear_analysis()


def sync_actual_from_slider() -> None:
    set_actual_cpi(st.session_state.actual_cpi_slider)


def sync_actual_from_input() -> None:
    set_actual_cpi(st.session_state.actual_cpi_input)


def sync_forecast_from_slider() -> None:
    set_forecast_cpi(st.session_state.forecast_cpi_slider)


def sync_forecast_from_input() -> None:
    set_forecast_cpi(st.session_state.forecast_cpi_input)


def load_sample_scenario() -> None:
    set_actual_cpi(default_actual_cpi)
    set_forecast_cpi(default_forecast_cpi)
    st.session_state.portfolio_amount = 10000.0
    st.session_state.analysis = None


def start_app() -> None:
    st.session_state.started = True


def clear_analysis() -> None:
    st.session_state.analysis = None


show_intro = st.session_state.analysis is None

if show_intro:
    st.markdown(
        f"""
        <div class="hero-panel">
        <div class="hero-kicker">CPI x SPX Impact Dashboard</div>
        <h1>CPI x SPX Impact Dashboard</h1>
        <div class="subtle">
            A guided investing simulator for first-time users. Enter a CPI release scenario, confirm the model run, then review the predicted SPX reaction, historical context, reasoning layer, and a stock/bond allocation tilt.
        </div>
        <div class="flow-grid">
            <div class="flow-card"><div class="flow-step">STEP 01</div><div class="flow-title">Read CPI context</div><div class="flow-copy">Start with the recent CPI trend so the current inflation backdrop is visible before entering a scenario.</div></div>
            <div class="flow-card"><div class="flow-step">STEP 02</div><div class="flow-title">Enter scenario</div><div class="flow-copy">Use the recent CPI context, then adjust <strong>actual CPI</strong>, <strong>forecast CPI</strong>, and the <strong>portfolio amount</strong> only if you want to test a scenario.</div></div>
            <div class="flow-card"><div class="flow-step">STEP 03</div><div class="flow-title">Analysis and reasoning showcase</div><div class="flow-copy">After you press the confirmation button, you can see <strong>the reasons behind the decision</strong> alongside the model outputs.</div></div>
            <div class="flow-card"><div class="flow-step">STEP 04</div><div class="flow-title">Read outputs</div><div class="flow-copy">Review the gauge, SPX signal, historical comparison, reasoning layer, and stock/bond split.</div></div>
        </div>
    </div>
        """,
        unsafe_allow_html=True,
    )

if not st.session_state.started:
    st.button("Start Analysis", use_container_width=True, on_click=start_app)
    st.stop()

with st.sidebar:
    st.markdown("## Analysis Flow")
    st.markdown("<div class='status-strip'>STEP 01 / CPI CONTEXT</div>", unsafe_allow_html=True)
    cpi_chart_df, cpi_source = cpi_trend_data(history)
    latest_cpi_value = float(cpi_chart_df["CPI_YoY"].iloc[-1]) if not cpi_chart_df.empty else latest.get("CPI_YoY")
    st.markdown(
        f"""
        <div class="sidebar-market-board">
            <div class="market-board-title">Recent CPI Trend</div>
            <div class="market-board-date">Past 12 months - latest {format_plain(latest_cpi_value)}% - {cpi_source}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(cpi_trend_chart(cpi_chart_df), use_container_width=True)
    st.markdown(f"<div class='placeholder-box'>{cpi_trend_summary(cpi_chart_df, cpi_source)}</div>", unsafe_allow_html=True)

    st.markdown("<div class='status-strip'>STEP 02 / INPUT SCENARIO</div>", unsafe_allow_html=True)
    st.slider(
        "Actual CPI YoY (%)",
        min_value=-5.0,
        max_value=20.0,
        step=0.10,
        key="actual_cpi_slider",
        help="The CPI inflation number that was actually released. YoY means year-over-year.",
        on_change=sync_actual_from_slider,
    )
    st.number_input(
        "Exact Actual CPI YoY (%)",
        label_visibility="collapsed",
        min_value=-5.0,
        max_value=20.0,
        step=0.10,
        format="%.2f",
        key="actual_cpi_input",
        help="Type an exact actual CPI value if the slider is not precise enough.",
        on_change=sync_actual_from_input,
    )
    st.slider(
        "Forecast CPI YoY (%)",
        min_value=-5.0,
        max_value=20.0,
        step=0.10,
        key="forecast_cpi_slider",
        help="The CPI number economists or the market expected before the release.",
        on_change=sync_forecast_from_slider,
    )
    st.number_input(
        "Exact Forecast CPI YoY (%)",
        label_visibility="collapsed",
        min_value=-5.0,
        max_value=20.0,
        step=0.10,
        format="%.2f",
        key="forecast_cpi_input",
        help="Type an exact forecast CPI value if the slider is not precise enough.",
        on_change=sync_forecast_from_input,
    )
    portfolio_amount = st.number_input(
        "Portfolio Amount ($)",
        min_value=0.0,
        max_value=1000000000.0,
        step=500.0,
        format="%.2f",
        key="portfolio_amount",
        help="The amount of money to split between stocks and bonds based on the model allocation tilt.",
        on_change=clear_analysis,
    )

    st.markdown("<div class='status-strip'>STEP 03 / LIVE MARKET BOARD</div>", unsafe_allow_html=True)
    spy_quote = fetch_live_market_quote("SPY")
    vix_quote = fetch_live_market_quote("^VIX")
    spx_return = spy_quote["change_pct"] if spy_quote.get("change_pct") is not None else latest.get("SP500_Daily_Return_Pct", latest.get("SP500_Return_Lag1", None))
    spx_label = f"${spy_quote['price']:.2f}" if spy_quote.get("price") is not None else "CSV"
    spx_class = "good" if pd.notna(spx_return) and float(spx_return) >= 0 else "bad"
    vix_value = vix_quote["change_pct"] if vix_quote.get("change_pct") is not None else latest.get("VIX_Return_Pct", None)
    vix_label = f"{vix_quote['price']:.2f}" if vix_quote.get("price") is not None else "CSV"
    vix_class = "bad" if pd.notna(vix_value) and float(vix_value) >= 0 else "good"
    live_source = "Live: Yahoo Finance" if spy_quote.get("price") is not None or vix_quote.get("price") is not None else "CSV fallback"
    st.markdown(
        f"""
        <div class="sidebar-market-board">
            <div class="market-board-title">Live Market Board</div>
            <div class="market-board-date">{live_source} - CPI/Fed from CSV {latest['Release_Date'].date()}</div>
            <div class="market-stat-row"><span>{term('SPY Price', 'The live price of SPY, an ETF commonly used as a tradable proxy for the S&P 500.')}</span><strong>{spx_label}</strong></div>
            <div class="market-stat-row"><span>{term('SPY 1D Change', 'The one-day percent move in SPY. Positive means the S&P 500 proxy is rising today.')}</span><strong class="{spx_class}">{format_pct(spx_return)}</strong></div>
            <div class="market-stat-row"><span>{term('VIX Level', 'The VIX is a market volatility index. Higher values usually mean investors expect more uncertainty.')}</span><strong>{vix_label}</strong></div>
            <div class="market-stat-row"><span>{term('VIX 1D Change', 'The one-day percent move in the VIX. A falling VIX often signals calmer market conditions.')}</span><strong class="{vix_class}">{format_pct(vix_value)}</strong></div>
            <div class="market-stat-row"><span>{term('Fed Funds', 'The Federal Funds Rate is the short-term policy rate influenced by the Federal Reserve.')}</span><strong>{format_plain(latest.get('FedFunds_Rate'))}%</strong></div>
            <div class="market-stat-row"><span>{term('Latest CPI YoY', 'The latest year-over-year inflation reading from the CPI data used by the model context.')}</span><strong>{format_plain(latest.get('CPI_YoY'))}%</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='status-strip'>STEP 04 / CONFIRM RUN</div>", unsafe_allow_html=True)
    run_analysis = st.button("Confirm & Run Models", use_container_width=True)

actual_cpi = float(st.session_state.actual_cpi)
forecast_cpi = float(st.session_state.forecast_cpi)
portfolio_amount = float(st.session_state.portfolio_amount)

if run_analysis:
    result = predict(actual_cpi=actual_cpi, forecast_cpi=forecast_cpi, recent_data=history)
    actual_surprise = float(result["actual_surprise"])
    base_allocation = allocation_signal(result, latest, actual_cpi)
    allocation = ai_allocation_signal(base_allocation, result, latest, actual_cpi, forecast_cpi)
    stocks_amount, bonds_amount = money_split(portfolio_amount, allocation)
    bullets, reasoning_source = openai_reasoning(actual_cpi, forecast_cpi, result, latest)
    st.session_state.analysis = {
        "result": result,
        "actual_surprise": actual_surprise,
        "allocation": allocation,
        "stocks_amount": stocks_amount,
        "bonds_amount": bonds_amount,
        "bullets": bullets,
        "reasoning_source": reasoning_source,
        "portfolio_amount": portfolio_amount,
    }

st.markdown("## Latest Data")
st.markdown(
    f"""
    <div class="context-grid">
        <div class="context-item"><div class="context-label">{term('Release Date', 'The date of the latest CPI release available in the CSV.')}</div><div class="context-value">{latest['Release_Date'].date()}</div></div>
        <div class="context-item"><div class="context-label">{term('Latest CPI YoY', 'The most recent year-over-year CPI inflation value in the dataset.')}</div><div class="context-value">{format_plain(latest['CPI_YoY'])}%</div></div>
        <div class="context-item"><div class="context-label">{term('VIX Return', 'Percent change in the VIX volatility index around the release.')}</div><div class="context-value">{format_pct(latest['VIX_Return_Pct'])}</div></div>
        <div class="context-item"><div class="context-label">{term('Fed Funds', 'The Federal Funds Rate, an important short-term interest rate.')}</div><div class="context-value">{format_plain(latest['FedFunds_Rate'])}%</div></div>
        <div class="context-item"><div class="context-label">{term('Inflation Regime', 'A low, medium, or high inflation environment label from the historical dataset.')}</div><div class="context-value">{latest['Inflation_Regime']}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.analysis is None:
    st.markdown(
        """
        <div class="status-strip">
            WAITING FOR CONFIRMATION: adjust the sidebar inputs, then press Confirm & Run Models to deploy the model flow and generate outputs.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

analysis = st.session_state.analysis
result = analysis["result"]
actual_surprise = analysis["actual_surprise"]
allocation = analysis["allocation"]
stocks_amount = analysis["stocks_amount"]
bonds_amount = analysis["bonds_amount"]
bullets = analysis["bullets"]
reasoning_source = analysis.get("reasoning_source", "Fallback reasoning")

st.markdown("## Model Outputs")
card_cols = st.columns(5, gap="medium")
with card_cols[0]:
    metric_card(f"Calculated {term('CPI Surprise', 'Actual CPI minus forecast CPI, measured in percentage points.')}", f"{actual_surprise:+.2f} pts", "actual CPI minus forecast CPI", tone_class(actual_surprise))
with card_cols[1]:
    metric_card(f"Expected {term('Surprise', 'Model 1 pre-release estimate of the CPI surprise before actual CPI is known.')}", f"{result['predicted_surprise']:+.2f} pts", "pre-release Model 1")
with card_cols[2]:
    metric_card(
        f"Expected {term('Direction', 'Model 2 pre-release class: Below, Match, or Above.')}",
        result["surprise_direction"],
        f"{result['surprise_confidence']:.1f}% pre-release confidence" if result["surprise_confidence"] else "fallback classification",
    )
with card_cols[3]:
    spx_class = "positive" if result["spx_direction"] == "Up" else "negative" if result["spx_direction"] == "Down" else "neutral"
    metric_card(
        f"Likely {term('SPX Reaction', 'The model prediction for whether the S&P 500 moves up or down after the CPI release.')}",
        result["spx_direction"],
        f"{result['spx_probability']:.1f}% model probability" if result["spx_probability"] else "fallback directional rule",
        spx_class,
    )
with card_cols[4]:
    metric_card(
        f"{term('Allocation Tilt', 'A model-based educational stock/bond split, not personalized financial advice.')}",
        allocation["allocation"],
        f"Stocks ${stocks_amount:,.0f} / Bonds ${bonds_amount:,.0f}",
        allocation["tone"],
    )

st.markdown(
    f"""
    <div class="allocation-guide">
        <div class="allocation-head">
            <div>
                <div class="allocation-title">Allocation Guide</div>
                <div class="allocation-subtitle">Current model/AI split plus common educational scenarios and purposes.</div>
            </div>
            <div>
                <div class="allocation-current">{allocation['allocation']}</div>
                <div class="allocation-money">Stocks ${stocks_amount:,.0f} / Bonds ${bonds_amount:,.0f}</div>
            </div>
        </div>
        <table class="allocation-table">
            <thead><tr><th>Situation</th><th>Purpose</th><th>Stock / Bond Split</th></tr></thead>
            <tbody>
                <tr><td>Current CPI + model setup</td><td>Blend allocation rules, model output, and AI reasoning for this dashboard scenario.</td><td>{allocation['allocation']}</td></tr>
                <tr><td>Young, long-term investing, 10+ years</td><td>Maximize long-run growth while accepting larger short-term swings.</td><td>80/20 or 90/10</td></tr>
                <tr><td>Moderate risk, 5-10 years</td><td>Balance growth from stocks with stability from bonds.</td><td>60/40</td></tr>
                <tr><td>Worried about short-term volatility or CPI shock</td><td>Reduce sensitivity to equity drawdowns and inflation-surprise volatility.</td><td>50/50 or 40/60</td></tr>
                <tr><td>Need money within 1-3 years</td><td>Prioritize liquidity and capital preservation over stock-market upside.</td><td>Mostly cash, T-bills, money market, short-term bonds</td></tr>
            </tbody>
        </table>
        <div class="allocation-note">{allocation['note']}. Educational research output only, not personalized financial advice.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
chart_left, chart_right = st.columns([0.42, 0.58], gap="large")
with chart_left:
    st.markdown(f"## {term('Surprise Gauge', 'A visual scale showing how far actual CPI was above or below forecast.')}", unsafe_allow_html=True)
    st.plotly_chart(gauge_chart(actual_surprise, history), use_container_width=True)
    st.plotly_chart(probability_bar(result), use_container_width=True)

with chart_right:
    st.markdown(f"## {term('Historical Comparison', 'Past CPI surprises plotted against one-day S&P 500 returns. Similar events are highlighted.')}", unsafe_allow_html=True)
    st.plotly_chart(scatter_chart(history, actual_surprise, result), use_container_width=True)

reason_col, diag_col = st.columns([0.58, 0.42], gap="large")
with reason_col:
    st.markdown(f"## {term('Reasoning Layer', 'Four causal explanations that describe why the CPI setup could matter for stocks, bonds, rates, and risk appetite.')}", unsafe_allow_html=True)
    st.markdown(
        "<ul class='reasoning-list'>" + "".join(f"<li>{bullet}</li>" for bullet in bullets[:4]) + "</ul>",
        unsafe_allow_html=True,
    )

with diag_col:
    st.markdown(f"## {term('Model Diagnostics', 'Simple model quality and feature-importance checks from the available CSV rows.')}", unsafe_allow_html=True)
    acc = evaluate_models(load_models(), history)
    result["model_accuracy"] = acc
    acc_cols = st.columns(2)
    with acc_cols[0]:
        metric_card(
            "Expected Direction Accuracy",
            "N/A" if acc["surprise_direction_acc"] is None else f"{acc['surprise_direction_acc'] * 100:.1f}%",
            "computed from available CSV rows",
        )
    with acc_cols[1]:
        metric_card(
            "SPX Direction Accuracy",
            "N/A" if acc["spx_direction_acc"] is None else f"{acc['spx_direction_acc'] * 100:.1f}%",
            "computed from available CSV rows",
        )

    if result["top_features"]:
        feature_df = pd.DataFrame(result["top_features"])
        feature_df["importance"] = feature_df["importance"].map(lambda value: f"{value * 100:.1f}%")
        st.dataframe(feature_df, use_container_width=True, hide_index=True)
    else:
        st.caption("Feature importance is unavailable for the loaded SPX model.")

    st.caption(f"Allocation note: {allocation['note']}.")
    st.caption("Research prototype only. Allocation tilt is a model-based educational signal, not personalized investment advice.")

