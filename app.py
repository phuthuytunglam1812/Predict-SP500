from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from predict_helper import DATA_PATH, FEATURE_ORDER, load_data, predict


load_dotenv()
load_dotenv("app.env", override=False)

st.set_page_config(
    page_title="CPI x SPX Impact Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)


CSS = """
<style>
    :root {
        --bg: #eef2f6;
        --panel: #ffffff;
        --ink: #111827;
        --muted: #526071;
        --line: #cbd5e1;
        --accent: #0f766e;
        --accent-soft: #dff3ef;
        --danger: #b23a48;
        --success: #1f7a3f;
        --warn: #a16207;
    }
    .stApp {
        background: var(--bg);
        color: var(--ink);
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.2rem;
        max-width: 1320px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
        color: var(--ink);
    }
    h1 {
        font-size: 2rem;
        margin-bottom: 0.15rem;
    }
    h2 {
        font-size: 1.05rem;
        margin-top: 0.25rem;
        margin-bottom: 0.7rem;
    }
    .subtle {
        color: var(--muted);
        font-size: 0.9rem;
        margin-bottom: 1.1rem;
    }
    .metric-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        height: 124px;
        justify-content: space-between;
        padding: 0.9rem 1rem;
    }
    .metric-label {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.055em;
        line-height: 1.2;
        margin-bottom: 0.15rem;
    }
    .metric-value {
        color: var(--ink);
        font-size: 1.55rem;
        line-height: 1.12;
        font-weight: 800;
        overflow-wrap: anywhere;
    }
    .metric-note {
        color: var(--muted);
        font-size: 0.76rem;
        line-height: 1.25;
        margin-top: 0.2rem;
    }
    .positive {
        color: var(--success);
    }
    .negative {
        color: var(--danger);
    }
    .neutral {
        color: var(--warn);
    }
    .section {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1rem;
        margin-top: 1rem;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
    }
    div[data-testid="stPlotlyChart"] {
        background:
            linear-gradient(135deg, rgba(15, 118, 110, 0.09), rgba(255, 255, 255, 0) 34%),
            #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 22px;
        box-shadow: none;
        padding: 0.8rem;
        overflow: hidden;
    }
    div[data-testid="stPlotlyChart"] > div {
        border-radius: 18px;
        overflow: hidden;
    }
    div[data-testid="stPlotlyChart"] .js-plotly-plot,
    div[data-testid="stPlotlyChart"] .plot-container,
    div[data-testid="stPlotlyChart"] .svg-container,
    div[data-testid="stPlotlyChart"] .main-svg {
        background: transparent !important;
        border-radius: 18px !important;
        overflow: hidden !important;
    }
    .context-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.65rem;
    }
    .context-item {
        border-left: 3px solid var(--accent);
        background: #fafbfc;
        padding: 0.55rem 0.7rem;
        min-height: 70px;
    }
    .context-label {
        color: var(--muted);
        font-size: 0.76rem;
        margin-bottom: 0.2rem;
    }
    .context-value {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 650;
    }
    .reasoning-list {
        margin: 0;
        padding-left: 1.1rem;
    }
    .reasoning-list li {
        margin: 0.38rem 0;
    }
    .placeholder-box {
        border: 1px dashed #aab4c0;
        background: #fbfcfd;
        border-radius: 8px;
        padding: 0.85rem;
        color: var(--muted);
    }
    .stButton > button {
        border-radius: 8px;
        border: 1px solid var(--accent);
        color: var(--accent);
        background: #ffffff;
        min-height: 2.45rem;
    }
    .stButton > button[kind="primary"] {
        background: var(--accent);
        color: #ffffff;
    }
    div[data-testid="stNumberInput"] label,
    div[data-testid="stNumberInput"] label p {
        color: var(--ink) !important;
        font-size: 0.9rem !important;
        font-weight: 650 !important;
        opacity: 1 !important;
    }
    div[data-testid="stNumberInput"] input {
        background: #ffffff !important;
        color: var(--ink) !important;
        border-color: var(--line) !important;
    }
    @media (max-width: 900px) {
        .context-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .metric-value {
            font-size: 1.35rem;
        }
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
    lower = min(-0.6, float(history["CPI_Surprise_Pct"].min()) - 0.05)
    upper = max(0.6, float(history["CPI_Surprise_Pct"].max()) + 0.05)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=actual_surprise,
            number={"suffix": " pts", "font": {"size": 36, "color": "#111827"}},
            gauge={
                "axis": {
                    "range": [lower, upper],
                    "tickwidth": 2,
                    "tickcolor": "#334155",
                    "tickfont": {"color": "#334155", "size": 13},
                },
                "bar": {"color": "#0f766e", "thickness": 0.36},
                "bgcolor": "#ffffff",
                "borderwidth": 1,
                "bordercolor": "#cbd5e1",
                "steps": [
                    {"range": [lower, -0.1], "color": "#c7ead9"},
                    {"range": [-0.1, 0.1], "color": "#f5df9e"},
                    {"range": [0.1, upper], "color": "#f2c4cb"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 5},
                    "thickness": 0.75,
                    "value": actual_surprise,
                },
            },
            title={"text": "Actual CPI Surprise", "font": {"color": "#334155", "size": 14}},
        )
    )
    fig.update_layout(
        height=300,
        margin={"l": 24, "r": 24, "t": 50, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#111827"},
    )
    return fig


def scatter_chart(history: pd.DataFrame, actual_surprise: float, result: dict) -> go.Figure:
    chart_df = history.copy()
    chart_df["Similar"] = "Historical"
    similar_dates = set(result["similar_events"]["Release_Date"]) if len(result["similar_events"]) else set()
    chart_df.loc[chart_df["Release_Date"].isin(similar_dates), "Similar"] = "Similar setup"

    fig = go.Figure()
    base = chart_df[chart_df["Similar"] == "Historical"]
    similar = chart_df[chart_df["Similar"] == "Similar setup"]

    fig.add_trace(
        go.Scatter(
            x=base["CPI_Surprise_Pct"],
            y=base["SP500_Daily_Return_Pct"],
            mode="markers",
            name="Historical",
            marker={"color": "#64748b", "size": 9, "opacity": 0.74},
            text=base["Release_Date"].dt.strftime("%Y-%m-%d"),
            hovertemplate="Date %{text}<br>CPI surprise %{x:.2f}<br>SPX 1D %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=similar["CPI_Surprise_Pct"],
            y=similar["SP500_Daily_Return_Pct"],
            mode="markers",
            name="Similar setup",
            marker={"color": "#0f766e", "size": 13, "line": {"color": "#ffffff", "width": 1.5}},
            text=similar["Release_Date"].dt.strftime("%Y-%m-%d"),
            hovertemplate="Date %{text}<br>CPI surprise %{x:.2f}<br>SPX 1D %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_vline(x=actual_surprise, line_width=3, line_dash="dash", line_color="#b23a48")
    fig.add_hline(y=0, line_width=2, line_color="#94a3b8")
    fig.update_layout(
        height=370,
        margin={"l": 10, "r": 10, "t": 28, "b": 10},
        xaxis_title="CPI surprise",
        yaxis_title="SPX 1D return (%)",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"color": "#111827", "size": 13},
            "bgcolor": "rgba(255,255,255,0.70)",
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.72)",
        font={"color": "#334155", "size": 13},
        xaxis={
            "gridcolor": "#d7dee8",
            "zerolinecolor": "#94a3b8",
            "linecolor": "#334155",
            "tickfont": {"color": "#334155"},
            "title": {"font": {"color": "#334155"}},
        },
        yaxis={
            "gridcolor": "#d7dee8",
            "zerolinecolor": "#94a3b8",
            "linecolor": "#334155",
            "tickfont": {"color": "#334155"},
            "title": {"font": {"color": "#334155"}},
        },
    )
    return fig


def probability_bar(result: dict) -> go.Figure:
    probability = float(result.get("spx_probability") or 0.0)
    direction = result.get("spx_direction", "N/A")
    color = "#1f7a3f" if direction == "Up" else "#b23a48" if direction == "Down" else "#a16207"
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
            "gridcolor": "#e2e8f0",
            "tickfont": {"color": "#334155"},
        },
        yaxis={"title": None, "tickfont": {"color": "#334155", "size": 13}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.72)",
        font={"color": "#334155"},
        showlegend=False,
    )
    return fig


def allocation_signal(result: dict) -> dict[str, str | int]:
    direction = result.get("spx_direction", "Neutral")
    probability = float(result.get("spx_probability") or 0.0)

    if probability == 0.0:
        return {
            "allocation": "60/40",
            "stocks_pct": 60,
            "bonds_pct": 40,
            "tone": "neutral",
            "note": "fallback neutral stock/bond mix",
        }

    if direction == "Up":
        if probability >= 80:
            return {"allocation": "80/20", "stocks_pct": 80, "bonds_pct": 20, "tone": "positive", "note": "strong risk-on model tilt"}
        if probability >= 65:
            return {"allocation": "70/30", "stocks_pct": 70, "bonds_pct": 30, "tone": "positive", "note": "moderate risk-on model tilt"}
        return {"allocation": "60/40", "stocks_pct": 60, "bonds_pct": 40, "tone": "neutral", "note": "mild risk-on signal"}

    if direction == "Down":
        if probability >= 80:
            return {"allocation": "30/70", "stocks_pct": 30, "bonds_pct": 70, "tone": "negative", "note": "strong defensive model tilt"}
        if probability >= 65:
            return {"allocation": "40/60", "stocks_pct": 40, "bonds_pct": 60, "tone": "negative", "note": "moderate defensive model tilt"}
        return {"allocation": "50/50", "stocks_pct": 50, "bonds_pct": 50, "tone": "neutral", "note": "mild defensive signal"}

    return {"allocation": "60/40", "stocks_pct": 60, "bonds_pct": 40, "tone": "neutral", "note": "neutral model signal"}


def money_split(amount: float, allocation: dict[str, str | int]) -> tuple[float, float]:
    stocks_amount = float(amount) * int(allocation["stocks_pct"]) / 100
    bonds_amount = float(amount) * int(allocation["bonds_pct"]) / 100
    return stocks_amount, bonds_amount


def fallback_reasoning(actual_cpi: float, forecast_cpi: float, result: dict, latest: pd.Series) -> list[str]:
    surprise = result["actual_surprise"]
    similar_count = result["similar_events_count"]
    up = result["similar_events_up"]
    down = result["similar_events_down"]
    direction_word = "above" if surprise > 0 else "below" if surprise < 0 else "in line with"
    spx_direction = result["spx_direction"]
    probability = result["spx_probability"]

    return [
        f"Actual CPI came in {direction_word} forecast by {abs(surprise):.2f} percentage points, using {actual_cpi:.2f} actual versus {forecast_cpi:.2f} forecast.",
        f"The latest market backdrop comes from {latest['Release_Date'].date()}: VIX return {latest['VIX_Return_Pct']:.2f}% and Fed funds {latest['FedFunds_Rate']:.2f}%.",
        f"The SPX model points to {spx_direction} with {probability:.1f}% confidence, while the pre-release model expected a surprise of {result['predicted_surprise']:+.2f} points.",
        f"Historically similar inflation-regime and surprise-size events total {similar_count}, with {up} up sessions and {down} down sessions.",
    ]


def openai_reasoning(actual_cpi: float, forecast_cpi: float, result: dict, latest: pd.Series) -> list[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", None)
        except Exception:
            api_key = None
    if not api_key:
        return fallback_reasoning(actual_cpi, forecast_cpi, result, latest)

    try:
        from openai import OpenAI

        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        try:
            model = st.secrets.get("OPENAI_MODEL", model)
        except Exception:
            pass

        client = OpenAI(api_key=api_key)
        prompt = f"""
        Create exactly 4 concise English bullets for a clean finance dashboard.
        Inputs:
        - Actual CPI: {actual_cpi}
        - Forecast CPI: {forecast_cpi}
        - Actual CPI surprise: {result['actual_surprise']}
        - Pre-release expected surprise: {result['predicted_surprise']}
        - Pre-release expected direction model: {result['surprise_direction']} at {result['surprise_confidence']:.1f}%
        - SPX impact model: {result['spx_direction']} at {result['spx_probability']:.1f}%
        - Similar historical events: {result['similar_events_count']}
        - Similar up/down: {result['similar_events_up']} up, {result['similar_events_down']} down
        - Latest VIX return: {latest['VIX_Return_Pct']}
        - Latest Fed funds: {latest['FedFunds_Rate']}
        Avoid investment advice and do not add a disclaimer.
        """
        response = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=260,
        )
        text = response.output_text
        bullets = [line.lstrip("-* ").strip() for line in text.splitlines() if line.strip()]
        return bullets[:4] or fallback_reasoning(actual_cpi, forecast_cpi, result, latest)
    except Exception:
        return fallback_reasoning(actual_cpi, forecast_cpi, result, latest)


history = cached_data()
latest = history.sort_values("Release_Date").iloc[-1]

if "actual_cpi" not in st.session_state:
    st.session_state.actual_cpi = 4.30
if "forecast_cpi" not in st.session_state:
    st.session_state.forecast_cpi = 4.20
if "portfolio_amount" not in st.session_state:
    st.session_state.portfolio_amount = 10000.0


def load_sample_scenario() -> None:
    st.session_state.actual_cpi = 4.30
    st.session_state.forecast_cpi = 4.20
    st.session_state.portfolio_amount = 10000.0


st.title("CPI x SPX Impact Dashboard")
st.markdown(
    '<div class="subtle">Hybrid CPI surprise and SPX reaction view using the Week 2 models.</div>',
    unsafe_allow_html=True,
)

input_col, action_col = st.columns([0.72, 0.28], gap="large")

with input_col:
    st.markdown("## Release Inputs")
    actual_cpi = st.number_input(
        "Actual CPI YoY (%)",
        min_value=-5.0,
        max_value=20.0,
        step=0.10,
        format="%.2f",
        key="actual_cpi",
    )
    forecast_cpi = st.number_input(
        "Forecast CPI YoY (%)",
        min_value=-5.0,
        max_value=20.0,
        step=0.10,
        format="%.2f",
        key="forecast_cpi",
    )
    portfolio_amount = st.number_input(
        "Portfolio Amount ($)",
        min_value=0.0,
        max_value=1000000000.0,
        step=500.0,
        format="%.2f",
        key="portfolio_amount",
    )

with action_col:
    st.markdown("## Scenario")
    st.button("Use Sample Scenario", use_container_width=True, on_click=load_sample_scenario)
    st.markdown(
        """
        <div class="placeholder-box">
            Auto-updating market and CPI inputs will connect here later through API keys or scheduled data refresh.
        </div>
        """,
        unsafe_allow_html=True,
    )

result = predict(actual_cpi=actual_cpi, forecast_cpi=forecast_cpi, recent_data=history)
actual_surprise = float(result["actual_surprise"])

st.markdown("## Latest Data Context")
st.markdown(
    f"""
    <div class="context-grid">
        <div class="context-item"><div class="context-label">Release Date</div><div class="context-value">{latest['Release_Date'].date()}</div></div>
        <div class="context-item"><div class="context-label">Latest CPI YoY</div><div class="context-value">{format_plain(latest['CPI_YoY'])}%</div></div>
        <div class="context-item"><div class="context-label">VIX Return</div><div class="context-value">{format_pct(latest['VIX_Return_Pct'])}</div></div>
        <div class="context-item"><div class="context-label">Fed Funds</div><div class="context-value">{format_plain(latest['FedFunds_Rate'])}%</div></div>
        <div class="context-item"><div class="context-label">Inflation Regime</div><div class="context-value">{latest['Inflation_Regime']}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

allocation = allocation_signal(result)
stocks_amount, bonds_amount = money_split(portfolio_amount, allocation)

card_cols = st.columns(5, gap="medium")
with card_cols[0]:
    metric_card("Calculated CPI Surprise", f"{actual_surprise:+.2f} pts", "actual CPI minus forecast CPI", tone_class(actual_surprise))
with card_cols[1]:
    metric_card("Expected Surprise", f"{result['predicted_surprise']:+.2f} pts", "pre-release Model 1")
with card_cols[2]:
    metric_card(
        "Expected Direction",
        result["surprise_direction"],
        f"{result['surprise_confidence']:.1f}% pre-release confidence" if result["surprise_confidence"] else "fallback classification",
    )
with card_cols[3]:
    spx_class = "positive" if result["spx_direction"] == "Up" else "negative" if result["spx_direction"] == "Down" else "neutral"
    metric_card(
        "Likely SPX Reaction",
        result["spx_direction"],
        f"{result['spx_probability']:.1f}% model probability" if result["spx_probability"] else "fallback directional rule",
        spx_class,
    )
with card_cols[4]:
    metric_card(
        "Model Allocation Tilt",
        allocation["allocation"],
        f"Stocks ${stocks_amount:,.0f} / Bonds ${bonds_amount:,.0f}",
        allocation["tone"],
    )

chart_left, chart_right = st.columns([0.42, 0.58], gap="large")
with chart_left:
    st.markdown("## Surprise Gauge")
    st.plotly_chart(gauge_chart(actual_surprise, history), use_container_width=True)
    st.plotly_chart(probability_bar(result), use_container_width=True)

with chart_right:
    st.markdown("## Historical Comparison")
    st.plotly_chart(scatter_chart(history, actual_surprise, result), use_container_width=True)

reason_col, diag_col = st.columns([0.58, 0.42], gap="large")
with reason_col:
    st.markdown("## Reasoning Layer")
    bullets = openai_reasoning(actual_cpi, forecast_cpi, result, latest)
    st.markdown(
        "<ul class='reasoning-list'>" + "".join(f"<li>{bullet}</li>" for bullet in bullets[:4]) + "</ul>",
        unsafe_allow_html=True,
    )

with diag_col:
    st.markdown("## Model Diagnostics")
    acc = result["model_accuracy"]
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
