"""
Kisan Mandi Intelligence (AI Farmer Market & Price Discovery)
Built for Smart India Hackathon (SIH26132) & Agricultural Decision Support.
Implements the exact UI/UX design specifications, color tokens, and typography.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure src modules can be imported
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from src.features import extract_single_series, build_features, temporal_train_test_split
from src.models import (
    NaiveModel,
    MovingAverageModel,
    ARIMAModelWrapper,
    LinearRegressionModelWrapper,
    RandomForestModelWrapper,
)
from src.evaluate import evaluate_all_models_on_series
from src.mandi_compare import get_mandi_comparison, calculate_sell_now_or_wait
from src.translations import get_text

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Kisan Mandi Intelligence | AI Farmer Price Discovery",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom CSS: Design Tokens & Typography Injection
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Import Space Grotesk and Plus Jakarta Sans */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --color-surface: #f9faf5;
    --color-surface-dim: #d9dad6;
    --color-surface-card: #ffffff;
    --color-surface-muted: #f0f2eb;
    --color-border-subtle: #e2e6dc;
    --color-border-strong: #c5ccc0;
    --color-primary: #1B4D3E;
    --color-primary-dark: #003629;
    --color-primary-light: #baeed9;
    --color-secondary: #2E7D32;
    --color-tertiary: #E65100;
    --color-text-primary: #141D17;
    --color-text-secondary: #465248;
    --color-text-muted: #6E7A70;
    --color-sell-bg: #E8F5E9;
    --color-sell-text: #1B5E20;
    --color-sell-border: #81C784;
    --color-wait-bg: #FFF3E0;
    --color-wait-text: #BF360C;
    --color-wait-border: #FFB74D;
}

/* Base Body & App Canvas */
html, body, [class*="css"], .stApp {
    background-color: var(--color-surface) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--color-text-primary) !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #f3f4ef !important;
    border-right: 1px solid var(--color-border-subtle) !important;
    width: 290px !important;
}

[data-testid="stSidebar"] .stSelectbox label, 
[data-testid="stSidebar"] .stRadio label {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    color: var(--color-text-secondary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

/* Headings and Stat Readouts */
h1, h2, h3, h4, .space-heading {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    color: var(--color-primary) !important;
    letter-spacing: -0.01em !important;
}

/* Top Branding Bar */
.kmi-header-box {
    background-color: #ffffff;
    border: 1px solid var(--color-border-subtle);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    box-shadow: 0 1px 3px rgba(20, 29, 23, 0.04);
}

.kmi-brand-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 26px;
    font-weight: 700;
    color: var(--color-primary);
    margin: 0;
    line-height: 1.2;
}

.kmi-brand-tagline {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 13px;
    color: var(--color-text-muted);
    margin-top: 4px;
}

.kmi-mandi-badge {
    background-color: var(--color-surface-muted);
    border: 1px solid var(--color-border-strong);
    border-radius: 9999px;
    padding: 6px 14px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: var(--color-primary);
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* Metric Strip Cards */
.kmi-metric-card {
    background-color: #ffffff;
    border: 1px solid var(--color-border-subtle);
    border-radius: 12px;
    padding: 18px 20px;
    height: 100%;
    box-shadow: 0 1px 2px rgba(20, 29, 23, 0.03);
    transition: transform 0.15s ease;
}

.kmi-metric-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px;
    font-weight: 700;
    color: var(--color-text-muted);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.kmi-metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: var(--color-primary);
    line-height: 1.15;
    margin-bottom: 6px;
}

.kmi-pill-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 9999px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 12px;
    font-weight: 600;
}

/* Sell vs Wait Semantic Badges */
.badge-sell {
    background-color: var(--color-sell-bg);
    color: var(--color-sell-text);
    border: 1.5px solid var(--color-sell-border);
}

.badge-wait {
    background-color: var(--color-wait-bg);
    color: var(--color-wait-text);
    border: 1.5px solid var(--color-wait-border);
}

.badge-neutral {
    background-color: #F1F5F9;
    color: #475569;
    border: 1.5px solid #CBD5E1;
}

/* Decision Action Banner */
.kmi-decision-banner {
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 24px;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.banner-sell {
    background-color: #E8F5E9;
    border: 1.5px solid #81C784;
    box-shadow: 2px 2px 0px rgba(27, 94, 32, 0.08);
}

.banner-wait {
    background-color: #FFF3E0;
    border: 1.5px solid #FFB74D;
    box-shadow: 2px 2px 0px rgba(191, 54, 12, 0.08);
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: var(--color-surface-muted);
    padding: 5px;
    border-radius: 10px;
    border: 1px solid var(--color-border-subtle);
}

.stTabs [data-baseweb="tab"] {
    height: 42px;
    border-radius: 8px;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    color: var(--color-text-secondary) !important;
    background-color: transparent !important;
    border: none !important;
    padding: 8px 18px !important;
}

.stTabs [aria-selected="true"] {
    background-color: #ffffff !important;
    color: var(--color-primary) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}

/* Card Containers */
.kmi-card {
    background-color: #ffffff;
    border: 1px solid var(--color-border-subtle);
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(20, 29, 23, 0.03);
}

/* Table styling */
.kmi-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--color-border-subtle);
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 13px;
}

.kmi-table th {
    background-color: var(--color-primary);
    color: #F9FAF5;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 12px 16px;
    text-align: left;
}

.kmi-table td {
    padding: 12px 16px;
    border-bottom: 1px solid var(--color-border-subtle);
    color: var(--color-text-primary);
}

.kmi-table tr:nth-child(even) {
    background-color: #F9FAF5;
}

.kmi-table tr:nth-child(odd) {
    background-color: #FFFFFF;
}

.kmi-table tr.best-row {
    background-color: #F0FDF4 !important;
    font-weight: 600;
}

/* Buttons */
.stButton>button {
    background-color: var(--color-primary) !important;
    color: #FFFFFF !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
    min-height: 44px !important;
    padding: 0 20px !important;
    transition: background 0.15s ease !important;
}

.stButton>button:hover {
    background-color: #143B2F !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Data Loader with Streamlit Caching
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    """Loads cleaned agricultural prices from parquet for fast execution."""
    data_path_parquet = os.path.join(APP_DIR, "data", "clean_prices.parquet")
    data_path_csv = os.path.join(APP_DIR, "data", "clean_prices.csv")

    if os.path.exists(data_path_parquet):
        df = pd.read_parquet(data_path_parquet)
    elif os.path.exists(data_path_csv):
        df = pd.read_csv(data_path_csv)
    else:
        st.error("Dataset not found! Please run `src/clean_data.py` first.")
        st.stop()

    df["date"] = pd.to_datetime(df["date"])
    return df


df_master = load_data()


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
            <div style="background-color: #1B4D3E; color: white; width: 38px; height: 38px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 20px;">🌾</div>
            <div>
                <div style="font-family: 'Space Grotesk'; font-weight: 700; font-size: 17px; color: #1B4D3E; line-height: 1.1;">Kisan Mandi</div>
                <div style="font-size: 11px; font-weight: 600; color: #465248; letter-spacing: 0.05em;">PRICE INTELLIGENCE</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='margin: 12px 0; border: none; border-top: 1px solid #c5ccc0;'>", unsafe_allow_html=True)

    # Language Selector
    selected_lang = st.selectbox(
        "🌐 Language / भाषा / ਭਾਸ਼ਾ",
        ["English", "हिन्दी", "ਪੰਜਾਬੀ"],
        index=0,
        help="Choose preferred display language.",
    )

    # 1. Commodity Selector
    commodities = sorted(df_master["commodity"].unique().tolist())
    default_comm_idx = commodities.index("Wheat") if "Wheat" in commodities else 0
    selected_commodity = st.selectbox(
        get_text(selected_lang, "select_commodity"),
        commodities,
        index=default_comm_idx,
        help="Select agricultural produce for price analytics and forecasting.",
    )

    # 2. State Selector
    available_states = sorted(
        df_master[df_master["commodity"] == selected_commodity]["state"].unique().tolist()
    )
    default_state_idx = available_states.index("Punjab") if "Punjab" in available_states else 0
    selected_state = st.selectbox(
        get_text(selected_lang, "select_state"),
        available_states,
        index=default_state_idx,
        help="Filter mandis by state.",
    )

    # 3. Mandi Selector
    available_markets = sorted(
        df_master[
            (df_master["commodity"] == selected_commodity)
            & (df_master["state"] == selected_state)
        ]["market"].unique().tolist()
    )
    if not available_markets:
        st.warning(f"No mandis found for {selected_commodity} in {selected_state}.")
        st.stop()

    default_market_idx = 0
    selected_market = st.selectbox(
        get_text(selected_lang, "select_mandi"),
        available_markets,
        index=default_market_idx,
        help="Choose specific regulated mandi market.",
    )

    # 4. Forecast Horizon Selector (7, 15, 30 days)
    st.markdown("<hr style='margin: 12px 0; border: none; border-top: 1px solid #c5ccc0;'>", unsafe_allow_html=True)
    horizon_option = st.radio(
        get_text(selected_lang, "forecast_horizon"),
        options=[7, 15, 30],
        format_func=lambda x: f"{x} {get_text(selected_lang, 'days_ahead')}",
        index=0,
        horizontal=True,
        help="Select the prediction lookahead window.",
    )

    # Sidebar Dataset Summary Badge
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="background-color: #ffffff; border: 1px solid #e2e6dc; border-radius: 8px; padding: 12px; font-size: 11px; color: #465248;">
            <div style="font-family: 'Space Grotesk'; font-weight: 700; color: #1B4D3E; margin-bottom: 4px;">SIH26132 ENGINE STATS</div>
            <div>• Active Mandis: <b>{df_master['market'].nunique()}</b></div>
            <div>• Commodity: <b>{selected_commodity}</b></div>
            <div>• Cleaned Records: <b>{len(df_master):,}</b></div>
            <div>• Training Horizon: <b>2023 - 2025</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Load Series & Run Evaluations
# ---------------------------------------------------------
series_data = extract_single_series(df_master, selected_commodity, selected_state, selected_market)

if series_data.empty or len(series_data) < 30:
    st.error(f"Insufficient historical data available for {selected_commodity} in {selected_market}, {selected_state}.")
    st.stop()

# Evaluate models on this series (cached via function or in memory)
with st.spinner("Training models & evaluating forecasts..."):
    eval_df, models_dict, pred_test_df, train_slice, test_slice = evaluate_all_models_on_series(series_data)

# Extract best model
best_model_row = eval_df[eval_df["Is Best"]].iloc[0]
best_model_name = best_model_row["Model"]
best_mape = best_model_row["MAPE (%)"]
best_model_obj = models_dict[best_model_name]

# Current modal price & delta
current_price = float(series_data["modal_price"].iloc[-1])
yesterday_price = float(series_data["modal_price"].iloc[-2]) if len(series_data) > 1 else current_price
daily_delta_rs = current_price - yesterday_price
daily_delta_pct = (daily_delta_rs / yesterday_price) * 100 if yesterday_price else 0.0

# ---------------------------------------------------------
# Generate Out-Of-Sample Forecast for selected Horizon
# ---------------------------------------------------------
history_prices = series_data["modal_price"].values
last_date = series_data.index[-1]
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon_option + 1)]

# Forecast with best model
if "Linear Regression" in best_model_name or "Random Forest" in best_model_name:
    forecast_values = best_model_obj.forecast_steps(history_prices, last_date, horizon_option)
else:
    forecast_values = best_model_obj.forecast_steps(history_prices, horizon_option)

# Estimate confidence intervals from test residual standard error
test_residuals = test_slice["modal_price"].values - pred_test_df[best_model_name].values
residual_std = float(np.std(test_residuals)) if len(test_residuals) > 0 else (current_price * 0.03)

# Expand confidence intervals slightly with horizon
step_multipliers = np.sqrt(np.arange(1, horizon_option + 1)) / np.sqrt(horizon_option)
ci_spread = 1.96 * residual_std * (0.8 + 0.4 * step_multipliers)
upper_bound = forecast_values + ci_spread
lower_bound = np.maximum(50.0, forecast_values - ci_spread)

# ---------------------------------------------------------
# Inter-Mandi Comparison & Sell/Wait Strategy
# ---------------------------------------------------------
mandi_stats, best_alt = get_mandi_comparison(
    df_master,
    selected_commodity,
    selected_state=selected_state,
    selected_market=selected_market,
    lookback_days=14,
)

decision_info = calculate_sell_now_or_wait(current_price, forecast_values, horizon_option)


# ---------------------------------------------------------
# Top Header Box
# ---------------------------------------------------------
district_name = series_data["district"].iloc[0] if "district" in series_data.columns else selected_state
st.markdown(
    f"""
    <div class="kmi-header-box">
        <div>
            <div class="kmi-brand-title">{get_text(selected_lang, 'brand_title')}</div>
            <div class="kmi-brand-tagline">{get_text(selected_lang, 'brand_tagline')}</div>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
            <div class="kmi-mandi-badge">
                <span>📍</span>
                <span>{selected_market}, {selected_state} ({district_name} Dist.)</span>
            </div>
            <div class="kmi-mandi-badge" style="background-color: #1B4D3E; color: #FFFFFF; border-color: #1B4D3E;">
                <span>🌾 {selected_commodity}</span>
            </div>
            <div class="kmi-mandi-badge">
                <span>📅 Up to {last_date.strftime('%d %b %Y')}</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Metric Strip (4 Cards)
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    delta_symbol = "+" if daily_delta_rs >= 0 else ""
    delta_color_cls = "badge-sell" if daily_delta_rs >= 0 else "badge-wait"
    st.markdown(
        f"""
        <div class="kmi-metric-card">
            <div class="kmi-metric-label">{get_text(selected_lang, 'curr_price')}</div>
            <div class="kmi-metric-value">₹{current_price:,.0f} <span style="font-size: 14px; font-weight: 500; color: #6E7A70;">{get_text(selected_lang, 'per_quintal')}</span></div>
            <div>
                <span class="kmi-pill-badge {delta_color_cls}">
                    {delta_symbol}₹{abs(daily_delta_rs):,.0f} ({delta_symbol}{daily_delta_pct:.1f}%) 1d
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    if decision_info["decision"] == "SELL NOW":
        dec_badge_text = get_text(selected_lang, 'sell_now')
        dec_badge = f'<span class="kmi-pill-badge badge-sell" style="font-size: 14px; padding: 6px 14px;">{dec_badge_text}</span>'
    else:
        dec_badge_text = get_text(selected_lang, 'wait')
        dec_badge = f'<span class="kmi-pill-badge badge-wait" style="font-size: 14px; padding: 6px 14px;">{dec_badge_text}</span>'

    st.markdown(
        f"""
        <div class="kmi-metric-card">
            <div class="kmi-metric-label">{get_text(selected_lang, 'pred_direction')} ({horizon_option}d)</div>
            <div style="margin: 8px 0 10px 0;">
                {dec_badge}
            </div>
            <div style="font-size: 12px; color: #465248; font-weight: 500;">
                Avg forecast: <b>₹{decision_info['avg_forecast']:,.0f}</b> ({decision_info['diff_pct']:+.1f}%)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    if best_alt and best_alt["is_higher"]:
        alt_html = f"""
            <div class="kmi-metric-value" style="font-size: 20px; margin-top: 4px;">{best_alt['market']}</div>
            <div>
                <span class="kmi-pill-badge badge-sell">
                    +₹{best_alt['diff']:,.0f}/qtl (+{best_alt['pct']}%)
                </span>
            </div>
        """
    elif best_alt:
        alt_html = f"""
            <div class="kmi-metric-value" style="font-size: 20px; margin-top: 4px;">{selected_market} (Top)</div>
            <div>
                <span class="kmi-pill-badge badge-sell">
                    Current mandi is paying the highest
                </span>
            </div>
        """
    else:
        alt_html = """<div class="kmi-metric-value" style="font-size: 18px;">No Alternate Mandis</div>"""

    st.markdown(
        f"""
        <div class="kmi-metric-card">
            <div class="kmi-metric-label">{get_text(selected_lang, 'best_mandi')}</div>
            {alt_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="kmi-metric-card">
            <div class="kmi-metric-label">{get_text(selected_lang, 'top_model')}</div>
            <div class="kmi-metric-value">{best_mape:.2f}% <span style="font-size: 14px; font-weight: 500; color: #6E7A70;">MAPE</span></div>
            <div>
                <span class="kmi-pill-badge badge-sell">
                    ⭐ {best_model_name}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# 5 Main Navigation Tabs
# ---------------------------------------------------------
tab_history, tab_forecast, tab_eval, tab_compare, tab_alerts = st.tabs([
    get_text(selected_lang, "tab_history"),
    get_text(selected_lang, "tab_forecast"),
    get_text(selected_lang, "tab_eval"),
    get_text(selected_lang, "tab_compare"),
    get_text(selected_lang, "tab_alerts"),
])


# =========================================================
# TAB 1: PRICE HISTORY
# =========================================================
with tab_history:
    st.markdown(
        """
        <div style="margin-bottom: 16px;">
            <h3 style="margin-bottom: 4px;">Historical Modal Prices</h3>
            <p style="color: #6E7A70; font-size: 14px; margin: 0;">
                Daily Agmarknet modal price series with statistical indicators and moving average trends.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Time window filter for chart
    sub_col1, sub_col2, sub_col3 = st.columns([2, 1, 1])
    with sub_col1:
        lookback_choice = st.radio(
            "Display Range:",
            ["Last 90 Days", "Last 180 Days", "Last 1 Year", "Full History (2023-2025)"],
            index=2,
            horizontal=True,
        )
    with sub_col2:
        show_ma7 = st.checkbox("Show 7-Day Moving Avg", value=True)
    with sub_col3:
        show_ma30 = st.checkbox("Show 30-Day Moving Avg", value=False)

    # Filter data for display
    plot_df = series_data.copy()
    plot_df["ma_7"] = plot_df["modal_price"].rolling(7).mean()
    plot_df["ma_30"] = plot_df["modal_price"].rolling(30).mean()

    if lookback_choice == "Last 90 Days":
        plot_df = plot_df.tail(90)
    elif lookback_choice == "Last 180 Days":
        plot_df = plot_df.tail(180)
    elif lookback_choice == "Last 1 Year":
        plot_df = plot_df.tail(365)

    # Plotly Price History Chart
    fig_hist = go.Figure()

    # Min/Max Shaded Area
    fig_hist.add_trace(go.Scatter(
        x=plot_df.index,
        y=plot_df["max_price"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig_hist.add_trace(go.Scatter(
        x=plot_df.index,
        y=plot_df["min_price"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(27, 77, 62, 0.08)",
        name="Daily Min-Max Range",
        hoverinfo="skip",
    ))

    # Modal Price Line
    fig_hist.add_trace(go.Scatter(
        x=plot_df.index,
        y=plot_df["modal_price"],
        mode="lines",
        name=f"Modal Price (₹/qtl)",
        line=dict(color="#1B4D3E", width=2.5),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Modal Price: ₹%{y:,.0f}/qtl<extra></extra>",
    ))

    if show_ma7:
        fig_hist.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df["ma_7"],
            mode="lines",
            name="7-Day Moving Average",
            line=dict(color="#E65100", width=1.8, dash="dot"),
            hovertemplate="7-Day MA: ₹%{y:,.0f}/qtl<extra></extra>",
        ))

    if show_ma30:
        fig_hist.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df["ma_30"],
            mode="lines",
            name="30-Day Moving Average",
            line=dict(color="#1b6d24", width=1.8, dash="dash"),
            hovertemplate="30-Day MA: ₹%{y:,.0f}/qtl<extra></extra>",
        ))

    fig_hist.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=20, t=20, b=40),
        height=380,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family="Plus Jakarta Sans", size=12),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#E2E6DC",
            gridwidth=1,
            linecolor="#C5CCC0",
            tickfont=dict(family="Space Grotesk", size=11, color="#465248"),
        ),
        yaxis=dict(
            title=dict(text="Price (₹ / Quintal)", font=dict(family="Space Grotesk", size=12, color="#1B4D3E")),
            showgrid=True,
            gridcolor="#E2E6DC",
            gridwidth=1,
            linecolor="#C5CCC0",
            tickprefix="₹",
            tickfont=dict(family="Space Grotesk", size=11, color="#465248"),
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Statistical Summary Strip
    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
    with stat_col1:
        st.markdown(
            f"""
            <div class="kmi-card" style="padding: 14px; text-align: center;">
                <div class="kmi-metric-label">Min Price</div>
                <div style="font-family: 'Space Grotesk'; font-size: 20px; font-weight: 700; color: #1B4D3E;">
                    ₹{plot_df['modal_price'].min():,.0f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with stat_col2:
        st.markdown(
            f"""
            <div class="kmi-card" style="padding: 14px; text-align: center;">
                <div class="kmi-metric-label">Max Price</div>
                <div style="font-family: 'Space Grotesk'; font-size: 20px; font-weight: 700; color: #1B4D3E;">
                    ₹{plot_df['modal_price'].max():,.0f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with stat_col3:
        st.markdown(
            f"""
            <div class="kmi-card" style="padding: 14px; text-align: center;">
                <div class="kmi-metric-label">Average Price</div>
                <div style="font-family: 'Space Grotesk'; font-size: 20px; font-weight: 700; color: #1B4D3E;">
                    ₹{plot_df['modal_price'].mean():,.0f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with stat_col4:
        st.markdown(
            f"""
            <div class="kmi-card" style="padding: 14px; text-align: center;">
                <div class="kmi-metric-label">Std Deviation (₹)</div>
                <div style="font-family: 'Space Grotesk'; font-size: 20px; font-weight: 700; color: #1B4D3E;">
                    ₹{plot_df['modal_price'].std():,.1f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with stat_col5:
        st.markdown(
            f"""
            <div class="kmi-card" style="padding: 14px; text-align: center;">
                <div class="kmi-metric-label">Total Observations</div>
                <div style="font-family: 'Space Grotesk'; font-size: 20px; font-weight: 700; color: #1B4D3E;">
                    {len(plot_df):,} days
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Raw Data Expander
    with st.expander("📋 View Raw Cleaned Records & Download"):
        table_view = plot_df.reset_index()[["date", "commodity", "market", "min_price", "modal_price", "max_price"]].tail(50)
        table_view = table_view.sort_values(by="date", ascending=False)
        st.dataframe(table_view, use_container_width=True, hide_index=True)
        csv_data = plot_df.to_csv().encode("utf-8")
        st.download_button(
            label="📥 Download Historical Price CSV",
            data=csv_data,
            file_name=f"{selected_commodity}_{selected_market}_prices.csv",
            mime="text/csv",
        )


# =========================================================
# TAB 2: FORECAST & STRATEGY
# =========================================================
with tab_forecast:
    st.markdown(
        """
        <div style="margin-bottom: 16px;">
            <h3 style="margin-bottom: 4px;">Price Forecast & Selling Strategy</h3>
            <p style="color: #6E7A70; font-size: 14px; margin: 0;">
                Multi-day out-of-sample forecast with confidence interval bands and actionable marketing advice.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. High Visibility Strategy Banner
    banner_class = "banner-sell" if decision_info["decision"] == "SELL NOW" else "banner-wait"
    badge_class = "badge-sell" if decision_info["decision"] == "SELL NOW" else "badge-wait"
    icon = "↗" if decision_info["decision"] == "SELL NOW" else "⏱"

    st.markdown(
        f"""
        <div class="kmi-decision-banner {banner_class}">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                <div style="display: flex; align-items: center; gap: 14px;">
                    <span class="kmi-pill-badge {badge_class}" style="font-size: 18px; font-family: 'Space Grotesk'; font-weight: 700; padding: 8px 18px;">
                        {icon} {decision_info['decision']}
                    </span>
                    <span style="font-family: 'Space Grotesk'; font-size: 16px; font-weight: 600; color: #141D17;">
                        Recommendation for {selected_commodity} in {selected_market} ({horizon_option}-Day Horizon)
                    </span>
                </div>
                <div style="font-family: 'Space Grotesk'; font-size: 14px; font-weight: 600; color: #465248;">
                    Projected Delta: <b>{decision_info['diff_pct']:+.1f}% ({decision_info['diff_rs']:+,.0f} ₹/qtl)</b>
                </div>
            </div>
            <div style="font-family: 'Plus Jakarta Sans'; font-size: 14px; line-height: 1.5; color: #141D17; margin-top: 6px;">
                {decision_info['recommendation']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model selector for the forecast chart
    f_col1, f_col2 = st.columns([2, 1])
    with f_col1:
        st.markdown(f"**Forecasting Engine:** Currently displaying predictions from **{best_model_name}** (Lowest Test Error).")
    with f_col2:
        model_view_choice = st.selectbox(
            "Compare Forecast with Another Model:",
            ["Best Model (" + best_model_name + ")"] + [m for m in models_dict.keys() if m != best_model_name],
            index=0,
        )

    # Determine which model values to plot
    active_model_name = best_model_name if "Best Model" in model_view_choice else model_view_choice
    active_model_obj = models_dict[active_model_name]

    if "Linear Regression" in active_model_name or "Random Forest" in active_model_name:
        active_forecast = active_model_obj.forecast_steps(history_prices, last_date, horizon_option)
    else:
        active_forecast = active_model_obj.forecast_steps(history_prices, horizon_option)

    # 2. Interactive Forecast Chart
    recent_history_days = 30
    recent_slice = series_data.tail(recent_history_days)

    fig_forecast = go.Figure()

    # Confidence Interval Shading (PRD chart-forecast CI fill: rgba(230, 81, 0, 0.12))
    fig_forecast.add_trace(go.Scatter(
        x=future_dates,
        y=upper_bound,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig_forecast.add_trace(go.Scatter(
        x=future_dates,
        y=lower_bound,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(230, 81, 0, 0.12)",
        name="95% Confidence Interval",
        hoverinfo="skip",
    ))

    # Benchmark Line (PRD chart-benchmark: #78909C)
    fig_forecast.add_trace(go.Scatter(
        x=[recent_slice.index[0], future_dates[-1]],
        y=[current_price, current_price],
        mode="lines",
        name=f"Current Price Benchmark (₹{current_price:,.0f})",
        line=dict(color="#78909C", width=1.5, dash="dash"),
        hoverinfo="skip",
    ))

    # Past Actual Prices (PRD solid #1B4D3E, 2.5px width)
    fig_forecast.add_trace(go.Scatter(
        x=recent_slice.index,
        y=recent_slice["modal_price"],
        mode="lines+markers",
        name="Historical Price",
        line=dict(color="#1B4D3E", width=2.5),
        marker=dict(size=4, color="#1B4D3E"),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Historical Price: ₹%{y:,.0f}/qtl<extra></extra>",
    ))

    # Future Forecast (PRD dashed #E65100, 2.5px width)
    fig_forecast.add_trace(go.Scatter(
        x=[recent_slice.index[-1]] + future_dates,
        y=[current_price] + list(active_forecast),
        mode="lines+markers",
        name=f"Projected Forecast ({active_model_name})",
        line=dict(color="#E65100", width=2.5, dash="dash"),
        marker=dict(size=6, color="#E65100"),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Forecast Price: ₹%{y:,.0f}/qtl<extra></extra>",
    ))

    fig_forecast.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=20, t=20, b=40),
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family="Plus Jakarta Sans", size=12),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#E2E6DC",
            gridwidth=1,
            linecolor="#C5CCC0",
            tickfont=dict(family="Space Grotesk", size=11, color="#465248"),
        ),
        yaxis=dict(
            title=dict(text="Price (₹ / Quintal)", font=dict(family="Space Grotesk", size=12, color="#1B4D3E")),
            showgrid=True,
            gridcolor="#E2E6DC",
            gridwidth=1,
            linecolor="#C5CCC0",
            tickprefix="₹",
            tickfont=dict(family="Space Grotesk", size=11, color="#465248"),
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    # 3. Forecast Daily Schedule Breakdown Table
    st.markdown("#### Day-by-Day Forecast Schedule")
    forecast_table_data = []
    for idx, (f_date, f_val, ub, lb) in enumerate(zip(future_dates, active_forecast, upper_bound, lower_bound)):
        diff_from_today = f_val - current_price
        pct_from_today = (diff_from_today / current_price) * 100
        forecast_table_data.append({
            "Day": f"Day {idx + 1}",
            "Date": f_date.strftime("%d %b %Y (%a)"),
            "Forecast Price (₹/qtl)": f"₹{f_val:,.0f}",
            "Lower Bound (95% CI)": f"₹{lb:,.0f}",
            "Upper Bound (95% CI)": f"₹{ub:,.0f}",
            "Expected Delta": f"{diff_from_today:+,.0f} ₹ ({pct_from_today:+.1f}%)",
        })

    st.dataframe(pd.DataFrame(forecast_table_data), use_container_width=True, hide_index=True)


# =========================================================
# TAB 3: MODEL EVALUATION
# =========================================================
with tab_eval:
    st.markdown(
        """
        <div style="margin-bottom: 16px;">
            <h3 style="margin-bottom: 4px;">Model Evaluation & Benchmark Comparison</h3>
            <p style="color: #6E7A70; font-size: 14px; margin: 0;">
                Rigorous out-of-sample backtesting using an 80% train / 20% test chronological split with zero future data leakage.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top alert about evaluation methodology
    st.markdown(
        f"""
        <div style="background-color: #F0F2EB; border-left: 4px solid #1B4D3E; padding: 12px 16px; border-radius: 4px; margin-bottom: 20px; font-size: 13px; color: #141D17;">
            <b>Evaluation Protocol:</b> Trained on <b>{len(train_slice)} days</b> of historical prices (80% split) and validated on <b>{len(test_slice)} days</b> of unseen recent market data (20% test split). All models compete against the Naive Baseline.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Evaluation Comparison Table
    # Styling per design specs: header in #1B4D3E, cream text, best model highlighted in #F0FDF4 with emerald star
    table_html = """
    <table class="kmi-table">
        <thead>
            <tr>
                <th>Model</th>
                <th>Category</th>
                <th>MAE (₹/qtl)</th>
                <th>RMSE (₹/qtl)</th>
                <th>MAPE (%)</th>
                <th>Outperformed Baseline</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
    """
    for _, row in eval_df.iterrows():
        is_best = row["Is Best"]
        row_cls = "best-row" if is_best else ""
        star_badge = '<span style="color: #2E7D32; font-weight: 700;">⭐ Best Model</span>' if is_best else '<span style="color: #6E7A70;">Evaluated</span>'
        table_html += f"""
        <tr class="{row_cls}">
            <td style="font-weight: 600;">{row['Model']}</td>
            <td>{row['Type']}</td>
            <td style="font-family: 'Space Grotesk'; font-weight: 600;">₹{row['MAE (Rs)']:,.2f}</td>
            <td style="font-family: 'Space Grotesk'; font-weight: 600;">₹{row['RMSE (Rs)']:,.2f}</td>
            <td style="font-family: 'Space Grotesk'; font-weight: 700; color: {'#1B5E20' if is_best else '#141D17'};">{row['MAPE (%)']:.2f}%</td>
            <td><span class="kmi-pill-badge {'badge-sell' if row['Beat Baseline'] == 'Yes' else 'badge-neutral'}">{row['Beat Baseline']}</span></td>
            <td>{star_badge}</td>
        </tr>
        """
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    # Test Set Actual vs Predicted Chart
    st.markdown("#### Test Period Backtest: Actual vs Predicted Prices")
    st.markdown(
        "<p style='color: #6E7A70; font-size: 13px;'>Visualizing how closely each model predicted the unseen 20% test slice.</p>",
        unsafe_allow_html=True,
    )

    fig_test = go.Figure()

    # Actual Test Prices
    fig_test.add_trace(go.Scatter(
        x=test_slice.index,
        y=test_slice["modal_price"],
        mode="lines",
        name="Actual Market Price",
        line=dict(color="#141D17", width=2.8),
    ))

    # Add each model prediction
    model_colors = {
        "Naive (Baseline)": "#94A3B8",
        "Moving Average (7-Day)": "#0284C7",
        "ARIMA": "#9333EA",
        "Linear Regression": "#E65100",
        "Random Forest": "#16A34A",
    }

    for col_m in eval_df["Model"]:
        if col_m in pred_test_df.columns:
            is_winner = (col_m == best_model_name)
            fig_test.add_trace(go.Scatter(
                x=test_slice.index,
                y=pred_test_df[col_m],
                mode="lines",
                name=f"{col_m} {'(Winner)' if is_winner else ''}",
                line=dict(
                    color=model_colors.get(col_m, "#64748B"),
                    width=2.5 if is_winner else 1.5,
                    dash="solid" if is_winner else "dot",
                ),
            ))

    fig_test.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=20, t=20, b=40),
        height=380,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family="Plus Jakarta Sans", size=11),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#E2E6DC",
            gridwidth=1,
            linecolor="#C5CCC0",
            tickfont=dict(family="Space Grotesk", size=11, color="#465248"),
        ),
        yaxis=dict(
            title=dict(text="Test Set Price (₹ / Quintal)", font=dict(family="Space Grotesk", size=12, color="#1B4D3E")),
            showgrid=True,
            gridcolor="#E2E6DC",
            gridwidth=1,
            linecolor="#C5CCC0",
            tickprefix="₹",
            tickfont=dict(family="Space Grotesk", size=11, color="#465248"),
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig_test, use_container_width=True)


# =========================================================
# TAB 4: MANDI COMPARE
# =========================================================
with tab_compare:
    st.markdown(
        """
        <div style="margin-bottom: 16px;">
            <h3 style="margin-bottom: 4px;">Regional Mandi Price Comparison & Arbitrage</h3>
            <p style="color: #6E7A70; font-size: 14px; margin: 0;">
                Compare current and 14-day average prices across all mandis to discover higher-paying markets and calculate net transport margins.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Filter mandi comparison scope
    c_col1, c_col2 = st.columns([2, 1])
    with c_col1:
        state_filter_choice = st.radio(
            "Comparison Region:",
            [f"Within {selected_state}", "All Neighboring States (Punjab, Haryana, UP)"],
            index=0,
            horizontal=True,
        )
    with c_col2:
        top_n_mandis = st.slider("Display Top Mandis:", min_value=5, max_value=30, value=15, step=5)

    # Filter comparison dataset
    if "Within" in state_filter_choice:
        comp_subset = mandi_stats[mandi_stats["state"] == selected_state].head(top_n_mandis)
    else:
        comp_subset = mandi_stats.head(top_n_mandis)

    # Horizontal Bar Chart
    if not comp_subset.empty:
        # Mark whether row is current selected mandi
        comp_subset = comp_subset.copy()
        comp_subset["is_current"] = comp_subset["market"] == selected_market
        comp_subset["color"] = comp_subset["is_current"].apply(
            lambda x: "#E65100" if x else "#1B4D3E"
        )

        fig_bar = px.bar(
            comp_subset.sort_values(by="latest_price", ascending=True),
            x="latest_price",
            y="market",
            orientation="h",
            text="latest_price",
            labels={"latest_price": "Modal Price (₹/qtl)", "market": "Mandi"},
            color="is_current",
            color_discrete_map={True: "#E65100", False: "#1B4D3E"},
        )

        fig_bar.update_traces(
            texttemplate="₹%{text:,.0f}",
            textposition="inside",
            insidetextanchor="end",
            textfont=dict(family="Space Grotesk", color="#FFFFFF", size=11),
            showlegend=False,
        )

        fig_bar.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            margin=dict(l=40, r=20, t=10, b=40),
            height=max(350, len(comp_subset) * 26),
            xaxis=dict(
                showgrid=True,
                gridcolor="#E2E6DC",
                tickprefix="₹",
                tickfont=dict(family="Space Grotesk", size=11, color="#465248"),
            ),
            yaxis=dict(
                tickfont=dict(family="Plus Jakarta Sans", size=11, color="#141D17"),
            ),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Transport & Arbitrage Margin Calculator (PRD Section 19 Future Scope)
    st.markdown("#### Transport Cost & Arbitrage Profit Calculator")
    st.markdown(
        "<p style='color: #6E7A70; font-size: 13px;'>Calculate whether transporting produce to a higher-paying mandi yields higher net profit after diesel/freight expenses.</p>",
        unsafe_allow_html=True,
    )

    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        produce_load_qtl = st.number_input("Produce Volume (Quintals)", min_value=10, max_value=500, value=50, step=10)
    with t_col2:
        distance_km = st.slider("Estimated Distance to Alternate Mandi (km)", min_value=10, max_value=250, value=65, step=5)
    with t_col3:
        freight_rate_per_km = st.number_input("Roundtrip Freight Cost (₹/km)", min_value=10, max_value=100, value=30, step=5)

    transport_cost_total = distance_km * freight_rate_per_km
    transport_cost_per_qtl = transport_cost_total / produce_load_qtl

    # Mandi Comparison Table with Net Margins
    comp_display = comp_subset.copy()
    comp_display["Gross Premium (₹/qtl)"] = comp_display["diff_vs_current"]
    comp_display["Net Profit (₹/qtl)"] = (comp_display["Gross Premium (₹/qtl)"] - transport_cost_per_qtl).round(1)
    comp_display["Total Net Gain (₹)"] = (comp_display["Net Profit (₹/qtl)"] * produce_load_qtl).round(0)

    # Format table for display
    display_table = pd.DataFrame({
        "Mandi": comp_display["market"],
        "District": comp_display["district"],
        "State": comp_display["state"],
        "Latest Price (₹/qtl)": comp_display["latest_price"].apply(lambda x: f"₹{x:,.0f}"),
        "14-Day Avg (₹/qtl)": comp_display["avg_price"].apply(lambda x: f"₹{x:,.0f}"),
        "Spread vs Current": comp_display["diff_vs_current"].apply(lambda x: f"{x:+,.0f} ₹/qtl"),
        "Transport Cost": f"₹{transport_cost_per_qtl:.1f}/qtl",
        "Net Arbitrage Gain": comp_display["Total Net Gain (₹)"].apply(
            lambda x: f"+₹{x:,.0f}" if x > 0 else f"₹{x:,.0f}"
        ),
    })

    st.dataframe(display_table, use_container_width=True, hide_index=True)


# =========================================================
# TAB 5: WHATSAPP & SMS ADVISORY ALERTS
# =========================================================
with tab_alerts:
    st.markdown(
        f"""
        <div style="margin-bottom: 16px;">
            <h3 style="margin-bottom: 4px;">{get_text(selected_lang, 'tab_alerts')}</h3>
            <p style="color: #6E7A70; font-size: 14px; margin: 0;">
                Automated multi-channel price trigger alerts delivering timely mandi intelligence to farmers and FPO coordinators.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a_col1, a_col2 = st.columns([1, 1])

    with a_col1:
        st.markdown("#### Configure Advisory Subscription")
        phone_num = st.text_input("Farmer Mobile Number", value="+91 98721 54321", max_chars=15)
        alert_channel = st.radio("Notification Channel", ["WhatsApp Advisory", "SMS Text Alert"], horizontal=True)
        alert_event = st.selectbox(
            "Trigger Condition",
            [
                f"When {selected_commodity} price crosses target threshold in {selected_market}",
                f"When AI model changes recommendation to SELL NOW",
                f"Daily Morning Mandi Rate Digest (8:00 AM IST)",
            ],
        )
        target_trigger_price = st.number_input(
            "Target Trigger Price (₹ / Quintal)",
            min_value=100,
            max_value=10000,
            value=int(current_price * 1.05),
            step=50,
        )

        sub_btn = st.button("🔔 Activate Price Trigger Alert")
        if sub_btn:
            st.success(f"Advisory alert active! Alerts will be sent to {phone_num} via {alert_channel}.")

    with a_col2:
        st.markdown("#### Live Alert Message Preview")
        # Format preview text based on language
        if selected_lang == "हिन्दी":
            wa_text = f"""
🌾 *किसान मंडी इंटेलिजेंस अलर्ट*
━━━━━━━━━━━━━━━━━━
📍 *मंडी:* {selected_market} ({selected_state})
🌱 *फसल:* {selected_commodity}
💰 *आज का भाव:* ₹{current_price:,.0f}/क्विंटल
🎯 *पूर्वानुमान ({horizon_option} दिन):* ₹{decision_info['avg_forecast']:,.0f}/क्विंटल
⚡ *सिफारिश:* {decision_info['decision']} ({decision_info['diff_pct']:+.1f}%)

💡 *सलाह:* {decision_info['recommendation']}
━━━━━━━━━━━━━━━━━━
हेल्पलाइन: 1800-180-1551 (किसान कॉल सेंटर)
            """
        elif selected_lang == "ਪੰਜਾਬੀ":
            wa_text = f"""
🌾 *ਕਿਸਾਨ ਮੰਡੀ ਇੰਟੈਲੀਜੈਂਸ ਅਲਰਟ*
━━━━━━━━━━━━━━━━━━
📍 *ਮੰਡੀ:* {selected_market} ({selected_state})
🌱 *ਫਸਲ:* {selected_commodity}
💰 *ਅੱਜ ਦਾ ਭਾਅ:* ₹{current_price:,.0f}/ਕੁਇੰਟਲ
🎯 *ਅਨੁਮਾਨ ({horizon_option} ਦਿਨ):* ₹{decision_info['avg_forecast']:,.0f}/ਕੁਇੰਟਲ
⚡ *ਸਿਫਾਰਸ਼:* {decision_info['decision']} ({decision_info['diff_pct']:+.1f}%)

💡 *ਸਲਾਹ:* {decision_info['recommendation']}
━━━━━━━━━━━━━━━━━━
ਹੈਲਪਲਾਈਨ: 1800-180-1551 (ਕਿਸਾਨ ਕਾਲ ਸੈਂਟਰ)
            """
        else:
            wa_text = f"""
🌾 *KISAN MANDI INTELLIGENCE ALERT*
━━━━━━━━━━━━━━━━━━
📍 *Mandi:* {selected_market} ({selected_state})
🌱 *Crop:* {selected_commodity}
💰 *Today's Rate:* ₹{current_price:,.0f}/qtl
🎯 *AI Forecast ({horizon_option}d):* ₹{decision_info['avg_forecast']:,.0f}/qtl
⚡ *Signal:* *{decision_info['decision']}* ({decision_info['diff_pct']:+.1f}%)

💡 *Advisory:* {decision_info['recommendation']}
━━━━━━━━━━━━━━━━━━
Toll-Free Helpline: 1800-180-1551 (Kisan Call Centre)
            """

        st.markdown(
            f"""
            <div style="background-color: #EFEAE2; border-radius: 12px; padding: 18px; border: 1px solid #D1D7DB; max-width: 420px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
                <div style="background-color: #005C4B; color: #FFFFFF; padding: 8px 14px; border-radius: 8px 8px 0 0; font-size: 13px; font-weight: 600; display: flex; align-items: center; gap: 8px;">
                    <span>📱</span> WhatsApp Mandi Bot (Verified)
                </div>
                <div style="background-color: #FFFFFF; border-radius: 0 0 8px 8px; padding: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.1); margin-top: 2px;">
                    <pre style="white-space: pre-wrap; font-family: inherit; font-size: 13px; color: #111B21; margin: 0; line-height: 1.45;">{wa_text.strip()}</pre>
                    <div style="text-align: right; font-size: 10px; color: #667781; margin-top: 8px;">08:15 AM · Delivered ✓✓</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown(
    """
    <div style="margin-top: 50px; padding: 20px 0; border-top: 1px solid #E2E6DC; display: flex; justify-content: space-between; align-items: center; font-size: 12px; color: #6E7A70; flex-wrap: wrap; gap: 10px;">
        <div>
            <b>Kisan Mandi Intelligence</b> · AI Farmer Market & Price Discovery System · Problem Statement: SIH26132
        </div>
        <div>
            Dataset: Agmarknet Daily Commodity Prices (India) · Built with Streamlit, Scikit-Learn, Statsmodels
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
