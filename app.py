# =============================================================================
# CAPM Attribution Dashboard
# Author: Edgar Alcántara & Daniel R. Barrera
# =============================================================================

# --- Libraries ----------------------------------------------------------------

import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import re

# Data
from src.portfolios_dashboard.data import (
    load_file,
    parse_dataframe,
    import_prices_data,
    log_returns,
)

# Plots
from src.portfolios_dashboard.plots import TimeSeriesPlot

# Regressions
from src.portfolios_dashboard.regression import rolling_capm_coefficients

# Attribution
from src.portfolios_dashboard.attribution import (
    factor_contribution,
    capm_risk_attribution,
)

# Risk metrics
from src.portfolios_dashboard.risk_measures import (
    sharpe_ratio,
    value_at_risk,
    expected_shortfall,
    max_drawdown,
    conditional_expected_drawdown,
    tracking_error,
)

# =============================================================================
# CONSTANTS
# =============================================================================

SAMPLE_PORTFOLIOS = {
    "— Select a sample portfolio —": None,
    "Betting-Against-Beta":          r"config/betting_against_beta_portfolio.csv",
    "Equal-Weighted":                r"config/equal_weighted_portfolio.csv",
    "Mean-Variance":                 r"config/mean_variance_portfolio.csv",
    "Momentum":                      r"config/momentum_portfolio.csv",
}

BENCHMARKS = {
    "S&P 500 ETF (SPY)": "SPY",
    "S&P 500 Index (^GSPC)": "^GSPC",
    "MSCI World (URTH)": "URTH",
    "MSCI Emerging Markets (EEM)": "EEM",
    "Total Stock Market (VTI)": "VTI",
    "MSCI ACWI (ACWI)": "ACWI",
    "Russell 2000 (^RUT)": "^RUT",
    "Dow Jones (^DJI)": "^DJI",
    "NASDAQ 100 (^NDX)": "^NDX",
}

RISK_FREE_RATES = {
    "US Treasury 3 Months (^IRX)": "^IRX",
    "US Treasury 5 Years (^FVX)": "^FVX",
    "US Treasury 10 Years (^TNX)": "^TNX",
    "US Treasury 30 Years (^TYX)": "^TYX",
}

ROLLING_WINDOWS = {
    "1 Year (252)": 252,
    "2 Years (504)": 504,
    "3 Years (756)": 756,
    "4 Years (1008)": 1008,
    "5 Years (1260)": 1260,
}

TRADING_DAYS = 252


# =============================================================================
# CACHED DATA LOADERS
# =============================================================================

@st.cache_data(show_spinner="Downloading benchmark data...")
def get_benchmark_returns(ticker: str, start_date, end_date) -> pd.Series:
    """Download benchmark price data and convert to log returns."""
    prices = import_prices_data(
        tickers=ticker,
        start_date=start_date - pd.Timedelta(days=1),
        end_date=end_date,
    )
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    returns = log_returns(prices).dropna()
    if isinstance(returns, pd.DataFrame):
        returns = returns.squeeze()
    returns.index = returns.index.tz_convert(None) if returns.index.tz is not None else returns.index
    return returns


@st.cache_data(show_spinner="Downloading risk-free rate data...")
def get_risk_free_rate(ticker: str, start_date, end_date) -> pd.Series:
    """Download risk-free rate data and convert to daily rate."""
    data = import_prices_data(
        tickers=ticker,
        start_date=start_date,
        end_date=end_date,
    )
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    daily_rate = (data / 100 / 360).ffill()
    if isinstance(daily_rate, pd.DataFrame):
        daily_rate = daily_rate.squeeze()
    daily_rate.index = daily_rate.index.tz_convert(None) if daily_rate.index.tz is not None else daily_rate.index
    return daily_rate


@st.cache_data(show_spinner="Running rolling CAPM regression...")
def get_rolling_capm(portfolio_returns, benchmark_returns, risk_free_rate, window):
    """Compute rolling CAPM coefficients and flatten MultiIndex columns."""
    capm = rolling_capm_coefficients(
        portfolio_returns,
        benchmark_returns,
        risk_free_rate,
        window=window,
    )
    capm_df = pd.concat(capm, axis=1)
    if isinstance(capm_df.columns, pd.MultiIndex):
        capm_df.columns = capm_df.columns.droplevel(1)
    return capm_df


# =============================================================================
# HELPER: DETECT RETURN COLUMN
# =============================================================================

def get_return_column(df: pd.DataFrame) -> str | None:
    for candidate in ["return", "returns", "Return", "Returns"]:
        if candidate in df.columns:
            return candidate
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        return numeric_cols[0]
    return None


# =============================================================================
# SECTION RENDERERS
# =============================================================================

def render_data_uploader():
    """Render file uploader and benchmark/RFR selectors. Returns user inputs."""
    st.markdown(
        "<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;color:#111440;"
        "opacity:0.45;text-transform:uppercase;margin-bottom:6px'>Data source</p>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Upload a file with **two columns**: one for **dates** (daily basis) and one for **returns**. "
        "Supported formats: CSV, Excel (.xlsx / .xls), Parquet, JSON, and TXT."
    )

    col_file, col_bench, col_rfr = st.columns([5, 2.5, 2.5])

    with col_file:
        source = st.radio(
            "Data source",
            options=["Upload my own portfolio", "Use a sample portfolio"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if source == "Upload my own portfolio":
            uploaded_file = st.file_uploader(
                "Upload your time series file",
                type=["csv", "xlsx", "xls", "parquet", "json", "txt"],
                label_visibility="collapsed",
            )

        else:
            sample_label = st.selectbox(
                "Select a sample portfolio",
                options=list(SAMPLE_PORTFOLIOS.keys()),
                label_visibility="collapsed",
            )
            sample_path = SAMPLE_PORTFOLIOS[sample_label]

            if sample_path:
                uploaded_file = open(sample_path, "rb")
            else:
                uploaded_file = None

    with col_bench:
        benchmark_label = st.selectbox("Benchmark", options=list(BENCHMARKS.keys()))

    with col_rfr:
        rfr_label = st.selectbox("Risk-Free Rate", options=list(RISK_FREE_RATES.keys()))

    return uploaded_file, benchmark_label, rfr_label


def render_overview(df: pd.DataFrame, benchmark_returns: pd.Series, rfr_series: pd.Series,
                    benchmark_label: str, rfr_label: str, return_col: str):
    """Render the Overview section: cumulative returns, market data table, and risk metrics."""

    col_left, _, col_right = st.columns([4.5, 1, 4.5])

    # ── Left column: portfolio info ──────────────────────────────────────────
    with col_left:
        st.success(f"✅ File loaded successfully — **{len(df):,}** records.")

        c1, c2 = st.columns(2)
        c1.metric("Start Date", str(df.index.min().date()))
        c2.metric("End Date", str(df.index.max().date()))

        st.markdown(
            "<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;color:#111440;"
            "opacity:0.45;text-transform:uppercase;margin:1rem 0 4px'>Performance</p>",
            unsafe_allow_html=True,
        )
        fig_portfolio = TimeSeriesPlot(
            df[[return_col]].mul(100).cumsum(),
            "Cumulative Returns (%)",
        )
        st.plotly_chart(fig_portfolio, use_container_width=True)

    # ── Right column: market data + risk metrics ─────────────────────────────
    with col_right:
        market_data = pd.concat([benchmark_returns, rfr_series], axis=1)
        market_data.columns = [benchmark_label, rfr_label]
        aligned = market_data.reindex(df.index)
        missing_pct = aligned.isna().mean().mul(100).round(1)

        if missing_pct.max() > 20:
            st.warning(
                f"⚠️ Up to {missing_pct.max():.1f}% of market data could not be aligned "
                "to your portfolio dates. Results may be affected."
            )

        # ── Risk Metrics ─────────────────────────────────────────────────────
        st.markdown(
            "<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;color:#111440;"
            "opacity:0.45;text-transform:uppercase;margin-bottom:6px'>Risk & Performance Metrics</p>",
            unsafe_allow_html=True,
        )

        portfolio_series = df[return_col]
        bm_series = benchmark_returns.reindex(df.index).dropna()

        ann_return = portfolio_series.mul(100).mean() * TRADING_DAYS
        ann_std    = portfolio_series.mul(100).std() * np.sqrt(TRADING_DAYS)
        daily_var  = portfolio_series.mul(100).var()
        te         = tracking_error(portfolio_series.mul(100), bm_series.mul(100))
        sharpe     = sharpe_ratio(df[[return_col]]) * np.sqrt(TRADING_DAYS)
        var_95     = value_at_risk(df[[return_col]])
        es         = expected_shortfall(df[[return_col]])
        md         = max_drawdown(df[[return_col]])
        ced        = conditional_expected_drawdown(df[[return_col]])

        c1, c2, c3 = st.columns(3)

        c1.metric("Ann. Return",      f"{ann_return:.2f}%")
        c1.metric("Ann. Volatility",  f"{ann_std:.2f}%")
        c1.metric("Daily Variance",   f"{daily_var:.6f}")

        c2.metric("Sharpe Ratio",     f"{sharpe:.4f}x")
        c2.metric("VaR (95%)",        f"{var_95:.2%}")
        c2.metric("Exp. Shortfall",   f"{es:.2%}")

        c3.metric("Tracking Error",        f"{te:.4f}%")
        c3.metric("Max. Drawdown",         f"{md:.2%}")
        c3.metric("Cond. Exp. Drawdown",   f"{ced:.2%}")

        st.markdown(
            "<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;color:#111440;"
            "opacity:0.45;text-transform:uppercase;margin:1rem 0 4px'>Market Data Preview</p>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            market_data.style.format("{:.6f}"),
            height=350,
            use_container_width=True,
        )


def render_capm_interpretation(capm_df: pd.DataFrame, capm_risk: pd.DataFrame):
    """Auto-generate a plain-english interpretation of CAPM results."""

    alpha_mean = capm_df["alpha"].mean()
    beta_mean  = capm_df["beta"].mean()
    beta_last  = capm_df["beta"].iloc[-1]
    sys_pct    = capm_risk.loc["Systematic", "percentage"]
    idio_pct   = capm_risk.loc["Idiosyncratic", "percentage"]

    # Alpha interpretation
    if alpha_mean > 0:
        alpha_msg = f"The portfolio has generated **positive alpha** on average ({alpha_mean*100:.3f}% daily), suggesting the strategy adds value beyond market exposure."
    else:
        alpha_msg = f"The portfolio shows **negative alpha** on average ({alpha_mean*100:.3f}% daily), indicating returns have lagged what the market exposure alone would predict."

    # Beta interpretation
    if beta_mean < 0.8:
        beta_msg = f"With an average beta of **{beta_mean:.2f}**, the portfolio is **defensive** — it moves less than the market and may offer downside protection."
    elif beta_mean <= 1.2:
        beta_msg = f"With an average beta of **{beta_mean:.2f}**, the portfolio tracks the market **closely** with similar risk exposure."
    else:
        beta_msg = f"With an average beta of **{beta_mean:.2f}**, the portfolio is **aggressive** — it amplifies market movements and carries higher systematic risk."

    # Recent trend
    if abs(beta_last - beta_mean) > 0.2:
        trend_msg = f"Recently, beta has shifted to **{beta_last:.2f}**, which is notably {'higher' if beta_last > beta_mean else 'lower'} than its historical average."
    else:
        trend_msg = f"Recent beta (**{beta_last:.2f}**) is consistent with the historical average — no significant regime change detected."

    # Risk decomposition
    risk_msg = f"**{sys_pct*100:.1f}%** of total variance is explained by market exposure (systematic risk), while **{idio_pct*100:.1f}%** is idiosyncratic to the strategy."

    st.markdown(
        "<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;color:#111440;"
        "opacity:0.45;text-transform:uppercase;margin-bottom:8px'>Interpretation</p>",
        unsafe_allow_html=True,
    )
    for msg in [alpha_msg, beta_msg, trend_msg, risk_msg]:
        html_msg = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', msg)
        st.markdown(
            f"<p style='font-size:13px;color:#444;line-height:1.6;margin-bottom:10px'>{html_msg}</p>",
            unsafe_allow_html=True,
        )


def render_attribution(df: pd.DataFrame, benchmark_returns: pd.Series, rfr_series: pd.Series,
                       benchmark_ticker: str, return_col: str):
    """Render the Attribution section: rolling CAPM coefficients and factor decomposition."""

    st.markdown(
        "<hr style='border:none;border-top:1px solid #e5e7ef;margin:1.5rem 0'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;color:#111440;"
        "opacity:0.45;text-transform:uppercase;margin-bottom:12px'>CAPM Attribution</p>",
        unsafe_allow_html=True,
    )

    col_coef, _, col_window = st.columns([4.5, 1, 4.5])

    with col_coef:
        coef = st.selectbox(
            "Coefficient",
            options=["Alpha", "Beta", "Sigma"],
            index=1,
        )

    with col_window:
        window_label = st.selectbox(
            "Rolling Window",
            options=list(ROLLING_WINDOWS.keys()),
        )

    window = ROLLING_WINDOWS[window_label]
    portfolio_series = df[return_col]

    capm_df = get_rolling_capm(
        portfolio_series,
        benchmark_returns,
        rfr_series,
        window,
    )

    # ── Compute risk attribution (keep clean copy for interpretation) ─────────
    capm_risk = capm_risk_attribution(
        portfolio_series,
        benchmark_returns,
        rfr_series,
    )
    capm_risk_clean = capm_risk.copy()
    capm_risk["ann_volatility"] = np.sqrt(capm_risk["variance"]) * np.sqrt(TRADING_DAYS) / 100

    col_left, _, col_right = st.columns([4.5, 1, 4.5])

    with col_left:
        coef_key = coef.lower()

        if coef == "Alpha":
            series_to_plot = capm_df[coef_key].cumsum()
            chart_title = "Cumulative Alpha (%)"
        else:
            series_to_plot = capm_df[coef_key]
            chart_title = f"Rolling {coef}"

        fig_coef = TimeSeriesPlot(series_to_plot, chart_title)
        st.plotly_chart(fig_coef, use_container_width=True)

        st.markdown(
            "<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;color:#111440;"
            "opacity:0.45;text-transform:uppercase;margin-bottom:6px'>CAPM Risk Attribution</p>",
            unsafe_allow_html=True,
        )

        display = capm_risk.copy()
        display.index = display.index.str.replace("_", " ").str.title()

        st.dataframe(
            display.style
            .format({
                "variance":       "{:.6f}",
                "percentage":     "{:.2%}",
                "ann_volatility": "{:.2%}",
            })
            .bar(subset=["percentage"], color="#111440", vmin=0, vmax=1)
            .background_gradient(subset=["ann_volatility"], cmap="Blues"),
            use_container_width=True,
            column_config={
                "variance":       st.column_config.NumberColumn("Variance"),
                "percentage":     st.column_config.NumberColumn("Weight"),
                "ann_volatility": st.column_config.NumberColumn("Ann. Volatility"),
            },
        )

    with col_right:
        r_i  = portfolio_series.reindex(capm_df.index).dropna()
        r_m  = benchmark_returns.reindex(capm_df.index).dropna()
        beta = capm_df["beta"]

        contribution = factor_contribution(r_i, r_m, beta)
        contribution = pd.concat([r_i, contribution], axis=1)
        contribution.columns = ["Portfolio Returns", "Market Factor", "Residual (Alpha)"]

        fig_contrib = TimeSeriesPlot(
            contribution.mul(100).cumsum(),
            "CAPM Attribution — Cumulative Returns (%)",
        )
        st.plotly_chart(fig_contrib, use_container_width=True)

        render_capm_interpretation(capm_df, capm_risk_clean)


# =============================================================================
# PAGE: PORTFOLIO ATTRIBUTION
# =============================================================================

def page_portfolio_attribution():
    st.markdown("<h2 style='margin-bottom:4px'>Portfolio Attribution</h2>", unsafe_allow_html=True)
    st.caption(
        "Verify your strategy — decompose portfolio returns into market exposure (beta) "
        "and manager skill (alpha)."
    )
    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

    uploaded_file, benchmark_label, rfr_label = render_data_uploader()

    if not uploaded_file:
        st.info("👆 Upload a returns file above to get started.")
        return

    df_raw = load_file(uploaded_file)
    if df_raw is None:
        st.error("❌ Could not read the uploaded file. Please check the format and try again.")
        return

    df = parse_dataframe(df_raw)
    if df is None:
        st.error("❌ Could not parse the file. Ensure it has a date column and a numeric returns column.")
        return

    return_col = get_return_column(df)
    if return_col is None:
        st.error("❌ No numeric returns column found in the uploaded file.")
        return

    if return_col not in ("return", "returns"):
        st.warning(
            f"⚠️ Column `{return_col}` will be used as returns. "
            "Rename it to `return` or `returns` to suppress this warning."
        )

    benchmark_ticker = BENCHMARKS[benchmark_label]
    rfr_ticker       = RISK_FREE_RATES[rfr_label]

    benchmark_returns = get_benchmark_returns(
        benchmark_ticker,
        start_date=df.index.min().date(),
        end_date=df.index.max().date(),
    )

    rfr_series = get_risk_free_rate(
        rfr_ticker,
        start_date=df.index.min().date(),
        end_date=df.index.max().date(),
    )

    render_overview(df, benchmark_returns, rfr_series, benchmark_label, rfr_label, return_col)
    render_attribution(df, benchmark_returns, rfr_series, benchmark_ticker, return_col)


# =============================================================================
# PAGE: MULTIFACTOR ATTRIBUTION (placeholder)
# =============================================================================

def page_multifactor_attribution():
    st.markdown("<h2 style='margin-bottom:4px'>Multifactor Attribution</h2>", unsafe_allow_html=True)
    st.info(
        "🚧 **Coming soon.** This section will support multi-factor attribution models "
        "(e.g., Fama-French 3/5-factor, Carhart 4-factor)."
    )


# =============================================================================
# APP LAYOUT
# =============================================================================

st.set_page_config(
    page_title="DTQ Attribution Dashboard",
    page_icon="config/DTQ_logo.png",
    layout="wide",
)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&display=swap');

        html, body, [class*="css"], p, div, span, label {
            font-family: 'Montserrat', sans-serif !important;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Montserrat', sans-serif !important;
            font-weight: 700 !important;
        }
        .block-container {
            padding-top: 1rem !important;
        }
        [data-testid="metric-container"] {
            background: #f4f5fa;
            border-radius: 8px;
            padding: 12px 16px;
        }
        [data-testid="stMetricLabel"] {
            font-size: 10px !important;
            font-weight: 600 !important;
            letter-spacing: 0.06em !important;
            color: #888 !important;
            text-transform: uppercase;
        }
        [data-testid="stMetricValue"] {
            font-size: 20px !important;
            font-weight: 700 !important;
            color: #111440 !important;
        }
        .stAlert { border-radius: 8px !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:3rem'></div>", unsafe_allow_html=True)

col_logo, col_nav, col_links = st.columns([2, 6, 2])

with col_logo:
    logo = Image.open("config/DTQ_logo.png")
    st.image(logo, width=50)

with col_nav:
    selected = option_menu(
        None,
        ["Portfolio Attribution", "Multifactor Attribution"],
        icons=["bar-chart-line", "layers"],
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "#111440"},
            "nav-link": {"color": "rgba(255,255,255,0.6)", "font-size": "13px"},
            "nav-link-selected": {"background-color": "transparent", "color": "#fff",
                                  "border-bottom": "2px solid #fff"},
        }
    )

with col_links:
    st.markdown(
        """
        <div style='display:flex;flex-direction:column;gap:6px;justify-content:center;padding-top:8px'>
            <div style='display:flex;align-items:center;gap:6px;justify-content:flex-end'>
                <span style='font-size:11px;color:#111440;font-family:Montserrat,sans-serif'>Daniel R. Barrera</span>
                <a href='https://www.linkedin.com/in/danielrbarrera/' target='_blank'>
                    <svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='#111440'>
                        <path d='M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z'/>
                    </svg>
                </a>
            </div>
            <div style='display:flex;align-items:center;gap:6px;justify-content:flex-end'>
                <span style='font-size:11px;color:#111440;font-family:Montserrat,sans-serif'>Edgar M. Alcántara-López</span>
                <a href='https://www.linkedin.com/in/edgarallo0/' target='_blank'>
                    <svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='#111440'>
                        <path d='M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z'/>
                    </svg>
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

# ── Route pages ───────────────────────────────────────────────────────────────
if selected == "Portfolio Attribution":
    page_portfolio_attribution()

elif selected == "Multifactor Attribution":
    page_multifactor_attribution()

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown(
    "<hr style='border:none;border-top:1px solid #e5e7ef;margin:2rem 0 0.5rem'>",
    unsafe_allow_html=True,
)
st.caption(
    "⚠️ **Alpha version — for academic and research purposes only.** "
    "This tool is under active development and may contain errors or incomplete features. "
    "Results should not be interpreted as financial advice or used for investment decisions. "
    "Use at your own discretion."
)
