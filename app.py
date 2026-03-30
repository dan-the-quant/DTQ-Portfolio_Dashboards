# =============================================================================
# CAPM Attribution Dashboard
# Author: Edgar Alcántara & Daniel R. Barrera
# =============================================================================

# --- Libraries ----------------------------------------------------------------

import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

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
    "Equal-Weight":                  r"config/equal_weighted_portfolio.csv",
    "Mean-Variance":                 r"config/mean_variance_portfolio.csv",
    "Momentum":                      r"config/momentum_portfolio.csv",
    "Zero-Beta":                     r"config/zero_beta_portfolio.csv",
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
    # Flatten MultiIndex columns if yfinance returns them
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    returns = log_returns(prices).dropna()
    # Ensure we return a Series
    if isinstance(returns, pd.DataFrame):
        returns = returns.squeeze()

    returns.index = pd.to_datetime(returns.index)
    return returns


@st.cache_data(show_spinner="Downloading risk-free rate data...")
def get_risk_free_rate(ticker: str, start_date, end_date) -> pd.Series:
    """Download risk-free rate data and convert to daily rate."""
    data = import_prices_data(
        tickers=ticker,
        start_date=start_date,
        end_date=end_date,
    )
    # Flatten MultiIndex columns if needed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    daily_rate = (data / 100 / 360).ffill()
    if isinstance(daily_rate, pd.DataFrame):
        daily_rate = daily_rate.squeeze()

    daily_rate.index = pd.to_datetime(daily_rate.index)
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
    """
    Return the name of the returns column.
    Checks for 'return', 'returns', or the first numeric column as fallback.
    """
    for candidate in ["return", "returns", "Return", "Returns"]:
        if candidate in df.columns:
            return candidate
    # Fallback: first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        return numeric_cols[0]
    return None


# =============================================================================
# SECTION RENDERERS
# =============================================================================

def render_data_uploader():
    """Render file uploader and benchmark/RFR selectors. Returns user inputs."""
    st.subheader("Data Uploader")
    st.info(
        "Upload a file with **two columns**: one for **dates** (daily basis) and one for **returns**. "
        "Supported formats: CSV, Excel (.xlsx / .xls), Parquet, JSON, and TXT."
    )

    col_file, col_bench, col_rfr = st.columns([5, 2.5, 2.5])

    with col_file:
        source = st.radio(
            "Data source",
            options=["Upload my own portfolio", "Use a sample portfolio"],
            horizontal=True,
        )

        if source == "Upload my own portfolio":
            uploaded_file = st.file_uploader(
                "Upload your time series file",
                type=["csv", "xlsx", "xls", "parquet", "json", "txt"],
            )

        else:
            sample_label = st.selectbox(
                "Select a sample portfolio",
                options=list(SAMPLE_PORTFOLIOS.keys()),
            )
            sample_path = SAMPLE_PORTFOLIOS[sample_label]

            if sample_path:
                uploaded_file = open(sample_path, "rb")
            else:
                uploaded_file = None

    with col_bench:
        benchmark_label = st.selectbox("Select a Benchmark", options=list(BENCHMARKS.keys()))

    with col_rfr:
        rfr_label = st.selectbox("Select a Risk-Free Rate", options=list(RISK_FREE_RATES.keys()))

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

        st.markdown("#### Cumulative Returns")
        fig_portfolio = TimeSeriesPlot(
            df[[return_col]].mul(100).cumsum(),
            "Portfolio Cumulative Returns (%)",
        )
        st.plotly_chart(fig_portfolio, use_container_width=True)

    # ── Right column: market data + risk metrics ─────────────────────────────
    with col_right:
        # Align benchmark and RFR to portfolio dates
        market_data = pd.concat([benchmark_returns, rfr_series], axis=1)
        market_data.columns = [benchmark_label, rfr_label]
        aligned = market_data.reindex(df.index)
        missing_pct = aligned.isna().mean().mul(100).round(1)

        if missing_pct.max() > 20:
            st.warning(
                f"⚠️ Up to {missing_pct.max():.1f}% of market data could not be aligned "
                "to your portfolio dates. Results may be affected."
            )

        st.markdown("#### Market Data Preview")
        st.dataframe(
            market_data.style.format("{:.6f}"),
            height=250,
            use_container_width=True,
        )

        # ── Risk Metrics ─────────────────────────────────────────────────────
        st.markdown("#### Risk Metrics")

        portfolio_series = df[return_col]
        bm_series = benchmark_returns.reindex(df.index).dropna()

        ann_return = portfolio_series.mul(100).mean() * TRADING_DAYS
        daily_std = portfolio_series.mul(100).std()
        daily_var = portfolio_series.mul(100).var()
        te = tracking_error(portfolio_series.mul(100), bm_series.mul(100))
        sharpe = sharpe_ratio(df[[return_col]])
        var_95 = value_at_risk(df[[return_col]])
        es = expected_shortfall(df[[return_col]])
        md = max_drawdown(df[[return_col]])
        ced = conditional_expected_drawdown(df[[return_col]])

        c1, c2, c3 = st.columns(3)

        c1.metric("Ann. Return", f"{ann_return:.2f}%")
        c1.metric("Daily Std Dev", f"{daily_std:.4f}%")
        c1.metric("Daily Variance", f"{daily_var:.6f}")

        c2.metric("Sharpe Ratio", f"{sharpe:.4f}x")
        c2.metric("VaR (95%)", f"{var_95:.2%}")
        c2.metric("Expected Shortfall", f"{es:.2%}")

        c3.metric("Tracking Error", f"{te:.4f}%")
        c3.metric("Max Drawdown", f"{md:.2%}")
        c3.metric("Cond. Exp. Drawdown", f"{ced:.2%}")


def render_attribution(df: pd.DataFrame, benchmark_returns: pd.Series, rfr_series: pd.Series,
                       benchmark_ticker: str, return_col: str):
    """Render the Attribution section: rolling CAPM coefficients and factor decomposition."""

    st.markdown("---")
    st.subheader("CAPM Attribution")

    col_coef, _, col_window = st.columns([4.5, 1, 4.5])

    with col_coef:
        coef = st.selectbox(
            "Select a Coefficient to plot",
            options=["Alpha", "Beta", "Sigma"],
        )

    with col_window:
        window_label = st.selectbox(
            "Select a Rolling Window",
            options=list(ROLLING_WINDOWS.keys()),
        )

    window = ROLLING_WINDOWS[window_label]

    portfolio_series = df[return_col]

    # ── Rolling CAPM ─────────────────────────────────────────────────────────
    capm_df = get_rolling_capm(
        portfolio_series,
        benchmark_returns,
        rfr_series,
        window,
    )

    # ── Static CAPM risk attribution ─────────────────────────────────────────
    capm_risk = capm_risk_attribution(
        portfolio_series,
        benchmark_returns,
        rfr_series,
    )
    capm_risk["volatility"] = np.sqrt(capm_risk["variance"])

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

        st.markdown("**CAPM Risk Attribution**")

        display = capm_risk.copy()
        display.index = display.index.str.replace("_", " ").str.title()

        st.dataframe(
            display.style
            .format({
                "variance": "{:.6f}",
                "percentage": "{:.2%}",
                "volatility": "{:.6f}",
            })
            .bar(subset=["percentage"], color="#378ADD", vmin=0, vmax=1),
            use_container_width=True,
            column_config={
                "variance": st.column_config.NumberColumn("Variance"),
                "percentage": st.column_config.NumberColumn("Weight"),
                "volatility": st.column_config.NumberColumn("Volatility"),
            },
        )

    with col_right:
        # Align to the rolling window period (capm_df index)
        r_i = portfolio_series.reindex(capm_df.index).dropna()
        r_m = benchmark_returns.reindex(capm_df.index).dropna()
        beta = capm_df["beta"]

        contribution = factor_contribution(r_i, r_m, beta)
        contribution = pd.concat([r_i, contribution], axis=1)
        contribution.columns = ["Portfolio Returns", "Market Factor", "Residual (Alpha)"]

        fig_contrib = TimeSeriesPlot(
            contribution.mul(100).cumsum(),
            "CAPM Attribution — Cumulative Returns (%)",
        )
        st.plotly_chart(fig_contrib, use_container_width=True)


# =============================================================================
# PAGE: PORTFOLIO ATTRIBUTION
# =============================================================================

def page_portfolio_attribution():
    st.header("Portfolio Attribution")

    # ── 1. Uploader ───────────────────────────────────────────────────────────
    uploaded_file, benchmark_label, rfr_label = render_data_uploader()

    if not uploaded_file:
        st.info("👆 Upload a returns file above to get started.")
        return

    # ── 2. Load & parse file ──────────────────────────────────────────────────
    df_raw = load_file(uploaded_file)
    if df_raw is None:
        st.error("❌ Could not read the uploaded file. Please check the format and try again.")
        return

    df = parse_dataframe(df_raw)
    if df is None:
        st.error("❌ Could not parse the file. Ensure it has a date column and a numeric returns column.")
        return

    # ── 3. Detect return column ───────────────────────────────────────────────
    return_col = get_return_column(df)
    if return_col is None:
        st.error("❌ No numeric returns column found in the uploaded file.")
        return

    if return_col not in ("return", "returns"):
        st.warning(
            f"⚠️ Column `{return_col}` will be used as returns. "
            "Rename it to `return` or `returns` to suppress this warning."
        )

    # ── 4. Download market data ───────────────────────────────────────────────
    benchmark_ticker = BENCHMARKS[benchmark_label]
    rfr_ticker = RISK_FREE_RATES[rfr_label]

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

    # ── 5. Render sections ────────────────────────────────────────────────────
    render_overview(df, benchmark_returns, rfr_series, benchmark_label, rfr_label, return_col)
    render_attribution(df, benchmark_returns, rfr_series, benchmark_ticker, return_col)


# =============================================================================
# PAGE: MULTIFACTOR ATTRIBUTION (placeholder)
# =============================================================================

def page_multifactor_attribution():
    st.header("Multifactor Attribution")
    st.info(
        "🚧 **Coming soon.** This section will support multi-factor attribution models "
        "(e.g., Fama-French 3/5-factor, Carhart 4-factor)."
    )


# =============================================================================
# APP LAYOUT
# =============================================================================

st.set_page_config(
    page_title="DTQ Attribution Dashboard",
    layout="wide",
)

with st.sidebar:
    st.image(r"config/DTQ_logo.png", use_container_width=True)

    st.header("Portfolios Dashboard")

    selected = option_menu(
        None,
        ["Portfolio Attribution", "Multifactor Attribution"],
        icons=["bar-chart-line", "layers"],
        menu_icon="cast",
        default_index=0,
    )

    st.markdown("---")
    st.subheader("A Work by Dan the Quant")
    st.caption("Author: Dan the Quant")

    st.link_button("Daniel R. Barrera", "https://www.linkedin.com/in/danielrbarrera/")
    st.link_button("Edgar M. Alcántara-López", "https://www.linkedin.com/in/edgarallo0/")

# ── Route pages ───────────────────────────────────────────────────────────────
if selected == "Portfolio Attribution":
    page_portfolio_attribution()

elif selected == "Multifactor Attribution":
    page_multifactor_attribution()

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️ **Alpha version — for academic and research purposes only.** "
    "This tool is under active development and may contain errors or incomplete features. "
    "Results should not be interpreted as financial advice or used for investment decisions. "
    "Use at your own discretion."
)
