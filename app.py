# =============================================================================
# Attribution Dashboard
# Author: Edgar Alcántara & Daniel R. Barrera
# =============================================================================

import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import re

from src.portfolios_dashboard.data import (
    load_file,
    parse_dataframe,
    import_prices_data,
    log_returns,
)
from src.portfolios_dashboard.plots import TimeSeriesPlot
from src.portfolios_dashboard.regression import rolling_capm_coefficients
from src.portfolios_dashboard.attribution import (
    factor_contribution,
    capm_risk_attribution,
)
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
    "Betting-Against-Beta": r"config/betting_against_beta_portfolio.csv",
    "Equal-Weighted": r"config/equal_weighted_portfolio.csv",
    "Mean-Variance": r"config/mean_variance_portfolio.csv",
    "Momentum": r"config/momentum_portfolio.csv",
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


@st.cache_data(show_spinner="Running rolling regression...")
def get_rolling_capm(portfolio_returns, benchmark_returns, risk_free_rate, window):
    capm = rolling_capm_coefficients(
        portfolio_returns, benchmark_returns, risk_free_rate, window=window,
    )
    capm_df = pd.concat(capm, axis=1)
    if isinstance(capm_df.columns, pd.MultiIndex):
        capm_df.columns = capm_df.columns.droplevel(1)
    return capm_df


# =============================================================================
# HELPERS
# =============================================================================

def get_return_column(df: pd.DataFrame) -> str | None:
    for candidate in ["return", "returns", "Return", "Returns"]:
        if candidate in df.columns:
            return candidate
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols[0] if numeric_cols else None


def section_label(text: str):
    st.markdown(
        f"<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;"
        f"color:#111440;opacity:0.45;text-transform:uppercase;margin-bottom:8px'>{text}</p>",
        unsafe_allow_html=True,
    )


def divider():
    st.markdown(
        "<hr style='border:none;border-top:1px solid #e5e7ef;margin:1.5rem 0'>",
        unsafe_allow_html=True,
    )


# =============================================================================
# HERO: PLAIN-ENGLISH PERFORMANCE SUMMARY
# =============================================================================

def render_hero_summary(df, capm_risk, capm_df, return_col, benchmark_label, portfolio_label):
    """Lead with the answer — one sentence that explains performance."""

    portfolio_series = df[return_col]
    ann_return = portfolio_series.mul(100).mean() * TRADING_DAYS

    try:
        sys_pct = capm_risk.loc["Systematic", "percentage"] * 100
        idio_pct = capm_risk.loc["Idiosyncratic", "percentage"] * 100
    except KeyError:
        display = capm_risk.copy()
        display.index = display.index.str.replace("_", " ").str.title()
        sys_pct = display.iloc[1]["percentage"] * 100
        idio_pct = display.iloc[2]["percentage"] * 100

    alpha_mean = capm_df["alpha"].mean()
    beta_last = capm_df["beta"].iloc[-1]
    beta_mean = capm_df["beta"].mean()

    alpha_tag = "Positive alpha" if alpha_mean > 0 else "Negative alpha"
    alpha_color = "#5DCAA5" if alpha_mean > 0 else "#F09595"

    if beta_last < beta_mean - 0.2:
        posture = "more defensive than its historical average"
    elif beta_last > beta_mean + 0.2:
        posture = "more aggressive than its historical average"
    else:
        posture = "in line with its historical average"

    date_range = f"{df.index.min().date()} – {df.index.max().date()}"

    summary_html = f"""
    <div style='background:#111440;border-radius:12px;padding:20px 24px;margin-bottom:20px;
                display:flex;align-items:flex-start;justify-content:space-between;gap:20px'>
        <div style='flex:1'>
            <p style='font-size:10px;font-weight:700;letter-spacing:0.1em;
                      color:rgba(255,255,255,0.4);text-transform:uppercase;margin-bottom:8px'>
                Performance summary &nbsp;·&nbsp; {portfolio_label} &nbsp;·&nbsp; {date_range}
            </p>
            <p style='font-size:16px;font-weight:600;color:#fff;line-height:1.6;margin:0'>
                This portfolio earned <span style='color:#7EC8C8'>{ann_return:.2f}% annualized</span>
                — of which <strong style='color:#fff'>{sys_pct:.1f}% came from broad market exposure</strong>
                and <strong style='color:#fff'>{idio_pct:.1f}% from strategy-specific decisions.</strong>
                Current market sensitivity is <span style='color:#7EC8C8'>{posture}</span>.
            </p>
        </div>
        <div style='text-align:right;flex-shrink:0'>
            <div style='font-size:28px;font-weight:700;color:#fff'>{ann_return:.2f}%</div>
            <div style='font-size:10px;color:rgba(255,255,255,0.4);text-transform:uppercase;
                        letter-spacing:0.06em;margin-top:2px'>Annualized return</div>
            <div style='margin-top:10px;display:inline-block;padding:3px 10px;border-radius:20px;
                        background:rgba(255,255,255,0.1);
                        font-size:11px;font-weight:600;color:{alpha_color}'>
                {alpha_tag}
            </div>
        </div>
    </div>
    """
    st.markdown(summary_html, unsafe_allow_html=True)


# =============================================================================
# SECTION: DATA CONTROLS  →  now rendered inside st.sidebar
# =============================================================================

def render_data_controls():
    """All portfolio controls live in the sidebar."""
    with st.sidebar:
        st.image("config/DTQ_logo.png", use_container_width=True)

        st.markdown(
            "<p style='font-size:10px;font-weight:700;letter-spacing:0.1em;"
            "color:#111440;opacity:0.45;text-transform:uppercase;margin-bottom:4px'>"
            "Portfolio setup</p>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Upload a file with **two columns**: dates (daily) and returns. "
            "Supported: CSV, Excel, Parquet, JSON, TXT."
        )

        source = st.radio(
            "Source",
            options=["Upload my own portfolio", "Use a sample portfolio"],
            horizontal=False,
            label_visibility="collapsed",
        )

        if source == "Upload my own portfolio":
            uploaded_file = st.file_uploader(
                "Upload your time series file",
                type=["csv", "xlsx", "xls", "parquet", "json", "txt"],
                label_visibility="collapsed",
            )
            portfolio_label = "Custom portfolio"
        else:
            sample_label = st.selectbox(
                "Portfolio",
                options=list(SAMPLE_PORTFOLIOS.keys()),
                label_visibility="collapsed",
            )
            sample_path = SAMPLE_PORTFOLIOS[sample_label]
            uploaded_file = open(sample_path, "rb") if sample_path else None
            portfolio_label = (
                sample_label
                if sample_label != "— Select a sample portfolio —"
                else "Sample portfolio"
            )

        st.markdown("---")

        benchmark_label = st.selectbox("Benchmark", options=list(BENCHMARKS.keys()))
        rfr_label = st.selectbox("Risk-Free Rate", options=list(RISK_FREE_RATES.keys()))

        st.markdown("---")

        # Rolling window also moved here so it's always visible
        window_label = st.selectbox(
            "Rolling window",
            options=list(ROLLING_WINDOWS.keys()),
        )

    return uploaded_file, benchmark_label, rfr_label, portfolio_label, window_label


# =============================================================================
# SECTION: OVERVIEW
# =============================================================================

def render_overview(df, benchmark_returns, rfr_series, benchmark_label, rfr_label, return_col):
    col_left, col_right = st.columns(2)

    with col_left:
        section_label("Cumulative return")

        c1, c2 = st.columns(2)
        c1.metric("Start date", str(df.index.min().date()))
        c2.metric("End date", str(df.index.max().date()))

        # ── CHANGE 2: overlay benchmark on the same chart ─────────────────────
        portfolio_cum = df[[return_col]].mul(100).cumsum()
        portfolio_cum.columns = ["Portfolio"]

        bm_aligned = benchmark_returns.reindex(df.index).fillna(method="ffill")
        bm_cum = bm_aligned.mul(100).cumsum().to_frame(name=benchmark_label)

        combined_cum = pd.concat([portfolio_cum, bm_cum], axis=1).dropna()

        fig_portfolio = TimeSeriesPlot(combined_cum, "Cumulative Returns (%)")
        st.plotly_chart(fig_portfolio, use_container_width=True)

    with col_right:
        # ── Data Quality minicard ─────────────────────────────────────────────
        section_label("Data quality")

        market_data = pd.concat([benchmark_returns, rfr_series], axis=1)
        market_data.columns = [benchmark_label, rfr_label]
        missing_pct = market_data.reindex(df.index).isna().mean().mul(100).round(1)

        n_obs = len(df)
        bm_missing = missing_pct[benchmark_label]
        rfr_missing = missing_pct[rfr_label]
        date_start = df.index.min().date()
        date_end = df.index.max().date()

        def _dq_icon(pct):
            return "✅" if pct < 5 else ("⚠️" if pct < 20 else "🔴")

        def _dq_color(pct):
            return "#1a7a5e" if pct < 5 else ("#b45309" if pct < 20 else "#991b1b")

        def _dq_bg(pct):
            return "#f0fdf9" if pct < 5 else ("#fffbeb" if pct < 20 else "#fff1f2")

        overall_ok = bm_missing < 5 and rfr_missing < 5
        card_icon = "✅" if overall_ok else "⚠️"
        card_title = "Data looks good" if overall_ok else "Data quality issues detected"
        card_color = "#1a7a5e" if overall_ok else "#b45309"
        card_bg = "#f0fdf9" if overall_ok else "#fffbeb"
        card_border = "#6ee7b7" if overall_ok else "#fcd34d"

        rows_html = f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding:6px 0;border-bottom:1px solid rgba(0,0,0,0.06)'>
                <span style='font-size:11px;color:#555'>Observations</span>
                <span style='font-size:12px;font-weight:600;color:#111440'>{n_obs:,}</span>
            </div>
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding:6px 0;border-bottom:1px solid rgba(0,0,0,0.06)'>
                <span style='font-size:11px;color:#555'>Date range</span>
                <span style='font-size:12px;font-weight:600;color:#111440'>{date_start} – {date_end}</span>
            </div>
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding:6px 0;border-bottom:1px solid rgba(0,0,0,0.06)'>
                <span style='font-size:11px;color:#555'>{_dq_icon(bm_missing)} Benchmark missing</span>
                <span style='font-size:12px;font-weight:600;color:{_dq_color(bm_missing)}'>{bm_missing:.1f}%</span>
            </div>
            <div style='display:flex;justify-content:space-between;align-items:center;padding:6px 0'>
                <span style='font-size:11px;color:#555'>{_dq_icon(rfr_missing)} Risk-free rate missing</span>
                <span style='font-size:12px;font-weight:600;color:{_dq_color(rfr_missing)}'>{rfr_missing:.1f}%</span>
            </div>
        """

        st.markdown(f"""
            <div style='background:{card_bg};border:1px solid {card_border};border-radius:10px;
                        padding:14px 18px;margin-bottom:16px'>
                <div style='display:flex;align-items:center;gap:8px;margin-bottom:10px'>
                    <span style='font-size:16px'>{card_icon}</span>
                    <span style='font-size:12px;font-weight:700;color:{card_color};
                                 letter-spacing:0.02em'>{card_title}</span>
                </div>
                {rows_html}
            </div>
        """, unsafe_allow_html=True)

        # ── Risk & performance metrics ────────────────────────────────────────
        section_label("Risk & performance")

        portfolio_series = df[return_col]
        bm_series = benchmark_returns.reindex(df.index).dropna()

        ann_return = portfolio_series.mul(100).mean() * TRADING_DAYS
        ann_std = portfolio_series.mul(100).std() * np.sqrt(TRADING_DAYS)
        te = tracking_error(portfolio_series.mul(100), bm_series.mul(100))
        sharpe = sharpe_ratio(df[[return_col]]) * np.sqrt(TRADING_DAYS)
        var_95 = value_at_risk(df[[return_col]])
        es = expected_shortfall(df[[return_col]])
        md = max_drawdown(df[[return_col]])
        ced = conditional_expected_drawdown(df[[return_col]])

        # 2 columns × 4 rows — bigger tiles, fills the remaining space
        c1, c2 = st.columns(2)
        c1.metric("Annualized return", f"{ann_return:.2f}%")
        c2.metric("Annualized vol.", f"{ann_std:.2f}%")
        c1.metric("Sharpe ratio", f"{sharpe:.4f}x")
        c2.metric("Value at risk (95%)", f"{var_95:.2%}")
        c1.metric("Expected shortfall", f"{es:.2%}")
        c2.metric("Tracking error", f"{te:.4f}%")
        c1.metric("Max drawdown", f"{md:.2%}")
        c2.metric("Cond. exp. drawdown", f"{ced:.2%}")


# =============================================================================
# SECTION: INTERPRETATION
# =============================================================================

def render_interpretation(capm_df, capm_risk):
    """Plain-english interpretation for advisors."""

    alpha_mean = capm_df["alpha"].mean()
    beta_mean = capm_df["beta"].mean()
    beta_last = capm_df["beta"].iloc[-1]

    try:
        sys_pct = capm_risk.loc["Systematic", "percentage"]
        idio_pct = capm_risk.loc["Idiosyncratic", "percentage"]
    except KeyError:
        display = capm_risk.copy()
        display.index = display.index.str.replace("_", " ").str.title()
        sys_pct = display.iloc[1]["percentage"]
        idio_pct = display.iloc[2]["percentage"]

    if alpha_mean > 0:
        alpha_msg = f"The portfolio has generated **positive alpha** on average ({alpha_mean * 100:.3f}% daily), suggesting the strategy adds value beyond market exposure."
    else:
        alpha_msg = f"The portfolio shows **negative alpha** on average ({alpha_mean * 100:.3f}% daily), indicating returns have lagged what market exposure alone would predict."

    if beta_mean < 0.8:
        beta_msg = f"With an average market sensitivity of **{beta_mean:.2f}**, the portfolio is **defensive** — it moves less than the market and may offer downside protection."
    elif beta_mean <= 1.2:
        beta_msg = f"With an average market sensitivity of **{beta_mean:.2f}**, the portfolio tracks the market **closely** with similar risk exposure."
    else:
        beta_msg = f"With an average market sensitivity of **{beta_mean:.2f}**, the portfolio is **aggressive** — it amplifies market movements and carries higher systematic risk."

    if abs(beta_last - beta_mean) > 0.2:
        trend_msg = f"Recently, market sensitivity has shifted to **{beta_last:.2f}**, which is notably {'higher' if beta_last > beta_mean else 'lower'} than its historical average."
    else:
        trend_msg = f"Current market sensitivity (**{beta_last:.2f}**) is consistent with the historical average — no significant change in posture detected."

    risk_msg = f"**{sys_pct * 100:.1f}%** of total variance comes from broad market exposure, while **{idio_pct * 100:.1f}%** is specific to the strategy."

    section_label("What this means")
    for msg in [alpha_msg, beta_msg, trend_msg, risk_msg]:
        html_msg = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', msg)
        st.markdown(
            f"<p style='font-size:13px;color:#444;line-height:1.7;margin-bottom:10px'>{html_msg}</p>",
            unsafe_allow_html=True,
        )


# =============================================================================
# SECTION: ATTRIBUTION
# =============================================================================

def render_attribution(df, benchmark_returns, rfr_series, benchmark_ticker, return_col, window_label):
    divider()
    section_label("Return decomposition")

    # Rolling window now comes from the sidebar; only the coef selector stays here
    coef = st.selectbox(
        "View",
        options=["Market sensitivity (Beta)", "Alpha over time", "Residual risk (Sigma)"],
        index=0,
    )

    window = ROLLING_WINDOWS[window_label]
    portfolio_series = df[return_col]

    capm_df = get_rolling_capm(
        portfolio_series, benchmark_returns, rfr_series, window,
    )

    capm_risk = capm_risk_attribution(portfolio_series, benchmark_returns, rfr_series)
    capm_risk_clean = capm_risk.copy()
    capm_risk["ann_volatility"] = np.sqrt(capm_risk["variance"]) * np.sqrt(TRADING_DAYS) / 100

    col_left, col_right = st.columns(2)

    with col_left:
        if "Beta" in coef:
            series_to_plot = capm_df["beta"]
            chart_title = "Market sensitivity over time"
        elif "Alpha" in coef:
            series_to_plot = capm_df["alpha"].cumsum()
            chart_title = "Cumulative alpha (%)"
        else:
            series_to_plot = capm_df["sigma"]
            chart_title = "Residual risk over time"

        fig_coef = TimeSeriesPlot(series_to_plot, chart_title)
        st.plotly_chart(fig_coef, use_container_width=True)

        section_label("Where the risk comes from")

        # ── CHANGE 5: drop variance column from risk decomposition table ──────
        display = capm_risk.copy()
        display = display.drop(columns=["variance"], errors="ignore")
        display.index = (
            display.index
            .str.replace("total_variance", "Total portfolio")
            .str.replace("systematic_variance", "Market exposure")
            .str.replace("idio_variance", "Strategy-specific")
        )

        st.dataframe(
            display.style
            .format({
                "percentage": "{:.2%}",
                "ann_volatility": "{:.2%}",
            })
            .bar(subset=["percentage"], color="#111440", vmin=0, vmax=1)
            .background_gradient(subset=["ann_volatility"], cmap="Blues"),
            use_container_width=True,
            column_config={
                "percentage": st.column_config.NumberColumn("Share of risk"),
                "ann_volatility": st.column_config.NumberColumn("Annualized vol."),
            },
        )

    with col_right:
        r_i = portfolio_series.reindex(capm_df.index).dropna()
        r_m = benchmark_returns.reindex(capm_df.index).dropna()
        beta = capm_df["beta"]

        contribution = factor_contribution(r_i, r_m, beta)
        contribution = pd.concat([r_i, contribution], axis=1)
        contribution.columns = ["Total portfolio", "Market exposure", "Strategy alpha"]

        fig_contrib = TimeSeriesPlot(
            contribution.mul(100).cumsum(),
            "What drove the return — cumulative (%)",
        )
        st.plotly_chart(fig_contrib, use_container_width=True)

        render_interpretation(capm_df, capm_risk_clean)


# =============================================================================
# PAGES
# =============================================================================

def page_portfolio_attribution():
    # render_data_controls now returns window_label as well
    uploaded_file, benchmark_label, rfr_label, portfolio_label, window_label = render_data_controls()

    if not uploaded_file:
        st.info("👆 Upload a returns file or select a sample portfolio in the sidebar to get started.")
        return

    df_raw = load_file(uploaded_file)
    if df_raw is None:
        st.error("❌ Could not read the file. Please check the format and try again.")
        return

    df = parse_dataframe(df_raw)
    if df is None:
        st.error("❌ Could not parse the file. Ensure it has a date column and a numeric returns column.")
        return

    return_col = get_return_column(df)
    if return_col is None:
        st.error("❌ No numeric returns column found.")
        return

    if return_col not in ("return", "returns"):
        st.warning(
            f"⚠️ Column `{return_col}` will be used as returns. "
            "Rename it to `return` or `returns` to suppress this warning."
        )

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

    # ── Compute CAPM early for hero summary ───────────────────────────────────
    capm_df_preview = get_rolling_capm(
        df[return_col], benchmark_returns, rfr_series, window=252,
    )
    capm_risk_preview = capm_risk_attribution(df[return_col], benchmark_returns, rfr_series)
    capm_risk_display = capm_risk_preview.copy()
    capm_risk_display.index = (
        capm_risk_display.index
        .str.replace("total_variance", "Total portfolio")
        .str.replace("systematic_variance", "Systematic")
        .str.replace("idio_variance", "Idiosyncratic")
    )

    render_hero_summary(df, capm_risk_display, capm_df_preview, return_col, benchmark_label, portfolio_label)
    render_overview(df, benchmark_returns, rfr_series, benchmark_label, rfr_label, return_col)
    render_attribution(df, benchmark_returns, rfr_series, benchmark_ticker, return_col, window_label)


def page_multifactor_attribution():
    st.markdown("<h2 style='margin-bottom:4px'>Multifactor attribution</h2>", unsafe_allow_html=True)
    st.info(
        "🚧 **Coming soon.** This section will support multi-factor attribution models "
        "(e.g., Fama-French 3/5-factor, Carhart 4-factor)."
    )


# =============================================================================
# APP ENTRY POINT
# =============================================================================

st.set_page_config(
    page_title="CAPM · Attribution",
    page_icon="config/DTQ_logo.png",
    layout="wide",
)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"], p, div, span, label {
            font-family: 'Inter', sans-serif !important;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
        }
        .block-container { padding-top: 0 !important; }
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

# ── Navbar ────────────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

st.markdown("""
    <div style='background:#111440;padding:12px 28px;display:flex;align-items:center;
                justify-content:space-between;margin-bottom:24px'>
        <div style='color:#fff;font-size:14px;font-weight:600;letter-spacing:0.02em'>
            CAPM Attribution
            <span style='opacity:0.4;font-weight:400;margin-left:6px;font-size:12px'>· Attribution</span>
        </div>
        <div style='display:flex;flex-direction:column;gap:3px;align-items:flex-end'>
            <a href='https://www.linkedin.com/in/danielrbarrera/' target='_blank'
               style='color:rgba(255,255,255,0.4);font-size:10px;text-decoration:none'>
               Daniel R. Barrera
            </a>
            <a href='https://www.linkedin.com/in/edgarallo0/' target='_blank'
               style='color:rgba(255,255,255,0.4);font-size:10px;text-decoration:none'>
               Edgar M. Alcántara-López
            </a>
        </div>
    </div>
""", unsafe_allow_html=True)

# ── Tab navigation ────────────────────────────────────────────────────────────
col_nav, _ = st.columns([4, 6])
with col_nav:
    selected = option_menu(
        None,
        ["Portfolio view", "Multifactor"],
        icons=["bar-chart-line", "layers"],
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {
                "padding": "0",
                "background-color": "transparent",
                "border-bottom": "1.5px solid #e5e7ef",
            },
            "nav-link": {
                "color": "#888",
                "font-size": "13px",
                "font-family": "Inter, sans-serif",
                "font-weight": "500",
                "padding": "8px 16px",
                "border-radius": "0",
                "background-color": "transparent",
            },
            "nav-link-selected": {
                "background-color": "transparent",
                "color": "#111440",
                "font-weight": "600",
                "font-family": "Inter, sans-serif",
                "border-bottom": "2px solid #111440",
            },
            "icon": {"display": "none"},
        }
    )

st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

# ── Router ────────────────────────────────────────────────────────────────────
if selected == "Portfolio view":
    page_portfolio_attribution()
elif selected == "Multifactor":
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
