# =============================================================================
# Attribution Dashboard — Narrative-first layout (alt.py)
# Author: Edgar Alcántara & Daniel R. Barrera
# =============================================================================

import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
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
    "Equal-Weighted":        r"config/equal_weighted_portfolio.csv",
    "Mean-Variance":         r"config/mean_variance_portfolio.csv",
    "Momentum":              r"config/momentum_portfolio.csv",
}

PORTFOLIO_DESCRIPTIONS = {
    "Betting-Against-Beta": (
        "**Betting-Against-Beta (BAB)** goes long low-beta stocks and short high-beta stocks, "
        "exploiting the tendency of lower-risk assets to outperform on a risk-adjusted basis. "
        "Developed by Frazzini & Pedersen (2014), it profits when investors over-leverage "
        "high-beta assets and underprice low-beta ones."
    ),
    "Equal-Weighted": (
        "**Equal-Weighted** allocates the same dollar amount to every asset in the universe, "
        "rebalancing periodically. It captures the size premium by overweighting smaller stocks "
        "relative to market-cap weighting, at the cost of higher turnover."
    ),
    "Mean-Variance": (
        "**Mean-Variance** (Markowitz, 1952) selects weights that maximize expected return "
        "for a given level of volatility. It sits on the efficient frontier and is sensitive "
        "to input estimates."
    ),
    "Momentum": (
        "**Momentum** buys recent winners and sells recent losers, based on the observation "
        "that assets trending upward over the past 3–12 months tend to continue doing so. "
        "It is one of the most robust cross-sectional anomalies in the literature."
    ),
}

UPLOAD_SAMPLE_PATH = r"config/zero_beta_portfolio.csv"

BENCHMARKS = {
    "S&P 500 ETF (SPY)":           "SPY",
    "S&P 500 Index (^GSPC)":       "^GSPC",
    "MSCI World (URTH)":           "URTH",
    "MSCI Emerging Markets (EEM)": "EEM",
    "Total Stock Market (VTI)":    "VTI",
    "MSCI ACWI (ACWI)":            "ACWI",
    "Russell 2000 (^RUT)":         "^RUT",
    "Dow Jones (^DJI)":            "^DJI",
    "NASDAQ 100 (^NDX)":           "^NDX",
}

RISK_FREE_RATES = {
    "US Treasury 3 Months (^IRX)": "^IRX",
    "US Treasury 5 Years (^FVX)":  "^FVX",
    "US Treasury 10 Years (^TNX)": "^TNX",
    "US Treasury 30 Years (^TYX)": "^TYX",
}

ROLLING_WINDOWS = {
    "1 Year (252)":  252,
    "2 Years (504)": 504,
    "3 Years (756)": 756,
    "4 Years (1008)":1008,
    "5 Years (1260)":1260,
}

TRADING_DAYS = 252


# =============================================================================
# CACHED DATA LOADERS  (unchanged from app.py)
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


def section_tag(text: str):
    """Small uppercase label — used as section eyebrow."""
    st.markdown(
        f"<p style='font-size:20px;font-weight:700;letter-spacing:0.12em;"
        f"color:#111440;opacity:0.4;text-transform:uppercase;margin:0 0 6px'>{text}</p>",
        unsafe_allow_html=True,
    )


def divider():
    st.markdown(
        "<hr style='border:none;border-top:1px solid #e5e7ef;margin:2rem 0'>",
        unsafe_allow_html=True,
    )


# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================

def render_sidebar():
    with st.sidebar:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("config/DTQ_logo.png", use_container_width=True)
        st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)

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
            # Sample file download
            try:
                with open(UPLOAD_SAMPLE_PATH, "rb") as f:
                    sample_bytes = f.read()
                st.download_button(
                    label="⬇ Download sample file",
                    data=sample_bytes,
                    file_name="sample_portfolio.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            except Exception:
                pass
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
            # Portfolio description box
            desc = PORTFOLIO_DESCRIPTIONS.get(sample_label)
            if desc:
                st.info(desc)

        st.markdown("---")
        benchmark_label = st.selectbox("Benchmark", options=list(BENCHMARKS.keys()))
        rfr_label       = st.selectbox("Risk-Free Rate", options=list(RISK_FREE_RATES.keys()))
        st.markdown("---")
        window_label    = st.selectbox(
            "Rolling window",
            options=list(ROLLING_WINDOWS.keys()),
            help=(
                "The rolling window sets how many trading days are used to estimate "
                "each point in the rolling regression.\n\n"
                "**Shorter window** (e.g. 1 year): more reactive — captures recent shifts "
                "in beta and alpha, but noisier.\n\n"
                "**Longer window** (e.g. 5 years): smoother and more stable estimates, "
                "but slower to reflect structural changes in the portfolio."
            ),
        )

    return uploaded_file, benchmark_label, rfr_label, portfolio_label, window_label


# =============================================================================
# SECTION 0 — HEADLINE
# =============================================================================

def render_headline(df, capm_risk, capm_df, return_col, benchmark_label, portfolio_label):
    portfolio_series = df[return_col]
    ann_return       = portfolio_series.mul(100).mean() * TRADING_DAYS
    gained           = ann_return >= 0
    verb             = "gained" if gained else "lost"
    value_color      = "#5DCAA5" if gained else "#F09595"
    date_range       = f"{df.index.min().date()} – {df.index.max().date()}"

    try:
        sys_pct  = capm_risk.loc["Systematic", "percentage"] * 100
        idio_pct = capm_risk.loc["Idiosyncratic", "percentage"] * 100
    except KeyError:
        tmp = capm_risk.copy()
        tmp.index = tmp.index.str.replace("_", " ").str.title()
        sys_pct  = tmp.iloc[1]["percentage"] * 100
        idio_pct = tmp.iloc[2]["percentage"] * 100

    alpha_mean  = capm_df["alpha"].mean()
    alpha_label = "positive alpha" if alpha_mean > 0 else "negative alpha"
    alpha_color = "#5DCAA5" if alpha_mean > 0 else "#F09595"

    st.markdown(f"""
        <div style='border-left:4px solid #111440;padding:18px 24px;margin-bottom:8px;
                    background:#fafbff;border-radius:0 8px 8px 0'>
            <p style='font-size:10px;font-weight:700;letter-spacing:0.12em;color:#111440;
                      opacity:0.4;text-transform:uppercase;margin:0 0 10px'>
                {portfolio_label} &nbsp;·&nbsp; {date_range} &nbsp;·&nbsp; vs. {benchmark_label}
            </p>
            <p style='font-size:26px;font-weight:700;color:#111440;line-height:1.35;margin:0'>
                The portfolio <span style='color:{value_color}'>{verb} {abs(ann_return):.2f}%
                annualized</span> — {sys_pct:.0f}% from market exposure,
                {idio_pct:.0f}% from strategy decisions,
                with <span style='color:{alpha_color}'>{alpha_label}</span>.
            </p>
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# SECTION 1 — MARKET EXPOSURE VS. STRATEGY
# =============================================================================

def render_insight_1(df, benchmark_returns, capm_df, return_col, benchmark_label):
    divider()

    portfolio_series = df[return_col]
    ann_return       = portfolio_series.mul(100).mean() * TRADING_DAYS

    try:
        r_i  = portfolio_series.reindex(capm_df.index).dropna()
        r_m  = benchmark_returns.reindex(capm_df.index).dropna()
        beta = capm_df["beta"]
        contribution = factor_contribution(r_i, r_m, beta)
        contribution = pd.concat([r_i, contribution], axis=1)
        contribution.columns = ["Total portfolio", "Market exposure", "Strategy alpha"]
        has_contrib = True
    except Exception:
        has_contrib = False

    # ── Insight text ──────────────────────────────────────────────────────────
    if has_contrib:
        excess_ann = (contribution["Market exposure"] + contribution["Strategy alpha"]).mean() * TRADING_DAYS * 100
        mkt_ann    = contribution["Market exposure"].mean() * TRADING_DAYS * 100
        alpha_ann  = contribution["Strategy alpha"].mean() * TRADING_DAYS * 100
        direction  = "added" if alpha_ann >= 0 else "detracted"
        insight = (
            f"Of the {excess_ann:.2f}% annualized excess return, market beta contributed "
            f"<strong>{mkt_ann:.2f}%</strong> while strategy decisions "
            f"<strong>{direction} {abs(alpha_ann):.2f}%</strong>."
        )
    else:
        insight = "Return decomposition between market exposure and strategy alpha."

    col_text, col_chart = st.columns([3, 5])

    with col_text:
        st.markdown(f"""
            <div>
                <p style='font-size:20px;font-weight:700;letter-spacing:0.12em;
                          color:#111440;opacity:0.4;text-transform:uppercase;margin:0 0 6px'>
                    01 · Market exposure vs. strategy
                </p>
                <p style='font-size:22px;font-weight:600;color:#111440;line-height:1.55;margin:0 0 16px'>
                    {insight}
                </p>
                <p style='font-size:15px;color:#888;line-height:1.7;margin:0'>
                    Cumulative return decomposed into the portion explained by broad market
                    movement (beta × benchmark return) and the residual attributed to
                    strategy-specific decisions (alpha).
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col_chart:
        if has_contrib:
            fig = TimeSeriesPlot(
                contribution.mul(100).cumsum(),
                "Cumulative return decomposition (%)",
            )
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# SECTION 2 — MARKET SENSITIVITY (BETA)
# =============================================================================

def render_insight_2(capm_df):
    divider()

    beta_mean = capm_df["beta"].mean()
    beta_last = capm_df["beta"].iloc[-1]
    delta     = beta_last - beta_mean

    if beta_mean < 0.8:
        character = "defensive"
        char_desc = "moves less than the market and may offer downside protection"
    elif beta_mean <= 1.2:
        character = "market-neutral"
        char_desc = "tracks broad market movements closely"
    else:
        character = "aggressive"
        char_desc = "amplifies market swings and carries elevated systematic risk"

    if abs(delta) > 0.2:
        shift = (
            f"Recently it has shifted to <strong>{beta_last:.2f}</strong> — "
            f"{'higher' if delta > 0 else 'lower'} than its historical average."
        )
    else:
        shift = f"Current sensitivity (<strong>{beta_last:.2f}</strong>) is consistent with its historical average."

    insight = (
        f"With an average beta of <strong>{beta_mean:.2f}</strong>, the portfolio is "
        f"<strong>{character}</strong> — it {char_desc}. {shift}"
    )

    col_chart, col_text = st.columns([4, 3])

    with col_text:
        st.markdown(f"""
            <div>
                <p style='font-size:20px;font-weight:700;letter-spacing:0.12em;
                          color:#111440;opacity:0.4;text-transform:uppercase;margin:0 0 6px'>
                    02 · Market sensitivity (Beta)
                </p>
                <p style='font-size:22px;font-weight:600;color:#111440;line-height:1.55;margin:0 0 16px'>
                    {insight}
                </p>
                <p style='font-size:15px;color:#888;line-height:1.7;margin:0'>
                    Rolling beta measures how much the portfolio moves relative to the benchmark
                    over time. A beta of 1 means the portfolio moves in lockstep; below 1 is
                    more defensive, above 1 more aggressive.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col_chart:
        fig = TimeSeriesPlot(capm_df["beta"], "Rolling market sensitivity (Beta)")
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# SECTION 3 — ALPHA CONSISTENCY
# =============================================================================

def render_insight_3(capm_df):
    divider()

    alpha_cum  = capm_df["alpha"].cumsum()
    alpha_mean = capm_df["alpha"].mean() * 252
    alpha_std  = capm_df["alpha"].std() * np.sqrt(252)
    pos_pct    = (capm_df["alpha"] > 0).mean() * 100

    if alpha_mean > 0 and pos_pct > 55:
        verdict = "consistently generated positive alpha"
        verdict_color = "#1a7a5e"
    elif alpha_mean > 0:
        verdict = "generated positive alpha on balance, though inconsistently"
        verdict_color = "#b45309"
    elif alpha_mean < 0 and pos_pct < 45:
        verdict = "consistently destroyed alpha relative to the benchmark"
        verdict_color = "#991b1b"
    else:
        verdict = "produced mixed alpha — positive in some periods, negative in others"
        verdict_color = "#b45309"

    insight = (
        f"The strategy <strong style='color:{verdict_color}'>{verdict}</strong>. "
        f"Alpha was positive in <strong>{pos_pct:.0f}%</strong> of rolling windows, "
        f"with an annualized average of <strong>{alpha_mean*100:.3f}%</strong> "
        f"and an annualized volatility of {alpha_std*100:.3f}%."
    )

    col_text, col_chart = st.columns([3, 4])

    with col_text:
        st.markdown(f"""
            <div>
                <p style='font-size:20px;font-weight:700;letter-spacing:0.12em;
                          color:#111440;opacity:0.4;text-transform:uppercase;margin:0 0 6px'>
                    03 · Alpha consistency
                </p>
                <p style='font-size:22px;font-weight:600;color:#111440;line-height:1.55;margin:0 0 16px'>
                    {insight}
                </p>
                <p style='font-size:15px;color:#888;line-height:1.7;margin:0'>
                    Cumulative alpha shows whether outperformance was structural or episodic.
                    A steadily rising line indicates a robust strategy; a flat or declining
                    line suggests returns were driven primarily by market exposure.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col_chart:
        fig = TimeSeriesPlot(alpha_cum.mul(100), "Cumulative alpha (%)")
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# APPENDIX — RISK METRICS + CUMULATIVE RETURN
# =============================================================================

def render_appendix(df, benchmark_returns, rfr_series, benchmark_label, rfr_label, return_col):
    divider()
    section_tag("04 · Risk metrics & cumulative return")

    col_left, col_right = st.columns(2)

    with col_right:
        # Geometric cumulative return (exp(cumsum of log returns) - 1) * 100
        portfolio_cum = (np.expm1(df[return_col].cumsum()) * 100).to_frame(name="Portfolio")
        bm_aligned    = benchmark_returns.reindex(df.index).ffill()
        bm_cum        = (np.expm1(bm_aligned.cumsum()) * 100).to_frame(name=benchmark_label)
        combined      = pd.concat([portfolio_cum, bm_cum], axis=1).dropna()

        fig = TimeSeriesPlot(combined, "Cumulative return (%) — geometric")
        st.plotly_chart(fig, use_container_width=True)

    METRIC_HELP = {
        "Annualized return":   "Average daily return scaled to a full year (×252 trading days). Positive means the portfolio grew on average.",
        "Annualized vol.":     "Standard deviation of daily returns scaled to a year. Higher vol. = wider swings up and down.",
        "Sharpe ratio":        "Excess return per unit of total risk (return − risk-free rate) ÷ volatility. Higher is better; >1 is generally considered good.",
        "Value at risk (95%)": "The worst expected daily loss 95% of the time. E.g. −1.5% means on most days losses won't exceed 1.5%.",
        "Expected shortfall":  "Average loss on the worst 5% of days (also called CVaR). A more conservative tail-risk measure than VaR.",
        "Max drawdown":        "Largest peak-to-trough decline in the portfolio's history. Measures the worst cumulative loss an investor could have experienced.",
        "Tracking error":      "Annualized standard deviation of the difference between portfolio and benchmark returns. Lower = closer to the benchmark.",
        "Cond. exp. drawdown": "Average of the worst drawdowns (conditional expected drawdown). Captures sustained loss periods, not just the single worst one.",
    }

    with col_left:
        portfolio_series = df[return_col]
        bm_series        = benchmark_returns.reindex(df.index).dropna()

        ann_return = portfolio_series.mul(100).mean() * TRADING_DAYS
        ann_std    = portfolio_series.mul(100).std() * np.sqrt(TRADING_DAYS)
        te         = tracking_error(portfolio_series.mul(100), bm_series.mul(100))
        sharpe     = sharpe_ratio(df[[return_col]]) * np.sqrt(TRADING_DAYS)
        var_95     = value_at_risk(df[[return_col]])
        es         = expected_shortfall(df[[return_col]])
        md         = max_drawdown(df[[return_col]])
        ced        = conditional_expected_drawdown(df[[return_col]])

        c1, c2 = st.columns(2)
        c1.metric("Annualized return",   f"{ann_return:.2f}%",  help=METRIC_HELP["Annualized return"])
        c2.metric("Annualized vol.",     f"{ann_std:.2f}%",     help=METRIC_HELP["Annualized vol."])
        c1.metric("Sharpe ratio",        f"{sharpe:.4f}x",      help=METRIC_HELP["Sharpe ratio"])
        c2.metric("Value at risk (95%)", f"{var_95:.2%}",       help=METRIC_HELP["Value at risk (95%)"])
        c1.metric("Expected shortfall",  f"{es:.2%}",           help=METRIC_HELP["Expected shortfall"])
        c2.metric("Tracking error",      f"{te:.4f}%",          help=METRIC_HELP["Tracking error"])
        c1.metric("Max drawdown",        f"{md:.2%}",           help=METRIC_HELP["Max drawdown"])
        c2.metric("Cond. exp. drawdown", f"{ced:.2%}",          help=METRIC_HELP["Cond. exp. drawdown"])

        # Data quality warning if needed
        market_data = pd.concat([benchmark_returns, rfr_series], axis=1)
        market_data.columns = [benchmark_label, rfr_label]
        missing_pct = market_data.reindex(df.index).isna().mean().mul(100).round(1)
        if missing_pct.max() > 20:
            st.warning(
                f"⚠️ Up to {missing_pct.max():.1f}% of market data could not be aligned "
                "to your portfolio dates. Results may be affected."
            )


# =============================================================================
# PAGE
# =============================================================================

def page_portfolio_attribution():
    uploaded_file, benchmark_label, rfr_label, portfolio_label, window_label = render_sidebar()

    if not uploaded_file:
        st.markdown(
            "<div style='margin-top:80px;text-align:center;color:#aaa;font-size:14px'>"
            "👈 Upload a returns file or select a sample portfolio in the sidebar."
            "</div>",
            unsafe_allow_html=True,
        )
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
    rfr_ticker       = RISK_FREE_RATES[rfr_label]
    window           = ROLLING_WINDOWS[window_label]

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

    capm_df = get_rolling_capm(df[return_col], benchmark_returns, rfr_series, window)

    capm_risk = capm_risk_attribution(df[return_col], benchmark_returns, rfr_series)
    capm_risk_labeled = capm_risk.copy()
    capm_risk_labeled.index = (
        capm_risk_labeled.index
        .str.replace("total_variance",      "Total portfolio")
        .str.replace("systematic_variance", "Systematic")
        .str.replace("idio_variance",       "Idiosyncratic")
    )

    # ── Narrative flow ────────────────────────────────────────────────────────
    render_headline(df, capm_risk_labeled, capm_df, return_col, benchmark_label, portfolio_label)
    render_insight_1(df, benchmark_returns, capm_df, return_col, benchmark_label)
    render_insight_2(capm_df)
    render_insight_3(capm_df)
    render_appendix(df, benchmark_returns, rfr_series, benchmark_label, rfr_label, return_col)


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
            <span style='opacity:0.4;font-weight:400;margin-left:6px;font-size:12px'>· Research View</span>
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
