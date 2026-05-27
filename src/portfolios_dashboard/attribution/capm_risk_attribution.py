# Libraries
import pandas as pd
import numpy as np
import streamlit as st

# Modules
from src.portfolios_dashboard.regression.capital_asset_pricing_model import capm_coefficients


# CAPM Attribution
@st.cache_data
def capm_risk_attribution(asset_returns, benchmark_returns, risk_free_rate):
    # Align first, before scaling
    common_index = (
        asset_returns.dropna().index
        .intersection(benchmark_returns.dropna().index)
        .intersection(risk_free_rate.dropna().index)
    )

    if len(common_index) <= 2:
        raise ValueError(
            f"Only {len(common_index)} aligned observations between portfolio, "
            f"benchmark and RFR. Check that date ranges overlap."
        )

    asset_returns     = asset_returns.loc[common_index] * 100
    benchmark_returns = benchmark_returns.loc[common_index] * 100
    risk_free_rate    = risk_free_rate.loc[common_index] * 100

    beta = capm_coefficients(
        asset_returns, benchmark_returns, risk_free_rate,
        coefficient='beta', weighted=False
    ).iloc[0]

    total_variance      = asset_returns.var()
    market_variance     = benchmark_returns.var()
    systematic_variance = (beta ** 2) * market_variance
    idio_variance       = total_variance - systematic_variance

    result_df = pd.DataFrame({
        'variance':   [total_variance, systematic_variance, idio_variance],
        'percentage': [1.0, systematic_variance / total_variance, idio_variance / total_variance]
    }, index=['Total', 'Systematic', 'Idiosyncratic'])

    return result_df
