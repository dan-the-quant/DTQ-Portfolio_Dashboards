# Libraries
import pandas as pd
import numpy as np
import streamlit as st

# Modules
from src.portfolios_dashboard.regression.capital_asset_pricing_model import capm_coefficients


@st.cache_data
# Risk Attribution Function
def capm_risk_attribution(
        asset_returns,
        benchmark_returns,
        risk_free_rate,
):
    # Rescaling Variances
    asset_returns = asset_returns * 100
    benchmark_returns = benchmark_returns * 100
    risk_free_rate = risk_free_rate * 100

    # Get beta from CAPM
    beta = capm_coefficients(
        asset_returns, benchmark_returns, risk_free_rate, coefficient='beta', weighted=False
    ).iloc[0]

    # Variance components
    total_variance = asset_returns.var()
    market_variance = benchmark_returns.var()
    systematic_variance = (beta ** 2) * market_variance
    idio_variance = total_variance - systematic_variance

    result_df = pd.DataFrame({
        'variance': [total_variance, systematic_variance, idio_variance],
        'percentage': [1.000, systematic_variance / total_variance, idio_variance / total_variance]
    }, index=['Total', 'Systematic', 'Idiosyncratic'])

    return result_df
