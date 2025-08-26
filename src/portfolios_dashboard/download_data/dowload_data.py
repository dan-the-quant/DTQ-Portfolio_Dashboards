# Import Data Management
import pandas as pd
import numpy as np

# Import Providers Libraries
import yfinance as yf

# Streamlit
import streamlit as st

# Calculate Logarithmic Returns
def log_returns(
        price_series: pd.Series
):
    return np.log(price_series / price_series.shift(1))


@st.cache_data
# Function to import data
def import_yf_financial_data(
        ticker: str,
        start_date: str = '2018-01-01',
        end_date: str = '2025-01-01',
        returns: bool = False,
):
    # Get the Data from Yahoo Finance
    data = yf.download(
        ticker,                 # Stock to import
        start=start_date,       # First Date
        end=end_date,           # Last Date
        interval='1d',          # Daily Basis
        auto_adjust=True,       # Adjusted Prices,
        progress=False          # Not printing
    )

    # Flat columns
    data.columns = data.columns.get_level_values(0)
    data.columns = data.columns.str.lower()

    if returns:
        data['returns'] = log_returns(data['close']).copy()

    return data
