# Import Data Management
import pandas as pd
import numpy as np

# Import Providers Libraries
import yfinance as yf

# Streamlit
import streamlit as st


# Calculate Logarithmic Returns
@st.cache_data
def log_returns(
        price_series: pd.Series
):
    return np.log(price_series / price_series.shift(1))


# Function to import data
@st.cache_data
def import_prices_data(
        tickers: str | list,
        start_date: str = '1999-01-01',
        end_date: str = '2025-01-01',
        price: str = 'Close'
):
    # Get the Data from Yahoo Finance
    data = yf.download(
        tickers,  # Stock to import
        start=start_date,  # First Date
        end=end_date,  # Last Date
        interval='1d',  # Daily Basis
        auto_adjust=True,  # Adjusted Prices,
        progress=False  # Not printing
    )

    # Get Price Data
    price_data = data.loc[:, price]

    return price_data
