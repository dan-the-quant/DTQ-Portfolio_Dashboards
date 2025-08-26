# Libraries

# Data
import pandas as pd
import datetime

# Streamlit
import streamlit as st
from streamlit_option_menu import option_menu

# Modules
from src.portfolios_dashboard.download_data import import_yf_financial_data
from src.portfolios_dashboard.plots import TimeSeriesPlot
from src.portfolios_dashboard.plots import CandlesticksPlot

# Set the website layouts
st.set_page_config(
    page_title="DTQ Portfolio Dashboard",
    layout="wide",
)

# Set the sidebar content
with st.sidebar:
    st.header('Portfolios Dashboard')

    selected = option_menu(
        None,
        ["Create a Portfolio", "Track a Portofolio"],
        icons=['house', "cash"],
        menu_icon="cast",
        default_index=0
    )

    st.subheader('A Work by Dan the Quant')

    st.text('Author: Edgar Alcántara')

    st.link_button(
        "Daniel Barrera",
        "https://www.linkedin.com/in/danielrbarrera/"
    )

    st.link_button(
        "Edgar Alcántara",
        "https://www.linkedin.com/in/edgarallo0/"
    )

if selected == 'Create a Portfolio':

    # ------------------------------
    # CREATE A PORTFOLIO DASHBOARD
    # ------------------------------

    st.title("Create your own Portfolio from Yahoo Finance")

    # Load available tickers
    tickers = pd.read_csv(r'Inputs\available_stocks.csv')
    options = tickers["available_stocks"].tolist()
    selection = st.multiselect("Select the Tickers you want", options)

    if selection:
        # ----------#
        # Load data #
        # ----------#
        data_close = pd.DataFrame()
        data_returns = pd.DataFrame()
        data_dict = {}

        for s in selection:
            df = import_yf_financial_data(
                ticker=s,
                start_date='2010-01-01',
                end_date='2026-01-01',
                returns=True
            )

            df.index = pd.to_datetime(df.index)

            # Store in the dictionary
            data_dict[s] = df

            # Store in DataFrames
            data_close[s] = df['close']
            data_returns[s] = df['returns']

        # ------------------------------#
        # Show raw data and basic stats #
        # ------------------------------#
        col_right, buff, col_left = st.columns([4.5, 1, 4.5])

        with col_right:
            tab1, tab2 = st.tabs(["Close Price", "Returns"])
            with tab1:
                st.dataframe(data_close, height=400)
            with tab2:
                st.dataframe(data_returns, height=400)

            st.write(data_returns.describe())

        # ------------------------------
        # Time Series Plots
        # ------------------------------
        with col_left:
            # Time Series
            fig1 = TimeSeriesPlot(data_returns.cumsum(), 'Cumulative Returns')
            st.plotly_chart(fig1, use_container_width=True)

            # Selector
            selected_ticker = st.selectbox(
                "Choose a ticker:",
                data_close.columns
            )

            # Candles
            fig2 = CandlesticksPlot(data_dict[selected_ticker].iloc[-252:], f'Last-Year {selected_ticker} Candle Sticks')
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("Portfolio Builder")

        # ------------------------------
        # Portfolio Inputs
        # ------------------------------
        col1, buff, col2 = st.columns([4.5, 1, 4.5])

        with col1:
            # Date input limitado al rango de datos
            min_date = data_close.index.min().date()
            max_date = data_close.index.max().date()
            date = st.date_input("Select the starting date", datetime.date.today(), min_value=min_date,
                                 max_value=max_date)

            # Budget
            budget = st.number_input("Set your Budget", min_value=0, value=1000, step=1)

        # Current prices
        current_price = data_close.loc[pd.Timestamp(date)].rename('Close Price')
        with col2:
            st.subheader("Current Prices")
            st.dataframe(current_price)

        # ------------------------------
        # Stock quantities input
        # ------------------------------
        qs = {}
        with col1:
            st.subheader("Number of Stocks")
            for stock in data_close.columns:
                q = st.number_input(
                    f"{stock}:", min_value=0,
                    value=1,
                    step=1
                )
                qs[stock] = q

        # --------------------#
        # Calculate portfolio #
        # --------------------#
        def calculate_portfolio(qs, prices, budget):
            qs_series = pd.Series(qs, name='Number')
            composition = qs_series * prices
            composition.name = 'Composition'
            weights = composition / budget
            weights.name = 'Weights'
            total_value = composition.sum()
            leftover = budget - total_value
            return qs_series, composition, weights, total_value, leftover


        qs_series, composition, weights, total_value, leftover = calculate_portfolio(qs, current_price, budget)

        # ---------------------------#
        # Display metrics and tables #
        # ---------------------------#

        with col2:
            st.subheader("Portfolio Composition")
            st.dataframe(composition)
            st.subheader("Weights")
            st.dataframe(weights)

        with col1:
            st.subheader("Summary Metrics")
            st.metric("Portfolio Value", total_value)
            st.metric("Leftover Budget", leftover)
            st.metric("Sum of Weights", weights.sum())

        # ----------------#
        # Track Portfolio #
        # ----------------#

        st.divider()
        st.subheader("Portfolio Tracking")

        col_3, col_4 = st.columns([1, 1])

        with col_3:
            portfolio = weights * data_returns.loc[pd.Timestamp(date):]
            portfolio = portfolio.sum(axis=1)
            portfolio.name = 'Portfolio'

            st.dataframe(portfolio, height=400)

        with col_4:
            fig3 = TimeSeriesPlot(portfolio.cumsum(), 'Portfolio Cumulative Returns')
            st.plotly_chart(fig3, use_container_width=True)