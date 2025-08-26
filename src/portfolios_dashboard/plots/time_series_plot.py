# StreamLit
import streamlit as st

# Data
import pandas as pd

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@st.cache_resource
def TimeSeriesPlot(
        data,
        title: str = "Time Series"
):
    if isinstance(data, pd.Series):
        df = data.to_frame(name=data.name or "Value")
    else:
        df = data.copy()

    # Create Plot
    fig = go.Figure()

    for col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=col
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value"
    )

    fig.update_layout(
        height=400
    )

    return fig


@st.cache_resource
def CandlesticksPlot(
        df: pd.DataFrame,
        title: str = "Candle Sticks"
):
    fig = make_subplots()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Candles'
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        xaxis=dict(
            rangeslider=dict(visible=False)
        )
    )

    fig.update_layout(
        height=400
    )

    return fig
