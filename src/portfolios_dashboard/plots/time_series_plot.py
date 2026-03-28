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
        title: str = "Time Series",
        height: int = 500,
):
    if isinstance(data, pd.Series):
        df = data.to_frame(name=data.name or "Value")
    else:
        df = data.copy()

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
        yaxis_title="Value",
        height=height,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.7)",
            borderwidth=0,
        )
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
