# StreamLit
import streamlit as st

# Data
import pandas as pd

# Plotly
import plotly.graph_objects as go

# DTQ Color palette
DTQ_COLORS = [
    "#111440",  # Navy — primary (Portfolio)
    "#8a9bc4",  # Steel blue — secondary (Market Factor)
    "#c0392b",  # Red — tertiary (Alpha/Residual)
    "#1a7a4a",  # Green
    "#c07000",  # Amber
]

DTQ_LAYOUT = dict(
    font=dict(family="Montserrat, sans-serif", color="#111440"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#f8f9fc",
    xaxis=dict(
        showgrid=False,
        showline=True,
        linecolor="#e5e7ef",
        tickfont=dict(size=11, color="#aaa"),
        title_font=dict(size=11, color="#aaa"),
    ),
    yaxis=dict(
        gridcolor="#e5e7ef",
        gridwidth=0.5,
        showline=False,
        tickfont=dict(size=11, color="#aaa"),
        title_font=dict(size=11, color="#aaa"),
        zeroline=True,
        zerolinecolor="#d0d3df",
        zerolinewidth=1,
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        font=dict(size=11, color="#888"),
        orientation="h",
        yanchor="bottom",
        y=-0.25,
        xanchor="left",
        x=0,
    ),
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(
        bgcolor="#111440",
        font_color="#fff",
        font_size=12,
        bordercolor="#111440",
    ),
)


@st.cache_resource
def TimeSeriesPlot(
        data,
        title: str = "Time Series",
        height: int = 350,
):
    if isinstance(data, pd.Series):
        df = data.to_frame(name=data.name or "Value")
    else:
        df = data.copy()

    fig = go.Figure()

    for i, col in enumerate(df.columns):
        color = DTQ_COLORS[i % len(DTQ_COLORS)]
        is_alpha = "alpha" in col.lower() or "residual" in col.lower()

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=col,
                line=dict(
                    color=color,
                    width=1.8,
                    dash="dot" if is_alpha else "solid",
                ),
                hovertemplate=f"<b>{col}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_layout(
        **DTQ_LAYOUT,
        title=dict(
            text=title,
            font=dict(size=13, color="#111440", family="Montserrat, sans-serif"),
            x=0,
            xanchor="left",
        ),
        height=height,
    )

    # COVID-19 shaded band — only if the data covers that period
    covid_start = pd.Timestamp("2020-02-19")
    covid_end   = pd.Timestamp("2020-06-30")
    if df.index.min() <= covid_end and df.index.max() >= covid_start:
        fig.add_vrect(
            x0=covid_start, x1=covid_end,
            fillcolor="rgba(180, 83, 9, 0.08)",
            layer="below", line_width=0,
            annotation_text="COVID-19",
            annotation_position="top left",
            annotation_font=dict(size=9, color="#b45309"),
        )

    return fig
