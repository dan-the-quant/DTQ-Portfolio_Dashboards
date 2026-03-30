# Libraries
import pandas as pd
import numpy as np
import streamlit

# Modules
from src.portfolios_dashboard.data.data_calculations import wexp
from src.portfolios_dashboard.regression.regression_helper import add_constant
from src.portfolios_dashboard.regression.linear_regression_model import linear_regression


# CAPM Coefficients
def capm_coefficients(
        asset_returns,
        benchmark_returns,
        risk_free_rate,
        coefficient='all',
        weighted=True,
        half_life=None,
):
    # Common Index
    common_index = asset_returns.dropna().index.intersection(benchmark_returns.dropna().index)
    common_index = common_index.intersection(risk_free_rate.dropna().index)

    # Components
    asset = asset_returns.loc[common_index]
    benchmark = benchmark_returns.loc[common_index]
    rfr = risk_free_rate.loc[common_index]
    n = len(asset)

    # Inputs
    y = asset.subtract(rfr, axis=0)
    x = benchmark - rfr
    x = add_constant(x)
    x.columns = ['alpha', 'beta']

    # Weights
    if weighted:
        hl = half_life if half_life is not None else n / 2
        w = n * wexp(n, hl)
    else:
        w = None

    # Regression
    coefficients = linear_regression(
        y_matrix=y,
        x_matrix=x,
        weights=w
    )

    # Extract Coefficient
    if coefficient == 'alpha':
        return coefficients.iloc[0]
    elif coefficient == 'beta':
        return coefficients.iloc[1]
    elif coefficient == 'sigma':
        return coefficients.iloc[2]
    elif coefficient == 'all':
        return coefficients
    else:
        raise ValueError('coefficient argument must be one of: ["alpha", "beta", "sigma", "all"]')


@streamlit.cache_data
# Rolling CAPM Coefficients
def rolling_capm_coefficients(
        asset_returns,
        benchmark_returns,
        risk_free_rate,
        window: int = 252,
        coefficient='all',
        weighted=True,
        half_life=None,
):
    # Common index
    common_index = (
        asset_returns.dropna().index
        .intersection(benchmark_returns.dropna().index)
        .intersection(risk_free_rate.dropna().index)
    )

    asset     = asset_returns.loc[common_index].values
    benchmark = benchmark_returns.loc[common_index].values
    rfr       = risk_free_rate.loc[common_index].values
    dates     = common_index[window - 1:]
    n_windows = len(dates)
    n         = len(common_index)

    # Weights
    if weighted:
        hl = half_life if half_life is not None else window / 2
        w = window * wexp(window, hl)  # shape (window,)
        sqrt_w = np.sqrt(w)
    else:
        sqrt_w = np.ones(window)

    alphas = np.full(n_windows, np.nan)
    betas  = np.full(n_windows, np.nan)
    sigmas = np.full(n_windows, np.nan)

    for i in range(n_windows):
        a  = asset    [i: i + window]
        bm = benchmark[i: i + window]
        rf = rfr      [i: i + window]

        y = (a - rf).reshape(-1, 1)
        x = np.column_stack([np.ones(window), bm - rf])

        # Apply weights
        x_w = x * sqrt_w[:, None]
        y_w = y * sqrt_w[:, None]

        coef, _, _, _ = np.linalg.lstsq(x_w.T @ x_w, x_w.T @ y_w, rcond=None)

        alphas[i] = coef[0, 0]
        betas [i] = coef[1, 0]

        resid      = y - x @ coef
        sigmas[i]  = np.std(resid)

    result = pd.DataFrame(
        {"alpha": alphas, "beta": betas, "sigma": sigmas},
        index=dates,
    )

    if coefficient == 'all':
        return {"alpha": result[["alpha"]], "beta": result[["beta"]], "sigma": result[["sigma"]]}
    elif coefficient in result.columns:
        return result[[coefficient]]
    else:
        raise ValueError('coefficient must be one of: ["alpha", "beta", "sigma", "all"]')