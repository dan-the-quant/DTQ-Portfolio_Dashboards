# Libraries
import pandas as pd
import streamlit

# Modules
from src.portfolios_dashboard.data.data_calculations import wexp
from src.portfolios_dashboard.regression.regression_helper import add_constant
from src.portfolios_dashboard.regression.linear_regression_model import linear_regression


# CAPM Coefficients
@streamlit.cache_data
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
    dates = asset_returns.index[window - 1:]

    results = {}

    for date in dates:
        asset_window = asset_returns.loc[:date].iloc[-window:]
        benchmark_window = benchmark_returns.loc[:date].iloc[-window:]
        rfr_window = risk_free_rate.loc[:date].iloc[-window:]

        try:
            coeffs = capm_coefficients(
                asset_returns=asset_window,
                benchmark_returns=benchmark_window,
                risk_free_rate=rfr_window,
                coefficient='all',
                weighted=weighted,
                half_life=half_life,
            )

            for name in coeffs.index:
                if name not in results:
                    results[name] = []
                row = coeffs.loc[name]
                row.name = date
                results[name].append(row)

        except Exception as e:
            print(f"Fail on {date}: {e}")
            continue

    all_coeffs = {name: pd.DataFrame(rows) for name, rows in results.items()}

    if coefficient == 'all':
        return all_coeffs
    elif coefficient in all_coeffs:
        return all_coeffs[coefficient]
    else:
        raise ValueError('coefficient argument must be one of: ["alpha", "beta", "sigma", "all"]')