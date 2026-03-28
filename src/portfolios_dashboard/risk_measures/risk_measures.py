# Libraries
import pandas as pd


# Intraday Price Range
def intraday_prices_range(
        high_prices: pd.Series,
        low_prices: pd.Series,
):
    return high_prices - low_prices


# Tracking Error
def tracking_error(
        asset_returns,
        benchmark_returns,
):
    aligned_asset, aligned_benchmark = asset_returns.align(benchmark_returns, join="inner")
    diff = aligned_asset.subtract(benchmark_returns, axis=0)
    return diff.std(skipna=True)


# Information Ratio
def information_ratio(
        asset_returns,
        benchmark_returns,
):
    excess_returns = asset_returns - benchmark_returns
    mu = excess_returns.mean()
    t_error = tracking_error(asset_returns, benchmark_returns)

    return mu / t_error


# Treynor Ratio
def treynor_ratio(
        returns,
        beta,
        risk_free_rate=0.0,
):
    mu = returns.mean()

    return (mu - risk_free_rate) / beta



