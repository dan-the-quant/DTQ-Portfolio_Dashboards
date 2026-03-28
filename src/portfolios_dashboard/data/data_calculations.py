# Import Data Management
import pandas as pd
import numpy as np


# Calculate Logarithmic Returns
def log_returns(
        price_series: pd.Series
):
    return np.log(price_series / price_series.shift(1))


# Create the Weights Function
def wexp(N: int, half_life: float):
    if half_life <= 0:
        raise ValueError("half_life must be > 0")

    c = np.log(0.5) / half_life
    n = np.arange(N)
    w = np.exp(c * n)
    return np.flip(w / np.sum(w))


# Helper: exclude tiny returns
def n_days_nonmiss(
        returns,
        tiny_ret=1e-6
):
    ix_ret_tiny = np.abs(returns) <= tiny_ret
    return np.sum(~ix_ret_tiny, axis=0)


# Relative Strength Calculation
def calc_rstr(
        returns,
        half_life=126,
        min_obs=100,
        yolo=True,
):
    # If returns are not log, we make them log
    if not yolo:
        rstr = np.log(1. + returns)
    else:
        rstr = returns

    # Calculate Weights
    if half_life == 0:
        weights = np.ones_like(rstr)
    else:
        weights = len(returns) * wexp(len(returns), half_life).reshape(-1, 1)

    rstr = (rstr * weights).sum()
    idx = n_days_nonmiss(returns) < min_obs
    rstr = rstr.where(~idx, other=np.nan)
    df = pd.Series(rstr)
    df.name = returns.index[-1]
    return df


# Rolling Relative Strength
def rolling_calc_rstr(
        returns,
        window_size=252,
        half_life=126,
        min_obs=100
):
    # Vectorizing with rolling + apply
    def rstr_window(window):
        window_df = returns.iloc[int(window.index[0]):int(window.index[0]) + window_size]
        return calc_rstr(
            returns=window_df,
            half_life=half_life,
            min_obs=min_obs
        )

    rolling_results = []
    range_to_iter = range(len(returns) - window_size + 1)
    for i in range_to_iter:
        window_returns = returns.iloc[i:i + window_size]
        rs_i = calc_rstr(
            returns=window_returns,
            half_life=half_life,
            min_obs=min_obs
        )
        rolling_results.append(rs_i)

    return pd.concat(rolling_results, axis=1)
