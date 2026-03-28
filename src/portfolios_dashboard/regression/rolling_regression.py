# Libraries
import pandas as pd
import numpy as np

# Modules
from src.portfolios_dashboard.regression.linear_regression_model import linear_regression


# Rolling Regression
def rolling_least_squares_regression(
        y_matrix: pd.DataFrame,
        x_matrix: pd.DataFrame,
        weights: np.ndarray | None = None,
        window: int = 252,
):
    # Trimmed Returns
    trimmed_y_matrix = y_matrix.iloc[window - 1:]

    # Define the dates
    dates = trimmed_y_matrix.index

    # Coefficients Dictionary — initialized after first regression to match actual index
    coefficients_dict = {}

    # Loop
    for date in dates:

        # Set the windows
        x_window = x_matrix.loc[:date].iloc[-window:]
        y_window = y_matrix.loc[:date].iloc[-window:]

        # Trim weights to match window size
        if weights is not None:
            weights_window = weights[-window:]
        else:
            weights_window = None

        # Select Valid Stocks (those with enough data)
        valid_stocks = y_window.count()[y_window.count() >= window].index
        if len(valid_stocks) < 1:
            continue

        valid_y_window = y_window[valid_stocks]

        try:
            coeffs = linear_regression(valid_y_window, x_window, weights_window)

            for x in coeffs.index:
                if x not in coefficients_dict:
                    coefficients_dict[x] = []

                s = coeffs.loc[x]
                s.name = date
                coefficients_dict[x].append(s)

        except ValueError as e:
            print(f"Fail in {date}: {e}")
            continue

    # Concatenate lists into DataFrames after loop
    return {x: pd.DataFrame(rows) for x, rows in coefficients_dict.items()}