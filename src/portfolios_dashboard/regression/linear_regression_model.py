# Data Management
import pandas as pd
import numpy as np

# Modules
from src.portfolios_dashboard.regression.regression_helper import sigma


# Linear Regression Coefficients
def linear_regression(
        y_matrix,
        x_matrix,
        weights=None,
        stds=True,
):
    """
    General OLS/WLS regression using matrix formulation.

    Parameters
    ----------
    y_matrix : pd.DataFrame or pd.Series
        Dependent variable(s).
    x_matrix : pd.DataFrame
        Independent variable(s) (already including constant if desired).
    weights : array-like, optional
        Observation weights. If None, assumes OLS.
    stds : bool, default True
        Whether to compute standard deviation of residuals.

    Returns
    -------
    coef : pd.DataFrame
        Estimated coefficients (and optionally sigma).
    """

    if isinstance(y_matrix, pd.Series):
        y_matrix = y_matrix.to_frame()
    if isinstance(x_matrix, pd.Series):
        x_matrix = x_matrix.to_frame()

    if x_matrix.shape[0] != y_matrix.shape[0]:
        raise ValueError("The number of rows in X and Y must be the same.")

    X = np.asarray(x_matrix)
    Y = np.asarray(y_matrix)
    n = X.shape[0]

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    if weights is None:
        weights = np.ones(n)

    # WLS via element-wise multiplication — avoids building dense n×n matrix
    sqrt_w = np.sqrt(weights).reshape(-1, 1)
    X_w = X * sqrt_w
    Y_w = Y * sqrt_w

    # lstsq is numerically stable even under multicollinearity
    coef, _, _, _ = np.linalg.lstsq(X_w.T @ X_w, X_w.T @ Y_w, rcond=None)

    if stds:
        sigmas = sigma(X, Y, coef)
        coef = np.vstack([coef, sigmas])

    coef = pd.DataFrame(coef, columns=y_matrix.columns)
    coef.index = list(x_matrix.columns) + (['sigma'] if stds else [])

    return coef
