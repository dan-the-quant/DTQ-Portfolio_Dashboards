# Data Management
import pandas as pd
import numpy as np


# Helper: Add a constant
def add_constant(
        x_matrix: pd.DataFrame,
):
    if isinstance(x_matrix, pd.Series):
        x_matrix = x_matrix.to_frame()

    ones = pd.Series(1, index=x_matrix.index, name="constant")
    x_matrix_with_constant = pd.concat([ones, x_matrix], axis=1)

    return x_matrix_with_constant


# Helper: Residual Calculator
def residuals(
        X,
        Y,
        coef,
):
    return Y - X @ coef


# Helper: Calculate Sigma
def sigma(
        X,
        Y,
        coef,
):
    if X.shape[0] <= X.shape[1]:
        raise ValueError("n must be greater than k to calculate sigma.")

    errors = residuals(X, Y, coef)
    std = np.sqrt(np.sum(errors ** 2, axis=0) / (X.shape[0] - X.shape[1]))

    return std
