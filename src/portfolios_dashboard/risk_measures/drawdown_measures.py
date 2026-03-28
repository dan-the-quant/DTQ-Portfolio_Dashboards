# Libraries

# Data
import pandas as pd
import numpy as np


# Helper: Drawdowns
def drawdowns(
        returns,
        log=True
):
    # --- Compute cumulative returns ---
    if log:
        cumulative = np.exp(returns.cumsum())
    else:
        cumulative = (1 + returns).cumprod()

    # --- Compute drawdowns ---
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1

    return drawdown


# Maximum Drawdown
def max_drawdown(
        returns, 
        log=True
):
    # Handle scalar input (rolling)
    if isinstance(returns, (float, int)):
        return np.nan

    # Convert to DataFrame for uniform processing
    df = pd.DataFrame(returns)

    mdd_values = pd.Series(index=df.columns, dtype=float)

    for col in df.columns:
        r = df[col].dropna()

        if len(r) == 0:
            mdd_values[col] = np.nan
            continue

        drawdown = drawdowns(r, log=log)

        # --- Maximum drawdown ---
        mdd_values[col] = drawdown.min()

    return mdd_values.squeeze().astype(float)


# Conditional Expected Drawdown
def conditional_expected_drawdown(
        returns,
        alpha=0.05,
        log=True,
):
    # Handle scalar input (rolling)
    if isinstance(returns, (float, int)):
        return np.nan

    # Convert to DataFrame for uniform processing
    df = pd.DataFrame(returns)
    ced_values = pd.Series(index=df.columns, dtype=float)

    for col in df.columns:
        r = df[col].dropna()

        if len(r) == 0:
            ced_values[col] = np.nan
            continue

        drawdown = drawdowns(r, log=log)

        # --- Conditional Expected Drawdown ---
        threshold = np.quantile(drawdown, alpha)
        ced = drawdown[drawdown <= threshold].mean()

        ced_values[col] = ced

    return ced_values.squeeze().astype(float)
