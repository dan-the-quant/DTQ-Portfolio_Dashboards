import numpy as np
import pandas as pd


# Helper: Calculate Semivariance
def semivariance(
        returns,
        threshold=0.0
):
    # Handle scalar input (rolling)
    if isinstance(returns, (float, int)):
        return np.nan

    # Convert to DataFrame for uniform processing
    df = pd.DataFrame(returns)
    semi_values = pd.Series(index=df.columns, dtype=float)

    for col in df.columns:
        r = df[col].dropna()
        downside = r[r <= threshold]

        if len(downside) == 0:
            semi_values[col] = 0.0
        else:
            semi_values[col] = downside.var(ddof=1)

    return semi_values.squeeze().astype(float)


# Sortino Ratio
def sortino_ratio(
        returns,
        risk_free_rate=0.0,
        threshold=0.0
):
    # Handle scalar input (rolling)
    if isinstance(returns, (float, int)):
        return np.nan

    # Convert to DataFrame for uniform processing
    df = pd.DataFrame(returns)

    # DataFrame
    sortino_values = pd.Series(index=df.columns, dtype=float)

    for col in df.columns:
        r = df[col].dropna()
        mu = r.mean()
        semi_sigma = np.sqrt(semivariance(r, threshold))

        # Avoid division by zero
        if semi_sigma == 0:
            sortino_values[col] = np.nan
        else:
            sortino_values[col] = (mu - risk_free_rate) / semi_sigma

    return sortino_values.squeeze().astype(float)
