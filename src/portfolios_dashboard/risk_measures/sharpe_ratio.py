# Libraries
import pandas as pd
import numpy as np


# Sharpe Ratio
def sharpe_ratio(
        returns,
        risk_free_rate=0.0
):

    # Nan if it receives an empty array
    if isinstance(returns, (float, int)):
        return np.nan

    # Convert to DataFrame
    df = pd.DataFrame(returns)

    # Excess Returns
    if isinstance(risk_free_rate, (pd.Series, pd.DataFrame)):
        # Use subtract function if pandas object
        excess = df.subtract(risk_free_rate, axis=0)
    else:
        # Use operator if float or int
        excess = df - risk_free_rate

    # Sharpe Ratio
    mu = excess.mean()
    sigma = excess.std()
    sharpe = mu / sigma

    return sharpe.squeeze().astype(float)
