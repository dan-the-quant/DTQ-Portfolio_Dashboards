# Libraries

import numpy as np
import pandas as pd
from scipy.stats import norm


# Value at Risk
def value_at_risk(
        returns,
        alpha=0.05,
        method="historical",
        simulations=100000
):
    # Handle scalar inputs
    if isinstance(returns, (float, int)):
        return np.nan

    # Convert to DataFrame for uniform handling
    df = pd.DataFrame(returns)

    # Container for results
    var_values = pd.DataFrame(index=df.columns, columns=["VaR"])

    for col in df.columns:
        r = df[col].dropna().values  # Clean numpy array

        if len(r) == 0:
            var_values.loc[col, "VaR"] = np.nan
            continue

        # === Historical Method ===
        if method == "historical":
            var = np.quantile(r, alpha)

        # === Parametric (Normal) Method ===
        elif method == "parametric":
            mu = np.mean(r)
            sigma = np.std(r, ddof=1)
            z_alpha = norm.ppf(alpha)
            var = mu + sigma * z_alpha

        # === Monte Carlo Simulation ===
        elif method == "montecarlo":
            mu = np.mean(r)
            sigma = np.std(r, ddof=1)
            simulated_returns = np.random.normal(mu, sigma, simulations)
            var = np.quantile(simulated_returns, alpha)

        else:
            raise ValueError("Method not recognized. Use 'historical', 'parametric', or 'montecarlo'.")

        var_values.loc[col, "VaR"] = var

    return var_values.T.squeeze().astype(float)
