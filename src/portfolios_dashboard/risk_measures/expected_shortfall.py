# Libraries
import numpy as np
import pandas as pd
from scipy.stats import norm

# Import VaR function
from src.portfolios_dashboard.risk_measures.value_at_risk import value_at_risk

# Expected Shortfall Function
def expected_shortfall(
    returns,
    alpha=0.05,
    method="historical",
    simulations=100000
):

    # Handle scalar input (rolling)
    if isinstance(returns, (float, int)):
        return np.nan

    # Convert to DataFrame for uniform processing
    df = pd.DataFrame(returns)

    es_values = pd.DataFrame(index=df.columns, columns=["ES"])

    for col in df.columns:
        r = df[col].dropna().values  # clean numpy array

        if len(r) == 0:
            es_values.loc[col, "ES"] = np.nan
            continue

        # Compute VaR first
        var = value_at_risk(r, alpha=alpha, method=method, simulations=simulations)

        # --- Compute Expected Shortfall based on method ---
        if method == "historical":
            es = r[r <= var].mean()
        elif method == "parametric":
            mu = np.mean(r)
            sigma = np.std(r, ddof=1)
            z = norm.ppf(alpha)
            es = mu - (sigma * norm.pdf(z) / alpha)
        elif method == "montecarlo":
            mu = np.mean(r)
            sigma = np.std(r, ddof=1)
            simulated_returns = np.random.normal(mu, sigma, simulations)
            es = simulated_returns[simulated_returns <= var].mean()
        else:
            raise ValueError("Method must be 'historical', 'parametric', or 'montecarlo'.")

        es_values.loc[col, "ES"] = es

    return es_values.T.squeeze().astype(float)
