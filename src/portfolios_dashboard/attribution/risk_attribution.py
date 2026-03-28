# Libraries

# Data
import pandas as pd


# Helper: Sigmas
def sigmas(
        holdings_returns,
):
    return holdings_returns.std()


def rhos(
        holdings_returns,
        portfolio_returns,
):
    return holdings_returns.corrwith(portfolio_returns)


# X-Sigma-Rho Attribution
def x_sigma_rho(
        holdings_returns,
        holdings_weights
):
    # Weights
    x = holdings_weights

    # Portfolio
    portfolio_returns = holdings_returns @ x

    # Sigmas
    sigma = sigmas(holdings_returns)

    # Rhos
    rho = rhos(holdings_returns, portfolio_returns)

    # Risk Contribution
    risk_contribution = x * sigma * rho

    # Relative Risk Contribution
    rrc = risk_contribution / portfolio_returns.std()

    return rrc * 100


# Marginal Risk Contribution
def marginal_risk_contribution(
        holdings_returns,
        holdings_weights,
):
    x = holdings_weights
    portfolio_returns = holdings_returns @ x
    sigma = sigmas(holdings_returns)
    rho = rhos(holdings_returns, portfolio_returns)

    return sigma * rho


# Correlation Drilldown
def correlation_drilldown(
        holdings_returns,
        holdings_weights,
):
    x = holdings_weights
    portfolio_returns = holdings_returns @ x
    sigma = sigmas(holdings_returns)
    rho = rhos(holdings_returns, portfolio_returns)
    sigma_P = portfolio_returns.std()

    # Let us check the drill-downs
    decomposition_matrix = pd.DataFrame(
        index=holdings_returns.columns,
        columns=holdings_returns.columns
    )

    # Loop
    for m in holdings_returns.columns:  # Stock m (rows)
        for n in holdings_returns.columns:  # Stock n (columns)

            term = x.loc[n] * (sigma.loc[n] / sigma_P) * rho.loc[m, n]

            decomposition_matrix.loc[m, n] = term

    return decomposition_matrix
