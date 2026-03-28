# Libraries
import pandas as pd
import streamlit as st


# Compute the Residual Returns
def compute_residual_returns(
        stock_excess_returns,
        factor_returns,
        betas,
):
    contribution = factor_returns.multiply(betas, axis=0)
    aligned_returns = stock_excess_returns.reindex(contribution.index)
    return aligned_returns - contribution.sum(axis=1)


# Factor Contribution Function
@st.cache_data
def factor_contribution(
        portfolio_returns,
        factor_returns,
        betas,
):
    # If betas is a Series with a datetime index → time-varying, single factor
    # If betas is a Series with a string index → static, multi-factor
    # If betas is a DataFrame → time-varying, multi-factor

    # Align portfolio and factor returns
    df = pd.concat([portfolio_returns, factor_returns], axis=1, join='inner').dropna()
    r_i = df.iloc[:, 0]
    factors = df.iloc[:, 1:]

    # Normalize betas to a DataFrame aligned by date
    if isinstance(betas, pd.Series):
        if pd.api.types.is_datetime64_any_dtype(betas.index):
            # Time-varying single factor: Series indexed by date
            betas_df = betas.reindex(factors.index).to_frame()
            betas_df.columns = factors.columns
        else:
            # Static multi-factor: Series indexed by factor name
            betas_df = pd.DataFrame(
                [betas.reindex(factors.columns)] * len(factors),
                index=factors.index,
                columns=factors.columns,
            )
    elif isinstance(betas, pd.DataFrame):
        # Time-varying multi-factor
        betas_df = betas.reindex(factors.index)
    else:
        raise ValueError("betas must be a pd.Series or pd.DataFrame.")

    # Validate columns match
    if set(factors.columns) != set(betas_df.columns):
        raise ValueError(
            f"Factor columns {list(factors.columns)} do not match "
            f"beta columns {list(betas_df.columns)}."
        )

    # Compute factor contributions
    contributions = factors.multiply(betas_df, axis=1)
    contributions.columns = [f"{c}_factor" for c in contributions.columns]

    # Compute residuals
    contributions['residual'] = compute_residual_returns(
        stock_excess_returns=r_i,
        factor_returns=factors,
        betas=betas_df,
    )

    return contributions
