"""Strategy weight table helpers."""

from __future__ import annotations

import pandas as pd


def align_weights(prices: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Align strategy weights to the price index and validate."""
    aligned = weights.reindex(prices.index).ffill()

    if aligned.isna().any().any():
        missing = aligned.columns[aligned.isna().any()].tolist()
        raise ValueError(f"Weights missing for assets after alignment: {missing}")

    if (aligned < 0).any().any():
        raise ValueError("Weights must be non-negative.")

    row_sums = aligned.sum(axis=1)
    if (row_sums > 1.0 + 1e-9).any():
        bad = row_sums[row_sums > 1.0 + 1e-9]
        raise ValueError(f"Row weights exceed 1.0 on dates: {bad.index.tolist()}")

    return aligned


def constant_weights(
    prices: pd.DataFrame,
    weights_dict: dict[str, float],
) -> pd.DataFrame:
    """Build a strategy with fixed target weights at every date."""
    _validate_weights_dict(weights_dict, prices.columns)
    full_weights = {asset: weights_dict.get(asset, 0.0) for asset in prices.columns}
    data = {asset: [weight] * len(prices) for asset, weight in full_weights.items()}
    weights = pd.DataFrame(data, index=prices.index)
    return align_weights(prices, weights)


def periodic_weights(
    prices: pd.DataFrame,
    weights_dict: dict[str, float],
    freq: str = "Y",
) -> pd.DataFrame:
    """Build target weights that rebalance only at the chosen cadence.

    Between rebalance dates, weights drift (no target row) so the backtester
    does not trade every period.

    freq: pandas offset alias, e.g. "M" (month end), "Q" (quarter end),
        "Y" (year end). Legacy aliases like "ME"/"YE" are also accepted.
    """
    _validate_weights_dict(weights_dict, prices.columns)

    period_freq = _to_period_freq(freq)
    periods = prices.index.to_period(period_freq)
    rebalance_dates = prices.groupby(periods, observed=False).tail(1).index
    if len(rebalance_dates) == 0:
        rebalance_dates = prices.index[[0]]

    full_weights = {asset: weights_dict.get(asset, 0.0) for asset in prices.columns}
    data = {asset: [weight] * len(rebalance_dates) for asset, weight in full_weights.items()}
    weights = pd.DataFrame(data, index=rebalance_dates)
    weights = weights.reindex(prices.index)
    _validate_weight_rows(weights)
    return weights


def buy_and_hold_weights(
    prices: pd.DataFrame,
    weights_dict: dict[str, float],
) -> pd.DataFrame:
    """Build weights for a single initial allocation with no rebalancing."""
    _validate_weights_dict(weights_dict, prices.columns)
    full_weights = {asset: weights_dict.get(asset, 0.0) for asset in prices.columns}
    data = {asset: [weight] for asset, weight in full_weights.items()}
    weights = pd.DataFrame(data, index=prices.index[[0]])
    weights = weights.reindex(prices.index)
    _validate_weight_rows(weights)
    return weights


def _validate_weight_rows(weights: pd.DataFrame) -> None:
    defined = weights.dropna(how="all")
    if defined.empty:
        raise ValueError("At least one rebalance date must have weights.")

    if (defined < 0).any().any():
        raise ValueError("Weights must be non-negative.")

    row_sums = defined.sum(axis=1)
    if (row_sums > 1.0 + 1e-9).any():
        bad = row_sums[row_sums > 1.0 + 1e-9]
        raise ValueError(f"Row weights exceed 1.0 on dates: {bad.index.tolist()}")


def _validate_weights_dict(weights_dict: dict[str, float], price_columns: pd.Index) -> None:
    if not weights_dict:
        raise ValueError("weights_dict must not be empty.")

    unknown = set(weights_dict) - set(price_columns)
    if unknown:
        raise ValueError(f"Unknown assets in weights_dict: {unknown}")

    if any(w < 0 for w in weights_dict.values()):
        raise ValueError("Weights must be non-negative.")

    if sum(weights_dict.values()) > 1.0 + 1e-9:
        raise ValueError("Weights must sum to at most 1.0.")


def _to_period_freq(freq: str) -> str:
    """Map pandas timestamp freq aliases to period-compatible aliases."""
    mapping = {
        "ME": "M",
        "QE": "Q",
        "YE": "Y",
        "YS": "Y",
        "AS": "Y",
    }
    return mapping.get(freq, freq)
