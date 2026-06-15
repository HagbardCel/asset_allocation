"""Strategy weight table helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
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
    else:
        rebalance_dates = rebalance_dates.union(prices.index[[0]]).sort_values()

    full_weights = {asset: weights_dict.get(asset, 0.0) for asset in prices.columns}
    data = {asset: [weight] * len(rebalance_dates) for asset, weight in full_weights.items()}
    weights = pd.DataFrame(data, index=rebalance_dates)
    weights = weights.reindex(prices.index)
    _validate_weight_rows(weights)
    return weights


def momentum_weights(
    prices: pd.DataFrame,
    *,
    lookbacks: Sequence[int] = (12,),
    vol_scaled: bool = False,
    vol_window: int = 36,
    allocation: str = "winner",
    threshold: float = 0.0,
    absolute: bool = False,
    skip: int = 1,
    rebalance: str = "M",
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Build target weights from trailing momentum scores.

    Computes per-asset momentum scores (optionally volatility-scaled in the
    spirit of MSCI World Momentum), ranks assets, and emits rebalance rows
    only when the momentum-gap band is breached.

    Parameters
    ----------
    lookbacks
        Trailing return windows in months. Multiple windows are averaged.
    vol_scaled
        If True, divide each lookback return by annualized trailing volatility.
    vol_window
        Months of monthly returns used for the volatility estimate.
    allocation
        ``"winner"`` allocates 100% to the top-scoring asset;
        ``"relative"`` allocates in proportion to positive scores.
    threshold
        In ``"winner"`` mode, only switch when the desired top asset's score
        exceeds the currently held asset's score by more than this amount.
        In ``"relative"`` mode, rebalance when the L1 distance between desired
        and current target weights exceeds this threshold.
    absolute
        If True, move to cash when the relevant momentum score(s) are <= 0.
    skip
        Months skipped at the recent end of the return window (default 1 for 12-1
        momentum; avoids look-ahead when trading at month-end).
    rebalance
        Rebalance cadence (pandas offset alias, e.g. ``"M"``, ``"Q"``, ``"Y"``).
    risk_free_rate
        Annual risk-free rate subtracted from monthly returns when vol-scaling.
    """
    lookback_tuple = tuple(lookbacks)
    _validate_momentum_params(
        lookback_tuple, vol_window, allocation, threshold, skip, vol_scaled
    )

    scores = _momentum_scores(
        prices,
        lookbacks=lookback_tuple,
        vol_scaled=vol_scaled,
        vol_window=vol_window,
        skip=skip,
        risk_free_rate=risk_free_rate,
    )

    warmup = max(lookback_tuple) + skip
    if vol_scaled:
        warmup = max(warmup, vol_window)

    if warmup >= len(prices):
        raise ValueError(
            f"Not enough price history: need more than {warmup} rows, got {len(prices)}."
        )

    rebalance_dates = _rebalance_dates(prices, rebalance)
    rebalance_dates = rebalance_dates[rebalance_dates >= prices.index[warmup]]

    assets = list(prices.columns)
    weight_rows: list[dict[str, float]] = []
    row_dates: list[pd.Timestamp] = []

    held_asset: str | None = None
    current_target: pd.Series | None = None

    for date in rebalance_dates:
        row_scores = scores.loc[date]
        if row_scores.isna().any():
            continue

        desired = _weights_from_scores(row_scores, allocation, absolute)
        desired_top = _top_asset(desired)

        if current_target is None:
            weight_rows.append(desired.to_dict())
            row_dates.append(date)
            current_target = desired
            held_asset = desired_top
            continue

        if desired_top is None:
            if held_asset is not None:
                weight_rows.append(desired.to_dict())
                row_dates.append(date)
                current_target = desired
                held_asset = None
            continue

        if allocation == "relative":
            weight_distance = float((desired - current_target).abs().sum())
            if weight_distance > threshold:
                weight_rows.append(desired.to_dict())
                row_dates.append(date)
                current_target = desired
                held_asset = desired_top
            continue

        if desired_top == held_asset:
            continue

        gap = float(row_scores[desired_top] - row_scores[held_asset])
        if gap > threshold:
            weight_rows.append(desired.to_dict())
            row_dates.append(date)
            current_target = desired
            held_asset = desired_top

    weights = pd.DataFrame(weight_rows, index=row_dates, columns=assets)
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


def _rebalance_dates(prices: pd.DataFrame, freq: str) -> pd.DatetimeIndex:
    period_freq = _to_period_freq(freq)
    periods = prices.index.to_period(period_freq)
    rebalance_dates = prices.groupby(periods, observed=False).tail(1).index
    if len(rebalance_dates) == 0:
        rebalance_dates = prices.index[[0]]
    return rebalance_dates


def _momentum_scores(
    prices: pd.DataFrame,
    *,
    lookbacks: tuple[int, ...],
    vol_scaled: bool,
    vol_window: int,
    skip: int,
    risk_free_rate: float,
) -> pd.DataFrame:
    """Per-asset momentum scores averaged across lookback windows."""
    score_frames: list[pd.DataFrame] = []
    period_rf = risk_free_rate / 12.0

    for lookback in lookbacks:
        start = prices.shift(skip + lookback)
        end = prices.shift(skip)
        trailing_return = end / start - 1.0

        if vol_scaled:
            monthly_returns = prices.pct_change()
            excess = monthly_returns - period_rf
            vol = excess.rolling(vol_window).std(ddof=1) * np.sqrt(12.0)
            score = trailing_return / vol
        else:
            score = trailing_return

        score_frames.append(score)

    return sum(score_frames) / len(score_frames)


def _weights_from_scores(
    scores: pd.Series,
    allocation: str,
    absolute: bool,
) -> pd.Series:
    if allocation == "winner":
        top_asset = scores.idxmax()
        top_score = float(scores[top_asset])
        weights = pd.Series(0.0, index=scores.index)
        if absolute and top_score <= 0.0:
            return weights
        weights[top_asset] = 1.0
        return weights

    positive = scores.clip(lower=0.0)
    total = float(positive.sum())
    if total > 0.0:
        return positive / total

    top_asset = scores.idxmax()
    weights = pd.Series(0.0, index=scores.index)
    if absolute:
        return weights
    weights[top_asset] = 1.0
    return weights


def _top_asset(weights: pd.Series) -> str | None:
    if weights.sum() <= 0.0:
        return None
    return str(weights.idxmax())


def _validate_momentum_params(
    lookbacks: tuple[int, ...],
    vol_window: int,
    allocation: str,
    threshold: float,
    skip: int,
    vol_scaled: bool,
) -> None:
    if not lookbacks:
        raise ValueError("lookbacks must not be empty.")
    if any(lb <= 0 for lb in lookbacks):
        raise ValueError("lookback windows must be positive.")
    if vol_scaled and vol_window <= 1:
        raise ValueError("vol_window must be > 1 when vol_scaled is True.")
    if allocation not in {"winner", "relative"}:
        raise ValueError('allocation must be "winner" or "relative".')
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative.")
    if skip < 0:
        raise ValueError("skip must be non-negative.")
