"""Anchored walk-forward evaluation for strategy parameter selection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd

from asset_allocation.backtest import BacktestResult, run_backtest
from asset_allocation.config import BacktestConfig
from asset_allocation.metrics import cagr, max_drawdown, sharpe_ratio
from asset_allocation.strategy import momentum_weights

ObjectiveFn = Callable[[BacktestResult], float]
StrategyFn = Callable[..., pd.DataFrame]

_NEGATIVE_INF = float("-inf")


@dataclass
class WalkForwardFold:
    """One anchored walk-forward fold."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    selected_name: str
    selected_params: dict[str, Any]
    train_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Stitched out-of-sample walk-forward evaluation."""

    folds: list[WalkForwardFold]
    oos_weights: pd.DataFrame
    oos_result: BacktestResult
    selections: pd.DataFrame


def walk_forward(
    prices: pd.DataFrame,
    param_grid: dict[str, dict[str, Any]],
    *,
    objective: str | ObjectiveFn = "sharpe",
    train_min_months: int = 60,
    test_months: int = 12,
    config: BacktestConfig | None = None,
    strategy_fn: StrategyFn = momentum_weights,
    liquidate_training_folds: bool = False,
) -> WalkForwardResult:
    """Run anchored walk-forward parameter selection and stitch OOS segments.

    At each fold an expanding training window is used to pick the best
    parameter set by ``objective``. The selected config's weight rows in the
    subsequent test window are stitched into one continuous OOS backtest.

    Training folds use ``liquidate_training_folds`` to control whether
    ``liquidate_at_end`` applies during in-sample scoring. The stitched OOS
    backtest always uses the caller's ``config`` unchanged.

  Note: path-dependent momentum state (threshold band, held asset) is reset
    when each candidate's weights are computed over full history; acceptable
    for v1 but may differ slightly from a strictly sequential state machine.
    """
    if not param_grid:
        raise ValueError("param_grid must not be empty.")
    if train_min_months < 1:
        raise ValueError("train_min_months must be >= 1.")
    if test_months < 1:
        raise ValueError("test_months must be >= 1.")
    if train_min_months + test_months > len(prices):
        raise ValueError(
            "Not enough data for walk-forward: "
            f"need at least {train_min_months + test_months} months, got {len(prices)}."
        )

    config = config or BacktestConfig()
    score_fn = _resolve_objective(objective, config)

    # Cache full-history weights per candidate (no lookahead at each rebalance date).
    cached_weights: dict[str, pd.DataFrame] = {}
    for name, params in param_grid.items():
        cached_weights[name] = strategy_fn(prices, **params)

    folds: list[WalkForwardFold] = []
    stitched_rows: list[pd.DataFrame] = []

    start = train_min_months
    while start < len(prices):
        test_end_idx = min(start + test_months, len(prices)) - 1
        train_prices = prices.iloc[:start]
        test_start = prices.index[start]
        test_end = prices.index[test_end_idx]
        train_start = prices.index[0]
        train_end = prices.index[start - 1]

        train_scores: dict[str, float] = {}
        for name, params in param_grid.items():
            train_scores[name] = _score_on_train(
                train_prices,
                params,
                score_fn,
                config,
                strategy_fn,
                liquidate_training_folds,
            )

        if all(score == _NEGATIVE_INF for score in train_scores.values()):
            raise ValueError(
                f"All parameter candidates failed for fold ending {train_end.date()}."
            )

        selected_name = max(train_scores, key=train_scores.get)  # type: ignore[arg-type]
        selected_params = param_grid[selected_name]

        fold = WalkForwardFold(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            selected_name=selected_name,
            selected_params=selected_params,
            train_scores=train_scores,
        )
        folds.append(fold)

        fold_weights = _extract_fold_weights(
            cached_weights[selected_name],
            test_start=test_start,
            test_end=test_end,
        )
        stitched_rows.append(fold_weights)

        start += test_months

    oos_start = prices.index[train_min_months]
    oos_prices = prices.loc[oos_start:]
    oos_weights = _stitch_oos_weights(stitched_rows, oos_prices.index)
    oos_result = run_backtest(oos_prices, oos_weights, config)

    selections = pd.DataFrame(
        [
            {
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "selected": fold.selected_name,
                "train_score": fold.train_scores[fold.selected_name],
            }
            for fold in folds
        ]
    ).set_index("test_start")

    return WalkForwardResult(
        folds=folds,
        oos_weights=oos_weights,
        oos_result=oos_result,
        selections=selections,
    )


def _resolve_objective(
    objective: str | ObjectiveFn,
    config: BacktestConfig,
) -> ObjectiveFn:
    if callable(objective):
        return objective

    key = objective.lower()
    if key == "sharpe":
        return lambda r: sharpe_ratio(
            r.returns, config.risk_free_rate, config.periods_per_year
        )
    if key == "cagr":
        return lambda r: cagr(r.equity, config.periods_per_year)
    if key == "calmar":
        return lambda r: _calmar(r, config.periods_per_year)

    raise ValueError(
        f'Unknown objective "{objective}". Use "sharpe", "cagr", "calmar", or a callable.'
    )


def _calmar(result: BacktestResult, periods_per_year: int) -> float:
    mdd = max_drawdown(result.equity)
    if mdd == 0:
        return 0.0
    return cagr(result.equity, periods_per_year) / abs(mdd)


def _score_on_train(
    train_prices: pd.DataFrame,
    params: dict[str, Any],
    score_fn: ObjectiveFn,
    config: BacktestConfig,
    strategy_fn: StrategyFn,
    liquidate_training_folds: bool,
) -> float:
    try:
        train_weights = strategy_fn(train_prices, **params)
        defined = train_weights.dropna(how="all")
        if defined.empty:
            return _NEGATIVE_INF
        train_config = replace(config, liquidate_at_end=liquidate_training_folds)
        result = run_backtest(train_prices, train_weights, train_config)
        if result.returns.empty:
            return _NEGATIVE_INF
        return score_fn(result)
    except (ValueError, KeyError):
        return _NEGATIVE_INF


def _weights_at_date(weights: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    """Last defined target weights on or before ``date``."""
    defined = weights.loc[:date].dropna(how="all")
    if defined.empty:
        return pd.Series(0.0, index=weights.columns)
    return defined.iloc[-1]


def _extract_fold_weights(
    full_weights: pd.DataFrame,
    *,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> pd.DataFrame:
    """Weight rows for one OOS fold, with a rebalance forced at ``test_start``."""
    mask = (full_weights.index >= test_start) & (full_weights.index <= test_end)
    segment = full_weights.loc[mask].dropna(how="all").copy()

    start_weights = _weights_at_date(full_weights, test_start)
    if segment.empty or segment.index[0] != test_start:
        start_row = pd.DataFrame([start_weights], index=[test_start])
        segment = pd.concat([start_row, segment])
        segment = segment[~segment.index.duplicated(keep="first")]

    return segment.sort_index()


def _stitch_oos_weights(
    segments: list[pd.DataFrame],
    oos_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    if not segments:
        raise ValueError("No OOS segments to stitch.")

    assets = segments[0].columns
    combined = pd.concat(segments)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.reindex(oos_index)
    return combined.reindex(columns=assets)
