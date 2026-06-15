"""Dynamic asset allocation strategy backtesting toolkit."""

from asset_allocation.backtest import BacktestResult, run_backtest
from asset_allocation.config import BacktestConfig
from asset_allocation.data import load_index_csv, load_prices
from asset_allocation.metrics import summary
from asset_allocation.strategy import (
    align_weights,
    buy_and_hold_weights,
    constant_weights,
    momentum_weights,
    periodic_weights,
)
from asset_allocation.walkforward import WalkForwardFold, WalkForwardResult, walk_forward

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "WalkForwardFold",
    "WalkForwardResult",
    "align_weights",
    "buy_and_hold_weights",
    "constant_weights",
    "load_index_csv",
    "load_prices",
    "momentum_weights",
    "periodic_weights",
    "run_backtest",
    "summary",
    "walk_forward",
]
