"""Performance metric correctness."""

from __future__ import annotations

import pandas as pd
import pytest

from asset_allocation.backtest import BacktestResult, run_backtest
from asset_allocation.config import BacktestConfig
from asset_allocation.metrics import (
    annual_turnover_ex_liquidation,
    annual_turnover_incl_liquidation,
    calendar_year_returns,
)


def test_calendar_year_returns_includes_january() -> None:
    dates = pd.to_datetime(["2019-12-31", "2020-01-31", "2020-12-31"])
    equity = pd.Series([100.0, 110.0, 121.0], index=dates)
    result = BacktestResult(
        equity=equity,
        returns=equity.pct_change().dropna(),
        held_weights=pd.DataFrame(),
        trade_log=pd.DataFrame(),
        total_taxes=0.0,
        total_costs=0.0,
        num_trades=0,
    )

    yearly = calendar_year_returns(result)

    assert yearly[2019] == pytest.approx(0.0, abs=1e-12)
    assert yearly[2020] == pytest.approx(0.21, rel=1e-9)
    assert yearly[2020] != pytest.approx(0.10, rel=1e-3)


def test_annual_turnover_excludes_liquidation() -> None:
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    prices = pd.DataFrame({"a": [100.0, 120.0, 130.0]}, index=dates)
    weights = pd.DataFrame({"a": [1.0, 1.0, 1.0]}, index=dates)
    config = BacktestConfig(
        apply_transaction_costs=False,
        apply_taxes=False,
        liquidate_at_end=True,
    )

    result = run_backtest(prices, weights, config, initial_capital=10_000.0)

    incl = annual_turnover_incl_liquidation(result)
    ex = annual_turnover_ex_liquidation(result)

    assert incl > ex
    assert incl > 0.0
