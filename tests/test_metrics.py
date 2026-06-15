"""Performance metric correctness."""

from __future__ import annotations

import pandas as pd
import pytest

from asset_allocation.backtest import BacktestResult
from asset_allocation.metrics import calendar_year_returns


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
