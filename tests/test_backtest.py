"""Backtest accounting and sparse-weight handling."""

from __future__ import annotations

import pandas as pd
import pytest

from asset_allocation.backtest import run_backtest
from asset_allocation.config import BacktestConfig


def test_transaction_cost_reduces_equity() -> None:
    dates = pd.date_range("2020-01-31", periods=1, freq="ME")
    prices = pd.DataFrame({"a": [100.0]}, index=dates)
    weights = pd.DataFrame({"a": [1.0]}, index=dates)
    config = BacktestConfig(
        apply_taxes=False,
        liquidate_at_end=False,
        transaction_cost_pct=0.001,
    )

    result = run_backtest(prices, weights, config, initial_capital=10_000.0)

    expected_equity = 10_000.0 / 1.001
    assert result.equity.iloc[0] == pytest.approx(expected_equity, rel=1e-9)
    assert result.equity.iloc[0] < 10_000.0
    assert result.total_costs == pytest.approx(expected_equity * 0.001, rel=1e-9)


def test_realized_gain_tax_reduces_equity() -> None:
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    prices = pd.DataFrame({"a": [100.0, 150.0]}, index=dates)
    weights = pd.DataFrame({"a": [1.0, 0.0]}, index=dates)
    config = BacktestConfig(
        apply_transaction_costs=False,
        liquidate_at_end=False,
        apply_taxes=True,
    )

    result = run_backtest(prices, weights, config, initial_capital=10_000.0)

    units = 10_000.0 / 100.0
    proceeds = units * 150.0
    gain = (150.0 - 100.0) * units
    tax = gain * config.effective_tax_rate
    expected_final_equity = proceeds - tax

    assert result.total_taxes == pytest.approx(tax, rel=1e-9)
    assert result.equity.iloc[-1] == pytest.approx(expected_final_equity, rel=1e-9)
    assert result.equity.iloc[-1] < proceeds


def test_final_liquidation_changes_final_equity_with_tax() -> None:
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    prices = pd.DataFrame({"a": [100.0, 120.0]}, index=dates)
    weights = pd.DataFrame({"a": [1.0, 1.0]}, index=dates)

    held = BacktestConfig(
        apply_transaction_costs=False,
        liquidate_at_end=False,
        apply_taxes=True,
    )
    liquidated = BacktestConfig(
        apply_transaction_costs=False,
        liquidate_at_end=True,
        apply_taxes=True,
    )

    result_held = run_backtest(prices, weights, held, initial_capital=10_000.0)
    result_liq = run_backtest(prices, weights, liquidated, initial_capital=10_000.0)

    units = 10_000.0 / 100.0
    held_equity = units * 120.0
    gain = (120.0 - 100.0) * units
    tax = gain * liquidated.effective_tax_rate
    liquidated_equity = units * 120.0 - tax

    assert result_held.equity.iloc[-1] == pytest.approx(held_equity, rel=1e-9)
    assert result_liq.equity.iloc[-1] == pytest.approx(liquidated_equity, rel=1e-9)
    assert result_liq.equity.iloc[-1] < result_held.equity.iloc[-1]


def test_sparse_weights_no_key_error() -> None:
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    prices = pd.DataFrame({"a": [100.0, 101.0, 102.0]}, index=dates)
    weights = pd.DataFrame({"a": [1.0]}, index=dates[[0]])

    result = run_backtest(
        prices,
        weights,
        BacktestConfig(liquidate_at_end=False, apply_transaction_costs=False),
    )

    assert len(result.equity) == 3
    assert result.equity.iloc[0] == pytest.approx(10_000.0, rel=1e-9)
