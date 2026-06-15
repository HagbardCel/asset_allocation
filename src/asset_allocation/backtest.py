"""Portfolio backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from asset_allocation.config import BacktestConfig


@dataclass
class BacktestResult:
    """Output of a portfolio backtest."""

    equity: pd.Series
    returns: pd.Series
    held_weights: pd.DataFrame
    trade_log: pd.DataFrame
    total_taxes: float
    total_costs: float
    num_trades: int
    config: BacktestConfig = field(default_factory=BacktestConfig)


def run_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    config: BacktestConfig | None = None,
    initial_capital: float = 10_000.0,
) -> BacktestResult:
    """Simulate a strategy given prices and target weights."""
    config = config or BacktestConfig()
    assets = list(prices.columns)
    dates = prices.index

    _ensure_weights_columns(weights, assets)

    cash = initial_capital
    units = {asset: 0.0 for asset in assets}
    avg_cost = {asset: 0.0 for asset in assets}

    equity_values: list[float] = []
    held_weight_rows: list[dict[str, float]] = []
    trade_rows: list[dict] = []
    total_taxes = 0.0
    total_costs = 0.0
    num_trades = 0

    period_rf = config.risk_free_rate / config.periods_per_year

    for i, date in enumerate(dates):
        price_row = prices.loc[date]

        if i > 0 and period_rf != 0.0:
            cash *= 1.0 + period_rf

        portfolio_value = cash + sum(units[a] * price_row[a] for a in assets)

        target_row = weights.loc[date]
        should_rebalance = not target_row.isna().all()

        if should_rebalance:
            target_weights = target_row.fillna(0.0)
            cash_weight = 1.0 - float(target_weights.sum())

            period_costs = 0.0
            period_taxes = 0.0
            period_trades = 0

            for asset in assets:
                target_value = portfolio_value * float(target_weights[asset])
                current_value = units[asset] * price_row[asset]
                trade_value = target_value - current_value

                if abs(trade_value) < config.trade_tolerance:
                    continue

                price = float(price_row[asset])
                trade_units = trade_value / price

                if trade_units < 0:
                    units_sold = -trade_units
                    proceeds = units_sold * price
                    gain = (price - avg_cost[asset]) * units_sold
                    tax = 0.0
                    if config.apply_taxes and gain > 0:
                        tax = gain * config.effective_tax_rate

                    cost = 0.0
                    if config.apply_transaction_costs:
                        cost = abs(trade_value) * config.transaction_cost_pct

                    units[asset] -= units_sold
                    if units[asset] <= config.trade_tolerance:
                        units[asset] = 0.0
                        avg_cost[asset] = 0.0

                    cash += proceeds - tax - cost
                    period_taxes += tax
                    period_costs += cost
                    period_trades += 1

                    trade_rows.append(
                        {
                            "date": date,
                            "asset": asset,
                            "side": "sell",
                            "units": units_sold,
                            "price": price,
                            "value": proceeds,
                            "tax": tax,
                            "cost": cost,
                            "realized_gain": max(gain, 0.0),
                        }
                    )
                else:
                    buy_value = trade_units * price
                    cost = 0.0
                    if config.apply_transaction_costs:
                        cost = buy_value * config.transaction_cost_pct

                    new_units = units[asset] + trade_units
                    if new_units > config.trade_tolerance:
                        avg_cost[asset] = (
                            units[asset] * avg_cost[asset] + buy_value
                        ) / new_units
                    units[asset] = new_units
                    cash -= buy_value + cost
                    period_costs += cost
                    period_trades += 1

                    trade_rows.append(
                        {
                            "date": date,
                            "asset": asset,
                            "side": "buy",
                            "units": trade_units,
                            "price": price,
                            "value": buy_value,
                            "tax": 0.0,
                            "cost": cost,
                            "realized_gain": 0.0,
                        }
                    )

            target_cash = portfolio_value * cash_weight
            cash_delta = target_cash - cash
            if abs(cash_delta) > config.trade_tolerance:
                cash = target_cash

            total_taxes += period_taxes
            total_costs += period_costs
            num_trades += period_trades

        portfolio_value = cash + sum(units[a] * price_row[a] for a in assets)
        equity_values.append(portfolio_value)

        if portfolio_value > 0:
            weight_row = {
                a: (units[a] * price_row[a]) / portfolio_value for a in assets
            }
            weight_row["cash"] = cash / portfolio_value
        else:
            weight_row = {a: 0.0 for a in assets}
            weight_row["cash"] = 0.0
        held_weight_rows.append(weight_row)

    equity = pd.Series(equity_values, index=dates, name="equity")
    returns = equity.pct_change().dropna()
    held_weights = pd.DataFrame(held_weight_rows, index=dates)
    trade_log = pd.DataFrame(trade_rows)

    return BacktestResult(
        equity=equity,
        returns=returns,
        held_weights=held_weights,
        trade_log=trade_log,
        total_taxes=total_taxes,
        total_costs=total_costs,
        num_trades=num_trades,
        config=config,
    )


def _ensure_weights_columns(weights: pd.DataFrame, assets: list[str]) -> None:
    missing = set(assets) - set(weights.columns)
    if missing:
        raise ValueError(f"Weights missing columns for assets: {missing}")

    defined = weights.dropna(how="all")
    if defined.empty:
        raise ValueError("Weights must define at least one rebalance date.")

    if (defined < 0).any().any():
        raise ValueError("Weights must be non-negative.")

    row_sums = defined.sum(axis=1)
    if (row_sums > 1.0 + 1e-9).any():
        raise ValueError("Row weights must sum to at most 1.0.")
