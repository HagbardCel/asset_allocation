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
    """Simulate a strategy given prices and target weights.

    When ``liquidate_at_end`` is enabled (default), all assets are sold on the
    final evaluation date so exit taxes and transaction costs are included.
    """
    config = config or BacktestConfig()
    assets = list(prices.columns)
    dates = prices.index

    _ensure_weights_columns(weights, assets)
    weights = weights.reindex(prices.index)

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

            period_costs = 0.0
            period_taxes = 0.0
            period_trades = 0

            trade_values: dict[str, float] = {}
            for asset in assets:
                target_value = portfolio_value * float(target_weights[asset])
                current_value = units[asset] * price_row[asset]
                trade_value = target_value - current_value
                if abs(trade_value) >= config.trade_tolerance:
                    trade_values[asset] = trade_value

            for asset, trade_value in trade_values.items():
                if trade_value >= 0:
                    continue

                price = float(price_row[asset])
                trade_units = trade_value / price
                cash_delta, tax, cost, gain, sold = _sell_asset(
                    units_sold=-trade_units,
                    price=price,
                    avg_cost=avg_cost[asset],
                    config=config,
                )
                units[asset] -= sold
                if units[asset] <= config.trade_tolerance:
                    units[asset] = 0.0
                    avg_cost[asset] = 0.0

                cash += cash_delta
                period_taxes += tax
                period_costs += cost
                period_trades += 1

                trade_rows.append(
                    _trade_record(
                        date, asset, "sell", sold, price, sold * price, tax, cost, gain
                    )
                )

            buy_values = {a: v for a, v in trade_values.items() if v > 0}
            if buy_values:
                cost_pct = (
                    config.transaction_cost_pct
                    if config.apply_transaction_costs
                    else 0.0
                )
                total_buy_need = sum(v * (1.0 + cost_pct) for v in buy_values.values())
                available_cash = max(cash, 0.0)
                if total_buy_need > config.trade_tolerance:
                    scale = min(1.0, available_cash / total_buy_need)
                else:
                    scale = 0.0

                for asset, trade_value in buy_values.items():
                    scaled_value = trade_value * scale
                    if scaled_value < config.trade_tolerance:
                        continue

                    price = float(price_row[asset])
                    trade_units = scaled_value / price
                    buy_value = trade_units * price
                    cost = buy_value * cost_pct

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
                        _trade_record(
                            date,
                            asset,
                            "buy",
                            trade_units,
                            price,
                            buy_value,
                            0.0,
                            cost,
                            0.0,
                        )
                    )

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

    if config.liquidate_at_end:
        final_date = dates[-1]
        price_row = prices.loc[final_date]
        period_taxes = 0.0
        period_costs = 0.0
        period_trades = 0

        for asset in assets:
            if units[asset] <= config.trade_tolerance:
                continue

            price = float(price_row[asset])
            units_sold = units[asset]
            cash_delta, tax, cost, gain, sold = _sell_asset(
                units_sold=units_sold,
                price=price,
                avg_cost=avg_cost[asset],
                config=config,
            )
            units[asset] = 0.0
            avg_cost[asset] = 0.0
            cash += cash_delta
            period_taxes += tax
            period_costs += cost
            period_trades += 1

            trade_rows.append(
                _trade_record(
                    final_date,
                    asset,
                    "sell",
                    sold,
                    price,
                    sold * price,
                    tax,
                    cost,
                    gain,
                    liquidation=True,
                )
            )

        total_taxes += period_taxes
        total_costs += period_costs
        num_trades += period_trades

        equity_values[-1] = cash
        held_weight_rows[-1] = {a: 0.0 for a in assets}
        held_weight_rows[-1]["cash"] = 1.0 if cash > 0 else 0.0

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


def _sell_asset(
    *,
    units_sold: float,
    price: float,
    avg_cost: float,
    config: BacktestConfig,
) -> tuple[float, float, float, float, float]:
    proceeds = units_sold * price
    gain = (price - avg_cost) * units_sold
    tax = gain * config.effective_tax_rate if config.apply_taxes and gain > 0 else 0.0
    cost = (
        proceeds * config.transaction_cost_pct
        if config.apply_transaction_costs
        else 0.0
    )
    cash_delta = proceeds - tax - cost
    return cash_delta, tax, cost, max(gain, 0.0), units_sold


def _trade_record(
    date: pd.Timestamp,
    asset: str,
    side: str,
    units: float,
    price: float,
    value: float,
    tax: float,
    cost: float,
    realized_gain: float,
    *,
    liquidation: bool = False,
) -> dict:
    return {
        "date": date,
        "asset": asset,
        "side": side,
        "units": units,
        "price": price,
        "value": value,
        "tax": tax,
        "cost": cost,
        "realized_gain": realized_gain,
        "liquidation": liquidation,
    }


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
