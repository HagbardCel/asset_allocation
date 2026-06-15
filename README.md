# Asset Allocation Strategy Backtester

Evaluate dynamic asset allocation strategies on historical MSCI index data. Strategies are expressed as a table of target weights per asset per date; the toolkit handles backtesting, a simplified German tax model, transaction costs, performance metrics, and analysis plots.

## Data

Monthly net total-return index levels (USD) in `data/`:

- `msci_world_momentum.csv` — MSCI World Momentum (from Jan 1997)
- `msci_world_value_enhanced.csv` — MSCI World Enhanced Value (from Dec 1997)

The joint price table starts in Dec 1997 when both series are available.

## Setup

```bash
uv sync
```

## Quick start (notebook)

```bash
uv run jupyter notebook notebooks/01_validation.ipynb
```

## API overview

```python
from asset_allocation.data import load_prices
from asset_allocation.strategy import constant_weights, periodic_weights
from asset_allocation.backtest import run_backtest
from asset_allocation.config import BacktestConfig
from asset_allocation.metrics import summary
from asset_allocation.plotting import plot_summary

prices = load_prices()
weights = constant_weights(prices, {"momentum": 1.0})
result = run_backtest(prices, weights)
print(summary(result))
plot_summary({"momentum": result})
```

### Strategy format

A strategy is a `DataFrame` of target weights (one column per asset, one row per rebalance date). Row weights should sum to ≤ 1; any remainder is held as cash.

Helpers:

- `constant_weights(prices, {"momentum": 1.0})` — fixed allocation, rebalanced every period
- `periodic_weights(prices, {"momentum": 0.5, "value": 0.5}, freq="Y")` — rebalance annually; weights drift between rebalance dates
- `momentum_weights(prices, lookbacks=(12,), allocation="winner")` — rank assets by trailing momentum (optional volatility scaling, relative weights, momentum-gap threshold, absolute/cash mode)

After transaction costs and taxes, held weights may deviate slightly from targets because buys are scaled to available cash.

### Momentum strategy

`momentum_weights` ranks assets on trailing returns (averaged across one or more lookback windows). Defaults: 12-month lookback, winner-take-all, monthly rebalance, fully invested.

```python
from asset_allocation.strategy import momentum_weights

# Classic 12-month winner-take-all
weights = momentum_weights(prices)

# MSCI-style blend with volatility scaling and a 2% momentum-gap band
weights = momentum_weights(
    prices,
    lookbacks=(6, 12),
    vol_scaled=True,
    allocation="relative",
    threshold=0.02,
)

# Defensive: move to cash when momentum is non-positive
weights = momentum_weights(prices, absolute=True)
```

### Taxes and costs (Germany defaults)

The default tax model is a simplified German equity ETF approximation: realized positive gains are taxed using Abgeltungsteuer + Soli after 30% Teilfreistellung. It does not model Vorabpauschale, distributions, loss carry-forward, or full tax-lot accounting.

`BacktestConfig` defaults:

- Abgeltungsteuer 25% + Soli → 26.375%, with 30% Teilfreistellung for equity ETFs (~18.46% effective rate on realized gains)
- Transaction costs: 0.1% of traded notional
- Taxes apply only on sells (average-cost basis)

Toggle with `apply_taxes=False` or `apply_transaction_costs=False`.

By default, all holdings are liquidated on the final evaluation date so exit taxes and costs are included in the reported return (`liquidate_at_end=True`).

### Walk-forward evaluation

Anchored walk-forward selects the best momentum hyperparameters on an expanding training window and stitches out-of-sample segments into one continuous backtest:

```python
from asset_allocation.walkforward import walk_forward

PARAM_GRID = {
    "mom 12m winner": {"lookbacks": (12,), "allocation": "winner"},
    "mom 6m winner": {"lookbacks": (6,), "allocation": "winner"},
}

wf = walk_forward(
    prices,
    PARAM_GRID,
    objective="sharpe",      # or "cagr", "calmar", or a custom callable
    train_min_months=60,
    test_months=12,
    config=BacktestConfig(),
)

print(wf.selections)       # which config won each fold
print(summary(wf.oos_result, "walk-forward OOS"))
```

### Metrics

CAGR, annual volatility, Sharpe, Sortino, Calmar, max drawdown, longest drawdown, win rate, best/worst month, calendar-year returns, annual turnover (ex- and incl. liquidation), time in market, number of trades, total taxes and costs.
