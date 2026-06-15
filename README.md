# Asset Allocation Strategy Backtester

Evaluate dynamic asset allocation strategies on historical MSCI index data. Strategies are expressed as a table of target weights per asset per date; the toolkit handles backtesting, German taxes and transaction costs, performance metrics, and analysis plots.

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
plot_summary(result, prices)
```

### Strategy format

A strategy is a `DataFrame` of target weights (one column per asset, one row per rebalance date). Row weights should sum to ≤ 1; any remainder is held as cash.

Helpers:

- `constant_weights(prices, {"momentum": 1.0})` — fixed allocation, rebalanced every period
- `periodic_weights(prices, {"momentum": 0.5, "value": 0.5}, freq="Y")` — rebalance annually; weights drift between rebalance dates

### Taxes and costs (Germany defaults)

`BacktestConfig` defaults:

- Abgeltungsteuer 25% + Soli → 26.375%, with 30% Teilfreistellung for equity ETFs (~18.46% effective rate on realized gains)
- Transaction costs: 0.1% of traded notional
- Taxes apply only on sells (average-cost basis)

Toggle with `apply_taxes=False` or `apply_transaction_costs=False`.

### Metrics

CAGR, annual volatility, Sharpe, Sortino, Calmar, max drawdown, longest drawdown, win rate, best/worst month, calendar-year returns, annual turnover, time in market, number of trades, total taxes and costs.
