"""Backtest configuration with German tax and cost defaults."""

from dataclasses import dataclass


@dataclass
class BacktestConfig:
    """Configuration for portfolio backtesting."""

    # German Abgeltungsteuer: 25% + 5.5% Soli = 26.375%
    tax_rate: float = 0.26375
    # Teilfreistellung for equity ETFs (30%)
    teilfreistellung: float = 0.30
    apply_taxes: bool = True

    # Transaction costs as fraction of traded notional (0.1% = typical broker fee)
    transaction_cost_pct: float = 0.001
    apply_transaction_costs: bool = True

    # Annual risk-free rate for Sharpe/Sortino and cash interest
    risk_free_rate: float = 0.0
    periods_per_year: int = 12

    # Minimum relative change in units to count as a trade
    trade_tolerance: float = 1e-9

    @property
    def effective_tax_rate(self) -> float:
        """Tax rate after Teilfreistellung on equity ETF gains."""
        return self.tax_rate * (1.0 - self.teilfreistellung)
