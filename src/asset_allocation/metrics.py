"""Performance metrics for backtest results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from asset_allocation.backtest import BacktestResult


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series, periods_per_year: int = 12) -> float:
    if len(equity) < 2:
        return 0.0
    years = (len(equity) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def annual_volatility(returns: pd.Series, periods_per_year: int = 12) -> float:
    if returns.empty:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
) -> float:
    vol = annual_volatility(returns, periods_per_year)
    if vol == 0 or returns.empty:
        return 0.0
    period_rf = risk_free_rate / periods_per_year
    excess = returns.mean() - period_rf
    return float(excess / returns.std(ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
) -> float:
    if returns.empty:
        return 0.0
    period_rf = risk_free_rate / periods_per_year
    excess = returns.mean() - period_rf
    downside = returns[returns < 0]
    if downside.empty or downside.std(ddof=1) == 0:
        return 0.0
    return float(excess / downside.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def longest_drawdown(equity: pd.Series) -> int:
    """Longest underwater stretch in number of periods (months)."""
    if equity.empty:
        return 0
    running_max = equity.cummax()
    underwater = equity < running_max

    max_len = 0
    current = 0
    for is_under in underwater:
        if is_under:
            current += 1
            max_len = max(max_len, current)
        else:
            current = 0
    return max_len


def best_month(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float(returns.max())


def worst_month(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float(returns.min())


def win_rate(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float((returns > 0).mean())


def calendar_year_returns(result: BacktestResult) -> pd.Series:
    """Compound return for each calendar year."""
    equity = result.equity
    year_end_equity = equity.groupby(equity.index.year).last()
    returns = year_end_equity.pct_change()
    returns.iloc[0] = year_end_equity.iloc[0] / equity.iloc[0] - 1.0
    returns.index.name = "year"
    return returns


def _turnover_from_trades(result: BacktestResult, trades: pd.DataFrame) -> float:
    if trades.empty or result.equity.empty:
        return 0.0

    trade_values = trades["value"].abs()
    years = max((len(result.equity) - 1) / result.config.periods_per_year, 1e-9)
    avg_equity = float(result.equity.mean())
    if avg_equity == 0:
        return 0.0
    return float(trade_values.sum() / years / avg_equity)


def annual_turnover_incl_liquidation(result: BacktestResult) -> float:
    """Average annual traded notional divided by average portfolio value."""
    if result.trade_log.empty:
        return 0.0
    return _turnover_from_trades(result, result.trade_log)


def annual_turnover_ex_liquidation(result: BacktestResult) -> float:
    """Annual turnover excluding terminal liquidation trades."""
    if result.trade_log.empty:
        return 0.0

    trades = result.trade_log
    if "liquidation" in trades.columns:
        trades = trades[~trades["liquidation"].fillna(False)]
    return _turnover_from_trades(result, trades)


def annual_turnover(result: BacktestResult) -> float:
    """Average annual traded notional excluding liquidation trades."""
    return annual_turnover_ex_liquidation(result)


def time_in_market(result: BacktestResult) -> float:
    """Average invested weight (1 minus average cash weight)."""
    if "cash" not in result.held_weights.columns:
        return 1.0
    return float(1.0 - result.held_weights["cash"].mean())


def summary(result: BacktestResult, name: str = "strategy") -> pd.DataFrame:
    """One-row DataFrame with all key metrics."""
    cfg = result.config
    metrics = {
        "name": name,
        "total_return": total_return(result.equity),
        "cagr": cagr(result.equity, cfg.periods_per_year),
        "annual_volatility": annual_volatility(result.returns, cfg.periods_per_year),
        "sharpe_ratio": sharpe_ratio(
            result.returns, cfg.risk_free_rate, cfg.periods_per_year
        ),
        "sortino_ratio": sortino_ratio(
            result.returns, cfg.risk_free_rate, cfg.periods_per_year
        ),
        "calmar_ratio": _calmar(result),
        "max_drawdown": max_drawdown(result.equity),
        "longest_drawdown_months": longest_drawdown(result.equity),
        "best_month": best_month(result.returns),
        "worst_month": worst_month(result.returns),
        "win_rate": win_rate(result.returns),
        "annual_turnover": annual_turnover(result),
        "annual_turnover_incl_liquidation": annual_turnover_incl_liquidation(result),
        "time_in_market": time_in_market(result),
        "num_trades": result.num_trades,
        "total_taxes": result.total_taxes,
        "total_costs": result.total_costs,
    }
    return pd.DataFrame([metrics]).set_index("name")


def _calmar(result: BacktestResult) -> float:
    mdd = max_drawdown(result.equity)
    if mdd == 0:
        return 0.0
    return cagr(result.equity, result.config.periods_per_year) / abs(mdd)
