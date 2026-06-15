"""Plotting helpers for backtest analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from asset_allocation.backtest import BacktestResult
from asset_allocation.metrics import calendar_year_returns


def plot_equity_curve(
    results: dict[str, BacktestResult],
    *,
    log_scale: bool = False,
    ax: plt.Axes | None = None,
    title: str = "Equity Curve",
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    for name, result in results.items():
        normalized = result.equity / result.equity.iloc[0]
        ax.plot(normalized.index, normalized.values, label=name)

    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_drawdown(
    results: dict[str, BacktestResult],
    *,
    ax: plt.Axes | None = None,
    title: str = "Drawdown",
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    for name, result in results.items():
        equity = result.equity
        dd = equity / equity.cummax() - 1.0
        ax.fill_between(dd.index, dd.values, 0, alpha=0.3, label=name)
        ax.plot(dd.index, dd.values, linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_allocation(
    result: BacktestResult,
    *,
    ax: plt.Axes | None = None,
    title: str = "Allocation Over Time",
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    weights = result.held_weights.copy()
    ax.stackplot(
        weights.index,
        [weights[col] for col in weights.columns],
        labels=list(weights.columns),
        alpha=0.8,
    )
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_return_distribution(
    results: dict[str, BacktestResult],
    *,
    ax: plt.Axes | None = None,
    title: str = "Monthly Return Distribution",
    bins: int = 30,
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    for name, result in results.items():
        ax.hist(
            result.returns * 100,
            bins=bins,
            alpha=0.5,
            label=name,
            density=True,
        )

    ax.set_title(title)
    ax.set_xlabel("Monthly return (%)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_calendar_year_returns(
    results: dict[str, BacktestResult],
    *,
    ax: plt.Axes | None = None,
    title: str = "Calendar Year Returns",
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    yearly = pd.DataFrame({name: calendar_year_returns(r) for name, r in results.items()})
    yearly.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Return")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_summary(
    results: dict[str, BacktestResult],
    *,
    focus: str | None = None,
) -> plt.Figure:
    """Arrange key plots in a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_equity_curve(results, ax=axes[0, 0])
    plot_drawdown(results, ax=axes[0, 1])
    plot_return_distribution(results, ax=axes[1, 0])
    plot_calendar_year_returns(results, ax=axes[1, 1])

    focus_name = focus or next(iter(results))
    if focus_name in results:
        fig.suptitle(f"Backtest Summary (allocation: {focus_name})", y=1.02)
    fig.tight_layout()
    return fig
