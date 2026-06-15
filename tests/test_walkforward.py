"""Walk-forward evaluation robustness."""

from __future__ import annotations

import pandas as pd
import pytest

from asset_allocation.config import BacktestConfig
from asset_allocation.strategy import momentum_weights
from asset_allocation.walkforward import _resolve_objective, _score_on_train, walk_forward


def test_walk_forward_raises_when_all_candidates_fail() -> None:
    dates = pd.date_range("2020-01-31", periods=72, freq="ME")
    prices = pd.DataFrame(
        {"a": range(100, 172), "b": range(200, 272)},
        index=dates,
        dtype=float,
    )
    param_grid = {
        "long_a": {"lookbacks": (60,)},
        "long_b": {"lookbacks": (59,)},
    }

    with pytest.raises(ValueError, match="All parameter candidates failed"):
        walk_forward(
            prices,
            param_grid,
            train_min_months=60,
            test_months=12,
        )


def test_liquidate_training_folds_changes_training_score() -> None:
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    prices = pd.DataFrame({"a": [100.0 + i * 5 for i in range(24)]}, index=dates)
    params = {"lookbacks": (3,), "skip": 0}
    config = BacktestConfig(
        apply_transaction_costs=False,
        apply_taxes=True,
        liquidate_at_end=True,
    )
    score_fn = _resolve_objective("cagr", config)

    score_no_liq = _score_on_train(
        prices, params, score_fn, config, momentum_weights, False
    )
    score_liq = _score_on_train(
        prices, params, score_fn, config, momentum_weights, True
    )

    assert score_no_liq != score_liq
