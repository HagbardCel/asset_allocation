"""Strategy signal correctness."""

from __future__ import annotations

import pandas as pd

from asset_allocation.strategy import momentum_weights


def test_momentum_no_lookahead_by_default() -> None:
    dates = pd.date_range("2019-01-31", periods=15, freq="ME")
    n = len(dates)

    prices_high_shock = pd.DataFrame(
        {
            "a": [100.0 + i for i in range(n)],
            "b": [100.0] * (n - 1) + [300.0],
        },
        index=dates,
    )
    prices_low_shock = prices_high_shock.copy()
    prices_low_shock.loc[dates[-1], "b"] = 50.0

    weights_high = momentum_weights(prices_high_shock, lookbacks=(12,))
    weights_low = momentum_weights(prices_low_shock, lookbacks=(12,))

    defined_high = weights_high.dropna(how="all")
    defined_low = weights_low.dropna(how="all")
    assert not defined_high.empty
    assert not defined_low.empty
    pd.testing.assert_series_equal(defined_high.iloc[-1], defined_low.iloc[-1])

    weights_high_la = momentum_weights(prices_high_shock, lookbacks=(12,), skip=0)
    weights_low_la = momentum_weights(prices_low_shock, lookbacks=(12,), skip=0)
    defined_high_la = weights_high_la.dropna(how="all")
    defined_low_la = weights_low_la.dropna(how="all")
    assert not defined_high_la.empty
    assert not defined_low_la.empty
    assert not defined_high_la.iloc[-1].equals(defined_low_la.iloc[-1])
