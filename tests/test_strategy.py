"""Strategy signal correctness."""

from __future__ import annotations

import pandas as pd
import pytest

from asset_allocation.strategy import _momentum_scores, momentum_weights, periodic_weights


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


def test_vol_scaled_momentum_no_lookahead_by_default() -> None:
    dates = pd.date_range("2019-01-31", periods=36, freq="ME")
    n = len(dates)

    prices_high_shock = pd.DataFrame(
        {
            "a": [100.0 * (1.01**i) for i in range(n)],
            "b": [100.0 * (1.005**i) for i in range(n - 1)] + [300.0],
        },
        index=dates,
    )
    prices_low_shock = prices_high_shock.copy()
    prices_low_shock.loc[dates[-1], "b"] = 50.0

    kwargs = {"lookbacks": (12,), "vol_scaled": True, "vol_window": 12}
    weights_high = momentum_weights(prices_high_shock, **kwargs)
    weights_low = momentum_weights(prices_low_shock, **kwargs)

    defined_high = weights_high.dropna(how="all")
    defined_low = weights_low.dropna(how="all")
    assert not defined_high.empty
    assert not defined_low.empty
    pd.testing.assert_series_equal(defined_high.iloc[-1], defined_low.iloc[-1])

    score_kwargs = {
        "lookbacks": (12,),
        "vol_scaled": True,
        "vol_window": 12,
        "risk_free_rate": 0.0,
    }
    last = dates[-1]
    scores_high_skip1 = _momentum_scores(prices_high_shock, skip=1, **score_kwargs)
    scores_low_skip1 = _momentum_scores(prices_low_shock, skip=1, **score_kwargs)
    pd.testing.assert_series_equal(
        scores_high_skip1.loc[last], scores_low_skip1.loc[last]
    )

    scores_high_skip0 = _momentum_scores(prices_high_shock, skip=0, **score_kwargs)
    scores_low_skip0 = _momentum_scores(prices_low_shock, skip=0, **score_kwargs)
    assert not scores_high_skip0.loc[last].equals(scores_low_skip0.loc[last])


def test_periodic_weights_invests_from_first_date() -> None:
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    prices = pd.DataFrame({"a": range(100, 112)}, index=dates, dtype=float)

    weights = periodic_weights(prices, {"a": 1.0}, freq="Y")
    defined = weights.dropna(how="all")

    assert defined.index[0] == dates[0]


def test_momentum_raises_on_insufficient_history() -> None:
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    prices = pd.DataFrame({"a": range(100, 105)}, index=dates, dtype=float)

    with pytest.raises(ValueError, match="Not enough price history"):
        momentum_weights(prices, lookbacks=(12,))


def test_relative_momentum_rebalances_on_weight_distance() -> None:
    dates = pd.date_range("2020-01-31", periods=15, freq="ME")
    a_prices = [100.0]
    b_prices = [100.0]
    for _ in range(12):
        a_prices.append(a_prices[-1] * 1.01)
        b_prices.append(b_prices[-1] * 1.01)
    a_prices.append(a_prices[-1] * 1.9)
    b_prices.append(b_prices[-1] * 1.1)
    a_prices.append(a_prices[-1] * 1.55)
    b_prices.append(b_prices[-1] * 1.45)

    prices = pd.DataFrame({"a": a_prices, "b": b_prices}, index=dates)
    weights = momentum_weights(
        prices,
        lookbacks=(1,),
        skip=0,
        allocation="relative",
        threshold=0.0,
    )
    defined = weights.dropna(how="all")

    assert len(defined) >= 2
    first = defined.iloc[-2]
    last = defined.iloc[-1]
    assert first["a"] == pytest.approx(0.9, abs=0.05)
    assert last["a"] == pytest.approx(0.55, abs=0.05)
    assert first["a"] != pytest.approx(last["a"], abs=0.05)
