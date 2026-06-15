"""Data loader validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from asset_allocation.data import load_index_csv

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_valid_index_csv() -> None:
    series = load_index_csv(FIXTURES / "valid_index.csv", "test")
    assert len(series) == 3
    assert series.iloc[0] == pytest.approx(1000.0)
    assert series.iloc[-1] == pytest.approx(1020.0)


def test_load_valid_index_csv_ignores_footer_and_parses_thousands() -> None:
    series = load_index_csv(FIXTURES / "valid_index.csv", "test")

    assert series.iloc[0] == pytest.approx(1000.0)
    assert series.iloc[1] == pytest.approx(1010.0)
    assert all(value > 0 for value in series)


def test_load_index_csv_rejects_empty_series() -> None:
    with pytest.raises(ValueError, match="No valid observations"):
        load_index_csv(FIXTURES / "empty_index.csv", "test")


def test_load_unsorted_index_csv() -> None:
    series = load_index_csv(FIXTURES / "unsorted_index.csv", "test")
    assert series.index.is_monotonic_increasing
    assert series.iloc[0] == pytest.approx(1000.0)
    assert series.iloc[-1] == pytest.approx(1020.0)


@pytest.mark.parametrize(
    ("fixture", "message"),
    [
        ("duplicate_dates.csv", "Duplicate dates"),
        ("missing_month.csv", "Missing monthly observations"),
        ("nonpositive.csv", "Non-positive index level"),
    ],
)
def test_load_invalid_index_csv_raises(fixture: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_index_csv(FIXTURES / fixture, "test")
