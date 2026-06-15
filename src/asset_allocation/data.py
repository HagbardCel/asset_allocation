"""Load MSCI index CSV files into a joint price table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_MAPPING: dict[str, str] = {
    "momentum": "msci_world_momentum.csv",
    "value": "msci_world_value_enhanced.csv",
}


def _default_data_dir() -> Path:
    """Resolve the data directory from common working directories."""
    package_root = Path(__file__).resolve().parents[2]
    candidates = [
        Path("data"),
        Path("../data"),
        package_root / "data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path("data")


def load_index_csv(path: str | Path, name: str) -> pd.Series:
    """Load a single MSCI index CSV file.

    The files have 6 metadata rows, a header row, monthly data rows, and a
    disclaimer footer. Values use comma as thousands separator.
    """
    path = Path(path)
    raw = pd.read_csv(path, sep=";", header=None, skiprows=7, dtype=str)

    dates = pd.to_datetime(raw.iloc[:, 0], format="%b %d, %Y", errors="coerce")
    values = (
        raw.iloc[:, 1]
        .astype(str)
        .str.replace(",", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    series = pd.Series(values.values, index=dates, name=name)
    series = series.dropna()
    series.index = pd.DatetimeIndex(series.index)
    series = series.sort_index()
    _validate_monthly_series(series, path)
    return series


def _validate_monthly_series(series: pd.Series, path: Path) -> None:
    if not series.index.is_unique:
        raise ValueError(f"Duplicate dates in {path}")

    if (series <= 0).any():
        raise ValueError(f"Non-positive index level in {path}")

    periods = series.index.to_period("M")
    expected = pd.period_range(periods.min(), periods.max(), freq="M")
    missing = expected.difference(periods)
    if len(missing):
        raise ValueError(f"Missing monthly observations in {path}: {missing.tolist()}")


def load_prices(
    data_dir: str | Path | None = None,
    mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load all index CSVs and return a joint price table.

    Rows are dates, columns are asset names. Leading rows where any asset
    is missing are dropped so all columns have data.
    """
    data_dir = Path(data_dir) if data_dir is not None else _default_data_dir()
    mapping = mapping or DEFAULT_MAPPING

    series_list: list[pd.Series] = []
    for name, filename in mapping.items():
        series_list.append(load_index_csv(data_dir / filename, name))

    prices = pd.concat(series_list, axis=1)
    prices = prices.sort_index()
    prices = prices.dropna(how="any")
    prices.index.name = "date"
    return prices
