"""Data loading utilities for historical price data."""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)


class PriceDataCache:
    """Fetches and caches daily OHLCV data from Yahoo Finance."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or Path(__file__).resolve().parents[2] / "data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str, start: str, end: str) -> Path:
        return self.cache_dir / f"{symbol}_{start}_{end}.parquet"

    def fetch(
        self,
        symbols: list[str],
        start: str,
        end: str,
        column: str = "Close",
    ) -> pd.DataFrame:
        """Fetch adjusted closing prices for multiple symbols."""
        frames = []
        for sym in symbols:
            cache = self._cache_path(sym, start, end)
            if cache.exists():
                df = pd.read_parquet(cache)
            else:
                df = yf.download(
                    sym, start=start, end=end, progress=False, auto_adjust=True
                )
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.to_parquet(cache)
            if column in df.columns:
                frames.append(df[column].rename(sym))
            else:
                cols = [c for c in df.columns if column in str(c)]
                if cols:
                    frames.append(df[cols[0]].rename(sym))
        if not frames:
            raise ValueError("No data loaded")
        prices = pd.concat(frames, axis=1)
        prices.index = pd.to_datetime(prices.index)
        return prices.dropna(how="all").ffill().dropna()


REQUIRED_PANEL_COLUMNS = {"date", "symbol"}


def prepare_candidate_panel(candidate_panel: pd.DataFrame) -> pd.DataFrame:
    """Normalize candidate-panel inputs for universe construction.

    Expected minimum columns are ``date`` and ``symbol``. Optional inputs:
    - ``close`` and ``volume`` to infer dollar volume
    - ``dollar_volume`` when already computed
    - ``is_available`` to explicitly flag eligible observations
    """
    missing = REQUIRED_PANEL_COLUMNS.difference(candidate_panel.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"candidate_panel missing required columns: {missing_list}")

    panel = candidate_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["symbol"] = panel["symbol"].astype(str)

    if "close" not in panel.columns:
        panel["close"] = 1.0
    if "volume" not in panel.columns:
        panel["volume"] = 1.0
    if "dollar_volume" not in panel.columns:
        panel["dollar_volume"] = panel["close"].astype(float) * panel["volume"].astype(float)
    else:
        panel["dollar_volume"] = panel["dollar_volume"].astype(float)

    if "is_available" not in panel.columns:
        panel["is_available"] = (
            panel["close"].notna()
            & panel["volume"].notna()
            & (panel["dollar_volume"] > 0)
        )
    else:
        panel["is_available"] = panel["is_available"].astype(bool)

    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)
