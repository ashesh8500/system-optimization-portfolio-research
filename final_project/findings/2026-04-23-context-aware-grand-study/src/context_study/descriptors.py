from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

DESCRIPTOR_COLUMNS = [
    "avg_pairwise_corr",
    "first_pc_share",
    "cross_sectional_dispersion",
    "trend_persistence",
    "vol_level",
    "vol_of_vol",
    "liquidity_proxy",
    "regime_switch_rate",
]


def compute_universe_descriptors(
    *,
    price_panel: pd.DataFrame | None = None,
    return_panel: pd.DataFrame | None = None,
    volume_panel: pd.DataFrame | None = None,
    window: int = 60,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute rolling structural descriptors for a tradable universe.

    Parameters
    ----------
    price_panel
        Asset-by-date price panel. Used to infer returns if ``return_panel`` is not supplied.
    return_panel
        Asset-by-date return panel.
    volume_panel
        Optional asset-by-date volume panel used to form a simple liquidity proxy.
    window
        Rolling lookback window in periods.
    min_periods
        Minimum non-missing history required before descriptors are emitted.
    """
    if return_panel is None:
        if price_panel is None:
            raise ValueError("Either return_panel or price_panel must be provided.")
        return_panel = price_panel.pct_change()
    else:
        return_panel = return_panel.copy()

    if window <= 1:
        raise ValueError("window must be greater than 1.")

    min_periods = window if min_periods is None else min_periods
    if min_periods <= 1 or min_periods > window:
        raise ValueError("min_periods must satisfy 1 < min_periods <= window.")

    returns = _coerce_panel(return_panel)
    prices = _coerce_panel(price_panel).reindex_like(returns) if price_panel is not None else None
    volumes = _coerce_panel(volume_panel).reindex_like(returns) if volume_panel is not None else None

    descriptor_frame = pd.DataFrame(index=returns.index, columns=DESCRIPTOR_COLUMNS, dtype=float)

    for end_idx in range(len(returns)):
        window_slice = returns.iloc[max(0, end_idx - window + 1) : end_idx + 1]
        valid_rows = window_slice.dropna(how="all")
        if len(valid_rows) < min_periods:
            continue

        descriptor_frame.iloc[end_idx] = _compute_window_descriptors(
            returns_window=valid_rows,
            prices_window=prices.loc[valid_rows.index] if prices is not None else None,
            volumes_window=volumes.loc[valid_rows.index] if volumes is not None else None,
        )

    return descriptor_frame


def _coerce_panel(panel: pd.DataFrame | None) -> pd.DataFrame:
    if panel is None:
        return panel
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("Input panels must be pandas DataFrames.")
    coerced = panel.sort_index().astype(float)
    coerced.index = pd.Index(coerced.index)
    return coerced


def _compute_window_descriptors(
    *,
    returns_window: pd.DataFrame,
    prices_window: pd.DataFrame | None,
    volumes_window: pd.DataFrame | None,
) -> dict[str, float]:
    daily_cross_sectional_std = returns_window.std(axis=1, ddof=0)
    equal_weight_return = returns_window.mean(axis=1)
    asset_vols = returns_window.std(axis=0, ddof=0)

    return {
        "avg_pairwise_corr": _avg_pairwise_corr(returns_window),
        "first_pc_share": _first_pc_share(returns_window),
        "cross_sectional_dispersion": _safe_mean(daily_cross_sectional_std),
        "trend_persistence": _mean_series_stat(returns_window, _lag1_autocorr),
        "vol_level": _safe_mean(asset_vols),
        "vol_of_vol": _safe_std(daily_cross_sectional_std),
        "liquidity_proxy": _liquidity_proxy(prices_window, volumes_window),
        "regime_switch_rate": _regime_switch_rate(equal_weight_return),
    }


def _avg_pairwise_corr(returns_window: pd.DataFrame) -> float:
    active = returns_window.dropna(axis=1, how="all")
    if active.shape[1] < 2:
        return 0.0
    corr = active.corr(min_periods=max(2, len(active) // 2))
    if corr.shape[0] < 2:
        return 0.0
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    values = corr.where(mask).stack().to_numpy()
    return _safe_mean(values) if values.size else 0.0


def _first_pc_share(returns_window: pd.DataFrame) -> float:
    active = returns_window.dropna(axis=1, how="all")
    if active.shape[1] < 2:
        return 1.0 if active.shape[1] == 1 else 0.0

    corr = active.corr(min_periods=max(2, len(active) // 2)).dropna(axis=0, how="all").dropna(axis=1, how="all")
    if corr.shape[0] < 2:
        return 1.0 if corr.shape[0] == 1 else 0.0
    corr_values = corr.to_numpy(dtype=float)
    corr_values = np.nan_to_num(corr_values, nan=0.0)
    corr_values = 0.5 * (corr_values + corr_values.T)
    np.fill_diagonal(corr_values, 1.0)

    eigenvalues = np.linalg.eigvalsh(corr_values)
    total = eigenvalues.sum()
    if total <= 0:
        return 1.0
    return float(eigenvalues[-1] / total)


def _lag1_autocorr(series: pd.Series) -> float:
    valid = series.dropna()
    if len(valid) < 3:
        return 0.0
    if np.isclose(valid.std(ddof=0), 0.0):
        return 0.0
    corr = valid.autocorr(lag=1)
    return 0.0 if pd.isna(corr) else float(corr)


def _mean_series_stat(panel: pd.DataFrame, stat_fn: Callable[[pd.Series], float]) -> float:
    values = [stat_fn(panel[column]) for column in panel.columns]
    return _safe_mean(values)


def _liquidity_proxy(prices_window: pd.DataFrame | None, volumes_window: pd.DataFrame | None) -> float:
    if volumes_window is None:
        return np.nan

    aligned_volume = volumes_window.dropna(axis=1, how="all")
    if aligned_volume.empty:
        return np.nan

    if prices_window is not None:
        aligned_prices = prices_window.reindex_like(aligned_volume)
        liquidity_panel = aligned_prices * aligned_volume
    else:
        liquidity_panel = aligned_volume

    median_liquidity = liquidity_panel.median(axis=0, skipna=True)
    if median_liquidity.dropna().empty:
        return np.nan
    return float(np.log1p(median_liquidity.mean()))


def _regime_switch_rate(equal_weight_return: pd.Series) -> float:
    valid = equal_weight_return.dropna()
    if len(valid) < 2:
        return np.nan
    signs = np.sign(valid.to_numpy())
    signs[signs == 0] = 1
    switches = signs[1:] != signs[:-1]
    return float(switches.mean())


def _safe_mean(values) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.nan
    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return np.nan
    return float(valid.mean())


def _safe_std(values) -> float:
    array = np.asarray(values, dtype=float)
    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return np.nan
    return float(valid.std(ddof=0))
