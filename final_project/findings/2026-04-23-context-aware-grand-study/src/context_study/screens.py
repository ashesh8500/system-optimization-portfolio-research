from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

SCREEN_OUTPUT_COLUMNS = [
    "as_of_date",
    "symbol",
    "rank",
    "score",
    "screen_rule",
    "lookback_start",
    "lookback_end",
]


@dataclass(frozen=True)
class ScreenSpec:
    screen_rule: str
    lookback: int
    top_k: int


def _empty_screen() -> pd.DataFrame:
    return pd.DataFrame(columns=SCREEN_OUTPUT_COLUMNS)


def _validate_prices(price_panel: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(price_panel, pd.DataFrame):
        raise TypeError("price_panel must be a pandas DataFrame.")
    if price_panel.empty:
        return price_panel.copy()
    if not isinstance(price_panel.index, pd.DatetimeIndex):
        raise TypeError("price_panel index must be a pandas DatetimeIndex.")
    return price_panel.sort_index().astype(float)


def _window(price_panel: pd.DataFrame, lookback: int, as_of_date: pd.Timestamp | str | None) -> tuple[pd.DataFrame, pd.Timestamp]:
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    prices = _validate_prices(price_panel)
    if prices.empty:
        return prices, pd.Timestamp(as_of_date) if as_of_date is not None else pd.NaT
    as_of = pd.Timestamp(as_of_date) if as_of_date is not None else pd.Timestamp(prices.index[-1])
    prices = prices.loc[prices.index <= as_of]
    if len(prices) <= lookback:
        return prices.iloc[0:0], as_of
    return prices.iloc[-(lookback + 1):], as_of


def _ranked_output(
    *,
    scores: pd.Series,
    top_k: int,
    screen_rule: str,
    as_of_date: pd.Timestamp,
    lookback_start: pd.Timestamp,
    lookback_end: pd.Timestamp,
) -> pd.DataFrame:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    clean_scores = pd.to_numeric(scores, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean_scores.empty:
        return _empty_screen()
    ranked = clean_scores.sort_values(ascending=False).head(top_k)
    records = [
        {
            "as_of_date": as_of_date,
            "symbol": symbol,
            "rank": rank,
            "score": float(score),
            "screen_rule": screen_rule,
            "lookback_start": lookback_start,
            "lookback_end": lookback_end,
        }
        for rank, (symbol, score) in enumerate(ranked.items(), start=1)
    ]
    return pd.DataFrame.from_records(records, columns=SCREEN_OUTPUT_COLUMNS)


def momentum_screen(
    price_panel: pd.DataFrame,
    *,
    lookback: int,
    top_k: int,
    as_of_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    prices, as_of = _window(price_panel, lookback, as_of_date)
    if prices.empty:
        return _empty_screen()
    scores = prices.iloc[-1] / prices.iloc[0] - 1.0
    return _ranked_output(
        scores=scores,
        top_k=top_k,
        screen_rule=f"momentum_{lookback}_top{top_k}",
        as_of_date=as_of,
        lookback_start=pd.Timestamp(prices.index[0]),
        lookback_end=pd.Timestamp(prices.index[-1]),
    )


def volatility_adjusted_momentum_screen(
    price_panel: pd.DataFrame,
    *,
    lookback: int,
    top_k: int,
    as_of_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    prices, as_of = _window(price_panel, lookback, as_of_date)
    if prices.empty:
        return _empty_screen()
    returns = prices.pct_change(fill_method=None).dropna()
    momentum = prices.iloc[-1] / prices.iloc[0] - 1.0
    realized_vol = returns.std().replace(0.0, np.nan)
    scores = momentum / realized_vol
    return _ranked_output(
        scores=scores,
        top_k=top_k,
        screen_rule=f"vol_adjusted_momentum_{lookback}_top{top_k}",
        as_of_date=as_of,
        lookback_start=pd.Timestamp(prices.index[0]),
        lookback_end=pd.Timestamp(prices.index[-1]),
    )


def liquidity_adjusted_momentum_screen(
    price_panel: pd.DataFrame,
    volume_panel: pd.DataFrame,
    *,
    lookback: int,
    top_k: int,
    as_of_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    if volume_panel is None:
        raise ValueError("volume_panel is required for liquidity-adjusted screening.")
    prices, as_of = _window(price_panel, lookback, as_of_date)
    volumes = _validate_prices(volume_panel)
    if prices.empty:
        return _empty_screen()
    volumes = volumes.reindex(prices.index).loc[:, prices.columns]
    dollar_volume = (prices * volumes).mean()
    liquidity_rank = dollar_volume.rank(pct=True)
    momentum = prices.iloc[-1] / prices.iloc[0] - 1.0
    positive_momentum = momentum.where(momentum > 0.0, np.nan)
    scores = positive_momentum * liquidity_rank
    return _ranked_output(
        scores=scores,
        top_k=top_k,
        screen_rule=f"liquidity_adjusted_momentum_{lookback}_top{top_k}",
        as_of_date=as_of,
        lookback_start=pd.Timestamp(prices.index[0]),
        lookback_end=pd.Timestamp(prices.index[-1]),
    )


def low_volatility_screen(
    price_panel: pd.DataFrame,
    *,
    lookback: int,
    top_k: int,
    as_of_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    prices, as_of = _window(price_panel, lookback, as_of_date)
    if prices.empty:
        return _empty_screen()
    realized_vol = prices.pct_change(fill_method=None).dropna().std()
    scores = -realized_vol
    return _ranked_output(
        scores=scores,
        top_k=top_k,
        screen_rule=f"low_volatility_{lookback}_top{top_k}",
        as_of_date=as_of,
        lookback_start=pd.Timestamp(prices.index[0]),
        lookback_end=pd.Timestamp(prices.index[-1]),
    )


def _correlation_clusters(corr: pd.DataFrame, threshold: float) -> dict[str, int]:
    remaining = list(corr.columns)
    cluster_ids: dict[str, int] = {}
    cluster_id = 0
    while remaining:
        seed = remaining.pop(0)
        queue = [seed]
        cluster_ids[seed] = cluster_id
        while queue:
            current = queue.pop(0)
            connected = [asset for asset in list(remaining) if corr.loc[current, asset] >= threshold]
            for asset in connected:
                remaining.remove(asset)
                cluster_ids[asset] = cluster_id
                queue.append(asset)
        cluster_id += 1
    return cluster_ids


def cluster_capped_momentum_screen(
    price_panel: pd.DataFrame,
    *,
    lookback: int,
    top_k: int,
    cluster_threshold: float = 0.9,
    max_per_cluster: int = 1,
    as_of_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    if max_per_cluster < 1:
        raise ValueError("max_per_cluster must be >= 1")
    prices, as_of = _window(price_panel, lookback, as_of_date)
    if prices.empty:
        return _empty_screen()
    returns = prices.pct_change(fill_method=None).dropna()
    if returns.empty:
        return _empty_screen()
    corr = returns.corr().fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)
    clusters = _correlation_clusters(corr, cluster_threshold)
    momentum = (prices.iloc[-1] / prices.iloc[0] - 1.0).sort_values(ascending=False)
    selected: list[tuple[str, float]] = []
    counts: dict[int, int] = {}
    for symbol, score in momentum.items():
        if not np.isfinite(score):
            continue
        cluster_id = clusters.get(symbol, -1)
        if counts.get(cluster_id, 0) >= max_per_cluster:
            continue
        selected.append((symbol, float(score)))
        counts[cluster_id] = counts.get(cluster_id, 0) + 1
        if len(selected) >= top_k:
            break
    scores = pd.Series({symbol: score for symbol, score in selected})
    return _ranked_output(
        scores=scores,
        top_k=top_k,
        screen_rule=f"cluster_capped_momentum_{lookback}_top{top_k}",
        as_of_date=as_of,
        lookback_start=pd.Timestamp(prices.index[0]),
        lookback_end=pd.Timestamp(prices.index[-1]),
    )


def _parse_rule(screen_rule: str) -> ScreenSpec:
    patterns = [
        (r"^momentum_(\d+)_top(\d+)$", "momentum"),
        (r"^vol_adjusted_momentum_(\d+)_top(\d+)$", "vol_adjusted_momentum"),
        (r"^liquidity_adjusted_momentum_(\d+)_top(\d+)$", "liquidity_adjusted_momentum"),
        (r"^low_volatility_(\d+)_top(\d+)$", "low_volatility"),
        (r"^cluster_capped_momentum_(\d+)_top(\d+)$", "cluster_capped_momentum"),
    ]
    for pattern, canonical in patterns:
        match = re.match(pattern, screen_rule)
        if match:
            return ScreenSpec(screen_rule=canonical, lookback=int(match.group(1)), top_k=int(match.group(2)))
    raise ValueError(f"unsupported screen_rule={screen_rule!r}")


def apply_screen(
    price_panel: pd.DataFrame,
    *,
    screen_rule: str,
    volume_panel: pd.DataFrame | None = None,
    as_of_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    if screen_rule in {"none", "all"}:
        prices = _validate_prices(price_panel)
        if prices.empty:
            return _empty_screen()
        as_of = pd.Timestamp(as_of_date) if as_of_date is not None else pd.Timestamp(prices.index[-1])
        scores = pd.Series(1.0, index=prices.columns)
        return _ranked_output(
            scores=scores,
            top_k=len(scores),
            screen_rule="none",
            as_of_date=as_of,
            lookback_start=pd.Timestamp(prices.index[0]),
            lookback_end=pd.Timestamp(prices.loc[prices.index <= as_of].index[-1]),
        )

    spec = _parse_rule(screen_rule)
    if spec.screen_rule == "momentum":
        return momentum_screen(price_panel, lookback=spec.lookback, top_k=spec.top_k, as_of_date=as_of_date)
    if spec.screen_rule == "vol_adjusted_momentum":
        return volatility_adjusted_momentum_screen(price_panel, lookback=spec.lookback, top_k=spec.top_k, as_of_date=as_of_date)
    if spec.screen_rule == "liquidity_adjusted_momentum":
        if volume_panel is None:
            raise ValueError("volume_panel is required for liquidity-adjusted screening.")
        return liquidity_adjusted_momentum_screen(price_panel, volume_panel, lookback=spec.lookback, top_k=spec.top_k, as_of_date=as_of_date)
    if spec.screen_rule == "low_volatility":
        return low_volatility_screen(price_panel, lookback=spec.lookback, top_k=spec.top_k, as_of_date=as_of_date)
    if spec.screen_rule == "cluster_capped_momentum":
        return cluster_capped_momentum_screen(price_panel, lookback=spec.lookback, top_k=spec.top_k, as_of_date=as_of_date)
    raise ValueError(f"unsupported screen_rule={screen_rule!r}")
