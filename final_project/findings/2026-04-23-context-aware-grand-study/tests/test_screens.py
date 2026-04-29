from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from context_study.screens import (
    SCREEN_OUTPUT_COLUMNS,
    apply_screen,
    cluster_capped_momentum_screen,
    liquidity_adjusted_momentum_screen,
    low_volatility_screen,
    momentum_screen,
    volatility_adjusted_momentum_screen,
)


def make_price_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=90)
    x = np.arange(len(dates), dtype=float)
    return pd.DataFrame(
        {
            "FAST": 100.0 + 1.20 * x,
            "MID": 100.0 + 0.65 * x,
            "SLOW": 100.0 + 0.15 * x,
            "CHOP": 100.0 + np.sin(x / 2.0) * 2.0,
            "DOWN": 120.0 - 0.25 * x,
        },
        index=dates,
    )


def make_volume_panel(prices: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "FAST": 2_000_000.0,
            "MID": 1_500_000.0,
            "SLOW": 500_000.0,
            "CHOP": 10_000_000.0,
            "DOWN": 8_000_000.0,
        },
        index=prices.index,
    )


def test_momentum_screen_uses_lagged_lookback_and_preserves_schema() -> None:
    prices = make_price_panel()

    selected = momentum_screen(prices, lookback=21, top_k=2, as_of_date=prices.index[-1])

    assert list(selected.columns) == SCREEN_OUTPUT_COLUMNS
    assert selected["symbol"].tolist() == ["FAST", "MID"]
    assert selected["rank"].tolist() == [1, 2]
    assert selected["screen_rule"].unique().tolist() == ["momentum_21_top2"]
    assert pd.Timestamp(selected["as_of_date"].iloc[0]) == prices.index[-1]
    assert pd.Timestamp(selected["lookback_start"].iloc[0]) < prices.index[-1]


def test_volatility_adjusted_momentum_prefers_smooth_trend() -> None:
    prices = make_price_panel()

    selected = volatility_adjusted_momentum_screen(prices, lookback=21, top_k=2)

    assert selected["symbol"].tolist()[0] in {"FAST", "MID", "SLOW"}
    assert "DOWN" not in selected["symbol"].tolist()
    assert "CHOP" not in selected["symbol"].tolist()
    assert selected["score"].is_monotonic_decreasing


def test_liquidity_adjusted_momentum_uses_volume_without_selecting_downtrend() -> None:
    prices = make_price_panel()
    volumes = make_volume_panel(prices)

    selected = liquidity_adjusted_momentum_screen(prices, volumes, lookback=21, top_k=3)

    assert len(selected) == 3
    assert "DOWN" not in selected["symbol"].tolist()
    assert selected["screen_rule"].unique().tolist() == ["liquidity_adjusted_momentum_21_top3"]


def test_low_volatility_screen_selects_lowest_realized_vol_assets() -> None:
    prices = make_price_panel()

    selected = low_volatility_screen(prices, lookback=30, top_k=2)

    assert len(selected) == 2
    assert selected["score"].is_monotonic_decreasing
    assert set(selected["symbol"]).issubset(set(prices.columns))


def test_cluster_capped_momentum_limits_near_duplicate_cluster_members() -> None:
    dates = pd.bdate_range("2022-01-03", periods=90)
    x = np.arange(len(dates), dtype=float)
    prices = pd.DataFrame(
        {
            "TREND_A": 100 + 1.0 * x,
            "TREND_B": 101 + 1.02 * x,
            "TREND_C": 99 + 0.98 * x,
            "DIVERSIFIER": 100 + np.sin(x / 4.0) * 4 + 0.20 * x,
        },
        index=dates,
    )

    selected = cluster_capped_momentum_screen(
        prices,
        lookback=63,
        top_k=2,
        cluster_threshold=0.95,
        max_per_cluster=1,
    )

    trend_count = selected["symbol"].isin(["TREND_A", "TREND_B", "TREND_C"]).sum()
    assert trend_count == 1
    assert len(selected) == 2


def test_apply_screen_dispatches_rules_and_rejects_missing_volume() -> None:
    prices = make_price_panel()

    selected = apply_screen(prices, screen_rule="momentum_21_top2")
    assert selected["symbol"].tolist() == ["FAST", "MID"]

    try:
        apply_screen(prices, screen_rule="liquidity_adjusted_momentum_21_top3")
    except ValueError as exc:
        assert "volume_panel is required" in str(exc)
    else:
        raise AssertionError("liquidity-adjusted screen should require a volume panel")


def test_empty_screen_result_preserves_schema_when_history_is_insufficient() -> None:
    prices = make_price_panel().iloc[:5]

    selected = momentum_screen(prices, lookback=21, top_k=3)

    assert selected.empty
    assert list(selected.columns) == SCREEN_OUTPUT_COLUMNS
