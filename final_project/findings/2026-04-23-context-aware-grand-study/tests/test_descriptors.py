from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from context_study.descriptors import compute_universe_descriptors


EXPECTED_COLUMNS = [
    "avg_pairwise_corr",
    "first_pc_share",
    "cross_sectional_dispersion",
    "trend_persistence",
    "vol_level",
    "vol_of_vol",
    "liquidity_proxy",
    "regime_switch_rate",
]


def make_synthetic_inputs(periods: int = 80, assets: int = 5):
    rng = np.random.default_rng(12345)
    dates = pd.date_range("2020-01-01", periods=periods, freq="B")
    columns = [f"asset_{idx}" for idx in range(assets)]

    market = rng.normal(0.0005, 0.01, size=periods)
    idio = rng.normal(0.0, 0.008, size=(periods, assets))
    returns = pd.DataFrame(0.5 * market[:, None] + idio, index=dates, columns=columns)

    prices = 100 * (1.0 + returns).cumprod()
    volumes = pd.DataFrame(
        rng.integers(100_000, 900_000, size=(periods, assets)),
        index=dates,
        columns=columns,
    )
    return prices, returns, volumes


def test_compute_universe_descriptors_has_expected_schema():
    prices, returns, volumes = make_synthetic_inputs()

    descriptors = compute_universe_descriptors(
        price_panel=prices,
        return_panel=returns,
        volume_panel=volumes,
        window=20,
    )

    assert list(descriptors.columns) == EXPECTED_COLUMNS
    assert descriptors.index.equals(returns.index)


def test_compute_universe_descriptors_is_finite_after_warmup():
    prices, returns, volumes = make_synthetic_inputs()

    descriptors = compute_universe_descriptors(
        price_panel=prices,
        return_panel=returns,
        volume_panel=volumes,
        window=20,
    )

    post_warmup = descriptors.iloc[19:]
    assert not post_warmup.empty
    assert np.isfinite(post_warmup.to_numpy()).all()


def test_compute_universe_descriptors_gracefully_handles_missing_liquidity():
    prices, returns, _ = make_synthetic_inputs()

    descriptors = compute_universe_descriptors(
        price_panel=prices,
        return_panel=returns,
        volume_panel=None,
        window=20,
    )

    assert "liquidity_proxy" in descriptors.columns
    assert descriptors["liquidity_proxy"].isna().all()

    non_liquidity = descriptors.drop(columns=["liquidity_proxy"]).iloc[19:]
    assert np.isfinite(non_liquidity.to_numpy()).all()


def test_compute_universe_descriptors_handles_constant_returns_without_nan():
    dates = pd.date_range("2020-01-01", periods=40, freq="B")
    returns = pd.DataFrame(0.0, index=dates, columns=["a", "b", "c"])
    prices = 100 * (1.0 + returns).cumprod()

    descriptors = compute_universe_descriptors(
        price_panel=prices,
        return_panel=returns,
        volume_panel=None,
        window=10,
    )

    post_warmup = descriptors.iloc[9:]
    assert np.isfinite(post_warmup.drop(columns=["liquidity_proxy"]).to_numpy()).all()


def test_regime_switch_rate_rises_for_alternating_signs():
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    alternating = np.array([0.01 if i % 2 == 0 else -0.01 for i in range(20)])
    returns = pd.DataFrame(
        {
            "a": alternating,
            "b": alternating,
            "c": alternating,
        },
        index=dates,
    )

    descriptors = compute_universe_descriptors(return_panel=returns, window=10)
    assert descriptors["regime_switch_rate"].iloc[-1] > 0.8
