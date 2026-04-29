from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from context_study.strategies import (
    CMVTConfig,
    CMVTController,
    EWController,
    HierarchicalRiskController,
    MLMConfig,
    MLMController,
    MRHConfig,
    MRHController,
    MRSConfig,
    MRSController,
    RACConfig,
    RACController,
    ScreenedUniverseController,
    build_controllers,
)


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    periods = 180
    dates = pd.date_range("2020-01-01", periods=periods, freq="B")

    cluster_a = rng.normal(0.0006, 0.006, periods)
    cluster_b = rng.normal(0.0003, 0.010, periods)

    returns = pd.DataFrame(
        {
            "A1": cluster_a + rng.normal(0.0, 0.0015, periods),
            "A2": cluster_a + rng.normal(0.0, 0.0018, periods),
            "A3": cluster_a + rng.normal(0.0, 0.0012, periods),
            "B1": cluster_b + rng.normal(0.0, 0.0025, periods),
            "B2": cluster_b + rng.normal(0.0, 0.0028, periods),
            "B3": cluster_b + rng.normal(0.0, 0.0022, periods),
        },
        index=dates,
    )
    return 100 * (1 + returns).cumprod()


def test_hierarchical_risk_controller_returns_diversified_long_only_weights(
    synthetic_prices: pd.DataFrame,
) -> None:
    controller = HierarchicalRiskController(cluster_threshold=0.55, lookback=126)

    weights = controller.compute_weights(synthetic_prices)

    assert weights.shape == (synthetic_prices.shape[1],)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0.0)
    assert np.count_nonzero(weights > 1e-8) >= 4
    assert np.max(weights) < 0.4

    cluster_a_weight = weights[:3].sum()
    cluster_b_weight = weights[3:].sum()
    assert cluster_a_weight > 0.2
    assert cluster_b_weight > 0.2


def test_screened_universe_controller_limits_active_names_and_maps_back(
    synthetic_prices: pd.DataFrame,
) -> None:
    controller = ScreenedUniverseController(
        base_controller=EWController(),
        top_n=2,
        signal_lookback=20,
    )

    weights = controller.compute_weights(synthetic_prices)
    signal = synthetic_prices.iloc[-1] / synthetic_prices.iloc[-21] - 1.0
    selected_names = set(signal.nlargest(2).index)

    assert weights.shape == (synthetic_prices.shape[1],)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0.0)
    assert np.count_nonzero(weights > 1e-8) == 2

    for asset_name, weight in zip(synthetic_prices.columns, weights):
        if asset_name in selected_names:
            assert weight == pytest.approx(0.5)
        else:
            assert weight == pytest.approx(0.0)


def test_existing_controllers_still_produce_valid_weights(
    synthetic_prices: pd.DataFrame,
) -> None:
    controllers = [
        EWController(),
        MRHController(MRHConfig(lookback=21, top_k=3)),
        MRSController(MRSConfig(window=20, temperature=1.0)),
        CMVTController(CMVTConfig(lookback=63, risk_aversion=1.0, turnover_penalty=0.1)),
        RACController(RACConfig()),
        MLMController(MLMConfig(horizons=(21, 63), top_k=3)),
    ]
    current_weights = np.ones(synthetic_prices.shape[1]) / synthetic_prices.shape[1]

    for controller in controllers:
        weights = controller.compute_weights(synthetic_prices, current_weights=current_weights)
        assert weights.shape == (synthetic_prices.shape[1],)
        assert np.isclose(weights.sum(), 1.0)
        assert np.all(weights >= -1e-10), controller.name


def test_build_controllers_returns_runtime_controllers(synthetic_prices: pd.DataFrame) -> None:
    controllers = build_controllers()
    assert controllers
    current_weights = np.ones(synthetic_prices.shape[1]) / synthetic_prices.shape[1]
    for controller in controllers:
        assert hasattr(controller, 'name')
        assert hasattr(controller, 'compute_weights')
        weights = controller.compute_weights(synthetic_prices, current_weights=current_weights)
        assert weights.shape == (synthetic_prices.shape[1],)
        assert np.isclose(weights.sum(), 1.0)
