"""Event-based backtest engine with transaction cost modeling."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    transaction_cost_bps: float = 10.0  # one-way
    rebalance_freq: int = 21  # trading days


@dataclass
class BacktestResult:
    controller_name: str
    universe: str
    equity_curve: pd.Series
    returns: pd.Series
    weights_history: pd.DataFrame
    trades: list[dict]
    annualized_return: float
    annualized_vol: float
    sharpe: float
    max_drawdown: float
    calmar: float
    total_turnover: float
    win_rate: float
    num_trades: int


def run_backtest(
    prices: pd.DataFrame,
    controller: object,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a full backtest for a single controller.

    The controller must implement::

        controller.compute_weights(prices_slice, current_weights) -> np.ndarray
    """
    cfg = config or BacktestConfig()
    dates = prices.index
    n_assets = len(prices.columns)

    capital = cfg.initial_capital
    current_weights = np.ones(n_assets) / n_assets
    equity = [capital]
    weights_hist = [current_weights.copy()]
    trades = []
    total_turnover = 0.0
    wins = 0

    # Warm-up period (need enough history)
    warmup = max(getattr(controller, "cfg", cfg) and getattr(getattr(controller, "cfg", cfg), "lookback", 63) or 63, 252)
    start_idx = min(warmup, len(prices) // 4)

    for i in range(start_idx, len(dates)):
        date = dates[i]
        price_today = prices.iloc[i].values
        price_yesterday = prices.iloc[i - 1].values if i > 0 else price_today

        # Compute portfolio return from yesterday's weights
        asset_returns = price_today / (price_yesterday + 1e-12) - 1.0
        port_return = float(current_weights @ asset_returns)
        capital *= (1.0 + port_return)
        equity.append(capital)

        # Rebalance check
        freq = getattr(controller, "cfg", cfg) and getattr(getattr(controller, "cfg", cfg), "rebalance_freq", cfg.rebalance_freq) or cfg.rebalance_freq
        if (i - start_idx) % freq == 0 and i > start_idx:
            prices_slice = prices.iloc[: i + 1]
            target_weights = controller.compute_weights(prices_slice, current_weights)
            target_weights = np.asarray(target_weights, dtype=float)
            target_weights = np.maximum(target_weights, 0.0)
            target_weights /= np.sum(target_weights) + 1e-12

            # Turnover cost
            turnover = np.sum(np.abs(target_weights - current_weights))
            cost = turnover * (cfg.transaction_cost_bps / 10_000)
            capital *= (1.0 - cost)
            total_turnover += turnover

            # Track trade statistics
            if turnover > 0.001:
                trades.append({
                    "date": date,
                    "turnover": turnover,
                    "cost": cost,
                })
                if port_return > 0:
                    wins += 1

            current_weights = target_weights

        weights_hist.append(current_weights.copy())

    equity_series = pd.Series(equity, index=dates[start_idx - 1:])
    returns_series = equity_series.pct_change(fill_method=None).dropna()

    # Metrics
    ann_ret = returns_series.mean() * 252
    ann_vol = returns_series.std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-12)

    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_dd = drawdown.min()
    calmar = ann_ret / (abs(max_dd) + 1e-12)

    win_rate = wins / len(trades) if trades else 0.0

    return BacktestResult(
        controller_name=getattr(controller, "name", controller.__class__.__name__),
        universe="",
        equity_curve=equity_series,
        returns=returns_series,
        weights_history=pd.DataFrame(weights_hist, index=dates[start_idx - 1:], columns=prices.columns),
        trades=trades,
        annualized_return=ann_ret,
        annualized_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        calmar=calmar,
        total_turnover=total_turnover,
        win_rate=win_rate,
        num_trades=len(trades),
    )
