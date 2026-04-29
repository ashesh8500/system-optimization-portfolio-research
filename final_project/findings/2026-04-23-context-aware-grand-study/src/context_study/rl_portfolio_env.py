"""PufferLib-compatible portfolio allocation environment.

The agent learns a policy pi(s) -> a that maps market state to portfolio
weights.  The environment is designed to be vectorized via PufferLib so
that many parallel market trajectories can be simulated simultaneously.

State features (per asset, normalized):
- log-price relative to 20-day moving average
- recent return (5-day)
- volatility (20-day std of returns)
- current portfolio weight

Action: continuous target weights (softmaxed to simplex)

Reward: risk-adjusted return minus turnover penalty
"""
from __future__ import annotations

import numpy as np


class PortfolioEnv:
    """Single-vectorized-step portfolio environment compatible with PufferLib."""

    def __init__(
        self,
        returns: np.ndarray,          # (T, n_assets)
        prices: np.ndarray | None = None,
        rebalance_every: int = 21,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.0,
    ) -> None:
        self.returns = returns
        self.prices = prices
        self.n_assets = returns.shape[1]
        self.T = returns.shape[0]
        self.rebalance_every = rebalance_every
        self.tc = transaction_cost
        self.rf = risk_free_rate

        # Feature dimension: 4 per asset + 1 global (time-to-rebalance)
        self.obs_size = self.n_assets * 4 + 1
        self.act_size = self.n_assets

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.t = self.rebalance_every * 2  # warm-up
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.port_value = 1.0
        return self._observe()

    def _observe(self) -> np.ndarray:
        t = self.t
        r = self.returns
        n = self.n_assets

        # Need at least 20 days of history
        start = max(0, t - 20)
        hist = r[start:t]

        # Features per asset
        if len(hist) > 0:
            ret_5d = hist[-5:].mean(axis=0) if len(hist) >= 5 else hist.mean(axis=0)
            vol_20d = hist.std(axis=0) + 1e-8
            # Price feature (if prices available, else use cumulative return)
            if self.prices is not None:
                p = self.prices[start:t + 1]
                mu = p.mean(axis=0) + 1e-8
                price_feat = (p[-1] - mu) / mu
            else:
                cumret = np.cumprod(1 + hist, axis=0)[-1] - 1.0
                price_feat = cumret
        else:
            ret_5d = np.zeros(n)
            vol_20d = np.ones(n)
            price_feat = np.zeros(n)

        feat = np.concatenate([
            price_feat,
            ret_5d,
            vol_20d,
            self.weights,
            [(t % self.rebalance_every) / self.rebalance_every],
        ])
        return feat.astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step."""
        t = self.t
        if t >= self.T - 1:
            return self._observe(), 0.0, True, False, {}

        # Action -> target weights via softmax
        target = np.exp(action - np.max(action))
        target = target / (np.sum(target) + 1e-12)

        # Market return
        asset_ret = self.returns[t + 1]
        port_ret_before = float(self.weights @ asset_ret)
        self.port_value *= (1.0 + port_ret_before)

        # Rebalance if needed
        if (t % self.rebalance_every) == 0:
            turnover = np.sum(np.abs(target - self.weights))
            tc_cost = turnover * self.tc
            self.port_value *= (1.0 - tc_cost)
            self.weights = target
        else:
            # Let weights drift with market
            new_w = self.weights * (1.0 + asset_ret)
            new_w = new_w / (np.sum(new_w) + 1e-12)
            self.weights = new_w

        self.t += 1

        # Reward: daily return - small volatility penalty
        reward = port_ret_before - 0.1 * (port_ret_before ** 2)
        done = self.t >= self.T - 1

        return self._observe(), float(reward), done, False, {}


def make_env_creator(returns: np.ndarray, prices: np.ndarray | None = None):
    """Factory for PufferLib env_creator."""
    def env_creator():
        return PortfolioEnv(returns=returns, prices=prices)
    return env_creator
