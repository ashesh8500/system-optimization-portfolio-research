"""Portfolio allocation strategies described by their mathematical mechanics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    exp = np.exp(x / temperature)
    return exp / (np.sum(exp) + 1e-12)


def compute_covariance(returns: pd.DataFrame, lookback: int = 63) -> pd.DataFrame:
    sub = returns.iloc[-lookback:]
    return sub.cov() * 252


def sharpe_from_returns(returns: pd.Series) -> float:
    ann = returns.mean() * 252
    vol = returns.std() * np.sqrt(252)
    return ann / (vol + 1e-12)


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, 0.0)
    total = np.sum(weights)
    if total <= 1e-12:
        return np.ones(len(weights)) / max(len(weights), 1)
    return weights / total


# ---------------------------------------------------------------------------
# 1. Momentum Rank-and-Hold (MRH)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MRHConfig:
    """Momentum Rank-and-Hold controller.

    At each rebalance, rank assets by lookback return r_i(t) = P_i(t)/P_i(t-tau) - 1,
    hold the top-k assets with equal weight.
    """

    lookback: int = 63
    top_k: int = 5
    rebalance_freq: int = 21


class MRHController:
    def __init__(self, config: MRHConfig | None = None) -> None:
        self.cfg = config or MRHConfig()
        self.name = f"MRH_tau{self.cfg.lookback}_k{self.cfg.top_k}"

    def compute_weights(self, prices: pd.DataFrame, current_weights: np.ndarray | None = None) -> np.ndarray:
        lb = min(self.cfg.lookback, len(prices) - 1)
        momentum = prices.iloc[-1] / prices.iloc[-lb - 1] - 1.0
        ranked = momentum.sort_values(ascending=False)
        top = ranked.index[: self.cfg.top_k]
        w = pd.Series(0.0, index=prices.columns)
        w.loc[top] = 1.0 / self.cfg.top_k
        return w.values


# ---------------------------------------------------------------------------
# 2. Mean-Reversion Scoring (MRS)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MRSConfig:
    """Mean-reversion controller based on Bollinger-band position.

    z_i(t) = (P_i(t) - mu_i(t,window)) / sigma_i(t,window)
    Score s_i(t) = -z_i(t)  (oversold is attractive)
    Weights via softmax: w_i propto exp(s_i(t) / T)
    """

    window: int = 20
    temperature: float = 1.0
    rebalance_freq: int = 21


class MRSController:
    def __init__(self, config: MRSConfig | None = None) -> None:
        self.cfg = config or MRSConfig()
        self.name = f"MRS_w{self.cfg.window}_T{self.cfg.temperature}"

    def compute_weights(self, prices: pd.DataFrame, current_weights: np.ndarray | None = None) -> np.ndarray:
        w = min(self.cfg.window, len(prices))
        mu = prices.iloc[-w:].mean()
        sigma = prices.iloc[-w:].std() + 1e-12
        z = (prices.iloc[-1] - mu) / sigma
        scores = -z.values
        return softmax(scores, self.cfg.temperature)


# ---------------------------------------------------------------------------
# 3. Convex Mean-Variance with Turnover (CMVT)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CMVTConfig:
    """Convex mean-variance controller with turnover penalty.

    max_w  alpha^T w - lambda w^T Sigma w - gamma ||w - w_prev||_1
    s.t.   sum(w) = 1, w >= 0

    alpha is forecasted from recent momentum signals.
    """

    lookback: int = 63
    risk_aversion: float = 1.0
    turnover_penalty: float = 0.5
    rebalance_freq: int = 21


class CMVTController:
    def __init__(self, config: CMVTConfig | None = None) -> None:
        self.cfg = config or CMVTConfig()
        self.name = f"CMVT_lam{self.cfg.risk_aversion}_gam{self.cfg.turnover_penalty}"

    def compute_weights(self, prices: pd.DataFrame, current_weights: np.ndarray | None) -> np.ndarray:
        n = len(prices.columns)
        returns = prices.pct_change(fill_method=None).dropna()
        if len(returns) < 10:
            return np.ones(n) / n

        # Expected returns from recent momentum
        lb = min(self.cfg.lookback, len(returns))
        alpha = returns.iloc[-lb:].mean().values * 252

        # Covariance
        Sigma = compute_covariance(returns, lookback=min(63, len(returns))).values

        # Previous weights
        w_prev = current_weights if current_weights is not None else np.ones(n) / n

        # Objective
        def objective(w):
            return -alpha @ w + self.cfg.risk_aversion * (w @ Sigma @ w) + self.cfg.turnover_penalty * np.sum(np.abs(w - w_prev))

        # Constraints
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(n)]
        x0 = np.ones(n) / n

        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
        w = np.maximum(result.x, 0.0)
        return w / (np.sum(w) + 1e-12)


# ---------------------------------------------------------------------------
# 4. Regime-Adaptive Composite (RAC)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RACConfig:
    """Regime-adaptive controller.

    Detects market regime via trend strength and volatility:
    - High trend + low vol -> trend-following regime
    - Low trend + high vol -> mean-reversion regime
    - Otherwise -> neutral

    Blends momentum and mean-reversion scores accordingly.
    """

    trend_window: int = 63
    vol_window: int = 20
    rebalance_freq: int = 21


class RACController:
    def __init__(self, config: RACConfig | None = None) -> None:
        self.cfg = config or RACConfig()
        self.name = "RAC"

    def _detect_regime(self, prices: pd.DataFrame) -> str:
        returns = prices.pct_change(fill_method=None).dropna()
        if len(returns) < self.cfg.trend_window:
            return "neutral"

        # Trend strength: annualized Sharpe of market (equal-weight proxy)
        market_ret = returns.mean(axis=1)
        trend_sharpe = sharpe_from_returns(market_ret.iloc[-self.cfg.trend_window:])

        # Volatility regime
        recent_vol = market_ret.iloc[-self.cfg.vol_window:].std() * np.sqrt(252)

        if trend_sharpe > 0.8 and recent_vol < 0.15:
            return "trending"
        elif trend_sharpe < 0.2 and recent_vol > 0.25:
            return "mean_reverting"
        return "neutral"

    def compute_weights(self, prices: pd.DataFrame, current_weights: np.ndarray | None = None) -> np.ndarray:
        regime = self._detect_regime(prices)
        n = len(prices.columns)

        if regime == "trending":
            # Momentum logic
            lb = min(63, len(prices) - 1)
            mom = prices.iloc[-1] / prices.iloc[-lb - 1] - 1.0
            return softmax(mom.values, temperature=0.5)
        elif regime == "mean_reverting":
            # Mean-reversion logic
            w = min(20, len(prices))
            mu = prices.iloc[-w:].mean()
            sigma = prices.iloc[-w:].std() + 1e-12
            z = (prices.iloc[-1] - mu) / sigma
            scores = -z.values
            return softmax(scores, temperature=1.0)
        else:
            # Neutral: equal weight blend of both signals
            lb = min(63, len(prices) - 1)
            mom = prices.iloc[-1] / prices.iloc[-lb - 1] - 1.0
            w = min(20, len(prices))
            mu = prices.iloc[-w:].mean()
            sigma = prices.iloc[-w:].std() + 1e-12
            z = (prices.iloc[-1] - mu) / sigma
            score = 0.5 * mom.values - 0.5 * z.values
            return softmax(score, temperature=1.0)


# ---------------------------------------------------------------------------
# 5. Meta-Learning Momentum (MLM)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MLMConfig:
    """Meta-learning momentum controller.

    Maintains K momentum experts with lookback windows tau_k.
    Each expert's weight is updated by inverse prediction error.
    Final score = sum_k beta_k * expert_k_score
    """

    horizons: tuple[int, ...] = (21, 63, 126, 252)
    top_k: int = 5
    memory: float = 0.9
    rebalance_freq: int = 21


class MLMController:
    def __init__(self, config: MLMConfig | None = None) -> None:
        self.cfg = config or MLMConfig()
        self.name = f"MLM_K{len(self.cfg.horizons)}"
        self._expert_errors = None

    def compute_weights(self, prices: pd.DataFrame, current_weights: np.ndarray | None = None) -> np.ndarray:
        n = len(prices.columns)
        K = len(self.cfg.horizons)

        if self._expert_errors is None:
            self._expert_errors = np.ones(K)

        expert_scores = []
        for tau in self.cfg.horizons:
            lb = min(tau, len(prices) - 1)
            if lb < 5:
                expert_scores.append(np.zeros(n))
                continue
            mom = prices.iloc[-1] / prices.iloc[-lb - 1] - 1.0
            expert_scores.append(mom.values)

        # Update errors using recent predictive performance
        if len(prices) > max(self.cfg.horizons) * 2 + 5:
            for idx, tau in enumerate(self.cfg.horizons):
                if len(prices) > 2 * tau + 5:
                    pred = prices.iloc[-tau - 1] / prices.iloc[-2 * tau - 1] - 1.0
                    realized = prices.iloc[-1] / prices.iloc[-tau - 1] - 1.0
                    err = np.mean((pred - realized) ** 2)
                    self._expert_errors[idx] = self.cfg.memory * self._expert_errors[idx] + (1 - self.cfg.memory) * err

        # Weight experts by inverse error
        inv_err = 1.0 / (self._expert_errors + 1e-12)
        beta = inv_err / (np.sum(inv_err) + 1e-12)

        # Composite score
        score = np.zeros(n)
        for idx in range(K):
            score += beta[idx] * expert_scores[idx]

        # Rank and hold top-k
        ranked_idx = np.argsort(score)[::-1][: self.cfg.top_k]
        w = np.zeros(n)
        w[ranked_idx] = 1.0 / self.cfg.top_k
        return w


# ---------------------------------------------------------------------------
# 6. Hierarchical Risk Baseline (HRB)
# ---------------------------------------------------------------------------
class HierarchicalRiskController:
    """Lightweight hierarchical risk allocation via simple correlation clusters."""

    def __init__(self, lookback: int = 126, cluster_threshold: float = 0.5) -> None:
        self.lookback = lookback
        self.cluster_threshold = cluster_threshold
        self.name = f"HRB_lb{lookback}_ct{cluster_threshold}"

    def _cluster_assets(self, corr: pd.DataFrame) -> list[list[str]]:
        remaining = list(corr.columns)
        clusters: list[list[str]] = []

        while remaining:
            seed = remaining.pop(0)
            cluster = [seed]
            queue = [seed]
            while queue:
                current = queue.pop(0)
                connected = [
                    asset
                    for asset in list(remaining)
                    if corr.loc[current, asset] >= self.cluster_threshold
                ]
                for asset in connected:
                    remaining.remove(asset)
                    cluster.append(asset)
                    queue.append(asset)
            clusters.append(cluster)
        return clusters

    def compute_weights(self, prices: pd.DataFrame, current_weights: np.ndarray | None = None) -> np.ndarray:
        n = len(prices.columns)
        returns = prices.pct_change(fill_method=None).dropna()
        if len(returns) < 5:
            return np.ones(n) / n

        window = returns.iloc[-min(self.lookback, len(returns)):]
        vol = window.std().clip(lower=1e-6)
        corr = window.corr().fillna(0.0)
        np.fill_diagonal(corr.values, 1.0)
        clusters = self._cluster_assets(corr)

        cluster_scores = []
        intra_cluster_weights: list[pd.Series] = []
        for cluster in clusters:
            cluster_vol = vol.loc[cluster]
            intra = (1.0 / cluster_vol).astype(float)
            intra = intra / (intra.sum() + 1e-12)
            intra_cluster_weights.append(intra)
            cluster_returns = window[cluster].mul(intra, axis=1).sum(axis=1)
            cluster_scores.append(1.0 / max(cluster_returns.std(), 1e-6))

        cluster_allocations = normalize_weights(np.asarray(cluster_scores, dtype=float))
        weights = pd.Series(0.0, index=prices.columns, dtype=float)
        for cluster, intra, cluster_weight in zip(clusters, intra_cluster_weights, cluster_allocations):
            weights.loc[cluster] = cluster_weight * intra

        return normalize_weights(weights.values)


# ---------------------------------------------------------------------------
# 7. Screened Universe Wrapper
# ---------------------------------------------------------------------------
class ScreenedUniverseController:
    """Reduce to a top-N signal subset, delegate weighting, then map back."""

    def __init__(
        self,
        base_controller,
        top_n: int,
        signal_lookback: int = 21,
        signal_fn: Callable[[pd.DataFrame], pd.Series] | None = None,
    ) -> None:
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        self.base_controller = base_controller
        self.top_n = top_n
        self.signal_lookback = signal_lookback
        self.signal_fn = signal_fn
        self.name = f"Screened_{getattr(base_controller, 'name', type(base_controller).__name__)}_top{top_n}"

    def _compute_signal(self, prices: pd.DataFrame) -> pd.Series:
        if self.signal_fn is not None:
            signal = self.signal_fn(prices)
            return pd.Series(signal, index=prices.columns, dtype=float)
        lookback = min(self.signal_lookback, len(prices) - 1)
        if lookback <= 0:
            return pd.Series(1.0, index=prices.columns, dtype=float)
        return (prices.iloc[-1] / prices.iloc[-lookback - 1] - 1.0).astype(float)

    def compute_weights(self, prices: pd.DataFrame, current_weights: np.ndarray | None = None) -> np.ndarray:
        n_assets = len(prices.columns)
        if self.top_n >= n_assets:
            delegated = self.base_controller.compute_weights(prices, current_weights=current_weights)
            return normalize_weights(np.asarray(delegated, dtype=float))

        signal = self._compute_signal(prices)
        selected = list(signal.nlargest(self.top_n).index)
        screened_prices = prices[selected]

        screened_current_weights = None
        if current_weights is not None:
            current = pd.Series(current_weights, index=prices.columns, dtype=float)
            screened_current_weights = normalize_weights(current.loc[selected].values)

        delegated = np.asarray(
            self.base_controller.compute_weights(screened_prices, current_weights=screened_current_weights),
            dtype=float,
        )
        if len(delegated) != len(selected):
            raise ValueError("base controller returned weights with unexpected dimension")

        full_weights = pd.Series(0.0, index=prices.columns, dtype=float)
        full_weights.loc[selected] = normalize_weights(delegated)
        return normalize_weights(full_weights.values)


# ---------------------------------------------------------------------------
# 8. Equal-Weight Benchmark (EW)
# ---------------------------------------------------------------------------
class EWController:
    def __init__(self) -> None:
        self.name = "EW"

    def compute_weights(self, prices: pd.DataFrame, current_weights: np.ndarray | None = None) -> np.ndarray:
        n = len(prices.columns)
        return np.ones(n) / n


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_controllers() -> list:
    """Instantiate all controller families for the study."""
    return [
        EWController(),
        MRHController(MRHConfig(lookback=21, top_k=5)),
        MRHController(MRHConfig(lookback=63, top_k=5)),
        MRHController(MRHConfig(lookback=126, top_k=5)),
        MRSController(MRSConfig(window=20, temperature=1.0)),
        MRSController(MRSConfig(window=20, temperature=0.5)),
        CMVTController(CMVTConfig(lookback=63, risk_aversion=1.0, turnover_penalty=0.5)),
        CMVTController(CMVTConfig(lookback=63, risk_aversion=2.0, turnover_penalty=1.0)),
        RACController(RACConfig()),
        MLMController(MLMConfig(horizons=(21, 63, 126, 252), top_k=5)),
        HierarchicalRiskController(),
        ScreenedUniverseController(EWController(), top_n=5),
    ]
