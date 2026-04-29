"""
Exploration: System Optimization Controllers for Portfolio Allocation
=====================================================================

Tests convex QP and hierarchical risk budgeting controllers within the
same layered (screen → weight → route) framework as the 60-arm experiment.

Uses cached price data from the existing study to avoid Yahoo Finance API limits.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY_ROOT = Path(
    "/Users/asheshkaji/Documents/nyu/26_spring/systemopt/final_project/"
    "findings/2026-04-23-context-aware-grand-study"
)
OUTPUT_DIR = Path(
    "/Users/asheshkaji/Documents/nyu/26_spring/systemopt/final_project/"
    "findings/2026-04-26-system-opt-controllers"
)
METRICS_DIR = OUTPUT_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = STUDY_ROOT / "data"

sys.path.insert(0, str(STUDY_ROOT / "src"))

from context_study.strategies import (
    CMVTConfig, CMVTController,
    EWController, HierarchicalRiskController,
    MRHConfig, MRHController,
)
from context_study.screens import apply_screen
from context_study.protocol import generate_walk_forward_splits
from context_study.backtest import BacktestConfig, run_backtest
from context_study.candidates import NAMED_CANDIDATE_SETS, get_candidate_symbols

# ──────────────────────────────────────────────────────────────
# Controllers tested
# ──────────────────────────────────────────────────────────────
CONTROLLERS = [
    EWController(),
    MRHController(MRHConfig(lookback=21, top_k=5)),
    MRHController(MRHConfig(lookback=63, top_k=5)),
    MRHController(MRHConfig(lookback=126, top_k=5)),
    # Convex QP variants
    CMVTController(CMVTConfig(lookback=63, risk_aversion=0.5, turnover_penalty=0.25)),
    CMVTController(CMVTConfig(lookback=63, risk_aversion=1.0, turnover_penalty=0.5)),
    CMVTController(CMVTConfig(lookback=63, risk_aversion=2.0, turnover_penalty=1.0)),
    # Hierarchical risk budgeting
    HierarchicalRiskController(lookback=126, cluster_threshold=0.5),
    HierarchicalRiskController(lookback=63, cluster_threshold=0.3),
]

SCREEN_RULES = [
    "momentum_21_top10",
    "momentum_63_top10",
    "low_volatility_63_top10",
    "vol_adjusted_momentum_63_top10",
    "cluster_capped_momentum_63_top10",
]

CANDIDATE_UNIVERSES = [
    "liquid_us_equity_100",
    "liquid_us_equity_250",
    "liquid_us_equity_500",
]

# ──────────────────────────────────────────────────────────────
# Data loading (bypass PriceDataCache's aggressive .dropna())
# ──────────────────────────────────────────────────────────────
def load_price_panel(universe_name: str, start: str = "2018-01-01", end: str = "2025-01-01") -> pd.DataFrame:
    """Load price panel from cached parquet, handling missing data gracefully."""
    symbols = get_candidate_symbols(universe_name)
    
    frames = []
    for sym in symbols:
        cache_path = CACHE_DIR / f"{sym}_{start}_{end}.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            if "Close" in df.columns:
                s = df["Close"].rename(sym)
                s.index = pd.DatetimeIndex(s.index)
                frames.append(s)
    
    prices = pd.concat(frames, axis=1).sort_index()
    
    # Forward-fill per column (handles holidays, minor gaps)
    prices = prices.ffill()
    
    # Drop columns with <70% coverage (recent IPOs, delistings)
    coverage = prices.notna().sum() / len(prices)
    good = coverage[coverage >= 0.70].index.tolist()
    dropped = len(prices.columns) - len(good)
    if dropped:
        print(f"    Dropped {dropped} low-coverage assets: "
              f"{[c for c in prices.columns if c not in good][:5]}")
    prices = prices[good]
    
    # Drop rows where >50% of assets are NaN (early periods with few listings)
    row_ok = prices.notna().sum(axis=1) / len(prices.columns) >= 0.5
    prices = prices.loc[row_ok]
    
    # Forward-fill any remaining NaN
    prices = prices.ffill().bfill()
    
    return prices

# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────
def compute_metrics(returns: pd.Series) -> dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "max_dd": 0.0}
    ann_ret = float(returns.mean() * 252)
    ann_vol = float(returns.std() * np.sqrt(252))
    sharpe = ann_ret / (ann_vol + 1e-12)
    eq = (1.0 + returns).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())
    return {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": max_dd}

# ──────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────
COST_BPS = (10.0,)
all_summary, all_subperiod = [], []

for universe_name in CANDIDATE_UNIVERSES:
    print(f"\n{'='*60}")
    print(f"UNIVERSE: {universe_name}")
    print(f"{'='*60}")

    prices = load_price_panel(universe_name)
    print(f"  Panel: {prices.shape[0]} dates × {prices.shape[1]} assets")
    print(f"  Range: {prices.index[0].date()} to {prices.index[-1].date()}")

    splits = generate_walk_forward_splits(
        dates=prices.index,
        train_size=min(252, len(prices) // 2),
        validation_size=63,
        evaluation_size=63,
        step_size=63,
        cost_grid=tuple(c / 10_000 for c in COST_BPS),
    )
    print(f"  Splits: {len(splits)}, first eval: {splits[0].evaluation.start.date()}")

    for screen_rule in SCREEN_RULES:
        for ctrl in CONTROLLERS:
            screen_data = []
            for split in splits:
                screen = apply_screen(
                    prices.loc[: split.train.end],
                    screen_rule=screen_rule,
                    as_of_date=split.train.end,
                )
                if screen.empty:
                    continue
                selected = screen["symbol"].tolist()
                if len(selected) < 2:
                    continue

                try:
                    screened_prices = prices.loc[: split.evaluation.end, selected]
                    # Ensure no all-NaN columns
                    screened_prices = screened_prices.loc[:, screened_prices.notna().sum() > 10]
                    if screened_prices.shape[1] < 2:
                        continue
                except KeyError:
                    continue

                for cost_bps in COST_BPS:
                    try:
                        result = run_backtest(
                            screened_prices,
                            ctrl,
                            config=BacktestConfig(transaction_cost_bps=cost_bps),
                        )
                        eval_ret = result.returns.loc[
                            (result.returns.index >= split.evaluation.start)
                            & (result.returns.index <= split.evaluation.end)
                        ]
                        m = compute_metrics(eval_ret)
                        m["turnover"] = result.total_turnover
                        row = {
                            "universe": universe_name,
                            "screen_rule": screen_rule,
                            "controller": ctrl.name,
                            "split_id": split.split_id,
                            "cost_bps": cost_bps,
                            "n_assets": len(selected),
                            **m,
                        }
                        all_subperiod.append(row)
                        screen_data.append(m)
                    except Exception:
                        continue

            if screen_data:
                avg = {
                    k: float(np.mean([s[k] for s in screen_data]))
                    for k in ["ann_return", "ann_vol", "sharpe", "max_dd", "turnover"]
                }
                all_summary.append({
                    "universe": universe_name,
                    "screen_rule": screen_rule,
                    "controller": ctrl.name,
                    "splits": len(screen_data),
                    **avg,
                })
                print(f"  {screen_rule:<38} {ctrl.name:<22} "
                      f"Sharpe={avg['sharpe']:.3f} n={len(screen_data)}")

# ──────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────
pd.DataFrame(all_summary).to_csv(METRICS_DIR / "summary.csv", index=False)
pd.DataFrame(all_subperiod).to_csv(METRICS_DIR / "subperiod_summary.csv", index=False)

print(f"\nSaved {len(all_summary)} summary rows")
print(f"Saved {len(all_subperiod)} subperiod rows")

# ──────────────────────────────────────────────────────────────
# Quick comparison
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BEST SHARPE BY UNIVERSE (10 bps)")
print("=" * 60)

existing = pd.read_csv(STUDY_ROOT / "metrics" / "screened_core_summary.csv")
summary_df = pd.DataFrame(all_summary)

for uni in CANDIDATE_UNIVERSES:
    uni_short = uni.replace("liquid_us_equity_", "")
    ex = existing[existing["universe"] == uni].sort_values("mean_sharpe", ascending=False)
    new = summary_df[summary_df["universe"] == uni].sort_values("sharpe", ascending=False)

    print(f"\n--- {uni_short} ---")
    print("  Top 3 existing (EW/MRH only):")
    for _, r in ex.head(3).iterrows():
        print(f"    {r['screen_rule']:<36} {r['controller']:<18} Sharpe={float(r['mean_sharpe']):.3f}")
    print("  Top 3 with opt controllers:")
    for _, r in new.head(3).iterrows():
        print(f"    {r['screen_rule']:<36} {r['controller']:<22} Sharpe={r['sharpe']:.3f}")
