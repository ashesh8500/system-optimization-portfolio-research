from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_study.data_loader import PriceDataCache
from context_study.runner import run_screened_candidate_benchmark_pilot


def main() -> None:
    cache = PriceDataCache(ROOT / "data")
    runs = [
        ("liquid_us_equity_100", 25),
        ("liquid_us_equity_250", 40),
        ("liquid_us_equity_500", 60),
    ]
    screen_rules = (
        "momentum_21_top10",
        "momentum_63_top10",
        "vol_adjusted_momentum_63_top10",
        "low_volatility_63_top10",
        "cluster_capped_momentum_63_top10",
    )
    for universe_name, symbol_limit in runs:
        print(f"RUN {universe_name} symbol_limit={symbol_limit}", flush=True)
        artifacts = run_screened_candidate_benchmark_pilot(
            root=ROOT,
            cache=cache,
            universe_name=universe_name,
            screen_rules=screen_rules,
            controller_limit=4,
            cost_grid_bps=(0.0, 10.0, 25.0),
            symbol_limit=symbol_limit,
        )
        summary = artifacts["summary"]
        print(summary.sort_values("mean_sharpe", ascending=False).head(8).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
