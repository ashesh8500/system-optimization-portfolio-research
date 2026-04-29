from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from context_study import runner


class FakeCache:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def fetch(self, symbols: list[str], start: str, end: str, column: str = "Close") -> pd.DataFrame:
        self.calls.append(
            {
                "symbols": list(symbols),
                "start": start,
                "end": end,
                "column": column,
            }
        )
        dates = pd.bdate_range("2020-01-01", periods=420)
        base_path = np.linspace(100.0, 130.0, len(dates))
        price_map = {
            symbol: base_path + (idx * 3.0) + np.sin(np.arange(len(dates)) / 20.0 + idx)
            for idx, symbol in enumerate(symbols)
        }
        return pd.DataFrame(price_map, index=dates)


def test_resolve_universe_spec_loads_candidate_symbols(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_get_candidate_symbols(
        universe_name: str,
        data_dir: str | Path | None = None,
        limit: int | None = None,
    ) -> list[str]:
        captured["universe_name"] = universe_name
        captured["data_dir"] = data_dir
        captured["limit"] = limit
        return ["AAA", "BBB", "CCC"][: limit or 3]

    monkeypatch.setattr(runner, "get_candidate_symbols", fake_get_candidate_symbols)

    spec = runner.resolve_universe_spec(
        "liquid_us_equity_100",
        data_dir=Path("/tmp/candidates"),
        symbol_limit=2,
    )

    assert spec["candidate_universe_name"] == "liquid_us_equity_100"
    assert spec["universe_kind"] == "candidate_static"
    assert spec["symbols"] == ["AAA", "BBB"]
    assert captured == {
        "universe_name": "liquid_us_equity_100",
        "data_dir": Path("/tmp/candidates"),
        "limit": 2,
    }


def test_run_candidate_benchmark_pilot_uses_fake_cache_and_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    candidate_symbols = ["AAA", "BBB", "CCC", "DDD"]
    monkeypatch.setattr(runner, "get_candidate_symbols", lambda *args, **kwargs: list(candidate_symbols))
    cache = FakeCache()

    summary, subperiod, cost = runner.run_candidate_benchmark_pilot(
        root=tmp_path,
        cache=cache,
        universe_name="liquid_us_equity_100",
        controller_limit=1,
        cost_grid_bps=(0.0, 10.0),
        symbol_limit=4,
    )

    assert len(cache.calls) == 1
    assert cache.calls[0]["symbols"] == candidate_symbols
    assert cache.calls[0]["start"] == runner.DEFAULT_BENCHMARK_START
    assert cache.calls[0]["end"] == runner.DEFAULT_BENCHMARK_END

    assert not summary.empty
    assert summary["universe"].unique().tolist() == ["liquid_us_equity_100"]
    assert summary["controller"].unique().tolist() == ["EW"]

    assert not subperiod.empty
    assert not cost.empty
    assert sorted(cost["cost_bps"].unique().tolist()) == [0.0, 10.0]

    output_dir = tmp_path / "metrics" / "candidate_pilot" / "liquid_us_equity_100"
    assert (output_dir / "walk_forward_manifest.csv").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "subperiod_summary.csv").exists()
    assert (output_dir / "cost_sensitivity.csv").exists()


def test_run_screened_candidate_benchmark_writes_screen_and_ledger_outputs(tmp_path: Path, monkeypatch) -> None:
    candidate_symbols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    monkeypatch.setattr(runner, "get_candidate_symbols", lambda *args, **kwargs: list(candidate_symbols))
    cache = FakeCache()

    outputs = runner.run_screened_candidate_benchmark_pilot(
        root=tmp_path,
        cache=cache,
        universe_name="liquid_us_equity_100",
        screen_rules=("momentum_21_top3",),
        controller_limit=1,
        cost_grid_bps=(10.0,),
        symbol_limit=6,
    )

    assert set(outputs) == {"summary", "subperiod", "cost", "screen_membership"}
    assert not outputs["summary"].empty
    assert outputs["summary"]["screen_rule"].unique().tolist() == ["momentum_21_top3"]
    assert outputs["summary"]["holding_budget"].unique().tolist() == [3]
    assert not outputs["screen_membership"].empty
    assert outputs["screen_membership"]["split_id"].nunique() >= 1
    assert outputs["screen_membership"]["symbol"].groupby(outputs["screen_membership"]["split_id"]).nunique().max() <= 3

    output_dir = tmp_path / "metrics" / "screened_candidate_pilot" / "liquid_us_equity_100"
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "subperiod_summary.csv").exists()
    assert (output_dir / "cost_sensitivity.csv").exists()
    assert (output_dir / "screen_membership.csv").exists()
    assert (output_dir / "model_selection_ledger.csv").exists()
    assert (output_dir / "all_trials.csv").exists()


def test_run_descriptor_winner_analysis_and_meta_controller_pilot_write_outputs(tmp_path: Path) -> None:
    descriptor_index = pd.to_datetime(["2020-01-31", "2020-04-30", "2020-07-31", "2020-10-31"])
    descriptors = pd.DataFrame(
        {
            "avg_pairwise_corr": [0.2, 0.8, 0.3, 0.7],
            "vol_level": [0.1, 0.4, 0.15, 0.35],
        },
        index=descriptor_index,
    )
    subperiod = pd.DataFrame(
        [
            {"universe": "demo", "controller": "EW", "split_id": 0, "cost_bps": 10.0, "evaluation_start": pd.Timestamp("2020-02-15"), "evaluation_end": pd.Timestamp("2020-03-31"), "sharpe": 0.2, "ann_return": 0.05},
            {"universe": "demo", "controller": "MRH_tau21_k5", "split_id": 0, "cost_bps": 10.0, "evaluation_start": pd.Timestamp("2020-02-15"), "evaluation_end": pd.Timestamp("2020-03-31"), "sharpe": 0.6, "ann_return": 0.04},
            {"universe": "demo", "controller": "EW", "split_id": 1, "cost_bps": 10.0, "evaluation_start": pd.Timestamp("2020-05-15"), "evaluation_end": pd.Timestamp("2020-06-30"), "sharpe": 0.7, "ann_return": 0.06},
            {"universe": "demo", "controller": "MRH_tau21_k5", "split_id": 1, "cost_bps": 10.0, "evaluation_start": pd.Timestamp("2020-05-15"), "evaluation_end": pd.Timestamp("2020-06-30"), "sharpe": 0.3, "ann_return": 0.02},
            {"universe": "demo", "controller": "EW", "split_id": 2, "cost_bps": 10.0, "evaluation_start": pd.Timestamp("2020-08-15"), "evaluation_end": pd.Timestamp("2020-09-30"), "sharpe": 0.25, "ann_return": 0.03},
            {"universe": "demo", "controller": "MRH_tau21_k5", "split_id": 2, "cost_bps": 10.0, "evaluation_start": pd.Timestamp("2020-08-15"), "evaluation_end": pd.Timestamp("2020-09-30"), "sharpe": 0.8, "ann_return": 0.07},
        ]
    )

    analysis_table, feature_summary = runner.run_descriptor_winner_analysis(
        root=tmp_path,
        universe_name="demo",
        descriptors=descriptors,
        subperiod_summary=subperiod,
    )
    assert not analysis_table.empty
    assert not feature_summary.empty
    assert (tmp_path / "metrics" / "analysis" / "demo_winner_table.csv").exists()
    assert (tmp_path / "metrics" / "analysis" / "demo_feature_summary.csv").exists()

    meta_eval = runner.run_meta_controller_pilot(
        root=tmp_path,
        universe_name="demo",
        analysis_table=analysis_table,
    )
    assert meta_eval["n_test"] >= 1
    assert "accuracy" in meta_eval
    assert (tmp_path / "metrics" / "meta_controller" / "demo_predictions.csv").exists()
    assert (tmp_path / "metrics" / "meta_controller" / "demo_evaluation.json").exists()


def test_rl_training_and_evaluation_advance_policy_state() -> None:
    dates = pd.bdate_range("2021-01-01", periods=180)
    x = np.arange(len(dates), dtype=float)
    prices = pd.DataFrame(
        {
            "AAA": 100.0 + 0.05 * x + np.sin(x / 12.0),
            "BBB": 90.0 + 0.03 * x + np.cos(x / 9.0),
            "CCC": 70.0 + 0.04 * x + np.sin(x / 7.0),
        },
        index=dates,
    )

    trained = runner.train_rl_policy(prices, n_updates=3, horizon=32, seed=11)
    assert trained["theta"].shape[1] == 3
    assert len(trained["reward_history"]) == 3

    result = runner.run_rl_evaluation(prices, trained["theta"])
    assert result.controller_name == "RL_Learned"
    assert len(result.equity_curve) > 20
    assert len(result.weights_history) == len(result.equity_curve)
    assert np.isfinite(result.sharpe)
    assert result.total_turnover > 0.0


def test_run_candidate_analysis_pipeline_writes_all_outputs(tmp_path: Path, monkeypatch) -> None:
    candidate_symbols = [f"S{i:03d}" for i in range(12)]
    monkeypatch.setattr(runner, "get_candidate_symbols", lambda *args, **kwargs: list(candidate_symbols))
    cache = FakeCache()

    outputs = runner.run_candidate_analysis_pipeline(
        root=tmp_path,
        cache=cache,
        universe_name="liquid_us_equity_100",
        controller_limit=2,
        cost_grid_bps=(0.0, 10.0),
        symbol_limit=12,
    )

    assert set(outputs.keys()) == {"summary", "subperiod", "cost", "analysis", "feature_summary", "meta"}
    assert not outputs["summary"].empty
    assert not outputs["analysis"].empty
    assert (tmp_path / "metrics" / "candidate_pilot" / "liquid_us_equity_100" / "summary.csv").exists()
    assert (tmp_path / "metrics" / "analysis" / "liquid_us_equity_100_winner_table.csv").exists()
    assert (tmp_path / "metrics" / "meta_controller" / "liquid_us_equity_100_evaluation.json").exists()
