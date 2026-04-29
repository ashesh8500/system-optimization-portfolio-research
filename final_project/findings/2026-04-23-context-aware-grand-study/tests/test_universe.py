from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from context_study.universe import STATIC_UNIVERSES, build_universe_membership


def make_candidate_panel() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=6, freq="M")
    records: list[dict[str, object]] = []

    liquidity_profiles = {
        "AAA": [900, 900, 900, 900, 900, 900],
        "BBB": [850, 850, 850, 850, 850, 850],
        "CCC": [800, 800, 800, 800, 800, 800],
        "DDD": [300, 300, 300, 300, 300, 300],
        "EEE": [200, 200, 200, 200, 200, 200],
        "FFF": [100, 100, 100, 100, 100, 1000],
    }

    for date_idx, date in enumerate(dates):
        for symbol, profile in liquidity_profiles.items():
            records.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "close": 100 + date_idx,
                    "volume": profile[date_idx],
                }
            )

    return pd.DataFrame.from_records(records)


def test_membership_snapshots_are_chronological_and_size_respected(tmp_path: Path) -> None:
    panel = make_candidate_panel()

    membership = build_universe_membership(
        universe_name="liquid_us_equity_100",
        candidate_panel=panel,
        rebalance_frequency="M",
        lookback_periods=3,
        target_size=3,
        save_artifact=True,
        metrics_dir=tmp_path,
    )

    snapshot_dates = membership["snapshot_date"].drop_duplicates().tolist()
    assert snapshot_dates == sorted(snapshot_dates)

    counts = membership.groupby("snapshot_date")["symbol"].nunique()
    assert counts.tolist() == [3, 3, 3, 3]

    artifact = tmp_path / "universe_membership" / "liquid_us_equity_100.csv"
    assert artifact.exists()


def test_builder_does_not_use_future_dates_for_current_membership() -> None:
    panel = make_candidate_panel()

    membership = build_universe_membership(
        universe_name="liquid_us_equity_100",
        candidate_panel=panel,
        rebalance_frequency="M",
        lookback_periods=3,
        target_size=3,
        save_artifact=False,
    )

    april_members = membership.loc[
        membership["snapshot_date"] == pd.Timestamp("2020-04-30"), "symbol"
    ].tolist()

    may_members = membership.loc[
        membership["snapshot_date"] == pd.Timestamp("2020-05-31"), "symbol"
    ].tolist()

    assert "FFF" not in april_members
    assert "FFF" not in may_members
    assert april_members == ["AAA", "BBB", "CCC"]


def test_static_universes_are_declared() -> None:
    assert STATIC_UNIVERSES["sector_etf"] == (
        "XLB",
        "XLC",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLRE",
        "XLU",
        "XLV",
        "XLY",
    )
    assert STATIC_UNIVERSES["multi_asset"] == (
        "SPY",
        "QQQ",
        "IWM",
        "EFA",
        "EEM",
        "TLT",
        "IEF",
        "LQD",
        "HYG",
        "GLD",
        "DBC",
        "VNQ",
    )


def test_rolling_membership_handles_insufficient_history_gracefully() -> None:
    tiny_panel = pd.DataFrame(
        [
            {"date": "2020-01-31", "symbol": "AAA", "close": 100.0, "volume": 10.0},
            {"date": "2020-01-31", "symbol": "BBB", "close": 100.0, "volume": 5.0},
        ]
    )
    membership = build_universe_membership(
        universe_name="liquid_us_equity_100",
        candidate_panel=tiny_panel,
        rebalance_frequency="M",
        lookback_periods=3,
        target_size=3,
        save_artifact=False,
    )

    assert membership.empty
    assert list(membership.columns) == [
        "universe_name",
        "snapshot_date",
        "symbol",
        "rank",
        "score",
        "selection_source",
        "lookback_end",
    ]


def test_static_universe_membership_uses_explicit_snapshot_dates(tmp_path: Path) -> None:
    membership = build_universe_membership(
        universe_name="sector_etf",
        snapshot_dates=[pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")],
        save_artifact=True,
        metrics_dir=tmp_path,
    )

    assert membership["snapshot_date"].nunique() == 2
    assert membership.groupby("snapshot_date")["symbol"].nunique().tolist() == [11, 11]
    artifact = tmp_path / "universe_membership" / "sector_etf.csv"
    assert artifact.exists()


def test_static_universe_empty_input_preserves_schema() -> None:
    empty_panel = pd.DataFrame(columns=["date", "symbol", "close", "volume"])
    membership = build_universe_membership(
        universe_name="sector_etf",
        candidate_panel=empty_panel,
        save_artifact=False,
    )
    assert membership.empty
    assert list(membership.columns) == [
        "universe_name",
        "snapshot_date",
        "symbol",
        "rank",
        "score",
        "selection_source",
        "lookback_end",
    ]
