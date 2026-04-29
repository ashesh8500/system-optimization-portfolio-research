from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .data_loader import prepare_candidate_panel

STATIC_UNIVERSES: dict[str, tuple[str, ...]] = {
    "sector_etf": (
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
    ),
    "multi_asset": (
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
    ),
}

ROLLING_UNIVERSE_TARGETS: dict[str, int] = {
    "liquid_us_equity_100": 100,
    "liquid_us_equity_250": 250,
    "liquid_us_equity_500": 500,
}

DEFAULT_METRICS_DIR = Path(__file__).resolve().parents[2] / "metrics"
MEMBERSHIP_COLUMNS = [
    "universe_name",
    "snapshot_date",
    "symbol",
    "rank",
    "score",
    "selection_source",
    "lookback_end",
]


def build_universe_membership(
    universe_name: str,
    candidate_panel: pd.DataFrame | None = None,
    rebalance_frequency: str = "M",
    lookback_periods: int = 3,
    target_size: int | None = None,
    save_artifact: bool = True,
    metrics_dir: str | Path | None = None,
    snapshot_dates: Iterable[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    if universe_name in STATIC_UNIVERSES:
        membership = _build_static_membership(
            universe_name=universe_name,
            candidate_panel=candidate_panel,
            snapshot_dates=snapshot_dates,
        )
    elif universe_name in ROLLING_UNIVERSE_TARGETS:
        if candidate_panel is None:
            raise ValueError(f"candidate_panel is required for rolling universe {universe_name}")
        membership = _build_rolling_membership(
            universe_name=universe_name,
            candidate_panel=candidate_panel,
            rebalance_frequency=rebalance_frequency,
            lookback_periods=lookback_periods,
            target_size=target_size or ROLLING_UNIVERSE_TARGETS[universe_name],
        )
    else:
        supported = sorted(set(STATIC_UNIVERSES) | set(ROLLING_UNIVERSE_TARGETS))
        raise ValueError(f"unknown universe_name={universe_name!r}; supported={supported}")

    if save_artifact:
        _save_membership_artifact(
            universe_name=universe_name,
            membership=membership,
            metrics_dir=Path(metrics_dir) if metrics_dir is not None else DEFAULT_METRICS_DIR,
        )
    return membership


def _build_static_membership(
    universe_name: str,
    candidate_panel: pd.DataFrame | None,
    snapshot_dates: Iterable[pd.Timestamp] | None,
) -> pd.DataFrame:
    if snapshot_dates is not None:
        dates = [pd.Timestamp(date) for date in snapshot_dates]
    elif candidate_panel is not None and "date" in candidate_panel.columns:
        panel = prepare_candidate_panel(candidate_panel)
        dates = sorted(panel["date"].drop_duplicates().tolist())
    else:
        return pd.DataFrame(columns=MEMBERSHIP_COLUMNS)

    records = []
    for snapshot_date in dates:
        for rank, symbol in enumerate(STATIC_UNIVERSES[universe_name], start=1):
            records.append(
                {
                    "universe_name": universe_name,
                    "snapshot_date": pd.Timestamp(snapshot_date),
                    "symbol": symbol,
                    "rank": rank,
                    "score": None,
                    "selection_source": "static",
                    "lookback_end": pd.Timestamp(snapshot_date),
                }
            )
    if not records:
        return pd.DataFrame(columns=MEMBERSHIP_COLUMNS)
    return pd.DataFrame.from_records(records, columns=MEMBERSHIP_COLUMNS)


def _build_rolling_membership(
    universe_name: str,
    candidate_panel: pd.DataFrame,
    rebalance_frequency: str,
    lookback_periods: int,
    target_size: int,
) -> pd.DataFrame:
    if lookback_periods < 1:
        raise ValueError("lookback_periods must be >= 1")
    if target_size < 1:
        raise ValueError("target_size must be >= 1")

    panel = prepare_candidate_panel(candidate_panel)
    panel = panel.copy()
    panel["period"] = panel["date"].dt.to_period(rebalance_frequency)

    period_ends = (
        panel.groupby("period", as_index=False)["date"]
        .max()
        .sort_values("date")
        .reset_index(drop=True)
    )

    records: list[dict[str, object]] = []
    for end_idx in range(lookback_periods - 1, len(period_ends)):
        window_periods = period_ends.iloc[end_idx - lookback_periods + 1 : end_idx + 1]
        snapshot_date = pd.Timestamp(period_ends.iloc[end_idx]["date"])
        window_end = pd.Timestamp(window_periods.iloc[-1]["date"])
        eligible_periods = set(window_periods["period"].tolist())

        history = panel.loc[panel["period"].isin(eligible_periods)].copy()
        if history.empty:
            continue

        scores = (
            history.groupby("symbol")
            .agg(
                availability_ratio=("is_available", "mean"),
                liquidity_score=("dollar_volume", "mean"),
                observation_count=("date", "size"),
                last_observation=("date", "max"),
            )
            .reset_index()
        )
        scores = scores.loc[
            (scores["availability_ratio"] > 0)
            & (scores["last_observation"] <= window_end)
        ].copy()
        scores["score"] = scores["availability_ratio"] * scores["liquidity_score"]
        scores = scores.sort_values(
            ["score", "availability_ratio", "liquidity_score", "symbol"],
            ascending=[False, False, False, True],
        ).head(target_size)

        for rank, row in enumerate(scores.itertuples(index=False), start=1):
            records.append(
                {
                    "universe_name": universe_name,
                    "snapshot_date": snapshot_date,
                    "symbol": row.symbol,
                    "rank": rank,
                    "score": float(row.score),
                    "selection_source": "rolling_liquidity",
                    "lookback_end": window_end,
                }
            )

    if not records:
        return pd.DataFrame(columns=MEMBERSHIP_COLUMNS)
    return (
        pd.DataFrame.from_records(records, columns=MEMBERSHIP_COLUMNS)
        .sort_values(["snapshot_date", "rank", "symbol"])
        .reset_index(drop=True)
    )


def _save_membership_artifact(
    universe_name: str,
    membership: pd.DataFrame,
    metrics_dir: Path,
) -> Path:
    output_dir = metrics_dir / "universe_membership"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{universe_name}.csv"
    membership.to_csv(output_path, index=False)
    return output_path
