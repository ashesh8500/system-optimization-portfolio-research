from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from context_study.analysis import (
    build_descriptor_winner_table,
    compute_period_winners,
    summarize_features_by_winner,
)


def make_descriptors() -> pd.DataFrame:
    index = pd.to_datetime([
        "2020-01-31",
        "2020-02-03",
        "2020-02-28",
        "2020-03-31",
    ])
    return pd.DataFrame(
        {
            "avg_pairwise_corr": [0.20, 0.90, 0.40, 0.70],
            "vol_level": [0.10, 0.80, 0.20, 0.50],
        },
        index=index,
    )


def make_performance_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "universe": "demo",
                "split_id": 0,
                "period": "2020Q1",
                "evaluation_start": pd.Timestamp("2020-02-03"),
                "evaluation_end": pd.Timestamp("2020-02-28"),
                "controller": "EW",
                "sharpe": 0.40,
                "ann_return": 0.08,
            },
            {
                "universe": "demo",
                "split_id": 0,
                "period": "2020Q1",
                "evaluation_start": pd.Timestamp("2020-02-03"),
                "evaluation_end": pd.Timestamp("2020-02-28"),
                "controller": "MRH",
                "sharpe": 0.90,
                "ann_return": 0.04,
            },
            {
                "universe": "demo",
                "split_id": 1,
                "period": "2020Q2",
                "evaluation_start": pd.Timestamp("2020-03-31"),
                "evaluation_end": pd.Timestamp("2020-04-30"),
                "controller": "EW",
                "sharpe": 0.30,
                "ann_return": 0.03,
            },
            {
                "universe": "demo",
                "split_id": 1,
                "period": "2020Q2",
                "evaluation_start": pd.Timestamp("2020-03-31"),
                "evaluation_end": pd.Timestamp("2020-04-30"),
                "controller": "MRH",
                "sharpe": 0.10,
                "ann_return": 0.06,
            },
        ]
    )


def test_build_descriptor_winner_table_aligns_lagged_descriptors_and_winners() -> None:
    analysis = build_descriptor_winner_table(
        descriptors=make_descriptors(),
        performance=make_performance_table(),
        metric="sharpe",
    )

    assert analysis["winner_label"].tolist() == ["MRH", "EW"]
    assert analysis["descriptor_date"].tolist() == [
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-28"),
    ]
    assert analysis["avg_pairwise_corr"].tolist() == [0.20, 0.40]
    assert analysis["winner_score"].tolist() == [0.90, 0.30]


def test_compute_period_winners_supports_alternate_metric() -> None:
    winners = compute_period_winners(make_performance_table(), metric="ann_return")

    assert winners["winner_label"].tolist() == ["EW", "MRH"]
    assert winners["winner_score"].tolist() == [0.08, 0.06]


def test_summarize_features_by_winner_returns_interpretable_class_means() -> None:
    analysis = build_descriptor_winner_table(
        descriptors=make_descriptors(),
        performance=make_performance_table(),
        metric="sharpe",
    )

    summary = summarize_features_by_winner(analysis)

    assert summary["winner_label"].tolist() == ["EW", "MRH"]
    assert summary["observations"].tolist() == [1, 1]
    assert summary["avg_pairwise_corr_mean"].tolist() == [0.40, 0.20]
    assert summary["vol_level_mean"].tolist() == [0.20, 0.10]
