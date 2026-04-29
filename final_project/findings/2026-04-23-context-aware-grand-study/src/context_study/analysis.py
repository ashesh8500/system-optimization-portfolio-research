from __future__ import annotations

import pandas as pd


DEFAULT_PERIOD_COLUMNS = [
    "universe",
    "split_id",
    "period",
    "train_end",
    "validation_end",
    "evaluation_start",
    "evaluation_end",
    "cost_bps",
]


def compute_period_winners(
    performance: pd.DataFrame,
    *,
    metric: str = "sharpe",
    period_columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Collapse controller-level outcomes into one winner row per evaluation period."""
    performance = _coerce_performance_table(performance, metric=metric)
    period_cols = _resolve_period_columns(performance, period_columns=period_columns)

    ranked = performance.sort_values(period_cols + [metric, "controller"], ascending=[True] * len(period_cols) + [False, True])
    winners = ranked.groupby(period_cols, as_index=False, sort=True).first()
    winners = winners.rename(columns={"controller": "winner_label", metric: "winner_score"})

    keep_columns = period_cols + ["winner_label", "winner_score"]
    return winners.loc[:, keep_columns].reset_index(drop=True)


def build_descriptor_winner_table(
    *,
    descriptors: pd.DataFrame,
    performance: pd.DataFrame,
    metric: str = "sharpe",
    period_columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Join lagged descriptors onto period winners for downstream analysis."""
    winners = compute_period_winners(performance, metric=metric, period_columns=period_columns)
    descriptors = _coerce_descriptor_table(descriptors)

    descriptor_features = descriptors.reset_index(names="descriptor_date")
    aligned = pd.merge_asof(
        winners.sort_values("evaluation_start"),
        descriptor_features,
        left_on="evaluation_start",
        right_on="descriptor_date",
        direction="backward",
        allow_exact_matches=False,
    )
    ordered_columns = list(winners.columns) + ["descriptor_date"] + list(descriptors.columns)
    return aligned.loc[:, ordered_columns].reset_index(drop=True)


def summarize_features_by_winner(
    analysis_table: pd.DataFrame,
    *,
    feature_columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Compute lightweight class-conditional descriptor summaries."""
    if "winner_label" not in analysis_table.columns:
        raise ValueError("analysis_table must contain a winner_label column.")

    table = analysis_table.copy()
    if feature_columns is None:
        excluded = {
            "split_id",
            "winner_score",
        }
        excluded.update(column for column in table.columns if column.endswith("_start") or column.endswith("_end"))
        feature_columns = [
            column
            for column in table.columns
            if column not in excluded
            and column not in {"winner_label", "period", "universe", "descriptor_date"}
            and pd.api.types.is_numeric_dtype(table[column])
        ]

    grouped = table.groupby("winner_label", sort=True)
    summary = grouped.size().rename("observations").reset_index()

    for feature in feature_columns:
        summary[f"{feature}_mean"] = grouped[feature].mean().to_numpy()

    return summary


def _coerce_performance_table(performance: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    if not isinstance(performance, pd.DataFrame):
        raise TypeError("performance must be a pandas DataFrame.")
    required_columns = {"controller", metric, "evaluation_start"}
    missing = required_columns.difference(performance.columns)
    if missing:
        raise ValueError(f"performance is missing required columns: {sorted(missing)}")

    table = performance.copy()
    table["evaluation_start"] = pd.to_datetime(table["evaluation_start"])
    if "evaluation_end" in table.columns:
        table["evaluation_end"] = pd.to_datetime(table["evaluation_end"])
    return table


def _coerce_descriptor_table(descriptors: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(descriptors, pd.DataFrame):
        raise TypeError("descriptors must be a pandas DataFrame.")
    table = descriptors.sort_index().copy()
    table.index = pd.to_datetime(table.index)
    if not table.index.is_monotonic_increasing:
        raise ValueError("descriptor index must be sorted in ascending order.")
    if table.index.has_duplicates:
        raise ValueError("descriptor index must be unique.")
    return table


def _resolve_period_columns(
    performance: pd.DataFrame,
    *,
    period_columns: list[str] | tuple[str, ...] | None,
) -> list[str]:
    if period_columns is not None:
        resolved = list(period_columns)
    else:
        resolved = [column for column in DEFAULT_PERIOD_COLUMNS if column in performance.columns]

    if not resolved:
        raise ValueError("Could not infer period columns from performance table.")
    return resolved
