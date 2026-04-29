from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

UNIVERSE_PROVENANCE_COLUMNS = [
    "universe_name",
    "source",
    "construction_date",
    "point_in_time_available",
    "includes_delisted",
    "membership_changes_used",
    "survivorship_bias_flag",
    "missing_data_rule",
    "minimum_history_days",
]

MODEL_SELECTION_LEDGER_COLUMNS = [
    "run_id",
    "timestamp_utc",
    "universe_name",
    "universe_rule",
    "screen_rule",
    "holding_budget",
    "controller",
    "hyperparameters",
    "cost_bps",
    "selected_for_paper",
]

TRIAL_COLUMNS = [
    "run_id",
    "split_id",
    "universe_name",
    "universe_rule",
    "screen_rule",
    "holding_budget",
    "controller",
    "cost_bps",
    "train_start",
    "train_end",
    "evaluation_start",
    "evaluation_end",
    "gross_return",
    "net_return",
    "turnover",
    "sharpe",
    "max_drawdown",
    "selected_for_paper",
]


def build_universe_provenance_record(
    *,
    universe_name: str,
    source: str,
    construction_date: str | pd.Timestamp,
    point_in_time_available: bool,
    includes_delisted: bool,
    membership_changes_used: bool,
    missing_data_rule: str,
    minimum_history_days: int,
) -> dict[str, object]:
    if minimum_history_days < 0:
        raise ValueError("minimum_history_days must be non-negative")
    survivorship_bias_flag = not (point_in_time_available and includes_delisted and membership_changes_used)
    return {
        "universe_name": universe_name,
        "source": source,
        "construction_date": str(pd.Timestamp(construction_date).date()),
        "point_in_time_available": bool(point_in_time_available),
        "includes_delisted": bool(includes_delisted),
        "membership_changes_used": bool(membership_changes_used),
        "survivorship_bias_flag": bool(survivorship_bias_flag),
        "missing_data_rule": missing_data_rule,
        "minimum_history_days": int(minimum_history_days),
    }


def build_trial_record(
    *,
    run_id: str,
    split_id: int,
    universe_name: str,
    universe_rule: str,
    screen_rule: str,
    holding_budget: int | str,
    controller: str,
    cost_bps: float,
    train_start: str | pd.Timestamp,
    train_end: str | pd.Timestamp,
    evaluation_start: str | pd.Timestamp,
    evaluation_end: str | pd.Timestamp,
    gross_return: float,
    net_return: float,
    turnover: float,
    sharpe: float,
    max_drawdown: float,
    selected_for_paper: bool,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "split_id": int(split_id),
        "universe_name": universe_name,
        "universe_rule": universe_rule,
        "screen_rule": screen_rule,
        "holding_budget": holding_budget,
        "controller": controller,
        "cost_bps": float(cost_bps),
        "train_start": str(pd.Timestamp(train_start).date()),
        "train_end": str(pd.Timestamp(train_end).date()),
        "evaluation_start": str(pd.Timestamp(evaluation_start).date()),
        "evaluation_end": str(pd.Timestamp(evaluation_end).date()),
        "gross_return": float(gross_return),
        "net_return": float(net_return),
        "turnover": float(turnover),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "selected_for_paper": bool(selected_for_paper),
    }


def _coerce_rows(rows: Iterable[dict[str, object]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows), columns=columns)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"rows missing required columns: {missing}")
    extra = [column for column in frame.columns if column not in columns]
    if extra:
        raise ValueError(f"rows contain unexpected columns: {extra}")
    return frame.loc[:, columns]


@dataclass(frozen=True)
class ExperimentArtifactWriter:
    metrics_dir: Path | str

    def _write(self, filename: str, rows: Iterable[dict[str, object]], columns: list[str]) -> Path:
        metrics_dir = Path(self.metrics_dir)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        output_path = metrics_dir / filename
        _coerce_rows(rows, columns).to_csv(output_path, index=False)
        return output_path

    def write_universe_provenance(self, rows: Iterable[dict[str, object]]) -> Path:
        return self._write("universe_provenance.csv", rows, UNIVERSE_PROVENANCE_COLUMNS)

    def write_model_selection_ledger(self, rows: Iterable[dict[str, object]]) -> Path:
        return self._write("model_selection_ledger.csv", rows, MODEL_SELECTION_LEDGER_COLUMNS)

    def write_all_trials(self, rows: Iterable[dict[str, object]]) -> Path:
        return self._write("all_trials.csv", rows, TRIAL_COLUMNS)

    def write_all(
        self,
        *,
        universe_provenance: Iterable[dict[str, object]],
        model_selection_ledger: Iterable[dict[str, object]],
        all_trials: Iterable[dict[str, object]],
    ) -> dict[str, Path]:
        return {
            "universe_provenance": self.write_universe_provenance(universe_provenance),
            "model_selection_ledger": self.write_model_selection_ledger(model_selection_ledger),
            "all_trials": self.write_all_trials(all_trials),
        }
