from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from context_study.experiment_artifacts import (
    MODEL_SELECTION_LEDGER_COLUMNS,
    TRIAL_COLUMNS,
    UNIVERSE_PROVENANCE_COLUMNS,
    ExperimentArtifactWriter,
    build_trial_record,
    build_universe_provenance_record,
)


def test_build_universe_provenance_record_flags_current_constituent_bias() -> None:
    record = build_universe_provenance_record(
        universe_name="liquid_current_100",
        source="current public constituent list",
        construction_date="2026-04-23",
        point_in_time_available=False,
        includes_delisted=False,
        membership_changes_used=False,
        missing_data_rule="drop assets without full adjusted-close history",
        minimum_history_days=252,
    )

    assert list(record.keys()) == UNIVERSE_PROVENANCE_COLUMNS
    assert record["survivorship_bias_flag"] is True
    assert record["point_in_time_available"] is False
    assert record["includes_delisted"] is False


def test_build_trial_record_preserves_all_required_research_degrees_of_freedom() -> None:
    record = build_trial_record(
        run_id="demo",
        split_id=1,
        universe_name="liquid_current_250",
        universe_rule="lagged_dollar_volume_topN_with_buffer",
        screen_rule="momentum_21_topK",
        holding_budget=20,
        controller="equal_weight",
        cost_bps=10.0,
        train_start="2020-01-01",
        train_end="2021-12-31",
        evaluation_start="2022-01-01",
        evaluation_end="2022-06-30",
        gross_return=0.12,
        net_return=0.10,
        turnover=1.5,
        sharpe=1.1,
        max_drawdown=-0.08,
        selected_for_paper=False,
    )

    assert list(record.keys()) == TRIAL_COLUMNS
    assert record["holding_budget"] == 20
    assert record["selected_for_paper"] is False


def test_artifact_writer_writes_canonical_csv_schemas(tmp_path: Path) -> None:
    writer = ExperimentArtifactWriter(metrics_dir=tmp_path)
    provenance = build_universe_provenance_record(
        universe_name="demo",
        source="synthetic",
        construction_date="2026-04-23",
        point_in_time_available=True,
        includes_delisted=True,
        membership_changes_used=True,
        missing_data_rule="none",
        minimum_history_days=30,
    )
    trial = build_trial_record(
        run_id="demo",
        split_id=0,
        universe_name="demo",
        universe_rule="fixed",
        screen_rule="none",
        holding_budget="all",
        controller="equal_weight",
        cost_bps=10.0,
        train_start="2020-01-01",
        train_end="2020-12-31",
        evaluation_start="2021-01-01",
        evaluation_end="2021-06-30",
        gross_return=0.05,
        net_return=0.04,
        turnover=0.2,
        sharpe=0.8,
        max_drawdown=-0.03,
        selected_for_paper=True,
    )
    ledger = {
        "run_id": "demo",
        "timestamp_utc": "2026-04-23T00:00:00Z",
        "universe_name": "demo",
        "universe_rule": "fixed",
        "screen_rule": "none",
        "holding_budget": "all",
        "controller": "equal_weight",
        "hyperparameters": "{}",
        "cost_bps": 10.0,
        "selected_for_paper": True,
    }

    paths = writer.write_all(
        universe_provenance=[provenance],
        model_selection_ledger=[ledger],
        all_trials=[trial],
    )

    assert set(paths) == {"universe_provenance", "model_selection_ledger", "all_trials"}
    assert pd.read_csv(paths["universe_provenance"]).columns.tolist() == UNIVERSE_PROVENANCE_COLUMNS
    assert pd.read_csv(paths["model_selection_ledger"]).columns.tolist() == MODEL_SELECTION_LEDGER_COLUMNS
    assert pd.read_csv(paths["all_trials"]).columns.tolist() == TRIAL_COLUMNS
