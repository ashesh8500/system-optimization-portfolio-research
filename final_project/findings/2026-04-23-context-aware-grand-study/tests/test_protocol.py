from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from context_study.protocol import WalkForwardWindow, generate_walk_forward_splits


def make_dates(periods: int = 12) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-31", periods=periods, freq="M")


def test_validation_starts_after_training_ends() -> None:
    split = generate_walk_forward_splits(
        dates=make_dates(),
        train_size=4,
        validation_size=2,
        evaluation_size=2,
        step_size=2,
    )[0]

    assert split.validation.start > split.train.end
    assert split.validation.start == pd.Timestamp("2020-05-31")


def test_evaluation_starts_after_validation_ends() -> None:
    split = generate_walk_forward_splits(
        dates=make_dates(),
        train_size=4,
        validation_size=2,
        evaluation_size=2,
        step_size=2,
    )[0]

    assert split.evaluation.start > split.validation.end
    assert split.evaluation.start == pd.Timestamp("2020-07-31")


def test_windows_are_chronological_and_non_overlapping_within_each_split() -> None:
    splits = generate_walk_forward_splits(
        dates=make_dates(),
        train_size=4,
        validation_size=2,
        evaluation_size=2,
        step_size=2,
    )

    for split in splits:
        windows = [split.train, split.validation, split.evaluation]
        assert all(isinstance(window, WalkForwardWindow) for window in windows)
        assert windows == sorted(windows, key=lambda window: window.start)

        previous_end = None
        for window in windows:
            assert window.start <= window.end
            if previous_end is not None:
                assert window.start > previous_end
            previous_end = window.end


def test_multiple_splits_are_generated_for_expanding_windows() -> None:
    splits = generate_walk_forward_splits(
        dates=make_dates(periods=12),
        train_size=4,
        validation_size=2,
        evaluation_size=2,
        step_size=2,
        cost_grid=(0.0, 0.001, 0.002),
    )

    assert len(splits) == 3
    assert [split.split_id for split in splits] == [0, 1, 2]
    assert [split.train.end for split in splits] == [
        pd.Timestamp("2020-04-30"),
        pd.Timestamp("2020-06-30"),
        pd.Timestamp("2020-08-31"),
    ]
    assert all(split.cost_grid == (0.0, 0.001, 0.002) for split in splits)


def test_invalid_window_configuration_raises_value_error() -> None:
    with pytest.raises(ValueError):
        generate_walk_forward_splits(
            dates=make_dates(periods=7),
            train_size=4,
            validation_size=2,
            evaluation_size=2,
            step_size=1,
        )
