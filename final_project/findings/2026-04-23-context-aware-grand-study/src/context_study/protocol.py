from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True, order=True)
class WalkForwardWindow:
    start: pd.Timestamp
    end: pd.Timestamp
    dates: tuple[pd.Timestamp, ...]

    @property
    def size(self) -> int:
        return len(self.dates)


@dataclass(frozen=True)
class WalkForwardSplit:
    split_id: int
    train: WalkForwardWindow
    validation: WalkForwardWindow
    evaluation: WalkForwardWindow
    step_size: int
    cost_grid: tuple[float, ...] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "split_id": self.split_id,
            "train_start": self.train.start,
            "train_end": self.train.end,
            "train_dates": self.train.dates,
            "validation_start": self.validation.start,
            "validation_end": self.validation.end,
            "validation_dates": self.validation.dates,
            "evaluation_start": self.evaluation.start,
            "evaluation_end": self.evaluation.end,
            "evaluation_dates": self.evaluation.dates,
            "step_size": self.step_size,
            "cost_grid": self.cost_grid,
        }


def generate_walk_forward_splits(
    *,
    dates: Sequence[pd.Timestamp] | pd.Index,
    train_size: int,
    validation_size: int,
    evaluation_size: int,
    step_size: int,
    cost_grid: Sequence[float] | None = None,
) -> list[WalkForwardSplit]:
    """Build chronology-respecting expanding-window walk-forward splits.

    The training window expands from the start of the sample. Validation and
    evaluation windows roll forward in fixed-size blocks separated from prior
    windows. Each split contains explicit date tuples for direct runner use.
    """
    normalized_dates = _normalize_dates(dates)
    _validate_window_sizes(
        sample_size=len(normalized_dates),
        train_size=train_size,
        validation_size=validation_size,
        evaluation_size=evaluation_size,
        step_size=step_size,
    )

    splits: list[WalkForwardSplit] = []
    split_id = 0
    train_end_idx = train_size - 1
    normalized_cost_grid = None if cost_grid is None else tuple(float(cost) for cost in cost_grid)

    while True:
        validation_start_idx = train_end_idx + 1
        validation_end_idx = validation_start_idx + validation_size - 1
        evaluation_start_idx = validation_end_idx + 1
        evaluation_end_idx = evaluation_start_idx + evaluation_size - 1

        if evaluation_end_idx >= len(normalized_dates):
            break

        train_dates = normalized_dates[: train_end_idx + 1]
        validation_dates = normalized_dates[validation_start_idx : validation_end_idx + 1]
        evaluation_dates = normalized_dates[evaluation_start_idx : evaluation_end_idx + 1]

        splits.append(
            WalkForwardSplit(
                split_id=split_id,
                train=_build_window(train_dates),
                validation=_build_window(validation_dates),
                evaluation=_build_window(evaluation_dates),
                step_size=step_size,
                cost_grid=normalized_cost_grid,
            )
        )

        split_id += 1
        train_end_idx += step_size

    if not splits:
        raise ValueError("Window configuration does not yield any complete walk-forward splits.")

    return splits


def _normalize_dates(dates: Sequence[pd.Timestamp] | pd.Index) -> tuple[pd.Timestamp, ...]:
    normalized = tuple(pd.Timestamp(date) for date in dates)
    if not normalized:
        raise ValueError("dates must contain at least one timestamp.")
    if list(normalized) != sorted(normalized):
        raise ValueError("dates must be sorted in chronological order.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("dates must be unique.")
    return normalized


def _validate_window_sizes(
    *,
    sample_size: int,
    train_size: int,
    validation_size: int,
    evaluation_size: int,
    step_size: int,
) -> None:
    for name, value in {
        "train_size": train_size,
        "validation_size": validation_size,
        "evaluation_size": evaluation_size,
        "step_size": step_size,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be >= 1")

    minimum_required = train_size + validation_size + evaluation_size
    if sample_size < minimum_required:
        raise ValueError(
            "dates do not contain enough observations for one train/validation/evaluation split."
        )


def _build_window(dates: tuple[pd.Timestamp, ...]) -> WalkForwardWindow:
    return WalkForwardWindow(start=dates[0], end=dates[-1], dates=dates)
