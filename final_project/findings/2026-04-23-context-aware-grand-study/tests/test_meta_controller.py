from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from context_study.meta_controller import MetaController, compare_meta_controller_utility


def make_supervised_dataset() -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame(
        {
            "trend_strength": [0.90, 0.85, 0.80, 0.20, 0.15, 0.10, 0.55, 0.50, 0.45],
            "volatility": [0.10, 0.12, 0.11, 0.35, 0.32, 0.30, 0.18, 0.20, 0.22],
            "dispersion": [0.15, 0.18, 0.20, 0.45, 0.40, 0.42, 0.25, 0.28, 0.30],
        },
        index=pd.date_range("2021-01-01", periods=9, freq="B"),
    )
    labels = pd.Series(
        [
            "trend",
            "trend",
            "trend",
            "revert",
            "revert",
            "revert",
            "balanced",
            "balanced",
            "balanced",
        ],
        index=features.index,
        name="winner",
    )
    return features, labels


def test_meta_controller_fit_predict_and_evaluate() -> None:
    features, labels = make_supervised_dataset()
    model = MetaController()

    fitted = model.fit(features, labels)
    predictions = fitted.predict(features)
    metrics = fitted.evaluate(features, labels)

    assert fitted is model
    assert list(predictions.index) == list(features.index)
    assert predictions.tolist() == labels.tolist()
    assert metrics["n_samples"] == len(features)
    assert metrics["accuracy"] == 1.0
    assert metrics["labels"] == sorted(labels.unique().tolist())


def test_meta_controller_predictions_stay_within_training_label_set() -> None:
    features, labels = make_supervised_dataset()
    train_X = features.iloc[:6]
    train_y = labels.iloc[:6]
    test_X = features.iloc[6:]

    model = MetaController().fit(train_X, train_y)
    predictions = model.predict(test_X)

    assert set(predictions.unique()).issubset(set(train_y.unique()))


def test_compare_meta_controller_utility_returns_sensible_schema() -> None:
    chosen_labels = pd.Series(
        ["trend", "revert", "trend", "balanced"],
        index=pd.date_range("2022-01-03", periods=4, freq="B"),
        name="predicted_controller",
    )
    realized_panel = pd.DataFrame(
        {
            "trend": [0.02, -0.01, 0.015, 0.00],
            "revert": [0.01, 0.03, -0.02, 0.01],
            "balanced": [0.015, 0.01, 0.005, 0.005],
        },
        index=chosen_labels.index,
    )

    comparison = compare_meta_controller_utility(
        chosen_labels=chosen_labels,
        realized_panel=realized_panel,
        metric_name="net_return",
    )

    assert set(comparison.keys()) == {
        "metric_name",
        "n_periods",
        "meta_controller_total",
        "best_fixed_controller",
        "best_fixed_total",
        "relative_gain",
        "chosen_path",
    }
    assert comparison["metric_name"] == "net_return"
    assert comparison["n_periods"] == len(chosen_labels)
    assert comparison["best_fixed_controller"] in realized_panel.columns
    assert isinstance(comparison["chosen_path"], pd.Series)
    assert comparison["chosen_path"].index.equals(chosen_labels.index)
    assert np.isclose(comparison["chosen_path"].sum(), comparison["meta_controller_total"])
