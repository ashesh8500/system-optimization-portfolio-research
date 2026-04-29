from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SKLEARN_AVAILABLE = True


class MetaController:
    """Supervised controller selector from descriptor features to controller label.

    scikit-learn is a required dependency for this module. The implementation
    intentionally fails at import time if scikit-learn is absent instead of
    substituting another classifier with different behavior.
    """

    def __init__(self, *, random_state: int = 0, max_iter: int = 1000) -> None:
        self.random_state = random_state
        self.max_iter = max_iter
        self.feature_columns_: list[str] | None = None
        self.labels_: list[str] | None = None
        self.model_: Pipeline | None = None

    def fit(self, features: pd.DataFrame, winning_labels: pd.Series) -> "MetaController":
        X, y = self._validate_training_data(features, winning_labels)
        self.feature_columns_ = list(X.columns)
        self.labels_ = sorted(pd.Index(y).astype(str).unique().tolist())
        self.model_ = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self.model_.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=str))
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        X = self._validate_prediction_features(features)
        predictions = self.model_.predict(X.to_numpy(dtype=float))
        return pd.Series(predictions, index=X.index, name="predicted_controller", dtype=object)

    def evaluate(self, features: pd.DataFrame, winning_labels: pd.Series) -> dict[str, Any]:
        X, y = self._validate_training_data(features, winning_labels)
        predictions = self.predict(X)
        truth = y.astype(str)
        accuracy = float((predictions == truth).mean()) if len(truth) else np.nan
        return {
            "accuracy": accuracy,
            "n_samples": int(len(truth)),
            "labels": list(self.labels_ or sorted(truth.unique().tolist())),
            "predictions": predictions,
        }

    def _validate_training_data(self, features: pd.DataFrame, winning_labels: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame.")
        if not isinstance(winning_labels, pd.Series):
            raise TypeError("winning_labels must be a pandas Series.")
        if features.empty:
            raise ValueError("features must not be empty.")
        if len(features) != len(winning_labels):
            raise ValueError("features and winning_labels must have the same length.")

        X = features.astype(float)
        y = winning_labels.reindex(features.index)
        if y.isna().any():
            raise ValueError("winning_labels must align with the feature index without missing values.")
        return X, y.astype(str)

    def _validate_prediction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.model_ is None or self.feature_columns_ is None:
            raise RuntimeError("MetaController must be fitted before prediction.")
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame.")

        missing_columns = [column for column in self.feature_columns_ if column not in features.columns]
        if missing_columns:
            raise ValueError(f"Prediction features missing required columns: {missing_columns}")

        return features.loc[:, self.feature_columns_].astype(float)


def compare_meta_controller_utility(
    *,
    chosen_labels: pd.Series,
    realized_panel: pd.DataFrame,
    metric_name: str = "utility",
) -> dict[str, Any]:
    """Compare realized meta-controller utility against the best fixed controller."""
    if not isinstance(chosen_labels, pd.Series):
        raise TypeError("chosen_labels must be a pandas Series.")
    if not isinstance(realized_panel, pd.DataFrame):
        raise TypeError("realized_panel must be a pandas DataFrame.")
    if realized_panel.empty:
        raise ValueError("realized_panel must not be empty.")

    aligned_panel = realized_panel.reindex(chosen_labels.index)
    if aligned_panel.isna().all(axis=None):
        raise ValueError("realized_panel does not overlap with chosen_labels index.")

    missing_labels = sorted(set(chosen_labels.astype(str)) - set(aligned_panel.columns))
    if missing_labels:
        raise ValueError(f"chosen_labels contains controllers absent from realized_panel: {missing_labels}")

    chosen_path = pd.Series(
        [aligned_panel.loc[idx, str(label)] for idx, label in chosen_labels.astype(str).items()],
        index=chosen_labels.index,
        name="chosen_utility",
        dtype=float,
    )
    fixed_totals = aligned_panel.sum(axis=0)
    best_fixed_controller = str(fixed_totals.idxmax())
    best_fixed_total = float(fixed_totals.loc[best_fixed_controller])
    meta_total = float(chosen_path.sum())

    relative_gain = (meta_total - best_fixed_total) / (abs(best_fixed_total) + 1e-12)
    return {
        "metric_name": metric_name,
        "n_periods": int(len(chosen_path)),
        "meta_controller_total": meta_total,
        "best_fixed_controller": best_fixed_controller,
        "best_fixed_total": best_fixed_total,
        "relative_gain": float(relative_gain),
        "chosen_path": chosen_path,
    }
