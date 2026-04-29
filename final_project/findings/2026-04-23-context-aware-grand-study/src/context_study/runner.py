"""Main experiment runner for the context-aware grand study scaffold."""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from .analysis import (
    build_descriptor_winner_table,
    summarize_features_by_winner,
)
from .backtest import BacktestConfig, BacktestResult, run_backtest
from .candidates import NAMED_CANDIDATE_SETS, get_candidate_symbols
from .data_loader import PriceDataCache
from .descriptors import compute_universe_descriptors
from .experiment_artifacts import ExperimentArtifactWriter, build_trial_record
from .meta_controller import MetaController, compare_meta_controller_utility
from .protocol import generate_walk_forward_splits
from .rl_portfolio_env import PortfolioEnv
from .screens import apply_screen
from .strategies import (
    CMVTConfig,
    CMVTController,
    EWController,
    HierarchicalRiskController,
    MLMConfig,
    MLMController,
    MRHConfig,
    MRHController,
    MRSConfig,
    MRSController,
    RACConfig,
    RACController,
    ScreenedUniverseController,
    build_controllers,
)
from .universe import STATIC_UNIVERSES


# ---------------------------------------------------------------------------
# Universe definitions
# ---------------------------------------------------------------------------
DEFAULT_BENCHMARK_START = "2018-01-01"
DEFAULT_BENCHMARK_END = "2025-01-01"

STATIC_BENCHMARK_UNIVERSES = {
    "diversified_equity": {
        "symbols": [
            "AAPL", "MSFT", "AMZN", "GOOGL", "META",
            "NVDA", "TSLA", "JPM", "V", "JNJ",
            "WMT", "PG", "UNH", "HD", "MA",
            "BAC", "ABBV", "PFE", "KO", "PEP",
        ],
        "start": DEFAULT_BENCHMARK_START,
        "end": DEFAULT_BENCHMARK_END,
        "universe_kind": "legacy_static",
    },
    "sector_etf": {
        "symbols": list(STATIC_UNIVERSES["sector_etf"]),
        "start": DEFAULT_BENCHMARK_START,
        "end": DEFAULT_BENCHMARK_END,
        "universe_kind": "static",
    },
    "multi_asset": {
        "symbols": list(STATIC_UNIVERSES["multi_asset"]),
        "start": DEFAULT_BENCHMARK_START,
        "end": DEFAULT_BENCHMARK_END,
        "universe_kind": "static",
    },
}

CANDIDATE_BENCHMARK_UNIVERSES = tuple(
    universe_name
    for universe_name in ("liquid_us_equity_100", "liquid_us_equity_250", "liquid_us_equity_500")
    if universe_name in NAMED_CANDIDATE_SETS
)


def _build_candidate_universe_specs() -> dict[str, dict[str, object]]:
    return {
        universe_name: {
            "candidate_universe_name": universe_name,
            "start": DEFAULT_BENCHMARK_START,
            "end": DEFAULT_BENCHMARK_END,
            "universe_kind": "candidate_static",
        }
        for universe_name in CANDIDATE_BENCHMARK_UNIVERSES
    }


UNIVERSES = {
    **STATIC_BENCHMARK_UNIVERSES,
    **_build_candidate_universe_specs(),
}


def resolve_universe_spec(
    universe_name: str,
    *,
    data_dir: str | Path | None = None,
    symbol_limit: int | None = None,
) -> dict[str, object]:
    """Resolve a universe spec into a concrete symbol list and metadata."""
    if universe_name not in UNIVERSES:
        supported = sorted(UNIVERSES)
        raise ValueError(f"unknown universe_name={universe_name!r}; supported={supported}")

    raw_spec = UNIVERSES[universe_name]
    resolved_spec = dict(raw_spec)
    candidate_universe_name = raw_spec.get("candidate_universe_name")
    if candidate_universe_name is not None:
        resolved_spec["symbols"] = get_candidate_symbols(
            str(candidate_universe_name),
            data_dir=data_dir,
            limit=symbol_limit,
        )
    else:
        symbols = raw_spec.get("symbols", [])
        resolved_spec["symbols"] = list(symbols)
    return resolved_spec


def fetch_universe_prices(
    cache: PriceDataCache,
    universe_name: str,
    *,
    data_dir: str | Path | None = None,
    symbol_limit: int | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Resolve a universe and fetch its price panel through the configured cache."""
    universe_spec = resolve_universe_spec(
        universe_name,
        data_dir=data_dir,
        symbol_limit=symbol_limit,
    )
    prices = cache.fetch(
        universe_spec["symbols"],
        universe_spec["start"],
        universe_spec["end"],
    )
    return universe_spec, prices


def build_all_controllers() -> list:
    """Return all controller instances for the study."""
    return build_controllers()


def compute_and_save_descriptors(
    root: Path,
    universe_name: str,
    prices: pd.DataFrame,
    volume_panel: pd.DataFrame | None = None,
    window: int = 63,
) -> pd.DataFrame:
    """Compute and persist universe descriptors for a price panel."""
    descriptors = compute_universe_descriptors(
        price_panel=prices,
        volume_panel=volume_panel,
        window=window,
        min_periods=min(20, window),
    )
    output_path = root / "metrics" / "descriptors" / f"{universe_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptors.to_csv(output_path)
    return descriptors


def run_context_descriptor_pass(
    root: Path,
    cache: PriceDataCache,
    universe_names: tuple[str, ...] | list[str] | None = None,
    *,
    data_dir: str | Path | None = None,
    symbol_limit: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute descriptor tables for the currently configured benchmark universes."""
    outputs: dict[str, pd.DataFrame] = {}
    selected_universes = list(UNIVERSES) if universe_names is None else list(universe_names)
    for uni_name in selected_universes:
        _, prices = fetch_universe_prices(
            cache,
            uni_name,
            data_dir=data_dir,
            symbol_limit=symbol_limit,
        )
        outputs[uni_name] = compute_and_save_descriptors(root, uni_name, prices)
    return outputs


def run_model_based_study(
    root: Path,
    cache: PriceDataCache,
    universe_names: tuple[str, ...] | list[str] | None = None,
    *,
    data_dir: str | Path | None = None,
    symbol_limit: int | None = None,
) -> list[BacktestResult]:
    """Run all model-based controllers across all universes."""
    results = []
    controllers = build_all_controllers()
    selected_universes = list(UNIVERSES) if universe_names is None else list(universe_names)

    for uni_name in selected_universes:
        print(f"\n[universe] {uni_name}: fetching data...")
        _, prices = fetch_universe_prices(
            cache,
            uni_name,
            data_dir=data_dir,
            symbol_limit=symbol_limit,
        )
        print(f"  shape: {prices.shape}, dates: {prices.index[0]} to {prices.index[-1]}")

        for ctrl in controllers:
            print(f"  [controller] {ctrl.name}")
            res = run_backtest(prices, ctrl)
            res.universe = uni_name
            results.append(res)

    return results


def summarize_results(results: list[BacktestResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "universe": r.universe,
            "controller": r.controller_name,
            "ann_return": r.annualized_return,
            "ann_vol": r.annualized_vol,
            "sharpe": r.sharpe,
            "max_dd": r.max_drawdown,
            "calmar": r.calmar,
            "turnover": r.total_turnover,
            "win_rate": r.win_rate,
            "num_trades": r.num_trades,
        })
    return pd.DataFrame(rows)


def _compute_basic_metrics(returns: pd.Series) -> dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "calmar": 0.0,
        }
    ann_ret = float(returns.mean() * 252)
    ann_vol = float(returns.std() * np.sqrt(252))
    sharpe = ann_ret / (ann_vol + 1e-12)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min())
    calmar = ann_ret / (abs(max_dd) + 1e-12)
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": float(sharpe),
        "max_dd": max_dd,
        "calmar": float(calmar),
    }


def run_walk_forward_pilot(
    root: Path,
    cache: PriceDataCache,
    universe_name: str = "sector_etf",
    controller_limit: int = 2,
    cost_grid_bps: tuple[float, ...] = (0.0, 10.0, 25.0),
    *,
    data_dir: str | Path | None = None,
    symbol_limit: int | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a lightweight walk-forward benchmark pilot for the new protocol layer."""
    universe, prices = fetch_universe_prices(
        cache,
        universe_name,
        data_dir=data_dir,
        symbol_limit=symbol_limit,
    )
    pilot_output_dir = output_dir or (root / "metrics")
    pilot_output_dir.mkdir(parents=True, exist_ok=True)
    splits = generate_walk_forward_splits(
        dates=prices.index,
        train_size=min(252, len(prices) // 2),
        validation_size=63,
        evaluation_size=63,
        step_size=63,
        cost_grid=tuple(cost / 10_000 for cost in cost_grid_bps),
    )

    manifest = pd.DataFrame([split.as_dict() for split in splits])
    manifest_path = pilot_output_dir / "walk_forward_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    controllers = build_all_controllers()[:controller_limit]
    summary_rows = []
    subperiod_rows = []
    cost_rows = []

    for controller in controllers:
        for split in splits:
            for cost_bps in cost_grid_bps:
                result = run_backtest(
                    prices.loc[: split.evaluation.end],
                    controller,
                    config=BacktestConfig(transaction_cost_bps=cost_bps),
                )
                eval_returns = result.returns.loc[
                    (result.returns.index >= split.evaluation.start)
                    & (result.returns.index <= split.evaluation.end)
                ]
                metrics = _compute_basic_metrics(eval_returns)
                row = {
                    "universe": universe_name,
                    "controller": controller.name,
                    "split_id": split.split_id,
                    "cost_bps": cost_bps,
                    "train_end": split.train.end,
                    "validation_end": split.validation.end,
                    "evaluation_start": split.evaluation.start,
                    "evaluation_end": split.evaluation.end,
                    **metrics,
                }
                cost_rows.append(row)
                subperiod_rows.append(row)

        controller_rows = [r for r in cost_rows if r["controller"] == controller.name and r["cost_bps"] == 10.0]
        if controller_rows:
            df = pd.DataFrame(controller_rows)
            summary_rows.append(
                {
                    "universe": universe_name,
                    "controller": controller.name,
                    "splits": int(len(df)),
                    "mean_ann_return": float(df["ann_return"].mean()),
                    "mean_ann_vol": float(df["ann_vol"].mean()),
                    "mean_sharpe": float(df["sharpe"].mean()),
                    "worst_split_drawdown": float(df["max_dd"].min()),
                }
            )

    summary = pd.DataFrame(summary_rows)
    subperiod = pd.DataFrame(subperiod_rows)
    cost = pd.DataFrame(cost_rows)

    summary.to_csv(pilot_output_dir / "summary.csv", index=False)
    subperiod.to_csv(pilot_output_dir / "subperiod_summary.csv", index=False)
    cost.to_csv(pilot_output_dir / "cost_sensitivity.csv", index=False)
    return summary, subperiod, cost


def run_candidate_benchmark_pilot(
    root: Path,
    cache: PriceDataCache,
    universe_name: str = "liquid_us_equity_100",
    controller_limit: int = 2,
    cost_grid_bps: tuple[float, ...] = (0.0, 10.0, 25.0),
    *,
    data_dir: str | Path | None = None,
    symbol_limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a small controller subset on one candidate-based large universe."""
    if universe_name not in CANDIDATE_BENCHMARK_UNIVERSES:
        supported = sorted(CANDIDATE_BENCHMARK_UNIVERSES)
        raise ValueError(
            f"candidate benchmark pilot requires a candidate universe; got {universe_name!r}; supported={supported}"
        )

    output_dir = root / "metrics" / "candidate_pilot" / universe_name
    return run_walk_forward_pilot(
        root=root,
        cache=cache,
        universe_name=universe_name,
        controller_limit=controller_limit,
        cost_grid_bps=cost_grid_bps,
        data_dir=data_dir,
        symbol_limit=symbol_limit,
        output_dir=output_dir,
    )


def _holding_budget_from_screen(screen_membership: pd.DataFrame) -> int | str:
    if screen_membership.empty:
        return 0
    return int(screen_membership["symbol"].nunique())


def run_screened_candidate_benchmark_pilot(
    *,
    root: Path,
    cache: PriceDataCache,
    universe_name: str = "liquid_us_equity_100",
    screen_rules: tuple[str, ...] = ("momentum_21_top20",),
    controller_limit: int = 2,
    cost_grid_bps: tuple[float, ...] = (10.0,),
    data_dir: str | Path | None = None,
    symbol_limit: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Run walk-forward benchmarks after applying lagged screen rules per split."""
    if universe_name not in CANDIDATE_BENCHMARK_UNIVERSES:
        supported = sorted(CANDIDATE_BENCHMARK_UNIVERSES)
        raise ValueError(
            f"screened candidate benchmark requires a candidate universe; got {universe_name!r}; supported={supported}"
        )

    _, prices = fetch_universe_prices(
        cache,
        universe_name,
        data_dir=data_dir,
        symbol_limit=symbol_limit,
    )
    output_dir = root / "metrics" / "screened_candidate_pilot" / universe_name
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = generate_walk_forward_splits(
        dates=prices.index,
        train_size=min(252, len(prices) // 2),
        validation_size=63,
        evaluation_size=63,
        step_size=63,
        cost_grid=tuple(cost / 10_000 for cost in cost_grid_bps),
    )
    manifest = pd.DataFrame([split.as_dict() for split in splits])
    manifest.to_csv(output_dir / "walk_forward_manifest.csv", index=False)

    controllers = build_all_controllers()[:controller_limit]
    summary_rows: list[dict[str, object]] = []
    subperiod_rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, object]] = []
    ledger_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    for screen_rule in screen_rules:
        for controller in controllers:
            for split in splits:
                screen = apply_screen(
                    prices.loc[: split.train.end],
                    screen_rule=screen_rule,
                    as_of_date=split.train.end,
                )
                if screen.empty:
                    continue
                selected_symbols = screen["symbol"].tolist()
                for screen_row in screen.to_dict("records"):
                    membership_rows.append(
                        {
                            "universe": universe_name,
                            "screen_rule": screen_rule,
                            "split_id": split.split_id,
                            **screen_row,
                        }
                    )
                screened_prices = prices.loc[: split.evaluation.end, selected_symbols]
                holding_budget = _holding_budget_from_screen(screen)
                for cost_bps in cost_grid_bps:
                    result = run_backtest(
                        screened_prices,
                        controller,
                        config=BacktestConfig(transaction_cost_bps=cost_bps),
                    )
                    eval_returns = result.returns.loc[
                        (result.returns.index >= split.evaluation.start)
                        & (result.returns.index <= split.evaluation.end)
                    ]
                    metrics = _compute_basic_metrics(eval_returns)
                    row = {
                        "universe": universe_name,
                        "screen_rule": screen_rule,
                        "holding_budget": holding_budget,
                        "controller": controller.name,
                        "split_id": split.split_id,
                        "cost_bps": cost_bps,
                        "train_start": split.train.start,
                        "train_end": split.train.end,
                        "validation_end": split.validation.end,
                        "evaluation_start": split.evaluation.start,
                        "evaluation_end": split.evaluation.end,
                        "selected_symbols": " ".join(selected_symbols),
                        "turnover": result.total_turnover,
                        **metrics,
                    }
                    cost_rows.append(row)
                    subperiod_rows.append(row)
                    trial_rows.append(
                        build_trial_record(
                            run_id="screened_candidate_pilot",
                            split_id=split.split_id,
                            universe_name=universe_name,
                            universe_rule="fixed_current_constituents_symbol_limited",
                            screen_rule=screen_rule,
                            holding_budget=holding_budget,
                            controller=controller.name,
                            cost_bps=cost_bps,
                            train_start=split.train.start,
                            train_end=split.train.end,
                            evaluation_start=split.evaluation.start,
                            evaluation_end=split.evaluation.end,
                            gross_return=metrics["ann_return"],
                            net_return=metrics["ann_return"],
                            turnover=result.total_turnover,
                            sharpe=metrics["sharpe"],
                            max_drawdown=metrics["max_dd"],
                            selected_for_paper=False,
                        )
                    )

            controller_rows = [
                row
                for row in cost_rows
                if row["controller"] == controller.name
                and row["screen_rule"] == screen_rule
                and row["cost_bps"] == 10.0
            ]
            if controller_rows:
                df = pd.DataFrame(controller_rows)
                summary_rows.append(
                    {
                        "universe": universe_name,
                        "screen_rule": screen_rule,
                        "holding_budget": int(df["holding_budget"].median()),
                        "controller": controller.name,
                        "splits": int(len(df)),
                        "mean_ann_return": float(df["ann_return"].mean()),
                        "mean_ann_vol": float(df["ann_vol"].mean()),
                        "mean_sharpe": float(df["sharpe"].mean()),
                        "worst_split_drawdown": float(df["max_dd"].min()),
                        "mean_turnover": float(df["turnover"].mean()),
                    }
                )
                ledger_rows.append(
                    {
                        "run_id": "screened_candidate_pilot",
                        "timestamp_utc": timestamp,
                        "universe_name": universe_name,
                        "universe_rule": "fixed_current_constituents_symbol_limited",
                        "screen_rule": screen_rule,
                        "holding_budget": int(df["holding_budget"].median()),
                        "controller": controller.name,
                        "hyperparameters": "{}",
                        "cost_bps": 10.0,
                        "selected_for_paper": False,
                    }
                )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        best_idx = summary.groupby(["universe", "screen_rule"])["mean_sharpe"].idxmax()
        summary.loc[:, "selected_for_paper"] = False
        summary.loc[best_idx, "selected_for_paper"] = True
        for ledger_row in ledger_rows:
            match = summary.loc[
                (summary["controller"] == ledger_row["controller"])
                & (summary["screen_rule"] == ledger_row["screen_rule"])
            ]
            ledger_row["selected_for_paper"] = bool(match["selected_for_paper"].any())
    subperiod = pd.DataFrame(subperiod_rows)
    cost = pd.DataFrame(cost_rows)
    screen_membership = pd.DataFrame(membership_rows)

    summary.to_csv(output_dir / "summary.csv", index=False)
    subperiod.to_csv(output_dir / "subperiod_summary.csv", index=False)
    cost.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    screen_membership.to_csv(output_dir / "screen_membership.csv", index=False)
    ExperimentArtifactWriter(metrics_dir=output_dir).write_model_selection_ledger(ledger_rows)
    ExperimentArtifactWriter(metrics_dir=output_dir).write_all_trials(trial_rows)

    return {
        "summary": summary,
        "subperiod": subperiod,
        "cost": cost,
        "screen_membership": screen_membership,
    }


def run_descriptor_winner_analysis(
    *,
    root: Path,
    universe_name: str,
    descriptors: pd.DataFrame,
    subperiod_summary: pd.DataFrame,
    metric: str = "sharpe",
    cost_bps_filter: float | None = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and persist descriptor-to-winner analysis artifacts."""
    analysis_dir = root / "metrics" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    filtered_subperiod = subperiod_summary.copy()
    if cost_bps_filter is not None and "cost_bps" in filtered_subperiod.columns:
        filtered_subperiod = filtered_subperiod.loc[filtered_subperiod["cost_bps"] == cost_bps_filter].copy()
        if filtered_subperiod.empty:
            raise ValueError(f"No rows remain after applying cost_bps_filter={cost_bps_filter}.")

    analysis_table = build_descriptor_winner_table(
        descriptors=descriptors,
        performance=filtered_subperiod,
        metric=metric,
    )
    feature_summary = summarize_features_by_winner(analysis_table)

    analysis_table.to_csv(analysis_dir / f"{universe_name}_winner_table.csv", index=False)
    feature_summary.to_csv(analysis_dir / f"{universe_name}_feature_summary.csv", index=False)
    return analysis_table, feature_summary


def run_meta_controller_pilot(
    *,
    root: Path,
    universe_name: str,
    analysis_table: pd.DataFrame,
    feature_columns: list[str] | None = None,
    train_fraction: float = 0.67,
) -> dict[str, object]:
    """Fit a first supervised meta-controller and persist predictions/evaluation."""
    if analysis_table.empty:
        raise ValueError("analysis_table must not be empty")
    if "winner_label" not in analysis_table.columns:
        raise ValueError("analysis_table must contain winner_label")

    numeric_candidates = [
        column for column in analysis_table.columns
        if column not in {"winner_label", "winner_score", "split_id", "cost_bps", "period", "universe", "descriptor_date"}
        and not column.endswith("_start")
        and not column.endswith("_end")
        and pd.api.types.is_numeric_dtype(analysis_table[column])
    ]
    selected_features = numeric_candidates if feature_columns is None else feature_columns
    feature_frame = analysis_table.loc[:, selected_features].astype(float)
    labels = analysis_table["winner_label"].astype(str)

    split_point = max(1, int(len(feature_frame) * train_fraction))
    split_point = min(split_point, len(feature_frame) - 1) if len(feature_frame) > 1 else len(feature_frame)
    if split_point <= 0 or split_point >= len(feature_frame):
        raise ValueError("analysis_table must contain at least two rows for train/test evaluation")

    train_X = feature_frame.iloc[:split_point]
    train_y = labels.iloc[:split_point]
    test_X = feature_frame.iloc[split_point:]
    test_y = labels.iloc[split_point:]

    if train_y.nunique() < 2:
        majority_label = str(train_y.iloc[0])
        predictions = pd.Series(majority_label, index=test_X.index, name="predicted_controller", dtype=object)
        accuracy = float((predictions == test_y).mean()) if len(test_y) else 1.0
        evaluation = {
            "accuracy": accuracy,
            "n_samples": int(len(test_y)),
            "labels": [majority_label],
            "predictions": predictions,
        }
    else:
        model = MetaController().fit(train_X, train_y)
        predictions = model.predict(test_X)
        evaluation = model.evaluate(test_X, test_y)

    meta_dir = root / "metrics" / "meta_controller"
    meta_dir.mkdir(parents=True, exist_ok=True)

    prediction_frame = pd.DataFrame(
        {
            "winner_label": test_y.values,
            "predicted_controller": predictions.values,
        },
        index=test_X.index,
    )
    prediction_frame.to_csv(meta_dir / f"{universe_name}_predictions.csv")

    result = {
        "universe": universe_name,
        "n_train": int(len(train_X)),
        "n_test": int(len(test_X)),
        "feature_columns": list(selected_features),
        "accuracy": float(evaluation["accuracy"]),
        "labels": list(evaluation["labels"]),
    }
    with open(meta_dir / f"{universe_name}_evaluation.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, default=str, indent=2)
    return result


def run_candidate_analysis_pipeline(
    *,
    root: Path,
    cache: PriceDataCache,
    universe_name: str = "liquid_us_equity_100",
    controller_limit: int = 2,
    cost_grid_bps: tuple[float, ...] = (0.0, 10.0, 25.0),
    data_dir: str | Path | None = None,
    symbol_limit: int | None = None,
) -> dict[str, object]:
    """Run candidate benchmark, descriptor alignment, and meta-controller pilot end-to-end."""
    summary, subperiod, cost = run_candidate_benchmark_pilot(
        root=root,
        cache=cache,
        universe_name=universe_name,
        controller_limit=controller_limit,
        cost_grid_bps=cost_grid_bps,
        data_dir=data_dir,
        symbol_limit=symbol_limit,
    )
    _, prices = fetch_universe_prices(
        cache,
        universe_name,
        data_dir=data_dir,
        symbol_limit=symbol_limit,
    )
    descriptors = compute_and_save_descriptors(root, universe_name, prices)
    analysis_table, feature_summary = run_descriptor_winner_analysis(
        root=root,
        universe_name=universe_name,
        descriptors=descriptors,
        subperiod_summary=subperiod,
    )
    meta = run_meta_controller_pilot(
        root=root,
        universe_name=universe_name,
        analysis_table=analysis_table,
    )
    return {
        "summary": summary,
        "subperiod": subperiod,
        "cost": cost,
        "analysis": analysis_table,
        "feature_summary": feature_summary,
        "meta": meta,
    }


def train_rl_policy(
    prices: pd.DataFrame,
    n_updates: int = 100,
    horizon: int = 128,
    *,
    seed: int = 7,
    learning_rate: float = 0.001,
    exploration_sigma: float = 0.3,
) -> dict:
    """Train a simple policy-gradient controller on portfolio allocation.

    The controller is deliberately small and reproducible: a linear Gaussian
    policy maps environment observations to action logits, actions are projected
    to the simplex by the environment, and vanilla REINFORCE updates the linear
    coefficients.  This is not intended to be a production RL algorithm; it is a
    controlled first-class RL baseline for the study's controller hierarchy.
    """
    clean_prices = prices.dropna(axis=1, how="any")
    returns = clean_prices.pct_change(fill_method=None).dropna().values
    if returns.shape[0] < 60 or returns.shape[1] < 2:
        raise ValueError("RL training requires at least 60 return observations and two assets")

    rng = np.random.default_rng(seed)
    env = PortfolioEnv(returns=returns)
    obs_dim = env.obs_size
    act_dim = env.act_size
    theta = rng.normal(0.0, 0.01, size=(obs_dim, act_dim))
    baseline = 0.0
    gamma = 0.99
    reward_history: list[float] = []

    for update in range(n_updates):
        obs = env.reset(seed=seed + update)
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rewards: list[float] = []

        for _ in range(horizon):
            logits = obs @ theta
            action = logits + rng.normal(0.0, exploration_sigma, size=act_dim)
            observations.append(obs.copy())
            actions.append(action.copy())
            obs, reward, done, _, _ = env.step(action)
            rewards.append(float(reward))
            if done:
                break

        if not rewards:
            continue

        discounted = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + gamma * running
            discounted.insert(0, running)
        discounted_arr = np.asarray(discounted, dtype=float)
        baseline = 0.95 * baseline + 0.05 * float(discounted_arr.mean())

        grad = np.zeros_like(theta)
        variance = exploration_sigma ** 2
        for obs_t, action_t, ret_t in zip(observations, actions, discounted_arr):
            logits_t = obs_t @ theta
            score = (action_t - logits_t) / variance
            grad += (ret_t - baseline) * np.outer(obs_t, score)

        grad_norm = float(np.linalg.norm(grad))
        if np.isfinite(grad_norm) and grad_norm > 1.0:
            grad /= grad_norm
        theta += learning_rate * grad / max(1, len(discounted_arr))
        reward_history.append(float(np.sum(rewards)))

    return {
        "theta": theta,
        "mean_reward": float(np.mean(reward_history[-20:])) if reward_history else 0.0,
        "reward_history": reward_history,
        "n_updates": n_updates,
        "asset_columns": list(clean_prices.columns),
    }


def run_rl_evaluation(
    prices: pd.DataFrame,
    theta: np.ndarray,
    *,
    transaction_cost: float = 0.001,
    rebalance_every: int = 21,
) -> BacktestResult:
    """Evaluate a learned linear policy with chronological environment state.

    The previous scaffold reset the environment on every iteration, which made
    the learned policy effectively observe the same state repeatedly.  This
    version advances the environment exactly once per market step and records
    realized turnover/cost statistics from the environment weights.
    """
    clean_prices = prices.dropna(axis=1, how="any")
    returns = clean_prices.pct_change(fill_method=None).dropna()
    returns_arr = returns.values
    dates = returns.index
    n_assets = returns_arr.shape[1]
    if theta.shape[1] != n_assets:
        raise ValueError(f"theta action dimension {theta.shape[1]} does not match {n_assets} assets")

    env = PortfolioEnv(
        returns=returns_arr,
        rebalance_every=rebalance_every,
        transaction_cost=transaction_cost,
    )
    obs = env.reset(seed=42)
    capital = 1_000_000.0
    equity = [capital]
    weights_hist = [env.weights.copy()]
    trades: list[dict] = []
    total_turnover = 0.0
    wins = 0

    while True:
        current_t = env.t
        if current_t >= len(returns_arr) - 1:
            break
        prev_weights = env.weights.copy()
        logits = obs @ theta
        next_obs, reward, done, _, _ = env.step(logits)
        capital = env.port_value * 1_000_000.0
        equity.append(capital)
        weights_hist.append(env.weights.copy())

        turnover = float(np.sum(np.abs(env.weights - prev_weights)))
        if current_t % rebalance_every == 0 and turnover > 1e-8:
            total_turnover += turnover
            trades.append({
                "date": dates[min(current_t, len(dates) - 1)],
                "turnover": turnover,
                "cost": turnover * transaction_cost,
            })
            if reward > 0:
                wins += 1
        obs = next_obs
        if done:
            break

    equity_index = dates[:len(equity)]
    equity_series = pd.Series(equity, index=equity_index)
    ret_series = equity_series.pct_change(fill_method=None).dropna()
    ann_ret = float(ret_series.mean() * 252) if not ret_series.empty else 0.0
    ann_vol = float(ret_series.std() * np.sqrt(252)) if len(ret_series) > 1 else 0.0
    sharpe = ann_ret / (ann_vol + 1e-12)
    cummax = equity_series.cummax()
    max_dd = float(((equity_series - cummax) / cummax).min()) if not equity_series.empty else 0.0
    calmar = ann_ret / (abs(max_dd) + 1e-12)
    win_rate = wins / len(trades) if trades else 0.0

    return BacktestResult(
        controller_name="RL_Learned",
        universe="",
        equity_curve=equity_series,
        returns=ret_series,
        weights_history=pd.DataFrame(weights_hist, index=dates[:len(weights_hist)], columns=clean_prices.columns),
        trades=trades,
        annualized_return=ann_ret,
        annualized_vol=ann_vol,
        sharpe=float(sharpe),
        max_drawdown=max_dd,
        calmar=float(calmar),
        total_turnover=float(total_turnover),
        win_rate=float(win_rate),
        num_trades=len(trades),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cache = PriceDataCache(cache_dir=root / "data")
    static_universe_names = tuple(STATIC_BENCHMARK_UNIVERSES)

    # 1. Model-based controllers
    print("=" * 60)
    print("CONTEXT-AWARE GRAND STUDY")
    print("Dynamic Portfolio Allocation as a System Optimization Problem")
    print("=" * 60)

    print("[descriptors] Computing benchmark-universe descriptor tables...")
    descriptor_tables = run_context_descriptor_pass(root, cache, universe_names=static_universe_names)
    for universe_name, descriptor_table in descriptor_tables.items():
        latest = descriptor_table.dropna(how="all").tail(1)
        if latest.empty:
            print(f"  {universe_name:20s} -> no descriptor row yet")
        else:
            latest_row = latest.iloc[0]
            print(
                f"  {universe_name:20s} -> corr={latest_row['avg_pairwise_corr']:.3f}, "
                f"pc1={latest_row['first_pc_share']:.3f}, trend={latest_row['trend_persistence']:.3f}"
            )

    # 1. Walk-forward pilot benchmark
    print("\n[pilot] Running walk-forward benchmark on a small controller subset...")
    wf_summary, wf_subperiod, wf_cost = run_walk_forward_pilot(root, cache)
    print(f"  walk-forward summary rows: {len(wf_summary)}")
    print(f"  walk-forward split-cost rows: {len(wf_cost)}")

    print("\n[candidate-pilot] Running candidate-universe benchmark pilot...")
    candidate_summary, _, candidate_cost = run_candidate_benchmark_pilot(
        root,
        cache,
        universe_name="liquid_us_equity_100",
        controller_limit=2,
    )
    print(f"  candidate summary rows: {len(candidate_summary)}")
    print(f"  candidate split-cost rows: {len(candidate_cost)}")

    # 2. Legacy full-sample model-based benchmark
    results = run_model_based_study(root, cache, universe_names=static_universe_names)
    summary = summarize_results(results)

    summary_path = root / "metrics" / "full_sample_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"\n[saved] {summary_path}")

    # Print best by universe
    print("\n" + "=" * 60)
    print("BEST CONTROLLER BY UNIVERSE (Sharpe)")
    print("=" * 60)
    for uni in summary["universe"].unique():
        sub = summary[summary["universe"] == uni]
        best = sub.loc[sub["sharpe"].idxmax()]
        print(f"  {uni:20s} -> {best['controller']:30s}  Sharpe={best['sharpe']:.3f}")

    # 2. RL training on one universe (multi-asset as most diverse)
    print("\n" + "=" * 60)
    print("REINFORCEMENT LEARNING CONTROLLER")
    print("=" * 60)
    multi_prices = cache.fetch(
        UNIVERSES["multi_asset"]["symbols"],
        UNIVERSES["multi_asset"]["start"],
        UNIVERSES["multi_asset"]["end"],
    )

    trained_rl = train_rl_policy(
        multi_prices,
        n_updates=200,
        horizon=256,
        seed=7,
        learning_rate=0.001,
        exploration_sigma=0.3,
    )
    theta = trained_rl["theta"]
    reward_history = trained_rl["reward_history"]
    print(f"\n[RL] Final mean reward (last 20): {np.mean(reward_history[-20:]):.6f}")

    # Evaluate RL policy in backtest framework
    print("\n[RL] Evaluating learned policy...")
    rl_res = run_rl_evaluation(multi_prices, theta)
    rl_res.universe = "multi_asset"

    print(f"  RL Policy: Sharpe={rl_res.sharpe:.3f}, Return={rl_res.annualized_return:.3f}, Vol={rl_res.annualized_vol:.3f}")

    # Save RL result
    rl_summary = pd.DataFrame([{
        "universe": rl_res.universe,
        "controller": rl_res.controller_name,
        "ann_return": rl_res.annualized_return,
        "ann_vol": rl_res.annualized_vol,
        "sharpe": rl_res.sharpe,
        "max_dd": rl_res.max_drawdown,
        "calmar": rl_res.calmar,
        "turnover": rl_res.total_turnover,
        "win_rate": rl_res.win_rate,
        "num_trades": rl_res.num_trades,
    }])
    rl_path = root / "metrics" / "rl_result.csv"
    rl_summary.to_csv(rl_path, index=False)

    # Save equity curves
    for r in results:
        curve_path = root / "metrics" / "equity" / f"{r.universe}_{r.controller_name}.csv"
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        r.equity_curve.to_csv(curve_path)

    # Save RL equity
    rl_curve_path = root / "metrics" / "equity" / "multi_asset_RL_Learned.csv"
    rl_res.equity_curve.to_csv(rl_curve_path)

    print("\n[done] All experiments complete.")


if __name__ == "__main__":
    main()
