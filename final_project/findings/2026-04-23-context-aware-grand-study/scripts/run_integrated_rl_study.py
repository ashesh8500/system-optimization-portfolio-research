from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_study.hierarchical_rl_router import (  # noqa: E402
    build_router_reward_panel,
    evaluate_routing_baselines,
    run_router_repeated_study,
)

METRICS = ROOT / "metrics"
FIGURES = ROOT / "figures"
OUT = METRICS / "integrated_rl_study"
OUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    {"config_name": "conservative", "n_updates": 220, "episodes_per_update": 8, "learning_rate": 0.015, "gamma": 0.96},
    {"config_name": "baseline", "n_updates": 320, "episodes_per_update": 10, "learning_rate": 0.030, "gamma": 0.98},
    {"config_name": "aggressive", "n_updates": 420, "episodes_per_update": 12, "learning_rate": 0.060, "gamma": 0.99},
]
SEEDS = [3, 7, 11, 17, 23, 31, 43]


def summarize_trials(trials: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["scope", "config_name"]
    summary = (
        trials.groupby(group_cols)
        .agg(
            n_trials=("seed", "count"),
            rl_mean=("rl_total_reward", "mean"),
            rl_std=("rl_total_reward", "std"),
            rl_min=("rl_total_reward", "min"),
            rl_max=("rl_total_reward", "max"),
            rl_vs_train_selected_fixed_mean=("rl_vs_train_selected_fixed", "mean"),
            rl_vs_train_selected_fixed_min=("rl_vs_train_selected_fixed", "min"),
            rl_vs_trailing_window_mean=("rl_vs_trailing_window", "mean"),
            rl_vs_random_mean_mean=("rl_vs_random_mean", "mean"),
            train_selected_fixed_total_reward=("train_selected_fixed_total_reward", "first"),
            trailing_window_total_reward=("trailing_window_total_reward", "first"),
            test_best_fixed_total_reward=("test_best_fixed_total_reward", "first"),
            period_oracle_total_reward=("period_oracle_total_reward", "first"),
            random_mean_total_reward=("random_mean_total_reward", "first"),
            n_test_periods=("n_test_periods", "first"),
        )
        .reset_index()
    )
    return summary


def best_by_scope(config_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, group in config_summary.groupby("scope"):
        idx = group["rl_mean"].idxmax()
        rows.append(config_summary.loc[idx].to_dict())
    return pd.DataFrame(rows)


def plot_scope_comparison(best: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = range(len(best))
    width = 0.18
    ax.bar([i - 1.5 * width for i in x], best["rl_mean"], width=width, label="RL mean over seeds")
    ax.bar([i - 0.5 * width for i in x], best["train_selected_fixed_total_reward"], width=width, label="Train-selected fixed")
    ax.bar([i + 0.5 * width for i in x], best["trailing_window_total_reward"], width=width, label="Trailing-window router")
    ax.bar([i + 1.5 * width for i in x], best["random_mean_total_reward"], width=width, label="Random mean")
    ax.errorbar(list(x), best["rl_mean"], yerr=best["rl_std"].fillna(0.0), fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(list(x), best["scope"], rotation=25, ha="right")
    ax.set_ylabel("Test-period total reward")
    ax.set_title("Integrated hierarchical RL study: best RL configuration by scope")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "integrated_rl_study_scope_comparison.png", dpi=180)
    plt.close(fig)


def plot_hyperparameter_surface(config_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for scope, group in config_summary.groupby("scope"):
        ax.plot(group["config_name"], group["rl_vs_train_selected_fixed_mean"], marker="o", label=scope)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("RL mean reward minus train-selected fixed reward")
    ax.set_title("RL sensitivity across policy-gradient configurations")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "integrated_rl_study_config_sensitivity.png", dpi=180)
    plt.close(fig)


def main() -> None:
    data = pd.read_csv(METRICS / "screened_core_subperiod_summary.csv")
    scopes: list[tuple[str, pd.DataFrame]] = [("combined", data)]
    for universe in sorted(data["universe"].unique()):
        scopes.append((str(universe), data.loc[data["universe"] == universe].copy()))

    all_trials = []
    baseline_rows = []
    for scope, scope_data in scopes:
        panel = build_router_reward_panel(scope_data, cost_bps=10.0)
        train_fraction = 0.68 if len(panel.periods) > 12 else 0.60
        print(f"[integrated-rl] scope={scope} periods={len(panel.periods)} arms={len(panel.arm_labels)} train_fraction={train_fraction}", flush=True)
        baselines = evaluate_routing_baselines(
            panel,
            train_fraction=train_fraction,
            trailing_window=3,
            transaction_cost_penalty=0.0005,
            n_random_paths=1024,
            seed=101,
        )
        baseline_rows.append(
            {
                "scope": scope,
                "periods": len(panel.periods),
                "arms": len(panel.arm_labels),
                "train_fraction": train_fraction,
                "n_test_periods": baselines["n_test_periods"],
                "train_selected_fixed_arm": baselines["train_selected_fixed_arm"],
                "train_selected_fixed_total_reward": baselines["train_selected_fixed_total_reward"],
                "trailing_window_total_reward": baselines["trailing_window_total_reward"],
                "test_best_fixed_arm": baselines["test_best_fixed_arm"],
                "test_best_fixed_total_reward": baselines["test_best_fixed_total_reward"],
                "period_oracle_total_reward": baselines["period_oracle_total_reward"],
                "random_mean_total_reward": baselines["random_mean_total_reward"],
                "random_p05_total_reward": baselines["random_p05_total_reward"],
                "random_p95_total_reward": baselines["random_p95_total_reward"],
            }
        )
        baselines["baseline_actions"].to_csv(OUT / f"{scope}_baseline_actions.csv", index=False)
        trials = run_router_repeated_study(
            panel,
            scope=scope,
            configs=CONFIGS,
            seeds=SEEDS,
            train_fraction=train_fraction,
            transaction_cost_penalty=0.0005,
            trailing_window=3,
            n_random_paths=1024,
        )
        trials.to_csv(OUT / f"{scope}_rl_trials.csv", index=False)
        all_trials.append(trials)

    trial_table = pd.concat(all_trials, ignore_index=True)
    baseline_table = pd.DataFrame(baseline_rows)
    config_summary = summarize_trials(trial_table)
    best = best_by_scope(config_summary)

    trial_table.to_csv(METRICS / "integrated_rl_study_trials.csv", index=False)
    baseline_table.to_csv(METRICS / "integrated_rl_study_baselines.csv", index=False)
    config_summary.to_csv(METRICS / "integrated_rl_study_config_summary.csv", index=False)
    best.to_csv(METRICS / "integrated_rl_study_best_by_scope.csv", index=False)

    plot_scope_comparison(best)
    plot_hyperparameter_surface(config_summary)

    print("\nBest RL configuration by scope:")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
