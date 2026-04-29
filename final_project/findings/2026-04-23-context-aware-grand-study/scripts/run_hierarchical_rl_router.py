from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_study.hierarchical_rl_router import (  # noqa: E402
    build_router_reward_panel,
    evaluate_router_policy,
    train_pufferlib_router_policy,
)

METRICS = ROOT / "metrics"
FIGURES = ROOT / "figures"
OUT = METRICS / "hierarchical_rl_router"
OUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def run_scope(name: str, data: pd.DataFrame) -> dict[str, object]:
    panel = build_router_reward_panel(data, cost_bps=10.0)
    train_fraction = 0.68 if len(panel.periods) > 12 else 0.60
    result = train_pufferlib_router_policy(
        panel,
        train_fraction=train_fraction,
        n_updates=450,
        episodes_per_update=12,
        learning_rate=0.035,
        seed=31,
        transaction_cost_penalty=0.0005,
    )
    evaluation = evaluate_router_policy(
        panel,
        result.policy_weights,
        train_fraction=train_fraction,
        transaction_cost_penalty=0.0005,
        seed=37,
        n_random_paths=512,
    )

    pd.DataFrame(
        {
            "update": range(len(result.training_curve)),
            "mean_episode_reward": result.training_curve,
        }
    ).to_csv(OUT / f"{name}_training_curve.csv", index=False)
    evaluation["chosen_actions"].to_csv(OUT / f"{name}_chosen_actions.csv", index=False)

    return {
        "scope": name,
        "periods": len(panel.periods),
        "arms": len(panel.arm_labels),
        "universes": ";".join(panel.universe_labels),
        "train_fraction": train_fraction,
        "used_pufferlib": result.used_pufferlib,
        "n_updates": len(result.training_curve),
        "final_train_reward_mean_25": float(pd.Series(result.training_curve).tail(25).mean()),
        "n_test_periods": evaluation["n_test_periods"],
        "rl_total_reward": evaluation["rl_total_reward"],
        "best_fixed_arm": evaluation["best_fixed_arm"],
        "best_fixed_total_reward": evaluation["best_fixed_total_reward"],
        "period_oracle_total_reward": evaluation["period_oracle_total_reward"],
        "random_mean_total_reward": evaluation["random_mean_total_reward"],
        "random_p95_total_reward": evaluation["random_p95_total_reward"],
        "rl_vs_best_fixed": evaluation["rl_total_reward"] - evaluation["best_fixed_total_reward"],
        "rl_vs_random_mean": evaluation["rl_total_reward"] - evaluation["random_mean_total_reward"],
        "oracle_gap": evaluation["period_oracle_total_reward"] - evaluation["rl_total_reward"],
    }


def main() -> None:
    subperiod_path = METRICS / "screened_core_subperiod_summary.csv"
    data = pd.read_csv(subperiod_path)
    scopes: list[tuple[str, pd.DataFrame]] = [("combined", data)]
    for universe in sorted(data["universe"].unique()):
        scopes.append((universe, data.loc[data["universe"] == universe].copy()))

    rows = []
    for name, scope_data in scopes:
        print(f"[hierarchical-rl] {name}: rows={len(scope_data)}", flush=True)
        rows.append(run_scope(name, scope_data))

    summary = pd.DataFrame(rows)
    summary.to_csv(METRICS / "hierarchical_rl_router_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(summary))
    ax.bar([i - 0.25 for i in x], summary["rl_total_reward"], width=0.25, label="RL router")
    ax.bar(x, summary["best_fixed_total_reward"], width=0.25, label="Best fixed arm")
    ax.bar([i + 0.25 for i in x], summary["random_mean_total_reward"], width=0.25, label="Random mean")
    ax.set_xticks(list(x), summary["scope"], rotation=25, ha="right")
    ax.set_ylabel("Test-period total reward")
    ax.set_title("Hierarchical RL router versus fixed and random routing baselines")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "hierarchical_rl_router_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for path in sorted(OUT.glob("*_training_curve.csv")):
        curve = pd.read_csv(path)
        label = path.name.replace("_training_curve.csv", "")
        ax.plot(curve["update"], curve["mean_episode_reward"].rolling(15, min_periods=1).mean(), label=label)
    ax.set_xlabel("Policy-gradient update")
    ax.set_ylabel("Rolling mean train episode reward")
    ax.set_title("PufferLib hierarchical router training curves")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "hierarchical_rl_router_training_curves.png", dpi=180)
    plt.close(fig)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
