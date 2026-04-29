from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_study.hierarchical_rl_router import (  # noqa: E402
    HierarchicalRoutingEnv,
    build_router_reward_panel,
    evaluate_router_policy,
    evaluate_routing_baselines,
    run_router_repeated_study,
    train_pufferlib_router_policy,
)


def _toy_subperiod() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2021-01-01", periods=5, freq="90D")
    for split_id, start in enumerate(dates):
        for screen, controller, base in [
            ("momentum_21_top10", "EW", 0.02),
            ("low_volatility_63_top10", "EW", 0.01),
            ("momentum_21_top10", "MRH_tau21_k5", -0.005),
        ]:
            rows.append(
                {
                    "universe": "toy_universe",
                    "screen_rule": screen,
                    "controller": controller,
                    "split_id": split_id,
                    "cost_bps": 10.0,
                    "evaluation_start": start,
                    "evaluation_end": start + pd.Timedelta(days=63),
                    "ann_return": base + 0.01 * split_id,
                    "sharpe": 1.0 + base + split_id,
                    "turnover": 0.5 if controller != "EW" else 0.0,
                    "max_dd": -0.05,
                }
            )
    return pd.DataFrame(rows)


def test_build_router_reward_panel_creates_period_by_arm_matrix() -> None:
    panel = build_router_reward_panel(_toy_subperiod(), cost_bps=10.0)

    assert list(panel.periods["split_id"]) == [0, 1, 2, 3, 4]
    assert panel.rewards.shape == (5, 3)
    assert "momentum_21_top10|EW" in panel.arm_labels
    assert np.isfinite(panel.rewards).all()


def test_hierarchical_routing_env_is_native_pufferlib_env() -> None:
    panel = build_router_reward_panel(_toy_subperiod(), cost_bps=10.0)
    env = HierarchicalRoutingEnv(panel, transaction_cost_penalty=0.001)

    observations, infos = env.reset(seed=3)
    assert observations.shape == (1, env.single_observation_space.shape[0])
    assert len(infos) == 1
    assert env.single_action_space.n == len(panel.arm_labels)

    observations, rewards, terminals, truncations, infos = env.step(np.asarray([0], dtype=np.int32))
    assert observations.shape == (1, env.single_observation_space.shape[0])
    assert rewards.shape == (1,)
    assert terminals.shape == (1,)
    assert truncations.shape == (1,)
    assert infos[0]["chosen_arm"] == panel.arm_labels[0]


def test_train_and_evaluate_router_policy_beats_valid_random_baseline_on_toy_panel() -> None:
    panel = build_router_reward_panel(_toy_subperiod(), cost_bps=10.0)

    result = train_pufferlib_router_policy(
        panel,
        train_fraction=0.6,
        n_updates=30,
        episodes_per_update=4,
        learning_rate=0.05,
        seed=11,
    )
    evaluation = evaluate_router_policy(panel, result.policy_weights, train_fraction=0.6)

    assert result.used_pufferlib is True
    assert len(result.training_curve) == 30
    assert evaluation["n_test_periods"] == 2
    assert evaluation["rl_total_reward"] >= evaluation["random_mean_total_reward"]
    assert set(evaluation["chosen_actions"].columns) >= {"split_id", "chosen_arm", "reward"}


def test_evaluate_routing_baselines_reports_deployable_and_upper_bound_comparators() -> None:
    panel = build_router_reward_panel(_toy_subperiod(), cost_bps=10.0)

    baselines = evaluate_routing_baselines(
        panel,
        train_fraction=0.6,
        trailing_window=2,
        n_random_paths=64,
        seed=5,
    )

    assert baselines["train_selected_fixed_arm"] in panel.arm_labels
    assert baselines["train_selected_fixed_total_reward"] <= baselines["period_oracle_total_reward"]
    assert baselines["test_best_fixed_total_reward"] <= baselines["period_oracle_total_reward"]
    assert "trailing_window_total_reward" in baselines
    assert "random_p05_total_reward" in baselines
    assert baselines["baseline_actions"].shape[0] == baselines["n_test_periods"]


def test_run_router_repeated_study_returns_one_row_per_scope_config_seed() -> None:
    panel = build_router_reward_panel(_toy_subperiod(), cost_bps=10.0)

    rows = run_router_repeated_study(
        panel,
        scope="toy",
        configs=[{"config_name": "small", "n_updates": 5, "episodes_per_update": 2, "learning_rate": 0.03}],
        seeds=[1, 2],
        train_fraction=0.6,
        transaction_cost_penalty=0.0005,
    )

    assert len(rows) == 2
    assert set(rows.columns) >= {
        "scope",
        "config_name",
        "seed",
        "rl_total_reward",
        "train_selected_fixed_total_reward",
        "trailing_window_total_reward",
        "rl_vs_train_selected_fixed",
    }
    assert rows["used_pufferlib"].all()
