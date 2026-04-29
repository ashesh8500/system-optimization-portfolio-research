from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import setuptools  # noqa: F401  # required so legacy Gym exposes distutils on Python 3.12
import pufferlib
import pufferlib.spaces


@dataclass(frozen=True)
class RouterRewardPanel:
    """Period-by-arm reward matrix for hierarchical portfolio routing."""

    periods: pd.DataFrame
    rewards: np.ndarray
    arm_labels: list[str]
    universe_labels: list[str]

    def subset(self, start: int, stop: int) -> "RouterRewardPanel":
        periods = self.periods.iloc[start:stop].reset_index(drop=True)
        rewards = self.rewards[start:stop].copy()
        return RouterRewardPanel(
            periods=periods,
            rewards=rewards,
            arm_labels=list(self.arm_labels),
            universe_labels=list(self.universe_labels),
        )


@dataclass(frozen=True)
class RouterTrainingResult:
    policy_weights: np.ndarray
    training_curve: list[float]
    arm_labels: list[str]
    feature_dim: int
    used_pufferlib: bool


def _period_return_from_annualized(row: pd.Series) -> float:
    start = pd.Timestamp(row["evaluation_start"])
    end = pd.Timestamp(row["evaluation_end"])
    days = max((end - start).days, 1)
    scale = min(days / 252.0, 1.0)
    ann_return = float(row["ann_return"])
    return ann_return * scale


def build_router_reward_panel(
    subperiod_summary: pd.DataFrame,
    *,
    cost_bps: float = 10.0,
    reward_column: str = "period_return",
) -> RouterRewardPanel:
    """Convert screen-controller walk-forward rows into a routing reward panel.

    Each action corresponds to a hierarchical arm of the form
    ``screen_rule|controller``.  Each period corresponds to one universe and
    split.  The reward is the next evaluation-window return implied by the
    already-computed screen/controller backtest row.
    """
    required = {
        "universe",
        "screen_rule",
        "controller",
        "split_id",
        "cost_bps",
        "evaluation_start",
        "evaluation_end",
        "ann_return",
    }
    missing = required - set(subperiod_summary.columns)
    if missing:
        raise ValueError(f"subperiod_summary missing required columns: {sorted(missing)}")
    if subperiod_summary.empty:
        raise ValueError("subperiod_summary must not be empty")

    df = subperiod_summary.loc[subperiod_summary["cost_bps"].astype(float) == float(cost_bps)].copy()
    if df.empty:
        raise ValueError(f"no rows found for cost_bps={cost_bps}")
    df["evaluation_start"] = pd.to_datetime(df["evaluation_start"])
    df["evaluation_end"] = pd.to_datetime(df["evaluation_end"])
    df["arm_label"] = df["screen_rule"].astype(str) + "|" + df["controller"].astype(str)
    if reward_column == "period_return":
        df["period_return"] = df.apply(_period_return_from_annualized, axis=1)
    elif reward_column not in df.columns:
        raise ValueError(f"reward_column={reward_column!r} is not present")

    period_cols = ["universe", "split_id", "evaluation_start", "evaluation_end"]
    periods = (
        df.loc[:, period_cols]
        .drop_duplicates()
        .sort_values(["evaluation_start", "universe", "split_id"])
        .reset_index(drop=True)
    )
    arm_labels = sorted(df["arm_label"].unique().tolist())
    universe_labels = sorted(periods["universe"].astype(str).unique().tolist())

    pivot = df.pivot_table(
        index=period_cols,
        columns="arm_label",
        values=reward_column,
        aggfunc="mean",
    ).reindex(columns=arm_labels)
    pivot = pivot.reindex(pd.MultiIndex.from_frame(periods.loc[:, period_cols]))
    rewards = pivot.to_numpy(dtype=float)
    if np.isnan(rewards).any():
        col_means = np.nanmean(rewards, axis=0)
        global_min = np.nanmin(rewards)
        col_means = np.where(np.isfinite(col_means), col_means, global_min)
        row_idx, col_idx = np.where(np.isnan(rewards))
        rewards[row_idx, col_idx] = col_means[col_idx]
    if not np.isfinite(rewards).all():
        raise ValueError("reward panel contains non-finite rewards")

    return RouterRewardPanel(
        periods=periods,
        rewards=rewards.astype(np.float32),
        arm_labels=arm_labels,
        universe_labels=universe_labels,
    )


class HierarchicalRoutingEnv(pufferlib.PufferEnv):
    """Single-agent PufferLib environment for hierarchical strategy routing.

    At each walk-forward interval, the policy selects one arm combining a screen
    and controller.  The reward is the realized next-period return for that arm,
    penalized when the route switches between arms.
    """

    def __init__(
        self,
        reward_panel: RouterRewardPanel,
        *,
        transaction_cost_penalty: float = 0.0005,
        buf=None,
    ) -> None:
        if reward_panel.rewards.ndim != 2 or reward_panel.rewards.shape[0] == 0:
            raise ValueError("reward_panel must contain a non-empty 2D reward matrix")
        self.panel = reward_panel
        self.transaction_cost_penalty = float(transaction_cost_penalty)
        self.num_agents = 1
        self.n_periods, self.n_arms = reward_panel.rewards.shape
        self.n_universes = len(reward_panel.universe_labels)
        self.obs_dim = 1 + 1 + self.n_universes + self.n_arms * 2
        box_cls = pufferlib.spaces.Box[-1]
        discrete_cls = pufferlib.spaces.Discrete[-1]
        self.single_observation_space = box_cls(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.single_action_space = discrete_cls(self.n_arms)
        self.t = 0
        self.previous_action = -1
        self.trailing_sum = np.zeros(self.n_arms, dtype=np.float32)
        self.trailing_count = np.zeros(self.n_arms, dtype=np.float32)
        self.last_observed_rewards = np.zeros(self.n_arms, dtype=np.float32)
        super().__init__(buf=buf)

    def _observation(self) -> np.ndarray:
        period = self.panel.periods.iloc[min(self.t, self.n_periods - 1)]
        universe = str(period["universe"])
        universe_one_hot = np.asarray(
            [1.0 if universe == label else 0.0 for label in self.panel.universe_labels],
            dtype=np.float32,
        )
        progress = np.asarray([self.t / max(self.n_periods - 1, 1)], dtype=np.float32)
        bias = np.asarray([1.0], dtype=np.float32)
        trailing_mean = np.divide(
            self.trailing_sum,
            np.maximum(self.trailing_count, 1.0),
            out=np.zeros_like(self.trailing_sum),
            where=np.maximum(self.trailing_count, 1.0) > 0,
        ).astype(np.float32)
        obs = np.concatenate([bias, progress, universe_one_hot, trailing_mean, self.last_observed_rewards])
        return obs.astype(np.float32)

    def reset(self, seed: int | None = None):
        if seed is not None:
            np.random.default_rng(seed)
        self.t = 0
        self.previous_action = -1
        self.trailing_sum[:] = 0.0
        self.trailing_count[:] = 0.0
        self.last_observed_rewards[:] = 0.0
        self.observations[0] = self._observation()
        self.rewards[0] = 0.0
        self.terminals[0] = False
        self.truncations[0] = False
        return self.observations, [{"period": 0}]

    def step(self, actions):
        action = int(np.asarray(actions).reshape(-1)[0])
        if action < 0 or action >= self.n_arms:
            raise ValueError(f"action {action} outside [0, {self.n_arms})")

        raw_reward = float(self.panel.rewards[self.t, action])
        switch_penalty = self.transaction_cost_penalty if self.previous_action not in {-1, action} else 0.0
        reward = raw_reward - switch_penalty

        full_rewards = self.panel.rewards[self.t].astype(np.float32)
        self.trailing_sum += full_rewards
        self.trailing_count += 1.0
        self.last_observed_rewards = full_rewards
        self.previous_action = action
        chosen_period = self.panel.periods.iloc[self.t]
        self.t += 1
        terminal = self.t >= self.n_periods

        self.rewards[0] = reward
        self.terminals[0] = terminal
        self.truncations[0] = False
        if terminal:
            self.observations[0] = np.zeros(self.obs_dim, dtype=np.float32)
        else:
            self.observations[0] = self._observation()
        info = {
            "period": int(chosen_period["split_id"]),
            "universe": str(chosen_period["universe"]),
            "chosen_arm": self.panel.arm_labels[action],
            "raw_reward": raw_reward,
            "switch_penalty": switch_penalty,
        }
        return self.observations, self.rewards, self.terminals, self.truncations, [info]

    def close(self) -> None:
        return None


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    exp = np.exp(z)
    return exp / (np.sum(exp) + 1e-12)


def _split_panel(panel: RouterRewardPanel, train_fraction: float) -> tuple[RouterRewardPanel, RouterRewardPanel, int]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    split = int(np.floor(len(panel.periods) * train_fraction))
    split = min(max(split, 1), len(panel.periods) - 1)
    return panel.subset(0, split), panel.subset(split, len(panel.periods)), split


def train_pufferlib_router_policy(
    panel: RouterRewardPanel,
    *,
    train_fraction: float = 0.7,
    n_updates: int = 200,
    episodes_per_update: int = 8,
    learning_rate: float = 0.02,
    gamma: float = 0.98,
    seed: int = 17,
    transaction_cost_penalty: float = 0.0005,
) -> RouterTrainingResult:
    """Train a PufferLib-native hierarchical router with REINFORCE."""
    train_panel, _, _ = _split_panel(panel, train_fraction)
    env = HierarchicalRoutingEnv(train_panel, transaction_cost_penalty=transaction_cost_penalty)
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=(env.obs_dim, env.n_arms)).astype(np.float64)
    training_curve: list[float] = []
    baseline = 0.0

    for update in range(n_updates):
        update_returns: list[float] = []
        grad = np.zeros_like(weights)
        for episode in range(episodes_per_update):
            obs_buf, _ = env.reset(seed=seed + update * episodes_per_update + episode)
            observations: list[np.ndarray] = []
            actions: list[int] = []
            rewards: list[float] = []
            done = False
            while not done:
                obs = obs_buf[0].astype(np.float64)
                probabilities = _softmax(obs @ weights)
                action = int(rng.choice(env.n_arms, p=probabilities))
                observations.append(obs.copy())
                actions.append(action)
                obs_buf, reward_buf, terminal_buf, _, _ = env.step(np.asarray([action], dtype=np.int32))
                rewards.append(float(reward_buf[0]))
                done = bool(terminal_buf[0])

            returns = []
            running = 0.0
            for reward in reversed(rewards):
                running = reward + gamma * running
                returns.insert(0, running)
            returns_arr = np.asarray(returns, dtype=np.float64)
            update_returns.append(float(np.sum(rewards)))
            baseline = 0.95 * baseline + 0.05 * float(returns_arr.mean())

            for obs, action, ret in zip(observations, actions, returns_arr):
                probabilities = _softmax(obs @ weights)
                indicator = np.zeros(env.n_arms, dtype=np.float64)
                indicator[action] = 1.0
                grad += (ret - baseline) * np.outer(obs, indicator - probabilities)

        grad_norm = float(np.linalg.norm(grad))
        if np.isfinite(grad_norm) and grad_norm > 1.0:
            grad /= grad_norm
        weights += learning_rate * grad / max(1, episodes_per_update)
        training_curve.append(float(np.mean(update_returns)))

    return RouterTrainingResult(
        policy_weights=weights.astype(np.float32),
        training_curve=training_curve,
        arm_labels=list(panel.arm_labels),
        feature_dim=env.obs_dim,
        used_pufferlib=isinstance(env, pufferlib.PufferEnv),
    )


def evaluate_router_policy(
    panel: RouterRewardPanel,
    policy_weights: np.ndarray,
    *,
    train_fraction: float = 0.7,
    transaction_cost_penalty: float = 0.0005,
    seed: int = 23,
    n_random_paths: int = 256,
) -> dict[str, object]:
    """Evaluate a deterministic hierarchical router against routing baselines."""
    _, test_panel, split = _split_panel(panel, train_fraction)
    env = HierarchicalRoutingEnv(test_panel, transaction_cost_penalty=transaction_cost_penalty)
    if policy_weights.shape != (env.obs_dim, env.n_arms):
        raise ValueError(f"policy_weights shape {policy_weights.shape} does not match {(env.obs_dim, env.n_arms)}")

    obs_buf, _ = env.reset(seed=seed)
    done = False
    rows: list[dict[str, object]] = []
    total = 0.0
    while not done:
        obs = obs_buf[0].astype(float)
        action = int(np.argmax(obs @ policy_weights))
        obs_buf, reward_buf, terminal_buf, _, infos = env.step(np.asarray([action], dtype=np.int32))
        reward = float(reward_buf[0])
        total += reward
        period = test_panel.periods.iloc[len(rows)]
        rows.append(
            {
                "split_id": int(period["split_id"]),
                "universe": str(period["universe"]),
                "evaluation_start": period["evaluation_start"],
                "evaluation_end": period["evaluation_end"],
                "chosen_arm": infos[0]["chosen_arm"],
                "reward": reward,
                "raw_reward": float(infos[0]["raw_reward"]),
                "switch_penalty": float(infos[0]["switch_penalty"]),
            }
        )
        done = bool(terminal_buf[0])

    fixed_totals = test_panel.rewards.sum(axis=0)
    best_fixed_idx = int(np.argmax(fixed_totals))
    best_period_oracle = float(np.max(test_panel.rewards, axis=1).sum())
    rng = np.random.default_rng(seed)
    random_totals = []
    for _ in range(n_random_paths):
        actions = rng.integers(0, test_panel.rewards.shape[1], size=test_panel.rewards.shape[0])
        random_totals.append(float(test_panel.rewards[np.arange(len(actions)), actions].sum()))

    chosen_actions = pd.DataFrame(rows)
    return {
        "split_index": split,
        "n_test_periods": int(test_panel.rewards.shape[0]),
        "rl_total_reward": float(total),
        "best_fixed_arm": test_panel.arm_labels[best_fixed_idx],
        "best_fixed_total_reward": float(fixed_totals[best_fixed_idx]),
        "period_oracle_total_reward": best_period_oracle,
        "random_mean_total_reward": float(np.mean(random_totals)),
        "random_p95_total_reward": float(np.percentile(random_totals, 95)),
        "chosen_actions": chosen_actions,
    }



def evaluate_routing_baselines(
    panel: RouterRewardPanel,
    *,
    train_fraction: float = 0.7,
    transaction_cost_penalty: float = 0.0005,
    trailing_window: int = 3,
    seed: int = 23,
    n_random_paths: int = 256,
) -> dict[str, object]:
    """Evaluate deployable and upper-bound routing baselines.

    The train-selected fixed arm is the main deployable comparator: it chooses
    the arm with the largest total reward on the training slice and applies it
    unchanged to the test slice. The test-best fixed arm and period oracle are
    reported only as upper-bound diagnostics.
    """
    train_panel, test_panel, split = _split_panel(panel, train_fraction)
    if trailing_window <= 0:
        raise ValueError("trailing_window must be positive")

    train_totals = train_panel.rewards.sum(axis=0)
    train_selected_idx = int(np.argmax(train_totals))
    train_selected_reward = float(test_panel.rewards[:, train_selected_idx].sum())

    test_totals = test_panel.rewards.sum(axis=0)
    test_best_idx = int(np.argmax(test_totals))
    period_oracle = float(np.max(test_panel.rewards, axis=1).sum())

    history = train_panel.rewards.copy()
    trailing_rows: list[dict[str, object]] = []
    trailing_total = 0.0
    previous_action = -1
    for local_t in range(test_panel.rewards.shape[0]):
        lookback = history[-trailing_window:]
        trailing_scores = lookback.mean(axis=0)
        action = int(np.argmax(trailing_scores))
        raw_reward = float(test_panel.rewards[local_t, action])
        switch_penalty = transaction_cost_penalty if previous_action not in {-1, action} else 0.0
        reward = raw_reward - switch_penalty
        trailing_total += reward
        period = test_panel.periods.iloc[local_t]
        trailing_rows.append(
            {
                "split_id": int(period["split_id"]),
                "universe": str(period["universe"]),
                "evaluation_start": period["evaluation_start"],
                "evaluation_end": period["evaluation_end"],
                "trailing_window_arm": test_panel.arm_labels[action],
                "trailing_window_reward": reward,
                "trailing_window_raw_reward": raw_reward,
                "trailing_window_switch_penalty": switch_penalty,
            }
        )
        previous_action = action
        history = np.vstack([history, test_panel.rewards[local_t : local_t + 1]])

    rng = np.random.default_rng(seed)
    random_totals = []
    for _ in range(n_random_paths):
        actions = rng.integers(0, test_panel.rewards.shape[1], size=test_panel.rewards.shape[0])
        random_totals.append(float(test_panel.rewards[np.arange(len(actions)), actions].sum()))

    return {
        "split_index": split,
        "n_test_periods": int(test_panel.rewards.shape[0]),
        "train_selected_fixed_arm": test_panel.arm_labels[train_selected_idx],
        "train_selected_fixed_total_reward": train_selected_reward,
        "test_best_fixed_arm": test_panel.arm_labels[test_best_idx],
        "test_best_fixed_total_reward": float(test_totals[test_best_idx]),
        "period_oracle_total_reward": period_oracle,
        "trailing_window": int(trailing_window),
        "trailing_window_total_reward": float(trailing_total),
        "random_mean_total_reward": float(np.mean(random_totals)),
        "random_p05_total_reward": float(np.percentile(random_totals, 5)),
        "random_p50_total_reward": float(np.percentile(random_totals, 50)),
        "random_p95_total_reward": float(np.percentile(random_totals, 95)),
        "baseline_actions": pd.DataFrame(trailing_rows),
    }


def run_router_repeated_study(
    panel: RouterRewardPanel,
    *,
    scope: str,
    configs: list[dict[str, object]],
    seeds: list[int],
    train_fraction: float,
    transaction_cost_penalty: float = 0.0005,
    trailing_window: int = 3,
    n_random_paths: int = 256,
) -> pd.DataFrame:
    """Run a repeated-seed RL routing study and return one row per trial."""
    if not configs:
        raise ValueError("configs must not be empty")
    if not seeds:
        raise ValueError("seeds must not be empty")

    baseline = evaluate_routing_baselines(
        panel,
        train_fraction=train_fraction,
        transaction_cost_penalty=transaction_cost_penalty,
        trailing_window=trailing_window,
        n_random_paths=n_random_paths,
        seed=seeds[0],
    )
    rows: list[dict[str, object]] = []
    for config in configs:
        config_name = str(config.get("config_name", "unnamed"))
        for seed in seeds:
            result = train_pufferlib_router_policy(
                panel,
                train_fraction=train_fraction,
                n_updates=int(config.get("n_updates", 200)),
                episodes_per_update=int(config.get("episodes_per_update", 8)),
                learning_rate=float(config.get("learning_rate", 0.02)),
                gamma=float(config.get("gamma", 0.98)),
                seed=int(seed),
                transaction_cost_penalty=transaction_cost_penalty,
            )
            evaluation = evaluate_router_policy(
                panel,
                result.policy_weights,
                train_fraction=train_fraction,
                transaction_cost_penalty=transaction_cost_penalty,
                seed=int(seed) + 10_000,
                n_random_paths=n_random_paths,
            )
            rl_reward = float(evaluation["rl_total_reward"])
            rows.append(
                {
                    "scope": scope,
                    "config_name": config_name,
                    "seed": int(seed),
                    "periods": len(panel.periods),
                    "arms": len(panel.arm_labels),
                    "universes": ";".join(panel.universe_labels),
                    "train_fraction": float(train_fraction),
                    "n_updates": int(config.get("n_updates", 200)),
                    "episodes_per_update": int(config.get("episodes_per_update", 8)),
                    "learning_rate": float(config.get("learning_rate", 0.02)),
                    "gamma": float(config.get("gamma", 0.98)),
                    "used_pufferlib": bool(result.used_pufferlib),
                    "final_train_reward_mean_25": float(pd.Series(result.training_curve).tail(25).mean()),
                    "n_test_periods": int(evaluation["n_test_periods"]),
                    "rl_total_reward": rl_reward,
                    "train_selected_fixed_arm": baseline["train_selected_fixed_arm"],
                    "train_selected_fixed_total_reward": baseline["train_selected_fixed_total_reward"],
                    "trailing_window_total_reward": baseline["trailing_window_total_reward"],
                    "test_best_fixed_arm": baseline["test_best_fixed_arm"],
                    "test_best_fixed_total_reward": baseline["test_best_fixed_total_reward"],
                    "period_oracle_total_reward": baseline["period_oracle_total_reward"],
                    "random_mean_total_reward": baseline["random_mean_total_reward"],
                    "random_p05_total_reward": baseline["random_p05_total_reward"],
                    "random_p95_total_reward": baseline["random_p95_total_reward"],
                    "rl_vs_train_selected_fixed": rl_reward - float(baseline["train_selected_fixed_total_reward"]),
                    "rl_vs_trailing_window": rl_reward - float(baseline["trailing_window_total_reward"]),
                    "rl_vs_test_best_fixed": rl_reward - float(baseline["test_best_fixed_total_reward"]),
                    "rl_vs_random_mean": rl_reward - float(baseline["random_mean_total_reward"]),
                    "oracle_gap": float(baseline["period_oracle_total_reward"]) - rl_reward,
                    "dominant_chosen_arm": evaluation["chosen_actions"]["chosen_arm"].mode().iat[0],
                    "chosen_arm_count": int(evaluation["chosen_actions"]["chosen_arm"].value_counts().max()),
                }
            )
    return pd.DataFrame(rows)
