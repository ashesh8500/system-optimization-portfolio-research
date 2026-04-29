# Hierarchical Reinforcement-Learning Routing Brief

**Author:** Ashesh Kaji  
**Run package:** `findings/2026-04-23-context-aware-grand-study/`  
**Status:** Completed first empirical hierarchical-routing experiment on the screened-core reward panel

## Research question

The motivating question is whether a learned decision-maker can sit above the portfolio construction stack and choose, at each walk-forward interval, which universe/screen/controller arm should be active. This reframes reinforcement learning from a direct weight allocator into a higher-level routing policy over the hierarchy already studied in the repository:

1. candidate-universe breadth,
2. screen rule,
3. weighting/controller rule,
4. transaction-cost-aware realized reward.

The practical analogy is a human portfolio manager who reviews a broad menu of candidate tickers and strategy families, then decides which layer of the system should be trusted for the next interval.

## Experimental setup

The experiment uses the completed screened-core matrix rather than synthetic data. The reward panel is constructed from:

- `metrics/screened_core_subperiod_summary.csv`
- main transaction-cost slice: 10 bps
- universes: `liquid_us_equity_100`, `liquid_us_equity_250`, `liquid_us_equity_500`
- arms: 20 screen/controller combinations per scope
- state features: normalized time, previous-action flag, universe indicator, trailing arm rewards, and trailing arm drawdown proxies
- action: select one screen/controller arm for the next period
- reward: next-period net return with a small switching penalty

Four scopes were evaluated:

1. combined routing across all three universe breadths,
2. routing within `liquid_us_equity_100`,
3. routing within `liquid_us_equity_250`,
4. routing within `liquid_us_equity_500`.

The policy is a learned softmax router trained through policy-gradient updates in a vectorized reinforcement-learning environment. The implementation is intentionally explicit and dependency-backed; missing RL dependencies fail directly rather than being replaced by placeholder behavior.

## Results

| Scope | Periods | Test periods | RL total reward | Best fixed arm | Best fixed reward | Random mean reward | Period oracle reward | RL vs best fixed | RL vs random mean |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Combined | 42 | 14 | 1.146 | Vol-adjusted momentum 63 + 21-day rank-and-hold | 1.635 | 1.293 | 2.586 | -0.488 | -0.147 |
| Liquid US Equity 100 | 22 | 8 | 0.827 | Cluster-capped momentum 63 + equal weight | 0.835 | 0.661 | 1.355 | -0.008 | +0.166 |
| Liquid US Equity 250 | 11 | 5 | 0.305 | Vol-adjusted momentum 63 + 21-day rank-and-hold | 0.466 | 0.290 | 0.764 | -0.161 | +0.016 |
| Liquid US Equity 500 | 9 | 4 | 0.371 | Cluster-capped momentum 63 + 126-day rank-and-hold | 0.881 | 0.512 | 0.932 | -0.511 | -0.142 |

The selected test-period policies were concentrated rather than erratic:

| Scope | Dominant selected arm(s) in test |
|---|---|
| Combined | 21-day momentum + 126-day rank-and-hold in 9 of 14 periods; 21-day momentum + 63-day rank-and-hold in 5 of 14 periods |
| Liquid US Equity 100 | 21-day momentum + 126-day rank-and-hold in all 8 periods |
| Liquid US Equity 250 | 63-day low-volatility + equal weight in all 5 periods |
| Liquid US Equity 500 | 21-day momentum + 63-day rank-and-hold in all 4 periods |

## Interpretation

The result is useful precisely because it is mixed rather than promotional. The learned router did not dominate the best hindsight-fixed arm. However, it showed a meaningful signal in the smaller and intermediate scopes by outperforming the random-routing baseline and choosing coherent arms rather than oscillating randomly.

The strongest academic interpretation is:

> A hierarchical learned router is feasible and can learn stable selection behavior from the screened-core panel, but the current walk-forward sample size is too small for a strong claim that reinforcement learning improves over the best fixed stack.

This strengthens the paper in three ways:

1. It converts reinforcement learning from an isolated appendix into a first-class control layer inside the same universe/screen/controller hierarchy.
2. It provides an honest negative/partial result: learned routing is plausible, but not yet dominant.
3. It identifies the precise empirical bottleneck: more independent routing episodes are needed before a policy can learn when to switch stacks.

## Practical conclusion

For practical use today, the fixed stack selected from the screened-core matrix remains the preferred deployable rule. The RL router is not yet a production replacement for the best fixed screened controller. Its value is as a research layer for studying conditional stack selection.

The most promising next extension is to expand the number of routing episodes by generating more independent universe definitions and rebalance windows, then evaluate the router under stricter out-of-sample splits. A credible follow-up should test whether the learned router can beat:

- the best fixed arm selected on training data,
- a trailing-window winner rule,
- a supervised classifier over universe descriptors,
- random arm routing,
- and the period oracle as an unattainable upper bound.

## Artifacts

Summary and traces:

- `metrics/hierarchical_rl_router_summary.csv`
- `metrics/hierarchical_rl_router/combined_chosen_actions.csv`
- `metrics/hierarchical_rl_router/combined_training_curve.csv`
- `metrics/hierarchical_rl_router/liquid_us_equity_100_chosen_actions.csv`
- `metrics/hierarchical_rl_router/liquid_us_equity_100_training_curve.csv`
- `metrics/hierarchical_rl_router/liquid_us_equity_250_chosen_actions.csv`
- `metrics/hierarchical_rl_router/liquid_us_equity_250_training_curve.csv`
- `metrics/hierarchical_rl_router/liquid_us_equity_500_chosen_actions.csv`
- `metrics/hierarchical_rl_router/liquid_us_equity_500_training_curve.csv`

Figures:

- `figures/hierarchical_rl_router_comparison.png`
- `figures/hierarchical_rl_router_training_curves.png`

Source and tests:

- `src/context_study/hierarchical_rl_router.py`
- `scripts/run_hierarchical_rl_router.py`
- `tests/test_hierarchical_rl_router.py`

## Repro

```bash
cd findings/2026-04-23-context-aware-grand-study
uv run python scripts/run_hierarchical_rl_router.py
uv run pytest tests/test_hierarchical_rl_router.py -q
```
