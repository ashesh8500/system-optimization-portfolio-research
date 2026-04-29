# Integrated Reinforcement-Learning Routing Study

**Author:** Ashesh Kaji  
**Run package:** `findings/2026-04-23-context-aware-grand-study/`  
**Status:** Completed as part of the main universe × screen × controller experiment

## Objective

This experiment evaluates reinforcement learning as an internal component of the portfolio-control hierarchy, not as an external extension. The learned policy is asked to play the role of a decision-making agent that chooses among screen/controller stacks at each walk-forward interval.

The central question is:

> Can a learned routing policy exploit the hierarchical structure of the allocation system well enough to improve over simple deployable stack-selection rules?

## Setup

The reward panel comes from the completed screened-core matrix:

- source: `metrics/screened_core_subperiod_summary.csv`
- primary transaction-cost slice: 10 bps
- candidate-universe scopes: combined, 100-name, 250-name, and 500-name liquid-equity pilots
- actions: 20 screen/controller arms per scope
- policy input: time, universe indicator, trailing arm rewards, and most recent full arm reward vector
- action: select one screen/controller stack
- reward: next-period net return minus a small switching penalty

The reinforcement-learning study was run as a repeated-seed ML experiment:

- 3 policy-gradient configurations: conservative, baseline, aggressive
- 7 random seeds per configuration
- 4 scopes
- 84 total RL trials
- explicit comparison to deployable baselines and upper bounds

## Baselines

The study distinguishes deployable baselines from diagnostic upper bounds:

1. **Train-selected fixed arm:** select the best arm on the training slice and hold it through the test slice. This is the primary practical comparator.
2. **Trailing-window router:** choose the arm with the best trailing realized reward over the recent window. This is a simple adaptive non-RL comparator.
3. **Random routing:** sample arms uniformly to estimate a naive policy distribution.
4. **Test-best fixed arm:** best fixed arm on the test slice; reported as hindsight diagnostic, not deployable.
5. **Period oracle:** best arm in each test period; unattainable upper bound.

## Best RL configuration by scope

| Scope | Best RL config | Test periods | RL mean | RL std | Train-selected fixed | Trailing-window | Random mean | Test-best fixed | Oracle | RL vs train-selected | RL vs trailing |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Combined | Aggressive | 14 | 1.213 | 0.063 | 1.134 | 1.543 | 1.276 | 1.635 | 2.586 | +0.079 | -0.330 |
| Liquid US Equity 100 | Aggressive | 8 | 0.823 | 0.011 | 0.827 | 0.475 | 0.657 | 0.835 | 1.355 | -0.004 | +0.348 |
| Liquid US Equity 250 | Aggressive | 5 | 0.280 | 0.068 | 0.305 | 0.306 | 0.296 | 0.466 | 0.764 | -0.026 | -0.026 |
| Liquid US Equity 500 | Aggressive | 4 | 0.371 | 0.000 | 0.371 | 0.566 | 0.525 | 0.881 | 0.932 | +0.000 | -0.196 |

## Stability diagnostics

| Scope | Seeds | Wins vs train-selected fixed | Wins vs trailing-window | Wins vs random mean | Dominant learned selections |
|---|---:|---:|---:|---:|---|
| Combined | 7 | 6 / 7 | 0 / 7 | 1 / 7 | Mostly cluster-capped momentum + equal weight; also 21-day momentum + 126-day rank-and-hold |
| Liquid US Equity 100 | 7 | 6 / 7 | 7 / 7 | 7 / 7 | 21-day momentum + 126-day rank-and-hold in every best-config trial |
| Liquid US Equity 250 | 7 | 0 / 7 | 0 / 7 | 6 / 7 | 63-day low-volatility + equal weight in every best-config trial |
| Liquid US Equity 500 | 7 | 7 / 7 by numerical tie | 0 / 7 | 0 / 7 | 21-day momentum + 63-day rank-and-hold in every best-config trial |

## Interpretation

The integrated RL result is more informative than the initial single run. It shows that a learned routing policy can discover stable, economically interpretable selections, but it does not yet dominate simple adaptive routing.

The most important positive result is in the combined scope: the best RL configuration improves on the train-selected fixed arm by approximately 0.079 total test-period reward on average and wins against it in 6 of 7 seeds. This means the learned router is not merely decorative; it can exploit cross-scope structure better than a static arm selected from the training slice.

The most important caution is that the trailing-window router is still stronger in the combined and 500-name scopes. This means the current RL state and policy class are not yet extracting all available sequential information. The system contains exploitable hierarchy, but the simple adaptive rule remains a serious baseline.

The universe-specific results are also meaningful:

- In the 100-name scope, RL nearly matches the train-selected fixed arm and decisively beats random and trailing-window routing. The learned policy converges to a stable momentum/rank-and-hold stack.
- In the 250-name scope, RL underperforms train-selected and trailing-window selection, but its dominant action matches the core study's risk-filtering theme: low-volatility screening plus equal weighting.
- In the 500-name scope, RL collapses to the same deployable train-selected fixed arm, while trailing-window routing finds a better adaptive path. This indicates that broader universes create exploitable switching opportunities that the current learned router has not fully captured.

## Research conclusion

Reinforcement learning should be included in the main experiment as a meta-control layer, not as an appendix or extension. The result is not that RL is the best controller; the result is sharper:

> The hierarchical system is learnable enough for a policy-gradient router to recover stable stack preferences and sometimes improve over train-selected fixed selection, but a simple trailing-window adaptive rule remains a strong baseline. Therefore, the value of RL is currently explanatory and architectural rather than conclusively performance-dominant.

This is a useful academic result because it gives the paper a disciplined ML study: repeated seeds, hyperparameter sensitivity, deployable baselines, random baselines, hindsight upper bounds, and honest negative comparisons.

## Artifacts

Primary tables:

- `metrics/integrated_rl_study_trials.csv`
- `metrics/integrated_rl_study_baselines.csv`
- `metrics/integrated_rl_study_config_summary.csv`
- `metrics/integrated_rl_study_best_by_scope.csv`
- `metrics/integrated_rl_study_win_rates.csv`

Per-scope trial and baseline-action traces:

- `metrics/integrated_rl_study/combined_rl_trials.csv`
- `metrics/integrated_rl_study/combined_baseline_actions.csv`
- `metrics/integrated_rl_study/liquid_us_equity_100_rl_trials.csv`
- `metrics/integrated_rl_study/liquid_us_equity_100_baseline_actions.csv`
- `metrics/integrated_rl_study/liquid_us_equity_250_rl_trials.csv`
- `metrics/integrated_rl_study/liquid_us_equity_250_baseline_actions.csv`
- `metrics/integrated_rl_study/liquid_us_equity_500_rl_trials.csv`
- `metrics/integrated_rl_study/liquid_us_equity_500_baseline_actions.csv`

Figures:

- `figures/integrated_rl_study_scope_comparison.png`
- `figures/integrated_rl_study_config_sensitivity.png`

Source and tests:

- `src/context_study/hierarchical_rl_router.py`
- `scripts/run_integrated_rl_study.py`
- `tests/test_hierarchical_rl_router.py`

## Repro

```bash
cd findings/2026-04-23-context-aware-grand-study
uv run python scripts/run_integrated_rl_study.py
uv run pytest tests/test_hierarchical_rl_router.py -q
```
