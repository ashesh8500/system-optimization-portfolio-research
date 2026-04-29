# Exhaustive Next Experiment: Universe--Screen--Weight Factorial Study

**Author:** Ashesh Kaji  
**Parent run:** `2026-04-23-context-aware-grand-study`  
**Purpose:** Convert the current pilot finding into a deeper, literature-aligned empirical study that directly tests the layered-control thesis.

## Preserved research intuition

The next experiment is organized around the ideas that motivated the project:

1. Portfolio allocation has multiple control tiers, not only a final weight vector.
2. Universe construction changes strategy performance by changing breadth, dispersion, correlation, liquidity, and feasible turnover.
3. A meta-layer should not be trained until the lower layers produce enough leakage-safe reward samples.
4. Meaningful insight comes from following one motivated mechanism deeply: whether performance comes from universe eligibility, screening/cardinality, weighting, or routing.

## Upgraded thesis

The paper should test the following claim:

> Portfolio performance is determined by an interaction between universe eligibility, screen/cardinality choice, weighting controller, and context-aware routing; therefore, the practical optimization problem is a layered control problem rather than a single static mean-variance allocation.

## Current empirical trigger

The current pilot already shows a motivating pattern:

| Candidate universe | Symbol limit | Best controller | Mean Sharpe |
|---|---:|---|---:|
| Liquid US Equity 100 | 25 | Equal-weight allocation | 1.539 |
| Liquid US Equity 250 | 40 | 21-day rank-and-hold allocation | 0.994 |
| Liquid US Equity 500 | 60 | 21-day rank-and-hold allocation | 1.634 |

This pattern is not yet the final claim because the universes are symbol-limited and current-constituent biased. It is, however, strong enough to motivate the next deeper experiment.

## Experiment design

The next empirical run should use a factorial design:

### Layer 0: Parent candidate universe

- liquid equity 100
- liquid equity 250
- liquid equity 500
- ETF basket robustness set, if time permits

### Layer 1: Universe construction rule

- fixed current constituent list, explicitly flagged as survivorship-biased exploratory evidence
- lagged dollar-volume top-N
- lagged dollar-volume top-N with membership buffer
- sector-balanced liquidity screen
- low-correlation / cluster-diversified universe

### Layer 2: Screen and cardinality

- no screen, all eligible names
- top-K 21-day momentum
- top-K 63-day momentum
- top-K volatility-adjusted momentum
- top-K liquidity-adjusted momentum
- cluster-capped momentum
- low-volatility screen

Holdings budgets:

- all eligible names
- K = 10
- K = 20
- K = 30
- K = 50

### Layer 3: Weighting controller

- equal weight
- inverse volatility
- shrinkage minimum variance
- hierarchical risk allocation
- convex turnover-penalized allocation

### Layer 4: Context-aware routing

This should come after the factorial reward panel exists. Compare:

1. best fixed controller by training window,
2. trailing-best controller,
3. current supervised descriptor classifier,
4. soft-gated mixture over controllers,
5. contextual bandit router such as LinUCB/LinTS or expert-advice EXP4-style routing.

## Main ablation questions

1. Does broadening the candidate universe increase the value of screening?
2. Does the 21-day ranking advantage come from the ranking screen itself or from its equal-weighting of selected names?
3. Do cluster-capped or sector-balanced screens preserve momentum gains while reducing drawdown/concentration?
4. Does sophisticated weighting add value after a good screen, or does it mainly add estimation error?
5. Are descriptor features sufficient for routing, or are controller-health and market-state features necessary?

## Required outputs

The next run should not be considered paper-ready unless it writes:

- `universe_provenance.csv`
- `model_selection_ledger.csv`
- `all_trials.csv`
- `gross_vs_net_summary.csv`
- `cost_sensitivity.csv`
- `turnover_summary.csv`
- `subperiod_summary.csv`
- `ablation_summary.csv`
- `multiple_testing_summary.csv`
- `deflated_sharpe_summary.csv` or a documented fallback
- `router_reward_panel.csv`

The current repository now includes starter schemas / maps:

- `metrics/modern_literature_map.csv`
- `metrics/universe_screen_weight_factorial_matrix.csv`
- `metrics/empirical_rigor_artifact_checklist.csv`

## Paper claims discipline

Use three levels of claims:

**Strong claim:** survives walk-forward testing, realistic costs, simple baselines, ablations, and multiple-testing diagnostics.

**Moderate claim:** improves gross performance but weakens after costs or in some regimes.

**Exploratory claim:** selected from many trials, current-constituent universe only, or weak meta-controller sample size.

The current meta-controller remains exploratory. The next run should first build the full-information date-level controller reward panel; only then should routing be evaluated.

## Immediate implementation sequence

1. Add universe provenance and model-selection ledger writing to the current context study runner.
2. Implement screen modules: momentum, volatility-adjusted momentum, liquidity-adjusted momentum, cluster-capped momentum, and low-volatility.
3. Run the core subset of `universe_screen_weight_factorial_matrix.csv` before the extended matrix.
4. Build `router_reward_panel.csv` from realized controller arms at each rebalance date.
5. Evaluate best-fixed, trailing-best, soft-gated logistic, and LinUCB/LinTS routers.

## Why this is deeper than another broad sweep

This experiment isolates mechanism. Instead of asking which named strategy wins, it asks which layer contributes:

- universe eligibility,
- screen/cardinality,
- weighting rule,
- context/routing.

That is the academic contribution that can turn the project from a collection of backtests into a coherent system-optimization study.
