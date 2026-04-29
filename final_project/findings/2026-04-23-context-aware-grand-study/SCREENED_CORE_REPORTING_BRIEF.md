# Screened Core Matrix Reporting Brief

**Author:** Ashesh Kaji  
**Run:** `2026-04-23-context-aware-grand-study`  
**Status:** Completed breadth extension for the screen-integrated core matrix

## What was completed

The screen-integrated walk-forward experiment has now been extended from the first small pilot to the three candidate-universe breadth levels used by the context-aware study:

| Parent universe | Symbol limit | Screens | Controllers | Transaction costs | Splits |
|---|---:|---:|---:|---:|---:|
| Liquid US Equity 100 | 25 | 5 | 4 | 0, 10, 25 bps | 22 |
| Liquid US Equity 250 | 40 | 5 | 4 | 0, 10, 25 bps | 11 |
| Liquid US Equity 500 | 60 | 5 | 4 | 0, 10, 25 bps | 9 |

The matrix evaluates 60 universe--screen--controller arms at the canonical 10 bps cost slice, with split-level and cost-grid artifacts retained for robustness checks.

## Screens and controllers

Screens:

1. 21-day momentum top 10,
2. 63-day momentum top 10,
3. 63-day volatility-adjusted momentum top 10,
4. 63-day low-volatility top 10,
5. 63-day cluster-capped momentum top 10.

Controllers:

1. equal-weight allocation over the screened holdings,
2. 21-day rank-and-hold allocation,
3. 63-day rank-and-hold allocation,
4. 126-day rank-and-hold allocation.

## Primary empirical table

| Universe | Best screen | Best controller | Mean Sharpe | Mean ann. return | Worst split drawdown |
|---|---|---|---:|---:|---:|
| Liquid US Equity 100 | 21-day momentum top 10 | Equal weight | 1.474 | 0.204 | -0.449 |
| Liquid US Equity 250 | 63-day low-volatility top 10 | Equal weight | 0.859 | 0.112 | -0.105 |
| Liquid US Equity 500 | 63-day volatility-adjusted momentum top 10 | 21-day rank-and-hold | 1.756 | 0.351 | -0.155 |

## Paper-ready interpretation

The completed matrix supports a sharper version of the layered-control thesis. The best action is not a single global weighting rule. Instead, the preferred control stack changes with candidate-universe breadth:

- In the narrower 100-name pilot, a simple 21-day momentum screen followed by equal weighting gives the best Sharpe. This suggests that once a compact opportunity set is screened, additional ranking inside the screened set may add turnover without enough marginal signal.
- In the intermediate 250-name pilot, low-volatility screening dominates. The practical implication is risk control: when the broader opportunity set is not producing enough clean cross-sectional momentum dispersion, removing high-volatility names can improve the realized risk-adjusted profile.
- In the broader 500-name pilot, volatility-adjusted momentum plus a short-horizon rank-and-hold controller performs best. This is the most direct evidence for the universe-breadth thesis: broader candidate sets create more room for cross-sectional selection, but raw momentum benefits from volatility normalization.

## Mechanism-level findings

### 1. Screening is not a cosmetic layer

Averaging over controllers, the strongest screen changes by universe:

| Universe | Best screen averaged over controllers | Mean Sharpe across controllers |
|---|---|---:|
| Liquid US Equity 100 | 63-day low-volatility top 10 | 1.139 |
| Liquid US Equity 250 | 63-day low-volatility top 10 | 0.699 |
| Liquid US Equity 500 | 63-day momentum top 10 | 1.336 |

This implies that screen selection has independent explanatory power, not merely controller decoration.

### 2. More complex weighting is not automatically better after screening

Equal weighting is the best average controller for the 100- and 250-name pilots. In the 500-name pilot, the 21-day rank-and-hold controller narrowly exceeds equal weighting on average across screens.

| Universe | Best controller averaged over screens | Mean Sharpe across screens |
|---|---|---:|
| Liquid US Equity 100 | Equal weight | 1.380 |
| Liquid US Equity 250 | Equal weight | 0.572 |
| Liquid US Equity 500 | 21-day rank-and-hold | 1.433 |

This is practical: after a good screen, simple allocations may be more robust than layering another high-turnover signal, except when breadth is large enough for the second layer to find additional cross-sectional separation.

### 3. The 500-name pilot is the strongest evidence for breadth

The highest Sharpe in the expanded matrix occurs in the broadest universe with volatility-adjusted momentum screening and 21-day rank-and-hold weighting. The result is consistent with the intuition that larger universes make selection valuable, but only if the selection rule controls for volatility and not just raw returns.

## Claim discipline

Strong enough for the paper:

> The experiment shows that universe breadth, screening rule, and allocation controller interact materially; no single controller dominates across all breadth levels.

Moderate claim:

> Equal weighting is difficult to beat after a strong screen in smaller/intermediate candidate universes, while an additional short-horizon ranking layer becomes more useful in the broadest candidate universe.

Exploratory claim:

> Volatility-adjusted momentum appears especially promising for broad liquid-equity candidate sets, but the evidence remains symbol-limited and current-constituent biased.

Do **not** overclaim:

- These are not point-in-time constituent universes.
- Delisted names are not included.
- Walk-forward folds overlap, so multiple-testing diagnostics are descriptive rather than formal inference.
- The meta-controller remains under-sampled and should be framed as infrastructure plus early negative evidence, not as a solved routing model.

## Report artifact paths

Primary tables:

- `metrics/screened_core_summary.csv`
- `metrics/screened_universe_winners.csv`
- `metrics/screened_universe_screen_winners.csv`
- `metrics/screen_ablation_summary.csv`
- `metrics/controller_ablation_summary.csv`

Robustness / rigor:

- `metrics/screened_core_subperiod_summary.csv`
- `metrics/gross_vs_net_summary.csv`
- `metrics/turnover_summary.csv`
- `metrics/multiple_testing_summary.csv`
- `metrics/deflated_sharpe_summary.csv`
- `metrics/screened_core_all_trials.csv`
- `metrics/screened_core_model_selection_ledger.csv`
- `metrics/screened_core_membership.csv`
- `metrics/screen_membership_concentration.csv`

Figures:

- `figures/screened_core_sharpe_heatmap.png`
- `figures/screened_core_risk_return_map.png`

Scripts:

- `scripts/run_screened_core_matrix.py`
- `scripts/consolidate_screened_core.py`

## Recommended final-paper section placement

This should become the central empirical section after the original controller-comparison section:

1. **Baseline controller comparison:** show that controller identity matters.
2. **Context-aware extension:** introduce universe breadth and structural descriptors.
3. **Screened core matrix:** present this matrix as the deepest mechanism-identification experiment.
4. **Meta-controller and RL:** present as forward-looking control layers with honest current limitations.
5. **Discussion:** argue that practical portfolio optimization is a layered control problem.
