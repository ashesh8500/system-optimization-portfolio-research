# Paper Integration Memo: Context-Aware Portfolio Control

**Author:** Ashesh Kaji  
**Run:** `2026-04-23-context-aware-grand-study`  
**Purpose:** Convert the context-aware empirical engine into paper-ready claims, caveats, and next experiments.

## Thesis contribution

The new context-aware study strengthens the paper by moving the project from a broad controller comparison to a layered control interpretation of portfolio allocation. The allocation problem is no longer treated as only a weight-selection problem. It is modeled as a hierarchy:

1. **Universe definition:** which securities are eligible for allocation.
2. **Universe characterization:** what structural regime the chosen universe exhibits.
3. **Signal and controller selection:** which allocation rule is appropriate for that structure.
4. **Weight execution:** how the selected controller maps state into portfolio weights under turnover and cost constraints.
5. **Meta-control:** whether a higher-level classifier can learn the mapping from universe descriptors to controller choice.

The current empirical evidence supports the first three layers more strongly than the fifth. This is academically useful: the study identifies where structure exists and where additional data is required.

## Paper-ready empirical finding

Across three candidate-universe pilots, the best controller changes with universe breadth:

| Candidate universe | Symbol limit | Splits | Best controller | Mean Sharpe | Mean ann. return | Mean ann. vol |
|---|---:|---:|---|---:|---:|---:|
| Liquid US Equity 100 | 25 | 22 | Equal-weight allocation | 1.539 | 0.198 | 0.197 |
| Liquid US Equity 250 | 40 | 11 | 21-day rank-and-hold allocation | 0.994 | 0.182 | 0.221 |
| Liquid US Equity 500 | 60 | 9 | 21-day rank-and-hold allocation | 1.634 | 0.370 | 0.237 |

This result should be framed carefully: the symbol limits are pilot limits, not full-universe production runs. However, the pattern is already meaningful. Equal weighting is competitive in the narrower liquid-equity set, while short-horizon ranking becomes more attractive as the candidate universe broadens.

## Interpretation for the paper

A practical portfolio system should not only ask: *which weighting rule is best?* It should also ask: *for what universe is this rule being applied?* The universe layer changes the opportunity set, the dispersion of signals, the correlation structure, and the degree to which a ranking rule can exploit cross-sectional separation. This creates a natural system-optimization problem with control variables at multiple tiers.

The context-aware results therefore motivate an opinionated architecture:

- use descriptors to measure whether the current universe is concentrated, correlated, volatile, or trend-persistent;
- choose candidate controllers based on those descriptors;
- evaluate controller selection out of sample using walk-forward splits;
- treat weak routing results as evidence that more samples and richer universe construction are required, not as a reason to abandon the layered formulation.

## Honest negative result

The first supervised meta-controller is weak:

| Universe | Train rows | Test rows | Test accuracy |
|---|---:|---:|---:|
| Liquid US Equity 100 | 14 | 8 | 0.250 |
| Liquid US Equity 250 | 7 | 4 | 0.000 |
| Liquid US Equity 500 | 6 | 3 | 0.333 |

This should be included in the paper as a limitation. The result says that the present descriptor/winner table is too small to support a strong learned routing claim. It does **not** invalidate the universe-alignment thesis; it clarifies the next empirical requirement.

## RL integration note

The reinforcement-learning path is now technically valid in two forms. First, the repaired linear policy-gradient controller was trained and evaluated chronologically on a cached 10-equity panel:

| Controller | Assets | Updates | Ann. return | Ann. vol | Sharpe | Max drawdown | Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| Linear policy-gradient allocation | 10 | 30 | 0.085 | 0.182 | 0.468 | -0.214 | 1.964 |

Second, and more relevant to the layered-control thesis, a hierarchical reinforcement-learning router has been evaluated on the completed screened-core reward panel. In this formulation, the learned policy is not a direct weight allocator; it is a router over 20 screen/controller arms. The router observes compact state features and selects which control stack to apply at the next walk-forward interval.

| Scope | Test periods | RL reward | Best fixed reward | Random mean reward | Period oracle reward | RL vs best fixed |
|---|---:|---:|---:|---:|---:|---:|
| Combined | 14 | 1.146 | 1.635 | 1.293 | 2.586 | -0.488 |
| Liquid US Equity 100 | 8 | 0.827 | 0.835 | 0.661 | 1.355 | -0.008 |
| Liquid US Equity 250 | 5 | 0.305 | 0.466 | 0.290 | 0.764 | -0.161 |
| Liquid US Equity 500 | 4 | 0.371 | 0.881 | 0.512 | 0.932 | -0.511 |

This is an honest partial result. The router learned coherent choices and exceeded random routing in the 100- and 250-name scopes, but it did not beat the best fixed stack. The paper should present this as evidence that reinforcement learning is now integrated as a first-class meta-control layer, while also stating that more independent routing episodes are required before claiming learned routing superiority.

A subsequent integrated routing study turned the single-run probe into a repeated-seed ML experiment. Across four scopes, three policy-gradient configurations, and seven seeds per configuration, the best learned router improved over the train-selected fixed baseline in the combined scope by approximately 0.079 total test-period reward on average, nearly matched the train-selected fixed arm in the 100- and 500-name scopes, and underperformed in the 250-name scope. The learned router did not dominate a simple trailing-window adaptive baseline, which is an important practical caution. This is the version of the RL result that should be integrated into the final paper: reinforcement learning is not a separate extension, but a meta-control layer within the main hierarchy whose current contribution is architectural and explanatory rather than conclusively performance-dominant.

## Recommended final-paper framing

The completed screen-integrated matrix strengthens the framing. The paper can now move beyond the earlier controller-only claim and present a three-layer empirical result:

1. **Universe breadth matters:** the best stack changes between the 100-, 250-, and 500-name candidate pilots.
2. **Screen choice matters:** low-volatility screens dominate the smaller/intermediate average-screen ablation, while momentum-style screens become more effective in the broadest universe.
3. **Weighting value is conditional:** equal weighting is difficult to beat after screening in narrower/intermediate settings, but a short-horizon rank-and-hold layer becomes valuable in the broadest setting.

The strongest current title-level claim is:

> Portfolio optimization is better understood as a layered control problem in which universe construction, structural descriptors, controller choice, and allocation weights jointly determine performance.

The strongest current empirical claim is:

> In symbol-limited liquid-equity pilots, the best-performing control stack changed as the candidate universe broadened: a 21-day momentum screen with equal weighting led the 100-name pilot, low-volatility screening with equal weighting led the 250-name pilot, and volatility-adjusted momentum with a short-horizon rank-and-hold layer led the 500-name pilot.

The strongest limitation is:

> The first descriptor-based meta-controller did not yet generalize reliably because the number of walk-forward winner samples remains small.

## Next experiment that would deepen the paper most

The highest-value next experiment is not another broad controller comparison. It is a controlled ablation:

1. Fix the same date range and transaction cost.
2. Construct multiple universes from the same parent candidate set:
   - top liquidity,
   - low volatility,
   - high momentum,
   - low correlation / diversified,
   - sector-balanced.
3. Run the same controller set on each universe.
4. Report whether controller winners align with universe descriptors.
5. Train the meta-controller only after enough winner samples exist.

That experiment would turn the current motivating pattern into a deeper academic result.
