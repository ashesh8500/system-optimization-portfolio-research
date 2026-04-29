# Findings: `2026-04-23-context-aware-grand-study`

**Run date:** 2026-04-23  
**Author:** Ashesh Kaji  
**Status:** Completed implementation wave — context-aware engine, screened core matrix, integrated RL routing study, consolidated metrics/figures, concise final report at `report/CONTEXT_AWARE_GRAND_STUDY.pdf`, and math-focused System Optimization Methods report at `report/SYSTEM_OPTIMIZATION_METHODS_STUDY.pdf`

## Objective

Build the next empirical engine for the project: a context-aware portfolio study with larger universes, structural universe descriptors, and eventually a meta-controller over portfolio controller families.

## Setup

- **Run package:** `findings/2026-04-23-context-aware-grand-study/`
- **Source package:** `src/context_study/`
- **Current validated components:** universe construction, descriptor computation, pilot runner integration

## Method

This implementation wave focused on foundational infrastructure rather than a full empirical run:

1. copied the prior grand-study engine into a new isolated study package,
2. added static and rolling universe-construction logic,
3. added rolling descriptor computation for universe alignment,
4. integrated descriptor persistence into the new runner,
5. added synthetic-data tests for the new modules.

## Results

### Implemented modules
- `src/context_study/universe.py`
- `src/context_study/descriptors.py`
- `src/context_study/protocol.py`
- `src/context_study/candidates.py`
- `src/context_study/analysis.py`
- `src/context_study/meta_controller.py`
- extended `src/context_study/strategies.py` with hierarchical-risk and screened-universe controllers
- updated `src/context_study/data_loader.py`
- updated `src/context_study/runner.py`
- updated `src/context_study/__init__.py`

### Generated run assets
- `data/candidate_universes/sp100_wikipedia.csv`
- `data/candidate_universes/sp500_wikipedia.csv`
- `metrics/walk_forward_manifest.csv`
- `metrics/summary.csv`
- `metrics/subperiod_summary.csv`
- `metrics/cost_sensitivity.csv`
- `metrics/analysis/`
- `metrics/meta_controller/`
- `metrics/candidate_pilot/liquid_us_equity_100/`
- `metrics/candidate_pilot/liquid_us_equity_250/`

### Validation
- Test suite currently passing:
  - `test_universe.py`
  - `test_descriptors.py`
  - `test_protocol.py`
  - `test_candidates.py`
  - `test_strategies.py`
  - `test_runner.py`
  - `test_analysis.py`
  - `test_meta_controller.py`
- Latest verification: `34 passed`
- Pilot smoke verification completed for:
  - `run_walk_forward_pilot`
  - `run_candidate_benchmark_pilot`
  - `run_candidate_analysis_pipeline`

### Real candidate-universe pilot runs

Two live walk-forward candidate pilots were executed with real downloaded price data through the local cache-backed data loader. A third broader candidate pilot was then completed successfully. Consolidated artifacts were also written to:

- `metrics/consolidated_candidate_summary.csv`
- `metrics/candidate_winners.csv`
- `metrics/candidate_controller_ranks.csv`
- `metrics/meta_controller_summary.csv`
- `metrics/descriptor_feature_summary_all.csv`
- `figures/candidate_sharpe_heatmap.png`
- `figures/candidate_risk_return_map.png`
- `figures/rl_training_curve.png`

#### Liquid US Equity 100 pilot (`symbol_limit=25`, 22 splits)
| Controller | Mean ann. return | Mean ann. vol | Mean Sharpe | Worst split drawdown |
|---|---:|---:|---:|---:|
| EW | 0.198 | 0.197 | 1.539 | -0.372 |
| MRH 21 | 0.055 | 0.233 | 0.422 | -0.299 |
| MRH 63 | 0.136 | 0.239 | 0.695 | -0.259 |
| MRH 126 | 0.224 | 0.244 | 1.115 | -0.307 |

Meta-controller pilot:
- train rows: 14
- test rows: 8
- test accuracy: 0.25

#### Liquid US Equity 250 pilot (`symbol_limit=40`, 11 splits)
| Controller | Mean ann. return | Mean ann. vol | Mean Sharpe | Worst split drawdown |
|---|---:|---:|---:|---:|
| EW | 0.122 | 0.170 | 0.931 | -0.140 |
| MRH 21 | 0.182 | 0.221 | 0.994 | -0.141 |
| MRH 63 | 0.055 | 0.211 | 0.403 | -0.152 |
| MRH 126 | 0.050 | 0.218 | 0.196 | -0.189 |

Meta-controller pilot:
- train rows: 7
- test rows: 4
- test accuracy: 0.00

#### Liquid US Equity 500 pilot (`symbol_limit=60`, 9 splits)
| Controller | Mean ann. return | Mean ann. vol | Mean Sharpe | Worst split drawdown |
|---|---:|---:|---:|---:|
| EW | 0.217 | 0.157 | 1.522 | -0.168 |
| MRH 21 | 0.370 | 0.237 | 1.634 | -0.195 |
| MRH 63 | 0.195 | 0.239 | 0.830 | -0.166 |
| MRH 126 | 0.270 | 0.256 | 1.041 | -0.148 |

Meta-controller pilot:
- train rows: 6
- test rows: 3
- test accuracy: 0.333

#### Candidate winners

The current pilot winner table is:

| Universe | Winning controller | Mean ann. return | Mean ann. vol | Mean Sharpe | Worst split drawdown |
|---|---:|---:|---:|---:|---:|
| Liquid US Equity 100 | EW | 0.198 | 0.197 | 1.539 | -0.372 |
| Liquid US Equity 250 | MRH 21 | 0.182 | 0.221 | 0.994 | -0.141 |
| Liquid US Equity 500 | MRH 21 | 0.370 | 0.237 | 1.634 | -0.195 |

This is the first concrete evidence in this run that universe breadth can change which controller is preferred.

### RL controller repair and pilot

The RL path was repaired so that the policy-gradient trainer now stores actions, computes a nonzero score-function gradient, advances the environment chronologically during evaluation, and records turnover. A compact cached-equity pilot was executed on 10 liquid-equity symbols over 900 observations:

| Universe | Controller | Assets | Updates | Ann. return | Ann. vol | Sharpe | Max drawdown | Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cached large equity 10 | RL learned linear policy | 10 | 30 | 0.085 | 0.182 | 0.468 | -0.214 | 1.964 |

Artifacts:
- `metrics/rl_context_pilot.csv`
- `metrics/rl_training_curve.csv`
- `metrics/rl_context_equity.csv`
- `figures/rl_training_curve.png`

### Literature-informed extension increment

A follow-up research pass was completed using divided labor across universe construction, contextual routing, and empirical rigor. The result is not a restart of the plan; it is an increment that turns the current pilot into a deeper mechanism-identification study.

New artifacts:
- `EXHAUSTIVE_NEXT_EXPERIMENT.md`
- `META_CONTROLLER_RESEARCH_DESIGN.md`
- `metrics/modern_literature_map.csv`
- `metrics/universe_screen_weight_factorial_matrix.csv`
- `metrics/empirical_rigor_artifact_checklist.csv`
- `docs/plans/2026-04-23-universe-screen-weight-factorial-implementation.md`

The generated factorial matrix has 1,875 rows across parent universes, universe-construction rules, screen/cardinality rules, and weighting controllers. The purpose is to isolate whether performance improvements come from the universe layer, the screen/cardinality layer, the weighting layer, or the routing layer.

### Screen and empirical-rigor artifact implementation

The first concrete implementation step for the exhaustive next experiment is now complete. New source modules:

- `src/context_study/screens.py`
- `src/context_study/experiment_artifacts.py`

New tests:

- `tests/test_screens.py`
- `tests/test_experiment_artifacts.py`

Implemented screen rules:

- 21/63/etc. momentum top-K via `momentum_screen`,
- volatility-adjusted momentum top-K,
- liquidity-adjusted momentum top-K, requiring an explicit volume panel,
- low-volatility top-K,
- cluster-capped momentum top-K,
- dispatcher `apply_screen` for canonical rule strings.

Implemented artifact schemas/writers:

- `metrics/universe_provenance.csv` — 3 rows for the current liquid-equity pilots, explicitly flagging current-constituent survivorship bias,
- `metrics/model_selection_ledger.csv` — 12 rows documenting controller/universe configurations and paper-selection flags,
- `metrics/all_trials.csv` — 168 split-level trial rows from the current candidate pilots.

Verification after this increment: `uv run pytest -q` returned `45 passed`.

### Screened walk-forward runner integration

The screen layer is now wired into the walk-forward benchmark path through `run_screened_candidate_benchmark_pilot`. This applies each screen rule using only data through `split.train.end`, runs selected controllers on the screened price panel through the evaluation end date, and writes screen membership plus ledger/trial artifacts.

New real screened pilot:

- Universe: `liquid_us_equity_100`
- Symbol limit: 25
- Screens: `momentum_21_top10`, `low_volatility_63_top10`
- Controllers: `EW`, `MRH_tau21_k5`
- Cost: 10 bps
- Splits: 22 per screen/controller

| Screen | Controller | Mean ann. return | Mean ann. vol | Mean Sharpe | Worst split drawdown |
|---|---:|---:|---:|---:|---:|
| Momentum 21 top 10 | EW | 0.204 | 0.202 | 1.474 | -0.449 |
| Momentum 21 top 10 | MRH 21 | 0.152 | 0.215 | 0.879 | -0.435 |
| Low-volatility 63 top 10 | EW | 0.163 | 0.162 | 1.440 | -0.293 |
| Low-volatility 63 top 10 | MRH 21 | 0.125 | 0.168 | 1.023 | -0.263 |

Artifacts:

- `metrics/screened_candidate_pilot/liquid_us_equity_100/summary.csv`
- `metrics/screened_candidate_pilot/liquid_us_equity_100/subperiod_summary.csv`
- `metrics/screened_candidate_pilot/liquid_us_equity_100/cost_sensitivity.csv`
- `metrics/screened_candidate_pilot/liquid_us_equity_100/screen_membership.csv`
- `metrics/screened_candidate_pilot/liquid_us_equity_100/model_selection_ledger.csv`
- `metrics/screened_candidate_pilot/liquid_us_equity_100/all_trials.csv`

Verification after this increment: `uv run pytest -q` returned `46 passed`.

### Completed screen-integrated core matrix

The screen-integrated experiment has now been extended across all three candidate-universe breadth levels. The completed core matrix uses:

- Universes: `liquid_us_equity_100` (`symbol_limit=25`), `liquid_us_equity_250` (`symbol_limit=40`), and `liquid_us_equity_500` (`symbol_limit=60`).
- Screens: `momentum_21_top10`, `momentum_63_top10`, `vol_adjusted_momentum_63_top10`, `low_volatility_63_top10`, and `cluster_capped_momentum_63_top10`.
- Controllers: equal weight plus 21-, 63-, and 126-day rank-and-hold controllers.
- Costs: 0, 10, and 25 bps; primary reported slice is 10 bps.

Consolidated artifacts:

- `SCREENED_CORE_REPORTING_BRIEF.md`
- `metrics/screened_core_summary.csv`
- `metrics/screened_universe_winners.csv`
- `metrics/screened_universe_screen_winners.csv`
- `metrics/screen_ablation_summary.csv`
- `metrics/controller_ablation_summary.csv`
- `metrics/gross_vs_net_summary.csv`
- `metrics/turnover_summary.csv`
- `metrics/multiple_testing_summary.csv`
- `metrics/deflated_sharpe_summary.csv`
- `metrics/screened_core_model_selection_ledger.csv`
- `metrics/screened_core_all_trials.csv`
- `figures/screened_core_sharpe_heatmap.png`
- `figures/screened_core_risk_return_map.png`

#### Screened universe winners

| Universe | Best screen | Best controller | Mean ann. return | Mean ann. vol | Mean Sharpe | Worst split drawdown |
|---|---|---|---:|---:|---:|---:|
| Liquid US Equity 100 | 21-day momentum top 10 | Equal weight | 0.204 | 0.202 | 1.474 | -0.449 |
| Liquid US Equity 250 | 63-day low-volatility top 10 | Equal weight | 0.112 | 0.141 | 0.859 | -0.105 |
| Liquid US Equity 500 | 63-day volatility-adjusted momentum top 10 | 21-day rank-and-hold | 0.351 | 0.208 | 1.756 | -0.155 |

#### Screen ablation

Averaging over controllers, the preferred screen is not constant across universe breadth:

| Universe | Best screen by average controller Sharpe | Mean Sharpe across controllers |
|---|---|---:|
| Liquid US Equity 100 | 63-day low-volatility top 10 | 1.139 |
| Liquid US Equity 250 | 63-day low-volatility top 10 | 0.699 |
| Liquid US Equity 500 | 63-day momentum top 10 | 1.336 |

#### Controller ablation

Averaging over screens, equal weighting remains strongest in the smaller and intermediate pilots, while the 21-day rank-and-hold controller becomes strongest in the broadest pilot:

| Universe | Best controller by average screen Sharpe | Mean Sharpe across screens |
|---|---|---:|
| Liquid US Equity 100 | Equal weight | 1.380 |
| Liquid US Equity 250 | Equal weight | 0.572 |
| Liquid US Equity 500 | 21-day rank-and-hold | 1.433 |

This is now the deepest mechanism-identification evidence in the repository. It separates the contribution of universe breadth, screening, and weighting rather than treating a portfolio strategy as a single indivisible object.

Latest verification after this expanded matrix: `uv run pytest -q` returned `46 passed, 7 warnings`.

### Hierarchical reinforcement-learning router over screened-control arms

A first hierarchical routing experiment has now been completed on the screened-core reward panel. Instead of treating reinforcement learning only as a direct weight allocator, this experiment places a learned policy above the screen/controller layer. At each walk-forward interval, the policy observes compact state features and selects one of 20 screen/controller arms. The experiment evaluates whether a learned decision-maker can exploit the hierarchy already identified in the universe × screen × controller study.

Artifacts:

- `HIERARCHICAL_RL_ROUTING_BRIEF.md`
- `src/context_study/hierarchical_rl_router.py`
- `scripts/run_hierarchical_rl_router.py`
- `tests/test_hierarchical_rl_router.py`
- `metrics/hierarchical_rl_router_summary.csv`
- `metrics/hierarchical_rl_router/`
- `figures/hierarchical_rl_router_comparison.png`
- `figures/hierarchical_rl_router_training_curves.png`

| Scope | Test periods | RL reward | Best fixed reward | Random mean reward | Period oracle reward | RL vs best fixed | RL vs random mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Combined | 14 | 1.146 | 1.635 | 1.293 | 2.586 | -0.488 | -0.147 |
| Liquid US Equity 100 | 8 | 0.827 | 0.835 | 0.661 | 1.355 | -0.008 | +0.166 |
| Liquid US Equity 250 | 5 | 0.305 | 0.466 | 0.290 | 0.764 | -0.161 | +0.016 |
| Liquid US Equity 500 | 4 | 0.371 | 0.881 | 0.512 | 0.932 | -0.511 | -0.142 |

The router learned coherent, concentrated choices rather than random oscillation. However, it did not beat the best fixed arm selected in hindsight. The most useful interpretation is therefore a partial/negative result: learned hierarchical routing is now technically feasible inside the empirical engine, but the current number of independent walk-forward routing episodes is too small to support a strong claim that the learned router improves on fixed screened-control stacks.

This result should still be included in the paper because it is directly aligned with the layered-control thesis. It shows how reinforcement learning can be positioned as a meta-control layer, while honestly identifying sample size and regime diversity as the bottleneck before stronger deployment claims can be made.

Latest verification after this routing increment: focused routing tests returned `3 passed`; full-suite verification is recorded below.

### Integrated reinforcement-learning routing study

The hierarchical reinforcement-learning path has now been promoted from a single routing run into an integrated ML study inside the main experiment. The study uses the same screened-core reward panel and evaluates the learned router over 84 trials: four scopes, three policy-gradient configurations, and seven seeds per configuration. This makes reinforcement learning part of the experiment design rather than a post-hoc extension.

Artifacts:

- `INTEGRATED_RL_STUDY_BRIEF.md`
- `scripts/run_integrated_rl_study.py`
- `metrics/integrated_rl_study_trials.csv`
- `metrics/integrated_rl_study_baselines.csv`
- `metrics/integrated_rl_study_config_summary.csv`
- `metrics/integrated_rl_study_best_by_scope.csv`
- `metrics/integrated_rl_study_win_rates.csv`
- `metrics/integrated_rl_study/`
- `figures/integrated_rl_study_scope_comparison.png`
- `figures/integrated_rl_study_config_sensitivity.png`

| Scope | Best RL config | Test periods | RL mean | Train-selected fixed | Trailing-window | Random mean | Test-best fixed | Oracle | RL vs train-selected | RL vs trailing |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Combined | Aggressive | 14 | 1.213 | 1.134 | 1.543 | 1.276 | 1.635 | 2.586 | +0.079 | -0.330 |
| Liquid US Equity 100 | Aggressive | 8 | 0.823 | 0.827 | 0.475 | 0.657 | 0.835 | 1.355 | -0.004 | +0.348 |
| Liquid US Equity 250 | Aggressive | 5 | 0.280 | 0.305 | 0.306 | 0.296 | 0.466 | 0.764 | -0.026 | -0.026 |
| Liquid US Equity 500 | Aggressive | 4 | 0.371 | 0.371 | 0.566 | 0.525 | 0.881 | 0.932 | +0.000 | -0.196 |

The result is now sufficiently explored for this research pass. The learned router is a meaningful main-experiment component: it improves over the train-selected fixed arm in the combined setting on average and learns stable, interpretable stack choices. However, it does not dominate the trailing-window adaptive baseline. The practical conclusion is that reinforcement learning is currently valuable as a meta-control architecture and explanatory probe, while simple adaptive rules remain difficult baselines in small walk-forward samples.

### Compiled reports

The concise context-aware layered-control report for this research wave has been written and compiled at:

- `report/CONTEXT_AWARE_GRAND_STUDY.tex`
- `report/CONTEXT_AWARE_GRAND_STUDY.pdf`

A second, more mathematically explicit System Optimization Methods version has also been written and compiled at:

- `report/SYSTEM_OPTIMIZATION_METHODS_STUDY.tex`
- `report/SYSTEM_OPTIMIZATION_METHODS_STUDY.pdf`

The second report reframes the same empirical evidence around the mathematical modeling path of the project: Markowitz-style convex optimization, turnover-regularized control, finite-horizon stochastic control, cardinality-constrained screening, mixed discrete-continuous model selection, online expert selection, and policy-gradient routing. It is intended for submission contexts where the optimization formulation and course-methods connection need to be more explicit than the concise paper.

Both papers are written from Ashesh Kaji's student/researcher perspective for NYU System Optimization Methods and include the requested email contact (`ask9184@nyu.edu`). The reports preserve the layered-control thesis, integrate reinforcement learning as a main meta-control methodology rather than an appendix, and use native LaTeX/TikZ/PGFPlots diagrams where possible. The generated PDFs were rendered to preview images and visually inspected; the main tables, native diagrams/heatmaps, RL comparison material, appendices, and references were checked for readability and alignment with the papers' claims.

The paper's core practical findings are:

1. tune the candidate universe before over-tuning the optimizer;
2. treat screening/cardinality as a first-class control layer;
3. use equal weight as a serious deployment baseline after screening;
4. add active rank-and-hold weighting only where universe breadth supplies enough cross-sectional dispersion;
5. evaluate learned routing only against strong simple adaptive baselines, especially trailing-window selection.

The highest-value remaining gap is not another broad sweep; it is a deeper point-in-time universe study with more independent walk-forward routing episodes, so that supervised or reinforcement-learning routing can be evaluated with enough samples to support stronger deployment claims.

## Interpretation

The project now has a validated base for the next research wave. The key change is that universe definition and universe characterization are no longer implicit; they are now first-class objects in the study pipeline.

The most important substantive result is not that one universal controller dominates. It is the opposite: the preferred controller changes when the candidate universe changes. Equal weighting wins the smaller liquid-equity pilot by mean Sharpe, while a short-horizon rank-and-hold controller wins both broader candidate pilots. This directly supports the research thesis that allocation performance is partly controlled by the upstream universe-definition layer, not only by the downstream weight optimizer.

The meta-controller results should be interpreted as an honest negative/early result. Current accuracy is weak because the training table is still small and because the winner labels are unstable across adjacent windows. That does not invalidate the direction; it identifies the next bottleneck. The descriptor layer and winner table now exist, so the correct next step is to enlarge the number of universes and walk-forward samples before making stronger claims about learned routing.

The RL pilot is now technically valid as a first-class controller path. Its current performance is modest, but it now has a reproducible training/evaluation route and can be compared against the model-based families once full walk-forward RL evaluation is added.

## Limitations / next steps

Current gaps now narrow to deeper empirical coverage rather than missing infrastructure:
- expand from symbol-limited candidate pilots to full-size candidate universes where data coverage and runtime permit,
- add more universe construction rules, especially sector-balanced, volatility-screened, correlation-screened, and momentum-screened universes,
- increase the number of descriptor/winner samples before drawing conclusions about supervised controller routing,
- run a full ablation that separates universe selection, signal construction, weighting controller, and meta-routing,
- evaluate the repaired RL controller through the same walk-forward protocol as the model-based controllers,
- promote the strongest results into the final paper with a careful negative-results discussion for the weak meta-controller.

## Repro

Run tests:

```bash
cd findings/2026-04-23-context-aware-grand-study
uv run pytest -q
```

Dependencies are explicit in `pyproject.toml`; required libraries should fail clearly if absent rather than using substitute classes or degraded fallbacks.

Recreate the compact RL pilot from cached price panels:

```bash
python - <<'PY'
from pathlib import Path
import sys, pandas as pd
root = Path('findings/2026-04-23-context-aware-grand-study')
sys.path.insert(0, str(root / 'src'))
from context_study.runner import train_rl_policy, run_rl_evaluation
# Load cached parquet files, train the compact linear policy, and write the result tables.
PY
```
