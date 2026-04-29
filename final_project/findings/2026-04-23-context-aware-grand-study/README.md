# Context-Aware Grand Study

This run package extends the grand-system study toward larger universes, universe-alignment descriptors, and contextual controller selection.

Current implementation status:
- base study engine scaffold copied into `src/context_study/`
- rolling/static universe construction implemented
- descriptor computation implemented
- walk-forward protocol implemented
- S&P-based candidate-universe scaffolding implemented under `data/candidate_universes/`
- hierarchical-risk and screened-universe controller extensions implemented
- descriptor-to-winner analysis and a first supervised meta-controller implemented
- pilot runner integration for descriptor persistence, walk-forward benchmark outputs, candidate-based benchmark pilots, and analysis/meta-controller artifacts implemented
- RL training/evaluation path repaired and validated with a compact cached-equity pilot
- hierarchical reinforcement-learning router implemented and evaluated on the completed screened-core reward panel
- integrated reinforcement-learning routing study completed as part of the main experiment with repeated seeds, hyperparameter sensitivity, deployable baselines, random baselines, and oracle bounds
- consolidated metrics and figures generated under `metrics/` and `figures/`
- tests currently cover universe construction, descriptors, protocol logic, candidate-list loading, strategy extensions, runner integration, winner analysis, meta-controller logic, RL train/evaluate plumbing, screen rules, empirical-rigor artifact writers, and screened runner integration


Primary source code:
- `src/context_study/`

Current test command:
- `uv run pytest -q` from `findings/2026-04-23-context-aware-grand-study/`
- fallback command only when already inside an equivalent configured environment: `pytest findings/2026-04-23-context-aware-grand-study/tests -q`

Key empirical artifacts:
- `PAPER_INTEGRATION_MEMO.md` — paper-ready interpretation, claims, caveats, and next deep experiment
- `EXHAUSTIVE_NEXT_EXPERIMENT.md` — literature-informed universe/screen/weight factorial extension
- `META_CONTROLLER_RESEARCH_DESIGN.md` — contextual-bandit / mixture-of-experts routing design
- `metrics/modern_literature_map.csv` — citations mapped to concrete design changes
- `metrics/universe_screen_weight_factorial_matrix.csv` — generated 1,875-row next-run experiment matrix
- `metrics/empirical_rigor_artifact_checklist.csv` — required rigor artifacts for paper-ready claims
- `metrics/universe_provenance.csv` — provenance and survivorship-bias flags for current pilots
- `metrics/model_selection_ledger.csv` — controller/universe trial ledger for current pilots
- `metrics/all_trials.csv` — split-level trial table for current pilots
- `SCREENED_CORE_REPORTING_BRIEF.md` — paper-ready interpretation of the completed screen-integrated core matrix
- `HIERARCHICAL_RL_ROUTING_BRIEF.md` — paper-ready interpretation of the learned router over screen/controller arms
- `INTEGRATED_RL_STUDY_BRIEF.md` — main-experiment reinforcement-learning routing study with repeated seeds and robust baselines
- `metrics/screened_candidate_pilot/liquid_us_equity_100/summary.csv` — screen-integrated pilot for the 100-name candidate universe
- `metrics/screened_candidate_pilot/liquid_us_equity_250/summary.csv` — screen-integrated pilot for the 250-name candidate universe
- `metrics/screened_candidate_pilot/liquid_us_equity_500/summary.csv` — screen-integrated pilot for the 500-name candidate universe
- `metrics/screened_core_summary.csv` — consolidated 60-arm universe × screen × controller summary at 10 bps
- `metrics/screened_universe_winners.csv` — best screen/controller stack per universe breadth
- `metrics/screen_ablation_summary.csv` — screen contribution averaged across controllers
- `metrics/controller_ablation_summary.csv` — controller contribution averaged across screens
- `metrics/gross_vs_net_summary.csv`, `metrics/turnover_summary.csv`, `metrics/multiple_testing_summary.csv`, `metrics/deflated_sharpe_summary.csv` — robustness and rigor diagnostics
- `metrics/screened_core_model_selection_ledger.csv` and `metrics/screened_core_all_trials.csv` — consolidated ledger and split-level trial records
- `figures/screened_core_sharpe_heatmap.png` and `figures/screened_core_risk_return_map.png` — report figures for the screened core matrix
- `metrics/hierarchical_rl_router_summary.csv` — learned router evaluation against best-fixed, random, and period-oracle baselines
- `metrics/hierarchical_rl_router/` — router chosen-action traces and training curves by scope
- `figures/hierarchical_rl_router_comparison.png` and `figures/hierarchical_rl_router_training_curves.png` — router comparison and training diagnostics
- `metrics/integrated_rl_study_trials.csv`, `metrics/integrated_rl_study_baselines.csv`, `metrics/integrated_rl_study_config_summary.csv`, `metrics/integrated_rl_study_best_by_scope.csv`, and `metrics/integrated_rl_study_win_rates.csv` — integrated RL study tables
- `metrics/integrated_rl_study/` — per-scope RL trial logs and adaptive baseline action traces
- `figures/integrated_rl_study_scope_comparison.png` and `figures/integrated_rl_study_config_sensitivity.png` — integrated RL study figures
- `metrics/candidate_winners.csv` — winner table for liquid-equity 100/250/500 pilots
- `metrics/consolidated_candidate_summary.csv`
- `metrics/meta_controller_summary.csv`
- `metrics/rl_context_pilot.csv`
- `figures/candidate_sharpe_heatmap.png`
- `figures/candidate_risk_return_map.png`
- `figures/rl_training_curve.png`
