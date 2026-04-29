# Reproducibility guide

This guide explains how to study or extend the portfolio-allocation research without depending on local-only context.

## 1. Open the presentation

```bash
cd /path/to/systemopt
python3 -m http.server 8080
open http://localhost:8080/final_project/presentation/
```

The website is self-contained. Its charts embed the canonical 60-arm summary and router results directly in `final_project/presentation/index.html`.

## 2. Read the research narrative

Recommended order:

1. `final_project/proposal/final_submission/FINAL_PROJECT_REPORT.pdf`
2. `final_project/PRESENTATION_SCRIPT.md`
3. `final_project/docs/RESEARCH_CONTEXT.md`
4. `final_project/findings/2026-04-23-context-aware-grand-study/SCREENED_CORE_REPORTING_BRIEF.md`
5. `final_project/findings/2026-04-23-context-aware-grand-study/report/SYSTEM_OPTIMIZATION_METHODS_STUDY.pdf`

## 3. Inspect canonical metrics

The presentation is generated from these CSV artifacts:

```text
final_project/findings/2026-04-23-context-aware-grand-study/metrics/screened_core_summary.csv
final_project/findings/2026-04-23-context-aware-grand-study/metrics/screened_universe_winners.csv
final_project/findings/2026-04-23-context-aware-grand-study/metrics/screen_ablation_summary.csv
final_project/findings/2026-04-23-context-aware-grand-study/metrics/controller_ablation_summary.csv
final_project/findings/2026-04-23-context-aware-grand-study/metrics/gross_vs_net_summary.csv
final_project/findings/2026-04-23-context-aware-grand-study/metrics/integrated_rl_study_best_by_scope.csv
```

## 4. Recreate the Python environment

```bash
cd final_project/findings/2026-04-23-context-aware-grand-study
uv sync --dev
uv run pytest
```

The project requires Python 3.12+.

## 5. Extend the study

Good next experiments:

- rebuild point-in-time universe membership to reduce survivorship bias;
- compute universe descriptors such as correlation concentration, dispersion, realized volatility, trend persistence, and liquidity proxies;
- increase the number of walk-forward episodes before making strong claims about learned routing;
- replace proportional cost with a liquidity-aware nonlinear impact model.

When adding new results, keep metrics in CSV form with clear schemas and document the interpretation in a `FINDINGS.md` file under a dedicated run folder.
