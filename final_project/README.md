# System Optimization Final Project: Portfolio Allocation as Layered Control

This folder contains the working research materials, final report, interactive presentation, and reproducible experiment artifacts for Ashesh Kaji's System Optimization Methods final project.

## Presentation

Open the interactive website:

```bash
cd /path/to/systemopt
python3 -m http.server 8080
# visit http://localhost:8080/final_project/presentation/
```

Controls:

- `Space` / `→`: next slide
- `←`: previous slide
- `N`: speaker notes
- `Scroll` button: switch from flip-through deck mode to a vertical page

## Final report

- Final submission source: `proposal/final_submission/FINAL_PROJECT_REPORT.tex`
- Final PDF: `proposal/final_submission/FINAL_PROJECT_REPORT.pdf`
- Proposal materials: `proposal/`

## Main empirical run

The final presentation and report draw primarily from:

```text
findings/2026-04-23-context-aware-grand-study/
```

Key artifacts:

- `metrics/screened_core_summary.csv` — 60-arm main matrix.
- `metrics/screened_universe_winners.csv` — best stack by universe breadth.
- `metrics/screen_ablation_summary.csv` — screen-layer ablation.
- `metrics/controller_ablation_summary.csv` — weighting-controller ablation.
- `metrics/gross_vs_net_summary.csv` — transaction-cost sensitivity.
- `metrics/integrated_rl_study_best_by_scope.csv` — adaptive router comparison.
- `report/SYSTEM_OPTIMIZATION_METHODS_STUDY.pdf` — math-focused study paper.

## Reproduce environment

```bash
cd findings/2026-04-23-context-aware-grand-study
uv sync --dev
uv run pytest
```

The experiment package is under `src/context_study/`; scripts live under `scripts/`.

## Core research claim

Portfolio allocation is better modeled as a layered sequential control problem than as a single static optimizer. The binding layer changes with candidate-universe breadth:

- **Top-100:** 21-day momentum screen + equal weighting.
- **Top-250:** 63-day low-volatility screen + equal weighting.
- **Top-500:** 63-day volatility-adjusted momentum screen + 21-day rank-and-hold weighting.

The adaptive router learns stable preferences over these arms, but in this finite-sample study it does not dominate a simple trailing-window router. That comparison is central to the research: additional optimization complexity is only useful when it beats transparent baselines.
