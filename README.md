# Portfolio Allocation as Layered System Optimization

Interactive research presentation and reproducible study materials for Ashesh Kaji's Spring 2026 System Optimization Methods final project at New York University.

## Live presentation

- Local entry point: `final_project/presentation/index.html`
- GitHub Pages entry point after deployment: repository root (`index.html`) redirects to the presentation.
- Presentation mode: use `Space` / `→` to advance, `←` to go back, and `N` to open speaker notes.

## Research summary

This project studies portfolio allocation as a layered optimization/control problem rather than as a single static weighting rule. The final experiment decomposes the decision into:

1. candidate-universe breadth: Top-100, Top-250, Top-500 liquid-equity scopes;
2. cardinality-constrained screening: momentum, low-volatility, volatility-adjusted momentum, and cluster-capped variants;
3. weighting controllers: equal weight and short/medium/long rank-and-hold rules;
4. transaction-cost control: proportional turnover cost, reported primarily at 10 bps;
5. adaptive routing: fixed, trailing-window, random, oracle, and policy-gradient model-selection policies.

Main finding: the best control stack is not invariant. In the tested walk-forward design, Top-100 favors 21-day momentum screening plus equal weighting, Top-250 favors 63-day low-volatility screening plus equal weighting, and Top-500 favors 63-day volatility-adjusted momentum plus 21-day rank-and-hold weighting.

## Repository map

```text
final_project/
├── presentation/                 # Flip-through interactive website
├── proposal/                     # Proposal, final report sources, and final PDF
├── findings/                     # Experiment runs, metrics, reports, scripts
│   └── 2026-04-23-context-aware-grand-study/
│       ├── metrics/              # CSV ledgers used by the presentation/report
│       ├── report/               # Compiled research reports
│       ├── scripts/              # Reproduction and consolidation scripts
│       ├── src/context_study/    # Research package
│       └── tests/                # Synthetic and pipeline tests
├── docs/                         # Reproducibility and research-context notes
├── README.md                     # Project overview inside the course folder
└── PRESENTATION_SCRIPT.md        # Rehearsal script for the final talk
```

## Quick setup

This repository uses `uv` for Python environments.

```bash
# Clone
git clone https://github.com/ashesh8500/system-optimization-portfolio-research.git
cd system-optimization-portfolio-research

# Open the interactive presentation
python3 -m http.server 8080
# then open http://localhost:8080/final_project/presentation/

# Reproduce the main context-aware study environment
cd final_project/findings/2026-04-23-context-aware-grand-study
uv sync --dev
uv run pytest
```

The presentation itself is static HTML/CSS/JavaScript and has no build step. It uses D3 from a CDN for interactive charts; the experiment data needed for the presentation is embedded into `final_project/presentation/index.html`.

## Reproducing study artifacts

The canonical metrics used in the final presentation live under:

```text
final_project/findings/2026-04-23-context-aware-grand-study/metrics/
```

Important files:

- `screened_core_summary.csv` — 60-arm universe × screen × controller summary.
- `screened_universe_winners.csv` — best stack by candidate-universe breadth.
- `screen_ablation_summary.csv` — screen-layer ablation.
- `controller_ablation_summary.csv` — weighting-controller ablation.
- `gross_vs_net_summary.csv` — cost-sensitivity slices.
- `integrated_rl_study_best_by_scope.csv` — router comparison summary.

The raw cached market-data parquet files are intentionally ignored by git because they are large and regenerable. Metrics CSVs and report artifacts are retained for study and review.

## GitHub Pages deployment

A static Pages workflow is included at `.github/workflows/pages.yml`. After pushing to `main`:

1. In GitHub repository settings, set Pages source to **GitHub Actions**.
2. Run or wait for the `Deploy static research presentation` workflow.
3. Open the deployment URL shown in the workflow summary.

The workflow follows GitHub's static Pages pattern: checkout → configure Pages → upload artifact → deploy Pages.

## Academic-use note

This repository is a research/coursework artifact. The empirical results should be interpreted as mechanism evidence from the tested walk-forward design, not as investment advice or a deployable trading recommendation.
