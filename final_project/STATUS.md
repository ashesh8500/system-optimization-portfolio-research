# Project status

**Last updated:** 2026-04-29  
**Course:** System Optimization Methods, Spring 2026  
**Owner:** Ashesh Kaji

## Current research state

The project has converged on a final layered-control framing for portfolio allocation. The canonical empirical package is:

```text
findings/2026-04-23-context-aware-grand-study/
```

The final presentation and report focus on the completed 60-arm walk-forward matrix over candidate-universe breadth, screening rule, weighting controller, transaction-cost sensitivity, and adaptive routing.

## Canonical deliverables

- Interactive presentation: `presentation/index.html`
- Rehearsal script: `PRESENTATION_SCRIPT.md`
- Final report PDF: `proposal/final_submission/FINAL_PROJECT_REPORT.pdf`
- Main research package: `findings/2026-04-23-context-aware-grand-study/`
- Reproducibility guide: `docs/REPRODUCIBILITY.md`
- Research literature context: `docs/RESEARCH_CONTEXT.md`

## Main empirical finding

The best portfolio-control stack is not invariant across universe breadth:

- **Top-100:** 21-day momentum screen + equal weighting.
- **Top-250:** 63-day low-volatility screen + equal weighting.
- **Top-500:** 63-day volatility-adjusted momentum + 21-day rank-and-hold weighting.

The learned routing layer is informative because it learns preferences over interpretable arms, but in the finite-sample study it does not dominate a simple trailing-window router. This is kept as a first-class result: additional optimization complexity must beat transparent baselines.

## Reproduction status

- Presentation is static HTML/CSS/JavaScript and can be served with `python3 -m http.server 8080`.
- Main Python package uses `uv` from `findings/2026-04-23-context-aware-grand-study/`.
- Raw cached market-data files are intentionally not published; compact metrics, code, figures, and report artifacts are kept.

## Next research steps

1. Rebuild point-in-time universe membership to reduce survivorship bias.
2. Increase independent walk-forward episodes for routing evaluation.
3. Add richer liquidity and market-impact cost models.
4. Evaluate whether universe descriptors can condition the router better than trailing-window selection.
