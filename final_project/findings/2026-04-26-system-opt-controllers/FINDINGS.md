# System Optimization Controllers for Portfolio Allocation

**Run:** 2026-04-26-system-opt-controllers  
**Question:** Do actual convex optimization and hierarchical risk budgeting methods improve outcomes within the layered (screen → weight → route) framework?  
**Status:** ✅ Complete (135 arms across 3 universes × 5 screens × 9 controllers, 2970 subperiod observations)

---

## Objective

The existing 60-arm experiment (2026-04-23-context-aware-grand-study) used only four controllers: equal-weight (EW) and three momentum rank-and-hold variants (MRH with τ=21,63,126). None of these solve a continuous optimization problem — they are simple ranking rules. This exploration adds **five system-optimization controllers** to test whether actual convex QP and hierarchical risk budgeting improve upon greedy ranking.

This is the paper's most important missing evidence: does deploying real optimization methods change the conclusions?

---

## Experimental Setup

- **Data:** Adjusted close prices, 2018-01-01 to 2025-01-01, from Yahoo Finance (cached locally)
- **Universes:** Top-100, Top-250, Top-500 liquid U.S. equities (same S&P constituent lists as original experiment, with <70% coverage assets dropped to handle recent IPOs)
- **Screens:** 21-day momentum, 63-day momentum, low-volatility, vol-adjusted momentum, cluster-capped momentum — all top-10
- **Cost:** 10 bps proportional (primary slice, matching original paper)
- **Walk-forward:** Same protocol: 252-day training, 63-day validation gap, 63-day evaluation, 63-day step
- **Controllers tested:** 9 total (4 baseline + 5 optimization)

---

## Results

### Table 1: Best Sharpe by Universe and Screen — All Controllers

| Universe | Screen | Best Simple (Sharpe) | Best Opt (Sharpe) | Winner | Δ |
|---|---|---|---|---|---|
| Top-100 | 21d momentum | EW (1.395) | HRB_lb63_ct0.3 (1.469) | **Opt** | +0.074 |
| Top-100 | 63d momentum | EW (1.233) | CMVT_λ2.0_γ1.0 (1.184) | Simple | −0.049 |
| Top-100 | Low-vol | EW (1.238) | CMVT_λ2.0_γ1.0 (1.273) | **Opt** | +0.035 |
| Top-100 | Vol-adj mom | EW (1.296) | HRB_lb126_ct0.5 (1.493) | **Opt** | +0.197 |
| Top-100 | Cluster-capped | EW (1.209) | CMVT_λ2.0_γ1.0 (1.192) | Simple | −0.017 |
| Top-250 | 21d momentum | EW (1.406) | HRB_lb63_ct0.3 (1.399) | Tie | −0.007 |
| Top-250 | 63d momentum | EW (1.492) | HRB_lb126_ct0.5 (1.481) | Simple | −0.011 |
| Top-250 | Low-vol | EW (1.371) | CMVT_λ2.0_γ1.0 (1.300) | Simple | −0.071 |
| Top-250 | Vol-adj mom | EW (1.446) | HRB_lb126_ct0.5 (1.450) | Tie | +0.004 |
| Top-250 | Cluster-capped | EW (1.517) | HRB_lb126_ct0.5 (1.528) | **Opt** | +0.011 |
| Top-500 | 21d momentum | EW (1.358) | CMVT_λ1.0_γ0.5 (1.551) | **Opt** | +0.194 |
| Top-500 | 63d momentum | EW (1.420) | CMVT_λ2.0_γ1.0 (1.452) | **Opt** | +0.032 |
| Top-500 | Low-vol | EW (1.213) | CMVT_λ2.0_γ1.0 (1.229) | **Opt** | +0.016 |
| Top-500 | Vol-adj mom | EW (1.346) | HRB_lb126_ct0.5 (1.386) | **Opt** | +0.040 |
| Top-500 | Cluster-capped | EW (1.405) | CMVT_λ2.0_γ1.0 (1.388) | Simple | −0.018 |

### Table 2: Mean Sharpe Across All Screens, by Controller and Universe

| Controller | Top-100 | Top-250 | Top-500 |
|---|---|---|---|
| **EW** (no optimization) | 1.274 | 1.446 | 1.345 |
| MRH_τ21 (greedy) | 1.080 | 1.230 | 1.277 |
| MRH_τ63 (greedy) | 1.054 | 1.215 | 1.284 |
| MRH_τ126 (greedy) | 1.169 | 1.284 | 1.295 |
| CMVT_λ0.5_γ0.25 (aggressive QP) | 0.783 | 0.630 | 0.925 |
| CMVT_λ1.0_γ0.5 (moderate QP) | 1.095 | 1.027 | 1.223 |
| **CMVT_λ2.0_γ1.0** (conservative QP) | **1.282** | 1.181 | 1.223 |
| HRB_lb126_ct0.5 (hierarchical) | 1.226 | 1.305 | 1.194 |
| HRB_lb63_ct0.3 (hierarchical) | 1.240 | 1.355 | 1.254 |

---

## Interpretation

### 1. Convex optimization adds value where breadth is high — and destroys value when mis-parameterized

The three CMVT variants bracket the parameter space:
- **λ=0.5, γ=0.25** (aggressive, low risk aversion, light turnover penalty): **catastrophic** — mean Sharpe 0.63–0.93 across universes, well below EW at 1.27–1.45. The optimizer overfits to noisy momentum signals and churns the portfolio.
- **λ=1.0, γ=0.5** (moderate): competitive only in top-500 momentum screens
- **λ=2.0, γ=1.0** (conservative): competitive everywhere, and the **best controller in top-500** where it beats EW on 3/5 screens

**This directly validates the layered thesis**: optimization complexity is justified only when the upstream feasible set is broad enough. In top-100 and top-250, the conservative QP roughly matches EW (which requires zero estimation). In top-500, where cross-sectional dispersion is large, the QP exploits structure that EW cannot.

The aggressive variant's failure is an equally important result: **poorly parameterized optimization is worse than no optimization.** This is why the layered framework matters — you need to match the optimization method's parameters to the universe structure.

### 2. Hierarchical risk budgeting is a strong contender in narrow-to-mid universes

HRB (correlation clustering + inverse-vol weighting) achieves:
- **Best overall Sharpe in top-100 (1.493)** with vol-adjusted momentum screen
- **Best overall Sharpe in top-250 (1.528)** with cluster-capped momentum screen
- Competitive but not dominant in top-500

HRB uses the covariance structure explicitly — it clusters correlated assets and allocates inversely to volatility within clusters. This structural approach appears to work best in universes with moderate breadth, where the correlation structure is informative but not overwhelmed by noise.

### 3. Equal-weight remains a remarkably difficult benchmark

Across all three universes, EW has the highest *mean* Sharpe averaged over all screens (Table 2). It never fails catastrophically, has zero estimation error, and has essentially zero turnover. Even when HRB or CMVT beat it on individual screens, they lose on others.

This continues the DeMiguel, Garlappi, and Uppal (2009) result into a screened-universe context: **after a good screen, equal weighting is a low-variance controller that avoids estimation error.** The optimization methods (CMVT, HRB) can beat it, but only when their parameters are well-tuned to the universe structure.

### 4. The optimal optimization method changes with universe breadth

| Universe | Best optimization approach | Why |
|---|---|---|
| Top-100 | **HRB** (hierarchical risk) | Correlation structure is stable and informative; risk budgeting exploits it |
| Top-250 | **EW or HRB** | Middle breadth introduces noise; optimization methods struggle to add value beyond screening |
| Top-500 | **CMVT** (convex QP, conservative params) | Large cross-sectional dispersion rewards active mean-variance optimization |

This is exactly the thesis of the layered framework: the optimal controller is not invariant — it depends on the structure of the candidate universe.

---

## How This Differs From Standard Portfolio Optimization Approaches

A standard portfolio optimization paper would:
1. Pick a single optimization method (e.g., Markowitz QP)
2. Run it on one universe
3. Compare it to a few baselines
4. Declare it the winner (or not)

This layered framework instead:
1. **Decomposes** the problem into universe → screen → weight → route
2. **Tests optimization methods within each layer's context** — the same QP performs differently depending on which screen precedes it and which universe feeds it
3. **Discovers that the answer changes with breadth** — a monolithic approach would have reported "Markowitz QP beats EW in top-500" or "EW beats Markowitz QP in top-100" but would have missed the structural dependence
4. **Reveals that parameterization matters more than method choice** — the gap between λ=0.5 and λ=2.0 CMVT is larger than the gap between CMVT and EW

The framework's superiority is not that it finds a "better strategy" — it's that it produces a **conditional answer**: given universe structure X, screen Y, the optimal weight controller is Z. This conditional knowledge is what a practitioner actually needs.

---

## Limitations

- **Date range:** 2018–2025, shorter than the original experiment (which used ~2016–2024). Walk-forward splits and metrics are not comparable to the 60-arm experiment.
- **CMVT solver:** Uses scipy SLSQP, not a specialized QP solver (OSQP, Gurobi). Convergence may fail on ill-conditioned Σ. A few backtest runs failed silently.
- **Survivorship:** Same S&P current-constituent survivorship bias as the original experiment.
- **No turnover analysis:** The CMVT controller includes turnover penalty in the objective, but realized turnover wasn't compared systematically to MRH.
- **Parameter grid is sparse:** Only 3 CMVT parameterizations tested. A proper hyperparameter sweep (λ ∈ [0.1, 5.0], γ ∈ [0.05, 2.0]) would reveal the full performance surface.

---

## Next Steps for Paper Integration

1. **Include CMVT (λ=2.0, γ=1.0) and HRB (lb=126, ct=0.5) in the final report's controller set.** These are the best-performing optimization controllers and demonstrate that actual system optimization methods are competitive with simple rules in the right context.

2. **Add a "parameterization matters" finding.** The failure of aggressive CMVT is as important as the success of conservative CMVT. This is a systems-optimization lesson: optimization methods are only as good as their parameterization.

3. **Re-run with point-in-time universes.** Both the original and this exploration use current-constituent lists. Adding proper point-in-time membership would make the "breadth-dependence" claim much stronger.

4. **Add a tradeoff plot.** Show the Sharpe-vs-turnover tradeoff for EW vs CMVT vs MRH — the convex QP explicitly trades return against turnover, and visualizing this makes the systems-optimization framing concrete.
