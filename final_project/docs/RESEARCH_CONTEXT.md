# Research context and literature anchors

This project treats portfolio allocation as a layered system-optimization problem rather than a single static weight-selection problem.

## Conceptual anchors

- **Mean-variance portfolio choice.** Markowitz's portfolio-selection formulation provides the classical starting point: choose weights by balancing expected return and variance.
- **Turnover and transaction-cost control.** Portfolio optimization with linear/proportional transaction costs motivates treating turnover as a control effort term. In the presentation, this appears as the κ‖w_t − w_{t−1}‖₁ penalty.
- **Online portfolio selection.** Cover's universal-portfolio work motivates evaluating allocation policies sequentially and nonanticipatively rather than only as hindsight-fitted portfolios.
- **Reinforcement-learning portfolio control.** Recent RL portfolio-management work motivates adaptive policies, but this study keeps the learned layer interpretable by routing among screen/controller arms and compares it against fixed, random, trailing-window, and oracle baselines.

## How these map to this study

1. **Universe layer:** Top-100, Top-250, and Top-500 scopes change the feasible opportunity set.
2. **Screen layer:** momentum, low-volatility, volatility-adjusted momentum, and cluster-capped screens implement cardinality-constrained asset selection.
3. **Weighting layer:** equal weight and rank-and-hold controllers solve different approximations to the weight-update problem.
4. **Cost layer:** proportional turnover cost evaluates whether an active controller earns enough to pay for movement.
5. **Routing layer:** policy-gradient routing tests adaptive model selection over interpretable arms, with transparent baselines used as the standard for comparison.

## References used for presentation framing

- Markowitz, H. (1952). *Portfolio Selection*. The Journal of Finance, 7(1), 77–91. DOI: 10.1111/j.1540-6261.1952.tb01525.x.
- Cover, T. M. (1991). *Universal Portfolios*. Mathematical Finance, 1(1), 1–29. DOI: 10.1111/j.1467-9965.1991.tb00002.x.
- Boyd et al. portfolio-optimization work on transaction costs and convex optimization motivates the linear-cost framing used in the website intuition section.
- Recent surveys of reinforcement learning in portfolio management motivate the policy-gradient routing layer while reinforcing the need for strict out-of-sample baselines.

The empirical results in this repository should be read as mechanism evidence from the tested walk-forward design, not as investment advice.
