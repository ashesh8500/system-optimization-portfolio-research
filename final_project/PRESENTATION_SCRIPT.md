# Rough presentation rehearsal script

Target length: 7–10 minutes. Use the interactive website in deck mode and advance one section at a time.

## Slide 1 — Thesis

Today I am presenting my final project, **Portfolio Allocation as Layered System Optimization**. The main idea is that portfolio allocation should not be treated as just one static weighting problem. In practice, the system makes several linked decisions: what universe is available, which assets are screened into the active set, how weights are assigned, how much turnover is allowed, and whether a higher-level rule should route among strategies.

The final experiment studies those decisions as a layered control system. I evaluate 60 walk-forward arms: three candidate-universe breadths, five screen rules, and four weighting controllers, with an adaptive routing layer on top.

## Slide 2 — Study framing

The project begins with the classical portfolio-allocation question, but it does not stop at choosing weights. The central framing is that a portfolio system also chooses an opportunity set, a screen objective, a rebalancing controller, a cost tolerance, and sometimes a routing policy.

I will first explain why the Markowitz formulation is only the starting point. Then I will show how the system was decomposed into layers, how the walk-forward experiment was built, and what the results say about where optimization actually helped.

## Slide 3 — Research question

The research question is: **where should optimization be applied?**

A standard portfolio optimizer assumes the investment universe is already fixed and then solves for weights. But that skips several choices that can dominate the result. If the universe is too narrow, too noisy, or structurally mismatched to the signal, a more sophisticated weighting rule may just optimize the wrong feasible set.

So the project separates three core decisions: the opportunity set, the screen objective, and the weighting controller. The hypothesis is that the best controller depends on the upstream universe and screen.

## Slide 4 — Research flow

The project started from a proposal about signal-based portfolio allocation as constrained optimization, with possible game-theoretic extensions. As the work developed, the most important step was to first build a rigorous single-system control substrate.

The research flow was:

1. formulate allocation as a sequential control problem;
2. define universe, screen, and controller layers;
3. build walk-forward ledgers so every arm is recorded, not just the selected winner;
4. scale the study to Top-100, Top-250, and Top-500 candidate universes;
5. add diagnostics: ablations, cost sensitivity, winner-gap checks, and a policy-gradient router.

That progression matters because it turns the project from a strategy search into a mechanism study.

## Slide 5 — System decomposition

This slide shows the core architecture. At each rebalance time, the system observes market information and current holdings. It chooses a universe breadth, applies a screen, chooses a weighting controller, executes a portfolio, and observes net reward after turnover cost.

There are two different optimization types here. The weighting layer is continuous: choose weights on a simplex while balancing expected return, risk, and turnover. The screening layer is discrete: choose a top-K subset according to a score. This is why the full problem is mixed discrete-continuous and sequential, not just a single quadratic program.

## Slide 6 — Risk and cost intuition

The first visual is the familiar Markowitz tradeoff. As risk aversion lambda increases, the optimum moves toward lower risk on the efficient frontier.

The second visual adds the dynamic-control view. The previous portfolio is part of the state. A high transaction-cost coefficient kappa pulls the new portfolio closer to the old portfolio. That is why I treat turnover as control effort rather than just a fee. A high-churn controller must earn enough extra return to pay for its own movement.

## Slide 7 — Screening intuition

This slide is the bridge to the main result. Screening is not preprocessing. It is a discrete optimization layer.

If I select by momentum, I choose names with high recent returns. If I select by low volatility, I choose stable names. If I select by volatility-adjusted momentum, I look for trend strength relative to noise. Each score creates a different active set, so each downstream weighting controller is solving a different portfolio problem.

## Slide 8 — Experiment matrix

The main empirical matrix has three universe scopes, five screen rules, and four weighting controllers. That gives 60 primary arms at the 10 basis point cost slice.

Every arm is evaluated chronologically. The study stores train and test dates, selected symbols, turnover, return, volatility, Sharpe ratio, and drawdown. The important design choice is that I keep the full ledger. This lets me study the surface by universe, screen, and controller rather than only reporting the best backtest.

## Slide 9 — Full result surface

This heatmap is the central empirical object. The rows are screens, the columns are controllers, and the tabs switch universe breadth. The outlined cell is the selected winner.

The key observation is that the winning region moves. In the Top-100 universe, the best stack is momentum screening plus equal weighting. In the Top-250 universe, low-volatility screening wins. In the Top-500 universe, volatility-adjusted momentum plus short-horizon rank-and-hold wins.

That means the optimal stack is not invariant. The best weighting rule depends on the feasible set created upstream.

## Slide 10 — Winning stacks

Here are the three headline results.

For Top-100, the best result is 21-day momentum screening plus equal weighting, with Sharpe around 1.47. The interpretation is that the narrow universe is already concentrated enough that screening does most of the useful selection.

For Top-250, the best result is low-volatility screening plus equal weighting, with Sharpe around 0.86. This is the most interesting middle case because it prevents a simple “bigger is always better” story. The middle universe appears broad enough to add noise, but not broad enough for momentum ranking to reliably dominate.

For Top-500, the best result is volatility-adjusted momentum plus 21-day rank-and-hold, with Sharpe around 1.76. In the broadest universe, there is enough cross-sectional dispersion for a second active ranking layer to pay for its turnover.

## Slide 11 — Layer ablations

The ablations ask which layer is doing useful work.

Averaged over screens, equal weighting is strongest in Top-100 and Top-250. That supports the idea that once the active set is chosen, a simple low-estimation controller can be robust.

In Top-500, the 21-day rank-and-hold controller becomes competitive or best. That means extra control effort is not universally good. It is only valuable when the upstream feasible set is broad enough to contain exploitable dispersion.

## Slide 12 — Selection risk

This slide is about caution. Each universe selects a winner from 20 arms, so the winning Sharpe is a selected maximum. I therefore treat the result as mechanism evidence, not as proof of a unique trading rule.

The winner gaps are small to moderate, and bootstrap stability is only about one-third across the scopes. That says the exact winning arm can be fragile, especially with limited walk-forward splits. But the broader mechanism — that the surface changes across universe breadth and layers — is the robust research insight.

## Slide 13 — Cost sensitivity

The cost-sensitivity check asks whether the result is just an artifact of the 10 basis point transaction-cost assumption.

The same winning families persist from 0 to 25 basis points. But the Top-500 active rank-and-hold winner visibly pays turnover cost. This supports the control interpretation: active ranking can be worth it, but only when the signal benefit is large enough to overcome movement cost.

## Slide 14 — Adaptive router

The reinforcement-learning component is placed at the system level. It does not learn raw portfolio weights from scratch. Instead, it chooses among interpretable screen/controller arms.

The learned router improves over train-selected fixed routing in the combined panel, but it does not dominate the trailing-window router. That is an important result rather than a failure. It means a simple transparent online optimizer is a strong baseline in this finite-sample setting. A learned router should only be considered practically useful if it consistently beats that kind of baseline.

## Slide 15 — Limitations

The main limitations are also the next research steps.

First, the universes are approximate rather than fully point-in-time, so survivorship bias remains a threat. Second, the number of independent walk-forward splits is limited, especially for the broader universes. Third, the grid creates model-selection risk because each result is chosen from many arms. Fourth, the cost model is proportional and does not yet include nonlinear market impact or liquidity constraints.

The next version should rebuild point-in-time universe membership, compute structural descriptors for each universe, and test whether a contextual router can beat trailing-window selection with more independent episodes.

## Slide 16 — Research anchors

This slide connects the implementation back to the research literature. Markowitz provides the risk-return baseline. Transaction-cost portfolio optimization explains why turnover belongs directly in the objective. Online portfolio selection motivates evaluating decisions sequentially rather than with hindsight. Reinforcement-learning portfolio work motivates an adaptive policy layer, but the important standard is still comparative: a learned policy has to beat simple online baselines before it is worth the extra complexity.

That is why the contribution here is not just another backtest. The contribution is the layered decomposition and the evidence that the binding layer changes with market breadth.

## Slide 17 — Conclusion

The main takeaway is that portfolio allocation is a layered control problem. Universe construction, screening, weighting, turnover, and routing are all optimization decisions.

The empirical result is not “this one strategy always wins.” It is that the binding layer changes with universe breadth. In narrow universes, screening plus simple equal weighting works best. In the middle universe, risk filtering matters. In the broad universe, active ranking becomes worthwhile.

So the research shifts the question from **which portfolio algorithm is best?** to **which layer should be optimized under the current market structure?** That is the system-optimization contribution of the project.
