# Meta-controller research design: contextual bandits, MoE, and regime-aware routing

## Why the current layer is weak
The current `context_study.meta_controller.MetaController` is a static supervised selector from rolling universe descriptors to a winner label. With only a few universe-level winner samples, it is vulnerable to label noise, class imbalance, and overfitting. Treat the current logistic/nearest-centroid selector as a diagnostic baseline, not the final routing layer.

## Literature signals to import

1. Contextual bandit / expert routing for online portfolio selection
- Zhu, Zheng, Wang, Liang, Zhang, "Online Portfolio Selection with Cardinality Constraint and Transaction Costs based on Contextual Bandit," IJCAI 2020. URL: https://www.ijcai.org/Proceedings/2020/646 ; PDF: https://ijcai.org/proceedings/2020/0646.pdf
  - Uses side information and an Exp4-style contextual bandit to select asset subsets, then allocates wealth. Important for this repo: choose among controller arms online, include transaction-cost-aware reward, and report regret versus best fixed/expert policy.
- Beygelzimer, Langford, Li, Reyzin, Schapire, "Contextual Bandit Algorithms with Supervised Learning Guarantees," AISTATS 2011. URL: https://proceedings.mlr.press/v15/beygelzimer11a.html
  - Exp4.P competes with large policy classes using partial-feedback rewards. Important for this repo: supervised predictors can be candidate policies, while bandit weighting handles online uncertainty.
- Li et al., "A Contextual-Bandit Approach to Personalized News Article Recommendation," WWW 2010 / LinUCB. URL: https://arxiv.org/abs/1003.0146
  - Classic LinUCB framing: reward is linear in context-action features plus confidence bonus. Important for this repo: start with transparent LinUCB/linear Thompson before neural routers.
- Contextual bandit software reference with BootstrappedUCB/TS, LinUCB/LinTS. URL: https://contextual-bandits.readthedocs.io/
  - Practical option if adding online contextual bandit baselines.

2. Mixture-of-experts and mixture-of-policies portfolio allocation
- Wei et al., "Deep reinforcement learning portfolio model based on mixture of experts," Applied Intelligence, 2025. DOI/URL: https://doi.org/10.1007/s10489-025-06242-6
  - Uses spatio-temporal attention, a router selecting experts, and market-index risk to adjust invested capital. Important for this repo: the router can output soft controller weights, not only hard labels; include market-risk/capital-throttle features.
- Shen, Liu, Chen, "Meta-Learning the Optimal Mixture of Strategies for Online Portfolio Selection," arXiv:2505.03659. URL: https://arxiv.org/abs/2505.03659 ; HTML: https://arxiv.org/html/2505.03659v1
  - Learns mixture weights over candidate portfolio policies rather than direct asset weights; decomposes long history into short tasks. Important for this repo: train on rolling tasks/windows and route over controller policies.
- Lim, Zohren, Roberts, "DeepUnifiedMom: Unified Time-series Momentum Portfolio Construction via Multi-Task Learning with Multi-Gate Mixture of Experts," arXiv:2406.08742. URL: https://arxiv.org/abs/2406.08742
  - Multi-gate MoE over time-series momentum horizons/assets. Important for this repo: multi-horizon momentum experts should be separate arms with learned gates.
- Jacobs et al., "Adaptive Mixtures of Local Experts," Neural Computation, 1991. URL: https://direct.mit.edu/neco/article-abstract/3/1/79/5560/Adaptive-Mixtures-of-Local-Experts
  - Foundational gating-network formulation.

3. Regime-aware dynamic allocation and regime labels
- Shu, Yu, Mulvey, "Dynamic Asset Allocation with Asset-Specific Regime Forecasts," arXiv:2406.09578. URL: https://arxiv.org/abs/2406.09578
  - Uses statistical jump models for asset-specific bull/bear labels, supervised GBDT forecasts, and regime-conditioned return/risk inputs to Markowitz. Important for this repo: add asset/factor-specific regime probabilities, not just global universe descriptors.
- Shu, Mulvey, "Dynamic Factor Allocation Leveraging Regime-Switching Signals," arXiv:2410.14841. URL: https://arxiv.org/abs/2410.14841
  - Sparse jump model labels factor bull/bear states; regime inferences feed Black-Litterman and improve IR. Important for this repo: use stable, interpretable jump/HMM regime states as auxiliary features and as evaluation strata.
- RegimeFolio, "A Regime Aware ML System for Sectoral Portfolio Optimization in Dynamic Markets," arXiv:2510.14986. URL: https://arxiv.org/abs/2510.14986
  - Volatility-regime segmentation, sector-specific ensembles, and regime-aware allocation. Important for this repo: regime-specific experts by sector/universe class.

4. Hierarchical / meta-controller RL
- Chen, Li, Wang, "MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management," arXiv:2508.01173. URL: https://arxiv.org/html/2508.01173v2
  - Heterogeneous agent ensemble with conservative/neutral/aggressive agents and a high-level meta-adaptive controller assigning weights from market state. Important for this repo: controller arms should be behaviorally diverse risk profiles, not only algorithm families.
- SAMP-HDRL, "Segmented Allocation with Momentum-Adjusted Utility for Multi-agent Portfolio Management via Hierarchical Deep Reinforcement Learning," arXiv. URL: https://arxiv.org/html/2512.22895v1
  - Dynamic asset grouping, upper-level global agent, lower-level group allocators, utility capital allocation. Important for this repo: separate universe-clustering/segmentation from intra-segment allocation; evaluate turbulent regimes separately.

## Recommended design for this repo

### Controller arms
Keep existing candidates, but expose them as arms with daily/monthly realized rewards:
- equal_weight / minimum-variance / risk-parity baseline
- momentum rank-and-hold, with 21/63/126-day variants as separate arms
- mean-reversion / Bollinger arm
- convex optimizer / meta-signal convex arm
- regime-adaptive attractiveness arm
- defensive/cash-throttled arm or low-vol arm
- optional RL pilot arm only after leakage-safe training

Prefer soft routing weights over hard winner labels when possible. Hard labels are useful for diagnostics; production routing should permit top-k blends.

### Features: expand the descriptor vector
Current descriptors to retain:
- avg_pairwise_corr, first_pc_share, cross_sectional_dispersion, trend_persistence, vol_level, vol_of_vol, liquidity_proxy, regime_switch_rate.

Add market-state features:
- equal-weight trailing returns: 1m, 3m, 6m, 12m.
- equal-weight realized vol: 21/63/126d and vol z-score versus 2y history.
- drawdown from 252d high; days since high.
- downside semivol, skew, kurtosis, 5% rolling VaR/CVaR.
- VIX / VIX term structure if available; SPY/QQQ benchmark trend and drawdown.
- rate/dollar stress proxies if available: TLT return, UUP return, credit proxy LQD/HYG spread return.

Add cross-sectional opportunity features:
- dispersion by return quantile, not only mean std.
- momentum dispersion: std of 63d returns across assets.
- percentage of assets above 50/200d moving average.
- top-minus-bottom momentum spread.
- sector concentration / effective number of names if sector metadata exists.
- covariance condition number and eigenvalue entropy.
- average beta to benchmark and beta dispersion.

Add regime-belief features:
- HMM/GMM/Sparse Jump Model state probabilities on equal-weight returns and on benchmark returns.
- state age/duration and transition probability.
- asset-specific bullish share: fraction of names in bullish regime.
- factor/regime breadth: fraction of sectors/factors risk-on.

Add controller-health features:
- trailing arm rewards over 1/3/6 rebalances.
- trailing arm Sharpe/Sortino, drawdown, hit rate.
- trailing arm turnover and cost drag.
- disagreement/entropy among candidate controller weights.
- last selected controller and switching cost from current holdings.

### Labels and rewards
Avoid one global winner per universe. Construct many samples by rolling rebalance date x horizon.

For each date t, controller arm a, and horizon H in {21, 63} trading days, compute label/reward from the portfolio weights that would have been chosen using only data available at t:
- net_forward_return_a(t,H) = realized return net of estimated transaction costs.
- risk_adjusted_reward_a = net_return - lambda_vol * realized_vol - lambda_dd * max_drawdown - lambda_turnover * turnover.
- optional utility = log(1 + net_return) - lambda * variance - cost.

Label options:
1. hard argmax label: winner(t) = argmax_a risk_adjusted_reward_a(t,H).
2. top-k label: all arms within epsilon of best; train multilabel or ranking model.
3. soft target: softmax(reward_a / temperature); train MoE gate by cross-entropy/KL.
4. contextual bandit reward: observe reward for chosen arm in live simulation; for offline backtest, use full-information panel to compare algorithms.

Use purged labels: feature window ends at t; reward window is (t, t+H]; purge H days between train/test folds.

### Model ladder
1. Baseline supervised classifier currently present: logistic regression / nearest centroid.
2. Ranking baseline: pairwise logistic or LightGBM/XGBoost ranker if dependency allowed; target is arm ordering not just winner.
3. Soft MoE gate: multinomial logistic/gradient boosting outputs p(arm|context); final weights = sum_a p_a w_a.
4. Contextual bandit: LinUCB/LinTS over action-context features; arms are controllers; reward is net utility. Add epsilon floor and switching penalty.
5. EXP4/EXP4.P style expert-advice layer: experts are simple policies such as "choose momentum in high trend", "choose defensive in high vol", learned classifier, and best trailing Sharpe. This is attractive with small data because it competes with policies rather than fitting a high-capacity gate.

### Evaluation protocol
Use walk-forward, purged, anchored splits:
- Train: first N months; validate: next 12 months; test: next 12 months; roll forward.
- Purge/embargo: at least H days after train before validation/test.
- No random K-fold on time series.

Report portfolio metrics:
- annual return, annual vol, Sharpe, Sortino, max drawdown, Calmar, Omega.
- turnover, estimated bps cost drag, average gross/net exposure, concentration/HHI.
- downside capture in stress windows.

Report routing metrics:
- regret versus oracle best-per-date and best fixed controller.
- hit rate: selected arm equals ex-post winner; top-2/top-3 hit rate.
- normalized regret: (oracle - chosen)/(oracle - best_fixed).
- switch frequency and transition matrix between controllers.
- gate entropy/effective number of active experts.
- per-regime performance: low/mid/high vol, trend/reversal, high/low correlation, high/low dispersion.
- ablations: descriptors only; descriptors + market state; + regime beliefs; + controller-health features.

### Data augmentation for few samples
- Use date-level samples rather than universe-level winners.
- Generate samples across liquid_us_equity_100/250/500 plus sector, factor, macro, commodity, and bootstrapped sub-universes.
- Bootstrap sub-universes with constraints: keep 30-100 names, sector balanced, liquidity filtered.
- Train on earlier universes/time windows; test on held-out universe class and later period.
- Use simpler models until samples exceed ~10x number of features per class.

### Implementation target
Add a new module, e.g. `src/context_study/routing.py`, with:
- `build_arm_reward_panel(candidate_weight_paths, returns, costs, horizon)`
- `build_meta_dataset(descriptors, regime_features, arm_reward_panel, horizon, label_mode)`
- `SoftGatingMetaController` returning probabilities over arms
- `LinUCBMetaRouter` or `EXP4Router` for online walk-forward routing
- `evaluate_router(predicted_arm_or_probs, arm_reward_panel, realized_weight_paths)`

The minimal high-value next step is not RL; it is a leakage-safe full-information reward panel plus purged walk-forward comparison of: best fixed, trailing-best, current logistic selector, soft-gated logistic, and LinUCB/EXP4.
