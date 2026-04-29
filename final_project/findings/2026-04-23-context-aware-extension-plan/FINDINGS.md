# Findings: `2026-04-23-context-aware-extension-plan`

**Run date:** 2026-04-23  
**Author:** Ashesh Kaji  
**Type:** Literature-informed research extension memo

---

## 1. Objective

Extend the repository's compiled-study format into a more modern, context-aware empirical research program that is aligned with recent portfolio-optimization literature and with the project's strongest original ideas:

1. control exists at multiple tiers in portfolio allocation,
2. asset-universe choice materially changes controller performance,
3. a higher-level classifier or router may be able to choose the right controller for the right universe and regime,
4. stronger claims require larger universes, stricter out-of-sample protocols, richer robustness checks, and clearer negative-result reporting.

This run does not introduce a new backtest. It is a research-planning increment that translates current repository evidence plus recent literature into the next experimental program.

---

## 2. Starting point in the repository

The repository already supports four real strengths:

1. **Constrained portfolio control is mature.** Convex allocation with turnover/risk penalties has been studied repeatedly and is the strongest empirical engine in the current work.
2. **Universe dependence is real.** Several recent runs already show that concentrated growth baskets, sector baskets, diversified equity baskets, commodity baskets, and defensive baskets prefer different controller families.
3. **Context-sensitive adaptivity is promising but not yet decisive.** Regime-adaptive and meta-learning layers often improve turnover or robustness, but they do not yet win consistently across all samples.
4. **Strategic interaction is conceptually valuable but empirically incomplete.** Crowding-aware penalties and toy best-response models point in a useful direction, but the historical interaction-aware evidence remains partial.

The next step should therefore be vertical deepening, not additional breadth without structure.

---

## 3. Literature-informed takeaways

Recent literature points toward six upgrades that distinguish a modern portfolio study from a broad exploratory project.

### 3.1 Asset-specific context matters

Recent regime-forecasting work emphasizes that regime information should not only be market-wide. Asset-specific regime forecasts can materially improve dynamic allocation when embedded inside a classical optimizer. This is especially relevant for mixed universes where one asset may be trending while another is in a defensive or mean-reverting state.

### 3.2 Large universes and direct cross-sectional modeling matter

Recent end-to-end portfolio-construction work studies thousands of tradable names rather than only small curated baskets. Even if the final course project cannot fully match that scale, the direction is clear: larger and regularly reconstituted universes are necessary to make claims about universe alignment rather than only basket idiosyncrasies.

### 3.3 Stock selection and weighting should be separated analytically

Several modern studies use a two-stage pipeline:
- first select or screen the active universe,
- then optimize weights under risk and cost constraints.

That separation is directly useful here because it maps onto the user's original idea that there are distinct tiers of control in the allocation problem.

### 3.4 Hierarchical or contextual routing is becoming a serious design pattern

Recent hierarchical RL, contextual bandit, and mixture-of-experts studies all point in the same direction: use context to decide which sub-policy, expert, or controller should act. In this project, the natural analogue is a top-layer router that selects among controller families conditioned on universe descriptors and regime descriptors.

### 3.5 Cost realism, walk-forward evaluation, and negative results are now central

Strong modern empirical studies do not rely on a single headline Sharpe table. They emphasize:
- strict train/validation/test or walk-forward splits,
- transaction-cost and turnover sensitivity,
- inference and confidence intervals,
- ablations,
- explicit reporting of when complex methods fail.

### 3.6 Explainability is a real competitive advantage

Recent stock-selection and regime studies increasingly track feature importance, regime labels, or routing decisions over time. This project should exploit that strength rather than compete only on black-box return claims.

---

## 4. Research thesis for the next program

The best extension of the current project is:

**Portfolio allocation should be modeled as a layered control problem in which the appropriate controller depends on the structural properties of the asset universe and on the current market context.**

That thesis is stronger than a generic "many strategies were compared" narrative because it organizes the research around a falsifiable question:

**Which controller tier is appropriate for which universe structure, and can a higher-level context model learn that mapping?**

---

## 5. Proposed hierarchy of control

The next study program should make the tiers of control explicit.

### Tier 0: Universe definition
- which assets are even eligible at time t,
- liquidity and tradability filters,
- optional screening/ranking layer.

### Tier 1: Signal construction
- momentum,
- reversal,
- volatility/risk,
- cross-sectional or macro-sensitive features,
- asset-specific regime labels.

### Tier 2: Weighting controller
- equal-weight,
- rank-and-hold,
- volatility-scaled,
- convex mean-variance-turnover,
- hierarchical risk allocation.

### Tier 3: Context adaptation
- regime-conditioned blending,
- online expert weighting,
- context-conditioned hyperparameters,
- dynamic turnover/risk budgets.

### Tier 4: Meta-routing / controller selection
- choose which controller family should govern the current universe,
- potentially choose the active sub-universe as well,
- contextual bandit / classifier / router over controller families.

### Tier 5: Strategic interaction
- crowding-aware penalties,
- best-response approximations,
- stress replay and reflexive fragility.

This hierarchy directly addresses the previously unattended idea that control is present at multiple distinct layers of the portfolio problem.

---

## 6. Proposed experimental program

The next phase should be organized as four linked studies rather than one monolithic omnibus run.

### Study A — Large-Universe Controller Benchmark

**Question:** Do the controller-family conclusions survive when the asset universe is much larger and more systematically defined?

**Scope:**
- 3 to 5 universes, each with transparent reconstitution rules
- target sizes such as top-100, top-250, and top-500 liquid US equities, plus sector ETF and multi-asset robustness universes
- walk-forward evaluation over a longer sample where available

**Purpose:**
- upgrade the current basket-based evidence into scalable universe-level evidence
- separate small-basket luck from genuine structural effects

### Study B — Universe Alignment and Descriptor Study

**Question:** Which measurable properties of a universe predict which controller family will work best?

**Core descriptors:**
- concentration
- average pairwise correlation
- rolling dispersion
- trend persistence / autocorrelation
- volatility state
- turnover burden
- liquidity / depth proxy
- regime-switch frequency

**Purpose:**
- move from descriptive results ("controller X won here") to explanatory results ("controller X wins when the universe exhibits these structural properties")

### Study C — Context-Aware Meta-Controller

**Question:** Can a higher-level classifier or contextual bandit learn to choose the right controller for the right universe/regime?

**Action space:** choose among a small family of already-vetted controllers rather than outputting raw weights directly.

**Why this matters:**
- it respects the project's system-optimization framing,
- it is more implementable and interpretable than replacing everything with a monolithic deep policy,
- it directly operationalizes the user's idea of a meta layer of classifiers using optimization/problem understanding.

### Study D — Historical Strategic Layer

**Question:** Does an interaction-aware extension improve robustness when crowding pressure is proxied in historical data?

**Scope:**
- keep this narrower than the main controller benchmark,
- focus on one or two universes where crowding is economically plausible,
- compare: base convex controller vs crowding-penalized controller vs contextual router with crowding feature.

**Purpose:**
- preserve the proposal's strategic-interaction motivation without letting it dominate the entire empirical program.

---

## 7. Concrete universe design increment

The compiled study format should be extended from curated baskets to a tiered universe stack.

### Primary universes
1. **Liquid US Equity 100** — top 100 names by rolling dollar volume and data completeness.
2. **Liquid US Equity 250** — broader, still tractable.
3. **Liquid US Equity 500** — closest to a serious cross-sectional equity universe.
4. **Sector ETF universe** — interpretable macro/sector structure.
5. **Multi-asset universe** — equity, duration, credit, commodities, real estate, gold, possibly international equity.

### Optional derived universes
6. **Cluster-defined subuniverses** — correlation clusters or style clusters extracted from the large equity universe.
7. **Regime-sensitive subuniverses** — high-trend, low-trend, high-volatility, low-liquidity subsets.
8. **Screened active universes** — top-N selected by a stock-screening layer before optimization.

### Reconstitution rules
- fixed monthly or quarterly membership updates,
- liquidity filters using rolling ADV/price availability,
- no lookahead in membership,
- survivorship-safe construction where feasible.

---

## 8. Concrete controller increment

The next benchmark should keep the controller set broad enough to be interesting but narrow enough to support ablation and statistical discipline.

### Recommended controller families
1. equal-weight / passive diversification
2. rank-based momentum
3. reversal / mean-reversion scoring
4. convex mean-variance-turnover
5. hierarchical risk-based allocation
6. regime-adaptive blend
7. online expert-weighted controller
8. contextual meta-controller choosing among 2–7

### Recommended strategic extension set
- crowding-aware convex controller
- reflexive stress overlay
- optional best-response-inspired penalty update

### Recommended RL scope
RL should no longer be treated as a single monolithic direct allocator unless it is competitive. The more promising role is:
- a **meta-controller** over controller families,
- or a **hierarchical high-level selector** over sub-universes and controller types.

This gives RL a cleaner and more defensible role than asking it to beat every structured model directly from scratch.

---

## 9. Required empirical discipline upgrade

A modern extension must tighten the protocol.

### Sample design
- strict walk-forward or expanding-window evaluation
- separate tuning from final evaluation
- subperiod reporting: pre-stress / stress / post-stress where available
- large-universe studies should not rely only on 2020–2025

### Baselines
- market benchmark
- equal weight
- inverse volatility / risk parity
- simple top-k momentum
- simple mean-reversion
- classical shrinkage mean-variance with turnover penalty

### Ablations
- regime layer on/off
- expert weighting on/off
- universe-screening layer on/off
- cost-aware optimization vs after-the-fact cost deduction
- routing features on/off
- crowding term on/off

### Frictions and realism
- multiple cost levels (0 / 5 / 10 / 25 / 50 bps)
- turnover reporting
- concentration reporting
- average names held
- optional liquidity-scaled penalties for large universes

### Statistical reporting
- bootstrap confidence intervals for Sharpe / drawdown-sensitive metrics where feasible
- factor-style alpha tables where appropriate
- model-comparison tests where possible
- explicit negative-result section

---

## 10. Practical outputs the paper should aim to deliver

The extended paper should not only rank methods. It should produce practical allocation rules such as:

1. **When to prefer simple signal-following controllers**
   - concentrated, persistent leadership universes
2. **When to prefer convex diversification-aware controllers**
   - heterogeneous universes with meaningful covariance structure
3. **When adaptation helps**
   - high regime-switch frequency, unstable factor leadership, elevated volatility transitions
4. **When strategic/crowding penalties help**
   - fragile, crowded, liquidation-sensitive segments
5. **When a meta-controller is justified**
   - no single controller dominates and descriptors are predictive of winner shifts

These are the kinds of practical conclusions that can survive beyond a one-off backtest table.

---

## 11. Recommended next implementation order

### Phase 1: Scale the data and universe layer
- add transparent large-universe constructors,
- store universe membership snapshots,
- add liquidity and reconstitution metadata.

### Phase 2: Unify controller benchmarking
- run the current controller families on all universes under a common protocol,
- produce a single scoreboard with costs, turnover, and subperiods.

### Phase 3: Compute universe descriptors
- generate per-universe and rolling descriptor tables,
- analyze which descriptors predict controller winners.

### Phase 4: Add the meta-controller
- first as a supervised classifier over controller winners,
- then as a contextual bandit or hierarchical policy if useful.

### Phase 5: Add the strategic layer selectively
- use only where the crowding mechanism is economically plausible,
- do not let it overwhelm the main controller-by-universe thesis.

---

## 12. Main decision

The project should not expand into an unrestricted survey of every modern portfolio idea. The disciplined extension is:

**compiled study -> larger universes -> universe descriptors -> controller routing -> selective strategic layer**

That sequence preserves the current paper perspective while making the research deeper, more modern, more explanatory, and more practically useful.

---

## 13. Limitations

- This run is a literature-and-repository synthesis, not a fresh empirical result.
- Some recent papers are early-stage, preprint-based, or application-specific; they should guide experimental design more than headline performance claims.
- Scaling to very large stock universes may require data-engineering compromises depending on available local data pipelines.

---

## 14. Reproduction

This memo was assembled from the repository status files, compiled-study findings, and recent public research sources. The most relevant literature and the proposed study matrix are recorded in the CSV artifacts under `metrics/`.

---

## 15. Artifacts

- `metrics/literature_map.csv`
- `metrics/proposed_study_matrix.csv`
- `metrics/universe_descriptor_schema.csv`
- `README.md`
