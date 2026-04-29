"""
Toy Dynamic Programming: 2-asset, 3-period portfolio problem
=============================================================

Solves the Bellman equation explicitly for a discretized 2-asset portfolio
problem to demonstrate dynamic programming in a portfolio context.

State: (wealth, w_prev) where w_prev ∈ {0, 0.25, 0.5, 0.75, 1.0}
Action: w ∈ {0, 0.25, 0.5, 0.75, 1.0}
Reward: expected return − λ * risk − κ * |w − w_prev|
"""
import numpy as np

# ── Parameters ──
mu = np.array([0.08, 0.12])      # annual expected returns
sigma = np.array([0.15, 0.25])   # annual volatilities
rho = 0.3                         # correlation
lam = 1.0                         # risk aversion
kappa = 0.005                     # turnover cost per unit |dw|
T = 3                             # periods
rf = 0.03                         # risk-free (unused)

# Covariance matrix
cov = np.array([
    [sigma[0]**2, rho * sigma[0] * sigma[1]],
    [rho * sigma[0] * sigma[1], sigma[1]**2]
])

# Discretized portfolio weights
W = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
n_W = len(W)

# ── One-step reward ──
def reward(w, w_prev):
    """Expected return minus risk penalty minus turnover cost."""
    ret = w * mu[0] + (1 - w) * mu[1]
    risk = lam * (w**2 * cov[0,0] + (1-w)**2 * cov[1,1] + 2*w*(1-w)*cov[0,1])
    cost = kappa * abs(w - w_prev)
    return ret - risk - cost

# Precompute reward matrix: R[w_prev_idx, w_idx]
R = np.zeros((n_W, n_W))
for i, wp in enumerate(W):
    for j, w in enumerate(W):
        R[i, j] = reward(w, wp)

# ── Value iteration ──
V = np.zeros((T + 1, n_W))   # V[t, w_prev_idx]
policy = np.zeros((T, n_W), dtype=int)

for t in reversed(range(T)):
    for i in range(n_W):
        Q = R[i, :] + V[t + 1, :]   # Q-values for all actions
        policy[t, i] = np.argmax(Q)
        V[t, i] = Q[policy[t, i]]

# ── Results ──
print("=" * 70)
print("TOY DP: 2-Asset, 3-Period Portfolio Problem")
print("=" * 70)
print(f"\nParameters: μ={mu}, σ={sigma}, ρ={rho}, λ={lam}, κ={kappa}")
print(f"Discretized weights: W = {W}")

for t in range(T):
    print(f"\n── Period t={t} ──")
    print(f"  V[t](w_prev): {V[t]}")
    print(f"  Optimal w given w_prev:")
    for i, wp in enumerate(W):
        print(f"    w_prev = {wp:.2f}  →  w* = {W[policy[t,i]]:.2f}  (V = {V[t,i]:.4f})")

print(f"\n── Terminal t={T} ──")
print(f"  V[T] = {V[T]}  (all zero — no future rewards)")

# ── Interpretation ──
print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("""
1. The optimal policy changes with w_prev due to the turnover penalty κ.
   When κ > 0, the policy is 'sticky' — it prefers staying near w_prev
   unless the risk-return benefit of moving is large enough.

2. As t → 0 (more periods remaining), the value function increases because
   there are more opportunities to earn returns. The optimal policy may
   also change because early moves compound over more periods.

3. The discretized DP directly solves the Bellman recursion from the paper's
   Equation (8): V_t(x_t) = max{g(x_t,a_t) + δ E[V_{t+1}(x_{t+1})]}.

4. In a live system, this DP would be solved repeatedly with updated
   estimates of μ and Σ, making it a proper sequential optimization.
""")
