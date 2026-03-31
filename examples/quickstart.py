"""
examples/quickstart.py
───────────────────────
MacroGym quickstart — demonstrates the full API.

Shows:
  1. Creating an economy with controlled nonlinearity
  2. Simulating a trajectory
  3. Computing exact ground-truth counterfactuals
  4. Evaluating a naive model against the ground truth
  5. Comparing linear vs nonlinear economies

Run:
  cd /home/andrew/children/macrogym
  uv run examples/quickstart.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from macrogym import MacroEconomy, print_benchmark_table, compute_full_metrics

# ── 1. Create economy ─────────────────────────────────────────────────────────
print("\n[1] Creating MacroEconomy")
env = MacroEconomy(nonlinearity=0.5, seed=42)
print(f"  {env}")

# ── 2. Simulate trajectory ────────────────────────────────────────────────────
print("\n[2] Simulating 500-month trajectory")
traj = env.simulate(T=500)
print(f"  Shape: {traj.shape}")
print(f"  Factor stds: {env.factor_stds.round(3)}")

# Train/test split
train, test = env.get_train_test_split(traj, train_frac=0.7)
print(f"  Train: {len(train)} months  Test: {len(test)} months")

# ── 3. Ground-truth counterfactual ────────────────────────────────────────────
print("\n[3] Computing ground-truth counterfactuals")
shock_time = 350   # apply shock at month 350

# Method 1: exact re-simulation (default)
result_resim = env.counterfactual(
    shock_time=shock_time, shock_factor=0, shock_size=-2.0,
    horizon=24, method="resimulation")
print(f"  Re-simulation: causal effect F0 at h=6: "
      f"{result_resim.causal_effect[5, 0]:+.4f}")

# Method 2: analytical (local linear approximation)
result_anal = env.counterfactual(
    shock_time=shock_time, shock_factor=0, shock_size=-2.0,
    horizon=24, method="analytical")
print(f"  Analytical:    causal effect F0 at h=6: "
      f"{result_anal.causal_effect[5, 0]:+.4f}")

# Method 3: Monte Carlo (N=100 seeds)
result_mc = env.counterfactual(
    shock_time=shock_time, shock_factor=0, shock_size=-2.0,
    horizon=24, method="montecarlo")
print(f"  Monte Carlo:   causal effect F0 at h=6: "
      f"{result_mc.causal_effect[5, 0]:+.4f}")

print(f"\n  True causal effect (first 6 months, F0 and F1):")
print(f"  {'h':>4}  {'F0 (activity)':>15}  {'F1 (inflation)':>15}")
for h in range(6):
    print(f"  {h+1:>4}  {result_resim.causal_effect[h,0]:>+15.4f}  "
          f"{result_resim.causal_effect[h,1]:>+15.4f}")

# ── 4. Evaluate a naive model ─────────────────────────────────────────────────
print("\n[4] Evaluating models against ground truth")

# Naive model 1: random walk (predicts no effect)
rw_baseline = result_resim.baseline.copy()
rw_cf       = result_resim.baseline.copy()   # predicts same as baseline
scores_rw   = env.evaluate(rw_cf, rw_baseline, result_resim)
print(f"\n  Random Walk (predicts zero effect):")
print(f"    Direction accuracy: {scores_rw['direction_accuracy']:.3f}")
print(f"    RMSE effect:        {scores_rw['rmse_effect']:.4f}")

# Naive model 2: linear VAR (analytical IRF — best linear predictor)
var_effect = result_anal.causal_effect
var_cf     = result_resim.baseline + var_effect
scores_var = env.evaluate(var_cf, result_resim.baseline, result_resim)
print(f"\n  Linear VAR (analytical IRF):")
print(f"    Direction accuracy: {scores_var['direction_accuracy']:.3f}")
print(f"    RMSE effect:        {scores_var['rmse_effect']:.4f}")

# Naive model 3: perfect model (upper bound)
scores_perfect = env.evaluate(
    result_resim.counterfactual,
    result_resim.baseline,
    result_resim)
print(f"\n  Perfect model (upper bound):")
print(f"    Direction accuracy: {scores_perfect['direction_accuracy']:.3f}")
print(f"    RMSE effect:        {scores_perfect['rmse_effect']:.4f}")

# ── 5. Compare linear vs nonlinear economies ──────────────────────────────────
print("\n[5] Comparing linear vs nonlinear economies")
for nl in [0.0, 0.5, 1.0]:
    env_nl  = MacroEconomy(nonlinearity=nl, seed=42)
    traj_nl = env_nl.simulate(T=500)
    res_nl  = env_nl.counterfactual(shock_time=350, shock_factor=0,
                                     shock_size=-2.0, horizon=12)
    # How well does the linear analytical IRF approximate the true CF?
    res_anal = env_nl.counterfactual(shock_time=350, shock_factor=0,
                                      shock_size=-2.0, horizon=12,
                                      method="analytical")
    scores = env_nl.evaluate(
        res_anal.baseline + res_anal.causal_effect,
        res_anal.baseline,
        res_nl)
    print(f"  nonlinearity={nl:.1f}: VAR dir_acc={scores['direction_accuracy']:.3f}  "
          f"RMSE={scores['rmse_effect']:.4f}")

# ── 6. Plot ───────────────────────────────────────────────────────────────────
print("\n[6] Saving figure")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
horizons  = np.arange(1, 25)

for i, (ax, fname) in enumerate(zip(axes.flatten()[:5],
                                    env.factor_names)):
    ax.plot(horizons, result_resim.causal_effect[:, i],
            color="royalblue", lw=2, label="Re-simulation (exact)")
    ax.plot(horizons, result_anal.causal_effect[:, i],
            color="darkorange", lw=1.5, linestyle="--", label="Analytical (linear)")
    ax.plot(horizons, result_mc.causal_effect[:, i],
            color="green", lw=1.5, linestyle=":", label="Monte Carlo (N=200)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title(f"{fname}", fontsize=9)
    ax.set_xlabel("Horizon (months)")
    if i == 0: ax.legend(fontsize=7)

# Nonlinearity comparison on F0
ax = axes[1, 2]
for nl, col in [(0.0, "green"), (0.5, "orange"), (1.0, "red")]:
    env_nl = MacroEconomy(nonlinearity=nl, seed=42)
    env_nl.simulate(T=500)
    res = env_nl.counterfactual(shock_time=350, shock_factor=0,
                                 shock_size=-2.0, horizon=24)
    ax.plot(horizons, res.causal_effect[:, 0],
            color=col, lw=2, label=f"nl={nl}")
ax.axhline(0, color="gray", lw=0.5)
ax.set_title("F0 effect by nonlinearity", fontsize=9)
ax.set_xlabel("Horizon")
ax.legend(fontsize=7)

fig.suptitle(
    "MacroGym: Ground-truth counterfactual causal effects\n"
    "−2σ shock to F0 (Real Activity), method comparison",
    fontsize=11)
fig.tight_layout()
fig.savefig("examples/quickstart_output.png", dpi=150, bbox_inches="tight")
print("  → examples/quickstart_output.png")
print("\nDone.")
