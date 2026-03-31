# MacroGym

**Controlled nonlinear macroeconomic benchmark for counterfactual evaluation of world models.**

MacroGym provides exact ground-truth counterfactuals for evaluating whether a macroeconomic world model correctly captures the causal effect of policy shocks. It is the first controlled benchmark specifically designed for this task.

## Why MacroGym?

| Benchmark | Ground-truth CF | Nonlinear | Impartial |
|---|---|---|---|
| DSGE model | ✗ (assumption-contaminated) | ✓ | ✗ |
| Agent-based | ✗ (noisy simulation) | ✓ | ✓ |
| Linear VAR | ✓ (exact) | ✗ | ✗ |
| **MacroGym** | **✓ (exact)** | **✓** | **✓** |

DSGE counterfactuals contaminate the benchmark with their own assumptions — a model that mimics DSGE assumptions passes trivially. Agent-based counterfactuals are noisy and not analytically tractable. MacroGym provides exact causal effects from a fully specified nonlinear DGP.

## Quick start

```python
from macrogym import MacroEconomy

# Create economy (nonlinearity=0: linear VAR, nonlinearity=1: maximally nonlinear)
env = MacroEconomy(nonlinearity=0.5, seed=42)

# Simulate 500 months
trajectory = env.simulate(T=500)

# Exact ground-truth counterfactual: -2σ shock to real activity at month 350
result = env.counterfactual(
    shock_time=350,
    shock_factor=0,    # F0 = real activity
    shock_size=-2.0,   # -2 standard deviations
    horizon=24
)

print(result.causal_effect[:6])  # causal effect for first 6 months

# Evaluate your model
scores = env.evaluate(
    model_counterfactual=my_model_cf,   # (24, 5)
    model_baseline=my_model_base,       # (24, 5)
    true_result=result
)
print(f"Direction accuracy: {scores['direction_accuracy']:.3f}")
print(f"RMSE of causal effect: {scores['rmse_effect']:.4f}")
```

## Installation

```bash
cd macrogym
uv sync
```

## Factor structure

The default economy has 5 factors mirroring standard macro decompositions:

| Factor | Interpretation | Key transmission |
|---|---|---|
| F0 | Real activity (output gap) | Drives inflation via Phillips curve |
| F1 | Inflation | Responds to activity and monetary policy |
| F2 | Monetary policy stance | Taylor rule response to inflation |
| F3 | Financial conditions | Amplifies downturns (accelerator) |
| F4 | External sector | Exogenous driver of activity |

## Nonlinearity

The `nonlinearity` parameter controls the degree of regime-switching:

- `nonlinearity=0.0`: Pure linear VAR. TVP-VAR achieves perfect counterfactuals.
- `nonlinearity=0.5`: Moderate asymmetry. Recessions propagate differently from expansions.
- `nonlinearity=1.0`: Full nonlinearity. Only flexible models can recover the true causal effect.

This lets you benchmark models across a spectrum of difficulty.

## Counterfactual methods

```python
# Exact re-simulation (default — uses stored noise path)
result = env.counterfactual(..., method="resimulation")

# Analytical linear IRF (local approximation, exact for nonlinearity=0)
result = env.counterfactual(..., method="analytical")

# Monte Carlo average (N=200 seeds, converges to expected CF)
result = env.counterfactual(..., method="montecarlo")
```

## Running the benchmark

```python
from macrogym import run_benchmark, print_benchmark_table

results = run_benchmark(
    models={"TVP-VAR": tvpvar_fn, "JEPA-Flow": jepaflow_fn},
    env=MacroEconomy(nonlinearity=0.5),
    n_trajectories=50,
    horizon=24
)
print_benchmark_table(results)
```

## Tests

```bash
uv run pytest tests/ -v
```

## Citation

```bibtex
@software{macrogym2026,
  title   = {MacroGym: Controlled Nonlinear Macroeconomic Benchmark
             for Counterfactual Evaluation},
  author  = {Garcia, Andrew},
  year    = {2026},
  url     = {https://github.com/andrewgarcia/macrogym}
}
```
