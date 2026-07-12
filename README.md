# MacroGym

**A macroeconomic simulator with ground-truth counterfactuals for testing causal predictions.**

MacroGym generates synthetic economies where the *true effect* of a policy shock is known exactly. This makes it possible to measure whether a model's counterfactual claims are correct — something that cannot be verified on real data, where the counterfactual outcome is never observed.

Presented at the **XLIV Encuentro de Economistas del BCRP** (Banco Central de Reserva del Perú), July 2026.

---

## Why MacroGym?

| Benchmark | Ground-truth counterfactuals | Nonlinear dynamics | Model-free evaluation |
|---|---|---|---|
| DSGE model | ✗ (answers depend on its own assumptions) | ✓ | ✗ |
| Agent-based | ✗ (effect exists, but not exactly recoverable) | ✓ | ✓ |
| Linear VAR | ✗ (exact only if the world is linear) | ✗ | ✗ |
| **MacroGym** | **✓ (exact re-simulation)** | **✓** | **✓** |

**DSGE** counterfactuals are circular: they grade models against the very class of theory being tested, so a model that mirrors those assumptions can look correct by construction rather than by capturing the true mechanism.

**Agent-based models** do have a true effect, but it cannot be isolated exactly — emergent dynamics can't be re-run with the noise sequence held fixed, so the causal effect is confounded with simulation noise.

**Linear VAR IRFs** are exact, but only under linearity, and become systematically biased when the true system is nonlinear.

MacroGym provides exact causal effects from a fully specified nonlinear data-generating process with known parameters: hold the noise sequence fixed, perturb the state at the shock date, re-integrate. The difference between the two trajectories is the causal effect, exact to machine precision.

---

## Results

Benchmarking the standard toolkit on counterfactual **direction accuracy** (share of correct signs vs. the true effect; oracle = 1.000, coin flip = 0.500), at k = 20 factors and T = 350 training months:

| Model | DA (h = 12, α = 0.5) | Parameters |
|---|---|---|
| **VAR(1)** | **0.931** | 400 |
| MS-VAR | 0.898 | 804 |
| TVP-VAR | 0.883 | ~400 × T |
| MLP | 0.807 | ~21,500 |
| TVAR | 0.802 | 800 |
| Random | 0.496 | — |

**Three findings:**

1. **A large, growing oracle gap.** Even the best model is wrong on ~1 directional claim in 7 at a 24-month horizon. The gap grows with horizon: 0.069 (h = 6) → 0.152 (h = 24).
2. **Parsimony wins.** VAR(1) outperforms every flexible alternative at *every* horizon and *every* nonlinearity level — even though the DGP is genuinely nonlinear. Estimation error in the transition operator compounds through the rollout faster than added flexibility pays off.
3. **This is not an artifact of one calibration.** Regenerating the economy five times (fresh structural matrices, same economic architecture) and rerunning the full grid: VAR(1) ranks first in **all 75 cells** (5 economies × 5 α × 3 horizons, 50 trajectories each). Oracle gap at h = 24, α = 0.5: 0.157 ± 0.062 across economies.

Higher nonlinearity does **not** favor flexible models. At realistic macro sample sizes, the binding constraint is information, not architecture.

---

## Quick start

```python
from macrogym import MacroEconomy, evaluate_model_counterfactual

# Default k=5 economy; or arbitrary dimensionality (k=20 matches large macro panels)
env = MacroEconomy.with_dimension(k=20, nonlinearity=0.5, seed=42)

# Simulate 500 months
trajectory = env.simulate(T=500)

# Exact ground-truth counterfactual: -2σ shock to real activity at month 350
result = env.counterfactual(
    shock_time=350,
    shock_factor=0,    # F0 = real activity
    shock_size=-2.0,   # -2 standard deviations
    horizon=24,
)
print(result.causal_effect[:6])   # exact causal effect, first 6 months

# Score your model's counterfactual against the truth
scores = evaluate_model_counterfactual(
    model_cf=my_model_cf,       # (24, k) your model's counterfactual rollout
    model_base=my_model_base,   # (24, k) your model's baseline rollout
    true_result=result,
)
print(f"Direction accuracy: {scores['direction_accuracy']:.3f}")
print(f"RMSE of causal effect: {scores['rmse_effect']:.4f}")
```

The shock displaces the **state**, never the operator: `F_t0 → F_t0 + δ`. Baseline and counterfactual are then rolled forward with the same transition mechanism and the same noise draws.

---

## Installation

```bash
git clone https://github.com/andrewrgarcia/macrogym
cd macrogym
pip install -e .

# Or with uv
uv sync
```

---

## Factor structure

The default economy has 5 interpretable macro factors:

| Factor | Interpretation | Key transmission |
|---|---|---|
| F0 | Real activity (output gap) | Drives inflation via the Phillips curve |
| F1 | Inflation | Responds to activity; feeds the Taylor rule |
| F2 | Monetary policy stance | Taylor-rule response to inflation |
| F3 | Financial conditions | Amplifies downturns (financial accelerator) |
| F4 | External sector | Exogenous commodity/terms-of-trade driver |

Every non-zero entry of the structural matrix is an economic claim, calibrated to the literature (Phillips slope, Taylor rule, Bernanke–Gertler accelerator); zeros are structural restrictions. `MacroEconomy.with_dimension(k)` extends this to arbitrary k ≥ 2 via `make_structural_matrices`, preserving the labeled architecture and adding sparse couplings.

## Nonlinearity

The `nonlinearity` parameter (α) scales a smooth, state-dependent regime shift: the transition matrix bends toward the recession regime as real activity falls, through a sigmoid gate on F0. It is *not* Markov switching — there is no hidden state.

- `nonlinearity=0.0` — the operator is constant; the economy collapses exactly to a linear VAR(1), and re-simulation reproduces the analytical IRF `Aʰδ` to machine precision.
- `nonlinearity=0.5` — moderate asymmetry: recessions propagate differently from expansions. Most relevant calibration for emerging-market economies.
- `nonlinearity=1.0` — full expansion–recession asymmetry.

> **Note:** raising α does **not** favor flexible models. VAR(1) attains the highest counterfactual direction accuracy at every α tested — see [Results](#results).

## Counterfactual methods

```python
result = env.counterfactual(...)                          # default: exact re-simulation
result = env.counterfactual(..., method="resimulation")   # same; uses the stored noise path
result = env.counterfactual(..., method="analytical")     # linear IRF (exact only at α=0)
```

## Self-validity

The benchmark is checked before it grades anyone:

| Check | Result |
|---|---|
| At α = 0, re-simulation equals the analytical IRF | max error 3.2 × 10⁻¹⁵ |
| Zero shock → zero effect | 0 |
| Re-running the same setup is deterministic | 0 |
| Raising α raises measured nonlinearity | monotone (slope 0.227) |

```bash
uv run pytest tests/ -v
```

---

## Paper pipeline

The empirical results in the paper — the joint FRED + BCRP panel (376 × 10,277), the 20-factor DFM, the factor-grounding analysis, and the Peru counterfactual — come from a separate research pipeline, `macrogym-research`, which reproduces every table and figure end-to-end from raw API access. Available on request; release is planned alongside the working paper.

---

## Citation

If you use MacroGym in your research, please cite both the software and the paper:

```bibtex
@software{macrogym2026,
  title   = {MacroGym: Controlled Nonlinear Macroeconomic Benchmark
             for Counterfactual Evaluation},
  author  = {Garcia, Andrew R.},
  year    = {2026},
  url     = {https://github.com/andrewrgarcia/macrogym}
}

@techreport{garcia2026macrogym,
  title       = {MacroGym: A Controlled Benchmark for Counterfactual
                 Evaluation in Macroeconomics},
  author      = {Garcia, Andrew R. and Vega, Marco},
  institution = {XLIV Encuentro de Economistas del BCRP},
  year        = {2026}
}
```

## License

MIT