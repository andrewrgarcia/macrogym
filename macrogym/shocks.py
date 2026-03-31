"""
macrogym/shocks.py
───────────────────
Shock injection and exact counterfactual computation.

Two approaches to ground-truth counterfactuals
───────────────────────────────────────────────

Approach 1 — Exact re-simulation (used for nonlinear dynamics):
    Given a baseline trajectory with stored noise draws {ε_t},
    re-simulate from the shock point with the same noise sequence
    but a different initial condition F_shock_t = F_t + δ_F.
    
    The difference baseline - counterfactual is the exact causal
    effect of the shock, holding the noise path fixed.
    
    This is the gold standard for nonlinear systems where no
    analytical solution exists.

Approach 2 — Analytical impulse response (used for linear component):
    For the linear part of the dynamics (nonlinearity=0), the exact
    impulse response to a shock δ_F at time t is:
    
        IRF(h) = A^h · δ_F    for h = 1, 2, ..., H
    
    where A is the transition matrix. This is analytically exact
    and has zero noise — useful as a validation check.

Approach 3 — Monte Carlo averaging (robustness check):
    Run N re-simulations with different noise seeds and average.
    As N → ∞ this converges to the true expected counterfactual.
    Used to verify that approach 1 results are not noise-sensitive.
"""

import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class CounterfactualResult:
    """
    Container for a counterfactual experiment result.

    All arrays are (horizon, k_factors).
    """
    baseline:        np.ndarray    # baseline factor trajectory
    counterfactual:  np.ndarray    # counterfactual factor trajectory
    causal_effect:   np.ndarray    # counterfactual - baseline (exact)
    shock_factor:    int           # which factor was shocked
    shock_size:      float         # size of shock in std devs
    shock_time:      int           # time index of shock
    horizon:         int           # forecast horizon
    method:          str           # "resimulation" | "analytical" | "montecarlo"
    n_seeds:         int = 1       # number of seeds (for montecarlo)


def inject_shock(F_t: np.ndarray,
                 shock_factor: int,
                 shock_size: float,
                 factor_stds: np.ndarray) -> np.ndarray:
    """
    Apply a shock to a factor vector.

    shock_size is in units of that factor's standard deviation
    (computed on the training window).

    Returns F_shocked = F_t + δ_F where δ_F[shock_factor] = shock_size · std.
    """
    delta_F = np.zeros_like(F_t)
    delta_F[shock_factor] = shock_size * factor_stds[shock_factor]
    return F_t + delta_F, delta_F


def counterfactual_resimulation(
    trajectory:   np.ndarray,
    noises:       np.ndarray,
    transition,
    shock_time:   int,
    shock_factor: int,
    shock_size:   float,
    factor_stds:  np.ndarray,
    horizon:      int,
) -> CounterfactualResult:
    """
    Exact counterfactual via re-simulation with fixed noise.

    trajectory : (T, k)  baseline factor trajectory
    noises     : (T, k)  noise draws used to generate trajectory
    transition : NonlinearTransition instance
    shock_time : int     time index when shock is applied (0-indexed)
    shock_factor: int    which factor to shock
    shock_size : float   size in standard deviations (negative = contractionary)
    factor_stds: (k,)    per-factor standard deviations from training window
    horizon    : int     how many steps to simulate after shock

    Returns CounterfactualResult with exact causal effects.
    """
    T, k = trajectory.shape
    assert shock_time + horizon <= T, \
        f"shock_time ({shock_time}) + horizon ({horizon}) exceeds trajectory length ({T})"

    # Shocked initial state
    F_shocked, delta_F = inject_shock(
        trajectory[shock_time], shock_factor, shock_size, factor_stds)

    # Re-simulate from shock_time using the same noise draws
    baseline_window = trajectory[shock_time : shock_time + horizon]  # (H, k)
    cf_window = np.zeros((horizon, k))
    F_curr = F_shocked.copy()

    for h in range(horizon):
        noise = noises[shock_time + h]
        F_curr = transition.step_deterministic(F_curr, noise)
        cf_window[h] = F_curr

    causal_effect = cf_window - baseline_window

    return CounterfactualResult(
        baseline       = baseline_window.copy(),
        counterfactual = cf_window,
        causal_effect  = causal_effect,
        shock_factor   = shock_factor,
        shock_size     = shock_size,
        shock_time     = shock_time,
        horizon        = horizon,
        method         = "resimulation",
        n_seeds        = 1,
    )


def counterfactual_analytical(
    F_shock_time: np.ndarray,
    transition,
    shock_factor:  int,
    shock_size:    float,
    factor_stds:   np.ndarray,
    horizon:       int,
) -> np.ndarray:
    """
    Analytical impulse response function.

    For a linear VAR F_{t+1} = A·F_t + ε_t, the exact causal effect
    of a shock δ_F applied at time t is:

        IRF(h) = A^h · δ_F    for h = 1, 2, ..., H

    For the nonlinear model, this is a local approximation around F_shock_time
    using the state-dependent A(F_t) evaluated at the shock point.

    Returns causal_effect : (horizon, k)
    """
    _, delta_F = inject_shock(
        F_shock_time, shock_factor, shock_size, factor_stds)

    # Local linear approximation at shock point
    A = transition.linear_approximation(F_shock_time)

    irf = np.zeros((horizon, len(delta_F)))
    A_power = np.eye(len(delta_F))
    for h in range(horizon):
        A_power = A_power @ A
        irf[h] = A_power @ delta_F

    return irf


def counterfactual_montecarlo(
    trajectory:   np.ndarray,
    noises:       np.ndarray,
    transition,
    shock_time:   int,
    shock_factor: int,
    shock_size:   float,
    factor_stds:  np.ndarray,
    horizon:      int,
    n_seeds:      int = 200,
    base_seed:    int = 0,
) -> CounterfactualResult:
    """
    Monte Carlo counterfactual — average over many noise realisations.

    Rather than fixing the noise path (resimulation), this draws N
    independent noise sequences and averages the causal effect.
    As N → ∞ this converges to E[F_cf_{t+h} - F_base_{t+h}].

    Use this to:
    1. Verify that resimulation results are not noise-path-specific
    2. Provide expected counterfactual when noise path is not fixed
    """
    T, k = trajectory.shape
    F_base_t = trajectory[shock_time]
    F_shock_t, _ = inject_shock(
        F_base_t, shock_factor, shock_size, factor_stds)

    effects = np.zeros((n_seeds, horizon, k))

    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        F_base = F_base_t.copy()
        F_cf   = F_shock_t.copy()
        for h in range(horizon):
            # Same noise draw for both base and cf
            noise_base = rng.standard_normal(k)
            noise_cf   = noise_base.copy()   # identical noise path per seed
            A_base = transition.transition_matrix(F_base)
            A_cf   = transition.transition_matrix(F_cf)
            L_base = np.linalg.cholesky(transition.noise_covariance(F_base))
            L_cf   = np.linalg.cholesky(transition.noise_covariance(F_cf))
            F_base = A_base @ F_base + L_base @ noise_base
            F_cf   = A_cf   @ F_cf   + L_cf   @ noise_cf
            effects[s, h] = F_cf - F_base

    mean_effect  = effects.mean(axis=0)   # (H, k)
    baseline_mc  = trajectory[shock_time : shock_time + horizon]
    cf_mc        = baseline_mc + mean_effect

    return CounterfactualResult(
        baseline       = baseline_mc.copy(),
        counterfactual = cf_mc,
        causal_effect  = mean_effect,
        shock_factor   = shock_factor,
        shock_size     = shock_size,
        shock_time     = shock_time,
        horizon        = horizon,
        method         = "montecarlo",
        n_seeds        = n_seeds,
    )


def evaluate_model_counterfactual(
    model_cf:    np.ndarray,
    model_base:  np.ndarray,
    true_result: CounterfactualResult,
) -> Dict[str, float]:
    """
    Evaluate a model's counterfactual prediction against ground truth.

    model_cf   : (horizon, k)  model's counterfactual trajectory
    model_base : (horizon, k)  model's baseline trajectory
    true_result: CounterfactualResult  ground-truth from MacroGym

    Returns dict of evaluation metrics:
      direction_accuracy : fraction of (h, k) cells with correct sign
      rmse_effect        : RMSE of causal effect (cf - base)
      mae_effect         : MAE of causal effect
      corr_effect        : correlation between model and true effects
      sign_error_rate    : fraction of cells with wrong sign
    """
    model_effect = model_cf - model_base            # (H, k)
    true_effect  = true_result.causal_effect        # (H, k)

    # Direction accuracy — correct sign on each (h, factor) cell
    correct_sign   = np.sign(model_effect) == np.sign(true_effect)
    dir_accuracy   = correct_sign.mean()
    sign_error     = 1.0 - dir_accuracy

    # Magnitude metrics
    diff           = model_effect - true_effect
    rmse_effect    = float(np.sqrt((diff**2).mean()))
    mae_effect     = float(np.abs(diff).mean())

    # Correlation between model and true causal effects
    me_flat  = model_effect.flatten()
    te_flat  = true_effect.flatten()
    if te_flat.std() > 1e-10 and me_flat.std() > 1e-10:
        corr = float(np.corrcoef(me_flat, te_flat)[0, 1])
    else:
        corr = 0.0

    # Per-factor direction accuracy
    per_factor = {
        f"dir_acc_F{i}": float(correct_sign[:, i].mean())
        for i in range(true_effect.shape[1])
    }

    return {
        "direction_accuracy": float(dir_accuracy),
        "sign_error_rate":    float(sign_error),
        "rmse_effect":        rmse_effect,
        "mae_effect":         mae_effect,
        "corr_effect":        corr,
        **per_factor,
    }
