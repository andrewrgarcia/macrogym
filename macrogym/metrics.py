"""
macrogym/metrics.py
────────────────────
Standardised evaluation metrics for counterfactual benchmarking.

Three classes of metrics:

1. Direction metrics — does the model get the sign right?
   Most important for economic interpretation.
   A model that gets the direction wrong is worse than useless.

2. Magnitude metrics — how close is the effect size?
   Secondary to direction but important for policy quantification.

3. Trajectory metrics — does the dynamic path match?
   Tests whether the model captures the timing and persistence
   of the shock effect, not just the average.
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """
    Full benchmark result for one model on one economy configuration.
    """
    model_name:        str
    nonlinearity:      float
    direction_accuracy: float      # fraction correct signs
    sign_error_rate:   float      # fraction wrong signs
    rmse_effect:       float      # RMSE of causal effect
    mae_effect:        float      # MAE of causal effect
    corr_effect:       float      # correlation with true effect
    rmse_by_horizon:   np.ndarray  # (horizon,) RMSE at each step
    dir_by_horizon:    np.ndarray  # (horizon,) direction accuracy at each step
    per_factor:        Dict[str, float]  # per-factor metrics


def compute_full_metrics(model_effect: np.ndarray,
                          true_effect:  np.ndarray,
                          model_name:   str,
                          nonlinearity: float) -> BenchmarkResult:
    """
    Compute the full suite of benchmark metrics.

    model_effect : (horizon, k)  model's (CF - baseline)
    true_effect  : (horizon, k)  ground-truth (CF - baseline)
    """
    H, k = true_effect.shape

    # ── Direction ─────────────────────────────────────────────────────────────
    correct = np.sign(model_effect) == np.sign(true_effect)
    dir_acc = float(correct.mean())

    # Direction by horizon
    dir_by_h = correct.mean(axis=1)   # (H,)

    # ── Magnitude ─────────────────────────────────────────────────────────────
    diff         = model_effect - true_effect
    rmse_effect  = float(np.sqrt((diff**2).mean()))
    mae_effect   = float(np.abs(diff).mean())
    rmse_by_h    = np.sqrt((diff**2).mean(axis=1))   # (H,)

    # ── Correlation ───────────────────────────────────────────────────────────
    me = model_effect.flatten()
    te = true_effect.flatten()
    if te.std() > 1e-10 and me.std() > 1e-10:
        corr = float(np.corrcoef(me, te)[0, 1])
    else:
        corr = 0.0

    # ── Per factor ────────────────────────────────────────────────────────────
    per_factor = {}
    for i in range(k):
        c_i = np.sign(model_effect[:, i]) == np.sign(true_effect[:, i])
        per_factor[f"dir_F{i}"]  = float(c_i.mean())
        per_factor[f"rmse_F{i}"] = float(np.sqrt(((model_effect[:, i] -
                                                    true_effect[:, i])**2).mean()))

    return BenchmarkResult(
        model_name         = model_name,
        nonlinearity       = nonlinearity,
        direction_accuracy = dir_acc,
        sign_error_rate    = 1.0 - dir_acc,
        rmse_effect        = rmse_effect,
        mae_effect         = mae_effect,
        corr_effect        = corr,
        rmse_by_horizon    = rmse_by_h,
        dir_by_horizon     = dir_by_h,
        per_factor         = per_factor,
    )


def run_benchmark(models:          Dict,
                  env,
                  n_trajectories:  int   = 50,
                  T:               int   = 500,
                  shock_time_frac: float = 0.7,
                  shock_factor:    int   = 0,
                  shock_size:      float = -2.0,
                  horizon:         int   = 24,
                  seed:            int   = 0) -> Dict[str, BenchmarkResult]:
    """
    Run the full counterfactual benchmark across multiple trajectories.

    models : dict of {name: callable}
        Each callable receives (train_data, test_data) and returns
        a function cf_fn(F_seed, shock_factor, shock_size, horizon)
        that returns (baseline, counterfactual) arrays of shape (horizon, k).

    Returns dict of {model_name: BenchmarkResult}.
    """
    rng = np.random.default_rng(seed)
    all_effects = {name: [] for name in models}
    all_true    = []

    for trial in range(n_trajectories):
        trial_seed = int(rng.integers(0, 2**31))
        env.seed = trial_seed
        env._rng = np.random.default_rng(trial_seed)

        traj = env.simulate(T)
        shock_time = int(shock_time_frac * T)
        train = traj[:shock_time]

        true_result = env.counterfactual(
            trajectory   = traj,
            shock_time   = shock_time,
            shock_factor = shock_factor,
            shock_size   = shock_size,
            horizon      = horizon,
            method       = "resimulation",
        )
        all_true.append(true_result.causal_effect)

        for name, model_fn in models.items():
            cf_fn = model_fn(train, traj)
            base, cf = cf_fn(
                traj[shock_time],
                shock_factor,
                shock_size,
                horizon,
                env.factor_stds,
            )
            all_effects[name].append(cf - base)

    # Aggregate
    results = {}
    true_stack = np.array(all_true)   # (n_traj, H, k)

    for name in models:
        model_stack = np.array(all_effects[name])   # (n_traj, H, k)
        mean_model = model_stack.mean(axis=0)
        mean_true  = true_stack.mean(axis=0)
        results[name] = compute_full_metrics(
            mean_model, mean_true, name, env.nonlinearity)

    return results


def print_benchmark_table(results: Dict[str, BenchmarkResult]) -> None:
    """Print a formatted benchmark comparison table."""
    print(f"\n{'─'*70}")
    print(f"  {'Model':<20} {'Dir.Acc':>8} {'SignErr':>8} "
          f"{'RMSE':>8} {'Corr':>8}")
    print(f"{'─'*70}")
    for name, r in sorted(results.items(),
                           key=lambda x: -x[1].direction_accuracy):
        print(f"  {name:<20} {r.direction_accuracy:>8.3f} "
              f"{r.sign_error_rate:>8.3f} {r.rmse_effect:>8.4f} "
              f"{r.corr_effect:>8.3f}")
    print(f"{'─'*70}")
