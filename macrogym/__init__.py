"""
MacroGym — Controlled nonlinear macroeconomic benchmark for
counterfactual evaluation of world models.

Quick start
───────────
    from macrogym import MacroEconomy

    env  = MacroEconomy(nonlinearity=0.5, seed=42)
    traj = env.simulate(T=500)

    result = env.counterfactual(
        shock_time=350, shock_factor=0, shock_size=-2.0, horizon=24)

    scores = env.evaluate(
        model_counterfactual=my_model_cf,
        model_baseline=my_model_base,
        true_result=result)

    print(f"Direction accuracy: {scores['direction_accuracy']:.3f}")
"""

from macrogym.economy   import MacroEconomy
from macrogym.shocks    import CounterfactualResult, evaluate_model_counterfactual
from macrogym.metrics   import BenchmarkResult, compute_full_metrics, run_benchmark, print_benchmark_table
from macrogym.transitions import NonlinearTransition

__all__ = [
    "MacroEconomy",
    "CounterfactualResult",
    "BenchmarkResult",
    "NonlinearTransition",
    "evaluate_model_counterfactual",
    "compute_full_metrics",
    "run_benchmark",
    "print_benchmark_table",
]

__version__ = "0.1.0"
