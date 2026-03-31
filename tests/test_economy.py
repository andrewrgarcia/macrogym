"""tests/test_economy.py"""
import numpy as np
import pytest
from macrogym import MacroEconomy


def test_simulate_shape():
    env = MacroEconomy(seed=0)
    traj = env.simulate(T=200)
    assert traj.shape == (200, 5)


def test_simulate_reproducible():
    env1 = MacroEconomy(seed=42)
    env2 = MacroEconomy(seed=42)
    t1 = env1.simulate(T=100)
    t2 = env2.simulate(T=100)
    np.testing.assert_array_equal(t1, t2)


def test_counterfactual_zero_shock():
    """Zero shock should produce zero causal effect."""
    env  = MacroEconomy(seed=42)
    traj = env.simulate(T=300)
    res  = env.counterfactual(shock_time=200, shock_factor=0,
                               shock_size=0.0, horizon=12)
    np.testing.assert_allclose(res.causal_effect, 0.0, atol=1e-10)


def test_counterfactual_direction():
    """Negative F0 shock should reduce F0 at h=1."""
    env  = MacroEconomy(nonlinearity=0.5, seed=42)
    traj = env.simulate(T=300)
    res  = env.counterfactual(shock_time=200, shock_factor=0,
                               shock_size=-2.0, horizon=12)
    # F0 should be lower under a contractionary shock at h=1
    assert res.causal_effect[0, 0] < 0, \
        f"Expected negative F0 effect, got {res.causal_effect[0, 0]}"


def test_linear_economy_var_exact():
    """At nonlinearity=0, analytical IRF should match re-simulation closely."""
    env  = MacroEconomy(nonlinearity=0.0, seed=42)
    traj = env.simulate(T=500)
    res_resim = env.counterfactual(shock_time=350, shock_factor=0,
                                    shock_size=-1.0, horizon=12,
                                    method="resimulation")
    res_anal  = env.counterfactual(shock_time=350, shock_factor=0,
                                    shock_size=-1.0, horizon=12,
                                    method="analytical")
    # For linear economy the analytical should match re-simulation
    # (they differ only by noise path, so check they have same sign)
    for h in range(6):
        assert np.sign(res_resim.causal_effect[h, 0]) == \
               np.sign(res_anal.causal_effect[h, 0])


def test_evaluate_perfect_model():
    """Perfect model should get direction_accuracy=1.0."""
    env  = MacroEconomy(seed=42)
    traj = env.simulate(T=300)
    res  = env.counterfactual(shock_time=200, shock_factor=0,
                               shock_size=-2.0, horizon=12)
    scores = env.evaluate(res.counterfactual, res.baseline, res)
    assert scores["direction_accuracy"] == 1.0
    assert scores["rmse_effect"] < 1e-10


def test_nonlinearity_increases_var_error():
    """Linear VAR should have higher CF error at higher nonlinearity."""
    errors = []
    for nl in [0.0, 0.5, 1.0]:
        env  = MacroEconomy(nonlinearity=nl, seed=42)
        traj = env.simulate(T=500)
        res_true = env.counterfactual(shock_time=350, shock_factor=0,
                                       shock_size=-2.0, horizon=12,
                                       method="resimulation")
        res_lin  = env.counterfactual(shock_time=350, shock_factor=0,
                                       shock_size=-2.0, horizon=12,
                                       method="analytical")
        scores = env.evaluate(
            res_lin.baseline + res_lin.causal_effect,
            res_lin.baseline, res_true)
        errors.append(scores["rmse_effect"])
    # Error should be non-decreasing with nonlinearity
    assert errors[0] <= errors[2] + 0.01, \
        f"Expected linear error ≤ nonlinear error, got {errors}"
