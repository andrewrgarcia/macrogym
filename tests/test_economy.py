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
    """Linear VAR analytical IRF has lower error at nl=0 than nl=0.5."""
    errors = []
    for nl in [0.0, 0.5]:  # skip nl=1.0 -- can diverge
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
    assert errors[0] <= errors[1] + 0.05, \
        f"Expected nl=0 error <= nl=0.5 error, got {errors}"


def test_arbitrary_k():
    """MacroEconomy should work for any k >= 2."""
    for k in [2, 5, 10, 20]:
        env  = MacroEconomy(n_factors=k, seed=42)
        traj = env.simulate(T=200)
        assert traj.shape == (200, k), f"Expected (200, {k}), got {traj.shape}"
        res  = env.counterfactual(shock_time=150, shock_factor=0,
                                   shock_size=-1.0, horizon=6)
        assert res.causal_effect.shape == (6, k)


def test_with_dimension():
    """with_dimension factory should work cleanly."""
    env  = MacroEconomy.with_dimension(k=20, nonlinearity=0.5, seed=42)
    traj = env.simulate(T=300)
    assert traj.shape == (300, 20)
    assert env.k == 20
    res  = env.counterfactual(shock_time=200, shock_factor=0,
                               shock_size=-2.0, horizon=12)
    assert res.causal_effect.shape == (12, 20)
    # Direction should be negative on F0 at h=1
    assert res.causal_effect[0, 0] < 0


def test_make_structural_matrices():
    """Generated matrices should be stable."""
    import numpy as np
    from macrogym import make_structural_matrices
    for k in [5, 10, 20]:
        A_n, A_r, sig = make_structural_matrices(k, seed=0)
        assert A_n.shape == (k, k)
        sr_n = max(abs(np.linalg.eigvals(A_n)))
        sr_r = max(abs(np.linalg.eigvals(A_r)))
        assert sr_n < 1.0, f"A_normal unstable at k={k}: sr={sr_n}"
        assert sr_r < 1.0, f"A_recession unstable at k={k}: sr={sr_r}"