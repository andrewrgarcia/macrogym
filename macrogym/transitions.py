"""
macrogym/economy.py
────────────────────
Main MacroEconomy class — the public API for MacroGym.

MacroGym is a controlled nonlinear macroeconomic environment for
benchmarking counterfactual methods. It provides:

  1. Trajectory simulation with known dynamics
  2. Exact ground-truth counterfactuals (via re-simulation or analytical)
  3. Standardised evaluation metrics for model comparison

Why MacroGym instead of DSGE or agent-based models
───────────────────────────────────────────────────
DSGE models encode specific theoretical assumptions (rational expectations,
log-linearisation) that contaminate the benchmark — a model that matches
DSGE assumptions passes the test trivially regardless of its quality.

Agent-based models are realistic but their counterfactuals are not
analytically tractable — you cannot compute the true causal effect,
only simulate it with noise, which makes the validation metric noisy.

MacroGym provides exact ground-truth counterfactuals by design.
The data-generating process is fully known, so the true causal effect
of any shock is computable to machine precision. This makes it the
first controlled benchmark for counterfactual evaluation in macroeconomics.

Factor interpretation (default k=5)
────────────────────────────────────
    F0 — real activity       (output gap / growth)
    F1 — inflation           (price level dynamics)
    F2 — monetary policy     (interest rate gap)
    F3 — financial conditions (credit spreads / risk)
    F4 — external sector     (trade / commodity prices)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Union

from macrogym.transitions import NonlinearTransition, A_NORMAL, A_RECESSION, SIGMA_BASE
from macrogym.shocks import (
    CounterfactualResult,
    counterfactual_resimulation,
    counterfactual_analytical,
    counterfactual_montecarlo,
    evaluate_model_counterfactual,
)


FACTOR_NAMES = ["RealActivity", "Inflation", "MonetaryPolicy",
                "FinancialConditions", "ExternalSector"]


class MacroEconomy:
    """
    Synthetic nonlinear macroeconomy for counterfactual benchmarking.

    Parameters
    ──────────
    n_factors : int
        Number of factors (default 5, matches default structural matrices).
    nonlinearity : float ∈ [0, 1]
        Regime-switching strength.
        0 → linear VAR (TVP-VAR should win)
        1 → fully nonlinear (flexible models needed)
        0.5 → recommended for benchmarking
    sharpness : float
        Speed of regime transition (higher = more abrupt).
    vol_sensitivity : float
        Stochastic volatility sensitivity (0 = homoskedastic).
    seed : int
        Master random seed for reproducibility.

    Example
    ───────
    >>> env = MacroEconomy(nonlinearity=0.5, seed=42)
    >>> traj = env.simulate(T=500)
    >>> result = env.counterfactual(traj, shock_time=300,
    ...                             shock_factor=0, shock_size=-2.0,
    ...                             horizon=24)
    >>> print(result.causal_effect[:6, :2])  # F0, F1 for first 6 months
    """

    def __init__(self,
                 n_factors:       int   = 5,
                 nonlinearity:    float = 0.5,
                 sharpness:       float = 2.0,
                 vol_sensitivity: float = 0.3,
                 seed:            int   = 42):

        assert 0.0 <= nonlinearity <= 1.0, "nonlinearity must be in [0, 1]"
        assert n_factors == 5, \
            "Currently only n_factors=5 is supported (matches default matrices). " \
            "Custom matrices via MacroEconomy.from_matrices() for other sizes."

        self.k               = n_factors
        self.nonlinearity    = nonlinearity
        self.seed            = seed
        self.factor_names    = FACTOR_NAMES[:n_factors]

        self.transition = NonlinearTransition(
            nonlinearity    = nonlinearity,
            sharpness       = sharpness,
            vol_sensitivity = vol_sensitivity,
            A_normal        = A_NORMAL,
            A_recession     = A_RECESSION,
            sigma_base      = SIGMA_BASE,
        )

        self._rng = np.random.default_rng(seed)
        self._last_trajectory = None
        self._last_noises     = None
        self._factor_stds     = None

    @classmethod
    def from_matrices(cls,
                      A_normal:    np.ndarray,
                      A_recession: np.ndarray,
                      sigma_base:  np.ndarray,
                      nonlinearity: float = 0.5,
                      seed: int = 42) -> "MacroEconomy":
        """
        Create a MacroEconomy with custom transition matrices.
        Useful for calibrating to specific economy data.
        """
        k = A_normal.shape[0]
        env = cls.__new__(cls)
        env.k = k
        env.nonlinearity = nonlinearity
        env.seed = seed
        env.factor_names = [f"F{i}" for i in range(k)]
        env.transition = NonlinearTransition(
            nonlinearity=nonlinearity,
            A_normal=A_normal,
            A_recession=A_recession,
            sigma_base=sigma_base,
        )
        env._rng = np.random.default_rng(seed)
        env._last_trajectory = None
        env._last_noises = None
        env._factor_stds = None
        return env

    # ── Simulation ────────────────────────────────────────────────────────────

    def simulate(self, T: int = 500,
                 F0: Optional[np.ndarray] = None,
                 burn_in: int = 100) -> np.ndarray:
        """
        Simulate a trajectory of T months.

        Parameters
        ──────────
        T       : length of trajectory to return (after burn-in)
        F0      : initial state (default: zeros)
        burn_in : number of initial steps to discard for stationarity

        Returns
        ───────
        trajectory : (T, k) factor panel
            Also stored in self._last_trajectory and self._last_noises
            for use in counterfactual() without re-simulating.
        """
        T_total = T + burn_in
        F0 = np.zeros(self.k) if F0 is None else F0.copy()

        trajectory = np.zeros((T_total, self.k))
        noises     = np.zeros((T_total, self.k))
        trajectory[0] = F0

        for t in range(T_total - 1):
            F_next, noise = self.transition.step(trajectory[t], self._rng)
            trajectory[t + 1] = F_next
            noises[t + 1]     = noise

        # Discard burn-in
        trajectory = trajectory[burn_in:]
        noises     = noises[burn_in:]

        # Compute factor stds on first 70% (training window equivalent)
        train_end = int(0.7 * T)
        self._factor_stds = trajectory[:train_end].std(axis=0)
        self._factor_stds[self._factor_stds == 0] = 1.0

        self._last_trajectory = trajectory
        self._last_noises     = noises

        return trajectory

    def simulate_to_dataframe(self, T: int = 500,
                               start_date: str = "1950-01-01") -> pd.DataFrame:
        """Simulate and return as a pandas DataFrame with monthly index."""
        traj = self.simulate(T)
        idx  = pd.date_range(start_date, periods=T, freq="MS")
        return pd.DataFrame(traj, index=idx, columns=self.factor_names)

    # ── Counterfactuals ───────────────────────────────────────────────────────

    def counterfactual(self,
                       trajectory:   Optional[np.ndarray] = None,
                       shock_time:   int = 300,
                       shock_factor: int = 0,
                       shock_size:   float = -2.0,
                       horizon:      int = 24,
                       method:       str = "resimulation") -> CounterfactualResult:
        """
        Compute an exact counterfactual for a shock.

        Parameters
        ──────────
        trajectory   : (T, k) factor panel. If None, uses last simulated.
        shock_time   : time index (0-based) when shock is applied.
        shock_factor : which factor to shock (0=RealActivity default).
        shock_size   : shock magnitude in std devs (negative=contractionary).
        horizon      : how many months to simulate after shock.
        method       : "resimulation" | "analytical" | "montecarlo"
            resimulation — exact, uses stored noise path (default)
            analytical   — local linear approximation (fast, exact for linear)
            montecarlo   — average over N noise seeds (N=200 default)

        Returns
        ───────
        CounterfactualResult with exact causal effects.
        """
        if trajectory is None:
            assert self._last_trajectory is not None, \
                "Call simulate() first or pass a trajectory."
            trajectory = self._last_trajectory
            noises     = self._last_noises
        else:
            # If external trajectory passed, generate noises by re-simulation
            # from stored seed — caller must use self._last_noises if available
            noises = self._last_noises if self._last_noises is not None else \
                     np.zeros_like(trajectory)

        assert self._factor_stds is not None, \
            "Factor stds not computed. Call simulate() first."

        if method == "resimulation":
            return counterfactual_resimulation(
                trajectory   = trajectory,
                noises       = noises,
                transition   = self.transition,
                shock_time   = shock_time,
                shock_factor = shock_factor,
                shock_size   = shock_size,
                factor_stds  = self._factor_stds,
                horizon      = horizon,
            )
        elif method == "analytical":
            F_shock = trajectory[shock_time]
            irf = counterfactual_analytical(
                F_shock_time = F_shock,
                transition   = self.transition,
                shock_factor = shock_factor,
                shock_size   = shock_size,
                factor_stds  = self._factor_stds,
                horizon      = horizon,
            )
            baseline = trajectory[shock_time : shock_time + horizon]
            return CounterfactualResult(
                baseline       = baseline.copy(),
                counterfactual = baseline + irf,
                causal_effect  = irf,
                shock_factor   = shock_factor,
                shock_size     = shock_size,
                shock_time     = shock_time,
                horizon        = horizon,
                method         = "analytical",
            )
        elif method == "montecarlo":
            return counterfactual_montecarlo(
                trajectory   = trajectory,
                noises       = noises,
                transition   = self.transition,
                shock_time   = shock_time,
                shock_factor = shock_factor,
                shock_size   = shock_size,
                factor_stds  = self._factor_stds,
                horizon      = horizon,
            )
        else:
            raise ValueError(f"Unknown method: {method}. "
                             f"Use 'resimulation', 'analytical', or 'montecarlo'.")

    def evaluate(self,
                 model_counterfactual: np.ndarray,
                 model_baseline:       np.ndarray,
                 true_result:          CounterfactualResult) -> Dict[str, float]:
        """
        Evaluate a model's counterfactual against the ground truth.

        model_counterfactual : (horizon, k)  model's CF trajectory
        model_baseline       : (horizon, k)  model's baseline trajectory
        true_result          : CounterfactualResult from self.counterfactual()

        Returns dict of metrics:
            direction_accuracy  — fraction of correct sign predictions
            sign_error_rate     — fraction of wrong sign predictions
            rmse_effect         — RMSE of causal effect
            mae_effect          — MAE of causal effect
            corr_effect         — correlation of model vs true effect
            dir_acc_F{i}        — per-factor direction accuracy
        """
        return evaluate_model_counterfactual(
            model_cf   = model_counterfactual,
            model_base = model_baseline,
            true_result = true_result,
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_train_test_split(self, trajectory: np.ndarray,
                              train_frac: float = 0.7
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """Split trajectory into train and test windows."""
        T      = len(trajectory)
        t_end  = int(train_frac * T)
        return trajectory[:t_end], trajectory[t_end:]

    def expected_irf(self, F_t: np.ndarray,
                     shock_factor: int,
                     shock_size: float,
                     horizon: int) -> np.ndarray:
        """
        Analytical impulse response function at state F_t.
        Returns causal_effect : (horizon, k).
        """
        assert self._factor_stds is not None
        return counterfactual_analytical(
            F_shock_time = F_t,
            transition   = self.transition,
            shock_factor = shock_factor,
            shock_size   = shock_size,
            factor_stds  = self._factor_stds,
            horizon      = horizon,
        )

    @property
    def factor_stds(self) -> np.ndarray:
        """Per-factor standard deviations from training window."""
        assert self._factor_stds is not None, "Call simulate() first."
        return self._factor_stds

    def __repr__(self) -> str:
        return (f"MacroEconomy(k={self.k}, "
                f"nonlinearity={self.nonlinearity}, "
                f"seed={self.seed})")
