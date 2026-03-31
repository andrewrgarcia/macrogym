"""
macrogym/transitions.py
────────────────────────
Nonlinear state-dependent transition functions for the synthetic economy.

The transition matrix A(F_t) determines how factors evolve:

    F_{t+1} = A(F_t) · F_t + Σ(F_t)^{1/2} · ε_t

where A(F_t) and Σ(F_t) both depend on the current state.

Design principles
─────────────────
1. Nonlinearity is controlled by a single scalar parameter ∈ [0, 1].
   At 0 the economy is a linear VAR — the VAR wins every benchmark.
   At 1 the economy is maximally nonlinear — only flexible models work.
   Intermediate values produce calibrated challenges.

2. Nonlinearities are economically motivated:
   - Regime asymmetry: recessions propagate differently from expansions
   - Volatility clustering: uncertainty rises during downturns
   - Transmission nonlinearity: monetary policy has larger effects at extremes
   
3. The true causal effect of a shock is computable analytically for the
   linear component and by exact re-simulation for the nonlinear component.

Factor interpretation (default k=5):
   F0 — real activity (output gap / growth)
   F1 — inflation
   F2 — monetary policy stance (interest rate gap)
   F3 — financial conditions (credit spreads)
   F4 — external sector (trade / commodity prices)
"""

import numpy as np
from typing import Tuple


# ── Default structural matrices ───────────────────────────────────────────────
# Calibrated to produce realistic impulse responses:
#   - Real activity is persistent (0.85 own-lag)
#   - Inflation responds to activity (Phillips curve)
#   - Monetary policy responds to inflation (Taylor rule)
#   - Financial conditions amplify downturns (accelerator)
#   - External sector is exogenous but affects activity

A_NORMAL = np.array([
    [0.85,  0.00,  -0.10,  -0.15,   0.20],   # F0 real activity
    [0.15,  0.75,   0.00,   0.05,   0.10],   # F1 inflation
    [0.05,  0.30,   0.80,   0.00,   0.00],   # F2 monetary policy
    [-0.20, 0.00,  -0.10,   0.70,   0.00],   # F3 financial conditions
    [0.00,  0.00,   0.00,   0.00,   0.90],   # F4 external sector
], dtype=np.float64)

# Recession regime — tighter financial accelerator, stronger monetary response
A_RECESSION = np.array([
    [0.70,  0.00,  -0.15,  -0.35,   0.15],   # F0 more sensitive to finance
    [0.10,  0.80,   0.00,   0.10,   0.05],   # F1 inflation stickier
    [0.08,  0.35,   0.82,   0.00,   0.00],   # F2 stronger policy response
    [-0.35, 0.00,  -0.15,   0.75,   0.00],   # F3 financial amplification
    [0.00,  0.00,   0.00,   0.00,   0.88],   # F4 external weakens
], dtype=np.float64)

# Base noise covariance — calibrated to unit factor variance
SIGMA_BASE = np.diag([0.30, 0.20, 0.15, 0.25, 0.20]) ** 2


class NonlinearTransition:
    """
    State-dependent nonlinear transition for the synthetic economy.

    The transition interpolates smoothly between a normal regime matrix
    and a recession regime matrix based on the current state of real
    activity (F0):

        r_t = sigmoid(-sharpness · F0_t)     ∈ [0,1]
        A_t = (1 - r_t) · A_normal + r_t · A_recession

    r_t ≈ 0 in expansions (F0 > 0) → A_normal applies
    r_t ≈ 1 in recessions (F0 < 0) → A_recession applies

    Stochastic volatility: noise variance scales with monetary tightness:
        Σ_t = Σ_base · exp(vol_sensitivity · |F2_t|)

    Parameters
    ──────────
    nonlinearity : float ∈ [0, 1]
        Strength of regime-switching. 0 = pure linear VAR, 1 = full asymmetry.
    sharpness : float
        Speed of regime transition. Higher = more abrupt switch.
    vol_sensitivity : float
        Degree of stochastic volatility. 0 = homoskedastic.
    A_normal : np.ndarray (k, k)
        Transition matrix in normal/expansion regime.
    A_recession : np.ndarray (k, k)
        Transition matrix in recession regime.
    sigma_base : np.ndarray (k, k)
        Base noise covariance.
    """

    def __init__(self,
                 nonlinearity:    float = 0.5,
                 sharpness:       float = 2.0,
                 vol_sensitivity: float = 0.3,
                 A_normal:        np.ndarray = A_NORMAL,
                 A_recession:     np.ndarray = A_RECESSION,
                 sigma_base:      np.ndarray = SIGMA_BASE):

        self.nonlinearity    = nonlinearity
        self.sharpness       = sharpness
        self.vol_sensitivity = vol_sensitivity
        self.A_normal        = A_normal.copy()
        self.A_recession     = A_recession.copy()
        self.delta_A         = A_recession - A_normal   # (k, k)
        self.sigma_base      = sigma_base.copy()
        self.k               = A_normal.shape[0]

    def regime_weight(self, F_t: np.ndarray) -> float:
        """
        Smooth regime weight r_t ∈ [0, 1].
        r_t = 0 → expansion (A_normal)
        r_t = 1 → recession (A_recession)
        Driven by F0 (real activity).
        """
        x = np.clip(self.sharpness * F_t[0], -500, 500)
        return 1.0 / (1.0 + np.exp(x))

    def transition_matrix(self, F_t: np.ndarray) -> np.ndarray:
        """
        State-dependent transition matrix A(F_t).

        A(F_t) = A_normal + nonlinearity · r_t · ΔA

        Linear when nonlinearity=0: A(F_t) = A_normal for all F_t.
        """
        r = self.regime_weight(F_t)
        return self.A_normal + self.nonlinearity * r * self.delta_A

    def noise_covariance(self, F_t: np.ndarray) -> np.ndarray:
        """
        State-dependent noise covariance Σ(F_t).

        Σ(F_t) = Σ_base · exp(vol_sensitivity · |F2_t|)

        Volatility rises when monetary policy is far from neutral.
        Zero vol_sensitivity → homoskedastic baseline.
        """
        exponent = np.clip(self.nonlinearity * self.vol_sensitivity * abs(F_t[2]), -10, 3)
        scale = np.exp(exponent)
        return self.sigma_base * scale

    def step(self, F_t: np.ndarray,
             rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """
        One-step transition: F_t → F_{t+1}.

        Returns (F_{t+1}, noise) where noise is the drawn innovation.
        Returning noise separately enables exact counterfactual re-simulation
        with the same noise draw but a different initial state.
        """
        A   = self.transition_matrix(F_t)
        Sig = self.noise_covariance(F_t)
        L   = np.linalg.cholesky(Sig)
        eps = rng.standard_normal(self.k)
        noise = L @ eps
        return A @ F_t + noise, noise

    def step_deterministic(self, F_t: np.ndarray,
                            noise: np.ndarray) -> np.ndarray:
        """
        One-step transition with fixed noise draw.
        Used for exact counterfactual re-simulation:
            F_cf_{t+1} = A(F_cf_t) · F_cf_t + noise_t
        where noise_t is the same draw as in the baseline.
        This isolates the causal effect of the shock from noise.
        """
        A = self.transition_matrix(F_t)
        return A @ F_t + noise

    def expected_next(self, F_t: np.ndarray) -> np.ndarray:
        """
        Expected next state E[F_{t+1} | F_t] = A(F_t) · F_t.
        Used for analytical counterfactual computation.
        """
        return self.transition_matrix(F_t) @ F_t

    def linear_approximation(self, F_t: np.ndarray) -> np.ndarray:
        """
        First-order Taylor expansion of A(F_t) around F_t.
        Used for analytical impulse response computation.
        Returns the effective A matrix at F_t.
        """
        return self.transition_matrix(F_t)