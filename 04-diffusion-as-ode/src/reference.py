r"""Ground truth for the probability-flow ODE, two ways -- exact, and very-high-NFE.

A convergence-order study is only as good as what it measures error against. Using a
fine-grid numerical solution as "truth" is the usual compromise and it biases the
finest levels, which are precisely the ones that decide the fitted slope. Here the
exact solution is available, so the fine-grid reference is demoted to a cross-check.

(a) The exact affine flow, for a single Gaussian prior, in any dimension
-----------------------------------------------------------------------
If p_0 = N(mu, v I) then p_t = N(alpha_t mu, V_t I) with V_t = alpha_t^2 v + sigma_t^2.
Claim: the probability-flow trajectories are

    x(t) = m_t + sqrt(V_t / V_s) ( x(s) - m_s ),      m_t = alpha_t mu

Proof. Write x(t) = m_t + c(t) z with c = sqrt(V) and z fixed by the initial
condition; then dx/dt = m' + (V'/(2V))(x - m). The probability-flow field for this
p_t is

    F = a x - (1/2) g^2 grad log p_t = a x + (g^2/2)(x - m)/V

Matching the terms proportional to m gives m' = a m, which holds because
m = alpha mu and alpha' = a alpha. Matching the terms in (x - m) gives the condition

    V' = 2 a V + g^2

For VP (a = -beta/2, g^2 = beta) this is V' = beta (1 - V), and indeed
V = 1 - e^{-B}(1 - v) gives V' = beta e^{-B}(1 - v) = beta(1 - V). For VE (a = 0,
g^2 = (sigma^2)') it is V' = (sigma^2)', which is immediate. So the claim holds for
both families. []

(b) The exact flow for *any* 1-D prior: the quantile map
--------------------------------------------------------
The probability-flow ODE transports p_s to p_t by construction (it is the continuity
equation for that density). Its field is Lipschitz in x on compact time intervals --
for a Gaussian mixture the score is smooth and its derivative is bounded away from
t = 0 -- so solutions are unique and trajectories cannot cross. In one dimension a
non-crossing flow map is strictly increasing, and a strictly increasing map pushing
p_s to p_t is unique: it is the quantile map. Hence

    Phi_{s -> t}(x)  =  F_t^{-1}( F_s(x) )

exactly, for a Gaussian mixture with any number of components. This is the ground
truth the convergence study uses.

Two practical consequences:

  * Initial conditions are specified by *probability levels* p_i, not by x values.
    Then x(s) = F_s^{-1}(p_i) and the exact answer is x(t) = F_t^{-1}(p_i), and no
    forward CDF is ever evaluated -- which matters, because F_s(x) saturates to 0 or 1
    in double precision several standard deviations out and inverting a saturated
    value would manufacture error that has nothing to do with the integrator.
  * The map is exact to the accuracy of the quantile inversion, which is
    bisection-plus-Newton at ~1e-15 relative. Six orders of magnitude below the
    smallest discretization error measured anywhere in this repo.

(c) The high-NFE reference
--------------------------
`reference_trajectory` runs DPM-Solver-2 on a grid uniform in log-SNR with a large
step count. It is used where (b) does not apply -- dimensions above one, and any
sanity check on the reverse SDE -- and, more importantly, as an independent check on
(b): tests/test_reference.py requires the two to agree, which they could not do if
either the analytic argument or the sampler were wrong.
"""

from __future__ import annotations

import numpy as np

from .nfe import ScoreFn
from .samplers.exponential import exponential
from .schedule import uniform_logsnr_grid
from .sde import GaussianMixture

#: Step count for the numerical reference. 4096 DPM-Solver-2 steps = 8192 NFE. Its
#: measured RMS distance from the analytic quantile map is 1.2e-6, and that distance
#: falls by exactly 4.00x per doubling (tests/test_reference.py) -- which is the check
#: that matters: a second-order sampler converging at order 2 *to the analytic map*
#: cannot happen unless both are right. It is a cross-check, not the ground truth;
#: the ground truth is `quantile_states`, exact to rounding.
REFERENCE_STEPS = 4096


def analytic_gaussian_flow(sde, mean: np.ndarray, variance: float, x: np.ndarray,
                           t_from: float, t_to: float) -> np.ndarray:
    """Exact probability-flow map for a single isotropic Gaussian prior N(mean, v I)."""
    mean = np.atleast_2d(np.asarray(mean, dtype=np.float64))
    a0, s0 = float(sde.alpha(t_from)), float(sde.sigma(t_from))
    a1, s1 = float(sde.alpha(t_to)), float(sde.sigma(t_to))
    v0 = a0 * a0 * variance + s0 * s0
    v1 = a1 * a1 * variance + s1 * s1
    return a1 * mean + np.sqrt(v1 / v0) * (np.asarray(x, dtype=np.float64) - a0 * mean)


def quantile_states(sde, prior: GaussianMixture, probs: np.ndarray,
                    t: float) -> np.ndarray:
    """x = F_t^{-1}(probs): the exact state at time t of the trajectory carrying
    probability level `probs`. Shape (n, 1)."""
    return sde.marginal(prior, float(t)).quantile(np.asarray(probs)).reshape(-1, 1)


def exact_flow_map(sde, prior: GaussianMixture, x: np.ndarray, t_from: float,
                   t_to: float) -> np.ndarray:
    """Phi_{t_from -> t_to}(x) = F_{t_to}^{-1}(F_{t_from}(x)), exact in 1-D.

    Use `quantile_states` instead when you are free to choose the initial condition;
    this form evaluates a forward CDF and therefore loses accuracy for x far out in
    the tail, where F saturates.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    p = sde.marginal(prior, float(t_from)).cdf(x)
    p = np.clip(p, 1e-300, 1.0 - 1e-16)
    return sde.marginal(prior, float(t_to)).quantile(p).reshape(-1, 1)


def reference_trajectory(score: ScoreFn, sde, x_init: np.ndarray, t_from: float,
                         t_to: float, n_steps: int = REFERENCE_STEPS) -> np.ndarray:
    """Very-high-NFE numerical reference: DPM-Solver-2 on a uniform log-SNR grid."""
    grid = uniform_logsnr_grid(sde, n_steps, t_start=t_from, t_end=t_to)
    return exponential(score, sde, x_init, grid, order=2).x


def probability_levels(n: int, p_lo: float = 1e-3, p_hi: float = 1.0 - 1e-3) -> np.ndarray:
    """Evenly spaced probability levels, used as the initial conditions everywhere.

    Truncated at 1e-3 rather than run to the extreme tail: the quantile inversion is
    accurate out there, but the trajectories are not interesting and their large
    magnitudes would dominate an RMS error norm with the behaviour of one sample.
    """
    return np.linspace(p_lo, p_hi, n)
