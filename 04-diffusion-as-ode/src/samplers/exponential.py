r"""Exponential integrator (DPM-Solver style): integrate the linear part exactly.

This is the one classical idea in the whole sampler literature that is worth the most,
and it is not a deep-learning idea. It is exponential integration, which numerical
analysis has used on semi-linear stiff problems since the 1960s and which Hochbruck &
Ostermann's survey and Hairer-Norsett-Wanner treat as standard equipment.

**The structure.** Write the probability-flow ODE in the noise parameterization. With
x_t = alpha(t) x_0 + sigma(t) eps and eps(x,t) = -sigma(t) grad log p_t(x),

    dx/dt = a(t) x  -  (1/2) g(t)^2 grad log p_t(x)
          = a(t) x  +  ( g(t)^2 / (2 sigma(t)) ) eps(x, t)
            \_____/     \_______________________________/
            linear,     nonlinear, and *smooth*: eps stays O(1) as t -> 0 while the
            stiff,      score it is built from blows up like 1/sigma
            exact

Euler and Heun approximate the whole right-hand side by a polynomial in h. That is
wasteful: the linear part has a closed-form solution and there is no reason to spend
accuracy on it.

**Variation of constants.** With alpha(t) = exp(int a), the exact solution from s to t
is

    x(t) = (alpha_t / alpha_s) x(s)  +  alpha_t int_s^t ( g^2 / (2 alpha sigma) ) eps(u) du

Change variables to the log-SNR lambda = log(alpha/sigma). For both SDE families here
d lambda/du = -g^2 / (2 sigma^2), and alpha sigma d lambda = -(g^2/2) du, which turns
the integral into

    x(t) = (alpha_t / alpha_s) x(s)  -  alpha_t int_{lambda_s}^{lambda_t} e^{-lambda}
                                                  eps_hat(lambda) d lambda

with eps_hat(lambda) = eps(x(t_lambda), t_lambda). Everything except eps_hat is now
exact. The e^{-lambda} weight is the whole stiffness, and it is integrated in closed
form.

**Order 1 (DPM-Solver-1).** Freeze eps_hat at the left endpoint. With
h = lambda_t - lambda_s,

    x(t) = (alpha_t/alpha_s) x(s) - sigma_t (e^h - 1) eps(x(s), s)

**Order 2 (DPM-Solver-2).** Evaluate eps at the lambda-midpoint, the exponential
analogue of the explicit midpoint rule -- one extra evaluation per step:

    lambda_m = (lambda_s + lambda_t)/2,   t_m = t_lambda(lambda_m)
    u = (alpha_m/alpha_s) x(s) - sigma_m (e^{h/2} - 1) eps(x(s), s)
    x(t) = (alpha_t/alpha_s) x(s) - sigma_t (e^h - 1) eps(u, t_m)

Both are Lu et al. (2206.00927), Algorithms 1-2, rederived here from variation of
constants.

**When is order 1 already exact?** Exactly when eps_hat is constant along the
trajectory. For a point-mass prior delta(x - mu) the marginal is N(alpha mu, sigma^2),
the flow map is x(t) = alpha_t mu + (sigma_t/sigma_s)(x(s) - alpha_s mu), so
eps = (x - alpha mu)/sigma is invariant and one step of DPM-Solver-1 -- any step, of
any size -- is exact to machine precision. For a Gaussian prior N(mu, v) with v > 0 it
is not; the deficit is available in closed form. Both multipliers on x(s) - alpha_s mu
are scalars,

    R_exact = sqrt(V_t / V_s),        R_dpm1 = (alpha_t alpha_s v + sigma_t sigma_s) / V_s

with V_t = alpha_t^2 v + sigma_t^2, and subtracting the squares gives

    R_exact^2 - R_dpm1^2  =  v (alpha_t sigma_s - sigma_t alpha_s)^2 / V_s^2

so the error is proportional to the prior variance times the squared log-SNR gap, and
vanishes iff v = 0 or the step is empty. tests/test_exponential.py asserts that
identity to 1e-13; it is a much sharper statement about the implementation than "one
step is exact".

**Numerics.** sigma_t (e^h - 1) is evaluated as sigma_t * expm1(h) -- accurate for the
small h that fine grids produce, where the algebraically equivalent
alpha_t sigma_s/alpha_s - sigma_t cancels. On these schedules the largest single-step
h is about 9.6 (the whole lambda range), so e^h never approaches overflow.
"""

from __future__ import annotations

import numpy as np

from ..nfe import SamplerResult, ScoreCounter, ScoreFn


def exponential(score: ScoreFn, sde, x_init: np.ndarray, times: np.ndarray,
                order: int = 2) -> SamplerResult:
    """DPM-Solver of order 1 or 2 on the given decreasing time grid."""
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")
    counted = ScoreCounter(score)
    x = np.array(x_init, dtype=np.float64, copy=True)
    times = np.asarray(times, dtype=np.float64)

    for n in range(times.size - 1):
        t0, t1 = float(times[n]), float(times[n + 1])
        a0, s0 = float(sde.alpha(t0)), float(sde.sigma(t0))
        a1, s1 = float(sde.alpha(t1)), float(sde.sigma(t1))
        lam0, lam1 = float(sde.log_snr(t0)), float(sde.log_snr(t1))
        h = lam1 - lam0
        eps0 = -s0 * counted(x, t0)

        if order == 1:
            x = (a1 / a0) * x - s1 * np.expm1(h) * eps0
        else:
            tm = float(sde.t_of_log_snr(0.5 * (lam0 + lam1)))
            am, sm = float(sde.alpha(tm)), float(sde.sigma(tm))
            u = (am / a0) * x - sm * np.expm1(0.5 * h) * eps0
            eps_m = -sm * counted(u, tm)
            x = (a1 / a0) * x - s1 * np.expm1(h) * eps_m

    return SamplerResult(x=x, nfe=counted.nfe, steps=times.size - 1, times=times)


def dpm_solver_1_multiplier(sde, prior_variance: float, t_from: float,
                            t_to: float) -> tuple[float, float]:
    """(R_exact, R_dpm1): the two scalar multipliers of a one-step Gaussian flow.

    Used by the tests to check the closed-form deficit derived in this module's
    docstring, and by analysis/stability.py, where R is the amplification factor whose
    modulus decides stability.
    """
    a0, s0 = float(sde.alpha(t_from)), float(sde.sigma(t_from))
    a1, s1 = float(sde.alpha(t_to)), float(sde.sigma(t_to))
    v0 = a0 * a0 * prior_variance + s0 * s0
    v1 = a1 * a1 * prior_variance + s1 * s1
    return float(np.sqrt(v1 / v0)), float((a1 * a0 * prior_variance + s1 * s0) / v0)
