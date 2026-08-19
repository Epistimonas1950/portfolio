r"""Explicit Euler on the probability-flow ODE. The deterministic baseline, order 1.

The probability-flow ODE of Song et al. (2011.13456, eq. 13) has the same marginals
p_t as the reverse SDE but no noise:

    dx/dt = f(x,t) - (1/2) g(t)^2 grad_x log p_t(x)  =:  F(x,t)

Explicit Euler on a decreasing grid t_0 = T > t_1 > ... > t_N = t_eps, with
h_n = t_{n+1} - t_n < 0:

    x_{n+1} = x_n + h_n F(x_n, t_n)

Local truncation error is (h^2/2) x''(tau), so order 2 locally. Globally the errors
accumulate over N = O(1/h) steps and are amplified by the flow's own Lipschitz
constant, giving the standard bound

    ||e_N||  <=  (h C / (2 L)) ( e^{L (T - t_eps)} - 1 )   =  O(h)

-- order p locally gives order p globally, one power of h being spent on the step
count. That is the statement the convergence study measures rather than assumes:
1 NFE per step, fitted slope 1.

This is DDIM's deterministic limit written as what it is.
"""

from __future__ import annotations

import numpy as np

from ..nfe import SamplerResult, ScoreCounter, ScoreFn
from ..sde import probability_flow_field


def euler_ode(score: ScoreFn, sde, x_init: np.ndarray, times: np.ndarray) -> SamplerResult:
    """Integrate the probability-flow ODE from times[0] down to times[-1]."""
    counted = ScoreCounter(score)
    x = np.array(x_init, dtype=np.float64, copy=True)
    times = np.asarray(times, dtype=np.float64)
    for n in range(times.size - 1):
        t, h = float(times[n]), float(times[n + 1] - times[n])
        x = x + h * probability_flow_field(sde, counted(x, t), x, t)
    return SamplerResult(x=x, nfe=counted.nfe, steps=times.size - 1, times=times)
