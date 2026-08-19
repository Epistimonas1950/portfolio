r"""Heun's method (explicit trapezoidal rule) on the probability-flow ODE. Order 2.

    k1      = F(x_n, t_n)
    x_euler = x_n + h k1
    k2      = F(x_euler, t_{n+1})
    x_{n+1} = x_n + (h/2) (k1 + k2)

Two score evaluations per step, so at matched NFE it takes half the steps Euler does.
It wins anyway: the local error drops from O(h^2) to O(h^3), so global error goes
from O(h) to O(h^2), and doubling h to pay for the extra stage costs a factor 2 while
buying a factor 4.

Why it is second order, concretely: expand k2 = F(x_n + h k1, t_n + h) about (x_n,t_n),
so (k1+k2)/2 = F + (h/2)(F_t + F_x F) + O(h^2), and h(F + (h/2) dF/dt) is exactly the
first two terms of the Taylor expansion of x(t_n + h). The h^2 term matches; the h^3
term does not, and that residual is the local truncation error.

This is the sampler Karras et al. (2206.00364) use, and their argument for it is the
one above -- a numerical-analysis argument, not a modelling one.
"""

from __future__ import annotations

import numpy as np

from ..nfe import SamplerResult, ScoreCounter, ScoreFn
from ..sde import probability_flow_field


def heun(score: ScoreFn, sde, x_init: np.ndarray, times: np.ndarray) -> SamplerResult:
    counted = ScoreCounter(score)
    x = np.array(x_init, dtype=np.float64, copy=True)
    times = np.asarray(times, dtype=np.float64)
    for n in range(times.size - 1):
        t0, t1 = float(times[n]), float(times[n + 1])
        h = t1 - t0
        k1 = probability_flow_field(sde, counted(x, t0), x, t0)
        x_euler = x + h * k1
        k2 = probability_flow_field(sde, counted(x_euler, t1), x_euler, t1)
        x = x + 0.5 * h * (k1 + k2)
    return SamplerResult(x=x, nfe=counted.nfe, steps=times.size - 1, times=times)
