"""The five integrators, all of them counting network evaluations.

    euler_maruyama   reverse SDE,  1 NFE/step
    euler_ode        prob. flow,   1 NFE/step,  order 1
    heun             prob. flow,   2 NFE/step,  order 2
    exponential      prob. flow,   1 or 2 NFE/step, order 1 or 2, linear part exact
    adaptive_heun    prob. flow,   2 NFE/attempted step, PI-controlled
"""

from .adaptive import adaptive_heun
from .euler_maruyama import brownian_increments, coarsen_increments, euler_maruyama
from .euler_ode import euler_ode
from .exponential import dpm_solver_1_multiplier, exponential
from .heun import heun

__all__ = ["adaptive_heun", "brownian_increments", "coarsen_increments",
           "euler_maruyama", "euler_ode", "exponential", "heun",
           "dpm_solver_1_multiplier"]

#: Fixed-step probability-flow samplers, keyed by the name used in results/*.csv.
ODE_SAMPLERS = {
    "euler_ode": (euler_ode, 1),
    "heun": (heun, 2),
    "exponential_1": (lambda s, sde, x, t: exponential(s, sde, x, t, order=1), 1),
    "exponential_2": (lambda s, sde, x, t: exponential(s, sde, x, t, order=2), 2),
}
