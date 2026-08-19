r"""Euler-Maruyama on the reverse-time SDE. The stochastic baseline.

Anderson's reverse-time SDE, as used by Song et al. (2011.13456, eq. 6):

    dx = [ f(x,t) - g(t)^2 grad_x log p_t(x) ] dt + g(t) d w_bar

integrated from T down to t_eps. With h_n = t_{n+1} - t_n < 0 and
dW_n ~ N(0, |h_n|):

    x_{n+1} = x_n + h_n [ f(x_n,t_n) - g(t_n)^2 s(x_n,t_n) ] + g(t_n) dW_n

**On the expected order.** The usual quotation for Euler-Maruyama is strong order
1/2, weak order 1. The 1/2 comes from the Milstein correction term
(1/2) g g_x (dW^2 - dt), which is present only when the diffusion coefficient depends
on the state. Here it does not: g(t) is a function of time alone, the noise is
additive, and the Milstein term vanishes identically -- Euler-Maruyama *is* Milstein
for this SDE, and its strong order is 1. This is a standard result for additive-noise
SDEs (Kloeden & Platen, *Numerical Solution of Stochastic Differential Equations*).
Every diffusion model's reverse SDE has this structure, so the 1/2 that gets quoted
in the sampling literature is pessimistic. analysis/convergence_order.py measures it
rather than taking either number on trust.

Strong error is measured with common random numbers: the Brownian increments are
drawn once on the finest grid and *summed* over blocks to build every coarser grid,
so all resolutions are driven by the same Brownian path and the comparison is
pathwise. Without that, one is measuring Monte-Carlo noise.
"""

from __future__ import annotations

import numpy as np

from ..nfe import SamplerResult, ScoreCounter, ScoreFn
from ..sde import reverse_sde_field


def brownian_increments(rng: np.random.Generator, times: np.ndarray,
                        shape: tuple[int, ...]) -> np.ndarray:
    """dW_n ~ N(0, |h_n|) for each step of `times`; shape (n_steps, *shape)."""
    times = np.asarray(times, dtype=np.float64)
    dt = np.abs(np.diff(times))
    return np.sqrt(dt)[:, None, None] * rng.normal(size=(dt.size, *shape))


def coarsen_increments(dw: np.ndarray, factor: int) -> np.ndarray:
    """Sum blocks of `factor` fine increments into one coarse increment.

    The defining property of Brownian motion: W(t+2h) - W(t) is the sum of the two
    sub-increments, exactly. Coarsening this way (rather than redrawing) is what makes
    the strong-order fit a pathwise statement.
    """
    n = dw.shape[0]
    if n % factor:
        raise ValueError(f"{n} fine steps do not divide into blocks of {factor}")
    return dw.reshape(n // factor, factor, *dw.shape[1:]).sum(axis=1)


def euler_maruyama(score: ScoreFn, sde, x_init: np.ndarray, times: np.ndarray,
                   rng: np.random.Generator | None = None,
                   increments: np.ndarray | None = None) -> SamplerResult:
    """Integrate the reverse SDE. Supply `increments` for common random numbers."""
    counted = ScoreCounter(score)
    x = np.array(x_init, dtype=np.float64, copy=True)
    times = np.asarray(times, dtype=np.float64)
    n_steps = times.size - 1
    if increments is None:
        if rng is None:
            raise ValueError("euler_maruyama needs either `rng` or `increments`")
        increments = brownian_increments(rng, times, x.shape)
    if increments.shape[0] != n_steps:
        raise ValueError(f"{increments.shape[0]} increments for {n_steps} steps")

    for n in range(n_steps):
        t, h = float(times[n]), float(times[n + 1] - times[n])
        drift = reverse_sde_field(sde, counted(x, t), x, t)
        x = x + h * drift + float(sde.diffusion(t)) * increments[n]
    return SamplerResult(x=x, nfe=counted.nfe, steps=n_steps, times=times)
