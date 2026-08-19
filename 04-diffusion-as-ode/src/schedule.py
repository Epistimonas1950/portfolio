r"""Time grids: the step-size parameterization a sampler is handed.

Sampling integrates *backwards*, from t_max down to t_min, so every grid returned
here is strictly decreasing and every step h = t_{n+1} - t_n is negative. Two
families, and the difference between them is a real experimental variable:

uniform in t
    t_n = T - n (T - t_eps)/N. The obvious grid, and the one convergence-order
    studies should use, because halving N halves every h exactly and the fitted
    log-log slope is then unambiguous.

uniform in log-SNR lambda = log(alpha/sigma)
    lambda_n equally spaced, t_n = t_lambda(lambda_n). This is what production
    samplers actually use (DPM-Solver, EDM), and for good reason: on the VP schedule
    d lambda/dt = -beta(t) / (2 sigma(t)^2), which at t = 10^-3 is about 500x its
    value at t = 1. A uniform-t grid therefore spends most of its steps where the
    solution is barely moving in lambda and too few where it moves fastest.

The adaptive sampler is measured against *both*. Beating uniform-t is easy here and
means little; beating uniform-lambda is the claim worth making.
"""

from __future__ import annotations

import numpy as np


def uniform_time_grid(sde, n_steps: int, t_start: float | None = None,
                      t_end: float | None = None) -> np.ndarray:
    """Decreasing grid of n_steps+1 times, equally spaced in t."""
    t0 = sde.t_max if t_start is None else t_start
    t1 = sde.t_min if t_end is None else t_end
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    return np.linspace(t0, t1, n_steps + 1)


def uniform_logsnr_grid(sde, n_steps: int, t_start: float | None = None,
                        t_end: float | None = None) -> np.ndarray:
    """Decreasing grid of n_steps+1 times, equally spaced in lambda = log(alpha/sigma)."""
    t0 = sde.t_max if t_start is None else t_start
    t1 = sde.t_min if t_end is None else t_end
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    lam = np.linspace(float(sde.log_snr(t0)), float(sde.log_snr(t1)), n_steps + 1)
    t = np.asarray(sde.t_of_log_snr(lam), dtype=np.float64)
    t[0], t[-1] = t0, t1              # pin the endpoints against round-off in t_lambda
    return t


GRIDS = {"uniform_t": uniform_time_grid, "uniform_logsnr": uniform_logsnr_grid}


def time_grid(kind: str, sde, n_steps: int, **kw) -> np.ndarray:
    if kind not in GRIDS:
        raise ValueError(f"unknown grid {kind!r}; choose from {sorted(GRIDS)}")
    return GRIDS[kind](sde, n_steps, **kw)
