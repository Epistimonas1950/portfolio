r"""Adaptive step-size control from an embedded Heun/Euler pair.

The two stages of Heun already contain an order-1 result, so an error estimate is
free -- no extra network evaluation:

    k1 = F(x_n, t_n)            x_low  = x_n + h k1                (Euler,  order 1)
    k2 = F(x_low, t_n + h)      x_high = x_n + (h/2)(k1 + k2)      (Heun,   order 2)

    err_vec = x_high - x_low = (h/2)(k2 - k1)

That difference is an O(h^2) estimate of the local error of the *lower*-order member.
We advance with x_high anyway (local extrapolation), which is standard and makes the
controller slightly conservative: the advancing method is more accurate than the one
being measured.

**Scaling.** Componentwise tolerance sc_i = atol + rtol max(|x_i|, |x_high,i|), and

    err = sqrt( mean_i ( err_vec_i / sc_i )^2 )

accept if err <= 1. Mixed absolute/relative scaling is Hairer-Norsett-Wanner's
recommendation and matters here because samples pass through 0 on their way between
modes, where a pure relative tolerance is meaningless.

**PI control.** The plain deadbeat controller h_new = h err^{-1/k} oscillates: a step
that was slightly too large is followed by one that is too small, and so on. The
proportional-integral controller of Gustafsson, as given in Hairer & Wanner,

    h_new = h * safety * err_n^{-kI} * err_{n-1}^{kP},   kI = 0.7/k, kP = 0.4/k

damps that. Here k = 2, the order of the error estimate. The growth factor is clamped
to [0.2, 5] so a single anomalous step cannot make the solver take a wild one.

**NFE accounting.** Two evaluations per *attempted* step, accepted or not. Rejected
steps are counted; a solver that hid them would not be comparable with a fixed-step
one, and rejections are exactly where an adaptive method's overhead lives.

**Why there is one step size for the whole batch.** The score is evaluated on the
batch in one forward pass, so a per-sample step size would need a separate pass per
sample and destroy the NFE advantage. The error norm is therefore reduced over the
batch as well as over the state dimension -- the same choice production samplers make.
"""

from __future__ import annotations

import numpy as np

from ..nfe import SamplerResult, ScoreCounter, ScoreFn
from ..sde import probability_flow_field

SAFETY = 0.9
FAC_MIN, FAC_MAX = 0.2, 5.0
ERROR_ORDER = 2                 # x_high - x_low is O(h^2)
KI, KP = 0.7 / ERROR_ORDER, 0.4 / ERROR_ORDER


def adaptive_heun(score: ScoreFn, sde, x_init: np.ndarray, rtol: float = 1e-3,
                  atol: float = 1e-4, t_start: float | None = None,
                  t_end: float | None = None, h_init: float | None = None,
                  max_steps: int = 100_000) -> SamplerResult:
    """Integrate the probability-flow ODE with PI-controlled steps."""
    counted = ScoreCounter(score)
    x = np.array(x_init, dtype=np.float64, copy=True)
    t = sde.t_max if t_start is None else float(t_start)
    t_final = sde.t_min if t_end is None else float(t_end)
    span = t - t_final
    if span <= 0:
        raise ValueError("t_start must exceed t_end; sampling integrates backwards")

    h = -span / 32.0 if h_init is None else -abs(h_init)
    err_prev = 1.0
    accepted = rejected = 0
    ts = [t]

    while t > t_final + 1e-15:
        h = -min(abs(h), t - t_final)                 # never overshoot the endpoint
        t_next = t + h
        k1 = probability_flow_field(sde, counted(x, t), x, t)
        x_low = x + h * k1
        k2 = probability_flow_field(sde, counted(x_low, t_next), x_low, t_next)
        x_high = x + 0.5 * h * (k1 + k2)

        sc = atol + rtol * np.maximum(np.abs(x), np.abs(x_high))
        err = float(np.sqrt(np.mean(((x_high - x_low) / sc) ** 2)))
        err = max(err, 1e-12)                         # keep the exponent finite

        if err <= 1.0:
            x, t = x_high, t_next
            accepted += 1
            ts.append(t)
            factor = SAFETY * err ** (-KI) * err_prev ** KP
            err_prev = err
        else:
            rejected += 1
            factor = SAFETY * err ** (-1.0 / ERROR_ORDER)

        h *= float(np.clip(factor, FAC_MIN, FAC_MAX))
        if accepted + rejected > max_steps:
            raise RuntimeError(
                f"adaptive_heun exceeded {max_steps} attempted steps at t={t:.3e}; "
                "rtol/atol are too tight for this problem")

    return SamplerResult(x=x, nfe=counted.nfe, steps=accepted, accepted=accepted,
                         rejected=rejected, times=np.asarray(ts))
