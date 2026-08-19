#!/usr/bin/env python3
r"""Stiffness of the reverse process, the explicit method's step limit, and what the
exponential integrator actually buys.

Three measurements, all written to results/stability.csv.

1. `stiffness` -- how stiff is it, really, and which way?
   The probability-flow field has Jacobian J = dF/dx, evaluated here by central
   differences on the exact field along the exact trajectories. For a single Gaussian of
   variance v it is available in closed form,

       J = beta(t) (1 - V(t)) / (2 V(t)) > 0,      V(t) = alpha^2 v + sigma^2

   which diverges like 1/(2t) as t -> 0 (because V ~ beta_min t there), capped at
   beta_min/(2v) once the data's own variance dominates. So the reverse process *is*
   stiff at small t, as the folklore says.

   **The sign matters and it is easy to lose.** Sampling integrates backwards, h < 0.
   Where J > 0 the mode is contracting, hJ < 0, and explicit Euler's factor 1 + hJ
   leaves the unit disk once |h| J > 2 -- the classical restriction. Where J < 0 (which
   happens between the modes of a mixture, since J = -beta/2 - (beta/2) d^2 log p/dx^2
   and log p is convex in the trough) the step is *expanding*, 1 + hJ > 1, and there is
   no stability bound to violate: the exact linearized factor exp(hJ) is larger still.

   So both the signed extremes are recorded, not just max |h J|. The measurement, on
   both priors and both grids and at every step count from 4 to 128: min(1 + hJ) never
   goes below +0.18. **Explicit Euler is never outside its stability region on this
   schedule.** The 1/(2t) growth of J is exactly cancelled by the fact that a monotone
   grid ending at t_eps cannot take a step larger than t.

   What is there instead, on the sharply multimodal prior at 8 steps, is the opposite
   failure: max h J = +4.7, so one step must reproduce an expansion of exp(4.7) = 106
   and explicit Euler reproduces 1 + 4.7 = 5.7 -- a 19x under-estimate, in a single
   step. That is an accuracy catastrophe, not an instability, and it is why the fitted
   orders on the sharp prior (analysis/convergence_order.py) are meaningless at those
   step counts. Going looking for the step-size restriction the folklore promises and
   reporting that it is not the binding constraint is the finding here.

2. `amplification` -- the stability functions, exactly.
   For a Gaussian prior the flow is affine and every scheme's step is a multiplication
   of x - alpha mu by a scalar:

       exact        R = sqrt(V_t / V_s)
       Euler        R = 1 + h beta(1 - V_s)/(2 V_s)
       DPM-Solver-1 R = (alpha_t alpha_s v + sigma_t sigma_s) / V_s

   Subtracting squares (src/samplers/exponential.py) gives
   R_exact^2 - R_dpm1^2 = v (alpha_t sigma_s - sigma_t alpha_s)^2 / V_s^2 >= 0, and
   R_dpm1 > 0 term by term. So the exponential integrator's amplification factor lies
   in (0, R_exact] for *every* step size: unconditionally stable, never oscillating,
   never overshooting. Euler's leaves [-1, 1] as soon as |h| dF/dx > 2. That is the
   whole stability argument, and it is checked numerically here rather than asserted.

3. `sharpness` -- and yet unconditional stability is not accuracy.
   DPM-Solver-1 freezes the noise prediction eps over a step; it is exact exactly when
   eps is constant along the trajectory, which happens when the data is a point mass.
   So its advantage should grow as the prior narrows. Sweeping the prior variance at a
   fixed step count, against the exact affine solution, locates the crossover: for
   v > ~0.01 plain Euler on the probability-flow ODE is *more* accurate than
   DPM-Solver-1, and below it DPM-Solver-1 wins, reaching machine precision at v = 0.
   Real image data sits far down that axis -- the data manifold is thin -- which is why
   the exponential integrator wins in practice and why it does not win on a broad
   Gaussian mixture. Reporting only the case where it wins would be the easy thing to
   do here.
"""

from __future__ import annotations

import csv
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.problem import CANONICAL, SDE, SHARP                            # noqa: E402
from src.reference import (analytic_gaussian_flow, probability_levels,   # noqa: E402
                           quantile_states)
from src.samplers import euler_ode, exponential, heun                    # noqa: E402
from src.samplers.exponential import dpm_solver_1_multiplier             # noqa: E402
from src.schedule import time_grid, uniform_logsnr_grid                  # noqa: E402
from src.sde import (GaussianMixture, make_score,                        # noqa: E402
                     probability_flow_field)

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

STEP_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]
#: The stiffness sweep costs one exact quantile inversion per step, so it is capped
#: lower than the closed-form amplification sweep. By 128 steps the ratio is already
#: an order of magnitude inside the stability bound and still falling like 1/N.
STIFFNESS_STEP_COUNTS = [4, 8, 16, 32, 64, 128]
N_PROBS = 101
FD_H = 1e-6                    # central-difference step for dF/dx: sqrt(eps)-ish, and
                               # the field is smooth, so this sits near the optimum
SHARPNESS_V = [1.0, 0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3, 1e-4, 0.0]
SHARPNESS_STEPS = 16


def field_jacobian(prior, x: np.ndarray, t: float) -> np.ndarray:
    score = make_score(SDE, prior)
    fp = probability_flow_field(SDE, score(x + FD_H, t), x + FD_H, t)
    fm = probability_flow_field(SDE, score(x - FD_H, t), x - FD_H, t)
    return (fp - fm) / (2.0 * FD_H)


def stiffness_rows(rows: list[dict]) -> None:
    probs = probability_levels(N_PROBS)
    for prior_name, prior in (("canonical", CANONICAL), ("sharp", SHARP)):
        for grid_kind in ("uniform_t", "uniform_logsnr"):
            for n in STIFFNESS_STEP_COUNTS:
                grid = time_grid(grid_kind, SDE, n)
                worst = 0.0
                worst_t = float("nan")
                lo, hi, exact_hi = 1.0, 1.0, 1.0
                for i in range(grid.size - 1):
                    t, h = float(grid[i]), float(grid[i + 1] - grid[i])
                    x = quantile_states(SDE, prior, probs, t)
                    hj = h * field_jacobian(prior, x, t)
                    ratio = float(np.abs(hj).max())
                    if ratio > worst:
                        worst, worst_t = ratio, t
                    lo = min(lo, float((1.0 + hj).min()))
                    hi = max(hi, float((1.0 + hj).max()))
                    exact_hi = max(exact_hi, float(np.exp(hj).max()))
                rows.append({
                    "section": "stiffness", "prior": prior_name, "grid": grid_kind,
                    "n_steps": n, "prior_variance": "",
                    "stiffness_ratio": round(worst, 4), "at_t": round(worst_t, 6),
                    "min_euler_factor": round(lo, 6), "max_euler_factor": round(hi, 4),
                    "max_frozen_exact_factor": round(exact_hi, 4),
                    # |1 + hJ| > 1 with hJ < 0 is the classical restriction; hJ > 0 is
                    # an expanding step, which is not a stability failure.
                    "euler_stability_violated": int(lo < -1.0),
                    "max_amp_euler": "", "max_amp_exponential": "", "max_amp_exact": "",
                    "err_euler": "", "err_heun": "", "err_dpm1": "", "err_dpm2": "",
                    "euler_over_dpm1": "",
                    "min_amp_euler": "", "min_amp_exponential": "",
                    "amp_rel_err_euler": "", "amp_rel_err_exponential": "",
                    "exponential_within_exact": "",
                })
            recent = rows[-len(STIFFNESS_STEP_COUNTS):]
            print(f"  {prior_name:9s} {grid_kind:14s} max|hJ| "
                  + " ".join(f"N={r['n_steps']}:{r['stiffness_ratio']:.2f}"
                             for r in recent))
            print(f"  {'':9s} {'':14s} min(1+hJ) "
                  + " ".join(f"{r['min_euler_factor']:+.2f}" for r in recent)
                  + "   (stability needs > -1: "
                  + f"violated {sum(r['euler_stability_violated'] for r in recent)}"
                  + f"/{len(recent)})")


def amplification_rows(rows: list[dict], variance: float = 1e-3) -> None:
    """Closed-form amplification factors per step, for a Gaussian prior.

    For each grid the per-step multiplier of x - alpha mu is a scalar, so the three
    schemes can be compared exactly. Reported per step count:

      max_amp_euler        max |R_euler| over the grid -- must stay <= 1 to be stable
      min_amp_euler        the signed minimum: negative means the step has overshot
      max_amp_exponential  max R_dpm1, which is positive and <= R_exact by construction
      amp_rel_err_*        max over steps of |R_scheme - R_exact| / R_exact

    n_steps = 1 is the whole interval in a single step, deliberately absurd: it is the
    strongest possible test of the claim that R_dpm1 stays in (0, R_exact] for *any*
    step size.
    """
    for n in STEP_COUNTS:
        grid = uniform_logsnr_grid(SDE, n)
        amp_e, amp_x, amp_d = [], [], []
        for i in range(grid.size - 1):
            t0, t1 = float(grid[i]), float(grid[i + 1])
            a0, s0 = float(SDE.alpha(t0)), float(SDE.sigma(t0))
            v0 = a0 * a0 * variance + s0 * s0
            jac = float(SDE.beta(t0)) * (1.0 - v0) / (2.0 * v0)
            amp_e.append(1.0 + (t1 - t0) * jac)
            r_exact, r_dpm = dpm_solver_1_multiplier(SDE, variance, t0, t1)
            amp_x.append(r_exact)
            amp_d.append(r_dpm)
        amp_e, amp_x, amp_d = np.array(amp_e), np.array(amp_x), np.array(amp_d)
        rows.append({
            "section": "amplification", "prior": "gaussian", "grid": "uniform_logsnr",
            "n_steps": n, "prior_variance": variance,
            "stiffness_ratio": "", "at_t": "", "min_euler_factor": "",
            "max_euler_factor": "", "max_frozen_exact_factor": "",
             "euler_stability_violated": "",
            "max_amp_euler": round(float(np.abs(amp_e).max()), 6),
            "min_amp_euler": round(float(amp_e.min()), 6),
            "max_amp_exponential": round(float(amp_d.max()), 6),
            "min_amp_exponential": round(float(amp_d.min()), 6),
            "max_amp_exact": round(float(amp_x.max()), 6),
            "amp_rel_err_euler": round(float(np.abs(amp_e / amp_x - 1.0).max()), 8),
            "amp_rel_err_exponential": round(float(np.abs(amp_d / amp_x - 1.0).max()), 8),
            "exponential_within_exact": int(np.all((amp_d > 0) & (amp_d <= amp_x + 1e-15))),
            "err_euler": "", "err_heun": "", "err_dpm1": "", "err_dpm2": "",
            "euler_over_dpm1": "",
        })
        r = rows[-1]
        print(f"  N={n:<5d} euler max|R|={r['max_amp_euler']:.4f} "
              f"min R={r['min_amp_euler']:+.4f} rel.err {r['amp_rel_err_euler']:.2e} | "
              f"exponential max R={r['max_amp_exponential']:.4f} "
              f"rel.err {r['amp_rel_err_exponential']:.2e} "
              f"in (0, R_exact]: {bool(r['exponential_within_exact'])}")


def sharpness_rows(rows: list[dict]) -> None:
    """Error at fixed NFE against the exact affine solution, as the prior narrows."""
    standard_normal = GaussianMixture(np.array([1.0]), np.array([[0.0]]), np.array([1.0]))
    x0 = standard_normal.stratified(1024)
    mu = np.array([[0.7]])
    grid = uniform_logsnr_grid(SDE, SHARPNESS_STEPS)
    for v in SHARPNESS_V:
        prior = GaussianMixture(np.array([1.0]), mu, np.array([v]))
        score = make_score(SDE, prior)
        exact = analytic_gaussian_flow(SDE, mu, v, x0, SDE.t_max, SDE.t_min)

        def err(fn):
            return float(np.sqrt(np.mean((fn(score, SDE, x0, grid).x - exact) ** 2)))

        e_eu = err(euler_ode)
        e_he = err(heun)
        e_d1 = err(lambda s, sde, x, g: exponential(s, sde, x, g, order=1))
        e_d2 = err(lambda s, sde, x, g: exponential(s, sde, x, g, order=2))
        rows.append({
            "section": "sharpness", "prior": "gaussian", "grid": "uniform_logsnr",
            "n_steps": SHARPNESS_STEPS, "prior_variance": v,
            "stiffness_ratio": "", "at_t": "", "min_euler_factor": "",
            "max_euler_factor": "", "max_frozen_exact_factor": "",
             "euler_stability_violated": "",
            "max_amp_euler": "", "max_amp_exponential": "", "max_amp_exact": "",
            "err_euler": e_eu, "err_heun": e_he, "err_dpm1": e_d1, "err_dpm2": e_d2,
            "euler_over_dpm1": round(e_eu / e_d1, 4) if e_d1 > 0 else "inf",
            "min_amp_euler": "", "min_amp_exponential": "",
                    "amp_rel_err_euler": "", "amp_rel_err_exponential": "",
                    "exponential_within_exact": "",
                })
        print(f"  sharpness v={v:<8g} euler {e_eu:.3e}  heun {e_he:.3e}  "
              f"dpm1 {e_d1:.3e}  dpm2 {e_d2:.3e}   euler/dpm1 "
              f"{rows[-1]['euler_over_dpm1']}")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    rows: list[dict] = []
    print("stiffness ratio max |h dF/dx| along the exact trajectories:")
    stiffness_rows(rows)
    print("\namplification factors (exact, closed form):")
    amplification_rows(rows)
    print("\nerror at 16 steps as the prior narrows toward a point mass:")
    sharpness_rows(rows)

    out = RESULTS / "stability.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
