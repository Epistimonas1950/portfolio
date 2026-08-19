#!/usr/bin/env python3
r"""Empirical order of convergence -- the credibility anchor of the whole repo.

Halve the step size, measure the error against the *exact* solution, fit the slope of
log(error) against log(h) by least squares. A first-order method must give 1, a
second-order method 2. Nothing about this can be made to come out right by accident:
an implementation with the wrong sign, the wrong schedule derivative or a misplaced
factor of two still runs, still produces plausible samples, and shows the wrong slope.

Three measurements, all written to results/convergence.csv:

1. ODE trajectory error. Initial conditions are the states carrying probability levels
   p_i at t = T; the exact answer at t = t_eps is the state carrying the same level,
   because the 1-D probability-flow map is the quantile map (derived in
   src/reference.py). Error is the RMS over levels. No reference solver is involved,
   so nothing here is limited by the reference's own accuracy.

2. Euler-Maruyama strong order, with common random numbers: one Brownian path is drawn
   on the finest grid and *summed* into every coarser grid, so all resolutions are
   driven pathwise by the same path. Additive noise makes the Milstein correction
   vanish, so the expected answer is 1, not the 1/2 usually quoted.

3. Euler-Maruyama weak order, on E[X^2]. E[X] is also measured and also converges at
   order 1, but its error constant here is ~30x smaller and sits at the Monte-Carlo
   floor at this path count -- both are reported with their floors so the reader can
   see which numbers carry information.

The fit window is stated in the CSV (fit_n_min / fit_n_max) and excludes the coarsest
grids, which are pre-asymptotic. The `sharp` prior rows exist to show what happens
when that discipline is dropped: on a badly conditioned problem the same code, over
the same range of h, fits slopes well below the true orders.
"""

from __future__ import annotations

import csv
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.problem import CANONICAL, SDE, SHARP, flow_map_condition       # noqa: E402
from src.reference import probability_levels, quantile_states           # noqa: E402
from src.samplers import (brownian_increments, coarsen_increments,      # noqa: E402
                          euler_maruyama)
from src.samplers import ODE_SAMPLERS                                   # noqa: E402
from src.schedule import time_grid, uniform_time_grid                   # noqa: E402
from src.sde import make_score                                          # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

N_LEVELS = [8, 16, 32, 64, 128, 256, 512, 1024]
FIT_FROM = 32                  # coarser grids are pre-asymptotic; see the README
N_PROBS = 257                  # probability levels = trajectories tracked

# --- SDE budget. n_paths x n_ref_steps doubles of Brownian increments must fit in
# --- memory: 8192 x 2048 x 8 B = 134 MB, which is the binding constraint here.
SDE_PATHS = 8192
SDE_REF_STEPS = 2048
SDE_LEVELS = [8, 16, 32, 64, 128]


def fit_loglog(h: np.ndarray, err: np.ndarray) -> tuple[float, float]:
    """Least-squares slope of log(err) vs log(h), and the RMS residual in log10.

    The residual is the honest companion to the slope: a slope of 2.0 with a residual
    of 0.3 decades means the points are not on a line and the number means nothing.
    """
    lx, ly = np.log(np.asarray(h)), np.log(np.asarray(err))
    slope, intercept = np.polyfit(lx, ly, 1)
    resid = ly - (slope * lx + intercept)
    return float(slope), float(np.sqrt(np.mean(resid**2)) / np.log(10.0))


def pairwise_orders(err: list[float]) -> list[float]:
    """log2(e_h / e_{h/2}) per interval -- the settling table."""
    return [float("nan")] + [float(np.log2(err[i - 1] / err[i])) for i in range(1, len(err))]


def ode_convergence(prior, prior_name: str, grid_kind: str, rows: list[dict]) -> None:
    score = make_score(SDE, prior)
    probs = probability_levels(N_PROBS)
    x_start = quantile_states(SDE, prior, probs, SDE.t_max)
    x_exact = quantile_states(SDE, prior, probs, SDE.t_min)
    cond = flow_map_condition(SDE, prior)

    for name, (sampler, nfe_per_step) in ODE_SAMPLERS.items():
        errs, hs, nfes = [], [], []
        for n in N_LEVELS:
            grid = time_grid(grid_kind, SDE, n)
            res = sampler(score, SDE, x_start, grid)
            errs.append(float(np.sqrt(np.mean((res.x - x_exact) ** 2))))
            hs.append(float(np.max(np.abs(np.diff(grid)))))
            nfes.append(res.nfe)
        keep = [i for i, n in enumerate(N_LEVELS) if n >= FIT_FROM]
        slope, resid = fit_loglog(np.array(hs)[keep], np.array(errs)[keep])
        orders = pairwise_orders(errs)
        for i, n in enumerate(N_LEVELS):
            rows.append({
                "prior": prior_name, "mode": "ode_trajectory", "grid": grid_kind,
                "sampler": name, "n_steps": n, "h_max": round(hs[i], 8),
                "nfe": nfes[i], "error": errs[i], "mc_floor": "",
                "pairwise_order": "" if i == 0 else round(orders[i], 4),
                "fitted_slope": round(slope, 4), "fit_residual_decades": round(resid, 5),
                "fit_n_min": FIT_FROM, "fit_n_max": N_LEVELS[-1],
                "flow_map_condition": round(cond, 4),
            })
        print(f"  {prior_name:9s} {grid_kind:14s} {name:14s} slope {slope:5.3f} "
              f"(residual {resid:.3f} decades)")


def sde_convergence(rows: list[dict]) -> None:
    """Strong and weak order of Euler-Maruyama, with common random numbers."""
    score = make_score(SDE, CANONICAL)
    rng = np.random.default_rng(20240418)
    # Stratified start: the exact midpoint quantiles of p_T, so the initial ensemble
    # carries no Monte-Carlo error of its own and every level starts identically.
    x0 = SDE.marginal(CANONICAL, SDE.t_max).stratified(SDE_PATHS)
    fine = uniform_time_grid(SDE, SDE_REF_STEPS)
    dw = brownian_increments(rng, fine, x0.shape)
    ref_p = euler_maruyama(score, SDE, x0, fine, increments=dw).x
    ref_m = euler_maruyama(score, SDE, x0, fine, increments=-dw).x

    strong, weak2, weak1, floor2, floor1, hs, nfes = [], [], [], [], [], [], []
    for n in SDE_LEVELS:
        grid = uniform_time_grid(SDE, n)
        c = coarsen_increments(dw, SDE_REF_STEPS // n)
        a = euler_maruyama(score, SDE, x0, grid, increments=c)
        b = euler_maruyama(score, SDE, x0, grid, increments=-c).x
        strong.append(float(np.sqrt(0.5 * (np.mean((a.x - ref_p) ** 2)
                                           + np.mean((b - ref_m) ** 2)))))
        # Antithetic average of the paired difference: an unbiased estimator of the
        # weak error whose variance is far below that of either path set alone.
        d1 = 0.5 * ((a.x - ref_p) + (b - ref_m))
        d2 = 0.5 * ((a.x**2 - ref_p**2) + (b**2 - ref_m**2))
        weak1.append(float(abs(d1.mean())))
        weak2.append(float(abs(d2.mean())))
        floor1.append(float(d1.std() / np.sqrt(SDE_PATHS)))
        floor2.append(float(d2.std() / np.sqrt(SDE_PATHS)))
        hs.append(float(np.max(np.abs(np.diff(grid)))))
        nfes.append(a.nfe)

    for mode, err, floor in (("sde_strong", strong, [""] * len(strong)),
                             ("sde_weak_second_moment", weak2, floor2),
                             ("sde_weak_mean", weak1, floor1)):
        slope, resid = fit_loglog(np.array(hs), np.array(err))
        orders = pairwise_orders(err)
        for i, n in enumerate(SDE_LEVELS):
            rows.append({
                "prior": "canonical", "mode": mode, "grid": "uniform_t",
                "sampler": "euler_maruyama", "n_steps": n, "h_max": round(hs[i], 8),
                "nfe": nfes[i], "error": err[i],
                "mc_floor": floor[i] if floor[i] == "" else round(floor[i], 10),
                "pairwise_order": "" if i == 0 else round(orders[i], 4),
                "fitted_slope": round(slope, 4), "fit_residual_decades": round(resid, 5),
                "fit_n_min": SDE_LEVELS[0], "fit_n_max": SDE_LEVELS[-1],
                "flow_map_condition": "",
            })
        print(f"  canonical uniform_t      {mode:22s} slope {slope:5.3f} "
              f"(residual {resid:.3f} decades)")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    rows: list[dict] = []

    print("probability-flow ODE, trajectory error against the exact quantile map:")
    for grid_kind in ("uniform_logsnr", "uniform_t"):
        ode_convergence(CANONICAL, "canonical", grid_kind, rows)
    print("\nthe same code on a badly conditioned prior (max |Phi'| = 1.7e4):")
    ode_convergence(SHARP, "sharp", "uniform_logsnr", rows)

    print("\nreverse SDE, Euler-Maruyama, common random numbers:")
    sde_convergence(rows)

    out = RESULTS / "convergence.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
