#!/usr/bin/env python3
r"""Sample quality against network evaluations -- the number an engineering team uses.

Each sampling step is one forward pass, so NFE is the cost. The question is how much
quality each sampler buys per evaluation, and the answer is not "the higher-order
method is better" for free: Heun and DPM-Solver-2 spend two evaluations per step, so
at a fixed NFE budget they take half as many steps as Euler. They win anyway, and by
how much is what this script measures.

Every sampler is given the *same* 8192 initial points and the same grid family
(uniform in log-SNR, which is what production samplers use). The initial points are
the midpoint quantiles of N(0,1) rather than i.i.d. draws: the point of this table is
the sampler's error, and 8192 i.i.d. draws carry a W1 sampling error of 0.024 that
would swamp it. Stratifying the input drops the floor to 4.4e-5 for a deterministic
sampler (measured, and reported as the `exact_map` row). Euler-Maruyama keeps its own
floor near 0.011 no matter what the input is, because it injects fresh randomness --
that is not an artefact, it is the stochastic sampler's variance showing up where it
should, and analysis/stochastic_vs_deterministic.py takes it apart.

Two kinds of error, reported side by side because they answer different questions:

  trajectory RMSE   ||x_sampler - Phi(x_T)||, where Phi is the exact probability-flow
                    map (src/reference.py). Pure discretization error, per sample, no
                    statistical floor at all. This is the number that discriminates
                    between samplers at high NFE.

  W1 / energy /     distributional error against the exact marginal p_{t_eps}. This is
  moments / mode-TV what a practitioner cares about, and it stops improving once the
                    sampler's discretization error falls below the Monte-Carlo floor
                    of 8192 samples. That floor is measured, not asserted: it is the
                    row `exact_map`, produced by pushing the same x_T through Phi.

The `exact_map` row is the floor: the same inputs pushed through the exact flow map
Phi. It is not zero, and the reason is worth stating -- sampling starts from N(0, I)
rather than from the true p_T. At T = 1 the VP schedule has alpha_T = 6.6e-3, so p_T
is very close to N(0, I) but not equal to it, and no number of steps removes the
difference. Here that prior-mismatch floor is W1 = 4.4e-5, three orders below the
errors being compared, but on a schedule truncated at smaller T it would not be.

Euler-Maruyama's trajectory RMSE is left blank on purpose. The SDE does not follow the
ODE trajectory -- it has the same marginals, not the same paths -- so a pathwise
comparison against Phi(x_T) is a category error, not a large error.

Writes results/nfe_quality.csv (every measurement) and results/nfe_targets.csv (the
NFE each sampler needs to reach the target quality, and the speedup over the
first-order deterministic baseline).
"""

from __future__ import annotations

import csv
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import src.metrics as metrics                                            # noqa: E402
from src.problem import CANONICAL, SDE                                   # noqa: E402
from src.reference import exact_flow_map                                 # noqa: E402
from src.samplers import adaptive_heun, euler_maruyama, euler_ode        # noqa: E402
from src.samplers import exponential, heun                               # noqa: E402
from src.schedule import uniform_logsnr_grid                             # noqa: E402
from src.sde import GaussianMixture, make_score                          # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

N_SAMPLES = 8192
SEED = 7
NFE_BUDGETS = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
RTOLS = [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]

#: The headline targets. W1 = 0.02 is about 1.3% of the target distribution's standard
#: deviation, and sits above Euler-Maruyama's irreducible sampling noise (~0.011 at
#: 8192 samples) so that the stochastic sampler can reach it at all; below that the
#: table would be reporting Monte-Carlo noise rather than integration error. The
#: trajectory targets have no statistical floor and are the sharper comparison.
W1_TARGET = 0.02
TRAJ_TARGET = 1e-2
TRAJ_TARGET_TIGHT = 3e-3

FIXED = {
    "euler_maruyama": (1, None),
    "euler_ode": (1, euler_ode),
    "heun": (2, heun),
    "exponential_1": (1, lambda s, sde, x, t: exponential(s, sde, x, t, order=1)),
    "exponential_2": (2, lambda s, sde, x, t: exponential(s, sde, x, t, order=2)),
}


def evaluate(x: np.ndarray, exact: np.ndarray, target, quantiles, target_self) -> dict:
    row = {
        "traj_rmse": float(np.sqrt(np.mean((x - exact) ** 2))),
        "w1": metrics.wasserstein1(x, quantiles),
        "energy": metrics.energy_distance(x, target, target_self),
        "mode_tv": metrics.mode_weight_error(x, target),
    }
    row.update(metrics.moment_errors(x, target))
    return row


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    score = make_score(SDE, CANONICAL)
    # Stratified N(0,1) start: the midpoint quantiles, so the input ensemble itself
    # contributes no Monte-Carlo error and what is left is the sampler's.
    standard_normal = GaussianMixture(np.array([1.0]), np.array([[0.0]]), np.array([1.0]))
    x_start = standard_normal.stratified(N_SAMPLES)

    target = SDE.marginal(CANONICAL, SDE.t_min)
    quantiles = metrics.target_midpoint_quantiles(target, N_SAMPLES)
    target_self = metrics._mean_abs_target(target)
    exact = exact_flow_map(SDE, CANONICAL, x_start, SDE.t_max, SDE.t_min)

    rows: list[dict] = []
    floor = evaluate(exact, exact, target, quantiles, target_self)
    rows.append({"sampler": "exact_map", "nfe": 0, "steps": 0, "rejected": 0,
                 "rtol": "", **{k: round(v, 8) for k, v in floor.items()}})
    print(f"floor (exact flow map, {N_SAMPLES} samples): "
          f"W1={floor['w1']:.5f} energy={floor['energy']:.3e} "
          f"mean_err={floor['mean_error']:.5f} mode_TV={floor['mode_tv']:.4f}")

    for name, (per_step, fn) in FIXED.items():
        for nfe in NFE_BUDGETS:
            n_steps = nfe // per_step
            grid = uniform_logsnr_grid(SDE, n_steps)
            if name == "euler_maruyama":
                # Its own stream, reseeded per budget so the comparison across budgets
                # is not confounded by which draws happened to come first.
                res = euler_maruyama(score, SDE, x_start, grid,
                                     rng=np.random.default_rng(SEED + nfe))
            else:
                res = fn(score, SDE, x_start, grid)
            m = evaluate(res.x, exact, target, quantiles, target_self)
            m = {k: round(v, 8) for k, v in m.items()}
            if name == "euler_maruyama":
                m["traj_rmse"] = ""        # different paths by construction; see above
            rows.append({"sampler": name, "nfe": res.nfe, "steps": res.steps,
                         "rejected": 0, "rtol": "", **m})
        last = rows[-1]
        print(f"  {name:15s} at NFE {last['nfe']:4d}: W1={last['w1']:.5f} "
              f"traj_rmse={last['traj_rmse']}")

    for rtol in RTOLS:
        res = adaptive_heun(score, SDE, x_start, rtol=rtol, atol=rtol / 10.0)
        m = evaluate(res.x, exact, target, quantiles, target_self)
        rows.append({"sampler": "adaptive", "nfe": res.nfe, "steps": res.steps,
                     "rejected": res.rejected, "rtol": rtol,
                     **{k: round(v, 8) for k, v in m.items()}})
    print(f"  {'adaptive':15s} rtol {RTOLS[-1]:.0e}: NFE={rows[-1]['nfe']} "
          f"W1={rows[-1]['w1']:.5f} traj_rmse={rows[-1]['traj_rmse']:.3e} "
          f"(rejected {rows[-1]['rejected']})")

    out = RESULTS / "nfe_quality.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")

    # ---- the headline table -----------------------------------------------------
    def first_nfe(sampler: str, key: str, target_value: float):
        hits = [r for r in rows if r["sampler"] == sampler and r[key] != ""
                and float(r[key]) <= target_value]
        return min((r["nfe"] for r in hits), default=None)

    samplers = [s for s in FIXED] + ["adaptive"]
    keys = [("w1", W1_TARGET, "nfe_for_w1_le_%.3f" % W1_TARGET),
            ("traj_rmse", TRAJ_TARGET, "nfe_for_traj_rmse_le_%.0e" % TRAJ_TARGET),
            ("traj_rmse", TRAJ_TARGET_TIGHT,
             "nfe_for_traj_rmse_le_%.0e" % TRAJ_TARGET_TIGHT)]
    base = {label: first_nfe("euler_ode", key, tgt) for key, tgt, label in keys}
    target_rows = []
    for s in samplers:
        row = {"sampler": s}
        for key, tgt, label in keys:
            n = first_nfe(s, key, tgt)
            row[label] = n if n else "not reached"
            row["speedup_" + label] = (round(base[label] / n, 2)
                                       if (n and base[label]) else "")
        row["w1_floor"] = round(floor["w1"], 6)
        target_rows.append(row)
    out2 = RESULTS / "nfe_targets.csv"
    with out2.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(target_rows[0]))
        writer.writeheader()
        writer.writerows(target_rows)
    print(f"wrote {out2}   ({time.time() - t0:.1f} s)\n")
    for r in target_rows:
        print("  " + "  ".join(f"{k}={v}" for k, v in r.items() if k != "w1_floor"))


if __name__ == "__main__":
    main()
