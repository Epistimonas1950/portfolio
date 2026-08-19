#!/usr/bin/env python3
r"""The SDE/ODE tradeoff, measured on both sides.

The reverse SDE and the probability-flow ODE have *the same marginals* -- that is the
theorem, and it means the usual claim "the SDE gives more diverse samples" cannot be
about the marginal distribution. At convergence both produce p_{t_eps} exactly. What
differs is everything else, and this script measures three things that actually do
differ:

`accuracy`
    Distributional error against the exact marginal, per NFE. The ODE converges in
    NFE; the SDE cannot go below its own Monte-Carlo noise, because it injects fresh
    randomness at every step and a finite sample of a random map has O(n^-1/2) error
    however well the SDE is integrated. Both floors are measured here.

`diversity`
    Conditional spread: fix a single state x_{t_c} at some time t_c and run the sampler
    to t_eps many times. The ODE is a deterministic map, so the answer is one point and
    the spread is exactly zero at every t_c. The SDE returns a distribution, and the
    interesting quantity is how that distribution shrinks as t_c falls: it measures how
    much of the final sample is decided by the state you started from and how much by
    the sampler's own injected noise. Started at t_c = T the reverse SDE forgets its
    initialization completely -- the conditional mode entropy equals the prior's own
    mode entropy, 1.556 bits -- which is exactly why "the SDE is more diverse" is a
    statement about conditional, never marginal, distributions. Both samplers target
    the same marginal by construction.

`invertibility`
    The ODE map is invertible: integrate T -> t_eps and then t_eps -> T and you get
    back where you started, to the integrator's accuracy. That is what makes
    ODE samplers usable for inversion, latent interpolation and editing. The SDE has no
    such inverse, so there is no number in that column -- it is left blank rather than
    filled with something meaningless.

Writes results/sde_vs_ode.csv.
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
from src.samplers import euler_maruyama, euler_ode, heun                 # noqa: E402
from src.schedule import uniform_logsnr_grid                             # noqa: E402
from src.sde import GaussianMixture, make_score                          # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

N_SAMPLES = 8192
NFE_BUDGETS = [16, 32, 64, 128, 256, 512]
SEED = 11

#: Conditioning points for the diversity measurement, as quantiles of the N(0,1) start
#: distribution. 0.5 sits between the two dominant basins of the canonical prior, which
#: is where the stochastic sampler's mode reassignment is visible at all.
#: Conditioning levels, as quantiles of the marginal at the conditioning time.
COND_LEVELS = [0.2, 0.5, 0.8]
#: Conditioning times: where the fixed state is held. t_c = 1.0 is ordinary sampling.
COND_TIMES = [1.0, 0.6, 0.4, 0.25, 0.15, 0.08, 0.04, 0.02]
COND_REPEATS = 2048
COND_NFE = 128

#: One flat schema for three different measurements, so the CSV has a single header and
#: a blank means "this measurement does not have that column" rather than "zero".
FIELDS = ["section", "sampler", "nfe", "w1", "energy", "mode_tv", "conditioning_time",
          "prob_level", "conditional_std", "mode_entropy_bits", "n_modes_reached",
          "roundtrip_rmse"]


def row(**kw) -> dict:
    out = {k: "" for k in FIELDS}
    unknown = set(kw) - set(FIELDS)
    if unknown:
        raise KeyError(f"unknown result columns {sorted(unknown)}")
    out.update(kw)
    return out


def accuracy_rows(rows: list[dict]) -> None:
    score = make_score(SDE, CANONICAL)
    standard_normal = GaussianMixture(np.array([1.0]), np.array([[0.0]]), np.array([1.0]))
    x_start = standard_normal.stratified(N_SAMPLES)
    target = SDE.marginal(CANONICAL, SDE.t_min)
    quantiles = metrics.target_midpoint_quantiles(target, N_SAMPLES)
    target_self = metrics._mean_abs_target(target)

    # The floor a *perfect* stochastic sampler would still have: 8192 exact draws.
    rng = np.random.default_rng(SEED)
    w1_floor = float(np.mean([metrics.wasserstein1(target.sample(rng, N_SAMPLES), quantiles)
                              for _ in range(5)]))
    rows.append(row(section="accuracy", sampler="exact_iid_samples", nfe=0,
                    w1=round(w1_floor, 8)))
    print(f"  Monte-Carlo floor of any stochastic sampler ({N_SAMPLES} draws): "
          f"W1 = {w1_floor:.5f}")

    for name, fn in (("euler_maruyama", None), ("euler_ode", euler_ode), ("heun", heun)):
        per_step = 2 if name == "heun" else 1
        for nfe in NFE_BUDGETS:
            grid = uniform_logsnr_grid(SDE, nfe // per_step)
            if fn is None:
                res = euler_maruyama(score, SDE, x_start, grid,
                                     rng=np.random.default_rng(SEED + nfe))
            else:
                res = fn(score, SDE, x_start, grid)
            rows.append(row(
                section="accuracy", sampler=name, nfe=res.nfe,
                w1=round(metrics.wasserstein1(res.x, quantiles), 8),
                energy=round(metrics.energy_distance(res.x, target, target_self), 10),
                mode_tv=round(metrics.mode_weight_error(res.x, target), 6)))
        print(f"  {name:15s} W1 at NFE {NFE_BUDGETS[0]}..{NFE_BUDGETS[-1]}: "
              + " ".join(f"{r['w1']:.5f}" for r in rows[-len(NFE_BUDGETS):]))


def diversity_rows(rows: list[dict]) -> None:
    """How much of the sample is decided by x_{t_c}, and how much by the sampler?"""
    score = make_score(SDE, CANONICAL)
    target = SDE.marginal(CANONICAL, SDE.t_min)
    w = CANONICAL.weights
    marginal_entropy = float(-(w * np.log2(w)).sum())
    rows.append(row(section="diversity", sampler="prior_marginal", nfe=0,
                    conditional_std=round(float(np.sqrt(target.moments()[1])), 6),
                    mode_entropy_bits=round(marginal_entropy, 6),
                    n_modes_reached=target.n_components))
    print(f"  the prior itself: sd {np.sqrt(target.moments()[1]):.4f}, "
          f"mode entropy {marginal_entropy:.3f} bits -- the ceiling for any conditional")

    for t_c in COND_TIMES:
        x_cond = SDE.marginal(CANONICAL, t_c).quantile(np.array(COND_LEVELS))
        grid = uniform_logsnr_grid(SDE, COND_NFE, t_start=t_c, t_end=SDE.t_min)
        batch = np.repeat(x_cond, COND_REPEATS).reshape(-1, 1)
        sde_out = euler_maruyama(score, SDE, batch, grid,
                                 rng=np.random.default_rng(SEED + int(1e4 * t_c))).x
        ode_out = euler_ode(score, SDE, batch, grid).x
        for name, out in (("euler_maruyama", sde_out), ("euler_ode", ode_out)):
            blocks = out.reshape(len(COND_LEVELS), COND_REPEATS)
            for level, block in zip(COND_LEVELS, blocks):
                k = target.responsibilities(block.reshape(-1, 1)).argmax(axis=1)
                pk = np.bincount(k, minlength=target.n_components) / block.size
                nz = pk[pk > 0]
                entropy = max(0.0, float(-(nz * np.log2(nz)).sum()))
                rows.append(row(
                    section="diversity", sampler=name, nfe=COND_NFE,
                    conditioning_time=t_c, prob_level=level,
                    conditional_std=round(float(block.std()), 8),
                    mode_entropy_bits=round(entropy, 6),
                    n_modes_reached=int((pk > 0).sum())))
        recent = rows[-2 * len(COND_LEVELS):]
        sde_rows = [r for r in recent if r["sampler"] == "euler_maruyama"]
        print(f"  t_c={t_c:<5g} SDE conditional sd "
              + "/".join(f"{r['conditional_std']:.3f}" for r in sde_rows)
              + "  entropy "
              + "/".join(f"{r['mode_entropy_bits']:.2f}" for r in sde_rows)
              + " bits   | ODE sd 0 exactly")


def invertibility_rows(rows: list[dict]) -> None:
    score = make_score(SDE, CANONICAL)
    standard_normal = GaussianMixture(np.array([1.0]), np.array([[0.0]]), np.array([1.0]))
    x_start = standard_normal.stratified(2048)
    for nfe in [16, 32, 64, 128, 256]:
        down = uniform_logsnr_grid(SDE, nfe // 2)
        up = uniform_logsnr_grid(SDE, nfe // 2, t_start=SDE.t_min, t_end=SDE.t_max)
        x_low = heun(score, SDE, x_start, down).x
        x_back = heun(score, SDE, x_low, up).x
        rows.append(row(
            section="invertibility", sampler="heun", nfe=2 * nfe,
            roundtrip_rmse=round(float(np.sqrt(np.mean((x_back - x_start) ** 2))), 10)))
        print(f"  Heun round trip T -> t_eps -> T at {2 * nfe} NFE: "
              f"RMSE {rows[-1]['roundtrip_rmse']:.3e}")
    # roundtrip_rmse deliberately blank: the stochastic map has no inverse.
    rows.append(row(section="invertibility", sampler="euler_maruyama"))


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    rows: list[dict] = []
    print("accuracy per NFE against the exact marginal:")
    accuracy_rows(rows)
    print("\nconditional diversity: many runs from one fixed x_T:")
    diversity_rows(rows)
    print("\ninvertibility of the deterministic map:")
    invertibility_rows(rows)

    out = RESULTS / "sde_vs_ode.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
