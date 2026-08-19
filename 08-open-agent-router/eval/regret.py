#!/usr/bin/env python3
"""Cumulative regret against T, log-log, and the exponent the theory predicts.

This is the file the project's headline claim comes out of, so it is worth being exact
about what is being verified, because the obvious experiment does not verify it.

The obvious experiment, and why it does not work
-----------------------------------------------
Fix a linear bandit instance, run LinUCB, plot cumulative regret on log-log axes, fit a
line, announce the slope. Do that here and you get whatever you want. The `fixed_instance`
block below measures the local slope of one instance over five sliding windows and it
falls monotonically:

    [100, 1000]      0.828
    [316, 3160]      0.742
    [1000, 10000]    0.623
    [3160, 31600]    0.456
    [6400, 64000]    0.377

A fixed instance has a smallest gap Delta, and once the algorithm has resolved it the
regret goes gap-dependent and logarithmic. Every fixed instance eventually leaves the
sqrt(T) regime; a curve bending from slope 1 towards slope 0 passes through 0.5 on the
way, and by choosing d, sigma and the fit window you can land on any exponent in (0,1).
Window [3160, 31600] gives 0.469 and would let you announce "the theory is confirmed".
The next window along gives 0.396. That is why this block is reported rather than hidden:
it is the reason the headline experiment is set up the way it is.

The experiment that does work
-----------------------------
Otilde(d sqrt(T)) is a **minimax** bound: a statement about the worst instance at each
horizon, not about any particular instance. So measure a supremum. Take a family of
instances indexed by a gap scale Delta, and at each horizon T report

    R*(T)  =  max over Delta of  E[ cumulative regret of the policy at horizon T ].   (1)

For a policy that learns, R(T; Delta) ~ min( c1 Delta T,  c2 d sqrt(T) ) -- linear while
the gap is unresolved, flat once it is -- and the max over Delta sits exactly where the
two meet, at Delta* ~ d/sqrt(T), giving R*(T) ~ sqrt(T). For a policy that does not
learn, R(T; Delta) = c Delta T for every Delta, so the max is attained at the largest
gap in the family and R*(T) ~ T. Same protocol, same family, same code: exponent 1/2 for
LinUCB, exponent 1 for random. That is the control, and it is what makes the number a
measurement of the policy rather than an artefact of the fitting.

The diagnostic that the supremum is real is that the maximising Delta is *interior* to
the grid and decreases like 1/sqrt(T); both are written to the CSV. For the random
policy the maximiser sits on the upper edge of the family by construction -- its regret
is c * Delta * T, monotone in Delta, so the sup is at the boundary. That is correct
behaviour and not a truncated grid: moving the boundary rescales the prefactor and
leaves the exponent at 1.

A result that was not the plan: LinGreedy -- the identical ridge estimator with the
exploration bonus switched off -- also comes out near 1/2 on this family, not near 1.
The reason is covariate diversity. Contexts here are drawn i.i.d. from the unit sphere,
so they explore the parameter space *for* the policy, and a greedy learner is
self-exploring; greedy's linear worst case needs instances that starve it of that
diversity. So on this family the exploration bonus is not what buys the exponent, and
saying so is more useful than quietly dropping the row. The place where the bonus does
pay is the fleet, where the linear model is misspecified: LinGreedy's mean reward there
is 0.652 against LinUCB's 0.748 (results/pareto.csv). Two measurements, one conclusion --
exploration earns its keep when the model is wrong, not merely when the problem is
stochastic.

Outputs (results/regret.csv), three experiments under one schema:

    minimax     the envelope (1) for LinUCB, Thompson, LinGreedy and random
    fixed_instance   local slopes of a single instance, showing the drift
    fleet       cumulative regret on the simulated fleet, where the linear model is
                misspecified and the exponent is correspondingly worse
"""

from __future__ import annotations

import csv
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.workload import (LinearBandit, loglog_slope,               # noqa: E402
                           run_fleet, run_linear_bandit)
from src.features import N_FEATURES                                   # noqa: E402
from src.fleet.simulator import LAMBDA, make_workload                 # noqa: E402
from src.routers.baselines import RandomRouter, ThresholdRouter        # noqa: E402
from src.routers.linucb import LinGreedy, LinUCB                       # noqa: E402
from src.routers.thompson import LinearThompson                        # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

# --- the instance family -----------------------------------------------------------
# d = 20 and sigma = 1.0: the dimension has to be large enough and the noise loud enough
# that the sqrt(T) stretch is wide in log-t at the horizons we can afford. K = 5 arms.
D = 20
N_ARMS = 5
SIGMA = 1.0
HORIZON = 64_000
# Five seeds, not three. The envelope is a maximum over 12 noisy curves, and a maximum
# over noisy things is both biased upward and jittery: at 2-3 seeds the fitted exponent
# moves by +-0.1 depending on which instance family is drawn, at 5 it moves by +-0.025.
# That is a fix to the experiment rather than to the tolerance, and it is the same fix
# applied in tests/test_bandits.py.
N_SEEDS = 5
# Gap scales spanning nearly three decades. The upper end matters as much as the lower:
# at small T the worst instance in the family is a *large*-gap one (regret Delta*T before
# the gap is resolved), and a grid that stops too low would truncate the supremum and
# bias the exponent upward.
GAP_SCALES = np.logspace(-2.0, 0.9, 12)
# Fit window. The lower end is where every instance in the family has left its transient
# and the envelope is smooth; the upper end is the horizon. Stated here, in the CSV, and
# in the README, because a log-log slope without its window is not a number.
FIT_LO, FIT_HI = 2_000, HORIZON

# The fleet block runs twenty times longer than the Pareto table (400k vs 20k queries)
# for one reason: at 20k the frozen difficulty-threshold baseline is ahead of LinUCB, and
# the interesting question is whether it stays ahead. A frozen policy accrues a constant
# expected regret per query, so its cumulative regret is linear in t; a learning policy's
# is sublinear, so the curves should cross. At 120k they had not yet, and extrapolating
# the two fitted exponents put the crossing near 150k -- so the horizon was extended
# until the answer is a measurement rather than an extrapolation. Whether it crosses at
# all is a real question and not a formality: the linear model is *misspecified* on this
# fleet, and a misspecified model's regret eventually turns linear too, at which point
# the learner stops catching up. Running long enough to see which happens is the point.
FLEET_QUERIES = 400_000
# A second, longer block restricted to the two policies whose comparison is the point:
# the learner and the frozen baseline. 1.2M queries with only those two costs a fraction
# of what running the whole set that far would, and it buys the one thing a single fit
# cannot show -- whether LinUCB's exponent is a number or a moving target.
LONG_QUERIES = 1_200_000
LONG_WINDOWS = (40_000, 120_000, 400_000, 1_200_000)
FLEET_SEEDS = 6

# Exploration constants, chosen on five TUNING workloads (seeds 1400-1404) that are
# disjoint from the evaluation seeds below. The reward surface is flat: LinUCB scores
# 0.7454 / 0.7471 / 0.7482 / 0.7477 at alpha = 0.4 / 0.8 / 1.0 / 1.2, so alpha = 1.0 is
# the middle of a plateau rather than a sharp optimum, and it happens to coincide with
# the value used in the linear-bandit experiment. Reporting this rather than a bare
# constant matters, because the difficulty-threshold baseline below is also fitted and
# a comparison between a tuned method and an untuned one is not a comparison.
LINUCB_ALPHA = 1.0
THOMPSON_V = 0.35


def _factories():
    """The four policies run through the minimax protocol.

    LinGreedy is here as a second control alongside random, and it is a more
    interesting one: it carries the identical ridge estimator to LinUCB and differs
    only in dropping the exploration bonus. Greedy has instances on which it commits to
    a wrong arm and never gathers the data that would correct it, so its *minimax*
    regret is linear even though on many individual instances it does well. If it comes
    out near 1 alongside random, the exploration bonus -- and not the linear model -- is
    what buys the 1/2.

    alpha = 1.0 here is the linear-bandit setting, separate from LINUCB_ALPHA below,
    which is the constant used on the fleet. They coincide numerically; they are not
    the same choice and are not tied together.
    """
    return {
        "LinUCB": lambda k, d: LinUCB(k, d, alpha=1.0),
        "Thompson": lambda k, d: LinearThompson(k, d, v=0.5, seed=17),
        "LinGreedy": lambda k, d: LinGreedy(k, d),
        "random": lambda k, d: RandomRouter(k, seed=23),
    }


def build_envelope(factory, d: int = D, n_arms: int = N_ARMS, sigma: float = SIGMA,
                   gaps: np.ndarray = GAP_SCALES, horizon: int = HORIZON,
                   n_seeds: int = N_SEEDS,
                   seed0: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Equation (1) evaluated on a grid: the sup over the family at every horizon.

    Returns (envelope, index of the maximising gap at each t). Factored out of the
    reporting code so tests/test_bandits.py exercises this function rather than a
    reimplementation of it -- a test of a copy of the experiment is a test of the copy.
    """
    curves = np.stack([
        np.mean([run_linear_bandit(factory,
                                   LinearBandit(d, n_arms, sigma, g, seed=seed0 + s),
                                   horizon, seed=10 * seed0 + s)
                 for s in range(n_seeds)], axis=0)
        for g in gaps])
    return curves.max(axis=0), curves.argmax(axis=0)


def minimax_envelope(rows: list[dict]) -> dict:
    """Equation (1): sup over the instance family, at every horizon."""
    summary = {}
    for name, factory in _factories().items():
        t0 = time.perf_counter()
        env, argmax = build_envelope(factory)
        slope, r2, pref = loglog_slope(env, FIT_LO, FIT_HI)
        summary[name] = (slope, r2, pref)
        for t in np.unique(np.round(np.logspace(2, np.log10(HORIZON), 48)).astype(int)):
            rows.append({
                "experiment": "minimax", "policy": name, "t": int(t),
                "cum_regret": round(float(env[t - 1]), 6),
                "argmax_gap_scale": round(float(GAP_SCALES[argmax[t - 1]]), 5),
                "argmax_interior": int(0 < argmax[t - 1] < len(GAP_SCALES) - 1),
                "fitted_slope": round(slope, 4), "fit_r2": round(r2, 5),
                "fit_prefactor": round(pref, 4), "fit_lo": FIT_LO, "fit_hi": FIT_HI,
                "n_seeds": N_SEEDS, "d": D, "n_arms": N_ARMS, "sigma": SIGMA,
            })
        print(f"  minimax {name:10s} slope={slope:.3f} R2={r2:.4f} "
              f"pref={pref:.3f}  [{time.perf_counter() - t0:.0f}s]")
    return summary


def fixed_instance_drift(rows: list[dict]) -> None:
    """The control experiment: one instance, and the slope you get depends on the window."""
    inst = LinearBandit(D, N_ARMS, SIGMA, gap_scale=1.0, seed=100)
    curve = np.mean([run_linear_bandit(_factories()["LinUCB"], inst, HORIZON,
                                       seed=1000 + s) for s in range(N_SEEDS)], axis=0)
    windows = [(100, 1_000), (316, 3_160), (1_000, 10_000), (3_160, 31_600),
               (6_400, 64_000)]
    for lo, hi in windows:
        slope, r2, pref = loglog_slope(curve, lo, hi, n_points=40)
        rows.append({
            "experiment": "fixed_instance", "policy": "LinUCB", "t": hi,
            "cum_regret": round(float(curve[hi - 1]), 6), "argmax_gap_scale": 1.0,
            "argmax_interior": "", "fitted_slope": round(slope, 4),
            "fit_r2": round(r2, 5), "fit_prefactor": round(pref, 4),
            "fit_lo": lo, "fit_hi": hi, "n_seeds": N_SEEDS, "d": D,
            "n_arms": N_ARMS, "sigma": SIGMA,
        })
        print(f"  fixed-instance LinUCB local slope over [{lo},{hi}] = {slope:.3f}")


def fleet_regret(rows: list[dict]) -> None:
    """The same measurement on the simulated fleet, where the linear model is wrong."""
    from src.fleet.simulator import expected_reward_matrix
    tune = make_workload(6_000, seed=901)
    thresh = ThresholdRouter.fit(tune.difficulty_score,
                                 expected_reward_matrix(tune, LAMBDA),
                                 score_index=6)

    def build(name):
        if name == "LinUCB":
            return LinUCB(3, N_FEATURES, alpha=LINUCB_ALPHA)
        if name == "Thompson":
            return LinearThompson(3, N_FEATURES, v=THOMPSON_V, seed=31)
        if name == "random":
            return RandomRouter(3, seed=41)
        return ThresholdRouter(thresh.cuts, score_index=6, n_arms=3)

    stored, per_seed = {}, {}
    for name in ("LinUCB", "Thompson", "difficulty-threshold", "random"):
        curves = []
        for s in range(FLEET_SEEDS):
            w = make_workload(FLEET_QUERIES, seed=300 + s)
            curves.append(run_fleet(build(name), w).regret)
        curve = np.mean(curves, axis=0)
        stored[name] = curve
        per_seed[name] = np.array([c[-1] for c in curves])
        slope, r2, pref = loglog_slope(curve, 2_000, FLEET_QUERIES)
        for t in np.unique(np.round(np.logspace(2, np.log10(FLEET_QUERIES), 40)).astype(int)):
            rows.append({
                "experiment": "fleet", "policy": name, "t": int(t),
                "cum_regret": round(float(curve[t - 1]), 6), "argmax_gap_scale": "",
                "argmax_interior": "", "fitted_slope": round(slope, 4),
                "fit_r2": round(r2, 5), "fit_prefactor": round(pref, 4),
                "fit_lo": 2_000, "fit_hi": FLEET_QUERIES, "n_seeds": FLEET_SEEDS,
                "d": N_FEATURES, "n_arms": 3, "sigma": "",
            })
        print(f"  fleet   {name:20s} slope={slope:.3f} R2={r2:.4f} "
              f"final regret={curve[-1]:.1f}")

    # Where the bandit overtakes the frozen baseline, measured rather than extrapolated.
    ahead = stored["LinUCB"] < stored["difficulty-threshold"]
    first = int(np.argmax(ahead)) + 1 if ahead.any() else -1
    # "Sustained" means it never gives the lead back: the first index after which the
    # inequality holds for the whole remaining run. A first-crossing index alone would
    # report a single step of noise near the crossover as the answer; a sustained index
    # alone would hide a lead that was taken and lost, which on a misspecified model is
    # exactly the outcome worth knowing about. Both go in the CSV.
    tail_ok = np.flip(np.cumprod(np.flip(ahead)))
    sustained = int(np.argmax(tail_ok)) + 1 if tail_ok.any() else -1
    # Paired over the workload seeds: both policies see the identical queries on each
    # seed, so the per-seed difference removes the workload variance. Without this the
    # final gap (a fraction of a percent of either total) is unreadable -- and the
    # conclusion turns on whether that gap is real.
    deltas = per_seed["difficulty-threshold"] - per_seed["LinUCB"]
    gap = float(deltas.mean())
    gap_se = float(deltas.std(ddof=1) / np.sqrt(FLEET_SEEDS))
    print(f"  fleet   LinUCB vs frozen threshold: first crossing at t = "
          f"{first if first > 0 else 'never'}, sustained from t = "
          f"{sustained if sustained > 0 else 'never'} "
          f"(within {FLEET_QUERIES})")
    verdict = ("LinUCB ahead" if gap > 2 * gap_se else
               "threshold ahead" if gap < -2 * gap_se else "indistinguishable")
    print(f"          paired final regret, threshold - LinUCB = {gap:+.1f} "
          f"+- {gap_se:.1f} over {FLEET_SEEDS} seeds  [{verdict}]")
    rows.append({
        "experiment": "fleet_crossover",
        "policy": "paired final regret: difficulty-threshold - LinUCB",
        "t": FLEET_QUERIES, "cum_regret": round(gap, 4), "argmax_gap_scale": "",
        "argmax_interior": "", "fitted_slope": "", "fit_r2": "",
        "fit_prefactor": round(gap_se, 4), "fit_lo": "", "fit_hi": FLEET_QUERIES,
        "n_seeds": FLEET_SEEDS, "d": N_FEATURES, "n_arms": 3, "sigma": "",
    })
    for label, value in (("first_crossing", first), ("sustained_crossing", sustained)):
        rows.append({
            "experiment": "fleet_crossover",
            "policy": f"LinUCB vs difficulty-threshold: {label}",
            "t": value,
            "cum_regret": round(float(stored["LinUCB"][value - 1]), 4)
            if value > 0 else "", "argmax_gap_scale": "", "argmax_interior": "",
            "fitted_slope": "", "fit_r2": "", "fit_prefactor": "", "fit_lo": "",
            "fit_hi": FLEET_QUERIES, "n_seeds": FLEET_SEEDS, "d": N_FEATURES,
            "n_arms": 3, "sigma": "",
        })


def fleet_long_horizon(rows: list[dict]) -> None:
    """Is LinUCB's fleet exponent a number, or a moving target?

    A well-specified learner's regret exponent settles as the horizon grows. A
    misspecified one's climbs toward 1, because the part of the reward its features
    cannot represent contributes a constant regret per query that no amount of data
    removes. Fitting over nested windows [2000, H] for increasing H distinguishes the
    two, and it is the cleanest evidence available here that the linear model -- not the
    exploration rule, and not the horizon -- is what limits the bandit on this fleet.

    Restricted to LinUCB and the frozen difficulty-threshold baseline, because those are
    the two whose comparison the README turns on and running Thompson and random this far
    would triple the cost for nothing.
    """
    from src.fleet.simulator import expected_reward_matrix
    tune = make_workload(6_000, seed=901)
    thresh = ThresholdRouter.fit(tune.difficulty_score,
                                 expected_reward_matrix(tune, LAMBDA), score_index=6)

    curves = {}
    for name in ("LinUCB", "difficulty-threshold"):
        per_seed = []
        for s in range(FLEET_SEEDS):
            w = make_workload(LONG_QUERIES, seed=300 + s)
            router = (LinUCB(3, N_FEATURES, alpha=LINUCB_ALPHA) if name == "LinUCB"
                      else ThresholdRouter(thresh.cuts, score_index=6, n_arms=3))
            per_seed.append(run_fleet(router, w).regret)
        curves[name] = np.stack(per_seed)

    for name, stack in curves.items():
        mean = stack.mean(axis=0)
        for h in LONG_WINDOWS:
            slope, r2, pref = loglog_slope(mean, 2_000, h)
            rows.append({
                "experiment": "fleet_long_horizon", "policy": name, "t": h,
                "cum_regret": round(float(mean[h - 1]), 4), "argmax_gap_scale": "",
                "argmax_interior": "", "fitted_slope": round(slope, 4),
                "fit_r2": round(r2, 5), "fit_prefactor": round(pref, 5),
                "fit_lo": 2_000, "fit_hi": h, "n_seeds": FLEET_SEEDS,
                "d": N_FEATURES, "n_arms": 3, "sigma": "",
            })
        mine = [r for r in rows if r["experiment"] == "fleet_long_horizon"
                and r["policy"] == name]
        shown = "  ".join(f"H={r['t'] // 1000}k:{r['fitted_slope']:.3f}" for r in mine)
        print(f"  long    {name:22s} exponent over [2000, H]:  {shown}")

    u, t_ = curves["LinUCB"].mean(axis=0), curves["difficulty-threshold"].mean(axis=0)
    ahead = u < t_
    first = int(np.argmax(ahead)) + 1 if ahead.any() else -1
    deltas = curves["difficulty-threshold"][:, -1] - curves["LinUCB"][:, -1]
    gap, gap_se = float(deltas.mean()), float(deltas.std(ddof=1) / np.sqrt(FLEET_SEEDS))
    verdict = ("LinUCB ahead" if gap > 2 * gap_se else
               "threshold ahead" if gap < -2 * gap_se else "indistinguishable")
    print(f"  long    crossing within {LONG_QUERIES}: "
          f"{'t=' + str(first) if first > 0 else 'never'};  paired final gap "
          f"(threshold - LinUCB) = {gap:+.1f} +- {gap_se:.1f}  [{verdict}]")
    rows.append({
        "experiment": "fleet_long_horizon",
        "policy": f"paired final gap: difficulty-threshold - LinUCB [{verdict}]",
        "t": LONG_QUERIES, "cum_regret": round(gap, 4), "argmax_gap_scale": "",
        "argmax_interior": "", "fitted_slope": "", "fit_r2": "",
        "fit_prefactor": round(gap_se, 4), "fit_lo": "", "fit_hi": LONG_QUERIES,
        "n_seeds": FLEET_SEEDS, "d": N_FEATURES, "n_arms": 3, "sigma": "",
    })


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows: list[dict] = []
    print("minimax envelope over the instance family "
          f"({len(GAP_SCALES)} gap scales x {N_SEEDS} seeds x T={HORIZON}):")
    minimax_envelope(rows)
    print("\nfixed-instance drift (the experiment that proves nothing):")
    fixed_instance_drift(rows)
    print("\nsimulated fleet (linear model misspecified):")
    fleet_regret(rows)
    print(f"\nlong-horizon exponent drift ({LONG_QUERIES} queries, "
          f"{FLEET_SEEDS} seeds, LinUCB vs the frozen baseline):")
    fleet_long_horizon(rows)

    out = RESULTS / "regret.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
