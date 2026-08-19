#!/usr/bin/env python3
"""The sharpest weakness in the project, made measurable.

BRIEF.md states the problem and asks for it to be volunteered before it is asked about:
the regret bound is over *the reward you actually optimise*, not the reward you care
about. A biased surrogate does not degrade the bound -- it removes its premise. The
policy still converges, at the rate the theory promises, to the maximiser of the wrong
objective, and nothing in its own diagnostics can tell it so. This script produces that
failure on purpose and measures both halves of it.

The surrogate. An LLM judge scoring its own fleet has a well-documented tendency to
reward the fluent, longer, more authoritative-sounding answer -- which is systematically
the bigger model's, correct or not. Model that as capability-proportional partial credit:

    q_tilde(t, a)  =  success(t, a)  +  b * w_a ,      w = (0, 0.386, 1),            (1)

with w the arms' skills rescaled to [0, 1] and b the bias magnitude. b = 0 is the true
reward. The bandit learns from r_tilde = q_tilde - lambda c and is *scored* on the true
r = success - lambda c. Nothing about the surrogate is visible to the policy; it is a
different number arriving in the same slot.

What is measured, per bias level:

  regret_vs_surrogate   cumulative regret against argmax_a E[r_tilde | x, a]. This is
                        the curve the policy would plot for itself, and it is the one
                        that looks fine.
  regret_vs_true        cumulative regret against argmax_a E[r | x, a]. This is the
                        curve that matters, and it grows linearly, because the policy is
                        converging to a fixed and wrong arm distribution.

The log-log slopes of the two are the compact statement of the failure: a sublinear
surrogate-regret exponent next to an exponent of ~1 on the true objective. There is no
threshold on b at which this switches on -- the damage is continuous in b, which is
worse than a threshold would be, because it means a small bias is invisible rather than
absent.

The honest conclusion, which no amount of bandit machinery fixes: the only defences are
to measure the surrogate's agreement with ground truth on a sample, and to bound how far
the optimum can move under a bias of a given size. The first needs human labels and is
listed in STATUS.md as not obtainable here; the second is the `arm_share_large` column
below, which is that displacement, measured.

Writes results/surrogate_bias.csv. Simulated fleet throughout.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.workload import loglog_slope, run_fleet                        # noqa: E402
from src.features import N_FEATURES                                      # noqa: E402
from src.fleet.simulator import (DEFAULT_FLEET, LAMBDA,                  # noqa: E402
                                 expected_reward_matrix, make_workload)
from src.routers.linucb import LinUCB                                    # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
N_QUERIES = 40_000
N_SEEDS = 3
BIAS_LEVELS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35)
LINUCB_ALPHA = 1.0
FIT_LO = 2_000


def bias_weights() -> np.ndarray:
    """w in equation (1): arm skills rescaled so the small arm gets none of the credit."""
    skills = np.array([a.skill for a in DEFAULT_FLEET])
    return (skills - skills[0]) / (skills[-1] - skills[0])


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    w_vec = bias_weights()
    print(f"bias weights w = {w_vec.round(3)} (small gets no credit, large gets all)")
    rows = []
    for b in BIAS_LEVELS:
        sur_curves, true_curves = [], []
        succ, cost, reward, shares = [], [], [], []
        for s in range(N_SEEDS):
            wl = make_workload(N_QUERIES, seed=700 + s)
            costs = wl.cost_matrix()
            surrogate_reward = (wl.success.astype(float) + b * w_vec[None, :]
                                - LAMBDA * costs)
            run = run_fleet(LinUCB(3, N_FEATURES, alpha=LINUCB_ALPHA), wl,
                            reward_override=surrogate_reward)
            mu_true = expected_reward_matrix(wl, LAMBDA)
            mu_sur = mu_true + b * w_vec[None, :]
            rowsidx = np.arange(N_QUERIES)
            sur_curves.append(np.cumsum(mu_sur.max(axis=1) - mu_sur[rowsidx, run.arms]))
            true_curves.append(run.regret)
            succ.append(run.success_rate)
            cost.append(run.mean_cost)
            reward.append(float(run.reward.mean()))
            shares.append(run.arm_shares(3))
        sur = np.mean(sur_curves, axis=0)
        tru = np.mean(true_curves, axis=0)
        s_sur, r2_sur, _ = loglog_slope(sur, FIT_LO, N_QUERIES)
        if tru[-1] > 1e-9:
            s_tru, r2_tru, _ = loglog_slope(tru, FIT_LO, N_QUERIES)
        else:
            s_tru, r2_tru = float("nan"), float("nan")
        sh = np.mean(shares, axis=0)
        for t in np.unique(np.round(np.logspace(2, np.log10(N_QUERIES), 40)).astype(int)):
            rows.append({
                "bias_b": b, "policy": "LinUCB", "t": int(t),
                "regret_vs_surrogate": round(float(sur[t - 1]), 4),
                "regret_vs_true": round(float(tru[t - 1]), 4),
                "slope_surrogate": round(s_sur, 4), "r2_surrogate": round(r2_sur, 5),
                "slope_true": round(s_tru, 4), "r2_true": round(r2_tru, 5),
                "true_success": round(float(np.mean(succ)), 5),
                "true_mean_cost": round(float(np.mean(cost)), 4),
                "true_mean_reward": round(float(np.mean(reward)), 5),
                "arm_share_small": round(float(sh[0]), 4),
                "arm_share_mid": round(float(sh[1]), 4),
                "arm_share_large": round(float(sh[2]), 4),
                "fit_lo": FIT_LO, "fit_hi": N_QUERIES, "n_seeds": N_SEEDS,
            })
        print(f"  b={b:<5} slope(surrogate)={s_sur:.3f} R2={r2_sur:.4f} | "
              f"slope(true)={s_tru:.3f} | true reward={np.mean(reward):.4f} "
              f"succ={np.mean(succ):.4f} cost={np.mean(cost):.3f} "
              f"share_large={sh[2]:.3f}")

    out = RESULTS / "surrogate_bias.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
