#!/usr/bin/env python3
"""Per-step success p, the p^n prediction, and the end-to-end rate actually measured.

BRIEF.md's compounding argument is the quantitative reason routing matters more in an
agent than in a chatbot: n independent steps at per-step success p give p^n, so 0.95 is
60% over ten steps and 0.90 is 35%, and a model four points worse per call is
catastrophically worse over a horizon. The argument is correct and the arithmetic is
not the measurement. This script runs actual episodes through src/agent/loop.py -- real
calculator, real file reads, real substring search, with only the *decision about
whether the model emitted a correct tool call* coming from the simulated fleet -- and
compares the measured end-to-end rate against p^n computed from the same runs.

Two regimes, and the second is the control that makes the first believable:

  correlated   every step of an episode inherits a shared episode-level difficulty plus
               jitter. Real agent tasks are like this: a hard task is hard at every
               step.
  independent  each step draws its own difficulty afresh. This is the regime in which
               p^n is *exactly* the right prediction, so the measurement has to land on
               it -- and if it does not, the instrument is broken and nothing else in
               this file means anything.

The expected sign is worth stating in advance, because it is the opposite of what most
people guess. With a shared per-episode difficulty the end-to-end rate is E_d[p(d)^n],
and q -> q^n is convex, so Jensen's inequality gives

    E_d[ p(d)^n ]  >=  ( E_d[ p(d) ] )^n  =  p_bar^n.                                (1)

Correlated failures make an agent do **better** than the independence formula predicts,
not worse: the failures bunch into a minority of hard episodes instead of spreading
evenly over all of them. The independence prediction is pessimistic, the gap grows with
n, and the size of the gap is a measurement of how correlated the steps are.

Writes results/compounding.csv.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.agent.loop import (ensure_corpus, make_episode, run_episode)   # noqa: E402
from src.features import N_FEATURES                                     # noqa: E402
from src.fleet.simulator import (DEFAULT_FLEET, LAMBDA,                 # noqa: E402
                                 success_probability)
from src.routers.baselines import FixedArm                              # noqa: E402
from src.routers.linucb import LinUCB                                   # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
N_EPISODES = 1_500
STEP_COUNTS = (1, 2, 3, 5, 8, 12)
LINUCB_ALPHA = 1.0
SCORE_NOISE = 0.12       # same serving-time classifier noise as the query workload


def step_features(rng: np.random.Generator):
    """Build the serving-time context for one step of an episode.

    The router sees the same ten features as in the query workload (src/features.py):
    a noisy difficulty score, the step's position in the chain as tool depth, and the
    interaction terms. It does not see the latent step difficulty, exactly as it does
    not in eval/pareto.py.
    """
    def fn(step, i):
        s = float(np.clip(step.difficulty + rng.normal(0.0, SCORE_NOISE), 0.0, 1.0))
        depth = (i + 1) / 4.0
        log_len = 0.8
        onehot = {"calculate": 0, "search_text": 1, "read_file": 2}[step.tool]
        x = np.zeros(N_FEATURES)
        x[0] = 1.0
        x[1] = log_len
        x[2 + onehot] = 1.0
        x[5] = depth
        x[6] = s
        x[7] = s * s
        x[8] = s * log_len
        x[9] = s * depth
        expected = np.array([a.base_seconds + a.seconds_per_token * 90
                             for a in DEFAULT_FLEET])
        return x, expected
    return fn


def run_block(policy_name: str, router_factory, n_steps: int, independent: bool,
              root: pathlib.Path, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    router = router_factory()
    n_ok_steps = 0
    n_steps_total = 0
    n_success = 0
    seconds = 0.0
    p_model = []
    for _ in range(N_EPISODES):
        ep_diff = float(np.clip(rng.beta(2.0, 2.0), 0.0, 1.0))
        ep = make_episode(rng, n_steps, root, ep_diff, independent)
        res = run_episode(ep, router, step_features(rng), DEFAULT_FLEET, rng, root,
                          lam=LAMBDA)
        n_ok_steps += int(sum(res.step_ok))
        n_steps_total += len(res.step_ok)
        n_success += int(res.succeeded)
        seconds += res.seconds
        for st, arm in zip(ep.steps, res.arms):
            p_model.append(float(success_probability(np.array([st.difficulty]),
                                                     DEFAULT_FLEET)[0, arm]))
    p_hat = n_ok_steps / n_steps_total
    measured = n_success / N_EPISODES
    predicted = p_hat ** n_steps
    return {
        "policy": policy_name,
        "regime": "independent" if independent else "correlated",
        "n_steps": n_steps,
        "per_step_success": round(p_hat, 5),
        "mean_model_p": round(float(np.mean(p_model)), 5),
        "predicted_end_to_end": round(predicted, 6),
        "measured_end_to_end": round(measured, 6),
        "gap_measured_minus_predicted": round(measured - predicted, 6),
        "ratio_measured_over_predicted": round(measured / predicted, 4)
        if predicted > 0 else "",
        "seconds_per_episode": round(seconds / N_EPISODES, 4),
        "n_episodes": N_EPISODES,
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    root = ensure_corpus()
    print(f"corpus: {len(list(root.glob('notes_*.txt')))} files under {root}")

    policies = {
        "always-small": lambda: FixedArm(0, name="always-small"),
        "always-large": lambda: FixedArm(2, name="always-large"),
        "LinUCB": lambda: LinUCB(3, N_FEATURES, alpha=LINUCB_ALPHA),
    }
    rows = []
    for name, factory in policies.items():
        for independent in (False, True):
            for n in STEP_COUNTS:
                row = run_block(name, factory, n, independent, root,
                                seed=4200 + 7 * n + int(independent))
                rows.append(row)
            tag = "independent" if independent else "correlated"
            last = rows[-1]
            print(f"  {name:13s} {tag:12s} n={last['n_steps']:2d}  "
                  f"p={last['per_step_success']:.4f}  p^n={last['predicted_end_to_end']:.4f}"
                  f"  measured={last['measured_end_to_end']:.4f}  "
                  f"ratio={last['ratio_measured_over_predicted']}")

    out = RESULTS / "compounding.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
