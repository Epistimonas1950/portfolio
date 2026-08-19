#!/usr/bin/env python3
"""The cost-quality table: every policy in the brief, plus the oracle that bounds them.

One row per policy on the simulated fleet, with the columns BRIEF.md asks for and two
it does not, because leaving them out would be flattering:

  peak_memory_gb        the *maximum* resident memory over the arms a policy actually
                        touches -- what you have to provision. A router that sends 2% of
                        its queries to the 32B arm still needs the 32B arm resident, so
                        its provisioning cost is the large arm's, not its average. The
                        mean is reported separately as `mean_memory_gb`; the two differ
                        by a factor of four here and conflating them is how routing
                        results get oversold.
  cost_at_matched_acc   the per-query cost of the cheapest *fixed-arm* policy whose
                        success rate is at least this policy's. The ratio
                        mean_cost / cost_at_matched_acc is the compute saving at equal
                        quality, which is the number a team actually wants, and it is
                        blank when the policy is more accurate than every fixed arm --
                        there is nothing to match it against and inventing an
                        extrapolation would be inventing a measurement.

Escalation rate is reported under a stated definition, because the brief's table uses
the word for two different things. For the conformal cascade it is the share of queries
not answered by the first tier. For a bandit router there is no escalation -- it makes
one call -- so the column holds the share of queries not sent to the cheapest arm, which
is the closest honest analogue, and the fixed-arm and oracle rows leave it blank.

A second block of rows traces the budgeted router's realized spend against the budget
line (B/T)t and its dual variable p_t, which is what "the single price is learned
online" looks like when you plot it.

Every number here is a simulated-fleet number.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.workload import replay_fixed, run_fleet                       # noqa: E402
from src.features import N_FEATURES                                     # noqa: E402
from src.fleet.simulator import (LAMBDA, cheapest_sufficient,           # noqa: E402
                                 expected_reward_matrix, make_workload,
                                 oracle_expected)
from src.routers.baselines import (FixedArm, RandomRouter,              # noqa: E402
                                   ThresholdRouter)
from src.routers.budgeted import BudgetedRouter                         # noqa: E402
from src.routers.linucb import LinGreedy, LinUCB                        # noqa: E402
from src.routers.thompson import LinearThompson                         # noqa: E402
from src.conformal.cascade import build_cascade, run_cascade, split_budget  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
N_QUERIES = 20_000
N_SEEDS = 5
TUNE_QUERIES = 6_000
SCORE_INDEX = 6          # index of difficulty_score in src.features.FEATURE_NAMES
CASCADE_ALPHA = 0.20

# Exploration constants, chosen on five TUNING workloads (seeds 1400-1404) that are
# disjoint from the evaluation seeds below. The reward surface is flat: LinUCB scores
# 0.7454 / 0.7471 / 0.7482 / 0.7477 at alpha = 0.4 / 0.8 / 1.0 / 1.2, so alpha = 1.0 is
# the middle of a plateau rather than a sharp optimum, and it happens to coincide with
# the value used in the linear-bandit experiment. Reporting this rather than a bare
# constant matters, because the difficulty-threshold baseline below is also fitted and
# a comparison between a tuned method and an untuned one is not a comparison.
LINUCB_ALPHA = 1.0
THOMPSON_V = 0.35

FIELDS = ["block", "policy", "task_success", "task_success_se", "seconds_per_query",
          "mean_cost", "peak_memory_gb", "mean_memory_gb", "escalation_rate",
          "mean_reward", "mean_reward_se", "paired_delta_vs_linucb",
          "paired_delta_se", "significant_vs_linucb",
          "cost_at_matched_acc", "cost_ratio_at_matched_acc", "final_regret",
          "share_small", "share_mid", "share_large", "budget", "total_spend",
          "t", "spend", "budget_line", "dual_price", "n_queries", "n_seeds"]

REFERENCE_POLICY = "LinUCB"


def _blank(**kw) -> dict:
    row = {f: "" for f in FIELDS}
    row.update(kw)
    return row


def summarise(run, w, fixed_acc, fixed_cost, escalation=None,
              escalation_applies: bool = True) -> dict:
    mem = np.array([a.peak_memory_gb for a in w.arms])
    used = np.bincount(run.arms, minlength=len(w.arms)) > 0
    shares = run.arm_shares(len(w.arms))
    acc = run.success_rate
    matched = [c for a, c in zip(fixed_acc, fixed_cost) if a >= acc]
    at_matched = min(matched) if matched else None
    # A fixed-arm policy and an oracle have no escalation to speak of, and the column
    # is blank for them in BRIEF.md's table. `escalation_applies` is an explicit flag
    # rather than a name check, because "oracle (cheapest sufficient)" does not equal
    # "oracle" and a string comparison would silently fill the cell.
    if not escalation_applies:
        esc = None
    elif escalation is not None:
        esc = escalation
    else:
        esc = float((run.arms != 0).mean())
    return _blank(
        block="policy", policy=run.name, task_success=round(acc, 5),
        seconds_per_query=round(float(run.seconds.mean()), 4),
        mean_cost=round(run.mean_cost, 4),
        peak_memory_gb=round(float(mem[used].max()), 2),
        mean_memory_gb=round(float((shares * mem).sum()), 3),
        escalation_rate="" if esc is None else round(esc, 5),
        mean_reward=round(float(run.reward.mean()), 5),
        cost_at_matched_acc="" if at_matched is None else round(at_matched, 4),
        cost_ratio_at_matched_acc="" if at_matched is None else
        round(run.mean_cost / at_matched, 4),
        final_regret=round(float(run.regret[-1]), 3),
        share_small=round(float(shares[0]), 4), share_mid=round(float(shares[1]), 4),
        share_large=round(float(shares[2]), 4),
        n_queries=len(w), n_seeds=N_SEEDS)


def average_rows(per_seed: list[list[dict]]) -> list[dict]:
    """Mean over seeds, plus the error bars that decide whether a gap is a result.

    The headline comparison in this table -- the difficulty-threshold baseline against
    LinUCB -- is a difference of a few thousandths in mean reward, which is the same
    order as the seed-to-seed spread. Reporting the two means and letting the reader
    infer an ordering would be reporting noise.

    So the difference is computed **paired**: both policies see the identical workload
    on each seed, so the per-seed difference removes the workload variance and leaves
    only the difference in policy behaviour. That is a much sharper instrument than
    comparing two unpaired means, and it is the difference between "beats" and
    "matches to within noise" -- which are different claims and want different English.
    `significant_vs_linucb` is 1 when |paired delta| > 2 standard errors.
    """
    n_seeds = len(per_seed)
    ref = next(i for i, r in enumerate(per_seed[0]) if r["policy"] == REFERENCE_POLICY)
    out = []
    for i in range(len(per_seed[0])):
        base = dict(per_seed[0][i])
        for f in FIELDS:
            vals = [r[i][f] for r in per_seed]
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
                base[f] = round(float(np.mean(vals)), 5)
        rewards = np.array([r[i]["mean_reward"] for r in per_seed], dtype=float)
        succ = np.array([r[i]["task_success"] for r in per_seed], dtype=float)
        deltas = rewards - np.array([r[ref]["mean_reward"] for r in per_seed],
                                    dtype=float)
        d_se = float(np.std(deltas, ddof=1) / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0
        base["mean_reward_se"] = round(float(np.std(rewards, ddof=1)
                                             / np.sqrt(n_seeds)), 6)
        base["task_success_se"] = round(float(np.std(succ, ddof=1)
                                              / np.sqrt(n_seeds)), 6)
        base["paired_delta_vs_linucb"] = round(float(deltas.mean()), 6)
        base["paired_delta_se"] = round(d_se, 6)
        base["significant_vs_linucb"] = int(abs(float(deltas.mean())) > 2 * d_se) \
            if i != ref else ""
        out.append(base)
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)

    tune = make_workload(TUNE_QUERIES, seed=901)
    threshold = ThresholdRouter.fit(tune.difficulty_score,
                                    expected_reward_matrix(tune, LAMBDA),
                                    score_index=SCORE_INDEX)
    print(f"threshold cut-points fitted on {TUNE_QUERIES} full-information tuning "
          f"queries: {threshold.cuts.round(3)}")

    per_seed_rows: list[list[dict]] = []
    budget_trace = None
    for s in range(N_SEEDS):
        w = make_workload(N_QUERIES, seed=400 + s)
        fixed_runs = [run_fleet(FixedArm(k, name=f"always-{w.arms[k].name}"), w)
                      for k in range(3)]
        fixed_acc = [r.success_rate for r in fixed_runs]
        fixed_cost = [r.mean_cost for r in fixed_runs]

        # Budget: 70% of what the unconstrained LinUCB policy spends on this workload.
        # Chosen by running it first, so the constraint genuinely binds -- a budget the
        # unconstrained policy already satisfies would make the comparison vacuous.
        probe = run_fleet(LinUCB(3, N_FEATURES, alpha=LINUCB_ALPHA), w)
        budget = 0.70 * probe.total_cost
        budgeted = BudgetedRouter(3, N_FEATURES, budget=budget, horizon=N_QUERIES,
                                  eta0=2.0, alpha=LINUCB_ALPHA)

        rows = []
        for k in (0, 2):     # always-smallest and always-largest bracket the plane
            rows.append(summarise(fixed_runs[k], w, fixed_acc, fixed_cost,
                                  escalation_applies=False))
        rows.append(summarise(run_fleet(RandomRouter(3, seed=700 + s), w), w,
                              fixed_acc, fixed_cost))
        rows.append(summarise(run_fleet(threshold, w), w, fixed_acc, fixed_cost))
        rows.append(summarise(run_fleet(LinUCB(3, N_FEATURES, alpha=LINUCB_ALPHA), w), w,
                              fixed_acc, fixed_cost))
        rows.append(summarise(run_fleet(LinGreedy(3, N_FEATURES), w), w,
                              fixed_acc, fixed_cost))
        rows.append(summarise(run_fleet(LinearThompson(3, N_FEATURES, v=THOMPSON_V,
                                                       seed=800 + s), w), w,
                              fixed_acc, fixed_cost))
        # The budgeted router learns from quality alone; the price handles the cost.
        brun = run_fleet(budgeted, w, reward_override=w.success.astype(float))
        brow = summarise(brun, w, fixed_acc, fixed_cost)
        brow["budget"] = round(budget, 3)
        brow["total_spend"] = round(brun.total_cost, 3)
        rows.append(brow)
        unc = rows[4]
        unc["budget"] = round(budget, 3)
        unc["total_spend"] = round(probe.total_cost, 3)

        # Conformal cascade as a policy: cost is the sum over every tier it visits.
        cal_w = make_workload(8_000, seed=950 + s)
        tiers = build_cascade(cal_w.probs, cal_w.label, (0, 1, 2),
                              split_budget(CASCADE_ALPHA, 3),
                              names=tuple(a.name for a in w.arms))
        rep = run_cascade(w.probs, w.label, tiers, costs=w.cost_matrix())
        answered = rep.answering_tier
        crun = replay_fixed(answered, w, name=f"conformal cascade (alpha={CASCADE_ALPHA})")
        crow = summarise(crun, w, fixed_acc, fixed_cost,
                         escalation=1.0 - rep.accept_rate[0])
        # Override cost: a cascade pays for every tier it consults, not just the last.
        crow["mean_cost"] = round(rep.mean_cost, 4)
        crow["seconds_per_query"] = round(rep.mean_cost, 4)
        matched = [c for a, c in zip(fixed_acc, fixed_cost) if a >= crun.success_rate]
        crow["cost_at_matched_acc"] = round(min(matched), 4) if matched else ""
        crow["cost_ratio_at_matched_acc"] = round(rep.mean_cost / min(matched), 4) \
            if matched else ""
        rows.append(crow)

        for arms_seq, nm in ((cheapest_sufficient(w), "oracle (cheapest sufficient)"),
                             (oracle_expected(w), "oracle (expected-reward)")):
            rows.append(summarise(replay_fixed(arms_seq, w, name=nm), w, fixed_acc,
                                  fixed_cost, escalation_applies=False))
        per_seed_rows.append(rows)

        if s == 0:
            step = max(1, N_QUERIES // 200)
            budget_trace = [(t + 1, budgeted.spend_history[t],
                             budget * (t + 1) / N_QUERIES, budgeted.price_history[t])
                            for t in range(0, N_QUERIES, step)]

    rows = average_rows(per_seed_rows)
    for r in rows:
        sig = r["significant_vs_linucb"]
        tag = "" if sig == "" else ("  *" if sig else "  (n.s.)")
        print(f"  {r['policy']:34s} succ={r['task_success']:.4f} "
              f"cost={r['mean_cost']:.3f} reward={r['mean_reward']:.4f}"
              f"+-{r['mean_reward_se']:.4f} regret={r['final_regret']:.0f} "
              f"dR_vs_LinUCB={r['paired_delta_vs_linucb']:+.5f}"
              f"+-{r['paired_delta_se']:.5f}{tag}")

    for t, spend, line, price in budget_trace:
        rows.append(_blank(block="budget_trace", policy="budgeted (single-price)",
                           t=t, spend=round(spend, 4), budget_line=round(line, 4),
                           dual_price=round(price, 6), n_queries=N_QUERIES, n_seeds=1))

    out = RESULTS / "pareto.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
