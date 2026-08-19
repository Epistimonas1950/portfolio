#!/usr/bin/env python3
"""Target coverage against empirical coverage, and the three ways it can go wrong.

Four experiments, all on the simulated fleet, all written to results/coverage.csv.

`marginal`     Sweep alpha and check P(Y in C(X)) >= 1 - alpha per arm on held-out
               data. This is the guarantee derived in src/conformal/calibrate.py and
               it is the second of the repo's two load-bearing claims. Averaged over
               `N_SPLITS` random calibration/test partitions, because a single split
               gives an empirical coverage with standard error sqrt(a(1-a)/n_test) and
               reporting one draw of that as "the coverage" would be noise.

`conditional`  The same sets, but coverage computed inside difficulty terciles. Split
               conformal promises *marginal* coverage. It does not promise coverage
               conditional on anything, and for a router that is precisely the wrong
               way round: 95% coverage that decomposes into 99% on easy queries and
               85% on hard ones is a system that is confidently wrong exactly when it
               matters. Measuring the decomposition is the honest thing to do and it is
               not a failure of the method -- it is what the method says.

`cascade`      Multi-tier composition. Reports the union bound sum_i alpha_i against the
               realized end-to-end miscoverage, per-tier acceptance, and the slack. Run
               on two workloads and two budget splits, because the cascade's behaviour
               turns on one comparison: an alpha_i smaller than tier i's error rate
               means the (1 - alpha_i) calibration quantile lands inside the mass of
               scores from wrong answers, q_hat goes high, the sets go wide, and the
               tier can never emit a singleton. So a tier can only ever answer when it
               is allotted more error budget than it actually makes. The nominal
               workload (tier error rates 0.30 / 0.13 / 0.02) and the easy workload
               (0.07 / 0.02 / 0.004) sit on opposite sides of that line for alpha <= 0.2,
               and the acceptance rates move accordingly.

`shift`        Exchangeability broken on purpose: calibrate on the easy workload, test
               on a hard one. Coverage collapses. BRIEF.md is explicit that diagnosing
               this is a result rather than a failure, so it is measured rather than
               avoided -- and it is the reason the guarantee's premise is worth stating
               out loud every time the guarantee is.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.conformal.calibrate import (covered, prediction_sets,        # noqa: E402
                                     set_sizes, split_conformal)
from src.conformal.cascade import build_cascade, run_cascade, split_budget  # noqa: E402
from src.fleet.simulator import make_workload                          # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

ALPHAS = (0.01, 0.02, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20)
N_POOL = 40_000
N_CAL = 4_000
N_SPLITS = 20
EASY_SHIFT = -0.35     # subtracted difficulty: makes every arm's error rate small
HARD_SHIFT = 0.25      # added difficulty: the out-of-calibration test distribution

FIELDS = ["experiment", "workload", "split", "unit", "alpha", "target_coverage",
          "empirical_coverage", "coverage_std", "mean_set_size", "empty_rate",
          "singleton_rate", "error_rate", "escalation_rate", "accept_rate",
          "union_bound", "slack", "mean_cost", "n_cal", "n_test", "n_splits"]


def _blank(**kw) -> dict:
    row = {f: "" for f in FIELDS}
    row.update(kw)
    return row


def marginal_and_conditional(rows: list[dict]) -> None:
    w = make_workload(N_POOL, seed=2024)
    rng = np.random.default_rng(11)
    n_test = N_POOL - N_CAL
    terciles = np.digitize(w.difficulty, np.quantile(w.difficulty, [1 / 3, 2 / 3]))

    for k, arm in enumerate(w.arms):
        err = float(1.0 - w.success[:, k].mean())
        for alpha in ALPHAS:
            cov, size, empty, sing = [], [], [], []
            cond = {0: [], 1: [], 2: []}
            for _ in range(N_SPLITS):
                perm = rng.permutation(N_POOL)
                cal, test = perm[:N_CAL], perm[N_CAL:]
                c = split_conformal(w.probs[cal, k, :], w.label[cal], alpha)
                mask = prediction_sets(w.probs[test, k, :], c)
                hit = covered(mask, w.label[test])
                sz = set_sizes(mask)
                cov.append(hit.mean()); size.append(sz.mean())
                empty.append((sz == 0).mean()); sing.append((sz == 1).mean())
                for g in (0, 1, 2):
                    sel = terciles[test] == g
                    cond[g].append(hit[sel].mean())
            rows.append(_blank(
                experiment="marginal", workload="nominal", split="-",
                unit=arm.name, alpha=alpha, target_coverage=round(1 - alpha, 4),
                empirical_coverage=round(float(np.mean(cov)), 5),
                coverage_std=round(float(np.std(cov)), 5),
                mean_set_size=round(float(np.mean(size)), 4),
                empty_rate=round(float(np.mean(empty)), 5),
                singleton_rate=round(float(np.mean(sing)), 5),
                error_rate=round(err, 5), n_cal=N_CAL, n_test=n_test,
                n_splits=N_SPLITS))
            for g, tag in enumerate(("easy", "medium", "hard")):
                rows.append(_blank(
                    experiment="conditional", workload="nominal", split="-",
                    unit=f"{arm.name}|difficulty-{tag}", alpha=alpha,
                    target_coverage=round(1 - alpha, 4),
                    empirical_coverage=round(float(np.mean(cond[g])), 5),
                    coverage_std=round(float(np.std(cond[g])), 5),
                    error_rate=round(err, 5), n_cal=N_CAL,
                    n_test=int(n_test / 3), n_splits=N_SPLITS))
        shown = []
        for a in (0.01, 0.1, 0.2):
            got = [r["empirical_coverage"] for r in rows
                   if r["experiment"] == "marginal" and r["unit"] == arm.name
                   and r["alpha"] == a]
            shown.append(f"a={a:.3f}:cov={float(np.mean(got)):.4f}")
        print(f"  marginal {arm.name:6s} error={err:.3f}  " + "  ".join(shown))


def cascades(rows: list[dict]) -> None:
    splits = {"equal": lambda a: split_budget(a, 3),
              # Front-loaded: the tier that sees every query gets most of the budget.
              # Still a legal split, since the three still sum to alpha.
              "front-loaded": lambda a: (0.7 * a, 0.2 * a, 0.1 * a)}
    for wl, shift in (("nominal", 0.0), ("easy", EASY_SHIFT)):
        cal_w = make_workload(12_000, seed=5100, difficulty_shift=shift)
        test_w = make_workload(20_000, seed=5200, difficulty_shift=shift)
        errs = 1.0 - test_w.success.mean(axis=0)
        costs = test_w.cost_matrix()
        for split_name, fn in splits.items():
            for alpha in ALPHAS:
                alphas = fn(alpha)
                tiers = build_cascade(cal_w.probs, cal_w.label, arms=(0, 1, 2),
                                      alphas=alphas,
                                      names=tuple(a.name for a in test_w.arms))
                rep = run_cascade(test_w.probs, test_w.label, tiers, costs=costs)
                rows.append(_blank(
                    experiment="cascade", workload=wl, split=split_name, unit="end-to-end",
                    alpha=alpha, target_coverage=round(1 - alpha, 4),
                    empirical_coverage=round(rep.empirical_coverage, 5),
                    mean_set_size=round(rep.mean_set_size, 4),
                    union_bound=round(rep.union_bound, 5),
                    slack=round(rep.slack, 5), mean_cost=round(rep.mean_cost, 4),
                    n_cal=len(cal_w), n_test=len(test_w), n_splits=1))
                for i, tier in enumerate(tiers):
                    rows.append(_blank(
                        experiment="cascade", workload=wl, split=split_name,
                        unit=f"tier{i}:{tier.name}", alpha=round(alphas[i], 5),
                        target_coverage=round(1 - alphas[i], 5),
                        empirical_coverage=round(1 - rep.per_tier_miscoverage[i], 5),
                        escalation_rate=round(rep.escalation_rate[i], 5),
                        accept_rate=round(rep.accept_rate[i], 5),
                        error_rate=round(float(errs[i]), 5),
                        n_cal=len(cal_w), n_test=len(test_w), n_splits=1))
            print(f"  cascade {wl:8s} {split_name:12s} errors={errs.round(3)}  "
                  f"a=0.2 accept={[round(r,3) for r in rep.accept_rate]} "
                  f"cov={rep.empirical_coverage:.4f} bound={rep.union_bound:.3f} "
                  f"slack={rep.slack:.4f}")


def exchangeability_break(rows: list[dict]) -> None:
    """Calibrate on easy queries, test on hard ones. The premise fails; so does (5)."""
    cal_w = make_workload(12_000, seed=6100, difficulty_shift=EASY_SHIFT)
    test_w = make_workload(20_000, seed=6200, difficulty_shift=HARD_SHIFT)
    for k, arm in enumerate(test_w.arms):
        for alpha in ALPHAS:
            c = split_conformal(cal_w.probs[:, k, :], cal_w.label, alpha)
            mask = prediction_sets(test_w.probs[:, k, :], c)
            cov = float(covered(mask, test_w.label).mean())
            sz = set_sizes(mask)
            rows.append(_blank(
                experiment="shift", workload="cal=easy,test=hard", split="-",
                unit=arm.name, alpha=alpha, target_coverage=round(1 - alpha, 4),
                empirical_coverage=round(cov, 5),
                mean_set_size=round(float(sz.mean()), 4),
                empty_rate=round(float((sz == 0).mean()), 5),
                singleton_rate=round(float((sz == 1).mean()), 5),
                error_rate=round(float(1 - test_w.success[:, k].mean()), 5),
                n_cal=len(cal_w), n_test=len(test_w), n_splits=1))
        got = [r["empirical_coverage"] for r in rows
               if r["experiment"] == "shift" and r["unit"] == arm.name]
        print(f"  shift    {arm.name:6s} target 0.99->0.80, got "
              f"{got[0]:.3f} -> {got[-1]:.3f}")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows: list[dict] = []
    print("marginal + conditional coverage:")
    marginal_and_conditional(rows)
    print("\ncascade composition:")
    cascades(rows)
    print("\nexchangeability deliberately broken:")
    exchangeability_break(rows)
    out = RESULTS / "coverage.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
