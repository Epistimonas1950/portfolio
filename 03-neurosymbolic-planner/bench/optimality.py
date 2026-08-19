#!/usr/bin/env python3
"""A* optimality and cost against the horizon.  Writes results/optimality.csv.

Table A of BRIEF.md.  Three regimes, and they are labelled in the CSV's `ground_truth`
column because they are not equally strong evidence:

    3-7 stops    exhaustive enumeration is affordable, so "optimal found" is measured
                 against the true optimum -- the only regime where the claim is proved
                 by this benchmark rather than assumed.
    8-12 stops   brute force is out of reach; A* is run with two different admissible
                 heuristics and their costs must agree.  That is a real check (the two
                 explore different node sets) but a weaker one.
    14+ stops    h2 only.  What is measured here is cost of search, not correctness.

The last block is the wall: horizons run with a hard node budget until A* fails to
finish inside it.  BRIEF.md asks for a 20-step row; the honest answer is a measured
ceiling rather than a blank, so the CSV records the horizon at which the budget bites.
"""

from __future__ import annotations

import csv
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.domain import make_instance, plan_cost                     # noqa: E402
from src.search import heuristics as H                              # noqa: E402
from src.search.astar import astar                                  # noqa: E402
from src.search.bruteforce import brute_force                       # noqa: E402
from src.search.instrument import BudgetExceeded                    # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

EXACT_HORIZONS = (3, 4, 5, 6, 7)
EXACT_SEEDS = range(12)
EXACT_HEURISTICS = ("h0", "h1", "h2", "hx")

CROSS_HORIZONS = (8, 10, 12)
CROSS_SEEDS = range(6)

LARGE_HORIZONS = (14, 16)
LARGE_SEEDS = range(5)

WALL_HORIZONS = (18, 20, 22, 24)
WALL_BUDGET = 2_000_000


def _summary(horizon: str | int, name: str, ground: str, recs: list[dict]) -> dict:
    n = len(recs)
    solved = [r for r in recs if r["cost"] is not None]
    return {
        "horizon": horizon,
        "heuristic": name,
        "ground_truth": ground,
        "instances": n,
        "optimal_found_pct": round(100.0 * sum(r["optimal"] for r in recs) / n, 2),
        "feasible_instances": len(solved),
        "mean_expansions": round(sum(r["expansions"] for r in recs) / n, 1),
        "max_expansions": max(r["expansions"] for r in recs),
        "mean_generated": round(sum(r["generated"] for r in recs) / n, 1),
        "mean_ebf": round(sum(r["ebf"] for r in solved) / max(len(solved), 1), 4),
        "mean_re_expansions": round(sum(r["re_expansions"] for r in recs) / n, 3),
        "mean_seconds": round(sum(r["seconds"] for r in recs) / n, 5),
        "budget_exceeded": sum(r["budget"] for r in recs),
    }


def run() -> list[dict]:
    rows: list[dict] = []

    # --- regime 1: proved against exhaustive enumeration -------------------------- #
    for n in EXACT_HORIZONS:
        truth = {}
        for seed in EXACT_SEEDS:
            inst = make_instance(n, seed)
            truth[seed] = brute_force(inst).cost
        for name in EXACT_HEURISTICS:
            recs = []
            for seed in EXACT_SEEDS:
                inst = make_instance(n, seed)
                res = astar(inst, H.make(name))
                ok = res.cost == truth[seed]
                if ok and res.cost is not None:
                    ok = plan_cost(inst, res.plan) == res.cost
                recs.append({"cost": res.cost, "optimal": int(ok),
                             "expansions": res.stats.expansions,
                             "generated": res.stats.generated,
                             "ebf": res.stats.ebf if res.cost is not None else 0.0,
                             "re_expansions": res.stats.re_expansions,
                             "seconds": res.stats.seconds, "budget": 0})
            rows.append(_summary(n, name, "brute-force", recs))
            print(f"  n={n:2d} {name:16s} optimal {rows[-1]['optimal_found_pct']:6.2f}%  "
                  f"exp {rows[-1]['mean_expansions']:9.1f}  b* {rows[-1]['mean_ebf']:.3f}")

    # --- regime 2: two admissible heuristics must agree ---------------------------- #
    for n in CROSS_HORIZONS:
        per = {name: [] for name in ("h1", "h2")}
        costs = {}
        for seed in CROSS_SEEDS:
            inst = make_instance(n, seed)
            for name in per:
                res = astar(inst, H.make(name))
                costs.setdefault(seed, []).append(res.cost)
                per[name].append(res)
        agree = {seed: len(set(c)) == 1 for seed, c in costs.items()}
        for name, results in per.items():
            recs = [{"cost": r.cost, "optimal": int(agree[s]),
                     "expansions": r.stats.expansions, "generated": r.stats.generated,
                     "ebf": r.stats.ebf if r.cost is not None else 0.0,
                     "re_expansions": r.stats.re_expansions,
                     "seconds": r.stats.seconds, "budget": 0}
                    for s, r in zip(CROSS_SEEDS, results)]
            rows.append(_summary(n, name, "h1-vs-h2 agreement", recs))
            print(f"  n={n:2d} {name:16s} agree   {rows[-1]['optimal_found_pct']:6.2f}%  "
                  f"exp {rows[-1]['mean_expansions']:9.1f}  b* {rows[-1]['mean_ebf']:.3f}")

    # --- regime 3: cost of search only -------------------------------------------- #
    for n in LARGE_HORIZONS:
        recs = []
        for seed in LARGE_SEEDS:
            inst = make_instance(n, seed)
            res = astar(inst, H.make("h2"))
            recs.append({"cost": res.cost, "optimal": 0,
                         "expansions": res.stats.expansions,
                         "generated": res.stats.generated,
                         "ebf": res.stats.ebf if res.cost is not None else 0.0,
                         "re_expansions": res.stats.re_expansions,
                         "seconds": res.stats.seconds, "budget": 0})
        row = _summary(n, "h2", "none (search cost only)", recs)
        row["optimal_found_pct"] = ""      # not measured here; do not imply otherwise
        rows.append(row)
        print(f"  n={n:2d} {'h2':16s} unverified      exp "
              f"{row['mean_expansions']:9.1f}  b* {row['mean_ebf']:.3f}  "
              f"{row['mean_seconds']:.2f}s")

    # --- regime 4: where the wall is ---------------------------------------------- #
    for n in WALL_HORIZONS:
        inst = make_instance(n, 0)
        t0 = time.perf_counter()
        try:
            res = astar(inst, H.make("h2"), node_budget=WALL_BUDGET)
            rec = {"cost": res.cost, "optimal": 0, "expansions": res.stats.expansions,
                   "generated": res.stats.generated, "ebf": res.stats.ebf,
                   "re_expansions": res.stats.re_expansions,
                   "seconds": res.stats.seconds, "budget": 0}
            hit = False
        except BudgetExceeded as exc:
            st = exc.stats
            rec = {"cost": None, "optimal": 0, "expansions": st.expansions,
                   "generated": st.generated, "ebf": 0.0, "re_expansions": 0,
                   "seconds": time.perf_counter() - t0, "budget": 1}
            hit = True
        row = _summary(n, "h2", f"wall probe, budget {WALL_BUDGET} generated", [rec])
        row["optimal_found_pct"] = ""
        rows.append(row)
        print(f"  n={n:2d} {'h2':16s} "
              f"{'BUDGET EXCEEDED' if hit else 'solved         '}  gen "
              f"{row['mean_generated']:10.0f}  {row['mean_seconds']:.2f}s")
        if hit:
            break
    return rows


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    print("A* optimality vs horizon")
    rows = run()
    out = RESULTS / "optimality.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
