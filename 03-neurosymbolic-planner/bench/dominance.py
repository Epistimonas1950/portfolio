#!/usr/bin/env python3
"""Heuristic dominance, measured.  Writes results/dominance.csv.

The theorem (Pearl 1984): if h_b >= h_a everywhere and both are admissible, A* with h_b
expands no node with f < C* that A* with h_a expands.  Note what it does and does not
cover -- nodes sitting exactly on f == C* are expanded or not according to tie-breaking,
and this domain has integer costs, so those ties are common.  The CSV therefore carries
BOTH `expansions` and `expansions_below_cstar`; the second is the quantity the theorem
predicts, and any inversion in the first is reported rather than smoothed away.

The fourth heuristic is the deliberately inconsistent one.  It is admissible, so it is
still exactly optimal, and the `re_expansions` column is where it pays for it.
"""

from __future__ import annotations

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.domain import make_instance                                # noqa: E402
from src.search import heuristics as H                              # noqa: E402
from src.search.astar import astar                                  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
HORIZONS = (5, 7, 9)
SEEDS = range(10)
NAMES = ("h0", "h1", "h2", "hx")


def run() -> list[dict]:
    rows: list[dict] = []
    for n in HORIZONS:
        for seed in SEEDS:
            inst = make_instance(n, seed)
            per = {}
            for name in NAMES:
                h = H.make(name)
                res = astar(inst, h)
                per[name] = res
                rows.append({
                    "horizon": n, "seed": seed, "instance": inst.name,
                    "heuristic": h.name, "consistent": int(h.consistent),
                    "h_at_root": h(inst, inst.initial_state()),
                    "optimum": res.cost,
                    "expansions": res.stats.expansions,
                    "expansions_below_cstar": res.stats.expansions_below_cstar,
                    "generated": res.stats.generated,
                    "re_expansions": res.stats.re_expansions,
                    "max_frontier": res.stats.max_frontier,
                    "depth": res.stats.depth,
                    "ebf": round(res.stats.ebf, 4),
                    "seconds": round(res.stats.seconds, 5),
                })
            costs = {r.cost for r in per.values()}
            assert len(costs) == 1, f"{inst.name}: heuristics disagree on cost {costs}"
    return rows


def report(rows: list[dict]) -> None:
    print("\nAggregate expansions per heuristic (lower is better):")
    for n in HORIZONS:
        sub = [r for r in rows if r["horizon"] == n]
        tot = {}
        for name in NAMES:
            rs = [r for r in sub if r["heuristic"].startswith(name)]
            tot[name] = (sum(r["expansions"] for r in rs),
                         sum(r["expansions_below_cstar"] for r in rs),
                         sum(r["re_expansions"] for r in rs))
        line = f"  n={n:2d}  " + "  ".join(
            f"{name}: {tot[name][0]:7d} (<C* {tot[name][1]:7d}, re-exp {tot[name][2]:3d})"
            for name in NAMES)
        print(line)
        print(f"        h0/h2 = {tot['h0'][0] / max(tot['h2'][0], 1):.2f}x   "
              f"h1/h2 = {tot['h1'][0] / max(tot['h2'][0], 1):.2f}x")

    # The fine print, counted rather than assumed away.
    inversions = 0
    pairs = 0
    for n in HORIZONS:
        for seed in SEEDS:
            got = {r["heuristic"][:2]: r for r in rows
                   if r["horizon"] == n and r["seed"] == seed}
            for weak, strong in (("h0", "h1"), ("h1", "h2")):
                pairs += 1
                if got[strong]["expansions"] > got[weak]["expansions"]:
                    inversions += 1
                assert (got[strong]["expansions_below_cstar"]
                        <= got[weak]["expansions"]), (n, seed, weak, strong)
    print(f"\n  total-expansion inversions (allowed by the theorem, at f == C*): "
          f"{inversions} / {pairs} instance-pairs")
    print("  surely-expanded (f < C*) dominance violations: 0 / "
          f"{pairs} -- asserted, not reported")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    print("Heuristic dominance")
    rows = run()
    out = RESULTS / "dominance.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
