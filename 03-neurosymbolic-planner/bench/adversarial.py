#!/usr/bin/env python3
"""Minimax vs alpha-beta vs alpha-beta with move ordering.  Writes results/adversarial.csv.

Two things are measured and they answer different questions.

VALUE.  Every variant must return the identical number.  Pruning that changes the answer
is a bug, and the `value` column exists so that a reader can check the claim rather than
take it on trust.

NODES, AND THE BRANCHING FACTOR.  Knuth & Moore (1975): with perfect move ordering
alpha-beta visits O(b^(d/2)) leaves rather than O(b^d) -- the same budget buys twice the
depth.  The effective branching factor is fitted from N = b* + b*^2 + ... + b*^d with N
the nodes generated, so a variant that halves the exponent shows up as b* moving toward
sqrt(b).

WHAT THIS DOMAIN CAN AND CANNOT SHOW.  The b^(d/2) result is asymptotic, for a UNIFORM
tree of branching factor b under PERFECT ordering.  Here the branching factor is the
number of unclaimed stops, so it falls by one at every ply, and the ordering is a cheap
static heuristic rather than an oracle.  So the measurement cannot land on sqrt(b) and
this benchmark does not pretend it does: the `nominal_b` and `sqrt_nominal_b` columns are
printed next to the measured b* so the reader can see the direction of travel and judge
the size of it.
"""

from __future__ import annotations

import csv
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.domain import make_instance                                # noqa: E402
from src.search.alphabeta import (alphabeta, iterative_deepening,   # noqa: E402
                                  make_contest, minimax)

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
STOPS = (9,)
SEEDS = range(6)
DEPTHS = (4, 6, 8)


def run() -> list[dict]:
    rows: list[dict] = []
    for n in STOPS:
        for seed in SEEDS:
            contest = make_contest(make_instance(n, seed), seed=seed)
            for depth in DEPTHS:
                variants = {
                    "minimax": lambda d=depth: minimax(contest, d),
                    "alpha-beta": lambda d=depth: alphabeta(contest, d, ordering=False,
                                                            use_tt=False),
                    "alpha-beta+ordering": lambda d=depth: alphabeta(contest, d,
                                                                     ordering=True,
                                                                     use_tt=False),
                    "alpha-beta+ordering+tt": lambda d=depth: alphabeta(contest, d,
                                                                        ordering=True,
                                                                        use_tt=True),
                    "alpha-beta+iterative-deepening": lambda d=depth:
                        iterative_deepening(contest, d),
                }
                ref = None
                for label, fn in variants.items():
                    res = fn()
                    if ref is None:
                        ref = res.value
                    assert res.value == ref, (
                        f"{contest.name} d={depth}: {label} returned {res.value}, "
                        f"minimax returned {ref}")
                    rows.append({
                        "n_stops": n, "seed": seed, "depth": depth, "variant": label,
                        "value": res.value, "best_move": res.best_move,
                        "expansions": res.stats.expansions,
                        "generated": res.stats.generated,
                        "ebf": round(res.stats.ebf, 4),
                        "nominal_b": n,
                        "sqrt_nominal_b": round(math.sqrt(n), 4),
                        "seconds": round(res.stats.seconds, 5),
                    })
    return rows


def report(rows: list[dict]) -> None:
    labels = ["minimax", "alpha-beta", "alpha-beta+ordering",
              "alpha-beta+ordering+tt", "alpha-beta+iterative-deepening"]
    print("\nNodes expanded, summed over seeds; b* averaged.  All variants returned "
          "identical values (asserted).")
    for depth in DEPTHS:
        print(f"  depth {depth}:")
        base = None
        for label in labels:
            sub = [r for r in rows if r["depth"] == depth and r["variant"] == label]
            tot = sum(r["expansions"] for r in sub)
            ebf = sum(r["ebf"] for r in sub) / len(sub)
            base = base or tot
            print(f"    {label:32s} {tot:9d} nodes  ({base / tot:6.1f}x fewer than "
                  f"minimax)  b* = {ebf:.3f}")
    n = rows[0]["n_stops"]
    print(f"\n  nominal branching factor at the root b = {n}, sqrt(b) = {math.sqrt(n):.3f}")
    print("  the tree is not uniform -- branching falls by one every ply -- so b* is a "
          "summary statistic,\n  and what the table shows is the direction and size of "
          "the movement, not attainment of sqrt(b).")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    print("Adversarial search: minimax vs alpha-beta")
    rows = run()
    out = RESULTS / "adversarial.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
