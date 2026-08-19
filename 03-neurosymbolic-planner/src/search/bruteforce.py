"""Exhaustive enumeration -- the ground truth that makes the optimality test mean something.

This is a plain depth-first enumeration of *every* feasible complete plan, using the
same `domain.successors` as A*, and with none of A*'s machinery: no heuristic, no
closed set, no earliest-arrival dominance, no cost-bound pruning.  It is exponential
and it is meant to be; its only job is to be obviously correct.

That distinction is the point.  A*'s optimality rests on two arguments -- admissibility
of h, and the dominance claim that licenses keying the closed set on (loc, U, cap) and
discarding a later arrival at the same key.  Both are proved, and both are the kind of
proof that is easy to believe and easy to get subtly wrong in code.  Enumerating the
whole feasible set and comparing the minimum, exactly, on integer costs, tests the code
against the mathematics instead of against itself.

The search space is finite: every move either removes a parcel from U or is a reload,
and `successors` forbids consecutive depot visits and reloads at full capacity, so any
plan has at most 2n+1 moves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..domain import Instance, Move, State, successors
from .instrument import BudgetExceeded, SearchStats


@dataclass
class BruteResult:
    cost: int | None
    plan: list[Move]
    plans_examined: int = 0
    stats: SearchStats = field(default_factory=SearchStats)


def brute_force(inst: Instance, node_budget: int = 4_000_000,
                start: State | None = None) -> BruteResult:
    """Minimum-cost feasible plan by exhaustive enumeration.

    `start` defaults to the instance's initial state.  Passing an arbitrary reachable
    state instead gives the exact cost-to-go h*(s) from there, which is what the
    pointwise admissibility test needs: h(s) <= h*(s) has to hold at EVERY reachable
    state, not only at the root, and the root is the one place a bound is easiest to get
    right by accident.  Cost is measured from `start`'s clock, so it is comparable with
    a heuristic value directly.
    """
    stats = SearchStats(algorithm="brute-force", heuristic="-")
    t0 = time.perf_counter()
    best_cost: int | None = None
    best_plan: list[Move] = []
    plans = 0
    start = start if start is not None else inst.initial_state()
    origin = start.t
    path: list[Move] = []

    def rec(s: State) -> None:
        nonlocal best_cost, best_plan, plans
        stats.expansions += 1
        if stats.generated > node_budget:
            stats.seconds = time.perf_counter() - t0
            stats.budget_exceeded = True
            raise BudgetExceeded("brute-force", node_budget, stats)
        if inst.is_goal(s):
            plans += 1
            cost = s.t - origin
            if best_cost is None or cost < best_cost:
                best_cost, best_plan = cost, list(path)
            return                    # nothing left to deliver: this branch is done
        for mv, nxt, _c in successors(inst, s):
            stats.generated += 1
            path.append(mv)
            rec(nxt)
            path.pop()

    rec(start)
    stats.seconds = time.perf_counter() - t0
    stats.solved = best_cost is not None
    stats.depth = len(best_plan)
    return BruteResult(best_cost, best_plan, plans, stats)
