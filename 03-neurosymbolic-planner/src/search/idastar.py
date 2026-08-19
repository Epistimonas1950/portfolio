"""IDA* -- A* with A*'s memory removed (Korf 1985).

A* stores every node it generates.  On this domain the frontier grows like the number
of (loc, U, cap) triples, i.e. O(n * 2^n * capacity), and that is the resource that runs
out first.  IDA* replaces the priority queue with a sequence of depth-first searches,
each cut off at an f-bound:

    bound_0 = h(start)
    bound_{i+1} = min { f(s) : s generated in round i with f(s) > bound_i }

Each round is a plain DFS with O(d) memory.  Because the next bound is the *smallest*
f that overflowed the last one, no node with f < C* is ever skipped, and the first round
that reaches a goal reaches it with f = g = C*.  So IDA* is optimal under exactly the
same condition as A*: h admissible.  It is asserted here by equality with A*'s cost.

WHAT IT COSTS.  IDA* keeps no closed set, so it cannot detect transpositions, and this
domain is nothing but transpositions -- every permutation of the same delivered prefix
lands on the same (loc, U, cap).  A* collapses those; IDA* re-explores them, in every
round.  The only pruning kept here is a cycle check against the states on the current
path, which is free and cannot discard an optimal plan (any plan revisiting a key at
equal-or-later time is dominated by the one that skips the loop).  The measured
node-count ratio against A* is the finding, and it is reported rather than hidden:
IDA* is the right algorithm when memory is the binding constraint and the state graph
is close to a tree, and this state graph is nothing like a tree.

Integer costs make the bound sequence finite and well behaved; with float costs the
classic failure mode is a round that advances the bound by an epsilon and re-expands
everything for one extra node.
"""

from __future__ import annotations

import time

from ..domain import Instance, Move, State, successors
from .astar import SearchResult
from .heuristics import Heuristic, ZeroHeuristic
from .instrument import BudgetExceeded, SearchStats

_INF = float("inf")


def idastar(inst: Instance, h: Heuristic | None = None,
            node_budget: int = 2_000_000) -> SearchResult:
    """Optimal plan for `inst` in memory linear in the plan length."""
    h = h or ZeroHeuristic()
    h.reset()
    stats = SearchStats(algorithm="IDA*", heuristic=h.name)
    t0 = time.perf_counter()

    start = inst.initial_state()
    bound: float = h(inst, start)
    plan: list[Move] = []
    path_keys: set[tuple[int, int, int]] = {start.key}
    found_cost: int | None = None

    def dfs(s: State, g: int, bound: float) -> float:
        """Return FOUND-sentinel -1.0, or the least f > bound seen in this subtree."""
        nonlocal found_cost
        stats.expansions += 1
        f = g + h(inst, s)
        if f > bound:
            return float(f)
        if inst.is_goal(s):
            found_cost = g
            return -1.0
        best = _INF
        for mv, nxt, c in successors(inst, s):
            if nxt.key in path_keys:
                continue                    # cycle: dominated by the loop-free prefix
            stats.generated += 1
            if stats.generated > node_budget:
                stats.seconds = time.perf_counter() - t0
                stats.budget_exceeded = True
                raise BudgetExceeded("IDA*", node_budget, stats)
            path_keys.add(nxt.key)
            plan.append(mv)
            t = dfs(nxt, g + c, bound)
            if t == -1.0:
                return -1.0
            plan.pop()
            path_keys.discard(nxt.key)
            if t < best:
                best = t
        return best

    while True:
        stats.iterations += 1
        t = dfs(start, 0, bound)
        if t == -1.0:
            break
        if t == _INF:
            break                            # no feasible plan at any bound
        bound = t

    stats.seconds = time.perf_counter() - t0
    stats.solved = found_cost is not None
    stats.depth = len(plan) if found_cost is not None else 0
    return SearchResult(found_cost, list(plan) if found_cost is not None else [], stats)
