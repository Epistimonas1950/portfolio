"""A* with a pluggable heuristic and the instrumentation the claims need.

    f(s) = g(s) + h(s),      g(s) = t(s) - start_time  (elapsed minutes)

expand the open node of least f, stop when a goal is popped.  With h admissible the
first goal popped is optimal (Hart, Nilsson & Raphael 1968).

THREE IMPLEMENTATION DECISIONS THAT ARE ACTUALLY MATHEMATICAL:

1. THE CLOSED-SET KEY IS (loc, U, cap), NOT THE STATE.  The clock is g itself.  Two
   arrivals at the same key differ only in when they happened, and the earliest-arrival
   dominance proved in `domain` says the earlier one is at least as good.  Keying on the
   full state instead would be correct but would deduplicate almost nothing, and the
   whole point of the heuristic comparison would be swamped by the transposition rate.

2. RE-OPENING IS ON.  A closed node is expanded again if a strictly cheaper path to it
   turns up.  With a consistent heuristic that never happens -- f is non-decreasing
   along every path, so a node's first pop already has optimal g -- and the counter
   `re_expansions` is provably 0.  With a merely admissible heuristic it can happen, and
   without re-opening the returned cost could be too high.  Keeping re-opening on is
   what makes optimality depend on admissibility alone, and it turns the
   admissibility-vs-consistency distinction into a number you can read off a CSV.

3. TIE-BREAKING IS (f, h, insertion order), IDENTICALLY FOR EVERY HEURISTIC.  The
   dominance theorem covers nodes with f < C*; nodes with f == C* are expanded or not
   according to the tie-break, and with integer minutes ties are everywhere.  So the
   stats carry `expansions` and `expansions_below_cstar` separately: the second is the
   quantity the theorem actually predicts, and it is the one the test asserts on
   per instance.  Preferring smaller h among equal f is the usual choice -- it favours
   nodes nearer the goal and shortens the tail of the search.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

from ..domain import Instance, Move, State, successors
from .heuristics import Heuristic, ZeroHeuristic
from .instrument import BudgetExceeded, SearchStats


@dataclass
class SearchResult:
    cost: int | None
    plan: list[Move] = field(default_factory=list)
    stats: SearchStats = field(default_factory=SearchStats)

    @property
    def solved(self) -> bool:
        return self.cost is not None


def astar(inst: Instance, h: Heuristic | None = None,
          node_budget: int = 2_000_000, reopen: bool = True) -> SearchResult:
    """Optimal plan for `inst` under any admissible `h`.

    node_budget caps *generated* nodes; exceeding it raises BudgetExceeded carrying the
    stats so far, so a benchmark can report where the wall is instead of hanging.
    """
    h = h or ZeroHeuristic()
    h.reset()
    stats = SearchStats(algorithm="A*", heuristic=h.name)
    t0 = time.perf_counter()

    start = inst.initial_state()
    g0 = 0
    h0 = h(inst, start)
    counter = 0
    open_heap: list[tuple[int, int, int, State]] = [(g0 + h0, h0, counter, start)]
    best_g: dict[tuple[int, int, int], int] = {start.key: g0}
    parent: dict[tuple[int, int, int], tuple[State, Move]] = {}
    closed: set[tuple[int, int, int]] = set()
    f_hist: dict[int, int] = {}
    stats.generated = 1

    goal_state: State | None = None
    cost: int | None = None

    while open_heap:
        stats.max_frontier = max(stats.max_frontier, len(open_heap))
        f, _hv, _c, s = heapq.heappop(open_heap)
        k = s.key
        g = s.t - inst.start_time
        if g > best_g.get(k, g):
            continue                                # stale queue entry
        if k in closed:
            if not reopen:
                continue
            stats.re_expansions += 1                # only reachable via a cheaper path
        closed.add(k)
        stats.expansions += 1
        f_hist[f] = f_hist.get(f, 0) + 1

        if inst.is_goal(s):
            goal_state, cost = s, g
            break

        for mv, nxt, c in successors(inst, s):
            stats.generated += 1
            if stats.generated > node_budget:
                stats.seconds = time.perf_counter() - t0
                stats.budget_exceeded = True
                raise BudgetExceeded("A*", node_budget, stats)
            nk = nxt.key
            ng = nxt.t - inst.start_time
            if ng < best_g.get(nk, 1 << 62):
                best_g[nk] = ng
                parent[nk] = (s, mv)
                hv = h(inst, nxt)
                counter += 1
                heapq.heappush(open_heap, (ng + hv, hv, counter, nxt))

    stats.seconds = time.perf_counter() - t0
    stats.solved = cost is not None
    stats.f_at_expansion = f_hist
    plan: list[Move] = []
    if goal_state is not None:
        k = goal_state.key
        # `reopen=False` can leave a parent pointer that no longer belongs to the path
        # A* actually took, so the walk back is guarded rather than trusted.  With
        # re-opening on -- the default, and the only configuration any result in this
        # repo uses -- the chain is a simple path and the guard never fires.
        seen_keys = {k}
        while k in parent:
            prev, mv = parent[k]
            plan.append(mv)
            k = prev.key
            if k in seen_keys:
                break
            seen_keys.add(k)
        plan.reverse()
        stats.depth = len(plan)
        stats.expansions_below_cstar = sum(n for fv, n in f_hist.items() if fv < cost)
    return SearchResult(cost, plan, stats)
