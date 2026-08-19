"""Admissible heuristics for the delivery domain, with the proofs.

THE LEMMA EVERYTHING RESTS ON.  Every move in `domain.successors` has cost

    c(s, s') = t' - t  >=  d(loc(s), loc(s'))                                    (L)

because the clock advances by at least the drive, and d is a metric on integer
coordinates (`domain.travel_matrix` uses ceil, not round, precisely to keep the
triangle inequality).  A completion of a state s is a walk

    loc(s) = v_0, v_1, ..., v_k = depot                                          (W)

whose vertex set is exactly U(s) union {loc(s), depot}: the only legal moves go to an
outstanding parcel or to the depot, so nothing else can appear.  By (L) the cost of
that completion is at least sum_i d(v_i, v_{i+1}), the walk's geometric length.  h0 and
h1 bound that geometric length and nothing else, so they throw away the waiting, the
handovers and the reloading entirely -- which is exactly where the slack in them lives.
h2 keeps the same drive-time bound and adds separate bounds on two of the three
discarded components; the waiting is the one nothing here bounds.

THE THREE HEURISTICS, in increasing strength:

    h0 = 0                         the control -- A* degenerates to Dijkstra
    h1 = max over q in U u {0} of  d(loc, q) + d(q, 0)
    h2 = max( h1, MST(U u {loc, 0}) )  +  sum_{p in U} service_p  +  reloads * reload_time

h2 >= h1 >= h0 pointwise by construction, so the dominance theorem applies:
if h_b >= h_a everywhere and both are admissible, A* with h_b expands no node that A*
with h_a does not (Pearl 1984).  The theorem is about nodes with f < C*; nodes
sitting exactly on f = C* are expanded or not according to tie-breaking, which is why
`astar.py` counts both totals and the surely-expanded set separately.

WHY h2 IS A MAX AND NOT JUST THE MST.  The MST bound does not dominate h1 on its own.
Put loc and q at distance 10 with the depot exactly between them, d(loc,0) = d(0,q) = 5:
MST({loc, 0, q}) = 10, but h1 = d(loc,q) + d(q,0) = 15.  The MST is blind to the fact
that the walk must *return*.  Taking the max of two admissible heuristics is admissible,
and the max of two consistent heuristics is consistent, so the combination costs nothing
in theory and buys a lot in practice.
"""

from __future__ import annotations

from ..domain import DEPOT, Instance, State


def mst_weight(travel: tuple[tuple[int, ...], ...], nodes: tuple[int, ...]) -> int:
    """Prim's algorithm, O(k^2), on the metric completion.  Integer weight."""
    k = len(nodes)
    if k <= 1:
        return 0
    in_tree = [False] * k
    best = [travel[nodes[0]][n] for n in nodes]
    in_tree[0] = True
    total = 0
    for _ in range(k - 1):
        j, bj = -1, None
        for i in range(k):
            if not in_tree[i] and (bj is None or best[i] < bj):
                j, bj = i, best[i]
        in_tree[j] = True
        total += bj
        row = travel[nodes[j]]
        for i in range(k):
            if not in_tree[i]:
                w = row[nodes[i]]
                if w < best[i]:
                    best[i] = w
    return total


class Heuristic:
    """Base class.  `__call__(inst, state) -> int` is the whole interface."""

    name = "h"
    admissible = True
    consistent = True
    description = ""

    def __call__(self, inst: Instance, s: State) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def reset(self) -> None:
        """Drop any memo tables.  Called between instances so timings are honest."""

    def __repr__(self) -> str:
        return f"<{self.name}>"


class ZeroHeuristic(Heuristic):
    """h0(s) = 0.

    ADMISSIBLE: h* >= 0 trivially, so 0 never overestimates.
    CONSISTENT: 0 <= c(s,s') + 0 for every edge, since costs are non-negative.

    A* with h0 is Dijkstra's algorithm.  It is here as the control: the point of the
    dominance experiment is the *ratio* of expansions, and a ratio needs a denominator
    that is not itself a heuristic.
    """

    name = "h0-zero"
    description = "zero (Dijkstra control)"

    def __call__(self, inst: Instance, s: State) -> int:
        return 0


class FarthestStopHeuristic(Heuristic):
    """h1(s) = max over q in U(s) u {depot} of [ d(loc, q) + d(q, depot) ].

    "Whatever else you do, you still have to reach the most awkward outstanding stop
    and then get home."

    ADMISSIBLE.  Fix q in U u {depot} and take any completion walk (W).  It visits q,
    say at index j (for q = depot, j = k works).  By the triangle inequality applied
    repeatedly, sum_{i<j} d(v_i,v_{i+1}) >= d(loc, q) and sum_{i>=j} d(v_i,v_{i+1}) >=
    d(q, depot).  So the walk's length is at least d(loc,q) + d(q,depot), for every q;
    hence at least the max.  With (L), h1 <= h*.  Including q = depot is what supplies
    the final drive home once U is empty -- without it h1 would collapse to 0 one move
    too early, and would stop being consistent.

    CONSISTENT.  Let s -> s' be any edge, c = c(s,s') >= d(loc, loc') by (L).
    Write q* for the argmax at s.
      * q* != loc':  then q* is still in U(s') u {depot}, and
        h1(s) = d(loc,q*) + d(q*,0) <= d(loc,loc') + d(loc',q*) + d(q*,0) <= c + h1(s').
      * q* == loc'  (the move delivers q*, or drives to the depot when q* = depot):
        h1(s) = d(loc,loc') + d(loc',0) <= c + d(loc',0), and d(loc',0) =
        d(loc',0) + d(0,0) <= h1(s') because depot is always in the max's index set.
    Both cases give h1(s) <= c + h1(s'), and h1(goal) = 0.  So f is non-decreasing along
    every path, and A* never re-expands a closed node -- asserted in the test suite by
    checking the re-expansion counter is exactly 0.
    """

    name = "h1-farthest"
    description = "max over outstanding stops of (drive there + drive home)"

    def __call__(self, inst: Instance, s: State) -> int:
        travel = inst.travel
        row = travel[s.loc]
        best = row[DEPOT]                     # the q = depot term
        mask, p = s.undelivered, 1
        while mask:
            if mask & 2:
                v = row[p] + travel[p][DEPOT]
                if v > best:
                    best = v
            mask >>= 1
            p += 1
        return best


class MSTHeuristic(Heuristic):
    """h2(s) = max( h1(s), MST(U u {loc, depot}) ) + service(U) + reloads(U, cap)*reload_time.

    Three lower bounds on three *disjoint* components of the remaining cost.  The clock
    from here to the depot is spent driving, waiting, serving, and reloading; bound each
    piece separately and the sum is still a bound.  h1 and the MST term bound driving
    only, which is why they leave so much on the table in a domain where a parcel's
    service time is comparable to a leg of the route.

    (a) THE SPANNING-TREE TERM -- the relaxed problem.  Drop the time windows, drop the
        capacity, drop the requirement that the route be a *walk* at all, and keep only
        "the outstanding parcels, where I am, and the depot must end up connected".  The
        cheapest connected structure on a vertex set is its minimum spanning tree, and
        that is solvable exactly in O(k^2) by Prim -- the textbook recipe for a
        heuristic: relax until the relaxed problem is polynomial, then solve it exactly.
        ADMISSIBLE: by (W) the completion walk's vertex set is exactly U u {loc, depot}
        and its edges connect that set; a connected spanning subgraph weighs at least
        its minimum spanning tree (delete edges until a tree remains, weights being
        non-negative).  With (L), MST <= remaining drive time.

    (b) THE SERVICE TERM.  S(U) = sum_{p in U} service_p.  Every outstanding parcel is
        handed over exactly once, and that time is not drive time.  ADMISSIBLE by
        inspection.

    (c) THE RELOAD TERM.  With D(U) = sum_{p in U} demand_p units still to hand over and
        cap on board, r reloads leave cap + r*capacity units available, so any feasible
        completion needs at least r = max(0, ceil((D(U) - cap) / capacity)) of them, each
        costing at least reload_time on top of its drive.  ADMISSIBLE by the same
        counting argument.  This is the only term that sees the capacity constraint at
        all, and it is what stops A* from cheerfully exploring plans that have stranded
        the load on the wrong side of town.

    CONSISTENT.  Consistency does not compose by adding consistent pieces, so the deficit
    h(s) - h(s') is bounded term by term against one edge cost c = c(s,s'):
      * MST term.  Let T' minimally span U(s') u {loc', depot}.  For `deliver`,
        U(s) u {loc, depot} = U(s') u {loc', depot} u {loc}, and T' + the edge (loc,loc')
        spans it, so MST(s) - MST(s') <= d(loc, loc').  For `reload`/`return`, loc' =
        depot is already in T' and the same edge addition works.  h1 obeys the same bound
        (proved in `FarthestStopHeuristic`), and max of two functions each satisfying
        deficit <= d(loc,loc') satisfies it too.
      * Service term.  `deliver p` gives S(s) - S(s') = service_p; `reload`/`return`
        give 0.
      * Reload term.  `deliver p` decreases D and cap by exactly demand_p, so D - cap is
        unchanged and the deficit is 0.  `reload` leaves D alone and sets cap = capacity;
        since 0 <= cap < capacity, r(s) <= ceil(D/capacity) and r(s') = ceil(D/capacity)
        - 1, so the deficit is at most one reload_time.  `return` has D = 0 on both sides.
      Summing per move type: `deliver` gives d(loc,loc') + service_p <= c; `reload` gives
      d(loc,loc') + reload_time <= c; `return` gives d(loc,loc') <= c.  Hence
      h2(s) <= c + h2(s') everywhere, and h2(goal) = 0.

    DOMINANCE.  h2 >= h1 by construction, since (b) and (c) are non-negative and the max
    already contains h1.  Both are admissible, so Pearl's theorem applies and h2 expands
    no node with f < C* that h1 does not.

    Memoized on (loc, U, cap); none of the three terms looks at the clock, which is also
    why the heuristic is well defined on the (loc, U, cap) closed-set key.
    """

    name = "h2-mst"
    description = "max(h1, MST over the outstanding set) + service + forced reloads"

    def __init__(self) -> None:
        self._cache: dict[tuple[int, int, int], int] = {}
        self._h1 = FarthestStopHeuristic()
        self._inst: Instance | None = None

    def reset(self) -> None:
        self._cache.clear()
        self._inst = None

    def __call__(self, inst: Instance, s: State) -> int:
        # The memo is keyed on the state only, so it is valid for one instance and
        # catastrophically wrong for the next.  Detect the switch here rather than
        # relying on every caller to remember reset(): a stale cache would silently
        # return a number that is not a bound at all, and silent is the one thing a
        # heuristic must never be.
        if inst is not self._inst:
            self._cache.clear()
            self._inst = inst
        k = (s.loc, s.undelivered, s.cap)
        got = self._cache.get(k)
        if got is not None:
            return got

        nodes = [s.loc] if s.loc != DEPOT else []
        nodes.append(DEPOT)
        stops = inst.stops_in(s.undelivered)
        nodes.extend(stops)
        drive = mst_weight(inst.travel, tuple(nodes))
        h1 = self._h1(inst, s)
        if h1 > drive:
            drive = h1

        service = sum(inst.service[p] for p in stops)
        demand = sum(inst.demand[p] for p in stops)
        shortfall = demand - s.cap
        reloads = -(-shortfall // inst.capacity) if shortfall > 0 else 0

        got = drive + service + reloads * inst.reload_time
        self._cache[k] = got
        return got


def _mix(x: int) -> int:
    """splitmix64 finalizer -- a deterministic, process-independent state hash.

    Python's built-in hash() is salted per process for strings and would make the
    inconsistency experiment irreproducible, so the mixing is written out.
    """
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


class InconsistentAdmissibleHeuristic(Heuristic):
    """h2 suppressed to 0 on a deterministic pseudo-random share of states.

    ADMISSIBLE: its value at every state is either 0 or h2(s), and both are lower
    bounds on h*(s).  Admissibility is a pointwise property, so it survives.

    NOT CONSISTENT: consistency is a property of *edges*, and the suppression is
    applied without reference to them.  Whenever an unsuppressed s sits next to a
    suppressed s' the edge inequality h(s) <= c(s,s') + h(s') = c(s,s') reads
    h2(s) <= c(s,s'), which fails as soon as the bound exceeds one move's cost.

    This exists to make the admissibility-vs-consistency distinction *measurable*
    rather than merely stated.  A* with this heuristic still returns the optimum --
    because the implementation re-opens a closed node when a strictly cheaper path to
    it is found -- but it pays for it with re-expansions, and the re-expansion counter
    is the observable.  With a consistent heuristic that counter is provably 0: along
    any path f is non-decreasing, so the first time a node is popped its g is already
    optimal.  Suppression is keyed on (loc, U) only, so the heuristic is a pure
    function of the closed-set key and the experiment reproduces exactly.
    """

    name = "hx-inconsistent"
    admissible = True
    consistent = False
    description = "h2, zeroed on a pseudo-random half of states (admissible, not consistent)"

    def __init__(self, suppress: float = 0.5, salt: int = 0x9E3779B97F4A7C15) -> None:
        self.suppress = suppress
        self.salt = salt
        self._base = MSTHeuristic()
        self._threshold = int(suppress * (1 << 64))

    def reset(self) -> None:
        self._base.reset()

    def __call__(self, inst: Instance, s: State) -> int:
        if _mix((s.loc * 0x100000001B3) ^ (s.undelivered * 0x9E3779B1) ^ self.salt) \
                < self._threshold:
            return 0
        return self._base(inst, s)


HEURISTICS: dict[str, type[Heuristic]] = {
    "h0": ZeroHeuristic,
    "h1": FarthestStopHeuristic,
    "h2": MSTHeuristic,
    "hx": InconsistentAdmissibleHeuristic,
}


def make(name: str) -> Heuristic:
    if name not in HEURISTICS:
        raise KeyError(f"unknown heuristic {name!r}; have {sorted(HEURISTICS)}")
    return HEURISTICS[name]()
