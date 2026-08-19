"""The formal problem: multi-stop delivery with time windows and vehicle capacity.

STATE.  One vehicle, one depot (node 0), n parcels at n distinct stops (nodes 1..n).

    s = (loc, U, t, cap)

    loc   node the vehicle is standing at
    U     bitmask of parcels not yet delivered (bit p set <=> parcel p outstanding)
    t     clock, in whole minutes
    cap   units of load still on board

MOVES.  Three, and their costs are all "time elapsed":

    deliver p   requires p in U and demand[p] <= cap.
                arrive = t + travel[loc][p];  infeasible if arrive > latest[p]
                start  = max(arrive, earliest[p])          <- waiting, and it is paid for
                t'     = start + service[p]
                cap'   = cap - demand[p],   U' = U without p

    reload      requires U non-empty, loc != 0, cap < capacity.
                t' = max(t + travel[loc][0], earliest[0]) + reload_time,  cap' = capacity

    return      requires U empty, loc != 0.  t' = max(t + travel[loc][0], earliest[0])

COST.  c(s, s') = t' - t >= travel[loc][loc'] >= 0 for all three moves.  That single
inequality is the lemma every admissibility and consistency proof in `heuristics.py`
rests on, so it is asserted in the test suite rather than assumed.  Plan cost is the
final clock reading, i.e. travel + waiting + service + reloading.

WHY THE CLOSED SET IS STILL SAFE.  Edge *feasibility* depends on t, so this is not
literally a static shortest-path problem.  It is a FIFO time-dependent one, and the
earliest-arrival dominance holds: if the same (loc, U, cap) is reachable at t1 <= t2,
then every completion feasible from t2 is feasible from t1 and costs no more.  Both
`arrive <= latest[p]` and `t' = max(t + travel, earliest) + service` are monotone
non-decreasing in t, so the claim follows by induction on the completion's length.
Therefore A* may key its closed set on (loc, U, cap) alone and keep the smallest t
seen -- which is exactly what `astar.py` does, and what `bruteforce.py` deliberately
does not, so that the optimality test checks the argument rather than restating it.

GEOMETRY.  travel[i][j] = ceil(euclidean distance between integer grid coordinates).
The ceiling matters: rounding to nearest can break the triangle inequality, whereas
ceil(a + b) <= ceil(a) + ceil(b) preserves it, and the heuristics below are only
admissible on a metric.  Integer minutes throughout also means plan costs are exact
integers, so `assertEqual(astar_cost, brute_force_cost)` is an honest exact test with
no tolerance to hide behind.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Iterable, NamedTuple

import numpy as np

DEPOT = 0


class State(NamedTuple):
    """A search node.  `key` deliberately drops `t`; see the module docstring."""

    loc: int
    undelivered: int
    t: int
    cap: int

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.loc, self.undelivered, self.cap)


class Move(NamedTuple):
    kind: str      # "deliver" | "reload" | "return"
    target: int    # stop index for deliver, 0 otherwise


@dataclass(frozen=True)
class Instance:
    """An immutable problem instance.  All times are whole minutes from `start_time`."""

    name: str
    n_stops: int
    coords: tuple[tuple[int, int], ...]
    travel: tuple[tuple[int, ...], ...]
    demand: tuple[int, ...]
    earliest: tuple[int, ...]
    latest: tuple[int, ...]
    service: tuple[int, ...]
    capacity: int
    reload_time: int
    start_time: int = 0
    stop_names: tuple[str, ...] = ()
    reference_cost: int | None = None   # cost of the plan the generator built the
                                        # windows around; a valid upper bound on the
                                        # optimum, and proof the instance is feasible

    @property
    def n_nodes(self) -> int:
        return self.n_stops + 1

    @property
    def all_mask(self) -> int:
        return ((1 << self.n_stops) - 1) << 1     # bit p <-> stop p, bit 0 unused

    def initial_state(self) -> State:
        return State(DEPOT, self.all_mask, self.start_time, self.capacity)

    def is_goal(self, s: State) -> bool:
        return s.undelivered == 0 and s.loc == DEPOT

    def stops_in(self, mask: int) -> tuple[int, ...]:
        return tuple(p for p in range(1, self.n_nodes) if mask >> p & 1)

    def label(self, node: int) -> str:
        if node == DEPOT:
            return "depot"
        if self.stop_names and node - 1 < len(self.stop_names):
            return self.stop_names[node - 1]
        return f"stop{node}"


# --------------------------------------------------------------------------------- #
# transitions
# --------------------------------------------------------------------------------- #

def successors(inst: Instance, s: State) -> list[tuple[Move, State, int]]:
    """All feasible (move, next state, cost) triples out of `s`.

    Two restrictions are baked in -- no reload while already at the depot, and no
    reload with a full load -- because both moves are provably non-improving (they
    strictly increase t and change nothing else), so excluding them removes no optimal
    plan.  They live here, in the shared successor function, so that A* and the brute
    force enumerate literally the same graph and the optimality test compares search
    strategies rather than problem definitions.
    """
    out: list[tuple[Move, State, int]] = []
    row = inst.travel[s.loc]

    p = 1
    mask = s.undelivered
    while mask:
        if mask & 2:                                # bit p of the original mask
            if inst.demand[p] <= s.cap:
                arrive = s.t + row[p]
                if arrive <= inst.latest[p]:
                    start = arrive if arrive >= inst.earliest[p] else inst.earliest[p]
                    t2 = start + inst.service[p]
                    out.append((Move("deliver", p),
                                State(p, s.undelivered & ~(1 << p), t2,
                                      s.cap - inst.demand[p]),
                                t2 - s.t))
        mask >>= 1
        p += 1

    if s.loc != DEPOT:
        arrive = s.t + row[DEPOT]
        if arrive <= inst.latest[DEPOT]:
            start = max(arrive, inst.earliest[DEPOT])
            if s.undelivered == 0:
                out.append((Move("return", DEPOT),
                            State(DEPOT, 0, start, s.cap), start - s.t))
            elif s.cap < inst.capacity:
                t2 = start + inst.reload_time
                out.append((Move("reload", DEPOT),
                            State(DEPOT, s.undelivered, t2, inst.capacity), t2 - s.t))
    return out


def apply_move(inst: Instance, s: State, move: Move) -> tuple[State, int]:
    """Apply `move` to `s`, raising if it is not legal.  Used by the validator."""
    for mv, nxt, cost in successors(inst, s):
        if mv == move:
            return nxt, cost
    raise ValueError(f"move {move} is not legal in state {s}")


# --------------------------------------------------------------------------------- #
# validation -- deliberately an independent re-simulation, not a call to successors()
# --------------------------------------------------------------------------------- #

@dataclass
class Validation:
    feasible: bool
    cost: int | None
    violations: list[str]
    timeline: list[dict]


def validate_plan(inst: Instance, plan: Iterable[Move]) -> Validation:
    """Check a plan against every hard constraint, from scratch.

    Written as a fresh forward simulation with explicit checks rather than by replaying
    `successors`, so that it can catch a bug in `successors` instead of inheriting it.
    Reports *all* violations, not the first, because "which constraint did I break" is
    the question a user of the explanation layer actually asks.
    """
    plan = list(plan)
    violations: list[str] = []
    timeline: list[dict] = []

    loc, t, cap = DEPOT, inst.start_time, inst.capacity
    delivered: list[int] = []

    for i, mv in enumerate(plan):
        target = mv.target if mv.kind == "deliver" else DEPOT
        if mv.kind == "deliver":
            if not 1 <= target < inst.n_nodes:
                violations.append(f"move {i}: stop {target} does not exist")
                continue
            if target in delivered:
                violations.append(f"move {i}: parcel {inst.label(target)} delivered twice")
            if inst.demand[target] > cap:
                violations.append(
                    f"move {i}: capacity -- {inst.label(target)} needs "
                    f"{inst.demand[target]} unit(s), {cap} on board")
            delivered.append(target)
        elif mv.kind not in ("reload", "return"):
            violations.append(f"move {i}: unknown move kind {mv.kind!r}")
            continue

        arrive = t + inst.travel[loc][target]
        if arrive > inst.latest[target]:
            violations.append(
                f"move {i}: time window -- arrive at {inst.label(target)} at "
                f"{fmt_time(arrive)}, window closes {fmt_time(inst.latest[target])}")
        start = max(arrive, inst.earliest[target])
        wait = start - arrive
        if mv.kind == "deliver":
            t = start + inst.service[target]
            cap -= inst.demand[target]
        elif mv.kind == "reload":
            t = start + inst.reload_time
            cap = inst.capacity
        else:
            t = start
        timeline.append({"move": mv.kind, "node": target,
                         "label": inst.label(target), "arrive": arrive,
                         "wait": wait, "depart": t, "cap_after": cap})
        loc = target

    missing = [p for p in range(1, inst.n_nodes) if p not in delivered]
    if missing:
        violations.append("undelivered parcels: "
                          + ", ".join(inst.label(p) for p in missing))
    if loc != DEPOT:
        violations.append(f"plan ends at {inst.label(loc)}, not back at the depot")

    ok = not violations
    return Validation(ok, t - inst.start_time if ok else None, violations, timeline)


def plan_cost(inst: Instance, plan: Iterable[Move]) -> int:
    v = validate_plan(inst, plan)
    if not v.feasible:
        raise ValueError("infeasible plan: " + "; ".join(v.violations))
    assert v.cost is not None
    return v.cost


def fmt_time(minutes: int) -> str:
    """Minutes-from-midnight to HH:MM.  Instances start at 08:00 by convention."""
    h, m = divmod(int(minutes), 60)
    return f"{h % 24:02d}:{m:02d}"


# --------------------------------------------------------------------------------- #
# instance generation
# --------------------------------------------------------------------------------- #

def travel_matrix(coords: Iterable[tuple[int, int]]) -> tuple[tuple[int, ...], ...]:
    """ceil of the Euclidean distance -- integer, symmetric, and metric.

    ceil is not cosmetic.  round() can violate the triangle inequality (three points
    with true distances 1.4, 1.4, 2.8 round to 1, 1, 3), and every heuristic in
    `heuristics.py` needs d(i,j) <= d(i,k) + d(k,j) to be admissible.  ceil is safe
    because ceil(a + b) <= ceil(a) + ceil(b).
    """
    pts = np.asarray(list(coords), dtype=float)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    m = np.ceil(d - 1e-12).astype(int)     # the epsilon keeps exact integers exact
    np.fill_diagonal(m, 0)
    return tuple(tuple(int(v) for v in row) for row in m)


def make_instance(n_stops: int, seed: int, capacity: int = 4, area: int = 40,
                  demand_max: int = 2, service: int = 6, reload_time: int = 10,
                  start_time: int = 480, slack_before: int = 120,
                  slack_after: int = 180, name: str | None = None) -> Instance:
    """A seeded random instance whose feasibility is guaranteed by construction.

    `n_stops` is the horizon knob.  Random time windows would make most instances
    infeasible and the benchmark meaningless, so instead the generator draws a random
    reference route, simulates it (inserting a depot reload whenever the load runs
    out), and hangs each window around the arrival time that route produced:

        [earliest_p, latest_p] = [a_p - Uniform(0, slack_before),
                                  a_p + Uniform(0, slack_after)]

    Every instance therefore admits at least one feasible plan -- the reference route,
    whose cost is stored as `reference_cost` and is a valid upper bound on the optimum.
    The slacks are the tightness knob: shrink them and the feasible set collapses
    toward that single route; widen them and the time windows stop binding.
    """
    if n_stops < 1:
        raise ValueError("n_stops must be at least 1")
    rng = np.random.default_rng(seed)

    # Distinct integer coordinates: coincident stops give zero-length edges, which are
    # legal but make the dominance measurements degenerate.
    seen: set[tuple[int, int]] = {(area // 2, area // 2)}
    coords = [(area // 2, area // 2)]
    while len(coords) < n_stops + 1:
        pt = (int(rng.integers(0, area + 1)), int(rng.integers(0, area + 1)))
        if pt not in seen:
            seen.add(pt)
            coords.append(pt)

    travel = travel_matrix(coords)
    demand = [0] + [int(rng.integers(1, demand_max + 1)) for _ in range(n_stops)]
    svc = [0] + [service] * n_stops

    order = list(rng.permutation(np.arange(1, n_stops + 1)))
    loc, t, cap = DEPOT, start_time, capacity
    arrival: dict[int, int] = {}
    ref_plan: list[Move] = []
    for p in order:
        p = int(p)
        if demand[p] > cap:
            t = t + travel[loc][DEPOT] + reload_time
            loc, cap = DEPOT, capacity
            ref_plan.append(Move("reload", DEPOT))
        t += travel[loc][p]
        arrival[p] = t
        t += svc[p]
        cap -= demand[p]
        loc = p
        ref_plan.append(Move("deliver", p))
    t += travel[loc][DEPOT]
    ref_plan.append(Move("return", DEPOT))
    ref_end = t

    earliest = [0] * (n_stops + 1)
    latest = [0] * (n_stops + 1)
    for p in range(1, n_stops + 1):
        lo = arrival[p] - int(rng.integers(0, slack_before + 1))
        hi = arrival[p] + int(rng.integers(0, slack_after + 1))
        earliest[p] = max(start_time, lo)
        latest[p] = max(hi, arrival[p])
    earliest[DEPOT] = 0
    # The depot must stay open long enough for the reference route plus any detour a
    # better plan might take; a hard depot close would silently prune optimal plans.
    latest[DEPOT] = ref_end + slack_after + 240

    inst = Instance(
        name=name or f"rand-n{n_stops}-s{seed}", n_stops=n_stops,
        coords=tuple(coords), travel=travel, demand=tuple(demand),
        earliest=tuple(earliest), latest=tuple(latest), service=tuple(svc),
        capacity=capacity, reload_time=reload_time, start_time=start_time,
        stop_names=tuple(f"stop{p}" for p in range(1, n_stops + 1)),
    )
    ref_cost = plan_cost(inst, ref_plan)
    return Instance(**{**inst.__dict__, "reference_cost": ref_cost})


# --------------------------------------------------------------------------------- #
# JSON <-> Instance, the interface the translation layer targets
# --------------------------------------------------------------------------------- #

def spec_from_instance(inst: Instance) -> dict:
    """The canonical JSON form: what `translate.RuleBackend` has to produce."""
    return {
        "capacity": inst.capacity,
        "service_minutes": int(inst.service[1]) if inst.n_stops else 0,
        "reload_minutes": inst.reload_time,
        "start_time": inst.start_time,
        "stops": [
            {"name": inst.label(p), "demand": int(inst.demand[p]),
             "earliest": int(inst.earliest[p]), "latest": int(inst.latest[p])}
            for p in range(1, inst.n_nodes)
        ],
    }


def instance_from_spec(spec: dict, gazetteer: dict[str, list[int]],
                       name: str = "from-spec") -> Instance:
    """Build a solvable Instance from a translated spec plus a coordinate gazetteer.

    Geometry is *not* the language model's job -- a dispatcher's address book already
    knows where the stops are.  Splitting it this way is what lets the translation
    benchmark measure translation and nothing else.
    """
    stops = spec["stops"]
    unknown = [s["name"] for s in stops if s["name"] not in gazetteer]
    if unknown:
        raise KeyError(f"not in the gazetteer: {unknown}")
    depot = gazetteer.get("depot")
    if depot is None:
        raise KeyError("gazetteer has no 'depot' entry")

    coords = [tuple(depot)] + [tuple(gazetteer[s["name"]]) for s in stops]
    svc = int(spec.get("service_minutes", 6))
    start = int(spec.get("start_time", 480))
    horizon = max(int(s["latest"]) for s in stops) + 600 if stops else start + 600
    return Instance(
        name=name, n_stops=len(stops), coords=tuple(coords),
        travel=travel_matrix(coords),
        demand=tuple([0] + [int(s["demand"]) for s in stops]),
        earliest=tuple([0] + [int(s["earliest"]) for s in stops]),
        latest=tuple([horizon] + [int(s["latest"]) for s in stops]),
        service=tuple([0] + [svc] * len(stops)),
        capacity=int(spec["capacity"]), reload_time=int(spec.get("reload_minutes", 10)),
        start_time=start, stop_names=tuple(s["name"] for s in stops),
    )


def load_gazetteer(path: str | pathlib.Path | None = None) -> dict[str, list[int]]:
    path = pathlib.Path(path) if path else (
        pathlib.Path(__file__).resolve().parents[1] / "data" / "gazetteer.json")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing -- it holds the stop-name -> coordinate address book "
            "the translation layer resolves against.")
    with open(path) as fh:
        return json.load(fh)
