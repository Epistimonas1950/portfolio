"""The adversarial variant: a competing courier claiming stops, and alpha-beta on it.

THE GAME.  A rival courier works the same map from its own base.  The two alternate;
whoever is to move drives to one unclaimed stop and takes the job, banking the fee minus
the drive:

    gain(side, p) = fee_p - d(loc_side, p)

The game is zero-sum in the *difference* of the two couriers' earnings.  MAX maximises
(my earnings - rival's earnings); the position is

    (claimed, loc_max, loc_min, side to move)

and the branching factor is the number of unclaimed stops, so it falls by one every
ply -- this is a shrinking tree, not a uniform one, which matters for what the
branching-factor measurement can honestly claim (see below).

WHAT THE SEARCH RETURNS.  `_search` returns the *incremental* value from a position
onward, never the accumulated score.  That is not a style choice: it is what makes the
transposition table sound.  A table entry keyed on the position must be valid however
the search arrived there, and an accumulated score is path-dependent.  Depth-limited
leaves are evaluated with a position-only term as well:

    eval = sum over unclaimed p of ( d(loc_min, p) - d(loc_max, p) ) // 2

"how much closer am I than the rival to what is left" -- a positional advantage, halved
so it informs the ordering without swamping the banked fees.  Note that a game
evaluation function needs no admissibility: nothing here is claimed to be optimal
against a real opponent, only that the *same* tree is searched by every variant.

ALPHA-BETA.  Prune a branch once it cannot affect the root: maintain [alpha, beta], the
best MAX can already force and the best MIN can already force, and cut when they cross.
The value at the root is *exactly* the minimax value -- alpha-beta never changes the
answer, only the work -- which is the equality the test suite asserts per instance.  It
is the cheapest possible check that the pruning conditions are right.

KNUTH & MOORE (1975): with perfect move ordering alpha-beta visits O(b^(d/2)) leaves
instead of O(b^d), so it searches twice as deep for the same budget.  The square root is
the entire point of the algorithm, and it is why move ordering -- here, "try the stop
with the best immediate gain for the side to move first" -- is not a micro-optimisation
but the thing that decides whether you get b or sqrt(b).  This domain's branching
shrinks with depth, so the measured effective branching factor cannot land on sqrt(b)
of anything; what `bench/adversarial.py` reports is the *direction* and size of the
movement, which is the honest claim.

TRANSPOSITION TABLE.  Positions repeat: any two orders of the same claimed set with the
same two locations are the same position.  Entries store (depth, flag, value, best move)
with flag EXACT / LOWER / UPPER; a stored bound is a true bound on the position's
depth-d value regardless of the window it was proved under, so reusing it is sound.
The stored best move seeds the ordering of the next, deeper iteration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..domain import Instance, travel_matrix
from .instrument import SearchStats

EXACT, LOWER, UPPER = 0, 1, 2


@dataclass
class Contest:
    """An Instance plus a rival courier: fees, a rival base, and a travel matrix."""

    inst: Instance
    fee: tuple[int, ...]                     # indexed by node; fee[0] unused
    travel: tuple[tuple[int, ...], ...]      # over inst nodes + the rival base
    rival: int                               # index of the rival base
    name: str = "contest"

    @property
    def n_stops(self) -> int:
        return self.inst.n_stops

    @property
    def full_mask(self) -> int:
        return ((1 << self.n_stops) - 1) << 1


def make_contest(inst: Instance, seed: int = 0, fee_lo: int = 25,
                 fee_hi: int = 60) -> Contest:
    """Attach a rival to an instance.  Seeded; the rival base is placed opposite the depot.

    The rival's base is the depot reflected through the centre of the service area, so
    the two couriers start with mirror-image positional advantages and the game is not
    decided by the map before a move is made.
    """
    rng = np.random.default_rng(seed)
    xs = [c[0] for c in inst.coords]
    ys = [c[1] for c in inst.coords]
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    depot = inst.coords[0]
    rival_coord = (int(round(2 * cx - depot[0])), int(round(2 * cy - depot[1])))
    coords = list(inst.coords) + [rival_coord]
    fee = tuple([0] + [int(rng.integers(fee_lo, fee_hi + 1))
                       for _ in range(inst.n_stops)])
    return Contest(inst=inst, fee=fee, travel=travel_matrix(coords),
                   rival=inst.n_stops + 1, name=f"{inst.name}-contest")


@dataclass
class GameResult:
    value: int
    best_move: int | None
    stats: SearchStats = field(default_factory=SearchStats)
    principal_variation: list[int] = field(default_factory=list)


def _static_eval(c: Contest, claimed: int, lm: int, ln: int) -> int:
    """Positional term: how much closer MAX is than MIN to what is still unclaimed."""
    total = 0
    mask, p = c.full_mask & ~claimed, 1
    m = mask >> 1
    while m:
        if m & 1:
            total += c.travel[ln][p] - c.travel[lm][p]
        m >>= 1
        p += 1
    return total // 2


def _moves(c: Contest, claimed: int, loc: int, ordered: bool) -> list[int]:
    stops = [p for p in range(1, c.n_stops + 1) if not (claimed >> p) & 1]
    if ordered:
        # Best immediate gain for the side to move, first.  Symmetric: MIN maximising
        # its own gain is exactly MIN minimising the differential's increment.
        stops.sort(key=lambda p: -(c.fee[p] - c.travel[loc][p]))
    return stops


def minimax(c: Contest, depth: int) -> GameResult:
    """Plain minimax, no pruning.  The reference value every other variant must match."""
    stats = SearchStats(algorithm="minimax", heuristic=f"depth={depth}")
    t0 = time.perf_counter()

    def rec(claimed: int, lm: int, ln: int, maxing: bool, d: int) -> int:
        stats.expansions += 1
        stops = _moves(c, claimed, lm if maxing else ln, ordered=False)
        if not stops or d == 0:
            return _static_eval(c, claimed, lm, ln) if stops else 0
        if maxing:
            best = -(1 << 60)
            for p in stops:
                stats.generated += 1
                v = (c.fee[p] - c.travel[lm][p]) + rec(claimed | 1 << p, p, ln, False, d - 1)
                if v > best:
                    best = v
            return best
        best = 1 << 60
        for p in stops:
            stats.generated += 1
            v = -(c.fee[p] - c.travel[ln][p]) + rec(claimed | 1 << p, lm, p, True, d - 1)
            if v < best:
                best = v
        return best

    root_moves = _moves(c, 0, 0, ordered=False)
    best_v, best_m = -(1 << 60), None
    for p in root_moves:
        stats.generated += 1
        v = (c.fee[p] - c.travel[0][p]) + rec(1 << p, p, c.rival, False, depth - 1)
        if v > best_v:
            best_v, best_m = v, p
    stats.expansions += 1
    stats.seconds = time.perf_counter() - t0
    stats.depth = depth
    stats.solved = True
    return GameResult(best_v, best_m, stats)


def alphabeta(c: Contest, depth: int, ordering: bool = True, use_tt: bool = True,
              tt: dict | None = None, root_first: int | None = None) -> GameResult:
    """Alpha-beta.  Returns the same value as `minimax(c, depth)`, always."""
    stats = SearchStats(
        algorithm="alpha-beta"
        + ("+ord" if ordering else "") + ("+tt" if use_tt else ""),
        heuristic=f"depth={depth}")
    t0 = time.perf_counter()
    table: dict = tt if tt is not None else {}

    def rec(claimed: int, lm: int, ln: int, maxing: bool, d: int,
            alpha: int, beta: int) -> int:
        stats.expansions += 1
        stops = _moves(c, claimed, lm if maxing else ln, ordered=ordering)
        if not stops:
            return 0
        if d == 0:
            return _static_eval(c, claimed, lm, ln)

        key = (claimed, lm, ln, maxing)
        hint = None
        if use_tt:
            entry = table.get(key)
            if entry is not None:
                e_depth, e_flag, e_val, e_move = entry
                if e_depth >= d:
                    if e_flag == EXACT:
                        return e_val
                    if e_flag == LOWER and e_val > alpha:
                        alpha = e_val
                    elif e_flag == UPPER and e_val < beta:
                        beta = e_val
                    if alpha >= beta:
                        return e_val
                hint = e_move
        if hint is not None and hint in stops:
            stops.remove(hint)
            stops.insert(0, hint)

        a0, b0 = alpha, beta
        best_move = stops[0]
        if maxing:
            best = -(1 << 60)
            for p in stops:
                stats.generated += 1
                # The window lives in this node's value scale; the child's does not.
                # v = edge + child, so the child must be searched in (alpha - edge,
                # beta - edge).  Forgetting this shift is the classic way to make
                # alpha-beta return something that is not the minimax value.
                edge = c.fee[p] - c.travel[lm][p]
                v = edge + rec(claimed | 1 << p, p, ln, False, d - 1,
                               alpha - edge, beta - edge)
                if v > best:
                    best, best_move = v, p
                if best > alpha:
                    alpha = best
                if alpha >= beta:
                    break                       # beta cutoff: MIN would avoid this line
        else:
            best = 1 << 60
            for p in stops:
                stats.generated += 1
                edge = -(c.fee[p] - c.travel[ln][p])
                v = edge + rec(claimed | 1 << p, lm, p, True, d - 1,
                               alpha - edge, beta - edge)
                if v < best:
                    best, best_move = v, p
                if best < beta:
                    beta = best
                if alpha >= beta:
                    break                       # alpha cutoff
        if use_tt:
            flag = EXACT if a0 < best < b0 else (LOWER if best >= b0 else UPPER)
            prev = table.get(key)
            if prev is None or prev[0] <= d:
                table[key] = (d, flag, best, best_move)
        return best

    alpha, beta = -(1 << 60), 1 << 60
    root_stops = _moves(c, 0, 0, ordered=ordering)
    if root_first is not None and root_first in root_stops:
        root_stops.remove(root_first)
        root_stops.insert(0, root_first)
    best_v, best_m = -(1 << 60), None
    for p in root_stops:
        stats.generated += 1
        edge = c.fee[p] - c.travel[0][p]
        v = edge + rec(1 << p, p, c.rival, False, depth - 1, alpha - edge, beta - edge)
        if v > best_v:
            best_v, best_m = v, p
        if best_v > alpha:
            alpha = best_v
    stats.expansions += 1
    stats.seconds = time.perf_counter() - t0
    stats.depth = depth
    stats.solved = True
    return GameResult(best_v, best_m, stats)


def iterative_deepening(c: Contest, max_depth: int, ordering: bool = True,
                        use_tt: bool = True) -> GameResult:
    """Search depths 1..max_depth, feeding each round's best root move to the next.

    Re-searching from scratch sounds wasteful and is not: in a tree of branching factor
    b the last iteration dominates the total, and the ordering information the shallow
    rounds hand over is worth far more than they cost.  It also makes the search
    interruptible -- there is always a legal best move in hand, which is what an
    anytime dispatcher needs.
    """
    tt: dict = {} if use_tt else None
    best = GameResult(0, None)
    total = SearchStats(algorithm="alpha-beta+ID", heuristic=f"depth={max_depth}")
    t0 = time.perf_counter()
    for d in range(1, max_depth + 1):
        best = alphabeta(c, d, ordering=ordering, use_tt=use_tt, tt=tt,
                         root_first=best.best_move)
        total.expansions += best.stats.expansions
        total.generated += best.stats.generated
        total.iterations += 1
    total.seconds = time.perf_counter() - t0
    total.depth = max_depth
    total.solved = True
    return GameResult(best.value, best.best_move, total)
