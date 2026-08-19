"""The claims this repo is built on, stated as assertions.

Optimality is the product.  If A* ever returns anything other than the exact optimum,
the architecture in BRIEF.md -- "the language model proposes, exact search guarantees"
-- has nothing to guarantee, and everything else here is decoration.
"""

import unittest

from src.domain import DEPOT, State, make_instance, plan_cost, successors
from src.search import heuristics as H
from src.search.astar import astar
from src.search.bruteforce import brute_force
from src.search.idastar import idastar

SMALL = [(n, seed) for n in (3, 4, 5, 6, 7) for seed in range(12)]
NAMED = ("h0", "h1", "h2", "hx")


def reachable_states(inst, limit=1500):
    """A sample of reachable states, for pointwise comparisons of heuristic values."""
    out, seen = [], set()
    stack = [inst.initial_state()]
    while stack and len(out) < limit:
        s = stack.pop()
        if s.key in seen:
            continue
        seen.add(s.key)
        out.append(s)
        for _mv, nxt, _c in successors(inst, s):
            stack.append(nxt)
    return out


class TestOptimality(unittest.TestCase):

    # === THE TEST THAT MATTERS ===
    # Fails if the mathematics is wrong, not merely if the code crashed.
    #
    # Over 60 seeded instances spanning 3 to 7 stops, A*'s returned cost equals the
    # exhaustive-enumeration optimum EXACTLY, for every admissible heuristic -- the
    # zero heuristic, both relaxed bounds, and the deliberately inconsistent one.
    # Costs are integer minutes by construction (domain.travel_matrix takes ceilings),
    # so this is assertEqual with no tolerance to hide in.
    #
    # It is a real test of the mathematics because A* and the brute force reach the
    # answer by completely different routes.  The brute force enumerates every feasible
    # plan and takes the minimum.  A* prunes on two proved arguments -- admissibility of
    # h, and the earliest-arrival dominance that lets the closed set key on
    # (loc, U, cap) and throw away later arrivals at the same key.  Either argument
    # being wrong, in the proof or in the code, shows up here as a cost that is too
    # high.  The returned plan is also re-validated from scratch, so a plan that merely
    # scores well but breaks a time window cannot pass.
    def test_astar_cost_equals_brute_force_optimum_for_every_heuristic(self):
        checked = 0
        for n, seed in SMALL:
            inst = make_instance(n, seed)
            truth = brute_force(inst)
            self.assertIsNotNone(truth.cost, f"{inst.name} has no feasible plan")
            self.assertLessEqual(truth.cost, inst.reference_cost)
            for name in NAMED:
                res = astar(inst, H.make(name))
                self.assertEqual(res.cost, truth.cost,
                                 f"{inst.name} with {name}: A* returned {res.cost}, "
                                 f"exhaustive optimum is {truth.cost}")
                # and the plan must actually be a plan
                self.assertEqual(plan_cost(inst, res.plan), truth.cost,
                                 f"{inst.name} with {name}: returned plan does not "
                                 "re-validate at the returned cost")
                checked += 1
        self.assertEqual(checked, len(SMALL) * len(NAMED))

    # === THE TEST THAT MATTERS (second half) ===
    # Heuristic dominance, empirically.  h2 >= h1 >= h0 pointwise and all three
    # admissible, so Pearl's theorem says A* with the stronger heuristic expands no
    # node with f < C* that the weaker one does not.
    #
    # The fine print is respected rather than ignored: the theorem says nothing about
    # nodes sitting exactly on f == C*, which are expanded or not according to
    # tie-breaking, and with integer minutes those ties are everywhere.  So the strict
    # per-instance assertion is on `expansions_below_cstar` -- the set the theorem
    # actually covers -- against the weaker heuristic's total, and the totals are
    # asserted in aggregate.
    def test_heuristic_dominance(self):
        totals = {name: 0 for name in ("h0", "h1", "h2")}
        for n, seed in SMALL:
            inst = make_instance(n, seed)
            stats = {}
            for name in totals:
                res = astar(inst, H.make(name))
                stats[name] = res.stats
                totals[name] += res.stats.expansions
            for weak, strong in (("h0", "h1"), ("h1", "h2"), ("h0", "h2")):
                self.assertLessEqual(
                    stats[strong].expansions_below_cstar, stats[weak].expansions,
                    f"{inst.name}: {strong} expanded a surely-expanded node set larger "
                    f"than {weak}'s total -- dominance is violated")
        self.assertLessEqual(totals["h2"], totals["h1"])
        self.assertLessEqual(totals["h1"], totals["h0"])
        # and it must be a real effect, not a tie
        self.assertLess(totals["h2"] * 1.5, totals["h1"])
        self.assertLess(totals["h1"] * 1.4, totals["h0"])

    def test_heuristics_are_pointwise_ordered_and_never_overestimate(self):
        # The hypothesis of the dominance theorem, checked directly: h2 >= h1 >= h0 at
        # every reachable state, and no heuristic exceeds the true cost-to-go at the
        # start state (where the true cost-to-go is the brute-force optimum).
        h0, h1, h2 = H.make("h0"), H.make("h1"), H.make("h2")
        for n, seed in [(4, 0), (5, 1), (6, 2), (7, 3)]:
            inst = make_instance(n, seed)
            for s in reachable_states(inst):
                a, b, c = h0(inst, s), h1(inst, s), h2(inst, s)
                self.assertLessEqual(a, b, f"h1 < h0 at {s}")
                self.assertLessEqual(b, c, f"h2 < h1 at {s}")
            optimum = brute_force(inst).cost
            start = inst.initial_state()
            for h in (h0, h1, h2, H.make("hx")):
                self.assertLessEqual(h(inst, start), optimum,
                                     f"{h.name} overestimates on {inst.name}")

    # The hypothesis of every result in this file, checked where it is actually needed:
    # at EVERY reachable state, not just the root.  h*(s) is obtained by exhaustive
    # enumeration restarted from s, so this compares each heuristic against the true
    # cost-to-go of the state it is looking at.  It matters here because h2 is a sum of
    # three separately-argued terms and the reload term ceil((D - cap)/capacity) is
    # evaluated at partially-delivered states with a partial load -- exactly the states
    # a root-only check never visits, and exactly where an over-count would hide.
    def test_no_heuristic_overestimates_at_any_reachable_state(self):
        checked = dead_ends = 0
        cases = [(3, 0), (4, 2), (5, 5), (6, 1), (5, 0), (5, 3), (6, 4), (6, 7), (7, 2)]
        for n, seed in cases:
            inst = make_instance(n, seed)
            heuristics = [H.make(name) for name in NAMED]
            for s in reachable_states(inst, limit=400):
                truth = brute_force(inst, start=s).cost
                if truth is None:
                    dead_ends += 1          # no feasible completion: h* is infinite
                    continue
                for h in heuristics:
                    self.assertLessEqual(
                        h(inst, s), truth,
                        f"{inst.name}: {h.name} says {h(inst, s)} from {s}, but the "
                        f"true cost-to-go is {truth}")
                checked += 1
        self.assertGreater(checked, 600, "too few states exercised to mean anything")
        # Most reachable states are doomed -- some window has already been missed -- and
        # h* is infinite there, so admissibility is vacuous.  Counted rather than
        # ignored, so the size of the live sample is visible.
        self.assertGreater(dead_ends, 0)

    def test_memoized_heuristic_does_not_leak_between_instances(self):
        # h2 caches its spanning-tree bound on (loc, U, cap), which is meaningless
        # across instances.  Reusing one heuristic object on two instances must not
        # return the first instance's numbers -- this failed once, silently, and the
        # value it returned was smaller than h1's, i.e. not even a valid bound.
        h1, h2 = H.make("h1"), H.make("h2")
        a, b = make_instance(4, 0), make_instance(5, 1)
        for inst in (a, b, a, b):
            for s in reachable_states(inst, limit=200):
                self.assertLessEqual(h1(inst, s), h2(inst, s))

    def test_heuristic_value_at_the_goal_is_zero(self):
        # h(goal) = 0 is required for consistency to imply admissibility.
        inst = make_instance(5, 4)
        goal = State(DEPOT, 0, inst.start_time, inst.capacity)
        for name in NAMED:
            self.assertEqual(H.make(name)(inst, goal), 0)


class TestConsistencyVersusAdmissibility(unittest.TestCase):

    # A CONSISTENT heuristic makes f non-decreasing along every path, so the first time
    # A* pops a node its g is already optimal and no closed node is ever re-opened.
    # That is a theorem, so the assertion is exact and per instance: the counter is 0.
    def test_consistent_heuristics_never_re_expand(self):
        for n, seed in SMALL:
            inst = make_instance(n, seed)
            for name in ("h0", "h1", "h2"):
                res = astar(inst, H.make(name))
                self.assertEqual(res.stats.re_expansions, 0,
                                 f"{inst.name} with {name}: {res.stats.re_expansions} "
                                 "re-expansions, but the heuristic is consistent")

    # An ADMISSIBLE-BUT-INCONSISTENT heuristic keeps optimality and loses the closed-set
    # guarantee.  Inconsistency does not force a re-expansion on any particular
    # instance, so existence is asserted in aggregate over the whole seeded set -- and
    # the optimum must still come out exactly right, which is the point: admissibility
    # alone buys optimality, consistency buys efficiency.
    def test_inconsistent_heuristic_re_expands_yet_stays_optimal(self):
        total_re = 0
        for n, seed in SMALL:
            inst = make_instance(n, seed)
            res = astar(inst, H.make("hx"))
            total_re += res.stats.re_expansions
            self.assertEqual(res.cost, brute_force(inst).cost,
                             f"{inst.name}: inconsistent-but-admissible h lost optimality")
        self.assertGreater(total_re, 0,
                           "the deliberately inconsistent heuristic produced no "
                           "re-expansions at all -- it is not exercising the distinction")

    def test_dropping_reopening_is_what_would_break(self):
        # Why re-opening is on by default: with an inconsistent heuristic, an A* that
        # refuses to re-open a closed node can return a cost that is too high.  If this
        # ever stops finding such an instance the demonstration above is vacuous.
        bad = 0
        for n, seed in SMALL:
            inst = make_instance(n, seed)
            truth = brute_force(inst).cost
            res = astar(inst, H.make("hx"), reopen=False)
            self.assertGreaterEqual(res.cost, truth)      # never better than optimal
            bad += res.cost > truth
        self.assertGreater(bad, 0, "no instance exposed the no-reopen failure")


class TestIDAStar(unittest.TestCase):

    def test_idastar_returns_the_same_optimum_as_astar(self):
        for n, seed in [(3, 0), (4, 1), (5, 2), (6, 3), (6, 7)]:
            inst = make_instance(n, seed)
            a = astar(inst, H.make("h2"))
            d = idastar(inst, H.make("h2"))
            self.assertEqual(d.cost, a.cost, f"{inst.name}: IDA* != A*")
            self.assertEqual(plan_cost(inst, d.plan), a.cost)

    def test_idastar_pays_for_its_memory_with_repeated_work(self):
        # The honest finding, asserted so it cannot quietly stop being true: this state
        # graph is dense with transpositions, IDA* keeps no closed set, so it generates
        # far more nodes than A* while holding only the current path.
        inst = make_instance(7, 1)
        a = astar(inst, H.make("h2"))
        d = idastar(inst, H.make("h2"))
        self.assertGreater(d.stats.generated, a.stats.generated)
        self.assertGreater(d.stats.iterations, 1)


if __name__ == "__main__":
    unittest.main()
