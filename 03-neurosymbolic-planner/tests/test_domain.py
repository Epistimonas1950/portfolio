"""The problem definition itself: metric geometry, the cost lemma, and the validator.

The optimality test in `test_search.py` compares two searches over the *same*
`domain.successors`, so it validates search, not modelling.  These tests close that gap
from the other side: they check the properties the heuristic proofs assume about the
domain, and they check the validator against plans built to break one constraint each.
"""

import unittest

from src.domain import (DEPOT, Instance, Move, make_instance, plan_cost,
                        successors, travel_matrix, validate_plan)


def hand_built() -> Instance:
    """A three-stop instance whose optimum is computed on paper in the test below."""
    coords = ((0, 0), (0, 4), (6, 0), (6, 4))      # depot, A, B, C
    return Instance(
        name="hand", n_stops=3, coords=coords, travel=travel_matrix(coords),
        demand=(0, 1, 1, 1), earliest=(0, 0, 0, 0), latest=(999, 999, 10, 999),
        service=(0, 0, 0, 0), capacity=2, reload_time=0, start_time=0,
        stop_names=("A", "B", "C"))


class TestGeometry(unittest.TestCase):

    def test_travel_matrix_is_a_metric(self):
        # Every admissibility proof in heuristics.py uses d(i,j) <= d(i,k) + d(k,j).
        # ceil() preserves it and round() does not, so this is a real assertion about
        # a choice made in domain.travel_matrix, not a smoke test.
        for n in (4, 7, 11):
            for seed in range(4):
                inst = make_instance(n, seed)
                d = inst.travel
                m = inst.n_nodes
                for i in range(m):
                    self.assertEqual(d[i][i], 0)
                    for j in range(m):
                        self.assertEqual(d[i][j], d[j][i], "travel must be symmetric")
                        for k in range(m):
                            self.assertLessEqual(
                                d[i][j], d[i][k] + d[k][j],
                                f"triangle inequality fails at {(i, j, k)} in {inst.name}")

    def test_round_would_have_broken_the_triangle_inequality(self):
        # The counterexample the ceil() comment in domain.travel_matrix refers to.
        coords = [(0, 0), (1, 1), (2, 2)]
        import math
        rounded = [[round(math.dist(a, b)) for b in coords] for a in coords]
        self.assertGreater(rounded[0][2], rounded[0][1] + rounded[1][2])
        t = travel_matrix(coords)
        self.assertLessEqual(t[0][2], t[0][1] + t[1][2])


class TestTransitions(unittest.TestCase):

    def test_every_move_costs_at_least_the_drive(self):
        # Lemma (L) of heuristics.py: c(s, s') >= d(loc, loc').  Every admissibility
        # and consistency proof in this repo is downstream of it.
        inst = make_instance(6, 2)
        seen = 0
        frontier = [inst.initial_state()]
        visited = {frontier[0].key}
        while frontier and seen < 4000:
            s = frontier.pop()
            for mv, nxt, cost in successors(inst, s):
                seen += 1
                self.assertGreaterEqual(cost, inst.travel[s.loc][nxt.loc])
                self.assertGreaterEqual(cost, 0)
                if nxt.key not in visited:
                    visited.add(nxt.key)
                    frontier.append(nxt)
        self.assertGreater(seen, 100)

    def test_no_reload_at_full_capacity_or_at_the_depot(self):
        inst = make_instance(5, 0)
        s = inst.initial_state()                       # at the depot, fully loaded
        self.assertFalse(any(m.kind == "reload" for m, _, _ in successors(inst, s)))

    def test_illegal_move_is_rejected_loudly(self):
        inst = make_instance(4, 0)
        with self.assertRaises(ValueError):
            from src.domain import apply_move
            apply_move(inst, inst.initial_state(), Move("deliver", 99))


class TestValidator(unittest.TestCase):

    def test_generator_reference_plan_is_feasible(self):
        # make_instance builds every time window around a reference route precisely so
        # that a feasible plan is guaranteed to exist; if that broke, the benchmarks
        # would be measuring infeasible instances.
        for n in (3, 6, 9):
            for seed in range(5):
                inst = make_instance(n, seed)
                self.assertIsNotNone(inst.reference_cost)
                self.assertGreater(inst.reference_cost, 0)

    def test_validator_rejects_a_capacity_violation(self):
        inst = hand_built()                            # capacity 2, three parcels of 1
        plan = [Move("deliver", 2), Move("deliver", 3), Move("deliver", 1),
                Move("return", DEPOT)]
        v = validate_plan(inst, plan)
        self.assertFalse(v.feasible)
        self.assertTrue(any("capacity" in x for x in v.violations), v.violations)

    def test_validator_rejects_a_time_window_violation(self):
        inst = hand_built()                            # stop B (=2) closes at t=10
        plan = [Move("deliver", 1), Move("deliver", 3), Move("reload", DEPOT),
                Move("deliver", 2), Move("return", DEPOT)]
        v = validate_plan(inst, plan)
        self.assertFalse(v.feasible)
        self.assertTrue(any("time window" in x for x in v.violations), v.violations)

    def test_validator_rejects_an_incomplete_plan(self):
        inst = hand_built()
        v = validate_plan(inst, [Move("deliver", 2)])
        self.assertFalse(v.feasible)
        self.assertTrue(any("undelivered" in x for x in v.violations), v.violations)
        self.assertTrue(any("not back at the depot" in x for x in v.violations))

    def test_validator_accepts_the_hand_optimum_and_prices_it(self):
        inst = hand_built()
        plan = [Move("deliver", 2), Move("deliver", 3), Move("reload", DEPOT),
                Move("deliver", 1), Move("return", DEPOT)]
        v = validate_plan(inst, plan)
        self.assertTrue(v.feasible, v.violations)
        # 0->B 6, B->C 4, C->0 8, 0->A 4, A->0 4  =  26, by hand.
        self.assertEqual(v.cost, 26)
        self.assertEqual(plan_cost(inst, plan), 26)

    def test_cost_of_infeasible_plan_raises(self):
        with self.assertRaises(ValueError):
            plan_cost(hand_built(), [Move("deliver", 1)])


if __name__ == "__main__":
    unittest.main()
