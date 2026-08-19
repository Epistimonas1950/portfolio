"""Alpha-beta must not change the answer, only the work.

Two claims, and they pull in opposite directions, which is what makes them worth
asserting together: pruning must be aggressive enough to matter and conservative enough
to be invisible in the value.
"""

import unittest

from src.domain import make_instance
from src.search.alphabeta import (alphabeta, iterative_deepening, make_contest,
                                  minimax)

CASES = [(8, seed) for seed in range(4)]
DEPTHS = (4, 6)


class TestAlphaBeta(unittest.TestCase):

    # === THE TEST THAT MATTERS (adversarial half) ===
    # Alpha-beta returns EXACTLY the minimax value, for every variant -- plain, with
    # move ordering, with a transposition table, and under iterative deepening -- while
    # expanding strictly fewer nodes.  Pruning that changes the value is not pruning,
    # it is a bug; and a cutoff condition that is too weak is invisible except in the
    # node counts.  This caught a real one: the alpha-beta window has to be shifted by
    # the edge value at every recursion, because the search returns the incremental
    # value of a position, not the accumulated score.  Without the shift the pruning
    # was sound-looking and the root value was wrong by tens of units.
    def test_alphabeta_equals_minimax_and_prunes(self):
        strictly_fewer = 0
        for n, seed in CASES:
            contest = make_contest(make_instance(n, seed), seed=seed)
            for depth in DEPTHS:
                ref = minimax(contest, depth)
                variants = {
                    "plain": alphabeta(contest, depth, ordering=False, use_tt=False),
                    "ordered": alphabeta(contest, depth, ordering=True, use_tt=False),
                    "ordered+tt": alphabeta(contest, depth, ordering=True, use_tt=True),
                    "iterative": iterative_deepening(contest, depth),
                }
                for label, res in variants.items():
                    self.assertEqual(
                        res.value, ref.value,
                        f"{contest.name} d={depth}: {label} returned {res.value}, "
                        f"minimax says {ref.value}")
                for label in ("plain", "ordered", "ordered+tt"):
                    self.assertLessEqual(variants[label].stats.expansions,
                                         ref.stats.expansions)
                strictly_fewer += variants["plain"].stats.expansions < ref.stats.expansions
        self.assertEqual(strictly_fewer, len(CASES) * len(DEPTHS),
                         "alpha-beta failed to prune anything somewhere")

    def test_move_ordering_and_the_table_each_pay_for_themselves(self):
        plain = ordered = tabled = 0
        for n, seed in CASES:
            contest = make_contest(make_instance(n, seed), seed=seed)
            depth = 6
            plain += alphabeta(contest, depth, ordering=False, use_tt=False).stats.expansions
            ordered += alphabeta(contest, depth, ordering=True, use_tt=False).stats.expansions
            tabled += alphabeta(contest, depth, ordering=True, use_tt=True).stats.expansions
        self.assertLess(ordered, plain, "move ordering bought nothing")
        self.assertLess(tabled, ordered, "the transposition table bought nothing")

    def test_effective_branching_factor_moves_toward_the_square_root(self):
        # Knuth & Moore: perfect ordering turns O(b^d) into O(b^(d/2)).  This tree's
        # branching shrinks by one every ply, so b* cannot literally land on sqrt(b);
        # the assertion is on the DIRECTION and on ordering beating no ordering, which
        # is the claim the mathematics actually licenses here.
        for n, seed in [(9, 1), (9, 2)]:
            contest = make_contest(make_instance(n, seed), seed=seed)
            depth = 6
            b_minimax = minimax(contest, depth).stats.ebf
            b_plain = alphabeta(contest, depth, ordering=False, use_tt=False).stats.ebf
            b_ord = alphabeta(contest, depth, ordering=True, use_tt=False).stats.ebf
            self.assertLess(b_plain, b_minimax)
            self.assertLess(b_ord, b_plain)
            self.assertLess(b_ord, n)          # nominal branching at the root

    def test_iterative_deepening_agrees_with_a_single_deep_search(self):
        for n, seed in [(7, 0), (8, 3)]:
            contest = make_contest(make_instance(n, seed), seed=seed)
            deep = alphabeta(contest, 5, ordering=True, use_tt=True)
            ident = iterative_deepening(contest, 5)
            self.assertEqual(ident.value, deep.value)

    def test_deeper_search_is_not_free(self):
        contest = make_contest(make_instance(9, 4), seed=4)
        counts = [alphabeta(contest, d, ordering=True, use_tt=True).stats.expansions
                  for d in (2, 4, 6)]
        self.assertLess(counts[0], counts[1])
        self.assertLess(counts[1], counts[2])


if __name__ == "__main__":
    unittest.main()
