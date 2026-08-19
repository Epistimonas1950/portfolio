"""Rank allocation under a budget, and the parameter accounting it depends on."""

import unittest

import numpy as np

from src.allocate import (LayerSpec, allocate_greedy, allocate_knapsack_dp,
                          allocate_lagrangian, allocate_uniform, total_dense)
from src.factorize import activation_error, whitened_spectrum, whitened_svd
from src.rebuild import (break_even_rank, compression_ratio, dense_params,
                         factored_params, rebuild_layer, report_layer,
                         stack_compression)
from src.synth import make_stack

#: Small enough that the whole suite stays under a couple of seconds, heterogeneous
#: enough that uniform allocation is genuinely the wrong answer: shapes differ, so a
#: unit of rank costs a different number of parameters per layer, and the activation
#: conditioning spans four decades, so it buys a different amount of loss reduction.
TEST_SHAPES = ((64, 128, 1e6), (96, 96, 1e4), (128, 64, 1e2), (96, 128, 1e5))
N_SAMPLES = 384
RIDGE = 1e-8          # effectively zero: the allocation objective is then exactly
                      # the measured squared activation error, which one test checks


def build_specs(seed: int = 0):
    stack = make_stack(n_samples=N_SAMPLES, seed=seed, shapes=TEST_SHAPES)
    specs, whitenings = [], []
    for layer in stack:
        sigma, wh = whitened_spectrum(layer.w, layer.x, ridge=RIDGE)
        specs.append(LayerSpec(name=layer.name, m=layer.w.shape[0],
                               n=layer.w.shape[1], sigma=sigma))
        whitenings.append(wh)
    return stack, specs, whitenings


class TestAllocationStrategies(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.stack, cls.specs, cls.whitenings = build_specs()
        cls.dense = total_dense(cls.specs)
        cls.budgets = [int(f * cls.dense) for f in (0.10, 0.20, 0.35, 0.50)]

    def test_every_strategy_respects_the_budget(self):
        for budget in self.budgets:
            for solve in (allocate_uniform, allocate_greedy, allocate_lagrangian,
                          allocate_knapsack_dp):
                alloc = solve(self.specs, budget)
                self.assertLessEqual(alloc.params, budget, alloc.strategy)
                self.assertEqual(len(alloc.ranks), len(self.specs))
                self.assertTrue(all(r >= 1 for r in alloc.ranks))

    # === THE TEST THAT MATTERS (second anchor) ===
    # Fails if the allocation mathematics is wrong, not merely if the code crashed.
    #
    # Uniform compression is what everybody ships and it is measurably the wrong
    # answer: at the same parameter budget, the marginal-gain and Lagrangian solvers
    # must beat it by a clear margin, and both must land within a fraction of a
    # percent of the knapsack optimum. The second half is the more interesting claim.
    # Because the per-layer losses sum_{i>r} sigma_i^2 have non-increasing marginal
    # gains, this is a separable convex allocation problem, and for that class
    # incremental greedy is optimal up to the last partial item -- so "greedy is
    # essentially exactly optimal" is a prediction of the theory, not an accident.
    def test_greedy_and_lagrangian_beat_uniform_and_nearly_match_the_optimum(self):
        for budget in self.budgets:
            uniform = allocate_uniform(self.specs, budget)
            greedy = allocate_greedy(self.specs, budget)
            lagrange = allocate_lagrangian(self.specs, budget)
            optimum = allocate_knapsack_dp(self.specs, budget)

            self.assertLessEqual(optimum.loss, min(uniform.loss, greedy.loss,
                                                   lagrange.loss) + 1e-12,
                                 f"budget {budget}: the DP is not the optimum")
            for alloc in (greedy, lagrange):
                self.assertLess(alloc.loss, 0.9 * uniform.loss,
                                f"budget {budget}: {alloc.strategy} only matched "
                                f"uniform ({alloc.loss:.6g} vs {uniform.loss:.6g})")
                gap = (alloc.loss - optimum.loss) / optimum.loss
                self.assertLess(gap, 0.05,
                                f"budget {budget}: {alloc.strategy} is {100*gap:.2f}% "
                                "off the optimum, which the convexity argument says "
                                "should not happen")

    def test_the_allocation_objective_is_the_real_error(self):
        # sum_l L_l(r_l) must equal sum_l ||(W_l - W_hat_l) X_l||_F^2 at zero ridge.
        # Without this the allocators are optimizing a number with no connection to
        # what the compressed stack actually does.
        budget = self.budgets[2]
        alloc = allocate_greedy(self.specs, budget)
        measured = 0.0
        for layer, wh, r in zip(self.stack, self.whitenings, alloc.ranks):
            fac = whitened_svd(layer.w, layer.x, r, whitening=wh)
            measured += activation_error(layer.w, fac.w_hat, layer.x) ** 2
        self.assertAlmostEqual(alloc.loss / measured, 1.0, places=5)

    def test_lagrangian_equalizes_the_marginal_loss_per_parameter(self):
        # The whole content of the relaxation: at the solution there is a single
        # threshold mu such that every layer has bought exactly the components whose
        # squared singular value per parameter exceeds it. So the cheapest component
        # anybody kept must be worth at least as much as the dearest one anybody
        # declined -- otherwise a swap would improve the objective.
        for budget in self.budgets:
            alloc = allocate_lagrangian(self.specs, budget)
            kept, declined = [], []
            for spec, r in zip(self.specs, alloc.ranks):
                cap = min(spec.max_rank, spec.break_even)
                if r > 1:
                    kept.append(spec.gain(r) / spec.cost_per_rank)
                if r < cap:
                    declined.append(spec.gain(r + 1) / spec.cost_per_rank)
            self.assertGreaterEqual(min(kept), max(declined) - 1e-12,
                                    f"budget {budget}: marginal rates not equalized")

    def test_more_budget_never_costs_more_loss(self):
        for solve in (allocate_greedy, allocate_lagrangian, allocate_knapsack_dp):
            losses = [solve(self.specs, b).loss for b in self.budgets]
            self.assertTrue(all(b <= a + 1e-12 for a, b in zip(losses, losses[1:])),
                            f"{solve.__name__} loss is not monotone in the budget: "
                            f"{losses}")

    def test_an_unaffordable_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            allocate_greedy(self.specs, 10)

    def test_layer_spec_rejects_an_unsorted_spectrum(self):
        with self.assertRaises(ValueError):
            LayerSpec(name="bad", m=8, n=8, sigma=np.array([1.0, 5.0, 2.0]))

    def test_break_even_cap_keeps_every_layer_smaller_than_dense(self):
        alloc = allocate_greedy(self.specs, int(0.99 * self.dense))
        for spec, r in zip(self.specs, alloc.ranks):
            self.assertLess(factored_params(spec.m, spec.n, r), spec.dense,
                            f"{spec.name} at rank {r} is larger than the dense layer")


class TestParameterAccounting(unittest.TestCase):

    def test_break_even_rank_is_the_last_rank_that_actually_compresses(self):
        for m, n in ((256, 256), (128, 512), (64, 4096), (3, 7)):
            r = break_even_rank(m, n)
            self.assertLess(factored_params(m, n, r), dense_params(m, n))
            self.assertGreaterEqual(factored_params(m, n, r + 1), dense_params(m, n))

    def test_square_layer_breaks_even_just_below_half_width(self):
        self.assertEqual(break_even_rank(256, 256), 127)
        self.assertAlmostEqual(compression_ratio(256, 256, 32), 4.0)
        self.assertAlmostEqual(compression_ratio(256, 256, 64), 2.0)
        self.assertAlmostEqual(compression_ratio(256, 256, 128), 1.0)

    def test_stack_compression_is_the_ratio_of_sums(self):
        reports = [report_layer("a", 100, 100, 10), report_layer("b", 10, 10, 2)]
        expected = (100 * 100 + 10 * 10) / (10 * 200 + 2 * 20)
        self.assertAlmostEqual(stack_compression(reports), expected)
        # The mean of per-layer ratios would be the flattering number; check we are
        # not accidentally computing it.
        mean_of_ratios = np.mean([r.ratio for r in reports])
        self.assertNotAlmostEqual(stack_compression(reports), float(mean_of_ratios))

    def test_rebuild_rejects_a_transposed_factor(self):
        with self.assertRaises(ValueError):
            rebuild_layer(np.zeros((8, 3)), np.zeros((4, 12)))
        self.assertEqual(rebuild_layer(np.zeros((8, 3)), np.zeros((3, 12))).shape,
                         (8, 12))


if __name__ == "__main__":
    unittest.main()
