"""The distributional metrics, checked against cases whose answers are known."""

import unittest

import numpy as np

import src.metrics as metrics
from src.problem import CANONICAL, SDE
from src.sde import GaussianMixture


class TestWasserstein(unittest.TestCase):
    def test_a_shift_costs_exactly_the_shift(self):
        # W1 between P and P + c is |c|, for any P.
        m = SDE.marginal(CANONICAL, SDE.t_min)
        n = 4096
        q = metrics.target_midpoint_quantiles(m, n)
        x = m.stratified(n)
        self.assertLess(metrics.wasserstein1(x, q), 1e-12)
        for c in (0.1, -0.35):
            self.assertAlmostEqual(metrics.wasserstein1(x + c, q), abs(c), places=10)

    def test_size_mismatch_raises(self):
        m = SDE.marginal(CANONICAL, SDE.t_min)
        with self.assertRaises(ValueError):
            metrics.wasserstein1(np.zeros(10), metrics.target_midpoint_quantiles(m, 11))


class TestEnergyDistance(unittest.TestCase):
    def test_it_is_zero_for_the_target_and_positive_otherwise(self):
        m = SDE.marginal(CANONICAL, SDE.t_min)
        self_term = metrics._mean_abs_target(m)
        x = m.stratified(4096)
        self.assertLess(abs(metrics.energy_distance(x, m, self_term)), 1e-6)
        for c in (0.05, 0.4):
            self.assertGreater(metrics.energy_distance(x + c, m, self_term), 0.0)
        self.assertGreater(metrics.energy_distance(x + 0.4, m, self_term),
                           metrics.energy_distance(x + 0.05, m, self_term))

    def test_closed_form_self_term_matches_a_pairwise_sum(self):
        # E|Y - Y'| by the K^2 folded-normal formula, against brute force on samples.
        m = GaussianMixture(np.array([0.6, 0.4]), np.array([[-1.0], [1.2]]),
                            np.array([0.4, 0.6]))
        rng = np.random.default_rng(0)
        a = m.sample(rng, 4000).ravel()
        b = m.sample(rng, 4000).ravel()
        brute = float(np.abs(a[:, None] - b[None, :]).mean())
        self.assertAlmostEqual(metrics._mean_abs_target(m), brute, delta=0.02)

    def test_closed_form_cross_term_matches_a_pairwise_sum(self):
        m = GaussianMixture(np.array([0.6, 0.4]), np.array([[-1.0], [1.2]]),
                            np.array([0.4, 0.6]))
        rng = np.random.default_rng(1)
        x = rng.normal(size=200)
        y = m.sample(rng, 200_000).ravel()
        brute = float(np.abs(x[:, None] - y[None, :]).mean())
        self.assertAlmostEqual(metrics._mean_abs_cross(x, m), brute, delta=0.01)

    def test_self_distance_of_samples_matches_brute_force(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=500)
        brute = float(np.abs(x[:, None] - x[None, :]).mean())
        self.assertAlmostEqual(metrics._mean_abs_self(x), brute, places=12)


class TestModeWeights(unittest.TestCase):
    def test_a_dropped_mode_is_detected(self):
        m = SDE.marginal(CANONICAL, SDE.t_min)
        good = m.stratified(4096)
        self.assertLess(metrics.mode_weight_error(good, m), 0.01)
        collapsed = good[good.ravel() < 1.0].reshape(-1, 1)   # third mode removed
        self.assertGreater(metrics.mode_weight_error(collapsed, m), 0.2)


if __name__ == "__main__":
    unittest.main()
