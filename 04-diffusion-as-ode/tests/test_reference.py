"""Ground truth: the analytic flow maps, and the high-NFE reference that checks them."""

import unittest

import numpy as np

from src.problem import CANONICAL, GAUSSIAN, SDE
from src.reference import (analytic_gaussian_flow, exact_flow_map, probability_levels,
                           quantile_states, reference_trajectory)
from src.sde import GaussianMixture, make_score, probability_flow_field


class TestAnalyticGaussianFlow(unittest.TestCase):
    def test_it_solves_the_ode(self):
        # Differentiate the claimed solution in t and compare with the field itself.
        # This is the derivation in src/reference.py, checked numerically.
        mu, v = np.array([[0.4]]), 0.3
        score = make_score(SDE, GaussianMixture(np.array([1.0]), mu, np.array([v])))
        x_t0 = np.linspace(-3, 3, 21).reshape(-1, 1)
        t0 = 0.9
        for t in (0.7, 0.4, 0.1, 0.01):
            dt = 1e-6
            xp = analytic_gaussian_flow(SDE, mu, v, x_t0, t0, t + dt)
            xm = analytic_gaussian_flow(SDE, mu, v, x_t0, t0, t - dt)
            x = analytic_gaussian_flow(SDE, mu, v, x_t0, t0, t)
            field = probability_flow_field(SDE, score(x, t), x, t)
            self.assertTrue(np.allclose((xp - xm) / (2 * dt), field, atol=1e-7), f"t={t}")

    def test_it_maps_the_marginal_onto_the_marginal(self):
        mu, v = np.array([[0.4]]), 0.3
        prior = GaussianMixture(np.array([1.0]), mu, np.array([v]))
        rng = np.random.default_rng(0)
        x_t = SDE.marginal(prior, 1.0).sample(rng, 200_000)
        y = analytic_gaussian_flow(SDE, mu, v, x_t, 1.0, 0.05)
        want = SDE.marginal(prior, 0.05)
        self.assertAlmostEqual(float(y.mean()), want.moments()[0], delta=5e-3)
        self.assertAlmostEqual(float(y.var()), want.moments()[1], delta=5e-3)


class TestQuantileFlow(unittest.TestCase):
    def test_exact_flow_map_agrees_with_quantile_states(self):
        probs = probability_levels(65)
        x_hi = quantile_states(SDE, CANONICAL, probs, SDE.t_max)
        x_lo = quantile_states(SDE, CANONICAL, probs, SDE.t_min)
        mapped = exact_flow_map(SDE, CANONICAL, x_hi, SDE.t_max, SDE.t_min)
        self.assertLess(float(np.abs(mapped - x_lo).max()), 1e-9)

    def test_the_map_is_the_gaussian_one_when_the_prior_is_gaussian(self):
        # Two independent derivations of the same object: the affine solution of the
        # linear ODE, and the 1-D monotone transport. They must coincide.
        probs = probability_levels(65)
        x_hi = quantile_states(SDE, GAUSSIAN, probs, SDE.t_max)
        by_quantile = quantile_states(SDE, GAUSSIAN, probs, SDE.t_min)
        by_affine = analytic_gaussian_flow(SDE, GAUSSIAN.means, float(GAUSSIAN.variances[0]),
                                           x_hi, SDE.t_max, SDE.t_min)
        self.assertLess(float(np.abs(by_quantile - by_affine).max()), 1e-10)

    # === THE TEST THAT MATTERS (ground-truth anchor) ===
    # Fails if either the analytic transport argument or the exponential integrator is
    # wrong. A second-order sampler can only converge at order 2 *towards the analytic
    # quantile map* if both are right; a wrong map would leave a constant offset and
    # the ratio below would collapse to 1.
    def test_high_nfe_reference_converges_to_the_analytic_map_at_order_two(self):
        score = make_score(SDE, CANONICAL)
        probs = probability_levels(65)
        x_hi = quantile_states(SDE, CANONICAL, probs, SDE.t_max)
        exact = quantile_states(SDE, CANONICAL, probs, SDE.t_min)
        errs = []
        for n in (256, 512, 1024, 2048):
            got = reference_trajectory(score, SDE, x_hi, SDE.t_max, SDE.t_min, n_steps=n)
            errs.append(float(np.sqrt(np.mean((got - exact) ** 2))))
        for a, b in zip(errs, errs[1:]):
            self.assertAlmostEqual(a / b, 4.0, delta=0.15, msg=f"errors {errs}")
        self.assertLess(errs[-1], 1e-5)


if __name__ == "__main__":
    unittest.main()
