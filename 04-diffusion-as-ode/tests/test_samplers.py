"""Sampler mechanics: NFE accounting, marginal preservation, adaptive control."""

import unittest

import numpy as np

from src.nfe import ScoreCounter
from src.problem import CANONICAL, SDE
from src.reference import exact_flow_map, probability_levels, quantile_states
from src.samplers import (adaptive_heun, brownian_increments, coarsen_increments,
                          euler_maruyama, euler_ode, exponential, heun)
from src.schedule import uniform_logsnr_grid, uniform_time_grid
from src.sde import GaussianMixture, make_score

STANDARD_NORMAL = GaussianMixture(np.array([1.0]), np.array([[0.0]]), np.array([1.0]))


class TestNfeAccounting(unittest.TestCase):
    def test_counter_counts_calls_not_samples(self):
        # One forward pass over a batch is one NFE. Counting per sample would make
        # every batched sampler look arbitrarily expensive.
        c = ScoreCounter(lambda x, t: x)
        c(np.zeros((1000, 1)), 0.5)
        c(np.zeros((3, 1)), 0.4)
        self.assertEqual(c.nfe, 2)

    def test_fixed_step_nfe_is_exactly_the_advertised_rate(self):
        score = make_score(SDE, CANONICAL)
        x = STANDARD_NORMAL.stratified(32)
        grid = uniform_logsnr_grid(SDE, 20)
        self.assertEqual(euler_ode(score, SDE, x, grid).nfe, 20)
        self.assertEqual(heun(score, SDE, x, grid).nfe, 40)
        self.assertEqual(exponential(score, SDE, x, grid, order=1).nfe, 20)
        self.assertEqual(exponential(score, SDE, x, grid, order=2).nfe, 40)
        rng = np.random.default_rng(0)
        self.assertEqual(euler_maruyama(score, SDE, x, grid, rng=rng).nfe, 20)

    def test_adaptive_counts_rejected_steps(self):
        score = make_score(SDE, CANONICAL)
        x = STANDARD_NORMAL.stratified(64)
        res = adaptive_heun(score, SDE, x, rtol=1e-3, atol=1e-4)
        self.assertEqual(res.nfe, 2 * (res.accepted + res.rejected))
        self.assertGreater(res.accepted, 0)

    def test_bad_order_and_missing_noise_raise(self):
        score = make_score(SDE, CANONICAL)
        x = STANDARD_NORMAL.stratified(4)
        with self.assertRaises(ValueError):
            exponential(score, SDE, x, uniform_logsnr_grid(SDE, 4), order=3)
        with self.assertRaises(ValueError):
            euler_maruyama(score, SDE, x, uniform_logsnr_grid(SDE, 4))


class TestBrownianRefinement(unittest.TestCase):
    # The strong-order measurement is only meaningful if the coarse path really is the
    # fine path. A bug here produces a plausible-looking wrong slope, so the instrument
    # is checked before it is used.
    def test_coarsening_sums_the_fine_increments_exactly(self):
        rng = np.random.default_rng(0)
        times = uniform_time_grid(SDE, 64)
        dw = brownian_increments(rng, times, (5, 1))
        c = coarsen_increments(dw, 8)
        self.assertEqual(c.shape, (8, 5, 1))
        self.assertTrue(np.array_equal(c[0], dw[:8].sum(axis=0)))
        self.assertAlmostEqual(float(np.abs(c.sum(0) - dw.sum(0)).max()), 0.0, places=15)

    def test_coarsening_rejects_a_non_divisor(self):
        rng = np.random.default_rng(0)
        dw = brownian_increments(rng, uniform_time_grid(SDE, 10), (2, 1))
        with self.assertRaises(ValueError):
            coarsen_increments(dw, 3)


class TestMarginalPreservation(unittest.TestCase):

    # === THE TEST THAT MATTERS (marginal anchor) ===
    # Fails if the mathematics is wrong, not merely if the code crashed.
    #
    # The probability-flow ODE is *defined* by the property that it transports p_T to
    # p_t. Integrating it from T down to t_eps must therefore reproduce the analytic
    # mean and variance of p_{t_eps}. Get the schedule derivative, the factor of 1/2 on
    # g^2, or the sign of the score wrong and the samples still look like samples --
    # but the moments move.
    #
    # The initial ensemble is the exact midpoint quantiles of p_T, so it carries no
    # Monte-Carlo error; the residual floor is the quadrature error of that ensemble,
    # measured in the first two assertions below and about 8e-6 in the mean and 4.5e-4
    # in the variance for n = 1024. The tolerances are set from that floor, not tuned.
    def test_probability_flow_reproduces_the_analytic_marginal(self):
        n = 1024
        score = make_score(SDE, CANONICAL)
        start = SDE.marginal(CANONICAL, SDE.t_max).stratified(n)
        target = SDE.marginal(CANONICAL, SDE.t_min)
        mean, var, _, _ = target.moments()

        floor = target.stratified(n)
        floor_mean = abs(float(floor.mean()) - mean)
        floor_var = abs(float(floor.var()) - var)
        self.assertLess(floor_mean, 5e-5)
        self.assertLess(floor_var, 1e-3)

        for name, sampler in (("heun", heun),
                              ("exponential_2",
                               lambda s, sde, x, g: exponential(s, sde, x, g, order=2))):
            out = sampler(score, SDE, start, uniform_logsnr_grid(SDE, 256)).x
            self.assertAlmostEqual(float(out.mean()), mean, delta=5e-5, msg=name)
            self.assertAlmostEqual(float(out.var()), var, delta=1.5e-3, msg=name)

    def test_the_sde_reproduces_the_same_marginal(self):
        # Same theorem, stochastic branch: the reverse SDE has the same marginals, so
        # its samples must match those moments too -- to Monte-Carlo accuracy this time.
        n = 60_000
        score = make_score(SDE, CANONICAL)
        rng = np.random.default_rng(4)
        start = SDE.marginal(CANONICAL, SDE.t_max).sample(rng, n)
        out = euler_maruyama(score, SDE, start, uniform_logsnr_grid(SDE, 400), rng=rng).x
        mean, var, _, _ = SDE.marginal(CANONICAL, SDE.t_min).moments()
        self.assertAlmostEqual(float(out.mean()), mean, delta=0.02)
        self.assertAlmostEqual(float(out.var()), var, delta=0.05)


class TestAdaptive(unittest.TestCase):
    def test_it_honours_its_tolerance_and_beats_the_fixed_grid(self):
        # "Fixed grid" here is uniform in log-SNR, which is the strong baseline --
        # what production samplers use -- not uniform in t, which would be easy to beat.
        score = make_score(SDE, CANONICAL)
        probs = probability_levels(257)
        start = quantile_states(SDE, CANONICAL, probs, SDE.t_max)
        exact = quantile_states(SDE, CANONICAL, probs, SDE.t_min)

        rtol = 1e-2
        res = adaptive_heun(score, SDE, start, rtol=rtol, atol=rtol / 10.0)
        err = float(np.sqrt(np.mean((res.x - exact) ** 2)))
        self.assertLess(err, rtol, "adaptive missed its requested tolerance")

        need = None
        for n in range(2, 200):
            got = heun(score, SDE, start, uniform_logsnr_grid(SDE, n))
            if float(np.sqrt(np.mean((got.x - exact) ** 2))) <= err:
                need = got.nfe
                break
        self.assertIsNotNone(need, "fixed-step Heun never reached the adaptive error")
        self.assertLess(res.nfe, need,
                        f"adaptive used {res.nfe} NFE, fixed-step Heun {need}")

    def test_it_refuses_an_impossible_request(self):
        score = make_score(SDE, CANONICAL)
        x = STANDARD_NORMAL.stratified(8)
        with self.assertRaises(RuntimeError):
            adaptive_heun(score, SDE, x, rtol=1e-14, atol=1e-16, max_steps=200)

    def test_it_rejects_a_forward_time_request(self):
        score = make_score(SDE, CANONICAL)
        x = STANDARD_NORMAL.stratified(4)
        with self.assertRaises(ValueError):
            adaptive_heun(score, SDE, x, t_start=SDE.t_min, t_end=SDE.t_max)


if __name__ == "__main__":
    unittest.main()
