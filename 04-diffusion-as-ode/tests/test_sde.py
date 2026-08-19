"""The forward SDE, its marginals, and the closed-form score."""

import unittest

import numpy as np

from src.problem import CANONICAL, SDE
from src.schedule import time_grid, uniform_logsnr_grid, uniform_time_grid
from src.sde import GaussianMixture, VESDE, VPSDE, make_score


class TestSchedule(unittest.TestCase):
    def test_vp_preserves_variance(self):
        t = np.linspace(1e-4, 1.0, 200)
        self.assertTrue(np.allclose(SDE.alpha(t) ** 2 + SDE.sigma(t) ** 2, 1.0, atol=1e-14))

    def test_sigma_is_accurate_at_small_t(self):
        # 1 - exp(-B) loses four digits at B ~ 1e-4; -expm1(-B) does not. Compare
        # against the series sigma^2 = B - B^2/2 + B^3/6, which is exact to 1e-20 there.
        t = 1e-5
        b = float(SDE.beta_integral(t))
        series = b - b * b / 2.0 + b**3 / 6.0
        self.assertAlmostEqual(float(SDE.sigma(t)) ** 2 / series, 1.0, places=13)

    def test_log_snr_inverts(self):
        t = np.linspace(SDE.t_min, SDE.t_max, 500)
        self.assertTrue(np.allclose(SDE.t_of_log_snr(SDE.log_snr(t)), t, atol=1e-12))

    def test_log_snr_is_decreasing_in_t(self):
        t = np.linspace(SDE.t_min, SDE.t_max, 500)
        self.assertTrue(np.all(np.diff(SDE.log_snr(t)) < 0))

    def test_ve_marginal_variance_adds(self):
        ve = VESDE()
        prior = GaussianMixture(np.array([1.0]), np.array([[0.3]]), np.array([0.2]))
        for t in (0.05, 0.5, 1.0):
            got = float(ve.marginal(prior, t).variances[0])
            self.assertAlmostEqual(got, 0.2 + float(ve.sigma(t)) ** 2, places=12)

    def test_grids_are_decreasing_and_hit_the_endpoints(self):
        for kind in ("uniform_t", "uniform_logsnr"):
            g = time_grid(kind, SDE, 17)
            self.assertEqual(g.size, 18)
            self.assertTrue(np.all(np.diff(g) < 0), kind)
            self.assertAlmostEqual(g[0], SDE.t_max)
            self.assertAlmostEqual(g[-1], SDE.t_min)

    def test_unknown_grid_is_rejected(self):
        with self.assertRaises(ValueError):
            time_grid("nope", SDE, 4)


class TestMixture(unittest.TestCase):
    def test_score_matches_finite_differences_of_log_density(self):
        # The score is the whole "network". If it is wrong every other number is too.
        x = np.linspace(-4.0, 4.0, 41).reshape(-1, 1)
        h = 1e-6
        for t in (SDE.t_min, 0.1, 0.5, 1.0):
            p = SDE.marginal(CANONICAL, t)
            fd = (np.log(p.pdf((x + h).ravel())) - np.log(p.pdf((x - h).ravel()))) / (2 * h)
            self.assertLess(float(np.abs(p.score(x).ravel() - fd).max()), 1e-6, f"t={t}")

    def test_marginal_matches_forward_noising(self):
        # p_t should be what you get by actually running the forward kernel on samples
        # from p_0: x_t = alpha x_0 + sigma z.
        rng = np.random.default_rng(3)
        n = 400_000
        x0 = CANONICAL.sample(rng, n)
        for t in (0.2, 0.8):
            a, s = float(SDE.alpha(t)), float(SDE.sigma(t))
            xt = a * x0 + s * rng.normal(size=x0.shape)
            mean, var, _, _ = SDE.marginal(CANONICAL, t).moments()
            self.assertAlmostEqual(float(xt.mean()), mean, delta=6e-3)
            self.assertAlmostEqual(float(xt.var()), var, delta=1e-2)

    def test_quantile_inverts_the_cdf(self):
        p = np.array([1e-6, 1e-3, 0.1, 0.5, 0.9, 1 - 1e-3, 1 - 1e-6])
        for t in (SDE.t_min, 0.3, 1.0):
            m = SDE.marginal(CANONICAL, t)
            self.assertTrue(np.allclose(m.cdf(m.quantile(p)), p, rtol=1e-11, atol=1e-14))

    def test_quantile_is_monotone(self):
        m = SDE.marginal(CANONICAL, SDE.t_min)
        q = m.quantile(np.linspace(1e-4, 1 - 1e-4, 500))
        self.assertTrue(np.all(np.diff(q) > 0))

    def test_moments_match_a_large_sample(self):
        rng = np.random.default_rng(5)
        x = CANONICAL.sample(rng, 500_000).ravel()
        mean, var, skew, _ = CANONICAL.moments()
        self.assertAlmostEqual(float(x.mean()), mean, delta=6e-3)
        self.assertAlmostEqual(float(x.var()), var, delta=2e-2)
        self.assertAlmostEqual(float(np.mean((x - x.mean()) ** 3) / x.var() ** 1.5),
                               skew, delta=2e-2)

    def test_bad_mixtures_raise(self):
        with self.assertRaises(ValueError):
            GaussianMixture(np.array([0.5, 0.4]), np.array([[0.0], [1.0]]),
                            np.array([1.0, 1.0]))
        with self.assertRaises(ValueError):
            GaussianMixture(np.array([1.0]), np.array([[0.0]]), np.array([-1.0]))
        with self.assertRaises(ValueError):
            CANONICAL.quantile(np.array([0.0]))

    def test_score_callable_has_the_network_signature(self):
        score = make_score(SDE, CANONICAL)
        x = np.zeros((5, 1))
        self.assertEqual(score(x, 0.4).shape, (5, 1))


if __name__ == "__main__":
    unittest.main()
