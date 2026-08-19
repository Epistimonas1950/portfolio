"""Empirical order of convergence -- the assertion the whole repo rests on."""

import unittest

import numpy as np

from src.problem import CANONICAL, GAUSSIAN, SDE
from src.reference import probability_levels, quantile_states
from src.samplers import (ODE_SAMPLERS, brownian_increments, coarsen_increments,
                          euler_maruyama)
from src.schedule import uniform_logsnr_grid, uniform_time_grid
from src.sde import make_score

#: Fit window. 32 steps is already asymptotic on the canonical prior (see
#: analysis/convergence_order.py, where the same fit is run from 8 to 1024 and the
#: per-interval orders are printed) and 512 is far above round-off: the smallest error
#: measured here is 1e-6, ten orders above double-precision noise.
STEP_COUNTS = [32, 64, 128, 256, 512]
N_PROBS = 129

EXPECTED = {"euler_ode": 1.0, "heun": 2.0, "exponential_1": 1.0, "exponential_2": 2.0}
TOLERANCE = 0.1


def fit_slope(h, err):
    slope, intercept = np.polyfit(np.log(h), np.log(err), 1)
    resid = np.log(err) - (slope * np.log(h) + intercept)
    return float(slope), float(np.sqrt(np.mean(resid**2)) / np.log(10.0))


class TestOrderOfConvergence(unittest.TestCase):

    # === THE TEST THAT MATTERS ===
    # Fails if the mathematics is wrong, not merely if the code crashed.
    #
    # Each sampler is run on a family of grids whose step size halves, and its error is
    # measured against the *exact* solution -- the 1-D probability-flow map is the
    # quantile map F_t^{-1} o F_s (derived in src/reference.py), so there is no
    # reference solver in the loop and nothing here is limited by a reference's own
    # accuracy. The least-squares slope of log(error) against log(h) must come out at
    # the design order of each method, to within 0.1.
    #
    # This is not a tolerance that can be satisfied by a broken implementation. A wrong
    # sign in the score, a missing factor of 1/2 on g^2, a mis-derived d lambda/dt, or a
    # midpoint evaluated at the wrong time all still run and still produce plausible
    # samples -- and all move the slope off its integer.
    def test_fitted_slopes_match_the_design_orders(self):
        score = make_score(SDE, CANONICAL)
        probs = probability_levels(N_PROBS)
        start = quantile_states(SDE, CANONICAL, probs, SDE.t_max)
        exact = quantile_states(SDE, CANONICAL, probs, SDE.t_min)

        for name, (sampler, _) in ODE_SAMPLERS.items():
            errs, hs = [], []
            for n in STEP_COUNTS:
                grid = uniform_logsnr_grid(SDE, n)
                errs.append(float(np.sqrt(np.mean((sampler(score, SDE, start, grid).x
                                                   - exact) ** 2))))
                hs.append(float(np.max(np.abs(np.diff(grid)))))
            slope, resid = fit_slope(np.array(hs), np.array(errs))
            self.assertLess(resid, 0.02,
                            f"{name}: log-log fit is not a straight line "
                            f"(residual {resid:.3f} decades), errors {errs}")
            self.assertAlmostEqual(slope, EXPECTED[name], delta=TOLERANCE,
                                   msg=f"{name}: fitted slope {slope:.3f}")

    def test_the_exponential_integrator_is_at_least_second_order(self):
        score = make_score(SDE, CANONICAL)
        probs = probability_levels(N_PROBS)
        start = quantile_states(SDE, CANONICAL, probs, SDE.t_max)
        exact = quantile_states(SDE, CANONICAL, probs, SDE.t_min)
        sampler, _ = ODE_SAMPLERS["exponential_2"]
        errs, hs = [], []
        for n in STEP_COUNTS:
            grid = uniform_logsnr_grid(SDE, n)
            errs.append(float(np.sqrt(np.mean((sampler(score, SDE, start, grid).x
                                               - exact) ** 2))))
            hs.append(float(np.max(np.abs(np.diff(grid)))))
        slope, _ = fit_slope(np.array(hs), np.array(errs))
        self.assertGreaterEqual(slope, 2.0 - TOLERANCE, f"fitted slope {slope:.3f}")

    def test_the_second_order_methods_actually_beat_the_first_order_ones(self):
        # An order claim that does not translate into a smaller error at a usable step
        # count is not worth much. At 128 NFE the gap should be at least 10x.
        score = make_score(SDE, CANONICAL)
        probs = probability_levels(N_PROBS)
        start = quantile_states(SDE, CANONICAL, probs, SDE.t_max)
        exact = quantile_states(SDE, CANONICAL, probs, SDE.t_min)
        err = {}
        for name, (sampler, per_step) in ODE_SAMPLERS.items():
            grid = uniform_logsnr_grid(SDE, 128 // per_step)
            got = sampler(score, SDE, start, grid)
            self.assertEqual(got.nfe, 128)
            err[name] = float(np.sqrt(np.mean((got.x - exact) ** 2)))
        self.assertLess(err["heun"] * 10, err["euler_ode"])
        self.assertLess(err["exponential_2"] * 10, err["exponential_1"])


class TestStochasticOrder(unittest.TestCase):

    # === THE TEST THAT MATTERS (stochastic anchor) ===
    # Euler-Maruyama is usually quoted at strong order 1/2. That figure comes from the
    # Milstein correction (1/2) g g_x (dW^2 - dt), which exists only when the diffusion
    # coefficient depends on the state. Here g = g(t) alone -- the noise is additive,
    # as it is in every diffusion model's reverse SDE -- so the correction vanishes
    # identically and Euler-Maruyama *is* Milstein. The strong order is therefore 1,
    # and this test asserts 1 rather than 1/2. Measuring 1/2 would mean the drift or
    # the increments were wrong.
    def test_strong_order_is_one_because_the_noise_is_additive(self):
        n_paths, n_ref = 512, 512
        levels = [8, 16, 32, 64]
        score = make_score(SDE, CANONICAL)
        rng = np.random.default_rng(17)
        x0 = SDE.marginal(CANONICAL, SDE.t_max).stratified(n_paths)
        fine = uniform_time_grid(SDE, n_ref)
        dw = brownian_increments(rng, fine, x0.shape)
        ref = euler_maruyama(score, SDE, x0, fine, increments=dw).x

        errs, hs = [], []
        for n in levels:
            grid = uniform_time_grid(SDE, n)
            got = euler_maruyama(score, SDE, x0, grid,
                                 increments=coarsen_increments(dw, n_ref // n)).x
            errs.append(float(np.sqrt(np.mean((got - ref) ** 2))))
            hs.append(float(np.max(np.abs(np.diff(grid)))))
        slope, resid = fit_slope(np.array(hs), np.array(errs))
        self.assertLess(resid, 0.03, f"strong-error fit is not straight: {errs}")
        self.assertAlmostEqual(slope, 1.0, delta=0.15, msg=f"strong slope {slope:.3f}")

    def test_weak_order_on_the_mean_is_one(self):
        # For a Gaussian prior the reverse SDE is linear with additive noise, so
        #     E[X_{n+1}] = E[X_n] (1 + h A_n) + h b_n
        # exactly -- the mean obeys the deterministic Euler recursion, which is what
        # running the sampler with zero Brownian increments computes. So the weak error
        # in the mean is available with no Monte-Carlo noise at all, and the exact
        # answer is E[X(t_eps)] = alpha(t_eps) mu, because the reverse SDE preserves
        # the marginals.
        score = make_score(SDE, GAUSSIAN)
        mu = float(GAUSSIAN.means[0, 0])
        x0 = np.full((1, 1), mu * float(SDE.alpha(SDE.t_max)))
        want = mu * float(SDE.alpha(SDE.t_min))
        errs, hs = [], []
        for n in (16, 32, 64, 128, 256):
            grid = uniform_time_grid(SDE, n)
            zero = np.zeros((n, 1, 1))
            got = euler_maruyama(score, SDE, x0, grid, increments=zero).x
            errs.append(abs(float(got[0, 0]) - want))
            hs.append(float(np.max(np.abs(np.diff(grid)))))
        slope, resid = fit_slope(np.array(hs), np.array(errs))
        self.assertLess(resid, 0.02, f"weak-error fit is not straight: {errs}")
        self.assertAlmostEqual(slope, 1.0, delta=0.1, msg=f"weak slope {slope:.3f}")

    def test_the_zero_noise_recursion_really_is_the_mean(self):
        # The claim above, checked by simulation: the mean of many noisy paths agrees
        # with the zero-increment run to Monte-Carlo accuracy.
        score = make_score(SDE, GAUSSIAN)
        rng = np.random.default_rng(23)
        n_paths = 40_000
        grid = uniform_time_grid(SDE, 64)
        x0 = np.full((n_paths, 1), float(GAUSSIAN.means[0, 0]) * float(SDE.alpha(SDE.t_max)))
        noisy = euler_maruyama(score, SDE, x0, grid, rng=rng).x
        quiet = euler_maruyama(score, SDE, x0[:1], grid,
                               increments=np.zeros((64, 1, 1))).x
        se = float(noisy.std()) / np.sqrt(n_paths)
        self.assertLess(abs(float(noisy.mean()) - float(quiet[0, 0])), 4.0 * se)


if __name__ == "__main__":
    unittest.main()
