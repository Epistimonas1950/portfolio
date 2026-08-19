"""The exponential integrator's exactness condition, and the closed-form deficit."""

import unittest

import numpy as np

from src.problem import POINT_MASS, SDE
from src.reference import analytic_gaussian_flow
from src.samplers import euler_ode, exponential, heun
from src.samplers.exponential import dpm_solver_1_multiplier
from src.schedule import uniform_logsnr_grid
from src.sde import GaussianMixture, VESDE, make_score

VARIANCES = [1.0, 0.3, 0.05, 1e-3, 1e-6]


def one_step(sde, t_from, t_to):
    return np.array([t_from, t_to])


class TestExactnessOnAPointMass(unittest.TestCase):

    # === THE TEST THAT MATTERS (exponential anchor) ===
    # Fails if the mathematics is wrong, not merely if the code crashed.
    #
    # DPM-Solver-1 is exact in one step precisely when the noise prediction eps is
    # constant along the trajectory. For a point-mass prior it is: p_t = N(alpha mu,
    # sigma^2), the flow is x(t) = alpha_t mu + (sigma_t/sigma_s)(x(s) - alpha_s mu),
    # so eps = (x - alpha mu)/sigma does not move. One step -- the whole interval, in a
    # single step -- must then reproduce the analytic solution to machine precision.
    # Nothing about this is a smoke test: a wrong sign, a wrong log-SNR, a missing
    # alpha_t/alpha_s, or expm1 of the wrong argument all leave a visible residual.
    #
    # On "machine precision": taking the whole interval in ONE step multiplies x by
    # alpha_{t_eps}/alpha_T = 152 and then subtracts a term of the same size, so the
    # attainable absolute accuracy is set by cancellation at ~1e-13 (1.3e-15 relative
    # to the intermediate magnitude), not by 1e-16. At two or more steps the
    # cancellation disappears and the residual drops to 2e-15, which the next test
    # asserts. Both tolerances are read off that arithmetic, not tuned until green.
    def test_one_step_is_exact_for_a_point_mass(self):
        mu = POINT_MASS.means
        score = make_score(SDE, POINT_MASS)
        x = np.linspace(-3.0, 3.0, 25).reshape(-1, 1)
        grid = one_step(SDE, SDE.t_max, SDE.t_min)
        exact = analytic_gaussian_flow(SDE, mu, 0.0, x, SDE.t_max, SDE.t_min)

        got = exponential(score, SDE, x, grid, order=1)
        self.assertEqual(got.nfe, 1)
        self.assertLess(float(np.abs(got.x - exact).max()), 1e-11,
                        "exponential integrator is not exact on a point mass")
        self.assertLess(float(np.abs(exponential(score, SDE, x, grid, order=2).x
                                     - exact).max()), 1e-9)

        # ... and the explicit methods are nowhere near, on the same single step.
        for name, sampler in (("euler", euler_ode), ("heun", heun)):
            err = float(np.abs(sampler(score, SDE, x, grid).x - exact).max())
            self.assertGreater(err, 1e-2, f"{name} was unexpectedly exact")

    def test_it_is_exact_on_every_grid_and_every_step_count(self):
        score = make_score(SDE, POINT_MASS)
        x = np.linspace(-3.0, 3.0, 17).reshape(-1, 1)
        exact = analytic_gaussian_flow(SDE, POINT_MASS.means, 0.0, x, SDE.t_max, SDE.t_min)
        for n in (2, 5, 37, 100):
            for order in (1, 2):
                got = exponential(score, SDE, x, uniform_logsnr_grid(SDE, n), order=order)
                self.assertLess(float(np.abs(got.x - exact).max()), 1e-13,
                                f"n={n}, order={order}")

    def test_the_same_holds_for_the_variance_exploding_schedule(self):
        ve = VESDE()
        prior = GaussianMixture(np.array([1.0]), np.array([[0.5]]), np.array([0.0]))
        score = make_score(ve, prior)
        x = np.linspace(-30.0, 30.0, 17).reshape(-1, 1)
        exact = analytic_gaussian_flow(ve, prior.means, 0.0, x, ve.t_max, ve.t_min)
        got = exponential(score, ve, x, one_step(ve, ve.t_max, ve.t_min), order=1)
        self.assertLess(float(np.abs(got.x - exact).max()), 1e-11)


class TestTheClosedFormDeficit(unittest.TestCase):
    # For a Gaussian prior of variance v > 0 the exponential integrator is *not* exact,
    # and the derivation in src/samplers/exponential.py says by exactly how much:
    #
    #     R_exact^2 - R_dpm1^2 = v (alpha_t sigma_s - sigma_t alpha_s)^2 / V_s^2
    #
    # Asserting that identity against the measured one-step error is a much sharper
    # statement about the implementation than the exactness test above, because it
    # pins the whole v-dependence rather than one point of it.
    def test_measured_one_step_error_matches_the_identity(self):
        for v in VARIANCES:
            mu = np.array([[0.7]])
            prior = GaussianMixture(np.array([1.0]), mu, np.array([v]))
            score = make_score(SDE, prior)
            for t_from, t_to in ((1.0, 0.001), (0.8, 0.2), (0.3, 0.05)):
                x = np.array([[2.5]])
                got = exponential(score, SDE, x, one_step(SDE, t_from, t_to), order=1).x
                exact = analytic_gaussian_flow(SDE, mu, v, x, t_from, t_to)
                a0 = float(SDE.alpha(t_from))
                offset = float(x[0, 0]) - a0 * float(mu[0, 0])
                r_exact, r_dpm = dpm_solver_1_multiplier(SDE, v, t_from, t_to)

                measured = float(exact[0, 0] - got[0, 0]) / offset
                self.assertAlmostEqual(measured, r_exact - r_dpm, places=11,
                                       msg=f"v={v}, {t_from}->{t_to}")

                a1, s0 = float(SDE.alpha(t_to)), float(SDE.sigma(t_from))
                s1 = float(SDE.sigma(t_to))
                v0 = a0 * a0 * v + s0 * s0
                predicted = v * (a1 * s0 - s1 * a0) ** 2 / (v0 * v0)
                self.assertAlmostEqual(r_exact**2 - r_dpm**2, predicted, places=14,
                                       msg=f"v={v}, {t_from}->{t_to}")

    def test_the_deficit_is_non_negative_and_vanishes_only_at_v_zero(self):
        for v in VARIANCES + [0.0]:
            r_exact, r_dpm = dpm_solver_1_multiplier(SDE, v, 1.0, 0.001)
            self.assertGreater(r_dpm, 0.0, f"v={v}")
            self.assertLessEqual(r_dpm, r_exact + 1e-15, f"v={v}")
            if v == 0.0:
                self.assertAlmostEqual(r_dpm, r_exact, places=15)
            else:
                self.assertLess(r_dpm, r_exact, f"v={v}")


if __name__ == "__main__":
    unittest.main()
