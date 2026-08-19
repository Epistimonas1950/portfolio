"""The whitening identities. If any of these fails, everything downstream is fiction."""

import unittest

import numpy as np

from src.synth import make_layer
from src.whiten import (apply_ridge, condition_number, second_moment,
                        solve_upper_triangular, unwhiten_rows, whiten,
                        whitening_factor)


class TestWhiteningIdentities(unittest.TestCase):

    def test_norm_identity_at_zero_ridge(self):
        # ||E X||_F = ||E S||_F with M = X X^T = S S^T. This is the change of variable
        # the entire repo is built on; without it, truncating W S has no bearing on
        # the activation-weighted error at all.
        layer = make_layer(n_out=32, n_in=48, n_samples=400, cond=1e4, seed=0)
        wh = whiten(layer.x, ridge_ratio=0.0)
        e = np.random.default_rng(1).normal(size=layer.w.shape)
        lhs = float(np.linalg.norm(e @ layer.x))
        rhs = float(np.linalg.norm(e @ wh.s))
        self.assertAlmostEqual(lhs / rhs, 1.0, places=10)

    def test_ridge_is_an_exact_interpolation_between_the_two_objectives(self):
        # ||E S||^2 = ||E X||^2 + lambda ||E||^2 for M + lambda I = S S^T. This is the
        # justification for the ridge that the README makes: it is not a fudge, it is
        # a convex combination of the activation-weighted and unweighted objectives,
        # and lambda -> infinity recovers plain truncated SVD.
        layer = make_layer(n_out=24, n_in=40, n_samples=300, cond=1e3, seed=2)
        e = np.random.default_rng(3).normal(size=layer.w.shape)
        for ratio in (1e-4, 1e-2, 1.0, 10.0):
            wh = whiten(layer.x, ridge_ratio=ratio)
            lhs = float(np.linalg.norm(e @ wh.s)) ** 2
            rhs = (float(np.linalg.norm(e @ layer.x)) ** 2
                   + wh.lam * float(np.linalg.norm(e)) ** 2)
            self.assertAlmostEqual(lhs / rhs, 1.0, places=9, msg=f"ratio={ratio}")

    def test_unwhiten_rows_inverts_the_whitening(self):
        # B = V^T S^{-1} must satisfy B S = V^T exactly, or the map back from the
        # whitened domain is wrong and W_hat is not the matrix we think it is.
        layer = make_layer(n_out=16, n_in=32, n_samples=200, cond=1e3, seed=4)
        wh = whiten(layer.x, ridge_ratio=1e-4)
        vt = np.linalg.svd(np.random.default_rng(5).normal(size=(8, 32)),
                           full_matrices=False)[2]
        b = unwhiten_rows(vt, wh.s)
        self.assertTrue(np.allclose(b @ wh.s, vt, atol=1e-9 * np.linalg.norm(vt)))


class TestTriangularSolve(unittest.TestCase):

    def test_back_substitution_matches_the_general_solver(self):
        # The hand-rolled back substitution exists because scipy is not available
        # here; it does not get to be trusted on that basis alone.
        rng = np.random.default_rng(7)
        u = np.triu(rng.normal(size=(40, 40))) + 40.0 * np.eye(40)
        for rhs_shape in ((40, 5), (40, 1)):
            b = rng.normal(size=rhs_shape)
            mine = solve_upper_triangular(u, b)
            reference = np.linalg.solve(u, b)
            self.assertTrue(np.allclose(mine, reference, atol=1e-10),
                            f"disagreement for rhs {rhs_shape}")

    def test_zero_pivot_is_reported_not_divided_by(self):
        u = np.eye(4)
        u[2, 2] = 0.0
        with self.assertRaises(np.linalg.LinAlgError):
            solve_upper_triangular(u, np.ones((4, 1)))

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            solve_upper_triangular(np.eye(4), np.ones((5, 1)))
        with self.assertRaises(ValueError):
            solve_upper_triangular(np.ones((4, 5)), np.ones((4, 1)))


class TestConditioning(unittest.TestCase):

    def test_undamped_rank_deficient_second_moment_fails_loudly(self):
        # Fewer calibration samples than input channels makes M = X X^T singular by
        # construction. Cholesky must raise rather than hand back an S whose inverse
        # silently amplifies noise by 10^8.
        rng = np.random.default_rng(0)
        x = rng.normal(size=(48, 16))
        with self.assertRaises(np.linalg.LinAlgError):
            whitening_factor(second_moment(x))

    def test_ridge_makes_the_rank_deficient_case_solvable(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(48, 16))
        m, lam = apply_ridge(second_moment(x), 1e-2)
        self.assertGreater(lam, 0.0)
        whitening_factor(m)                       # must not raise

    def test_ridge_ratio_is_scale_free(self):
        # The same dimensionless ratio must mean the same thing for layers whose
        # activation energies differ by orders of magnitude -- that is the entire
        # point of scaling lambda by mean(diag M) instead of using an absolute value.
        layer = make_layer(n_out=8, n_in=24, n_samples=120, cond=1e3, seed=9)
        c1 = condition_number(apply_ridge(second_moment(layer.x), 1e-2)[0])
        c2 = condition_number(apply_ridge(second_moment(layer.x * 1000.0), 1e-2)[0])
        self.assertAlmostEqual(c1 / c2, 1.0, places=6)

    def test_condition_number_of_a_singular_matrix_is_infinite(self):
        m = np.diag([1.0, 1.0, 0.0])
        self.assertEqual(condition_number(m), float("inf"))

    def test_negative_ridge_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_ridge(np.eye(3), -1e-3)


if __name__ == "__main__":
    unittest.main()
