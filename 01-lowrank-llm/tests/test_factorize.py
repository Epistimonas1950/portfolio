"""The core claim of the project, stated as assertions."""

import unittest

import numpy as np

from src.factorize import (activation_error, plain_truncated_svd,
                           relative_activation_error, weight_error, whitened_svd)
from src.synth import make_layer
from src.whiten import whiten, whitened_weights


class TestWhitenedVersusPlain(unittest.TestCase):

    # === THE TEST THAT MATTERS ===
    # Fails if the mathematics is wrong, not merely if the code crashed.
    #
    # At *equal rank* -- therefore at equal parameter count, equal compression, equal
    # everything -- the whitened factorization must beat plain truncated SVD on the
    # activation-weighted error ||(W - W_hat) X||_F when X is ill-conditioned. Plain
    # truncated SVD is the Eckart-Young optimum for a norm that ignores X entirely, so
    # on anisotropic activations it must lose, and it must lose by a margin that no
    # amount of seed luck could produce.
    def test_whitened_beats_plain_on_ill_conditioned_activations(self):
        margins = []
        for seed in range(6):
            layer = make_layer(n_out=96, n_in=128, n_samples=384, cond=1e5, seed=seed)
            plain = plain_truncated_svd(layer.w, 32)
            white = whitened_svd(layer.w, layer.x, 32, ridge=1e-4)
            e_plain = activation_error(layer.w, plain.w_hat, layer.x)
            e_white = activation_error(layer.w, white.w_hat, layer.x)
            self.assertLess(e_white, e_plain, f"seed {seed}: whitening did not help")
            margins.append(e_plain / e_white)
        # Measured on this configuration: 3.04x-3.28x, mean 3.15x. The threshold is
        # set below the worst observed seed, not tuned up to it.
        self.assertGreater(min(margins), 2.5,
                           f"worst seed only {min(margins):.2f}x")
        self.assertGreater(float(np.mean(margins)), 3.0,
                           f"mean margin {np.mean(margins):.2f}x is too small to be "
                           "the mechanism rather than luck")

    # The mirror test, and the reason to believe the one above. On isotropic
    # activations M is a multiple of the identity, S is a multiple of the identity,
    # and the two objectives coincide exactly -- so the advantage must collapse. If it
    # did not, the gain would be coming from somewhere other than the advertised
    # mathematics and the previous test would be measuring an artefact.
    #
    # n_samples = 64 * n_in so the sampled M is close to its isotropic population
    # value; the residual cond(M) ~= 1.6 is Wishart sampling noise, and the residual
    # margin below is that noise being exploited, not a real effect.
    def test_advantage_collapses_on_isotropic_activations(self):
        margins = []
        for seed in range(6):
            layer = make_layer(n_out=64, n_in=64, n_samples=4096, cond=1.0, seed=seed)
            plain = plain_truncated_svd(layer.w, 16)
            white = whitened_svd(layer.w, layer.x, 16, ridge=1e-4)
            margins.append(activation_error(layer.w, plain.w_hat, layer.x)
                           / activation_error(layer.w, white.w_hat, layer.x))
        # Measured: 1.0029x-1.0037x. Two orders of magnitude less advantage than the
        # anisotropic case, from identical code.
        self.assertLess(max(margins), 1.05,
                        f"claimed {max(margins):.4f}x on isotropic X, where S is a "
                        "multiple of the identity and there is nothing to exploit")

    # Plain truncated SVD pays for its blindness to X by being exactly optimal in the
    # unweighted norm. It must never be beaten there, by Eckart-Young. If the whitened
    # factorization won in weight space too, one of the two would not be solving the
    # problem it claims to.
    def test_plain_svd_is_never_beaten_in_the_unweighted_norm(self):
        for seed in range(4):
            layer = make_layer(n_out=64, n_in=96, n_samples=300, cond=1e4, seed=seed)
            plain = plain_truncated_svd(layer.w, 24)
            white = whitened_svd(layer.w, layer.x, 24, ridge=1e-4)
            self.assertLessEqual(weight_error(layer.w, plain.w_hat),
                                 weight_error(layer.w, white.w_hat) + 1e-9,
                                 f"seed {seed}: Eckart-Young violated")

    def test_large_ridge_degrades_whitened_svd_into_plain_svd(self):
        # ||E S||^2 = ||E X||^2 + lambda ||E||^2, so as lambda dominates, the whitened
        # problem becomes the unweighted one and the two factorizations must converge.
        layer = make_layer(n_out=48, n_in=64, n_samples=256, cond=1e5, seed=1)
        plain = plain_truncated_svd(layer.w, 16)
        gaps = []
        for ratio in (1e-2, 1.0, 1e3, 1e6):
            white = whitened_svd(layer.w, layer.x, 16, ridge=ratio)
            gaps.append(float(np.linalg.norm(white.w_hat - plain.w_hat))
                        / float(np.linalg.norm(plain.w_hat)))
        self.assertTrue(all(b < a for a, b in zip(gaps, gaps[1:])),
                        f"gap to plain SVD did not shrink monotonically: {gaps}")
        self.assertLess(gaps[-1], 1e-3, f"ridge 1e6 still {gaps[-1]:.2e} from plain")


class TestFactorizationIsWhatItClaims(unittest.TestCase):

    def test_truncation_is_pythagorean_in_the_whitened_domain(self):
        # ||W S||^2 = ||(W S)_r||^2 + ||(W - W_hat) S||^2. The residual and the kept
        # part are orthogonal, which is true only if the map back through S^{-1} is
        # right -- an S/S^T mix-up breaks this immediately without crashing anything.
        layer = make_layer(n_out=40, n_in=56, n_samples=280, cond=1e4, seed=6)
        wh = whiten(layer.x, ridge_ratio=1e-4)
        ws = whitened_weights(layer.w, wh.s)
        for r in (4, 12, 28):
            fac = whitened_svd(layer.w, layer.x, r, whitening=wh)
            kept = float(np.sum(fac.singular_values[:r] ** 2))
            residual = float(np.linalg.norm((layer.w - fac.w_hat) @ wh.s)) ** 2
            self.assertAlmostEqual((kept + residual) / float(np.linalg.norm(ws)) ** 2,
                                   1.0, places=8, msg=f"r={r}")

    def test_tail_energy_is_the_squared_activation_error(self):
        # The allocation objective in src/allocate.py is sum_{i>r} sigma_i^2. At zero
        # ridge that is not a proxy for ||(W - W_hat) X||_F^2, it *is* it -- and the
        # whole allocation chapter rests on that being true rather than approximately
        # true.
        layer = make_layer(n_out=40, n_in=56, n_samples=560, cond=1e4, seed=8)
        wh = whiten(layer.x, ridge_ratio=0.0)
        for r in (4, 16, 32):
            fac = whitened_svd(layer.w, layer.x, r, whitening=wh)
            measured = activation_error(layer.w, fac.w_hat, layer.x) ** 2
            self.assertAlmostEqual(fac.tail_energy / measured, 1.0, places=6,
                                   msg=f"r={r}")

    def test_factors_have_the_advertised_shapes_and_rank(self):
        layer = make_layer(n_out=40, n_in=56, n_samples=280, seed=10)
        for method in (plain_truncated_svd(layer.w, 13),
                       whitened_svd(layer.w, layer.x, 13, ridge=1e-4)):
            self.assertEqual(method.a.shape, (40, 13))
            self.assertEqual(method.b.shape, (13, 56))
            self.assertEqual(method.shape, (40, 56))
            self.assertEqual(np.linalg.matrix_rank(method.w_hat), 13)

    def test_whitened_solution_is_optimal_against_nearby_rank_r_matrices(self):
        # Eckart-Young in the whitened variable says W_hat is the global minimizer over
        # ALL rank-r matrices, not just the ones an algorithm happens to produce. A
        # cheap empirical check: perturb the factors, keeping the rank, and confirm
        # nothing found is better.
        layer = make_layer(n_out=32, n_in=48, n_samples=240, cond=1e4, seed=12)
        fac = whitened_svd(layer.w, layer.x, 10, ridge=1e-4)
        best = activation_error(layer.w, fac.w_hat, layer.x)
        rng = np.random.default_rng(13)
        for _ in range(25):
            scale = 10.0 ** rng.uniform(-3, -1)
            a = fac.a + scale * rng.normal(size=fac.a.shape) * np.abs(fac.a).mean()
            b = fac.b + scale * rng.normal(size=fac.b.shape) * np.abs(fac.b).mean()
            self.assertGreaterEqual(activation_error(layer.w, a @ b, layer.x),
                                    best - 1e-9)

    def test_relative_error_is_normalized_consistently(self):
        layer = make_layer(n_out=32, n_in=48, n_samples=240, seed=14)
        fac = whitened_svd(layer.w, layer.x, 10, ridge=1e-4)
        rel = relative_activation_error(layer.w, fac.w_hat, layer.x)
        expected = (activation_error(layer.w, fac.w_hat, layer.x)
                    / float(np.linalg.norm(layer.w @ layer.x)))
        self.assertAlmostEqual(rel, expected, places=12)


class TestArgumentChecking(unittest.TestCase):

    def test_rank_above_the_matrix_rank_is_rejected(self):
        layer = make_layer(n_out=16, n_in=32, n_samples=100, seed=0)
        with self.assertRaises(ValueError):
            plain_truncated_svd(layer.w, 17)
        with self.assertRaises(ValueError):
            whitened_svd(layer.w, layer.x, 0, ridge=1e-4)

    def test_shape_mismatch_between_w_and_x_is_reported_clearly(self):
        with self.assertRaises(ValueError):
            whitened_svd(np.zeros((4, 8)), np.zeros((7, 40)), 2)


if __name__ == "__main__":
    unittest.main()
