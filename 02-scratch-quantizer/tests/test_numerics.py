"""Damping, ordering, and the cross-layer error bound -- the numerical claims."""

import unittest

import numpy as np

from src.grid import Grid
from src.hessian import damp, hessian, inverse_cholesky
from src.ordering import ORDERINGS, column_order
from src.propagation import propagate
from src.sequential import sequential_quantize
from src.synth import make_layer, make_stack


class TestHessian(unittest.TestCase):
    def test_inverse_cholesky_reconstructs_the_inverse(self):
        layer = make_layer(n_out=8, n_in=48, n_samples=200, cond=1e3, seed=2)
        h, _ = damp(hessian(layer.x), 1e-3)
        r = inverse_cholesky(h)
        self.assertTrue(np.allclose(np.tril(r, -1), 0.0), "R must be upper triangular")
        self.assertTrue(np.allclose(r.T @ r, np.linalg.inv(h), atol=1e-8 * np.linalg.norm(np.linalg.inv(h))))

    def test_undamped_singular_hessian_fails_loudly(self):
        # Fewer samples than input channels => X X^T is rank-deficient by construction.
        # Cholesky must raise rather than return quietly wrong numbers.
        rng = np.random.default_rng(0)
        x = rng.normal(size=(40, 12))
        with self.assertRaises(np.linalg.LinAlgError):
            inverse_cholesky(hessian(x))

    def test_damping_makes_it_solvable(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(40, 12))
        h, lam = damp(hessian(x), 1e-2)
        self.assertGreater(lam, 0.0)
        inverse_cholesky(h)  # must not raise

    def test_damping_ratio_is_scale_free(self):
        # The same ratio must mean the same thing for activations whose energies
        # differ by orders of magnitude -- that is the point of scaling by the mean
        # diagonal instead of using an absolute lambda.
        layer = make_layer(n_in=32, n_samples=128, seed=4)
        h1 = hessian(layer.x)
        h2 = hessian(layer.x * 1000.0)
        c1 = np.linalg.cond(damp(h1, 1e-2)[0])
        c2 = np.linalg.cond(damp(h2, 1e-2)[0])
        self.assertAlmostEqual(c1 / c2, 1.0, places=4)


class TestOrdering(unittest.TestCase):
    def test_all_orderings_are_permutations(self):
        layer = make_layer(n_out=16, n_in=64, n_samples=128, seed=6)
        h = hessian(layer.x)
        for name in ORDERINGS:
            order = column_order(name, layer.w, h)
            self.assertEqual(sorted(order.tolist()), list(range(64)), name)

    def test_unknown_ordering_is_rejected(self):
        with self.assertRaises(ValueError):
            column_order("nope", np.zeros((2, 2)), np.eye(2))

    def test_salience_ordering_helps_when_activations_are_uneven(self):
        # With a few dominant input channels, quantizing them first leaves the whole
        # rest of the layer available to absorb their error. Against a weight-space
        # ordering that ignores activations entirely, that should show.
        wins = 0
        for seed in range(6):
            layer = make_layer(n_out=64, n_in=128, n_samples=256, cond=1e5,
                               n_outliers=8, outlier_scale=25.0, seed=seed)
            grid = Grid(3)
            sal = sequential_quantize(layer.w, layer.x, grid, ordering="salience")
            mag = sequential_quantize(layer.w, layer.x, grid, ordering="magnitude")
            wins += sal.output_error < mag.output_error
        self.assertGreaterEqual(wins, 5, f"salience won only {wins}/6")


class TestPropagation(unittest.TestCase):

    # === THE TEST THAT MATTERS (second anchor) ===
    # Fails if the derivation in src/propagation.py is wrong.
    #
    # The unrolled recursion is an upper bound on the measured end-to-end error, and
    # it must hold at every depth. If the measured error ever exceeds the prediction,
    # the bound is not a bound and the README's central plot is a lie.
    def test_bound_dominates_measurement_at_every_layer(self):
        weights, x0 = make_stack(n_layers=6, width=96, n_samples=192, seed=1)
        grid = Grid(3)
        hats, activations = [], x0
        for w in weights:
            res = sequential_quantize(w, activations, grid, damping=1e-2)
            hats.append(res.w_hat)
            activations = w @ activations
        rows = propagate(weights, hats, x0)
        for row in rows:
            self.assertGreaterEqual(row.predicted, row.measured - 1e-9,
                                    f"layer {row.layer}: bound violated")

    def test_bound_is_loose_and_gets_looser_with_depth(self):
        # Rounding errors from independent decisions are close to orthogonal, so the
        # triangle inequality overshoots -- and compounds. A tight bound here would
        # mean the layers' errors were conspiring, which would be the real finding.
        weights, x0 = make_stack(n_layers=6, width=96, n_samples=192, seed=1)
        grid = Grid(3)
        hats, activations = [], x0
        for w in weights:
            hats.append(sequential_quantize(w, activations, grid).w_hat)
            activations = w @ activations
        rows = propagate(weights, hats, x0)
        self.assertGreater(rows[-1].ratio, rows[0].ratio)


if __name__ == "__main__":
    unittest.main()
