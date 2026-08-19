"""The core claim of the project, stated as assertions."""

import unittest

import numpy as np

from src.grid import Grid
from src.rtn import rtn_quantize
from src.sequential import output_error, sequential_quantize
from src.synth import make_layer


class TestSequentialQuantization(unittest.TestCase):

    # === THE TEST THAT MATTERS ===
    # Fails if the mathematics is wrong, not merely if the code crashed.
    #
    # Sequential quantization with Cholesky error compensation must beat
    # round-to-nearest on the ACTIVATION-WEIGHTED error ||(W - W_hat) X||_F at low
    # bit-width. That is the entire thesis: RTN is exactly optimal for ||W - W_hat||_F
    # and blind to X, so on anisotropic activations it loses -- and it must lose by a
    # clear margin, not by noise.
    def test_compensation_beats_rtn_on_output_error_at_3_bits(self):
        grid = Grid(3)
        margins = []
        for seed in range(5):
            layer = make_layer(n_out=96, n_in=192, n_samples=384, cond=1e4,
                               n_outliers=0, seed=seed)
            rtn = rtn_quantize(layer.w, grid)
            seq = sequential_quantize(layer.w, layer.x, grid, damping=1e-2,
                                      ordering="salience")
            e_rtn = output_error(layer.w, rtn, layer.x)
            self.assertLess(seq.output_error, e_rtn,
                            f"seed {seed}: compensation did not beat RTN")
            margins.append(e_rtn / seq.output_error)
        self.assertGreater(float(np.mean(margins)), 2.0,
                           f"margin {np.mean(margins):.2f}x is too small to be the "
                           "mechanism rather than luck")

    # The other half of the claim: the advantage is bought by exploiting the anisotropy
    # of X. Remove the anisotropy and it must largely evaporate -- if it does not, the
    # improvement is coming from somewhere other than the mathematics we advertise.
    def test_advantage_collapses_on_isotropic_activations(self):
        grid = Grid(3)
        layer = make_layer(n_out=96, n_in=192, n_samples=768, cond=1.0,
                           n_outliers=0, seed=3)
        rtn = rtn_quantize(layer.w, grid)
        seq = sequential_quantize(layer.w, layer.x, grid, damping=1e-2,
                                  ordering="natural")
        ratio = output_error(layer.w, rtn, layer.x) / seq.output_error
        self.assertLess(ratio, 2.0,
                        f"claimed {ratio:.2f}x on isotropic X, where H is a multiple "
                        "of the identity and there is nothing to exploit")

    # RTN pays for weight-space optimality: it must never be beaten on ||W - W_hat||_F,
    # by construction. If the sequential quantizer wins there too, one of the two is
    # not solving the problem it claims to.
    def test_rtn_is_optimal_in_weight_space(self):
        grid = Grid(4)
        layer = make_layer(n_out=64, n_in=128, n_samples=256, seed=11)
        rtn = rtn_quantize(layer.w, grid)
        seq = sequential_quantize(layer.w, layer.x, grid)
        self.assertLessEqual(np.linalg.norm(layer.w - rtn), seq.weight_error + 1e-9)

    def test_eight_bits_is_nearly_free(self):
        # The result that is not a result: at 8 bits the grid is fine enough that the
        # choice of method barely registers. Asserted so the README's claim about it
        # is backed by something.
        layer = make_layer(seed=5)
        grid = Grid(8)
        rtn = rtn_quantize(layer.w, grid)
        rel = output_error(layer.w, rtn, layer.x) / np.linalg.norm(layer.w @ layer.x)
        self.assertLess(rel, 0.02)

    def test_outlier_channels_kept_in_fp16_reduce_error(self):
        layer = make_layer(n_outliers=6, outlier_scale=30.0, seed=13)
        grid = Grid(3)
        plain = sequential_quantize(layer.w, layer.x, grid, n_outliers=0)
        kept = sequential_quantize(layer.w, layer.x, grid, n_outliers=6)
        self.assertLess(kept.output_error, plain.output_error)
        # And it must find the channels that were actually spiked.
        self.assertTrue(set(kept.outlier_cols).issuperset(set(layer.outlier_channels)))

    def test_shape_mismatch_is_reported_clearly(self):
        with self.assertRaises(ValueError):
            sequential_quantize(np.zeros((4, 8)), np.zeros((7, 20)), Grid(4))


if __name__ == "__main__":
    unittest.main()
