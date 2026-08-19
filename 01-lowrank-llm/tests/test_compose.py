"""Composing low-rank factorization with quantization: the crossover and the ordering.

The brief for this project volunteers that quantization usually beats low-rank at equal
compression. These tests pin that claim down as an inequality, in both directions.
"""

import unittest

import numpy as np

from src.compose import (dense_bits, factored_bits, lowrank_only,
                         lowrank_then_quantize, quantize_only, quantize_then_lowrank,
                         quantized_bits, ranks_for_budget)
from src.factorize import relative_activation_error, whitened_svd
from src.quantize import Grid, rtn, sequential
from src.synth import make_layer

M = N = 128
SAMPLES = 256
FACTOR_BITS = (8, 4, 3, 2)


def best_lowrank_family(layer, target):
    best = None
    for bits in FACTOR_BITS:
        r = ranks_for_budget(M, N, bits, target)
        if r > 0:
            c = lowrank_then_quantize(layer.w, layer.x, r, bits)
            best = c if best is None or c.rel_error < best.rel_error else best
    r = min(M, int((dense_bits(M, N) / target) // ((M + N) * 16)))
    if r > 0:
        c = lowrank_only(layer.w, layer.x, r)
        best = c if best is None or c.rel_error < best.rel_error else best
    return best


def _layers(n=3, **kw):
    opts = dict(n_out=M, n_in=N, n_samples=SAMPLES, cond=1e5, n_spiked=2)
    opts.update(kw)
    return [make_layer(seed=s, **opts) for s in range(n)]


class TestStorageAccounting(unittest.TestCase):
    """The comparison is only meaningful if both arms are measured on the same axis."""

    def test_quantized_storage_counts_the_scales(self):
        # A "4-bit" layer with one fp16 scale per row is not 4 bits per weight, and
        # comparing it against a method whose metadata IS counted is the standard way
        # these tables mislead.
        self.assertEqual(quantized_bits(M, N, 4), M * N * 4 + M * 16)
        self.assertGreater(quantized_bits(M, N, 4) / (M * N), 4.0)

    def test_ranks_for_budget_respects_the_budget(self):
        for bits in FACTOR_BITS:
            for target in (2.0, 4.0, 8.0, 16.0):
                r = ranks_for_budget(M, N, bits, target)
                if r == 0:
                    continue
                budget = dense_bits(M, N) / target
                self.assertLessEqual(factored_bits(M, N, r, bits), budget)
                if r < min(M, N):     # not capped -- one more rank must overshoot
                    self.assertGreater(factored_bits(M, N, r + 1, bits), budget)

    def test_rank_is_capped_at_the_smaller_dimension(self):
        # Above min(m, n) there is nothing left to truncate, and a configuration that
        # asked for more would be quantization with extra steps.
        self.assertLessEqual(ranks_for_budget(M, N, 2, 1.05), min(M, N))


class TestOrdering(unittest.TestCase):

    # === THE TEST THAT MATTERS ===
    # Fails if the derivation in src/compose.py is wrong, not merely if the code crashed.
    #
    # Quantizing first and then factoring cannot save storage. The SVD factors of a
    # grid-valued matrix are not themselves grid-valued, so they must be stored in fp16
    # -- and the storage is then EXACTLY the low-rank-only cost, with the first
    # quantization having bought nothing at all. This is an exact structural identity,
    # asserted with assertEqual and no tolerance.
    def test_quantize_first_buys_no_storage(self):
        layer = _layers(1)[0]
        for r in (8, 32, 64):
            for bits in (4, 3, 2):
                q_then_lr = quantize_then_lowrank(layer.w, layer.x, r, bits,
                                                  requantize=False)
                lr_only = lowrank_only(layer.w, layer.x, r)
                self.assertEqual(q_then_lr.storage_bits, lr_only.storage_bits,
                                 f"r={r} bits={bits}: quantizing first changed the "
                                 "storage, which it cannot do")

    def test_requantizing_recovers_storage_but_rounds_twice(self):
        layer = _layers(1)[0]
        r, bits = 64, 4
        once = lowrank_then_quantize(layer.w, layer.x, r, bits)
        twice = quantize_then_lowrank(layer.w, layer.x, r, bits, requantize=True)
        self.assertEqual(once.storage_bits, twice.storage_bits)
        self.assertGreater(twice.rel_error, once.rel_error,
                           "rounding twice should not be free")


class TestCrossover(unittest.TestCase):

    # === THE TEST THAT MATTERS ===
    # The claim the brief volunteers, as a two-sided inequality.
    #
    # Both arms are given the SAME achieved compression -- the one the quantizer can
    # actually hit at that bit-width -- because quantization only offers a discrete
    # ladder of compressions and picking round targets silently forces it to the next
    # rung up. At 4 bits quantization must win; at 2 bits it must lose. If either
    # direction fails, the crossover is not where this repo says it is.
    def test_quantization_wins_at_four_bits_and_loses_at_two(self):
        layers = _layers(3)
        for bits, expected in ((4, "quantize"), (2, "lowrank")):
            target = dense_bits(M, N) / quantized_bits(M, N, bits)
            q = np.mean([quantize_only(l.w, l.x, bits).rel_error for l in layers])
            lr = np.mean([best_lowrank_family(l, target).rel_error for l in layers])
            winner = "quantize" if q < lr else "lowrank"
            self.assertEqual(winner, expected,
                             f"at {bits} bits ({target:.2f}x) quantize={q:.4f} "
                             f"lowrank={lr:.4f}, expected {expected} to win")

    def test_crossover_survives_the_isotropic_control(self):
        # The crossover must not be an artefact of the anisotropic generator. On
        # isotropic activations -- where whitening has nothing to exploit -- the
        # direction of both inequalities has to be the same.
        layers = _layers(3, cond=1.0, n_spiked=0)
        for bits, expected in ((4, "quantize"), (2, "lowrank")):
            target = dense_bits(M, N) / quantized_bits(M, N, bits)
            q = np.mean([quantize_only(l.w, l.x, bits).rel_error for l in layers])
            lr = np.mean([best_lowrank_family(l, target).rel_error for l in layers])
            self.assertEqual("quantize" if q < lr else "lowrank", expected)


class TestFactorQuantization(unittest.TestCase):

    def test_refit_is_optimal_for_the_quantized_second_factor(self):
        # A* = argmin ||W X - A B_hat X||_F is a least-squares solution, so it cannot be
        # beaten by the A that was chosen for the exact B. Compared BEFORE quantizing A,
        # so this tests the refit and nothing else.
        layer = _layers(1)[0]
        r, grid = 48, Grid(4)
        f = whitened_svd(layer.w, layer.x, r)
        b_hat = sequential(f.b, layer.x, grid)
        from src.compose import _refit_first_factor
        a_star = _refit_first_factor(layer.w, layer.x, b_hat)
        e_star = relative_activation_error(layer.w, a_star @ b_hat, layer.x)
        e_orig = relative_activation_error(layer.w, f.a @ b_hat, layer.x)
        self.assertLessEqual(e_star, e_orig + 1e-12)

    def test_activation_aware_factor_quantization_helps_where_rounding_binds(self):
        # The machinery only pays where rounding is the binding error. At rank near full
        # the truncation tail is negligible and rounding is ~all of the error, so the
        # aware+refit variant must beat naive RTN by a clear margin.
        layer = _layers(1)[0]
        r, bits = min(M, N) - 4, 4
        aware = lowrank_then_quantize(layer.w, layer.x, r, bits, aware=True, refit=True)
        naive = lowrank_then_quantize(layer.w, layer.x, r, bits, aware=False,
                                      refit=False)
        self.assertLess(aware.rel_error, naive.rel_error)

    def test_the_first_factor_sees_the_propagated_activations(self):
        # A multiplies B_hat X, not X. Quantizing it against the wrong Hessian is a real
        # mistake with a measurable cost -- here, against an isotropic surrogate.
        layer = _layers(1)[0]
        r, grid = min(M, N) - 4, Grid(3)
        f = whitened_svd(layer.w, layer.x, r)
        b_hat = sequential(f.b, layer.x, grid)
        from src.compose import _refit_first_factor
        a = _refit_first_factor(layer.w, layer.x, b_hat)

        z = b_hat @ layer.x                                   # the true input to A
        surrogate = np.random.default_rng(0).normal(size=z.shape)   # the wrong one
        right = relative_activation_error(layer.w, sequential(a, z, grid) @ b_hat,
                                          layer.x)
        wrong = relative_activation_error(layer.w, sequential(a, surrogate, grid) @ b_hat,
                                          layer.x)
        self.assertLess(right, wrong)


class TestVendoredQuantizer(unittest.TestCase):
    """Sanity for the vendored subset of project 02, so a bad copy is caught here."""

    def test_sequential_beats_rtn_on_activation_weighted_error(self):
        layer = _layers(1)[0]
        grid = Grid(3)
        e = lambda w_hat: float(np.linalg.norm((layer.w - w_hat) @ layer.x))
        self.assertLess(e(sequential(layer.w, layer.x, grid)), e(rtn(layer.w, grid)))

    def test_rtn_is_optimal_in_weight_space(self):
        layer = _layers(1)[0]
        grid = Grid(4)
        wq = lambda w_hat: float(np.linalg.norm(layer.w - w_hat))
        self.assertLessEqual(wq(rtn(layer.w, grid)),
                             wq(sequential(layer.w, layer.x, grid)) + 1e-9)

    def test_shape_mismatch_is_reported_clearly(self):
        with self.assertRaises(ValueError):
            sequential(np.zeros((4, 8)), np.zeros((7, 20)), Grid(4))


if __name__ == "__main__":
    unittest.main()


class TestOutputContract(unittest.TestCase):
    """The plot script cannot run here (no matplotlib), so pin what it reads.

    A renamed CSV column would otherwise break plotting silently on whatever machine
    does have matplotlib, and the failure would surface far from its cause.
    """

    import csv as _csv
    import pathlib as _pathlib

    RESULTS = _pathlib.Path(__file__).resolve().parents[1] / "results"

    def _columns(self, name):
        path = self.RESULTS / name
        if not path.exists():
            self.skipTest(f"{name} not generated yet -- run `make results`")
        with path.open() as fh:
            return set(next(self._csv.reader(fh)))

    def test_composition_csv_has_the_columns_the_plot_reads(self):
        cols = self._columns("composition.csv")
        for needed in ("spectrum", "achieved_compression", "quantize_only_error",
                       "best_lowrank_family_error", "winner"):
            self.assertIn(needed, cols)

    def test_order_csv_has_the_columns_the_readme_quotes(self):
        cols = self._columns("composition_order.csv")
        for needed in ("regime", "method", "relative_error", "effective_bits",
                       "compression"):
            self.assertIn(needed, cols)
