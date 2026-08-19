"""Grid arithmetic: round-trip fidelity, granularity, and the 8-bit non-result."""

import unittest

import numpy as np

from src.grid import Grid, effective_bits


class TestGrid(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_codes_stay_in_range(self):
        w = self.rng.normal(size=(16, 64)) * 5.0
        for bits in (2, 3, 4, 8):
            for sym in (True, False):
                g = Grid(bits, symmetric=sym)
                s, z = g.params(w)
                q = g.quantize(w, s, z)
                self.assertGreaterEqual(q.min(), g.qmin, f"bits={bits} sym={sym}")
                self.assertLessEqual(q.max(), g.qmax, f"bits={bits} sym={sym}")

    def test_error_bounded_by_half_a_step(self):
        # Round-to-nearest cannot be off by more than half the grid step, for any
        # value inside the represented range. If this fails the scale is wrong.
        w = self.rng.normal(size=(8, 32))
        g = Grid(4)
        s, z = g.params(w)
        err = np.abs(w - g.round_trip(w, s, z))
        self.assertTrue(np.all(err <= s / 2 + 1e-12))

    def test_extra_bit_halves_the_error(self):
        # A uniform grid's step halves with each added bit, so RMS error should too.
        w = self.rng.normal(size=(32, 128))
        errs = []
        for bits in (4, 5, 6, 7):
            g = Grid(bits)
            s, z = g.params(w)
            errs.append(float(np.sqrt(np.mean((w - g.round_trip(w, s, z)) ** 2))))
        for a, b in zip(errs, errs[1:]):
            self.assertAlmostEqual(a / b, 2.0, delta=0.15)

    def test_per_channel_beats_per_tensor_on_ragged_rows(self):
        # Rows deliberately spanning three orders of magnitude: one tensor-wide step
        # is set by the widest row and wastes the grid on all the others.
        #
        # Measured per row and relative. The plain Frobenius norm is the wrong
        # instrument here -- it is dominated by the single largest row, which both
        # methods handle identically, so it hides exactly the effect being tested.
        w = self.rng.normal(size=(12, 64)) * np.logspace(0, 3, 12).reshape(-1, 1)

        def mean_relative_row_error(grid):
            err = w - grid.round_trip(w, *grid.params(w))
            return float(np.mean(np.linalg.norm(err, axis=1)
                                 / np.linalg.norm(w, axis=1)))

        e_t = mean_relative_row_error(Grid(4, per_channel=False))
        e_c = mean_relative_row_error(Grid(4, per_channel=True))
        self.assertLess(e_c, e_t / 5.0)

        # The sharp version of the same statement: under one tensor-wide step, the
        # weakest row falls entirely below half a step and quantizes to exactly zero.
        # The row is not approximated badly -- it is deleted.
        per_tensor = Grid(4, per_channel=False)
        weakest = per_tensor.round_trip(w, *per_tensor.params(w))[0]
        self.assertTrue(np.all(weakest == 0.0))

    def test_effective_bits_counts_metadata(self):
        # 4 "bits" with a per-32-column group scale is not 4 bits per weight.
        # Per-32-column groups: 32 metadata bits per 32 weights per row = +1.0 bit.
        self.assertAlmostEqual(effective_bits(1024, 1024, 4, groupsize=32), 5.0,
                               places=3)
        # One scale+zero for the whole row is 32 bits amortized over 1024 weights.
        self.assertAlmostEqual(effective_bits(1024, 1024, 4, groupsize=None),
                               4.03125, places=5)


if __name__ == "__main__":
    unittest.main()
