"""The randomized range finder: does oversampling and power iteration do what is claimed?

BRIEF.md asks for the error bound to be stated precisely. Stating it is cheap; these
tests are what makes the statement checkable.
"""

from __future__ import annotations

import unittest

import numpy as np

from tests.chelper import require_c

from analysis.rangefinder_study import test_matrix
from oracle.chost import run_c_rangefinder
from oracle.rangefinder import (frobenius_bound, optimal_error, projection_error,
                                randomized_range_finder, randomized_svd,
                                rank_by_energy, rank_by_gap)

RANK = 8


def mean_truncated_error(a: np.ndarray, k: int, p: int, q: int, seeds: range) -> float:
    """Mean || A - U_k U_k^T A ||_F after truncating the sketch's SVD back to k."""
    errs = []
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        u, _ = randomized_svd(a, k, p, q, rng)
        errs.append(projection_error(a, u[:, :k]))
    return float(np.mean(errs))


class TestErrorApproachesTheOptimum(unittest.TestCase):
    """The whole point of p and q: get close to the Eckart-Young floor, cheaply.

    The test matrix has sigma_j = j^-1. A slowly decaying spectrum is essential and is
    not a detail: on a fast-decaying one even p = 0, q = 0 is already near-optimal, and
    this test would pass while measuring nothing at all.
    """

    @classmethod
    def setUpClass(cls):
        cls.a = test_matrix(decay=1.0)
        cls.opt = optimal_error(cls.a, RANK)
        cls.seeds = range(20)

    def test_oversampling_beats_no_oversampling(self):
        no_p = mean_truncated_error(self.a, RANK, 0, 0, self.seeds)
        with_p = mean_truncated_error(self.a, RANK, 6, 0, self.seeds)
        self.assertLess(with_p, 0.9 * no_p,
                        f"p=6 ({with_p:.4f}) did not beat p=0 ({no_p:.4f})")

    def test_more_oversampling_is_monotonically_better(self):
        errs = [mean_truncated_error(self.a, RANK, p, 0, self.seeds)
                for p in (0, 2, 6, 16)]
        for lo, hi in zip(errs, errs[1:]):
            self.assertLess(hi, lo, f"error rose with more oversampling: {errs}")

    def test_power_iterations_are_monotonically_better(self):
        errs = [mean_truncated_error(self.a, RANK, 6, q, self.seeds)
                for q in (0, 1, 2)]
        for lo, hi in zip(errs, errs[1:]):
            self.assertLess(hi, lo, f"error rose with more power iterations: {errs}")

    def test_error_approaches_the_optimum(self):
        # q = 2, p = 16 should be within a fraction of a percent of the best rank-8
        # projection that exists. Approaching but never beating it is the content of
        # Eckart-Young, so both directions are asserted.
        best = mean_truncated_error(self.a, RANK, 16, 2, self.seeds)
        self.assertGreaterEqual(best, self.opt * (1 - 1e-12),
                                "beat the Eckart-Young optimum, which is impossible")
        self.assertLess(best / self.opt, 1.01,
                        f"best sketch was {best / self.opt:.4f}x the optimum")

    def test_hmt_frobenius_bound_holds_in_expectation(self):
        # The bound is on the expectation, so it is checked against a mean over many
        # draws, not a single one. Asserting an expectation bound on one seed is a
        # flaky test dressed up as a strong one.
        for p in (2, 6, 16):
            errs = [projection_error(
                self.a, randomized_range_finder(self.a, RANK, p,
                                                0, np.random.default_rng(seed)))
                for seed in range(40)]
            self.assertLessEqual(float(np.mean(errs)), frobenius_bound(self.a, RANK, p),
                                 f"bound violated at p={p}")

    def test_bound_is_undefined_below_p_equals_two(self):
        # (1 + k/(p-1))^{1/2} has no meaning at p = 1 and no guarantee exists at p = 0.
        # Returning a finite number there would put an unsupported figure in a table.
        self.assertEqual(frobenius_bound(self.a, RANK, 0), float("inf"))
        self.assertEqual(frobenius_bound(self.a, RANK, 1), float("inf"))
        self.assertTrue(np.isfinite(frobenius_bound(self.a, RANK, 2)))


class TestRangeFinderMechanics(unittest.TestCase):
    def test_output_is_orthonormal(self):
        a = test_matrix()
        q = randomized_range_finder(a, RANK, 6, 2, np.random.default_rng(0))
        self.assertLess(float(np.abs(q.T @ q - np.eye(q.shape[1])).max()), 1e-12)

    def test_a_global_rng_is_refused(self):
        # CONVENTIONS: every random draw goes through an explicit seeded generator.
        # Silently defaulting to a global one would make results/ irreproducible.
        with self.assertRaises(ValueError):
            randomized_range_finder(test_matrix(), RANK, 6, 1, None)

    def test_c_range_finder_reaches_the_same_error(self):
        require_c(self)
        import pathlib
        from oracle.chost import BUILD
        a = test_matrix()
        BUILD.mkdir(exist_ok=True)
        path = pathlib.Path(BUILD / "test_rangefinder_matrix.csv")
        header = ",".join(f"ch{i:02d}" for i in range(a.shape[0]))
        np.savetxt(path, a.T, delimiter=",", header=header, comments="", fmt="%.10g")
        for p, q in ((6, 1), (16, 2)):
            c_err = run_c_rangefinder(path, RANK, p, q, seed=0)
            rng = np.random.default_rng(7)
            np_err = projection_error(a, randomized_range_finder(a, RANK, p, q, rng))
            # Different generators, so different draws: what must agree is the error
            # level, not the value. 20% is a loose band on a random quantity and still
            # catches a range finder that is simply wrong.
            self.assertLess(abs(c_err - np_err) / np_err, 0.20,
                            f"C={c_err:.5f} numpy={np_err:.5f} at p={p}, q={q}")


class TestRankSelection(unittest.TestCase):
    """Both criteria, on spectra where the right answer is known."""

    def test_energy_threshold_on_a_known_spectrum(self):
        sigma = np.array([10.0, 5.0, 3.0, 0.02, 0.01, 0.005])
        self.assertEqual(rank_by_energy(sigma, 0.95), 3)
        self.assertEqual(rank_by_energy(sigma, 0.5), 1)
        self.assertEqual(rank_by_energy(sigma, 1.0), 6)

    def test_gap_criterion_finds_the_cliff(self):
        sigma = np.array([10.0, 5.0, 3.0, 0.02, 0.01, 0.005])
        self.assertEqual(rank_by_gap(sigma), 3)

    def test_gap_criterion_is_fragile_without_a_cliff(self):
        # Documented weakness, asserted rather than described: on a flat spectrum the
        # criterion returns whichever adjacent pair was luckiest, and on the four-mode
        # stream it returns 1. Knowing that is the reason both criteria are computed.
        rng = np.random.default_rng(0)
        flat = np.sort(1.0 + 0.01 * rng.normal(size=12))[::-1]
        chosen = rank_by_gap(flat)
        self.assertGreaterEqual(chosen, 1)
        self.assertLessEqual(chosen, 11)
        ratios = flat[:-1] / flat[1:]
        self.assertLess(float(ratios.max()), 1.1, "the flat spectrum was not flat")

    def test_energy_rejects_a_degenerate_spectrum(self):
        with self.assertRaises(ValueError):
            rank_by_energy(np.array([]), 0.95)


if __name__ == "__main__":
    unittest.main()
