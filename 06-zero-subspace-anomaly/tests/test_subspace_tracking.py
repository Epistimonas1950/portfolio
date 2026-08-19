"""The two claims the whole project rests on: orthogonality, and agreement with batch.

Everything else in the repo could be right and, if these two fail, the tracker would
still be producing plausible-looking numbers from a basis that is no longer a basis.
"""

from __future__ import annotations

import pathlib
import unittest

import numpy as np

from tests.chelper import ROOT, require_c, require_data

from oracle.batch_svd import read_stream
from oracle.chost import TRACKER, TRACKER32, CDefaults, run_c_tracker
from oracle.incremental import (CHECK_EVERY, REORTH_TOL, orthogonality_drift,
                                principal_angles, projection_distance,
                                reorthonormalize, run_stream)

DATA = ROOT / "data"
FLOAT32_TOL = 100.0 * float(np.finfo(np.float32).eps)


# === THE TEST THAT MATTERS ===
# Fails if the mathematics is wrong, not merely if the code crashed.
class TestOrthogonalityDrift(unittest.TestCase):
    """Orthogonality drift: bounded with periodic repair, growing without it.

    Repeated rank-one updates lose orthogonality of U in floating point. The loss is
    SILENT: no exception, no NaN, no value out of range, and the detector keeps emitting
    scores from a basis whose columns are no longer orthonormal, so U U^T is no longer a
    projection and `|| a - U U^T a ||` is no longer a residual. The only defence is to
    measure `|| U^T U - I ||_F` and repair it, and this test asserts BOTH halves of that
    claim, because either one alone is worthless:

      1. WITH periodic reorthogonalization the drift stays pinned near the threshold.
         Without half 2, a monitor that never fires would pass this trivially.
      2. WITHOUT it the drift grows measurably over a long stream. Without half 1, an
         implementation that reorthonormalized on every sample -- destroying the O(mr)
         cost that is the entire point -- would pass this trivially.

    Run at both precisions. The shape is identical and the scale is set by eps, which is
    the actual content of the claim: this is a property of floating point, not of a bug.
    """

    REPEATS = 15          # 22 200 rank-one updates (15 passes, minus the 300 warm-up
                          # samples of the first pass, which are scored but not folded in)

    def _drift(self, binary: pathlib.Path, tol: float, reorth: bool):
        opts = CDefaults(lam=1.0, reorth=reorth, reorth_tol=tol,
                         check_every=CHECK_EVERY, repeat=self.REPEATS)
        return run_c_tracker(DATA / "normal.csv", opts, binary=binary)

    def _check(self, binary: pathlib.Path, tol: float, label: str) -> None:
        with_repair = self._drift(binary, tol, reorth=True)
        without = self._drift(binary, tol, reorth=False)

        # 1. The monitor does its job. It can only act every CHECK_EVERY samples, so the
        #    bound is a small multiple of the threshold, not the threshold itself.
        self.assertLessEqual(
            with_repair.max_drift, 3.0 * tol,
            f"{label}: drift reached {with_repair.max_drift:.3e} despite repair "
            f"(threshold {tol:.3e})")
        self.assertGreater(with_repair.summary["n_reorth"], "0",
                           f"{label}: the monitor never fired, so half 1 is vacuous")

        # 2. Without repair it grows -- both against where it started and against the
        #    repaired run. The statistic is the MEDIAN over the last tenth of the trace,
        #    not the single final value: drift is a random walk with a sawtooth on top,
        #    and one endpoint of a random walk is a noisy estimator that would make this
        #    assertion flap.
        without_late = float(np.median(without.drift[-without.drift.size // 10:]))
        with_late = float(np.median(with_repair.drift[-with_repair.drift.size // 10:]))
        initial = float(without.drift[0])
        self.assertGreater(
            without_late, 5.0 * initial,
            f"{label}: drift did not grow without repair "
            f"({initial:.3e} -> {without_late:.3e})")
        self.assertGreater(
            without_late, 3.0 * with_late,
            f"{label}: repair made no measurable difference "
            f"({with_late:.3e} vs {without_late:.3e})")

        # 3. And it accumulates rather than jitters: the second half of the stream is
        #    worse than the first.
        trace = without.drift
        half = trace.size // 2
        self.assertGreater(float(np.median(trace[half:])),
                           float(np.median(trace[:half])),
                           f"{label}: drift is not accumulating")

    def test_c_double(self):
        require_c(self)
        require_data(self, "normal.csv")
        self._check(TRACKER, REORTH_TOL, "C double")

    def test_c_float32(self):
        require_c(self)
        require_data(self, "normal.csv")
        if not TRACKER32.exists():
            self.skipTest("build/tracker32 missing; run `make host`")
        self._check(TRACKER32, FLOAT32_TOL, "C float32")

    def test_numpy_oracle(self):
        # Fewer repeats than the C runs: the numpy tracker is a Python loop, and the
        # separation is already an order of magnitude at 15 000 updates.
        require_data(self, "normal.csv")
        x = read_stream(DATA / "normal.csv")
        with_repair = run_stream(x, lam=1.0, reorth=True, reorth_tol=REORTH_TOL,
                                 repeat=10)
        without = run_stream(x, lam=1.0, reorth=False, repeat=10)
        late = slice(-without.drift.size // 10, None)
        self.assertLessEqual(with_repair.state.max_drift, 3.0 * REORTH_TOL)
        self.assertGreater(with_repair.state.n_reorth, 0)
        self.assertGreater(float(np.median(without.drift[late])),
                           5.0 * float(without.drift[0]))
        self.assertGreater(float(np.median(without.drift[late])),
                           3.0 * float(np.median(with_repair.drift[late])))


# === THE TEST THAT MATTERS ===
# Fails if the mathematics is wrong, not merely if the code crashed.
class TestAgreesWithBatchSVD(unittest.TestCase):
    """The incremental subspace against the batch full SVD, compared basis-independently.

    Brand's update is claimed to be EXACT, not approximate: with lambda = 1 and no
    truncation loss, streaming n samples through it must land on the same subspace a
    full SVD of all n samples finds. On a stationary rank-4 stream that is testable to
    high precision, and it is the claim that separates an incremental SVD from a
    heuristic that happens to track something.

    The comparison is basis-independent, and that is not a detail. U is defined only up
    to an r x r rotation within each singular subspace and a sign per column, so an
    elementwise comparison of two correct U's fails routinely -- see
    `test_elementwise_comparison_of_bases_is_meaningless` below, which demonstrates a
    subspace distance of 1e-16 alongside an elementwise difference of 0.84. Anything
    asserted here is asserted about the SUBSPACE (principal angles, or the distance
    between the projectors U U^T), never about the factors.
    """

    def setUp(self):
        require_data(self, "normal.csv")
        self.x = read_stream(DATA / "normal.csv")
        u, s, _ = np.linalg.svd(self.x, full_matrices=False)
        self.u_batch, self.s_batch = u, s

    def test_incremental_subspace_matches_batch_full_svd(self):
        res = run_stream(self.x, lam=1.0, reorth=True, reorth_tol=REORTH_TOL)
        r = res.r
        self.assertEqual(r, 4, "the rank criterion should recover the generator's rank")
        angles = np.degrees(principal_angles(res.state.u, self.u_batch[:, :r]))
        self.assertLess(angles.max(), 0.01,
                        f"largest principal angle {angles.max():.4f} deg")
        self.assertLess(projection_distance(res.state.u, self.u_batch[:, :r]), 1e-4)

    def test_incremental_singular_values_match_batch(self):
        # The subspace can be right while the spectrum is wrong -- that is what happens
        # if the warm-up block is folded in twice, or if forgetting leaks in when
        # lambda = 1. Sigma is checked separately for exactly that reason.
        res = run_stream(self.x, lam=1.0, reorth=True, reorth_tol=REORTH_TOL)
        rel = np.abs(res.state.sigma - self.s_batch[:res.r]) / self.s_batch[:res.r]
        self.assertLess(float(rel.max()), 1e-6, f"relative sigma error {rel}")

    def test_c_subspace_matches_batch_full_svd(self):
        require_c(self)
        c = run_c_tracker(DATA / "normal.csv", CDefaults(lam=1.0, reorth=True,
                                                         reorth_tol=REORTH_TOL),
                          want_basis=True)
        angles = np.degrees(principal_angles(c.basis, self.u_batch[:, :c.rank]))
        self.assertEqual(c.rank, 4)
        self.assertLess(angles.max(), 0.01,
                        f"C largest principal angle {angles.max():.4f} deg")

    def test_elementwise_comparison_of_bases_is_meaningless(self):
        # This test exists to justify the choice of metric in the two above it. U and
        # U R span the same subspace for any orthogonal R, so a test that compares them
        # entry by entry is testing the arbitrary choice of basis, not the mathematics.
        rng = np.random.default_rng(0)
        u = np.linalg.qr(rng.normal(size=(24, 4)))[0]
        rot = np.linalg.qr(rng.normal(size=(4, 4)))[0]
        rotated = u @ rot
        self.assertGreater(float(np.abs(u - rotated).max()), 0.1)
        self.assertLess(projection_distance(u, rotated), 1e-12)
        # The angle floor is sqrt(eps), not eps: theta = arccos(1 - delta) ~ sqrt(2 delta),
        # so a cosine computed to 1e-16 gives an angle no better than 1e-8 rad. That is a
        # property of measuring a small angle through its cosine, and it is why the
        # projector distance -- which has no such square root -- is the primary metric.
        self.assertLess(float(np.degrees(principal_angles(u, rotated)).max()), 1e-4)


class TestReorthonormalizationPreservesTheFactorization(unittest.TestCase):
    """Re-orthonormalizing U alone is wrong; U Sigma V^T is a factorization.

    Replacing U by the Q of U = QR without moving Sigma changes what the state factors.
    The repair in src/reorth.c does the small SVD of R Sigma so that U Sigma is
    invariant, and that invariance is what is asserted here.
    """

    def test_u_sigma_is_invariant(self):
        rng = np.random.default_rng(3)
        u = np.linalg.qr(rng.normal(size=(24, 4)))[0]
        u = u + 1e-8 * rng.normal(size=u.shape)          # deliberately non-orthonormal
        sigma = np.array([10.0, 6.0, 3.0, 1.0])
        before = u * sigma[None, :]
        u_new, s_new = reorthonormalize(u, sigma)
        after = u_new * s_new[None, :]
        # The repair rotates V, which is not tracked, so U Sigma itself is invariant only
        # up to a right multiplication. The quantity that IS invariant is the second
        # moment U Sigma^2 U^T -- which is exactly the object the subspace is extracted
        # from, so it is also the right thing to assert.
        lhs, rhs = before @ before.T, after @ after.T
        self.assertLess(float(np.abs(lhs - rhs).max()) / float(np.abs(lhs).max()), 1e-10)
        self.assertLess(orthogonality_drift(u_new), 1e-14)
        self.assertGreater(orthogonality_drift(u), 1e-10)

    def test_repair_is_not_needed_on_an_already_orthonormal_basis(self):
        rng = np.random.default_rng(4)
        u = np.linalg.qr(rng.normal(size=(24, 4)))[0]
        sigma = np.array([10.0, 6.0, 3.0, 1.0])
        u_new, s_new = reorthonormalize(u, sigma)
        self.assertTrue(np.allclose(np.sort(s_new), np.sort(sigma), atol=1e-12))
        self.assertLess(projection_distance(u, u_new), 1e-12)


if __name__ == "__main__":
    unittest.main()
