"""Claims about the C are tested against the C, by running it.

Every assertion here shells out to build/tracker. A Python re-implementation that agreed
with itself would prove nothing about the program that would actually be flashed onto a
board, which is the only artifact in this repo that BRIEF.md cares about.
"""

from __future__ import annotations

import subprocess
import unittest

import numpy as np

from tests.chelper import require_c, require_data

from oracle.batch_svd import DATA, read_stream
from oracle.chost import (TRACKER, TRACKER32, CDefaults, run_c_selftest,
                          run_c_tracker)
from oracle.incremental import (CHECK_EVERY, REORTH_TOL, principal_angles,
                                projection_distance, run_stream)
from oracle.roc import LAMBDA


class TestCSelfTest(unittest.TestCase):
    """The C's own checks of its linear algebra: QR, Jacobi SVD, rank, quantile.

    These are identities that must hold exactly (A = QR, U^T U = I, singular values
    descending), checked inside the C at its own working precision, so the float32 build
    is held to a float32 standard rather than a double one.
    """

    def test_double_build_passes(self):
        require_c(self)
        code, out = run_c_selftest(TRACKER)
        self.assertEqual(code, 0, out)
        self.assertIn("failures=0", out)

    def test_float32_build_passes(self):
        require_c(self)
        if not TRACKER32.exists():
            self.skipTest("build/tracker32 missing; run `make host`")
        code, out = run_c_selftest(TRACKER32)
        self.assertEqual(code, 0, out)

    def test_jacobi_sorts_descending(self):
        # Called out separately because it is the one failure in the list that is
        # completely silent: an unsorted truncation in incsvd_update drops the dominant
        # direction instead of the weakest and nothing else changes.
        require_c(self)
        _, out = run_c_selftest(TRACKER)
        for line in out.splitlines():
            if "descending" in line:
                self.assertIn("PASS", line)


class TestCAgreesWithTheOracle(unittest.TestCase):
    """Scores and subspaces, on every stream in data/.

    The comparison is deliberately not bit-for-bit. The two implementations seed
    different generators (PCG32 in C, PCG64 in numpy), so their initial sketches are
    different draws of the same randomized algorithm; the tolerance below is set by that
    approximation, not by rounding.
    """

    STREAMS = ("normal.csv", "anomalous.csv", "multimode.csv", "rotating.csv")

    def _compare(self, name: str, binary, tol_score: float, tol_subspace: float,
                 tol_angle_deg: float):
        x = read_stream(DATA / name)
        opts = CDefaults(lam=LAMBDA, reorth=True, reorth_tol=REORTH_TOL,
                         check_every=CHECK_EVERY)
        if binary is TRACKER32:
            opts.reorth_tol = 100.0 * float(np.finfo(np.float32).eps)
        c = run_c_tracker(DATA / name, opts, binary=binary, want_basis=True)
        npres = run_stream(x, lam=LAMBDA, reorth=True, reorth_tol=REORTH_TOL,
                           check_every=CHECK_EVERY)
        self.assertEqual(c.rank, npres.r, f"{name}: rank disagreement")
        rel = float(np.max(np.abs(c.scores - npres.scores)) / np.max(npres.scores))
        self.assertLess(rel, tol_score, f"{name}: relative score difference {rel:.3e}")
        dist = projection_distance(npres.state.u, c.basis)
        self.assertLess(dist, tol_subspace,
                        f"{name}: subspace projection distance {dist:.3e}")
        # And the invariant form of the same statement, which is the one a reader can
        # interpret. The angle tolerance is separate from the projector one because the
        # angle is read through an arccos and so has a sqrt(eps) floor of its own.
        angle = float(np.degrees(principal_angles(npres.state.u, c.basis).max()))
        self.assertLess(angle, tol_angle_deg,
                        f"{name}: principal angle {angle:.4f} deg")

    def test_double_build_matches_numpy(self):
        require_c(self)
        require_data(self, *self.STREAMS)
        for name in self.STREAMS:
            with self.subTest(stream=name):
                self._compare(name, TRACKER, 1e-4, 1e-4, 0.01)

    def test_float32_build_matches_numpy(self):
        require_c(self)
        require_data(self, *self.STREAMS)
        if not TRACKER32.exists():
            self.skipTest("build/tracker32 missing; run `make host`")
        # Single precision is ~10x looser on the subspace, which is the measurement a
        # person choosing a build for a 512 MB board actually wants.
        for name in self.STREAMS:
            with self.subTest(stream=name):
                self._compare(name, TRACKER32, 1e-3, 1e-3, 0.5)

    def test_nearly_degenerate_spectra_are_where_they_disagree_most(self):
        # manymode.csv has sigma_2 = 62.8 and sigma_3 = 62.5. Nearly repeated singular
        # values mean the corresponding singular subspace is nearly two-dimensional and
        # the split between the two directions is ill-conditioned: the two draws land on
        # different bases for very nearly the same subspace. Documented as an expected
        # 10^-3 rather than left to be discovered as a mysterious outlier in
        # results/c_vs_python.csv.
        require_c(self)
        require_data(self, "manymode.csv")
        self._compare("manymode.csv", TRACKER, 5e-2, 5e-2, 0.5)


class TestCommandLineBehaviour(unittest.TestCase):
    def test_unknown_mode_exits_non_zero_with_usage(self):
        require_c(self)
        proc = subprocess.run([str(TRACKER), "nonsense"], capture_output=True,
                              text=True, check=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("usage", proc.stderr)

    def test_missing_input_fails_loudly(self):
        require_c(self)
        proc = subprocess.run([str(TRACKER), "track", "--input", "/nonexistent.csv",
                               "--output", "/tmp/out.csv"],
                              capture_output=True, text=True, check=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("cannot open", proc.stderr)

    def test_a_stream_shorter_than_the_warm_up_is_refused(self):
        require_c(self)
        require_data(self, "normal.csv")
        proc = subprocess.run([str(TRACKER), "track", "--input",
                               str(DATA / "normal.csv"), "--output", "/tmp/out.csv",
                               "--warmup", "99999"],
                              capture_output=True, text=True, check=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("fewer than", proc.stderr)

    def test_reported_rank_and_spectrum_are_self_consistent(self):
        require_c(self)
        require_data(self, "anomalous.csv")
        c = run_c_tracker(DATA / "anomalous.csv", CDefaults(lam=LAMBDA))
        spectrum = [float(v) for k, v in c.summary.items() if k.startswith("spectrum")]
        self.assertTrue(all(a >= b for a, b in zip(spectrum, spectrum[1:])),
                        "reported spectrum is not descending")
        energy = np.cumsum(np.array(spectrum) ** 2) / np.sum(np.array(spectrum) ** 2)
        self.assertEqual(int(np.searchsorted(energy, 0.95) + 1),
                         int(c.summary["rank_energy"]))


if __name__ == "__main__":
    unittest.main()
