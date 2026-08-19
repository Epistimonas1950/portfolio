"""Exponential forgetting: does lambda mean what src/forget.h says it means?

Two separate claims, tested separately, because passing one while failing the other is
exactly the situation a lone smoke test would hide:

  - the ALGEBRA: lambda <-> effective window is a bijection, with the factor of two the
    Sigma-versus-Sigma^2 convention hides;
  - the BEHAVIOUR: on a subspace that turns 90 degrees over the stream, a tracker with a
    finite window follows it and lambda = 1 does not.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from tests.chelper import require_c, require_data

from oracle.batch_svd import DATA, read_stream
from oracle.chost import CDefaults, run_c_tracker
from oracle.generate_data import make_rotating
from oracle.incremental import principal_angles, run_stream


def lam(n_eff: float) -> float:
    return math.sqrt(1.0 - 1.0 / n_eff)


class TestWindowAlgebra(unittest.TestCase):
    def test_window_round_trip(self):
        for n_eff in (10.0, 100.0, 400.0, 5000.0):
            self.assertAlmostEqual(1.0 / (1.0 - lam(n_eff) ** 2), n_eff, places=6)

    def test_the_weights_really_sum_to_the_window(self):
        # N_eff = sum_k lambda^{2k}. Asserting the sum rather than the closed form is
        # what catches the Sigma-versus-Sigma^2 confusion: with weights lambda^k instead
        # the sum would be 1/(1 - lambda), which is 2 N_eff - 1, not N_eff.
        for n_eff in (50.0, 400.0):
            l = lam(n_eff)
            total = float(np.sum(l ** (2 * np.arange(200000))))
            self.assertAlmostEqual(total / n_eff, 1.0, places=4)

    def test_half_life(self):
        # lambda^{2k} = 1/2 at k = ln(1/2) / (2 ln lambda). N_eff = 400 -> 277 samples.
        l = lam(400.0)
        k = math.log(0.5) / (2.0 * math.log(l))
        self.assertAlmostEqual(k, 277.0, delta=1.0)
        self.assertAlmostEqual(l ** (2 * k), 0.5, places=9)


class TestTrackingARotatingSubspace(unittest.TestCase):
    """The behavioural claim, against a ground truth the generator can produce exactly."""

    def setUp(self):
        require_data(self, "rotating.csv")
        self.x = read_stream(DATA / "rotating.csv")
        self.truth = make_rotating().basis_final     # exact subspace at the last sample

    def _angle(self, u: np.ndarray) -> float:
        return float(np.degrees(principal_angles(u, self.truth[:, :u.shape[1]]).max()))

    def test_forgetting_tracks_where_lambda_one_does_not(self):
        never = run_stream(self.x, lam=1.0, reorth=True)
        window = run_stream(self.x, lam=lam(200.0), reorth=True)
        angle_never = self._angle(never.state.u)
        angle_window = self._angle(window.state.u)
        self.assertGreater(angle_never, 25.0,
                           f"lambda = 1 should be badly stale, got {angle_never:.2f} deg")
        self.assertLess(angle_window, 10.0,
                        f"N_eff = 200 should track, got {angle_window:.2f} deg")
        self.assertLess(angle_window, angle_never / 3.0)

    def test_shorter_windows_track_more_closely(self):
        angles = [self._angle(run_stream(self.x, lam=lam(n), reorth=True).state.u)
                  for n in (2000.0, 1000.0, 400.0, 200.0, 100.0)]
        for lo, hi in zip(angles, angles[1:]):
            self.assertLess(hi, lo, f"angle did not improve monotonically: {angles}")

    def test_the_tradeoff_is_real_the_residual_has_an_interior_minimum(self):
        # Shorter is not simply better: the subspace is estimated from fewer effective
        # samples, so it gets noisier. The detector's own quantity -- mean residual over
        # the tail -- therefore bottoms out at an interior window and rises again. If
        # this ever stopped being true, "choose lambda from the desired window" would be
        # a free lunch, and it is not.
        from oracle.batch_svd import residual_scores
        resid = []
        for n in (400.0, 200.0, 100.0, 50.0, 25.0):
            res = run_stream(self.x, lam=lam(n), reorth=True)
            resid.append(float(np.mean(residual_scores(self.x[:, -200:], res.state.u))))
        best = int(np.argmin(resid))
        self.assertGreater(best, 0, f"no interior minimum: {resid}")
        self.assertLess(best, len(resid) - 1, f"no interior minimum: {resid}")

    def test_c_matches_the_oracle_on_the_rotating_stream(self):
        require_c(self)
        for n_eff in (None, 200.0):
            l = 1.0 if n_eff is None else lam(n_eff)
            c = run_c_tracker(DATA / "rotating.csv", CDefaults(lam=l), want_basis=True)
            npres = run_stream(self.x, lam=l, reorth=True)
            self.assertAlmostEqual(self._angle(c.basis), self._angle(npres.state.u),
                                   places=2)


if __name__ == "__main__":
    unittest.main()
