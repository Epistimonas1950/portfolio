"""The detector: AUC on the anomaly it is built for, and on the case that breaks it.

The failing case is asserted as firmly as the working one. A repo that only tests the
scenario its method wins is not measuring the method, it is advertising it.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from tests.chelper import ROOT, require_c, require_data

from oracle.batch_svd import DATA, read_labels, read_stream, residual_scores
from oracle.chost import CDefaults, run_c_tracker
from oracle.incremental import CALIBRATION, WARMUP, run_stream
from oracle.roc import LAMBDA, _restrict, auc, roc_curve

_ = ROOT


def spike_auc(scores: np.ndarray, kind: list[str]) -> float:
    s, y = _restrict(scores, kind, "spike")
    return auc(s, y)


class TestAucMechanics(unittest.TestCase):
    """The AUC implementation itself, before it is used to judge anything."""

    def test_perfect_separation_is_one(self):
        scores = np.array([0.0, 0.1, 0.2, 5.0, 6.0])
        label = np.array([0, 0, 0, 1, 1])
        self.assertAlmostEqual(auc(scores, label), 1.0)

    def test_reversed_separation_is_zero(self):
        scores = np.array([5.0, 6.0, 0.0, 0.1, 0.2])
        label = np.array([0, 0, 1, 1, 1])
        self.assertAlmostEqual(auc(scores, label), 0.0)

    def test_a_constant_score_is_exactly_one_half(self):
        # The tie-handling test. With average ranks a detector that says the same thing
        # about everything must score exactly 0.5; assigning ties by array order instead
        # gives 1.0 or 0.0 depending on how the labels happen to be sorted.
        scores = np.ones(100)
        label = np.array([0] * 50 + [1] * 50)
        self.assertAlmostEqual(auc(scores, label), 0.5)
        self.assertAlmostEqual(auc(scores, label[::-1]), 0.5)

    def test_auc_is_undefined_with_one_class(self):
        self.assertTrue(math.isnan(auc(np.arange(5.0), np.zeros(5, dtype=int))))

    def test_matches_the_area_under_the_roc_curve(self):
        rng = np.random.default_rng(0)
        label = rng.integers(0, 2, size=500)
        scores = rng.normal(size=500) + 0.8 * label
        fpr, tpr = roc_curve(scores, label)          # np.trapz: numpy 1.26 has no trapezoid
        self.assertAlmostEqual(auc(scores, label), float(np.trapz(tpr, fpr)),
                               places=2)


class TestDetectionOnLabelledData(unittest.TestCase):
    """The headline number, and the case that ruins it."""

    def setUp(self):
        require_data(self, "anomalous.csv", "labels.csv", "multimode.csv",
                     "multimode_labels.csv", "manymode.csv", "manymode_labels.csv")

    def _run(self, stream: str, labels: str, **kwargs):
        x = read_stream(DATA / stream)
        _, kind = read_labels(DATA / labels)
        res = run_stream(x, lam=LAMBDA, reorth=True, **kwargs)
        return spike_auc(res.scores, kind), res

    def test_out_of_subspace_spikes_are_detected(self):
        value, res = self._run("anomalous.csv", "labels.csv")
        self.assertEqual(res.r, 4)
        self.assertGreater(value, 0.95, f"single-mode spike AUC {value:.4f}")

    def test_incremental_is_close_to_the_batch_ceiling(self):
        x = read_stream(DATA / "anomalous.csv")
        _, kind = read_labels(DATA / "labels.csv")
        u_batch, _, _ = np.linalg.svd(x[:, :WARMUP + CALIBRATION],
                                      full_matrices=False)
        ceiling = spike_auc(residual_scores(x, u_batch[:, :4]), kind)
        streamed = self._run("anomalous.csv", "labels.csv")[0]
        self.assertGreater(streamed, ceiling - 0.01,
                           f"streamed {streamed:.4f} vs batch ceiling {ceiling:.4f}")

    # === THE TEST THAT MATTERS (the limitation) ===
    # Fails if the failure case stops failing, which would mean the README's
    # "limitation I volunteer first" had quietly become untrue.
    #
    # BRIEF.md: "if the normal regime has several distinct operating modes, a single
    # subspace blurs them and the detector degrades." The anomalies in multimode.csv and
    # manymode.csv are IDENTICAL to those in anomalous.csv by construction -- same
    # indices, same directions, same amplitude, drawn from a stream-independent generator
    # -- so any AUC difference is attributable to the structure of the NORMAL class and
    # to nothing else.
    def test_multiple_operating_modes_degrade_the_detector(self):
        single = self._run("anomalous.csv", "labels.csv")[0]
        two_modes = self._run("multimode.csv", "multimode_labels.csv")[0]
        four_modes = self._run("manymode.csv", "manymode_labels.csv")[0]
        self.assertLess(two_modes, single - 0.25,
                        f"two modes: {two_modes:.4f} vs single mode {single:.4f}")
        self.assertLess(four_modes, single - 0.30,
                        f"four modes: {four_modes:.4f} vs single mode {single:.4f}")

    def test_the_degradation_is_a_rank_failure_and_the_union_rank_fixes_it(self):
        # The honest other half of the limitation, and the reason the README does not
        # simply say "multiple modes break subspace methods". Two overlapping rank-4
        # modes span a rank-6 union and four span a rank-13 one; forced to that rank the
        # detector comes back. What fails is AUTOMATIC rank selection -- a union-of-modes
        # spectrum has no cliff for the gap criterion to find and the energy criterion
        # undershoots -- not the subspace model.
        two = self._run("multimode.csv", "multimode_labels.csv",
                        r_max=6, energy=1.0)[0]
        four = self._run("manymode.csv", "manymode_labels.csv",
                         r_max=13, energy=1.0)[0]
        self.assertGreater(two, 0.95, f"two modes at the union rank: {two:.4f}")
        self.assertGreater(four, 0.95, f"four modes at the union rank: {four:.4f}")

    def test_neither_rank_criterion_finds_the_union_rank_of_four_modes(self):
        res = self._run("manymode.csv", "manymode_labels.csv")[1]
        self.assertLess(res.r_energy, 13, "energy criterion unexpectedly found 13")
        self.assertLess(res.r_gap, 3, "gap criterion unexpectedly did not collapse")


class TestThresholdCalibration(unittest.TestCase):
    def test_threshold_is_the_empirical_quantile_of_the_warm_up_scores(self):
        require_data(self, "normal.csv")
        x = read_stream(DATA / "normal.csv")
        res = run_stream(x, lam=LAMBDA, quantile=0.99)
        window = res.scores[WARMUP:WARMUP + CALIBRATION]
        self.assertAlmostEqual(res.threshold, float(np.quantile(window, 0.99)))

    def test_false_positive_rate_on_a_clean_stream_matches_the_design(self):
        # Calibrating at the 0.99 quantile of a known-normal window should give roughly
        # 1% flagged on the rest of a stream with no anomalies in it. "Roughly" is the
        # honest word: 300 calibration samples put a wide confidence interval on a 1%
        # tail, so the assertion is an order-of-magnitude one.
        require_data(self, "normal.csv")
        x = read_stream(DATA / "normal.csv")
        res = run_stream(x, lam=LAMBDA, quantile=0.99)
        rest = res.scores[WARMUP + CALIBRATION:]
        rate = float(np.mean(rest > res.threshold))
        self.assertLess(rate, 0.05, f"false positive rate {rate:.3f}")

    def test_c_and_numpy_agree_on_the_threshold(self):
        require_c(self)
        require_data(self, "anomalous.csv")
        x = read_stream(DATA / "anomalous.csv")
        res = run_stream(x, lam=LAMBDA)
        c = run_c_tracker(DATA / "anomalous.csv", CDefaults(lam=LAMBDA))
        self.assertLess(abs(c.threshold - res.threshold) / res.threshold, 1e-3)


if __name__ == "__main__":
    unittest.main()
