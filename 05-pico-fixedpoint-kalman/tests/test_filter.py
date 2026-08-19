"""Naive versus Joseph, measured on the compiled binary. The heart of the project.

Every number here comes from running build/kfhost, not from a Python model of it.
"""

from __future__ import annotations

import math
import pathlib
import unittest

import numpy as np

from tests._support import BUILD, kfhost, need_binaries, need_trace, steps, TRACE

from reference import kalman_float as kfl
from reference import kfparams as kp
from reference.error_budget import (GAIN_BITS, naive_pd_threshold_bits,
                                    naive_symmetry_prediction, state_error_budget)

RAD2DEG = 180.0 / math.pi
COV_ULP = 2.0 ** -kp.FRAC["cov"]


class TestCovarianceUpdateForms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        need_binaries()
        need_trace()
        cls.frac = dict(kp.FRAC)
        cls.frac["gain1"] = kp.FRAC["gain"] - 4
        cls.threshold = naive_pd_threshold_bits(kp.DT, int(kp.DURATION / kp.DT),
                                                kp.FRAC["cov"])

    # === THE TEST THAT MATTERS ===
    # Fails if the mathematics is wrong, not merely if the code crashed.
    #
    # The claim: the textbook covariance update P+ = (I - K H) P- stops producing a
    # covariance when the Kalman gain is coarsely represented, and the Joseph form does
    # not -- for structural reasons, at the same precision, on the same trace, through
    # the same predict step and the same gain.
    #
    # Three separate failures are asserted for the naive form, because any one of them
    # alone could be an artefact:
    #   1. min eigenvalue of the symmetric part goes NEGATIVE. A covariance cannot.
    #   2. ||P - P^T|| becomes non-trivially non-zero -- thousands of ulps, not one.
    #   3. it happens at the gain precision the error budget predicted in advance,
    #      log2(S_max/R) - 1, which is a number derived from the model and not fitted.
    # And for Joseph, at the same precision: eigenvalues stay positive and the symmetry
    # residual is EXACTLY zero, bit for bit, because A P A^T is a congruence.
    def test_naive_loses_positive_definiteness_where_joseph_does_not(self):
        bits = int(math.floor(self.threshold))          # 12, predicted, not searched
        naive = kfhost("naive", gain_bits=bits)
        joseph = kfhost("joseph", gain_bits=bits)

        self.assertLess(float(naive["min_lambda"]), 0.0,
                        f"naive covariance stayed positive definite at Q1.{bits}; "
                        "the predicted failure did not happen")
        self.assertGreater(float(joseph["min_lambda"]), 0.0,
                           f"Joseph lost positive definiteness at Q1.{bits} -- the "
                           "structural guarantee is broken")

        naive_sym = float(naive["max_sym_resid"])
        self.assertGreater(naive_sym, 1000.0 * COV_ULP,
                           f"naive asymmetry was only {naive_sym / COV_ULP:.1f} ulps")
        self.assertEqual(float(joseph["max_sym_resid"]), 0.0,
                         "Joseph produced an asymmetric covariance; A P A^T computes "
                         "(0,1) and (1,0) with the same expression and cannot")
        self.assertEqual(int(joseph["first_asymmetric_step"]), -1)

        # The threshold is where it was predicted: one bit more and the naive form
        # survives. If this fails, the prediction was luck.
        survives = kfhost("naive", gain_bits=bits + 1)
        self.assertGreater(float(survives["min_lambda"]), 0.0,
                           f"naive also failed at Q1.{bits + 1}: the predicted "
                           f"threshold {self.threshold:.2f} bits is wrong")

    def test_joseph_symmetry_is_exact_at_every_precision(self):
        # Not "small". Zero. The two off-diagonal entries are the same instruction
        # sequence on operands that differ only by a transpose, so at any word length
        # they produce the same int32. Any non-zero here means the congruence structure
        # was broken by an implementation shortcut -- e.g. expanding a completed square.
        for bits in GAIN_BITS:
            with self.subTest(gain_bits=bits):
                out = kfhost("joseph", gain_bits=bits)
                self.assertEqual(float(out["max_sym_resid"]), 0.0)

    def test_naive_asymmetry_matches_the_predicted_relaxation(self):
        # The prediction is d_k = (1-K0)d_{k-1} + eta with eta first order in the
        # gain's half-ulp, run on the float64 P sequence. Checked at every precision
        # in the sweep, not at one point. The measured residual is an integer number
        # of covariance ulps, so it comes out as a staircase and individual steps can
        # be flat -- the assertion is on the ratios and on the overall growth, which is
        # what the mechanism actually claims.
        ratios, resid = [], []
        for bits in GAIN_BITS:
            measured = float(kfhost("naive", gain_bits=bits)["max_sym_resid"])
            predicted = naive_symmetry_prediction(kp.DT, int(kp.DURATION / kp.DT),
                                                  kp.FRAC["cov"], bits, bits - 4)
            resid.append(measured)
            ratios.append(predicted / max(measured, COV_ULP))
        for bits, ratio in zip(GAIN_BITS, ratios):
            self.assertTrue(0.5 <= ratio <= 20.0,
                            f"at Q1.{bits} the prediction was {ratio:.2f}x the "
                            f"measurement, outside the stated factor of 20")
        self.assertLess(float(np.median(ratios)), 4.0,
                        f"median prediction/measurement = {np.median(ratios):.2f}")
        for a, b in zip(resid, resid[1:]):
            # 0.9 rather than 1.0: the residual is an integer number of covariance
            # ulps and its maximum over the run lands on different steps at different
            # precisions, so the staircase has a couple of 0.03% dips in it.
            self.assertGreaterEqual(b, 0.9 * a,
                                    f"asymmetry not monotone in bits: {resid}")
        self.assertGreater(resid[-1], 100.0 * resid[0],
                           f"asymmetry barely moved over the sweep: {resid}")

    def test_both_forms_agree_when_the_gain_is_exact_enough(self):
        # At the nominal Q1.30 gain the two forms must be numerically indistinguishable
        # in the state estimate. If they differed here, the divergence at Q1.12 could
        # not be attributed to the gain precision.
        naive = kfhost("naive", gain_bits=30)
        joseph = kfhost("joseph", gain_bits=30)
        self.assertAlmostEqual(float(naive["final_angle"]),
                               float(joseph["final_angle"]), places=10)


class TestAgainstFloat64(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        need_binaries()
        need_trace()
        cls.tr = kfl.load_trace(TRACE)
        cls.ref = kfl.run(cls.tr["gyro_rate"], cls.tr["accel_angle"], dt=kp.DT,
                          joseph=True)
        out = BUILD / "test_joseph.csv"
        cls.summary = kfhost("joseph", out=out)
        cls.steps = steps(out)
        frac = dict(kp.FRAC)
        frac["gain1"] = kp.FRAC["gain"] - 4
        cls.budget = state_error_budget(kp.DT, len(cls.tr["t"]), frac)

    # === THE TEST THAT MATTERS (second anchor) ===
    # The fixed-point Joseph filter must track the float64 reference inside the budget.
    # The budget brackets the answer -- a lower estimate that assumes independent
    # roundings and a strict l1 upper bound -- and the measurement has to land between
    # them. Falling outside either end means the budget is wrong, which is the only
    # claim this repo actually makes.
    def test_fixed_point_joseph_tracks_float64_within_the_budget(self):
        d = (self.steps["angle"] - self.ref.angle) * RAD2DEG
        measured_max = float(np.abs(d).max())
        upper = self.budget["bound_angle_rad"] * RAD2DEG
        lower = self.budget["rms_angle_rad"] * RAD2DEG
        self.assertLess(measured_max, upper,
                        f"measured max {measured_max:.3e} deg exceeded the l1 bound "
                        f"{upper:.3e} deg -- the bound is not a bound")
        self.assertGreater(measured_max, lower,
                           "measured error fell below the independent-rounding "
                           "estimate, which would mean the roundings are cancelling "
                           "and the model is wrong in the other direction")

    def test_there_is_no_secular_drift(self):
        # The budget predicts a bounded floor rather than a ramp, on the grounds that
        # every correction term stays far above its format's deadband. Falsify it by
        # fitting a slope over the last 30 s: the drift accumulated over the full 60 s
        # must stay well under the error floor itself.
        t = self.steps["t"]
        d = (self.steps["angle"] - self.ref.angle) * RAD2DEG
        tail = t >= t[-1] - 30.0
        slope = float(np.polyfit(t[tail], d[tail], 1)[0])
        floor = float(np.sqrt(np.mean(d[t >= t[-1] - 10.0] ** 2)))
        self.assertLess(abs(slope) * kp.DURATION, floor,
                        f"slope {slope:.3e} deg/s over 60 s exceeds the error floor "
                        f"{floor:.3e} deg: that is a drift, not a floor")
        self.assertGreater(self.budget["deadband_margin_bias"], 100.0)

    def test_the_filter_actually_works(self):
        # A smoke test, and it belongs here: everything above compares fixed point to
        # float64, and both could be equally wrong. Against the true angle the filter
        # must beat the raw accelerometer it is fusing.
        err = self.steps["angle"] - self.tr["true_angle"]
        raw = self.tr["accel_angle"] - self.tr["true_angle"]
        self.assertLess(np.sqrt(np.mean(err ** 2)), 0.5 * np.sqrt(np.mean(raw ** 2)))

    def test_the_filter_is_consistent(self):
        # A Kalman filter claims to know its own error. Once settled (after the diffuse
        # transient), the measured RMS bias error should agree with sqrt(P11) -- if it
        # did not, the covariance would be decorative and the whole naive-versus-Joseph
        # question would be about a number nobody uses.
        settled = self.steps["t"] >= 10.0
        e = (self.steps["bias"] - self.tr["true_bias"])[settled]
        claimed = math.sqrt(self.steps["p11"][-1])
        self.assertLess(float(np.sqrt(np.mean(e ** 2))), 2.0 * claimed)
        self.assertGreater(float(np.sqrt(np.mean(e ** 2))), 0.3 * claimed)

    def test_no_saturation_on_the_nominal_run(self):
        # The formats in qformat.h claim to bound every quantity in the filter. If any
        # of those bounds is wrong, this is where it shows -- it is how the K1 range
        # error was found in the first place.
        self.assertEqual(int(self.summary["saturation_events"]), 0)


if __name__ == "__main__":
    unittest.main()
