"""The step-size trade-off, and the sample-rate ceiling a number format imposes.

The claim under test is the one the brief calls out: there is an optimal dt, and in
this filter it is a numerical quantity rather than a modelling one. F = I + A dt is
exact (tests/test_reference.py checks that), so shrinking dt cannot break a
linearisation -- what it breaks is the covariance format, because Qd is proportional to
dt and eventually rounds to zero.

reference/error_budget.py predicts that ceiling in closed form:

    Qd = sigma^2 dt  rounds to 0  when  sigma^2 dt < 2^-(n+1)
    =>  fs_max = 2 sigma^2 2^n

which is 19.3 kHz for the nominal Q1.30 covariance and 302 Hz for a Q1.24 one. These
tests run the compiled filter either side of that prediction.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from tests._support import BUILD, kfhost, need_binaries, steps

from reference import generate_trace as gt
from reference import kalman_float as kfl
from reference import kfparams as kp

DURATION = 8.0          # s -- long enough to settle at every rate, cheap at 6.4 kHz
COARSE_COV = 24
RATES = (100.0, 400.0, 1600.0, 6400.0)


def ceiling_hz(cov_bits: int) -> float:
    return 2.0 * min(kp.SIGMA_GYRO, kp.SIGMA_BIAS) ** 2 * 2.0 ** cov_bits


class TestSampleRateCeiling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        need_binaries()
        (BUILD / "traces").mkdir(parents=True, exist_ok=True)
        cls.traces, cls.refs = {}, {}
        for hz in RATES:
            tr = gt.generate(dt=1.0 / hz, duration=DURATION, seed=kp.SEED)
            path = BUILD / "traces" / f"test_{hz:g}hz.csv"
            gt.write_csv(path, tr)
            cls.traces[hz] = (path, tr)
            ref = kfl.run(tr["gyro_rate"], tr["accel_angle"], dt=1.0 / hz, joseph=True)
            cls.refs[hz] = float(np.sqrt(np.mean((ref.angle - tr["true_angle"]) ** 2)))

    # === THE TEST THAT MATTERS (third anchor) ===
    # Fails if the predicted sample-rate ceiling is wrong.
    #
    # The prediction is a pure format calculation: fs_max = 2 sigma^2 2^n, with no
    # reference to any measurement. Below it the covariance must stay positive definite
    # at that word length; above it, it must not. Both directions are asserted, because
    # a prediction that only ever says "it will break" is not a prediction.
    def test_covariance_collapses_exactly_where_the_format_says_it_will(self):
        ceiling = ceiling_hz(COARSE_COV)
        self.assertTrue(RATES[0] < ceiling < RATES[-1],
                        "the test rates no longer bracket the predicted ceiling")
        below = [hz for hz in RATES if hz < ceiling]
        above = [hz for hz in RATES if hz > 2.0 * ceiling]
        for hz in below:
            out = kfhost("joseph", trace=self.traces[hz][0], cov_bits=COARSE_COV)
            self.assertGreater(float(out["min_lambda"]), 0.0,
                               f"Q1.{COARSE_COV} covariance already indefinite at "
                               f"{hz:g} Hz, below the predicted {ceiling:.0f} Hz")
        for hz in above:
            out = kfhost("joseph", trace=self.traces[hz][0], cov_bits=COARSE_COV)
            self.assertLess(float(out["min_lambda"]), 0.0,
                            f"Q1.{COARSE_COV} covariance survived {hz:g} Hz, well "
                            f"above the predicted ceiling {ceiling:.0f} Hz")

    def test_the_nominal_format_survives_the_whole_range(self):
        # Same rates, same trace, 6 more fractional bits: the ceiling moves to 19 kHz
        # and nothing in the sweep touches it. This is what the extra bits bought.
        self.assertGreater(ceiling_hz(kp.FRAC["cov"]), RATES[-1])
        for hz in RATES:
            out = kfhost("joseph", trace=self.traces[hz][0])
            self.assertGreater(float(out["min_lambda"]), 0.0, f"{hz:g} Hz")
            self.assertEqual(int(out["saturation_events"]), 0, f"{hz:g} Hz")


class TestStepSizeTradeoff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TestSampleRateCeiling.setUpClass()
        cls.traces = TestSampleRateCeiling.traces
        cls.refs = TestSampleRateCeiling.refs

    def test_float64_always_improves_with_more_samples(self):
        # The control. In exact arithmetic more measurements per second is never worse,
        # so any turning point in the fixed-point curve is a numerical effect and not a
        # property of the estimation problem.
        errs = [self.refs[hz] for hz in RATES]
        for a, b in zip(errs, errs[1:]):
            self.assertLess(b, a, f"float64 error not monotone in rate: {errs}")

    def test_fixed_point_stops_following_float64_above_the_ceiling(self):
        # Below the ceiling the Q1.24 build is indistinguishable from float64; above
        # it, the excess grows. That divergence IS the optimum -- past this rate you
        # are paying for samples that the covariance word length cannot use.
        excess = []
        for hz in RATES:
            out = kfhost("joseph", trace=self.traces[hz][0], out=BUILD / "dt_t.csv",
                         cov_bits=COARSE_COV)
            rms = float(out["rms_angle_error"]) * 180.0 / math.pi
            excess.append(rms / (self.refs[hz] * 180.0 / math.pi))
        self.assertLess(excess[0], 1.05, f"excess at {RATES[0]:g} Hz: {excess[0]:.3f}")
        self.assertGreater(excess[-1], 1.2,
                           f"Q1.{COARSE_COV} still matched float64 at {RATES[-1]:g} Hz: "
                           f"{excess}")
        self.assertGreater(excess[-1], excess[0])

    def test_larger_dt_costs_accuracy_in_both_arithmetics(self):
        # The other branch of the trade-off, and the point is that it is NOT a
        # fixed-point effect: the zero-order hold on the gyro and the lower measurement
        # rate hurt float64 by the same factor, so a dt study that only looked at the
        # fixed-point curve would misattribute it.
        ratio_float = self.refs[RATES[0]] / self.refs[RATES[-1]]
        lo = float(kfhost("joseph", trace=self.traces[RATES[0]][0])["rms_angle_error"])
        hi = float(kfhost("joseph", trace=self.traces[RATES[-1]][0])["rms_angle_error"])
        self.assertGreater(ratio_float, 2.0)
        self.assertAlmostEqual(lo / hi, ratio_float, delta=0.15 * ratio_float)


if __name__ == "__main__":
    unittest.main()
