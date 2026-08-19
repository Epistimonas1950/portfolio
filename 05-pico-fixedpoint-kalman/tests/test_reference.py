"""The float64 reference, the trace generator, and the C/Python constant agreement.

Nothing here touches fixed point. These are the checks that earn the right to call
reference/kalman_float.py "ground truth" -- if the reference is wrong, or if it is
solving a different problem from the C, then every fixed-versus-float number in
results/ is measuring a modelling mismatch and calling it arithmetic.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from tests._support import KFHOST, TRACE, need_binaries, need_trace

import subprocess

from reference import generate_trace as gt
from reference import kalman_float as kfl
from reference import kfparams as kp


class TestConstantsAgree(unittest.TestCase):
    # The C macros in firmware/kalman_fixed.h and the Python constants in
    # reference/kfparams.py are written out twice, because this repo has to build with
    # bare gcc and run with bare python3 and no generator step between them. That is a
    # real risk and this is the mitigation: ask the binary what it was compiled with.
    def test_c_and_python_filter_parameters_are_identical(self):
        need_binaries()
        out = subprocess.run([str(KFHOST), "--params"], capture_output=True, text=True,
                             check=True).stdout
        got = dict(line.split("=", 1) for line in out.splitlines())
        want = {"dt": kp.DT, "sigma_gyro": kp.SIGMA_GYRO, "sigma_bias": kp.SIGMA_BIAS,
                "sigma_acc": kp.SIGMA_ACC, "p0_angle": kp.P0_ANGLE,
                "p0_bias": kp.P0_BIAS}
        for key, val in want.items():
            self.assertEqual(float(got[key]), val, f"C and Python disagree on {key}")

    def test_c_and_python_q_formats_are_identical(self):
        need_binaries()
        out = subprocess.run([str(KFHOST), "--params"], capture_output=True, text=True,
                             check=True).stdout
        got = dict(line.split("=", 1) for line in out.splitlines())
        for name, frac in kp.FRAC.items():
            self.assertEqual(int(got[f"q_{name}_frac"]), frac, name)
        self.assertEqual(int(got["q_gain1_frac"]), kp.FRAC["gain"] - 4)
        self.assertEqual(int(got["q_global_frac"]), kp.GLOBAL_FRAC)


class TestDiscretisation(unittest.TestCase):
    def test_transition_matrix_is_exact_not_first_order(self):
        # A = [[0,-1],[0,0]] is nilpotent, so exp(A dt) = I + A dt terminates. The whole
        # dt study rests on this: any dt dependence found there is a sampling or
        # rounding effect, NOT a truncated matrix exponential.
        for dt in (1e-4, 5e-3, 0.1, 1.0):
            a = np.array([[0.0, -1.0], [0.0, 0.0]])
            series = np.eye(2)
            term = np.eye(2)
            for k in range(1, 12):
                term = term @ (a * dt) / k
                series = series + term
            self.assertTrue(np.allclose(series, np.eye(2) + a * dt, atol=1e-15), dt)

    def test_process_noise_matches_the_van_loan_integral(self):
        # Qd = int_0^dt exp(As) Qc exp(As)^T ds, checked by quadrature rather than by
        # re-deriving the same closed form twice.
        dt = kp.DT
        q00, q01, q11 = kp.process_noise(dt)
        a = np.array([[0.0, -1.0], [0.0, 0.0]])
        qc = np.diag([kp.SIGMA_GYRO ** 2, kp.SIGMA_BIAS ** 2])
        s = np.linspace(0.0, dt, 20001)
        integrand = np.array([(np.eye(2) + a * si) @ qc @ (np.eye(2) + a * si).T
                              for si in s])
        num = np.trapz(integrand, s, axis=0)
        self.assertAlmostEqual(num[0, 0] / q00, 1.0, places=8)
        self.assertAlmostEqual(num[0, 1] / q01, 1.0, places=6)
        self.assertAlmostEqual(num[1, 1] / q11, 1.0, places=8)


class TestFloatReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        need_trace()
        cls.tr = kfl.load_trace(TRACE)

    def test_naive_and_joseph_agree_in_float64(self):
        # In 53-bit arithmetic the two covariance forms are the same algebra and must
        # agree to rounding. If they did not, the fixed-point divergence measured in
        # results/ would not be a fixed-point effect at all.
        j = kfl.run(self.tr["gyro_rate"], self.tr["accel_angle"], dt=kp.DT, joseph=True)
        n = kfl.run(self.tr["gyro_rate"], self.tr["accel_angle"], dt=kp.DT, joseph=False)
        self.assertLess(np.abs(j.angle - n.angle).max(), 1e-12)
        self.assertLess(np.abs(j.p - n.p).max(), 1e-14)

    def test_reference_covariance_stays_positive_definite(self):
        j = kfl.run(self.tr["gyro_rate"], self.tr["accel_angle"], dt=kp.DT, joseph=True)
        lam = np.array([np.linalg.eigvalsh(j.p[i]).min() for i in range(len(j.t))])
        self.assertGreater(lam.min(), 0.0)
        # And comfortably above one covariance ulp, or the format could not hold it and
        # the comparison would be measuring the reference's own conditioning.
        self.assertGreater(lam.min(), 1000.0 * 2.0 ** -kp.FRAC["cov"])

    def test_gain_sequence_does_not_depend_on_the_measurements(self):
        # The error budget predicts from K_k without ever seeing data. That is only
        # legitimate because the Riccati recursion is measurement-independent.
        rng = np.random.default_rng(7)
        other = self.tr["accel_angle"] + rng.normal(0.0, 1.0, len(self.tr["t"]))
        a = kfl.run(self.tr["gyro_rate"], self.tr["accel_angle"], dt=kp.DT)
        b = kfl.run(self.tr["gyro_rate"], other, dt=kp.DT)
        self.assertTrue(np.allclose(a.k, b.k, atol=0.0, rtol=0.0))


class TestTraceGenerator(unittest.TestCase):
    def test_deterministic_for_a_seed(self):
        a = gt.generate(seed=123)
        b = gt.generate(seed=123)
        c = gt.generate(seed=124)
        for key in a:
            self.assertTrue(np.array_equal(a[key], b[key]), key)
        self.assertFalse(np.array_equal(a["gyro_rate"], c["gyro_rate"]))

    def test_true_rate_is_the_analytic_derivative(self):
        # generate_trace differentiates the motion in closed form rather than by
        # differencing the angle. A differenced "truth" would carry its own O(dt^2)
        # error straight into the quantity the dt study measures.
        t = np.linspace(0.0, 10.0, 100001)
        theta, omega = gt.true_motion(t)
        num = np.gradient(theta, t)
        # np.gradient is one-sided at the ends, so compare on the interior,
        # where it is a second-order central difference with error O(h^2) ~ 2e-7.
        self.assertLess(np.abs(num[2:-2] - omega[2:-2]).max(), 1e-6)

    def test_every_quantity_stays_inside_its_declared_format(self):
        tr = gt.generate()
        self.assertLess(np.abs(tr["true_angle"]).max(), 2.0 ** (31 - kp.FRAC["ang"]))
        self.assertLess(np.abs(tr["gyro_rate"]).max(), 2.0 ** (31 - kp.FRAC["rate"]))
        self.assertLess(np.abs(tr["true_bias"]).max(), 2.0 ** (31 - kp.FRAC["bias"]))
        # And inside the sensor's own full scale, or the trace would be unphysical.
        self.assertLess(np.abs(tr["gyro_rate"]).max(), 500.0 * math.pi / 180.0)

    def test_noise_densities_scale_with_the_sample_rate(self):
        # sigma_gyro is a density, so the per-sample deviation must go as 1/sqrt(dt).
        # Getting this wrong makes the filter look mistuned at one rate and fine at
        # another, and would silently invalidate the whole dt study.
        for dt in (0.01, 0.0025):
            tr = gt.generate(dt=dt, duration=200.0)
            residual = tr["gyro_rate"] - (np.gradient(tr["true_angle"], dt)
                                          + tr["true_bias"])
            self.assertAlmostEqual(np.std(residual) * math.sqrt(dt) / kp.SIGMA_GYRO,
                                   1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
