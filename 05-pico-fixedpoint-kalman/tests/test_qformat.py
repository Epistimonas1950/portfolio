"""The fixed-point arithmetic contract, checked by running the compiled selftest.

host/qformat_selftest.c does the checking; this file runs it and insists that every
named check passed. Keeping the assertions in C means they are testing the same
inlined code the filter calls, including the compiler's treatment of the int64
intermediates -- which is the part that would actually differ on a 32-bit target.
"""

from __future__ import annotations

import subprocess
import unittest

from tests._support import QTEST, need_binaries


class TestQFormatSelfTest(unittest.TestCase):
    def setUp(self):
        need_binaries()
        proc = subprocess.run([str(QTEST)], capture_output=True, text=True)
        self.raw = proc.stdout
        self.rc = proc.returncode

    def test_every_check_passed(self):
        self.assertEqual(self.rc, 0, f"qtest reported failures:\n{self.raw}")
        failed = [line for line in self.raw.splitlines() if "=FAIL" in line]
        self.assertEqual(failed, [], "\n".join(failed))

    # === THE TEST THAT MATTERS (second anchor) ===
    # Fails if the arithmetic wraps instead of saturating.
    #
    # A wrapping int32 covariance entry is the worst failure mode available here: +1.9
    # rad^2 becomes -1.9 rad^2, still a perfectly ordinary integer, and every downstream
    # consumer believes it. Nothing detects it, the filter runs on with a negative
    # variance, and the resulting estimate is wrong in a way that looks like a modelling
    # problem forever after. Saturation is also wrong, but boundedly and detectably.
    def test_multiply_saturates_rather_than_wrapping(self):
        checks = {line.split("=")[0]: line for line in self.raw.splitlines()}
        for name in ("mul_saturates_positive", "mul_saturates_negative",
                     "add_saturates_high", "add_saturates_low", "sub_saturates_low",
                     "div_by_zero_contained"):
            self.assertIn(name, checks, f"{name} was not run at all")
            self.assertIn("=PASS", checks[name], checks[name])
        # 1.9 * 1.9 = 3.61 wraps to -0.39 in Q1.30 if the intermediate is int32.
        line = checks["mul_saturates_positive"]
        self.assertIn("2.000000", line, f"expected clamp to +2.0, got: {line}")

    def test_no_spurious_saturation_on_the_working_range(self):
        line = [ln for ln in self.raw.splitlines()
                if ln.startswith("no_spurious_saturation")][0]
        self.assertIn("=PASS", line, line)
        self.assertIn("events=0", line, line)

    def test_rounding_is_to_nearest_not_truncating(self):
        # Truncation toward -inf is a biased -ulp/2 per operation, and a bias inside a
        # recursion integrates into drift. The whole "no secular drift" prediction in
        # the error budget rests on this.
        line = [ln for ln in self.raw.splitlines()
                if ln.startswith("rounding_is_unbiased")][0]
        self.assertIn("=PASS", line, line)


if __name__ == "__main__":
    unittest.main()
