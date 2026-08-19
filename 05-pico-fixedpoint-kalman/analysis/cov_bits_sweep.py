#!/usr/bin/env python3
"""How many fractional bits does the covariance need, and what does one global format cost?

The covariance is where the dynamic range lives: P00 has to hold the diffuse prior
1.0 rad^2 at boot and still resolve the per-step process noise Qd(1,1) = sigma_b^2 dt =
4.5e-8 rad^2/s^2 sixty seconds later. That is 24.4 bits of range before any precision.
reference/error_budget.py predicts the cliff:

    2^-n <= min(Q00, Q11) / 2    =>    n >= 25.4 fractional bits

Below it the process noise rounds to zero every step, the filter stops admitting that
it can be wrong, P collapses and the estimate freezes. This failure has nothing to do
with the update form: BOTH the naive and the Joseph filter go over the same cliff at
the same place, and that is worth measuring precisely because it is easy to blame the
wrong thing. Joseph fixes the gain-precision failure. It does not fix a covariance
format that cannot hold the process noise.

The sweep also runs the "one global format" build, where every quantity is forced into
the Q4.27 that the gyro's +-500 deg/s range demands. That is the beginner's mistake the
brief names, and this is its price in numbers.

Writes results/cov_bits_sweep.csv.
"""

from __future__ import annotations

import csv
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from analysis._hostrun import RESULTS, require_binary, require_trace, run   # noqa: E402
from reference import kfparams as kp                                        # noqa: E402
from reference.error_budget import COV_BITS, cov_bits_threshold             # noqa: E402

RAD2DEG = 180.0 / math.pi


def main() -> None:
    require_binary()
    require_trace()
    RESULTS.mkdir(exist_ok=True)
    threshold = cov_bits_threshold()
    q00, _, q11 = kp.process_noise(kp.DT)

    rows = []
    for bits in list(COV_BITS) + ["global"]:
        entry: dict[str, object] = {}
        if bits == "global":
            entry["config"] = f"one global format Q4.{kp.GLOBAL_FRAC}"
            entry["cov_bits"] = kp.GLOBAL_FRAC
            flags = {"global_format": kp.GLOBAL_FRAC}
        else:
            entry["config"] = f"Q1.{bits}"
            entry["cov_bits"] = bits
            flags = {"cov_bits": bits}
        entry["q11_in_ulps"] = "%.4g" % (q11 * 2.0 ** entry["cov_bits"])
        entry["predicted_ok"] = "yes" if entry["cov_bits"] >= threshold else "no"
        for variant in ("naive", "joseph"):
            summary, _ = run(variant, **flags)
            entry[f"{variant}_min_lambda"] = summary["min_lambda"]
            entry[f"{variant}_final_p00"] = summary["final_p00"]
            entry[f"{variant}_rms_vs_truth_deg"] = "%.6g" % (
                float(summary["rms_angle_error"]) * RAD2DEG)
            entry[f"{variant}_max_err_vs_truth_deg"] = "%.6g" % (
                float(summary["max_abs_angle_error"]) * RAD2DEG)
            entry[f"{variant}_first_negative_step"] = summary["first_negative_step"]
            entry[f"{variant}_saturation_events"] = summary["saturation_events"]
        rows.append(entry)

    out = RESULTS / "cov_bits_sweep.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"predicted threshold: {threshold:.2f} covariance fractional bits "
          f"(Q00={q00:.3g}, Q11={q11:.3g} rad^2)")
    print(f"{'config':>24} {'Q11/ulp':>9} {'pred':>5} {'naive lam_min':>14} "
          f"{'joseph lam_min':>15} {'naive rms(deg)':>15} {'joseph rms(deg)':>16}")
    for r in rows:
        print(f"{r['config']:>24} {float(r['q11_in_ulps']):>9.3g} "
              f"{r['predicted_ok']:>5} {float(r['naive_min_lambda']):>14.3e} "
              f"{float(r['joseph_min_lambda']):>15.3e} "
              f"{float(r['naive_rms_vs_truth_deg']):>15.4g} "
              f"{float(r['joseph_rms_vs_truth_deg']):>16.4g}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
