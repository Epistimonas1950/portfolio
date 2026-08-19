#!/usr/bin/env python3
"""The discretisation step-size study: there is an optimal dt, and it is a numerical one.

F = I + A dt is EXACT for this model (A is nilpotent), so unlike a generic EKF there is
no linearisation error in the transition to trade off. The trade-off is still there, and
it is worth being precise about where it comes from, because "small dt is more accurate"
is the intuition this study exists to break:

  dt too large   the gyro is held constant across the interval, so each predict carries
                 (dt^2/2)|omega_dot| of unmodelled motion. Grows as dt^2, and it is the
                 same error in float64 -- it is a discretisation error, not a rounding
                 one, and the float64 curve shows it too.
  dt too small   more steps per second, each injecting its roundings; and Qd is
                 proportional to dt, so past a certain rate the process noise falls
                 below half a covariance ulp, rounds to zero, and the filter's
                 covariance collapses. reference/error_budget.py puts that cliff at
                 sigma^2 * 2^n = 9664 Hz for the nominal Q1.30 and 1208 Hz for the one
                 global format build -- a sample-rate ceiling set by a number format,
                 which is as concrete as this project's thesis gets.

The optimum is therefore visible only in fixed point. The float64 curve keeps improving
(or flattens) exactly where the fixed-point curve turns up, and that contrast is the
result. Both are measured here against the TRUE angle, because the discretisation error
cancels in a fixed-vs-float comparison.

Writes results/dt_study.csv.
"""

from __future__ import annotations

import csv
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from analysis._hostrun import BUILD, RESULTS, require_binary, run          # noqa: E402
from reference import kalman_float as kfl                                  # noqa: E402
from reference import kfparams as kp                                       # noqa: E402
from reference.error_budget import (COARSE_COV_BITS, DT_HZ,               # noqa: E402
                                    DT_STUDY_DURATION)
from reference.generate_trace import generate, write_csv                   # noqa: E402

RAD2DEG = 180.0 / math.pi


def main() -> None:
    require_binary()
    RESULTS.mkdir(exist_ok=True)
    (BUILD / "traces").mkdir(parents=True, exist_ok=True)

    q_slope = min(kp.SIGMA_GYRO, kp.SIGMA_BIAS) ** 2
    # Qd = sigma^2 dt rounds to zero once it drops below half an ulp, so each format
    # has a sample-rate ceiling fs = 2 sigma^2 2^n. Predicted before the sweep runs.
    configs = (("nominal", {}, kp.FRAC["cov"]),
               ("global", {"global_format": kp.GLOBAL_FRAC}, kp.GLOBAL_FRAC),
               ("coarse", {"cov_bits": COARSE_COV_BITS}, COARSE_COV_BITS))
    ceilings = {name: 2.0 * q_slope * 2.0 ** bits for name, _, bits in configs}

    rows = []
    for hz in DT_HZ:
        dt = 1.0 / hz
        # Same seed and same closed-form motion at every rate, so the only thing that
        # changes between rows is the sampling -- not the trajectory.
        trace = generate(dt=dt, duration=DT_STUDY_DURATION, seed=kp.SEED)
        path = BUILD / "traces" / f"imu_{hz:g}hz.csv"
        write_csv(path, trace)

        ref = kfl.run(trace["gyro_rate"], trace["accel_angle"], dt=dt, joseph=True)
        ref_err = ref.angle - trace["true_angle"]

        entry = {
            "sample_rate_hz": "%g" % hz,
            "dt_s": "%.6g" % dt,
            "n_samples": len(trace["t"]),
            "float64_rms_vs_truth_deg": "%.6g" % (
                float(np.sqrt(np.mean(ref_err ** 2))) * RAD2DEG),
        }
        for name in ceilings:
            entry[f"predicted_qd_zero_{name}"] = "yes" if hz > ceilings[name] else "no"
        for label, flags, _ in configs:
            summary, steps = run("joseph", trace=path,
                                 out=BUILD / f"dt_{hz:g}_{label}.csv", **flags)
            d = steps["angle"] - ref.angle
            entry[f"fixed_{label}_rms_vs_truth_deg"] = "%.6g" % (
                float(summary["rms_angle_error"]) * RAD2DEG)
            entry[f"fixed_{label}_rms_vs_float64_deg"] = "%.6g" % (
                float(np.sqrt(np.mean(d ** 2))) * RAD2DEG)
            entry[f"fixed_{label}_min_lambda"] = summary["min_lambda"]
            # The quantity that actually answers "is a higher rate still buying me
            # anything": the fixed-point error divided by what float64 achieves at the
            # same rate. 1.0 means the arithmetic is free; it stops being free long
            # before the covariance visibly collapses.
            entry[f"fixed_{label}_excess_over_float64"] = "%.4g" % (
                float(entry[f"fixed_{label}_rms_vs_truth_deg"])
                / float(entry["float64_rms_vs_truth_deg"]))
        rows.append(entry)

    out = RESULTS / "dt_study.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def best(key):
        vals = [float(r[key]) for r in rows]
        return rows[int(np.argmin(vals))]["sample_rate_hz"], min(vals)

    print("predicted Qd-rounds-to-zero ceilings: " + ", ".join(
        f"{n} {v:.0f} Hz" for n, v in ceilings.items()))
    print(f"{'fs(Hz)':>8} {'float64':>9} {'Q1.30':>9} {'Q4.27glob':>10} {'Q1.24':>9} "
          f"{'Q1.24 lam_min':>14}")
    for r in rows:
        print(f"{r['sample_rate_hz']:>8} {float(r['float64_rms_vs_truth_deg']):>9.4g} "
              f"{float(r['fixed_nominal_rms_vs_truth_deg']):>9.4g} "
              f"{float(r['fixed_global_rms_vs_truth_deg']):>10.4g} "
              f"{float(r['fixed_coarse_rms_vs_truth_deg']):>9.4g} "
              f"{float(r['fixed_coarse_min_lambda']):>14.3e}")
    for key, name in (("float64_rms_vs_truth_deg", "float64"),
                      ("fixed_nominal_rms_vs_truth_deg", "fixed, nominal Q1.30"),
                      ("fixed_global_rms_vs_truth_deg", "fixed, one global Q4.27"),
                      ("fixed_coarse_rms_vs_truth_deg", "fixed, coarse Q1.24")):
        hz, val = best(key)
        print(f"  best {name:<26} {hz:>7} Hz  ({val:.4g} deg RMS)")
    for label, _, bits in configs:
        neg = [r["sample_rate_hz"] for r in rows
               if float(r[f"fixed_{label}_min_lambda"]) < 0]
        usable = [r["sample_rate_hz"] for r in rows
                  if float(r[f"fixed_{label}_excess_over_float64"]) <= 1.05]
        print(f"  {label:<8} Q.{bits}: predicted Qd-zero ceiling "
              f"{ceilings[label]:>7.0f} Hz | first indefinite P at "
              f"{(neg[0] + ' Hz') if neg else 'never in sweep':>14} | "
              f"still within 5% of float64 up to {usable[-1] if usable else 'n/a'} Hz")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
