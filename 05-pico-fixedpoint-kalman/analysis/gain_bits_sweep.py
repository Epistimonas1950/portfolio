#!/usr/bin/env python3
"""How many bits does the Kalman gain need before the textbook update falls over?

This is the bit-depth study the brief calls "how many bits before it falls over", and
it is the error budget in its most direct form. The covariance stays at its nominal
Q1.30; only the gain's fractional bits are cut, K0 from Q1.30 downwards and K1 from
Q5.26 downwards in lockstep (they keep their four-bit range difference, so this is a
precision sweep and not a saturation artefact).

Two quantities are tracked, and reference/error_budget.py predicted both before this
script was ever run:

  max ||P - P^T||     naive: the relaxation d_k = (1-K0)d_{k-1} + eta has a floor.
                      Joseph: identically zero, because its two off-diagonal entries
                      are the same expression with P01 and P10 exchanged.
  min lambda(P)       naive: P+00 = P00 - K0 P00 is a cancellation and goes negative
                      once the gain has fewer than log2(S_max/R) - 1 fractional bits.
                      Joseph: P+00 = a(a P00) + K0(K0 R), a sum of non-negative
                      products, so there is no threshold.

Writes results/gain_bits_sweep.csv.
"""

from __future__ import annotations

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from analysis._hostrun import RESULTS, require_binary, require_trace, run   # noqa: E402
from reference.error_budget import GAIN_BITS                                # noqa: E402


def read_predictions() -> dict[int, float]:
    path = RESULTS / "error_budget.csv"
    if not path.exists():
        sys.exit(f"{path} missing -- run reference/error_budget.py first.")
    out = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row["prediction"] == "naive_max_sym_resid":
                out[int(row["config"].split("=")[1])] = float(row["value"])
    return out


def main() -> None:
    require_binary()
    require_trace()
    RESULTS.mkdir(exist_ok=True)
    predicted = read_predictions()

    rows = []
    for bits in GAIN_BITS:
        entry = {"gain_bits": bits, "gain0_format": f"Q1.{bits}",
                 "gain1_format": f"Q5.{bits - 4}",
                 "predicted_naive_sym_resid": "%.6g" % predicted[bits]}
        for variant in ("naive", "joseph"):
            summary, _ = run(variant, gain_bits=bits)
            entry[f"{variant}_max_sym_resid"] = summary["max_sym_resid"]
            entry[f"{variant}_min_lambda"] = summary["min_lambda"]
            entry[f"{variant}_first_negative_step"] = summary["first_negative_step"]
            entry[f"{variant}_rms_vs_truth_deg"] = "%.6g" % (
                float(summary["rms_angle_error"]) * 180.0 / 3.141592653589793)
            entry[f"{variant}_saturation_events"] = summary["saturation_events"]
        m = float(entry["naive_max_sym_resid"])
        entry["naive_sym_predicted_over_measured"] = (
            "%.4g" % (predicted[bits] / m) if m > 0 else "inf")
        rows.append(entry)

    out = RESULTS / "gain_bits_sweep.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'gain':>5} {'naive sym':>11} {'pred sym':>11} {'p/m':>7} "
          f"{'naive lam_min':>14} {'joseph sym':>11} {'joseph lam_min':>15}")
    for r in rows:
        print(f"{r['gain_bits']:>5} {float(r['naive_max_sym_resid']):>11.3e} "
              f"{float(r['predicted_naive_sym_resid']):>11.3e} "
              f"{r['naive_sym_predicted_over_measured']:>7} "
              f"{float(r['naive_min_lambda']):>14.3e} "
              f"{float(r['joseph_max_sym_resid']):>11.3e} "
              f"{float(r['joseph_min_lambda']):>15.3e}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
