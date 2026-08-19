#!/usr/bin/env python3
"""Run the compiled fixed-point filter, compare it to float64, and score the budget.

This is the measuring end. reference/error_budget.py has already written its
predictions to results/error_budget.csv without running any C; this script runs the C,
computes the same quantities from the output, and puts them side by side. The number
the project is judged on is the last column: predicted / measured.

Definitions, fixed here so the same statistic is used on both sides:

  drift over 60 s   RMS of (theta_fixed - theta_float64) over the FINAL 10 s of the
                    60 s run. Not |d(60s)|: the budget predicts a bounded, zero-mean
                    floor, and the endpoint of a zero-mean process is one sample of it
                    -- on this trace it happens to land at 1.3e-8 deg, which would be a
                    meaningless headline. The settled RMS is the statistic the budget
                    predicts and it is the one used on both sides.
  drift rate        least-squares slope of the same difference over the final 30 s.
                    THIS is the falsifier for "there is no secular drift": the budget
                    predicts 0, and a bias-channel deadband would show up here as a
                    non-zero slope.
  max abs error     max_k |theta_fixed - theta_float64|, over the whole run, transient
                    included. Compared against the l1 bound.
  rms error         the same difference, RMS over the whole run.
  symmetry residual max_k |P01 - P10| as reported by the C binary.
  min eigenvalue    min_k lambda_min of the symmetric part of P, from the C binary.
  n negative steps  how many of the 12000 steps had an indefinite covariance. Carried
                    because "it went negative" and "it stayed negative" are different
                    claims and the second one is not the one this trace supports.

Errors are measured against float64, not against the true angle. Against truth the
number would be dominated by sensor noise (0.13 deg here) and would say nothing about
arithmetic; the fixed-vs-float difference is the arithmetic and nothing else. The
error-versus-truth column is carried alongside anyway, because a reader is entitled to
know that the filter works at all.

Writes results/comparison.csv and results/divergence_trace.csv.
"""

from __future__ import annotations

import csv
import math
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from reference import kalman_float as kfl                              # noqa: E402
from reference import kfparams as kp                                   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
RESULTS = ROOT / "results"
BINARY = BUILD / "kfhost"
TRACE = ROOT / "data" / "imu_capture.csv"
RAD2DEG = 180.0 / math.pi

# The gain precision at which the naive form is predicted to lose positive
# definiteness (error_budget.py: log2(S_max/R) - 1 = 12.29 bits). Chosen from the
# prediction, not from a search over what happens to break.
BREAKING_GAIN_BITS = 12
SAFE_GAIN_BITS = 30


def require_binary() -> None:
    if not BINARY.exists():
        sys.exit(f"{BINARY} not found. Build it first:\n    make host")


def run_host(variant: str, out: pathlib.Path, **flags) -> tuple[dict, np.ndarray]:
    """Run kfhost, returning (parsed summary, per-step array)."""
    cmd = [str(BINARY), "--trace", str(TRACE), "--variant", variant,
           "--out", str(out)]
    for key, val in flags.items():
        cmd += ["--" + key.replace("_", "-"), str(val)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    summary = {}
    for line in proc.stdout.splitlines():
        key, _, val = line.partition("=")
        summary[key] = val
    steps = np.genfromtxt(out, delimiter=",", names=True)
    return summary, steps


def stats(fixed_angle: np.ndarray, ref_angle: np.ndarray, truth: np.ndarray,
          summary: dict, t: np.ndarray) -> dict:
    d = (fixed_angle - ref_angle) * RAD2DEG
    e = fixed_angle - truth
    tail10 = t >= t[-1] - 10.0
    tail30 = t >= t[-1] - 30.0
    slope = float(np.polyfit(t[tail30], d[tail30], 1)[0])
    return {
        "drift_60s_deg": float(np.sqrt(np.mean(d[tail10] ** 2))),
        "drift_rate_deg_per_s": slope,
        "max_abs_err_deg": float(np.abs(d).max()),
        "rms_err_deg": float(np.sqrt(np.mean(d ** 2))),
        "rms_vs_truth_deg": float(np.sqrt(np.mean(e ** 2))) * RAD2DEG,
        "max_sym_resid": float(summary["max_sym_resid"]),
        "min_eigenvalue": float(summary["min_lambda"]),
        "first_negative_step": int(summary["first_negative_step"]),
        "n_negative_steps": int(summary["n_negative_steps"]),
        "saturation_events": int(summary["saturation_events"]),
    }


def read_budget() -> dict[tuple[str, str], float]:
    path = RESULTS / "error_budget.csv"
    if not path.exists():
        sys.exit(f"{path} missing -- run reference/error_budget.py first. The budget "
                 "is supposed to be written before the measurement, not after it.")
    with path.open() as fh:
        return {(r["prediction"], r["config"]): float(r["value"])
                for r in csv.DictReader(fh)}


def main() -> None:
    require_binary()
    RESULTS.mkdir(exist_ok=True)
    BUILD.mkdir(exist_ok=True)
    if not TRACE.exists():
        sys.exit(f"{TRACE} missing -- run reference/generate_trace.py first.")

    budget = read_budget()
    tr = kfl.load_trace(TRACE)
    dt = float(tr["t"][1] - tr["t"][0])
    truth = tr["true_angle"]

    ref = kfl.run(tr["gyro_rate"], tr["accel_angle"], dt=dt, joseph=True)
    ref_naive = kfl.run(tr["gyro_rate"], tr["accel_angle"], dt=dt, joseph=False)
    float_gap = np.abs(ref.angle - ref_naive.angle).max()

    runs = {}
    steps = {}
    for name, variant, bits in (
            ("fixed_joseph_gain_q1.30", "joseph", SAFE_GAIN_BITS),
            ("fixed_naive_gain_q1.30", "naive", SAFE_GAIN_BITS),
            ("fixed_joseph_gain_q1.%d" % BREAKING_GAIN_BITS, "joseph", BREAKING_GAIN_BITS),
            ("fixed_naive_gain_q1.%d" % BREAKING_GAIN_BITS, "naive", BREAKING_GAIN_BITS)):
        out = BUILD / f"steps_{variant}_g{bits}.csv"
        summary, arr = run_host(variant, out, gain_bits=bits)
        runs[name] = stats(arr["angle"], ref.angle, truth, summary, arr["t"])
        steps[name] = arr

    rows = []
    ref_err = ref.angle - truth
    rows.append({
        "implementation": "float64 reference (numpy)",
        "drift_60s_deg": 0.0, "drift_rate_deg_per_s": 0.0,
        "max_abs_err_deg": 0.0, "rms_err_deg": 0.0,
        "rms_vs_truth_deg": float(np.sqrt(np.mean(ref_err ** 2))) * RAD2DEG,
        "max_sym_resid": 0.0,
        "min_eigenvalue": float(min(np.linalg.eigvalsh(ref.p[i]).min()
                                    for i in range(len(ref.t)))),
        "first_negative_step": -1, "n_negative_steps": 0, "saturation_events": 0,
        "note": "ground truth by definition; naive and Joseph agree to %.2e rad here"
                % float_gap,
    })
    notes = {
        "fixed_joseph_gain_q1.30":
            "the design as specified in qformat.h",
        "fixed_naive_gain_q1.30":
            "textbook update, nominal gain precision: survives, but P is already "
            "asymmetric",
        "fixed_joseph_gain_q1.%d" % BREAKING_GAIN_BITS:
            "gain cut to %d fractional bits: suboptimal, still a covariance"
            % BREAKING_GAIN_BITS,
        "fixed_naive_gain_q1.%d" % BREAKING_GAIN_BITS:
            "gain cut to %d fractional bits: predicted to lose positive definiteness"
            % BREAKING_GAIN_BITS,
    }
    for name, st in runs.items():
        rows.append({"implementation": name, **st, "note": notes[name]})

    pred_rms = budget[("angle_error_rms", "nominal")]
    pred_peak_rms = budget[("angle_error_peak_rms", "nominal")]
    pred_bound = budget[("angle_error_bound", "nominal")]
    pred_sym_safe = budget[("naive_max_sym_resid", f"gain_bits={SAFE_GAIN_BITS}")]
    joseph = runs["fixed_joseph_gain_q1.30"]
    naive = runs["fixed_naive_gain_q1.30"]

    rows.append({
        "implementation": "PREDICTED lower (independent roundings)",
        "drift_60s_deg": pred_rms, "drift_rate_deg_per_s": 0.0,
        "max_abs_err_deg": pred_peak_rms, "rms_err_deg": pred_peak_rms,
        "rms_vs_truth_deg": float("nan"),
        "max_sym_resid": pred_sym_safe, "min_eigenvalue": float("nan"),
        "first_negative_step": -1, "n_negative_steps": 0, "saturation_events": 0,
        "note": "variance propagation: every rounding independent and uniform. "
                "Under-predicts, because roundings in a deterministic recursion are "
                "correlated in time -- the diagnosis, not an excuse",
    })
    rows.append({
        "implementation": "PREDICTED upper (l1 bound)",
        "drift_60s_deg": pred_bound, "drift_rate_deg_per_s": 0.0,
        "max_abs_err_deg": pred_bound, "rms_err_deg": pred_bound,
        "rms_vs_truth_deg": float("nan"),
        "max_sym_resid": pred_sym_safe, "min_eigenvalue": float("nan"),
        "first_negative_step": -1, "n_negative_steps": 0, "saturation_events": 0,
        "note": "l1 gain of the closed loop with every rounding at its full half-ulp "
                "and perfectly aligned in time; a strict upper bound",
    })
    rows.append({
        "implementation": "measured / PREDICTED lower",
        "drift_60s_deg": joseph["drift_60s_deg"] / pred_rms,
        "drift_rate_deg_per_s": float("nan"),
        "max_abs_err_deg": joseph["max_abs_err_deg"] / pred_peak_rms,
        "rms_err_deg": joseph["rms_err_deg"] / pred_peak_rms,
        "rms_vs_truth_deg": float("nan"),
        "max_sym_resid": (max(naive["max_sym_resid"], 2.0 ** -kp.FRAC["cov"])
                          / pred_sym_safe),
        "min_eigenvalue": float("nan"),
        "first_negative_step": -1, "n_negative_steps": 0, "saturation_events": 0,
        "note": "how far the independent-rounding model under-predicts",
    })
    rows.append({
        "implementation": "PREDICTED upper / measured",
        "drift_60s_deg": pred_bound / joseph["drift_60s_deg"],
        "drift_rate_deg_per_s": float("nan"),
        "max_abs_err_deg": pred_bound / joseph["max_abs_err_deg"],
        "rms_err_deg": pred_bound / joseph["rms_err_deg"],
        "rms_vs_truth_deg": float("nan"),
        "max_sym_resid": (pred_sym_safe
                          / max(naive["max_sym_resid"], 2.0 ** -kp.FRAC["cov"])),
        "min_eigenvalue": float("nan"),
        "first_negative_step": -1, "n_negative_steps": 0, "saturation_events": 0,
        "note": "how far the bound over-predicts; the measurement is bracketed",
    })

    fields = ["implementation", "drift_60s_deg", "drift_rate_deg_per_s",
              "max_abs_err_deg", "rms_err_deg", "rms_vs_truth_deg", "max_sym_resid",
              "min_eigenvalue", "first_negative_step", "n_negative_steps",
              "saturation_events", "note"]
    out = RESULTS / "comparison.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: (("%.6g" % v) if isinstance(v, float) else v)
                        for k, v in row.items()})

    # A decimated side-by-side trace at the breaking precision, for the divergence
    # figure. Committed because it is small and because it is the picture the brief
    # asks for; analysis/plot_results.py draws it.
    nkey = "fixed_naive_gain_q1.%d" % BREAKING_GAIN_BITS
    jkey = "fixed_joseph_gain_q1.%d" % BREAKING_GAIN_BITS
    dec = 20
    dpath = RESULTS / "divergence_trace.csv"
    with dpath.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "naive_lambda_min", "joseph_lambda_min", "naive_sym_resid",
                    "joseph_sym_resid", "naive_p00", "joseph_p00",
                    "naive_angle_err_deg", "joseph_angle_err_deg"])
        n = len(steps[nkey]["t"])
        for i in list(range(0, min(200, n))) + list(range(200, n, dec)):
            w.writerow(["%.6g" % v for v in (
                steps[nkey]["t"][i], steps[nkey]["lambda_min"][i],
                steps[jkey]["lambda_min"][i], steps[nkey]["sym_resid"][i],
                steps[jkey]["sym_resid"][i], steps[nkey]["p00"][i],
                steps[jkey]["p00"][i],
                (steps[nkey]["angle"][i] - ref.angle[i]) * RAD2DEG,
                (steps[jkey]["angle"][i] - ref.angle[i]) * RAD2DEG)])

    print(f"{'implementation':<40} {'drift60s':>10} {'drift/s':>10} {'max|d|':>10} "
          f"{'sym resid':>11} {'min eig':>11}")
    for row in rows:
        print(f"{row['implementation']:<40} {row['drift_60s_deg']:>10.4g} "
              f"{row['drift_rate_deg_per_s']:>10.3g} "
              f"{row['max_abs_err_deg']:>10.4g} {row['max_sym_resid']:>11.4g} "
              f"{row['min_eigenvalue']:>11.4g}")
    print(f"\nwrote {out}\nwrote {dpath}")


if __name__ == "__main__":
    main()
