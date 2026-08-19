#!/usr/bin/env python3
"""Synthesise an IMU trace with a KNOWN true orientation.

No board is attached to the machine this repo was built on, so there is no recorded
capture to replay. That is stated plainly rather than worked around: the trace here is
synthetic, seeded and reproducible, and it is what makes the entire numerical
comparison runnable today. What a real capture would change is the *realism* of the
input (vibration, linear acceleration, temperature-dependent bias); what it would not
change is any of the arithmetic being measured, because the fixed-point error budget is
a property of the recursion and the formats, not of the data.

The motion is a sum of four incommensurate sinusoids -- smooth, band-limited, and with
no repeating period inside 60 s, so the filter never sees the same state twice:

    theta(t)  = sum_i A_i sin(2 pi f_i t + phi_i)
    omega(t)  = dtheta/dt, in closed form (not differenced -- a differenced "truth"
                would carry its own O(dt^2) error into the thing being measured)

The gyro sees the true rate plus a bias that both ramps and random-walks, plus white
noise at the density the filter's Q assumes. The accelerometer sees the true angle plus
white noise at the density the filter's R assumes. The filter is therefore correctly
tuned by construction: any discrepancy between the fixed-point and float64 runs is
numerical, not a modelling mismatch. That is the whole point of using a synthetic trace
for this particular measurement.

Writes data/imu_capture.csv:  t, gyro_rate, accel_angle, true_angle, true_bias
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from reference import kfparams as kp                                    # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Amplitudes (rad), frequencies (Hz), phases (rad). Chosen so that
#   max|theta| ~ 0.9 rad  (inside the Q3.28 bound of 8 rad with large margin)
#   max|omega| ~ 2.4 rad/s (inside the +-500 deg/s = 8.73 rad/s gyro full scale)
# and so that no two frequencies are rationally related inside the window.
_MODES = (
    (0.50, 0.13, 0.00),
    (0.25, 0.37, 1.10),
    (0.10, 0.83, 2.40),
    (0.08, 1.70, 0.55),
)

BIAS0 = 0.03          # rad/s, a 1.7 deg/s turn-on offset -- typical MEMS
BIAS_RAMP = 2.0e-4    # rad/s per s, a slow thermal ramp over the 60 s window


def true_motion(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (theta, omega) in closed form."""
    theta = np.zeros_like(t)
    omega = np.zeros_like(t)
    for amp, freq, phase in _MODES:
        w = 2.0 * np.pi * freq
        theta += amp * np.sin(w * t + phase)
        omega += amp * w * np.cos(w * t + phase)
    return theta, omega


def generate(dt: float = kp.DT, duration: float = kp.DURATION,
             seed: int = kp.SEED) -> dict[str, np.ndarray]:
    """One reproducible trace. Every draw goes through this rng, no global seeding."""
    rng = np.random.default_rng(seed)
    n = int(round(duration / dt))
    t = np.arange(n) * dt
    theta, omega = true_motion(t)

    # Bias: deterministic turn-on offset + thermal ramp + rate random walk. The walk's
    # per-step standard deviation is sigma_bias*sqrt(dt), which is exactly the Qd(1,1)
    # the filter uses -- so the filter's process model is the truth, not an
    # approximation of it.
    walk = rng.normal(0.0, kp.SIGMA_BIAS * np.sqrt(dt), size=n).cumsum()
    bias = BIAS0 + BIAS_RAMP * t + walk

    # Gyro white noise is specified as a density; the per-sample standard deviation is
    # sigma_gyro/sqrt(dt). Getting this scaling wrong is the classic way to produce a
    # filter that looks mistuned at one sample rate and fine at another, and it is also
    # what makes the dt study in analysis/ mean anything.
    gyro = omega + bias + rng.normal(0.0, kp.SIGMA_GYRO / np.sqrt(dt), size=n)
    accel = theta + rng.normal(0.0, kp.SIGMA_ACC, size=n)

    return {"t": t, "gyro_rate": gyro, "accel_angle": accel,
            "true_angle": theta, "true_bias": bias}


def write_csv(path: pathlib.Path, trace: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["t", "gyro_rate", "accel_angle", "true_angle", "true_bias"]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for row in zip(*(trace[c] for c in cols)):
            w.writerow(["%.10g" % v for v in row])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dt", type=float, default=kp.DT)
    ap.add_argument("--duration", type=float, default=kp.DURATION)
    ap.add_argument("--seed", type=int, default=kp.SEED)
    ap.add_argument("--out", type=pathlib.Path,
                    default=ROOT / "data" / "imu_capture.csv")
    args = ap.parse_args()

    trace = generate(args.dt, args.duration, args.seed)
    write_csv(args.out, trace)
    n = len(trace["t"])
    print(f"wrote {args.out}  ({n} samples, dt={args.dt} s, {args.duration} s)")
    print(f"  max|theta| = {np.abs(trace['true_angle']).max():.4f} rad "
          f"(Q3.28 bound 8 rad)")
    print(f"  max|gyro|  = {np.abs(trace['gyro_rate']).max():.4f} rad/s "
          f"(Q4.27 bound 16 rad/s, sensor full scale 8.73)")
    print(f"  bias range = [{trace['true_bias'].min():.5f}, "
          f"{trace['true_bias'].max():.5f}] rad/s (Q1.30 bound 2 rad/s)")


if __name__ == "__main__":
    main()
