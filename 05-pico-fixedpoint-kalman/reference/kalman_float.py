#!/usr/bin/env python3
"""Float64 ground truth: the same two-state filter, in numpy, at 53-bit precision.

This is the reference every fixed-point number in the repo is measured against. It is
deliberately the *same* recursion, in the *same* order, on the *same* trace as
firmware/kalman_fixed.c -- predict with sample k's gyro reading, then update with
sample k's accelerometer reading -- so that the difference between the two is
arithmetic and nothing else.

    predict     x- = F x + B omega,        F = [[1, -dt], [0, 1]],  B = [dt, 0]^T
                P- = F P F^T + Qd
    update      S  = H P- H^T + R = P-_00 + R
                K  = P- H^T / S
                x+ = x- + K (z - H x-)
                P+ = (I - K H) P-                        naive
                P+ = (I-KH) P- (I-KH)^T + K R K^T        Joseph

In float64 the two covariance forms agree to about 1e-16 relative on this problem, and
test_reference.py asserts that they do -- if they disagreed here, the divergence seen in
fixed point would not be a fixed-point effect. That check is what earns the right to
call this "ground truth".

The gain sequence K_k depends only on (F, Qd, H, R, P0), not on the measurements, so it
is fully determined before any data arrives. reference/error_budget.py exploits that:
it propagates rounding error through this exact gain sequence to predict the
fixed-point error *without ever running the fixed-point filter*.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from reference import kfparams as kp                                    # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclass
class FloatRun:
    """Everything a comparison or an error budget needs from one float64 run."""
    t: np.ndarray            # (n,)      s
    angle: np.ndarray        # (n,)      rad
    bias: np.ndarray         # (n,)      rad/s
    p: np.ndarray            # (n, 2, 2) covariance after the update
    p_pred: np.ndarray       # (n, 2, 2) covariance after the predict, before update
    k: np.ndarray            # (n, 2)    Kalman gain
    s: np.ndarray            # (n,)      innovation covariance
    innovation: np.ndarray   # (n,)      rad


def load_trace(path: pathlib.Path) -> dict[str, np.ndarray]:
    raw = np.genfromtxt(path, delimiter=",", names=True)
    return {name: np.asarray(raw[name], dtype=np.float64) for name in raw.dtype.names}


def run(gyro: np.ndarray, accel: np.ndarray, dt: float = kp.DT,
        joseph: bool = True, p0_angle: float = kp.P0_ANGLE,
        p0_bias: float = kp.P0_BIAS, sigma_acc: float = kp.SIGMA_ACC,
        sigma_gyro: float = kp.SIGMA_GYRO,
        sigma_bias: float = kp.SIGMA_BIAS) -> FloatRun:
    """Run the filter in float64. `joseph` selects the covariance update form."""
    n = len(gyro)
    q00, q01, q11 = kp.process_noise(dt, sigma_gyro, sigma_bias)
    qd = np.array([[q00, q01], [q01, q11]])
    f = np.array([[1.0, -dt], [0.0, 1.0]])
    h = np.array([[1.0, 0.0]])
    r = sigma_acc ** 2
    eye = np.eye(2)

    x = np.zeros(2)
    p = np.diag([p0_angle, p0_bias])

    out = FloatRun(np.arange(n) * dt, np.zeros(n), np.zeros(n),
                   np.zeros((n, 2, 2)), np.zeros((n, 2, 2)),
                   np.zeros((n, 2)), np.zeros(n), np.zeros(n))

    for i in range(n):
        # predict
        x = np.array([x[0] + (gyro[i] - x[1]) * dt, x[1]])
        p = f @ p @ f.T + qd
        out.p_pred[i] = p

        # update
        s = p[0, 0] + r
        k = p[:, 0] / s
        y = accel[i] - x[0]
        x = x + k * y
        if joseph:
            a = eye - np.outer(k, h[0])
            p = a @ p @ a.T + r * np.outer(k, k)
        else:
            p = (eye - np.outer(k, h[0])) @ p

        out.angle[i], out.bias[i] = x
        out.p[i] = p
        out.k[i] = k
        out.s[i] = s
        out.innovation[i] = y
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=pathlib.Path,
                    default=ROOT / "data" / "imu_capture.csv")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="optional per-step CSV")
    ap.add_argument("--naive", action="store_true",
                    help="use the textbook covariance update instead of Joseph")
    ap.add_argument("--decimate", type=int, default=1)
    args = ap.parse_args()

    tr = load_trace(args.trace)
    dt = float(tr["t"][1] - tr["t"][0])
    res = run(tr["gyro_rate"], tr["accel_angle"], dt=dt, joseph=not args.naive)

    err = res.angle - tr["true_angle"]
    print(f"float64 {'naive' if args.naive else 'joseph'}: "
          f"rms angle error = {np.sqrt(np.mean(err ** 2)) * 180 / np.pi:.6f} deg, "
          f"max = {np.abs(err).max() * 180 / np.pi:.6f} deg")
    print(f"  steady-state P = {res.p[-1].tolist()}")
    print(f"  min eigenvalue over run = "
          f"{min(np.linalg.eigvalsh(res.p[i]).min() for i in range(len(res.t))):.6e}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["k", "t", "angle", "bias", "p00", "p01", "p10", "p11",
                        "k0", "k1", "true_angle", "true_bias"])
            for i in range(0, len(res.t), args.decimate):
                w.writerow(["%d" % i] + ["%.10g" % v for v in
                                         (res.t[i], res.angle[i], res.bias[i],
                                          res.p[i, 0, 0], res.p[i, 0, 1],
                                          res.p[i, 1, 0], res.p[i, 1, 1],
                                          res.k[i, 0], res.k[i, 1],
                                          tr["true_angle"][i], tr["true_bias"][i])])
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
