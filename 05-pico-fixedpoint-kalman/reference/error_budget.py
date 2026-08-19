#!/usr/bin/env python3
"""THE ERROR BUDGET. Executable, and written before anything was measured.

The claim this project exists to support is "I can tell you the numerical error budget
in advance, not measure it afterwards and hope". So this file computes, from the format
choices alone, what the fixed-point filter's error will be -- and it does it WITHOUT
running the fixed-point filter. Nothing here imports the C binary or reads its output.
reference/compare.py does the measuring, afterwards, and the agreement factor between
the two is the deliverable.

--------------------------------------------------------------------------------------
1. THE UNIT OF THE BUDGET
--------------------------------------------------------------------------------------
Every arithmetic routine in firmware/qformat.h rounds to nearest, so one operation into
a Qm.n destination contributes an error e with |e| <= 2^-(n+1), and (for a value not
already on the grid) e is well modelled as uniform on [-2^-(n+1), 2^-(n+1)], variance
(2^-n)^2/12. Those two readings give the two predictions below: a deterministic worst
case that is a true bound, and an RMS prediction that is what you should actually
expect. Reporting only one of them is how error budgets get a reputation for being
useless -- the bound is 30x pessimistic and the RMS figure is not a guarantee.

--------------------------------------------------------------------------------------
2. WHERE THE ROUNDINGS ARE
--------------------------------------------------------------------------------------
Walking firmware/kalman_fixed.c one line at a time. Errors are attributed to the state
channel they land in.

  predict, angle channel (rad)
    gyro sample -> Q4.27                        e_rate,  scaled by dt
    bias Q1.30 -> Q4.27 rescale                 e_rate,  scaled by dt
    (omega - b) * dt -> Q3.28                   e_ang
    theta + dtheta                              exact (same format, no shift)
  update, angle channel (rad)
    accel sample -> Q3.28                       e_ang,   scaled by K0
    K0 = P00/S -> Q1.30                         e_gain0, scaled by |y|
    K0 * y -> Q3.28                             e_ang
  update, bias channel (rad/s)
    K1 = P10/S -> Q5.26                         e_gain1, scaled by |y|
    K1 * y -> Q1.30                             e_bias

There is a fourth channel and it turned out to be the dominant one, which is the whole
argument for writing the budget as code instead of as a paragraph. The COVARIANCE is
computed in fixed point too, and its error feeds the gain:

    dK0 = R dP00 / S^2 ,        dK1 = (dP10 - K1 dP00) / S

with S = P00 + R ~ 1e-4 rad^2 at steady state. One ulp of error in P10 therefore
becomes 1e4 ulps of error in K1 -- an amplification of four orders of magnitude, from
the same divide that makes the gain cheap. My first budget left this out entirely and
under-predicted the measured error by a factor of 70. The covariance error needs its
own recursion:

    dP_k = A_k dP_{k-1} A_k^T + eta_k

which is the same closed-loop matrix as the state error, because at the optimal gain
the Riccati map's derivative with respect to P IS the closed-loop congruence. One
matrix, two error channels.

|y| is not free data: E[y^2] = S = P00 + R, and the sequence P_k -- hence S_k and K_k --
is determined by (F, Qd, H, R, P0) before any measurement arrives. That is what makes
this file a prediction rather than a fit.

--------------------------------------------------------------------------------------
3. HOW THEY PROPAGATE
--------------------------------------------------------------------------------------
Let e_k be the difference between the fixed-point state and the float64 state. The two
filters run the same recursion, so subtracting them cancels everything except the
roundings, and to first order

    e_k = (I - K_k H) F e_{k-1}  +  (I - K_k H) u_pred  +  u_upd
        = A_k e_{k-1} + w_k

The closed-loop matrix A_k = (I - K_k H) F is a contraction (that is what a stable
filter is), so the error does NOT random-walk: it reaches a floor. Two propagations:

    RMS          V_k = A_k V_{k-1} A_k^T + W_k       W_k = diag of the variances
    worst case   E = sum_{j>=0} |A_ss^j| |w|         the l1 gain of the loop

The obvious worst case, E_k = |A_k| E_{k-1} + |w_k|, is WRONG here and it is worth
saying why, because it looks right. Taking |.| entrywise throws away the sign structure
that provides the damping: |A| has spectral radius above 1 for this loop even though A
has both eigenvalues at 0.996 and 0.978, so that recursion diverges and predicts 1e14
degrees. It is not a bad bound, it is not a bound at all. The correct worst case over
all bounded rounding sequences is the l1 gain -- the summed absolute impulse response
sum_j |A^j| -- which converges geometrically because A does. It is evaluated at the
steady-state gain, so it is a statement about the settled filter, not the transient.

--------------------------------------------------------------------------------------
4. WHY THERE IS NO SECULAR DRIFT, AND WHEN THERE WOULD BE
--------------------------------------------------------------------------------------
A bounded floor is only the right answer while every correction term stays above the
deadband of its destination format. If |K1 * y| ever fell below half an ulp of Q1.30,
the bias correction would round to zero, the bias state would freeze at whatever offset
it had, and the angle would then drift at that offset -- a genuine secular drift, linear
in t. The budget therefore predicts drift = 0 CONDITIONALLY, and prints the margin. It
is the conditional that is the engineering content: `deadband_margin_bias` below is how
many times larger the typical bias correction is than the smallest number the bias
format can hold.

--------------------------------------------------------------------------------------
5. THE NAIVE FORM'S TWO FAILURE THRESHOLDS, IN BITS
--------------------------------------------------------------------------------------
(a) SYMMETRY. The naive off-diagonals are P01(1-K0) and P10 - K1 P00, equal in exact
    arithmetic because K0 P01 = K1 P00 = P00 P01 / S. With separately rounded gains the
    difference d = P01 - P10 obeys, to first order,

        d_k = (1 - K0_k) d_{k-1} + eta_k ,
        eta_k = e_gain0 P01_k + e_gain1 P00_k + 2 e_cov

    so it neither cancels nor random-walks: it relaxes with rate K0 to a floor eta/K0.
    Running that scalar recursion on the float64 P sequence predicts max_k |d_k| for
    every gain precision. The Joseph form's prediction is exactly 0, because its two
    off-diagonal entries are the same expression with P01 and P10 exchanged.

(b) POSITIVE DEFINITENESS. The naive P+00 = P00 - K0 P00 has true value P00 R / S, so
    it is a cancellation whose surviving digits are a fraction R/S of the operands. An
    error e_gain0 = 2^-(g+1) in K0 lands on it scaled by P00, so

        P+00 < 0  <=>  2^-(g+1) P00 > P00 R / S  <=>  g < log2(S/R) - 1

    Taking the largest S over the run -- the diffuse-prior transient, S = P0 + R -- gives
    the predicted threshold. Joseph's P+00 = a(a P00) + K0(K0 R) is a sum of products of
    non-negative numbers and has no such threshold at any precision.

(c) COVARIANCE FORMAT. Independent of the update form: if the per-step process noise
    Qd rounds to zero the filter stops admitting error, P collapses and the estimate
    freezes. Requires 2^-n <= min(Q00, Q11)/2.

--------------------------------------------------------------------------------------
6. THE dt TRADE-OFF
--------------------------------------------------------------------------------------
F = I + A dt is exact for this model, so shrinking dt buys nothing in the transition.
What it does change:
  - larger dt: the gyro is held constant across the interval, an error of
    (dt^2/2)|omega_dot| per step -- grows as dt^2;
  - smaller dt: more steps per second, each injecting the roundings above, and each
    correction |K1 y| shrinking toward its deadband -- grows as 1/dt.
Feeding both into the same recursion gives a predicted optimum, which
analysis/dt_study.py then measures.

Writes results/error_budget.csv.
"""

from __future__ import annotations

import csv
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from reference import kfparams as kp                                    # noqa: E402
from reference.generate_trace import _MODES                             # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RAD2DEG = 180.0 / math.pi

# Gain precisions the sweeps will use. Listed here, before measurement, so the
# predicted column of results/comparison.csv cannot be chosen after the fact.
GAIN_BITS = (30, 28, 26, 24, 22, 20, 18, 16, 15, 14, 13, 12, 11, 10)
COV_BITS = (30, 29, 28, 27, 26, 25, 24, 22, 20, 18)
DT_HZ = (25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0,
         12800.0)
DT_STUDY_DURATION = 20.0   # s; the dt sweep runs shorter so 12.8 kHz stays cheap
COARSE_COV_BITS = 24       # a third dt-sweep build whose Qd cliff lands mid-range


def covariance_sequence(dt: float, n: int, frac_cov: int | None = None):
    """The Riccati recursion alone: P_k, S_k, K_k, with no measurements involved.

    If `frac_cov` is given, Qd and R are first rounded to that format, so the
    process-noise underflow of section 5(c) is included in the prediction rather than
    being a separate story.
    """
    q00, q01, q11 = kp.process_noise(dt)
    r = kp.SIGMA_ACC ** 2
    if frac_cov is not None:
        step = 2.0 ** (-frac_cov)
        q00, q01, q11, r = (round(v / step) * step for v in (q00, q01, q11, r))
    qd = np.array([[q00, q01], [q01, q11]])
    f = np.array([[1.0, -dt], [0.0, 1.0]])
    p = np.diag([kp.P0_ANGLE, kp.P0_BIAS])
    eye = np.eye(2)
    ps, ss, ks, ppred = np.zeros((n, 2, 2)), np.zeros(n), np.zeros((n, 2)), \
        np.zeros((n, 2, 2))
    for i in range(n):
        p = f @ p @ f.T + qd
        ppred[i] = p
        s = p[0, 0] + r
        k = p[:, 0] / s if s > 0 else np.zeros(2)
        a = eye - np.outer(k, [1.0, 0.0])
        p = a @ p @ a.T + r * np.outer(k, k)
        ps[i], ss[i], ks[i] = p, s, k
    return ps, ppred, ss, ks


def rms_omega_dot() -> float:
    """RMS angular acceleration of the synthetic motion, in closed form.

    omega_dot = -sum_i A_i (2 pi f_i)^2 sin(...), and the RMS of a sum of sinusoids at
    distinct frequencies is sqrt(sum of half-squares.)
    """
    return math.sqrt(sum((a * (2 * math.pi * f) ** 2) ** 2 / 2.0 for a, f, _ in _MODES))


def state_error_budget(dt: float, n: int, frac: dict[str, int]) -> dict[str, float]:
    """Propagate the per-operation representation errors to a predicted angle error."""
    e_ang = kp.half_ulp(frac["ang"])
    e_rate = kp.half_ulp(frac["rate"])
    e_bias = kp.half_ulp(frac["bias"])
    e_g0 = kp.half_ulp(frac["gain"])
    e_g1 = kp.half_ulp(frac["gain1"])

    _, _, ss, ks = covariance_sequence(dt, n, frac["cov"])
    y_rms = np.sqrt(ss)                       # E[y^2] = S, no data needed
    # Per-step injections. Deterministic magnitudes for the bound; variances (e^2/3 per
    # independent rounding, i.e. ulp^2/12) for the RMS propagation. The zero-order-hold
    # error is NOT here: the float64 reference holds the gyro across the interval in
    # exactly the same way, so it cancels in (fixed - float64). It reappears in the dt
    # study, which measures against the true angle.
    up_ang = 2.0 * dt * e_rate + e_ang
    vp_ang = (2.0 * (dt * e_rate) ** 2 + e_ang ** 2) / 3.0

    eye = np.eye(2)
    f = np.array([[1.0, -dt], [0.0, 1.0]])
    var = np.zeros((2, 2))
    # Covariance-error covariance, on vec(dP) in row-major order, so that
    # vec(A dP A^T) = (A kron A) vec(dP).
    var_p = np.zeros((4, 4))
    e_cov = kp.half_ulp(frac["cov"])
    # Roundings per covariance entry in kalman_joseph.c + kf_predict, counted from the
    # source: 7, 6, 6, 9. Independent, so variances add: n * e_cov^2 / 3.
    q_p = np.diag([7.0, 6.0, 6.0, 9.0]) * (e_cov ** 2 / 3.0)
    peak_rms = 0.0
    min_bias_correction = np.inf
    dk0_rms = dk1_rms = 0.0

    for i in range(n):
        k = ks[i]
        a_cl = eye - np.outer(k, [1.0, 0.0])
        a = a_cl @ f
        s_i = ss[i]

        # Gain error caused by covariance error, as linear functionals of vec(dP).
        g0 = np.array([kp.SIGMA_ACC ** 2 / s_i ** 2, 0.0, 0.0, 0.0])
        g1 = np.array([-k[1] / s_i, 0.0, 1.0 / s_i, 0.0])
        dk0_rms = math.sqrt(max(g0 @ var_p @ g0, 0.0))
        dk1_rms = math.sqrt(max(g1 @ var_p @ g1, 0.0))

        vu_ang = (2.0 * e_ang ** 2 + (e_g0 * y_rms[i]) ** 2) / 3.0 \
            + (dk0_rms * y_rms[i]) ** 2
        vu_bias = ((e_g1 * y_rms[i]) ** 2 + e_bias ** 2) / 3.0 \
            + (dk1_rms * y_rms[i]) ** 2
        w_var = a_cl @ np.diag([vp_ang, 0.0]) @ a_cl.T + np.diag([vu_ang, vu_bias])
        var = a @ var @ a.T + w_var

        m = np.kron(a, a)
        var_p = m @ var_p @ m.T + q_p

        peak_rms = max(peak_rms, math.sqrt(max(var[0, 0], 0.0)))
        min_bias_correction = min(min_bias_correction, abs(k[1]) * y_rms[i])

    # ---- worst case ---------------------------------------------------------------
    # The l1 gain of the settled loop, sum_j |A^j|, driven by every rounding at its
    # full half-ulp. This is a strict upper bound over all rounding sequences: no
    # RMS quantity enters it. The covariance channel gets the same treatment through
    # A kron A, and its bound feeds the gain error, which feeds the state.
    k = ks[-1]
    a_cl = eye - np.outer(k, [1.0, 0.0])
    a = a_cl @ f
    s_ss = ss[-1]
    y_ss = y_rms[-1]

    def l1_gain(mat: np.ndarray) -> np.ndarray:
        acc, power = np.zeros_like(mat), np.eye(mat.shape[0])
        for _ in range(500000):
            acc += np.abs(power)
            power = power @ mat
            if np.abs(power).max() < 1e-16:
                break
        return acc

    eta = np.array([7.0, 6.0, 6.0, 9.0]) * e_cov     # per-entry rounding magnitudes
    dp_bound = l1_gain(np.kron(a, a)) @ eta
    g0 = np.array([kp.SIGMA_ACC ** 2 / s_ss ** 2, 0.0, 0.0, 0.0])
    g1 = np.array([-k[1] / s_ss, 0.0, 1.0 / s_ss, 0.0])
    dk0_bound = float(np.abs(g0) @ dp_bound)
    dk1_bound = float(np.abs(g1) @ dp_bound)

    w = a_cl @ np.array([up_ang, 0.0]) + np.array(
        [2.0 * e_ang + (e_g0 + dk0_bound) * y_ss,
         (e_g1 + dk1_bound) * y_ss + e_bias])
    bound = l1_gain(a) @ np.abs(w)

    return {
        "bound_angle_rad": float(bound[0]),
        "rms_angle_rad": float(math.sqrt(max(var[0, 0], 0.0))),
        "rms_angle_peak_rad": float(peak_rms),
        "rms_bias_rad_s": float(math.sqrt(max(var[1, 1], 0.0))),
        "loop_spectral_radius": float(max(abs(np.linalg.eigvals(a)))),
        "gain_error_amplification": float(dk1_rms / kp.half_ulp(frac["gain1"])),
        "dk0_rms": float(dk0_rms),
        "dk1_rms": float(dk1_rms),
        "dk0_bound": dk0_bound,
        "dk1_bound": dk1_bound,
        "dp00_rms": float(math.sqrt(max(var_p[0, 0], 0.0))),
        "dp00_bound": float(dp_bound[0]),
        "deadband_margin_bias": float(min_bias_correction / kp.half_ulp(frac["bias"])),
        "deadband_margin_angle": float(
            (np.abs(ks[:, 0]) * y_rms).min() / kp.half_ulp(frac["ang"])),
    }


def naive_symmetry_prediction(dt: float, n: int, frac_cov: int, gain_bits: int,
                              gain1_bits: int) -> float:
    """max_k |P01 - P10| for the naive update, from the scalar relaxation of 5(a)."""
    e_g0 = kp.half_ulp(gain_bits)
    e_g1 = kp.half_ulp(gain1_bits)
    e_cov = kp.half_ulp(frac_cov)
    ps, ppred, _, ks = covariance_sequence(dt, n, frac_cov)
    d = 0.0
    worst = 0.0
    for i in range(n):
        eta = e_g0 * abs(ppred[i, 0, 1]) + e_g1 * ppred[i, 0, 0] + 2.0 * e_cov
        d = (1.0 - ks[i, 0]) * d + eta
        worst = max(worst, d)
    return worst


def naive_pd_threshold_bits(dt: float, n: int, frac_cov: int) -> float:
    """Largest gain fractional-bit count at which naive P+00 can still go negative."""
    _, _, ss, _ = covariance_sequence(dt, n, frac_cov)
    r = kp.SIGMA_ACC ** 2
    return math.log2(ss.max() / r) - 1.0


def cov_bits_threshold() -> float:
    """Fewest covariance fractional bits that RESOLVE the process noise (>= 2 ulps).

    The soft criterion. A Qd represented by one ulp is quantised beyond usefulness even
    though it is not zero, so this is where the design should sit.
    """
    q00, _, q11 = kp.process_noise(kp.DT)
    return math.log2(2.0 / min(q00, q11))


def cov_bits_cliff() -> float:
    """Fewest covariance fractional bits before Qd rounds to exactly ZERO.

    The hard criterion, and a different number: round-to-nearest sends Qd to 0 once
    Qd * 2^n < 1/2, so the cliff is at log2(0.5 / min(Q00, Q11)) -- two bits below the
    soft threshold. Between the two the filter is degraded but alive; below the cliff
    the process noise is gone and the covariance collapses.
    """
    q00, _, q11 = kp.process_noise(kp.DT)
    return math.log2(0.5 / min(q00, q11))


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    n = int(round(kp.DURATION / kp.DT))
    frac = dict(kp.FRAC)
    frac["gain1"] = kp.FRAC["gain"] - 4          # K1 needs 4 more integer bits
    rows: list[dict] = []

    def add(pred, config, value, units, note):
        rows.append({"prediction": pred, "config": config,
                     "value": "%.10g" % value, "units": units, "note": note})

    # --- the formats themselves -----------------------------------------------------
    for name, f in sorted(frac.items()):
        add("half_ulp", f"{name}:Q.{f}", kp.half_ulp(f), "value units",
            "2^-(n+1), the per-operation representation error")

    # --- state error at the nominal format ------------------------------------------
    nom = state_error_budget(kp.DT, n, frac)
    add("angle_error_rms", "nominal", nom["rms_angle_rad"] * RAD2DEG, "deg",
        "LOWER estimate: every rounding independent and uniform, variances added. "
        "Roundings in a deterministic recursion are not white, so this under-predicts")
    add("angle_error_bound", "nominal", nom["bound_angle_rad"] * RAD2DEG, "deg",
        "UPPER bound: l1 gain of the loop with every rounding at its full half-ulp "
        "and perfectly aligned in time")
    add("angle_error_peak_rms", "nominal", nom["rms_angle_peak_rad"] * RAD2DEG, "deg",
        "the white-noise estimate at its largest, i.e. including the transient")
    add("cov_error_rms", "nominal", nom["dp00_rms"], "rad^2",
        "RMS error in P00 from the vec(dP) = (A kron A) vec(dP) + eta recursion")
    add("cov_error_bound", "nominal", nom["dp00_bound"], "rad^2",
        "same recursion, l1 gain, every covariance rounding at its half-ulp")
    add("angle_drift_60s", "nominal", nom["rms_angle_rad"] * RAD2DEG, "deg",
        "the loop is a contraction, so |d(60s)| is the floor, not a growing walk")
    add("angle_drift_rate", "nominal", 0.0, "deg/s",
        "predicted ZERO secular drift, conditional on the deadband margins below")
    add("bias_error_rms", "nominal", nom["rms_bias_rad_s"] * RAD2DEG, "deg/s",
        "RMS of (fixed - float64) in the bias state")
    add("gain_error_amplification", "nominal", nom["gain_error_amplification"], "x",
        "RMS error in K1 caused by covariance rounding, in half-ulps of K1's own "
        "format: the covariance channel dominates the divide's own rounding by this "
        "factor")
    add("deadband_margin_bias", "nominal", nom["deadband_margin_bias"], "x half-ulp",
        "smallest |K1 y| over the run, in half-ulps of Q1.30; >1 means no freeze")
    add("deadband_margin_angle", "nominal", nom["deadband_margin_angle"], "x half-ulp",
        "smallest |K0 y| over the run, in half-ulps of Q3.28")

    # --- one global format, the beginner's mistake ----------------------------------
    gfrac = {k: kp.GLOBAL_FRAC for k in frac}
    glob = state_error_budget(kp.DT, n, gfrac)
    add("angle_error_rms", "global_Q4.27", glob["rms_angle_rad"] * RAD2DEG, "deg",
        "every quantity forced into the format the gyro range demands")
    add("angle_error_ratio", "global_over_nominal",
        glob["rms_angle_rad"] / nom["rms_angle_rad"], "x",
        "cost of using one format for everything")

    # --- naive-vs-Joseph thresholds -------------------------------------------------
    pd_bits = naive_pd_threshold_bits(kp.DT, n, frac["cov"])
    add("naive_pd_threshold", "gain_bits", pd_bits, "bits",
        "naive P+00 goes negative for gain fractional bits below this; "
        "Joseph has no such threshold")
    add("joseph_pd_threshold", "gain_bits", float("-inf"), "bits",
        "P+00 = a(a P00) + K0(K0 R) is a sum of non-negative products")
    add("cov_bits_threshold", "cov_bits", cov_bits_threshold(), "bits",
        "soft: below this Qd is resolved by fewer than 2 ulps")
    add("cov_bits_cliff", "cov_bits", cov_bits_cliff(), "bits",
        "hard: below this Qd rounds to exactly zero, P collapses and the estimate "
        "freezes. Affects BOTH update forms -- Joseph does not fix this one")

    for g in GAIN_BITS:
        add("naive_max_sym_resid", f"gain_bits={g}",
            naive_symmetry_prediction(kp.DT, n, frac["cov"], g, g - 4), "rad^2",
            "relaxation d_k = (1-K0)d_{k-1} + eta, run on the float64 P sequence")
        add("joseph_max_sym_resid", f"gain_bits={g}", 0.0, "rad^2",
            "exactly zero: congruence, same expression with P01 and P10 exchanged")

    # --- the dt trade-off -----------------------------------------------------------
    # Two mechanisms, opposite signs, both predicted in closed form. The measured
    # optimum in analysis/dt_study.py has to sit between them.
    wdot = rms_omega_dot()
    add("rms_omega_dot", "trace", wdot, "rad/s^2",
        "closed form from the four motion modes; sets the zero-order-hold error")
    for hz in DT_HZ:
        dt = 1.0 / hz
        nn = int(round(DT_STUDY_DURATION * hz))
        b = state_error_budget(dt, nn, frac)
        _, _, ss, ks = covariance_sequence(dt, nn, frac["cov"])
        zoh = 0.5 * dt * dt * wdot / ks[-1, 0]     # per-step ZOH error / loop DC gain
        add("zoh_error_vs_dt", f"fs={hz:g}Hz", zoh * RAD2DEG, "deg",
            "(dt^2/2)|omega_dot| per step, amplified by the loop DC gain 1/K0; "
            "an upper estimate because the term is oscillatory, not DC")
        add("rounding_error_vs_dt", f"fs={hz:g}Hz", b["rms_angle_rad"] * RAD2DEG,
            "deg", "RMS of (fixed - float64) from the variance propagation")
        add("filter_sigma_vs_dt", f"fs={hz:g}Hz",
            math.sqrt(ss[-1] - kp.SIGMA_ACC ** 2) * RAD2DEG, "deg",
            "sqrt(P00) at steady state: the estimation error the filter itself claims")

    # "Too small dt" has a hard edge, not just a soft trend: Qd is proportional to dt,
    # so below dt = 2^-n / sigma^2 the process noise rounds to zero at every step and
    # the covariance collapses. That is a sample RATE ceiling set by a number format.
    q_slope = min(kp.SIGMA_GYRO ** 2, kp.SIGMA_BIAS ** 2)
    for label, bits in (("nominal_Q1.30", frac["cov"]),
                        ("global_Q4.27", kp.GLOBAL_FRAC),
                        ("coarse_Q1.24", COARSE_COV_BITS)):
        # Qd = sigma^2 dt rounds to zero once sigma^2 dt < 2^-(n+1), so the sample-rate
        # ceiling is fs = 2 sigma^2 2^n. It is a number format setting a sample rate.
        add("max_sample_rate", label, 2.0 * q_slope * 2.0 ** bits, "Hz",
            "above this, Qd = sigma^2 dt rounds to zero and the covariance collapses")

    out = RESULTS / "error_budget.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["prediction", "config", "value", "units",
                                           "note"])
        w.writeheader()
        w.writerows(rows)

    print("ERROR BUDGET (predicted, before any fixed-point run)")
    print(f"  angle error, white-noise    {nom['rms_angle_rad'] * RAD2DEG:.3e} deg "
          f"(lower estimate)")
    print(f"  angle error, l1 bound       {nom['bound_angle_rad'] * RAD2DEG:.3e} deg "
          f"(upper bound)")
    print(f"  P00 error, white / bound    {nom['dp00_rms'] / 2 ** -30:.2f} / "
          f"{nom['dp00_bound'] / 2 ** -30:.2f} ulps")
    print(f"  loop spectral radius        {nom['loop_spectral_radius']:.6f}")
    print(f"  gain error from cov rounding {nom['gain_error_amplification']:.4g}x the "
          f"gain format's own half-ulp")
    print(f"  secular drift rate          0 deg/s  (bias deadband margin "
          f"{nom['deadband_margin_bias']:.3g}x, angle "
          f"{nom['deadband_margin_angle']:.3g}x)")
    print(f"  one-global-format penalty   {glob['rms_angle_rad'] / nom['rms_angle_rad']:.2f}x")
    print(f"  naive loses PD below        {pd_bits:.2f} gain fractional bits")
    print(f"  Qd under-resolved below     {cov_bits_threshold():.2f} covariance bits")
    print(f"  Qd rounds to zero below     {cov_bits_cliff():.2f} covariance bits")
    q_slope = min(kp.SIGMA_GYRO, kp.SIGMA_BIAS) ** 2
    print(f"  Qd rounds to zero above     "
          f"{2 * q_slope * 2.0 ** frac['cov']:.0f} Hz (Q1.30), "
          f"{2 * q_slope * 2.0 ** kp.GLOBAL_FRAC:.0f} Hz (global Q4.27), "
          f"{2 * q_slope * 2.0 ** COARSE_COV_BITS:.0f} Hz (Q1.24)")
    print(f"\nwrote {out}  ({len(rows)} predictions)")


if __name__ == "__main__":
    main()
