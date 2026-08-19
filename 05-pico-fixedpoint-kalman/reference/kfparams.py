"""The filter's physical constants, in one place, in SI units.

These duplicate the KF_* macros in firmware/kalman_fixed.h. They are duplicated rather
than generated because a generator is a build step and this project has to compile with
`gcc file.c` and run with `python3 file.py` and nothing else. The duplication is made
safe by tests/test_reference.py, which shells out to `kfhost --params` and asserts the
two agree exactly -- a divergence between the C constants and the Python reference
would silently turn a numerical comparison into a modelling comparison, which is the
single most likely way to fake a result in this repo.

The model, stated once (the derivation is in README.md and firmware/kalman_fixed.h):

    d/dt [theta; b] = [[0,-1],[0,0]] [theta; b] + [omega_meas; 0] + w,
    w ~ N(0, diag(sigma_gyro^2, sigma_bias^2))     continuous-time PSDs
    z = theta + v,   v ~ N(0, sigma_acc^2)
"""

from __future__ import annotations

DT = 0.005            # s, 200 Hz. The IMU's ODR; also the filter's step.
SIGMA_GYRO = 3.0e-3   # rad/s/sqrt(Hz). Angular random walk of the gyro channel.
SIGMA_BIAS = 3.0e-3   # rad/s^1.5. Rate random walk that drives the bias state.
SIGMA_ACC = 1.0e-2    # rad. Accelerometer inclination noise, ~0.57 deg.
P0_ANGLE = 1.0        # rad^2. Diffuse prior: attitude unknown to about 1 rad at boot.
P0_BIAS = 0.25        # rad^2/s^2. Bias unknown to about 0.5 rad/s at boot.

DURATION = 60.0       # s. The brief's measurement window.
SEED = 20240517       # every random draw in this repo comes through this seed.

# Nominal fixed-point formats, mirroring qformat.h. Used by the error budget, which
# must be able to compute 2^-(n+1) for each quantity without running any C.
FRAC = {"ang": 28, "rate": 27, "bias": 30, "dt": 31, "cov": 30, "gain": 30}
GLOBAL_FRAC = 27      # the "one format for everything" build: forced by the gyro range


def half_ulp(frac: int) -> float:
    """Representation error of a Qm.n format: 2^-(n+1)."""
    return 2.0 ** (-(frac + 1))


def process_noise(dt: float, sigma_gyro: float = SIGMA_GYRO,
                  sigma_bias: float = SIGMA_BIAS) -> tuple[float, float, float]:
    """Van Loan discrete process noise (Q00, Q01, Q11) for this exactly-linear model.

    Qd = int_0^dt exp(A s) Qc exp(A s)^T ds with A nilpotent, so the integral is a
    polynomial in dt and the discretisation carries no truncation error at all.
    """
    sg2, sb2 = sigma_gyro ** 2, sigma_bias ** 2
    return (sg2 * dt + sb2 * dt ** 3 / 3.0, -0.5 * sb2 * dt ** 2, sb2 * dt)
