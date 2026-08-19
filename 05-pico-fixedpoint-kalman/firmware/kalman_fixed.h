/* kalman_fixed.h -- a two-state attitude Kalman filter in fixed point.
 *
 * THE MODEL
 * ---------
 * State x = [theta, b]^T : gravity-referenced angle (rad) and gyro bias (rad/s).
 * The gyro measures the true rate plus the bias, so in continuous time
 *
 *     d/dt theta = omega_meas - b + w_theta
 *     d/dt b     =              0 + w_b
 *
 * i.e. xdot = A x + B omega_meas + w with
 *
 *     A = [ 0  -1 ]      B = [ 1 ]      Qc = diag(sigma_g^2, sigma_b^2)
 *         [ 0   0 ]          [ 0 ]
 *
 * A is nilpotent (A^2 = 0), so the matrix exponential terminates after one term and
 * the discretisation is EXACT for any dt, not first order:
 *
 *     F = exp(A dt) = I + A dt = [ 1  -dt ]
 *                                [ 0   1  ]
 *
 * There is therefore no linearisation error in F. The dt dependence that
 * analysis/dt_study.py finds comes from the two places it actually lives: the
 * zero-order hold on omega_meas between samples (error O(dt^2 * omega_dot)), and the
 * per-step rounding, which accumulates in proportion to the number of steps. Those
 * pull in opposite directions and there is an optimum.
 *
 * The discrete process noise is the Van Loan integral, also exact:
 *
 *     Qd = int_0^dt exp(A s) Qc exp(A s)^T ds
 *        = [ sigma_g^2 dt + sigma_b^2 dt^3/3   -sigma_b^2 dt^2/2 ]
 *          [ -sigma_b^2 dt^2/2                  sigma_b^2 dt      ]
 *
 * At dt = 5 ms the off-diagonal term is -1.1e-10 rad^2/s, which is 0.12 of one Q1.30
 * ulp: it rounds to zero, and Qd degenerates to diag(Q00, Q11). That is a modelling
 * change of relative size 6e-6 in det(Qd) and it leaves Qd positive definite, so it is
 * recorded here as a known, bounded approximation rather than pretended away.
 *
 * The measurement is the accelerometer's inclination estimate, z = theta + v, so
 * H = [1 0] and R = sigma_a^2. It is a scalar measurement, which is why the gain is a
 * divide by a scalar and not a 2x2 inverse -- the single most important structural
 * decision for a fixed-point implementation.
 *
 * THE TWO COVARIANCE UPDATES
 * --------------------------
 *   naive   P+ = (I - K H) P-
 *   Joseph  P+ = (I - K H) P- (I - K H)^T + K R K^T
 *
 * They are algebraically identical at the optimal gain. They are not numerically
 * identical, and the difference is structural, not incidental:
 *
 *   - Joseph is a congruence plus a congruence. Entry (i,j) and entry (j,i) of
 *     A P A^T are the same expression in the same operands with i and j exchanged, so
 *     on a symmetric P they are computed by identical instruction sequences and come
 *     out BIT-EXACTLY equal. Symmetry is preserved by construction, at any precision.
 *   - The naive form is not a congruence. Its (0,1) entry is P01 - K0*P01 and its
 *     (1,0) entry is P10 - K1*P00: different operands, different roundings. The
 *     residual ||P - P^T|| is first order in the gain's representation error, and
 *     reference/error_budget.py predicts it as 2^-(g+1) * max_k P00(k).
 *   - Joseph's P+ is the exact covariance of the estimator that uses gain K, for ANY
 *     K, so a mis-rounded gain makes it suboptimal but never indefinite. Written as
 *     a*(a*P00) + K0*(K0*R) the (0,0) entry is a sum of two non-negative products and
 *     cannot go negative. The naive (0,0) entry is P00 - K0*P00, a subtraction of two
 *     nearly equal numbers when P00 >> R -- exactly the diffuse-prior transient at
 *     power-on -- and it goes negative once 2^-(g+1) * P00 > P00 R / (P00 + R).
 *
 * DO NOT expand the completed square when porting this. Computing c*c and K1*K1 in
 * the gain format annihilates both terms at coarse precision (K1 ~ 1e-2, so K1^2 is
 * below one ulp long before K1 is) and destroys the positive-definiteness guarantee
 * while leaving the indefinite cross term intact. Every square here is chained:
 * c*(c*P00), K1*(K1*R).
 *
 * PRECISION KNOBS
 * ---------------
 * The fractional-bit counts are fields of kf_t rather than compile-time constants.
 * On the RP2040 they would be the macros in qformat.h and every shift would fold away
 * at compile time; they are run-time here for exactly one reason -- the bit-depth
 * sweeps in analysis/ need to ask "how many bits before it falls over" without
 * rebuilding, and that question IS the error budget. The arithmetic is otherwise
 * identical: same int32 storage, same int64 intermediates, same rounding.
 */

#ifndef KALMAN_FIXED_H
#define KALMAN_FIXED_H

#include <stdint.h>

#include "qformat.h"

/* ---- nominal filter parameters ---------------------------------------------------
 * These are the values reference/generate_trace.py draws the synthetic trace with, so
 * the filter is correctly tuned by construction and every discrepancy against the
 * float64 reference is numerical rather than a modelling mismatch. host/kfhost
 * --params prints them and tests/test_reference.py checks the Python side agrees. */
#define KF_DT_DEFAULT     0.005    /* s        200 Hz sample rate                    */
#define KF_SIGMA_GYRO     3.0e-3   /* rad/s/sqrt(Hz)  gyro angular-random-walk       */
#define KF_SIGMA_BIAS     3.0e-3   /* rad/s^1.5       bias random-walk (rate ramp)   */
#define KF_SIGMA_ACC      1.0e-2   /* rad             accelerometer inclination noise*/
#define KF_P0_ANGLE       1.0      /* rad^2    diffuse prior: angle unknown to ~1 rad*/
#define KF_P0_BIAS        0.25     /* rad^2/s^2  bias unknown to ~0.5 rad/s          */

typedef enum {
    KF_NAIVE  = 0,   /* P+ = (I - K H) P-                                */
    KF_JOSEPH = 1    /* P+ = (I - K H) P- (I - K H)^T + K R K^T          */
} kf_variant_t;

/* Fractional bit counts. Defaults are the macros in qformat.h; the sweeps override
 * them. Constraints: gain <= 30, because 1 - K0 must be representable and 1.0 in
 * Q1.31 does not fit an int32; and gain1 = gain - 4, because K1 needs five integer
 * bits where K0 needs one (see qformat.h -- this is the sharpest example in the repo
 * of why per-quantity formats are not a stylistic preference). */
typedef struct {
    int ang;    /* angle, innovation            */
    int rate;   /* gyro rate                    */
    int bias;   /* bias state                   */
    int dt;     /* sample interval              */
    int cov;    /* P, Q, R, S                   */
    int gain;   /* K0, dimensionless, in [0,1)  */
    int gain1;  /* K1, 1/s, |K1| <= 25          */
} kf_frac_t;

typedef struct {
    /* state */
    int32_t angle;      /* frac.ang   rad     */
    int32_t bias;       /* frac.bias  rad/s   */
    int32_t p[4];       /* frac.cov, row-major: P00 P01 P10 P11 */

    /* constants, all in frac.cov except dt */
    int32_t dt;         /* frac.dt    s       */
    int32_t q00, q01, q11;
    int32_t r;

    kf_frac_t frac;

    /* diagnostics, not used by the arithmetic */
    int32_t k0;         /* frac.gain,  last K0 */
    int32_t k1;         /* frac.gain1, last K1 */
    int32_t s;          /* frac.cov,  last innovation covariance */
    int32_t innovation; /* frac.ang */
} kf_t;

/* Default fractional-bit set (the nominal design in qformat.h). */
kf_frac_t kf_frac_default(void);

/* Every quantity a run needs, in SI units. Converted to fixed point once, at init;
 * on device these are compile-time constants and this function is a table load. */
typedef struct {
    double dt;          /* s                */
    double sigma_gyro;  /* rad/s/sqrt(Hz)   */
    double sigma_bias;  /* rad/s^1.5        */
    double sigma_acc;   /* rad              */
    double p0_angle;    /* rad^2            */
    double p0_bias;     /* rad^2/s^2        */
    double angle0;      /* rad, initial estimate */
    double bias0;       /* rad/s            */
} kf_params_t;

kf_params_t kf_params_default(void);

void kf_init(kf_t *kf, const kf_params_t *par, kf_frac_t frac);

/* One predict with a gyro sample in frac.rate format. */
void kf_predict(kf_t *kf, int32_t gyro_rate);

/* One measurement update with an accelerometer angle in frac.ang format. */
void kf_update(kf_t *kf, int32_t accel_angle, kf_variant_t variant);

/* Split out so the two forms can be read side by side. kf_update dispatches. */
void kf_update_naive(kf_t *kf, int32_t accel_angle);
void kf_update_joseph(kf_t *kf, int32_t accel_angle);

#endif /* KALMAN_FIXED_H */
