/* kalman_fixed.c -- init, predict, and the TEXTBOOK covariance update.
 *
 * The naive update in this file is expected to fail at coarse gain precision. It is
 * here on purpose: it is what almost every embedded Kalman filter on the internet
 * does, and the point of the project is to show exactly where it stops working and to
 * have predicted that point in advance. kalman_joseph.c is the fix.
 *
 * The mathematics is documented in kalman_fixed.h; this file is the arithmetic.
 */

#include "kalman_fixed.h"

kf_frac_t kf_frac_default(void)
{
    kf_frac_t f;
    f.ang  = Q_ANG_FRAC;
    f.rate = Q_RATE_FRAC;
    f.bias = Q_BIAS_FRAC;
    f.dt   = Q_DT_FRAC;
    f.cov  = Q_COV_FRAC;
    f.gain = Q_GAIN_FRAC;
    f.gain1 = Q_GAIN1_FRAC;
    return f;
}

kf_params_t kf_params_default(void)
{
    kf_params_t p;
    p.dt         = KF_DT_DEFAULT;
    p.sigma_gyro = KF_SIGMA_GYRO;
    p.sigma_bias = KF_SIGMA_BIAS;
    p.sigma_acc  = KF_SIGMA_ACC;
    p.p0_angle   = KF_P0_ANGLE;
    p.p0_bias    = KF_P0_BIAS;
    p.angle0     = 0.0;
    p.bias0      = 0.0;
    return p;
}

void kf_init(kf_t *kf, const kf_params_t *par, kf_frac_t frac)
{
    const double dt = par->dt;
    const double sg2 = par->sigma_gyro * par->sigma_gyro;
    const double sb2 = par->sigma_bias * par->sigma_bias;

    kf->frac = frac;
    kf->angle = q_from_double(par->angle0, frac.ang);
    kf->bias  = q_from_double(par->bias0,  frac.bias);
    kf->dt    = q_from_double(dt,          frac.dt);

    /* Van Loan discrete process noise; see kalman_fixed.h. q01 is 0.12 ulp at Q1.30
     * and rounds to zero, which is recorded there rather than hidden here. */
    kf->q00 = q_from_double(sg2 * dt + sb2 * dt * dt * dt / 3.0, frac.cov);
    kf->q01 = q_from_double(-0.5 * sb2 * dt * dt,                frac.cov);
    kf->q11 = q_from_double(sb2 * dt,                            frac.cov);
    kf->r   = q_from_double(par->sigma_acc * par->sigma_acc,     frac.cov);

    kf->p[0] = q_from_double(par->p0_angle, frac.cov);
    kf->p[1] = 0;
    kf->p[2] = 0;
    kf->p[3] = q_from_double(par->p0_bias,  frac.cov);

    kf->k0 = kf->k1 = kf->s = kf->innovation = 0;
}

/* theta <- theta + (omega - b) dt ,   P <- F P F^T + Qd
 *
 * The covariance propagation is done as two congruence stages, A = F P then P = A F^T,
 * rather than by expanding F P F^T into monomials. Two reasons: it is one rounding
 * cheaper per entry, and the off-diagonals then come out of the SAME expression with
 * P01 and P10 exchanged, so a symmetric P stays bit-exactly symmetric through the
 * predict. A predict that manufactured asymmetry would contaminate the very quantity
 * the naive-vs-Joseph comparison measures.
 */
void kf_predict(kf_t *kf, int32_t gyro_rate)
{
    const int fr = kf->frac.rate, fa = kf->frac.ang, fb = kf->frac.bias;
    const int fc = kf->frac.cov,  fd = kf->frac.dt;
    int32_t rate_corr, dtheta;
    int32_t a00, a01, a10, a11;

    /* The bias state lives in a finer format than the rate; narrow it once, rounding,
     * rather than carrying the rate in the bias format and clipping at 2 rad/s. */
    rate_corr = q_sub(gyro_rate, q_rescale(kf->bias, fb, fr));
    dtheta    = q_mul_f(rate_corr, kf->dt, fr, fd, fa);
    kf->angle = q_add(kf->angle, dtheta);
    /* bias is constant under F */

    /* A = F P,  F = [[1, -dt], [0, 1]] */
    a00 = q_sub(kf->p[0], q_mul_f(kf->dt, kf->p[2], fd, fc, fc));
    a01 = q_sub(kf->p[1], q_mul_f(kf->dt, kf->p[3], fd, fc, fc));
    a10 = kf->p[2];
    a11 = kf->p[3];

    /* P- = A F^T + Qd,  F^T = [[1, 0], [-dt, 1]] */
    kf->p[0] = q_add(q_sub(a00, q_mul_f(kf->dt, a01, fd, fc, fc)), kf->q00);
    kf->p[1] = q_add(a01,                                          kf->q01);
    kf->p[2] = q_add(q_sub(a10, q_mul_f(kf->dt, a11, fd, fc, fc)), kf->q01);
    kf->p[3] = q_add(a11,                                          kf->q11);
}

/* Gain and state update, shared by both covariance forms.
 *
 * S = P00 + R is a scalar because H = [1 0] and the measurement is scalar, so the
 * "matrix inverse" is one divide. K0 = P00/S is in [0,1) by construction and K1 =
 * P10/S carries units of 1/s. */
static void kf_gain_and_state(kf_t *kf, int32_t accel_angle)
{
    const int fa = kf->frac.ang, fb = kf->frac.bias;
    const int fc = kf->frac.cov, fg = kf->frac.gain, fg1 = kf->frac.gain1;
    int32_t y;

    y  = q_sub(accel_angle, kf->angle);
    kf->innovation = y;
    kf->s  = q_add(kf->p[0], kf->r);
    kf->k0 = q_div_f(kf->p[0], kf->s, fc, fc, fg);
    kf->k1 = q_div_f(kf->p[2], kf->s, fc, fc, fg1);   /* different format, see header */

    kf->angle = q_add(kf->angle, q_mul_f(kf->k0, y, fg,  fa, fa));
    kf->bias  = q_add(kf->bias,  q_mul_f(kf->k1, y, fg1, fa, fb));
}

/* P+ = (I - K H) P- , written out for H = [1 0]:
 *
 *     P+00 = P00 - K0 P00        P+01 = P01 - K0 P01
 *     P+10 = P10 - K1 P00        P+11 = P11 - K1 P01
 *
 * Note what the off-diagonals do. In exact arithmetic K0 P01 = K1 P00, because both
 * equal P00 P01 / S, so the two entries agree. In fixed point K0 and K1 are separately
 * rounded and the two products are not the same number: the result is asymmetric by
 * roughly |dK| * P00 with |dK| <= 2^-(gain+1), and that residual is what
 * analysis/gain_bits_sweep.py measures against the prediction.
 *
 * P+00 = P00 - K0 P00 is also a cancellation. Its true value is P00 R / S, which
 * during the diffuse-prior transient (P00 = 1 rad^2, R = 1e-4 rad^2) is four orders of
 * magnitude below either operand. An error of 2^-(gain+1) in K0 lands on P+00 scaled
 * by P00, so P+00 goes negative once 2^-(gain+1) P00 > P00 R / S, i.e. once the gain
 * has fewer than log2(S/R) ~ 13.3 fractional bits.
 */
void kf_update_naive(kf_t *kf, int32_t accel_angle)
{
    const int fc = kf->frac.cov, fg = kf->frac.gain, fg1 = kf->frac.gain1;
    int32_t n00, n01, n10, n11;

    kf_gain_and_state(kf, accel_angle);

    n00 = q_sub(kf->p[0], q_mul_f(kf->k0, kf->p[0], fg,  fc, fc));
    n01 = q_sub(kf->p[1], q_mul_f(kf->k0, kf->p[1], fg,  fc, fc));
    n10 = q_sub(kf->p[2], q_mul_f(kf->k1, kf->p[0], fg1, fc, fc));
    n11 = q_sub(kf->p[3], q_mul_f(kf->k1, kf->p[1], fg1, fc, fc));

    kf->p[0] = n00; kf->p[1] = n01; kf->p[2] = n10; kf->p[3] = n11;
}

void kf_update(kf_t *kf, int32_t accel_angle, kf_variant_t variant)
{
    if (variant == KF_JOSEPH) kf_update_joseph(kf, accel_angle);
    else                      kf_update_naive(kf, accel_angle);
}

/* kf_gain_and_state is shared with kalman_joseph.c through this shim rather than
 * being duplicated, so that the ONLY difference between the two variants under test
 * is the covariance update. */
void kf_joseph_gain_and_state(kf_t *kf, int32_t accel_angle);
void kf_joseph_gain_and_state(kf_t *kf, int32_t accel_angle)
{
    kf_gain_and_state(kf, accel_angle);
}
