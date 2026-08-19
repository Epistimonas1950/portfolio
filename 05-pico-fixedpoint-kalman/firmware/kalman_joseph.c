/* kalman_joseph.c -- the Joseph-form covariance update. The fix.
 *
 *     P+ = (I - K H) P- (I - K H)^T + K R K^T
 *
 * With H = [1 0] the closed-loop matrix is lower triangular,
 *
 *     A = I - K H = [ 1-K0   0 ]   =  [ a  0 ]
 *                   [  -K1   1 ]      [ c  1 ]
 *
 * so A P A^T expands to
 *
 *     P+00 = a (a P00)                     + K0 (K0 R)
 *     P+01 = c (a P00) + a P01             + K1 (K0 R)
 *     P+10 = c (a P00) + a P10             + K1 (K0 R)
 *     P+11 = c (c P00) + c P01 + c P10 + P11 + K1 (K1 R)
 *
 * Three properties of this arrangement, each of which is load-bearing:
 *
 * 1. SYMMETRY BY CONSTRUCTION. P+01 and P+10 differ only in that one reads P01 where
 *    the other reads P10. Every other operand, every shift and every rounding is
 *    identical, so on a symmetric input they are bit-identical integers -- not "equal
 *    to within a rounding", identical. ||P - P^T|| is exactly 0 at every precision.
 *    The naive form has no such structure: its (0,1) entry is built from K0 and P01
 *    and its (1,0) entry from K1 and P00.
 *
 * 2. NON-NEGATIVE DIAGONAL. a = 1 - K0 is in [0,1] because K0 = P00/(P00+R) is, so
 *    a*(a*P00) is a product of non-negative numbers and K0*(K0*R) likewise. Round to
 *    nearest of a non-negative exact value is non-negative. P+00 cannot go negative,
 *    at any gain precision. The naive P+00 = P00 - K0 P00 is a subtraction and can.
 *
 * 3. THE SQUARES ARE CHAINED, NEVER FORMED. c*(c*P00), not (c*c)*P00; K1*(K1*R), not
 *    (K1*K1)*R. With K1 ~ 1e-2 the quantity K1*K1 ~ 1e-4 falls below one ulp of the
 *    gain format long before K1 does, and a coarse-gain build that computes it that
 *    way silently deletes the two terms that make the result positive semi-definite
 *    while keeping the indefinite cross term c(P01 + P10). The result is a Joseph
 *    filter that diverges *worse* than the naive one -- an easy and completely
 *    invisible way to get the wrong answer from the right formula.
 *
 * The cost is 9 multiplies against the naive form's 4, plus one subtract for a. On an
 * M0+ at 133 MHz that is roughly 40 extra cycles per update, or 8 us at 200 Hz -- 0.16%
 * of the sample budget. That is the entire price of the guarantee.
 *
 * Joseph's deeper property, and the reason it tolerates a mis-rounded gain: for ANY
 * gain K, optimal or not, A P A^T + K R K^T is the exact covariance of the estimator
 * that uses K. A rounded gain therefore makes the filter suboptimal but leaves the
 * covariance a genuine covariance. The naive form is only equal to it when K is
 * exactly optimal, and its error is first order in the gain error.
 */

#include "kalman_fixed.h"

/* Defined in kalman_fixed.c so that the gain and state update are literally the same
 * code in both variants. */
void kf_joseph_gain_and_state(kf_t *kf, int32_t accel_angle);

void kf_update_joseph(kf_t *kf, int32_t accel_angle)
{
    const int fc = kf->frac.cov, fg = kf->frac.gain, fg1 = kf->frac.gain1;
    const int32_t one = (int32_t)1 << fg;      /* 1.0 in K0's format; needs fg <= 30 */
    int32_t a, c, w, t, v, u, r0, r1, off;
    int32_t n00, n01, n10, n11;

    kf_joseph_gain_and_state(kf, accel_angle);

    a = q_sub(one, kf->k0);          /* a = 1 - K0, in [0,1] */
    c = q_neg(kf->k1);               /* c = -K1              */

    w  = q_mul_f(a,      kf->p[0], fg,  fc, fc);   /* a P00   */
    t  = q_mul_f(c,      w,        fg1, fc, fc);   /* c a P00 */
    v  = q_mul_f(c,      kf->p[0], fg1, fc, fc);   /* c P00   */
    u  = q_mul_f(c,      v,        fg1, fc, fc);   /* c^2 P00, never forming c^2 */
    r0 = q_mul_f(kf->k0, kf->r,    fg,  fc, fc);   /* K0 R    */
    r1 = q_mul_f(kf->k1, kf->r,    fg1, fc, fc);   /* K1 R    */

    n00 = q_add(q_mul_f(a, w, fg, fc, fc),
                q_mul_f(kf->k0, r0, fg, fc, fc));

    /* The shared part of both off-diagonals, computed once so the two entries cannot
     * drift apart through a difference in evaluation order. */
    off = q_add(t, q_mul_f(kf->k1, r0, fg1, fc, fc));
    n01 = q_add(off, q_mul_f(a, kf->p[1], fg, fc, fc));
    n10 = q_add(off, q_mul_f(a, kf->p[2], fg, fc, fc));

    n11 = q_add(u, q_mul_f(c, kf->p[1], fg1, fc, fc));
    n11 = q_add(n11, q_mul_f(c, kf->p[2], fg1, fc, fc));
    n11 = q_add(n11, kf->p[3]);
    n11 = q_add(n11, q_mul_f(kf->k1, r1, fg1, fc, fc));

    kf->p[0] = n00; kf->p[1] = n01; kf->p[2] = n10; kf->p[3] = n11;
}
