/* qformat.h -- every fixed-point format decision in this filter, with its arithmetic.
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * The RP2040 has no floating-point unit. Software float costs ~50-100 cycles for a
 * multiply where a 32x32->64 integer multiply costs 1-4, so an estimator that has to
 * close a control loop is written in fixed point. Fixed point is not "float with a
 * shift": every quantity carries a *statically chosen* scale, and choosing one global
 * scale for the whole filter is the classic beginner's mistake. The quantities here
 * span eleven orders of magnitude --
 *
 *     Kalman gain K1 up to  1.5e+1  1/s
 *     gyro rate      up to  8.7e+0  rad/s
 *     covariance     down to 4.5e-8 rad^2   (the per-step process-noise increment)
 *
 * -- so a single format wide enough for the rates throws away, by construction, the
 * bits the covariance needs. Section 3 below quantifies that: 3 bits, and 3 bits is
 * exactly the margin that decides whether the naive covariance update survives.
 *
 * NOTATION
 * --------
 * Qm.n is a signed two's-complement int32 with m integer bits, n fractional bits and
 * one sign bit, so m + n = 31. The stored integer i represents the real number
 * i * 2^-n. Consequently
 *
 *     representable range   [-2^m, 2^m - 2^-n]
 *     quantization step     ulp = 2^-n
 *     representation error  |x - Q(x)| <= 2^-(n+1)      (round to nearest)
 *
 * That last line is the unit of the error budget in reference/error_budget.py. Every
 * arithmetic routine below rounds to nearest, so every operation contributes at most
 * one half-ulp of the *destination* format -- which is what makes the budget a bound
 * rather than a guess.
 *
 * 1. THE FORMATS
 * --------------
 * Ranges are hard bounds derived from the sensor configuration and the filter's own
 * algebra, not from watching a trace. Where a bound is only a design assumption it
 * says so, and the saturating arithmetic below is what catches a violated assumption.
 *
 *  quantity          bound                     format   step 2^-n    half-ulp
 *  ---------------------------------------------------------------------------------
 *  angle theta       |theta| <= pi rad          Q3.28   3.73e-09     1.86e-09 rad
 *  innovation y      |y|     <= 2*pi rad        Q3.28   3.73e-09     1.86e-09 rad
 *  gyro rate omega   |omega| <= 8.73 rad/s      Q4.27   7.45e-09     3.73e-09 rad/s
 *  gyro bias b       |b|     <= 0.5 rad/s       Q1.30   9.31e-10     4.66e-10 rad/s
 *  sample step dt    0 < dt  <= 0.5 s           Q0.31   4.66e-10     2.33e-10 s
 *  covariance P      0 <= P00 <= 1 rad^2        Q1.30   9.31e-10     4.66e-10 rad^2
 *  innovation cov S  R <= S <= 1 + R            Q1.30   9.31e-10     4.66e-10 rad^2
 *  Kalman gain K0    0 <= K0 < 1 (dimensionless) Q1.30  9.31e-10     4.66e-10
 *  Kalman gain K1    |K1| <= 25 1/s             Q5.26   1.49e-08     7.45e-09 1/s
 *
 * Justifications, one line each -- these are the decisions an interviewer should be
 * able to interrogate:
 *
 *  theta   A gravity-referenced pitch/roll angle is in [-pi, pi]. Q3.28 spans +-8,
 *          which is 2.5x the bound; the spare bit absorbs an unwrapped innovation
 *          without saturating. Q4.27 would waste a bit, Q2.29 would clip at 4 rad and
 *          a clipped angle is a wrong angle that never announces itself.
 *  omega   The IMU is configured for +-500 deg/s = 8.727 rad/s full scale. Q4.27 is
 *          the tightest format that holds it. At the part's +-2000 deg/s setting the
 *          rate needs Q6.25 and everything downstream loses two more bits -- the
 *          full-scale setting is a numerical decision, not just a sensor one.
 *  b       MEMS gyro bias including temperature drift is well under 0.5 rad/s
 *          (29 deg/s). Q1.30 gives 4x headroom. This is the one *assumption* in the
 *          table; q_sat_events() exists so a violation is visible rather than silent.
 *  dt      dt multiplies the state in every predict, so it gets the finest format an
 *          int32 offers. Q0.31 caps dt at 1 s, far above the 5 ms this filter uses.
 *  P       P00 is an angle variance in rad^2. It starts at the diffuse prior 1.0 and
 *          decays to ~2.5e-6; P11 is a bias variance in rad^2/s^2 and P01 a covariance
 *          in rad^2/s. All three are bounded by 1 in magnitude after the first
 *          predict, so Q1.30 with 2x headroom serves all three. The binding constraint
 *          is at the *bottom* of the range, not the top: see section 2.
 *  K0      K0 = P00/(P00+R) lies in [0,1) identically, for any P00 >= 0 and R > 0.
 *          Q1.30 is the finest format that still represents 1.0 exactly, and 1 - K0
 *          is what the Joseph form needs, so Q1.30 it is. This is the one bound in
 *          the table that is an identity rather than an estimate.
 *  K1      THE ONE I GOT WRONG FIRST, and the reason q_sat_events() exists.
 *          K1 = P10/S has units 1/s and my first bound was |K1| <= sqrt(P11/P00) by
 *          Cauchy-Schwarz, which I read off as "about 0.5" -- but that bound blows up
 *          as P00 -> 0 and says nothing. The correct one maximises over the reachable
 *          set: with |P10| <= sqrt(P00 P11) and S = P00 + R >= 2 sqrt(P00 R),
 *
 *              |K1| <= sqrt(P00 P11) / (P00 + R) <= sqrt(P11_max / R) / 2
 *                    = sqrt(0.25 / 1e-4) / 2 = 25 1/s
 *
 *          attained near P00 = R, i.e. in the middle of the diffuse-prior transient
 *          when the angle has just become as well known as one measurement. The
 *          float64 reference peaks at |K1| = 15.4 at step 4, so the bound is tight to
 *          within 1.6x and Q5.26 (range +-32) is the right format. Built with Q1.30
 *          instead, the gain saturates at 2 on 106 of the 12000 steps -- the counter
 *          found it, which is the entire argument for having a counter. K0 and K1 are
 *          both "the Kalman gain" and they need formats four bits apart; if you were
 *          looking for the single cleanest example of why one global format is wrong,
 *          this is it.
 *
 * 2. THE BINDING CONSTRAINT IS THE SMALLEST COVARIANCE INCREMENT, NOT THE LARGEST
 * ------------------------------------------------------------------------------
 * P has to hold the diffuse prior P00(0) = 1 rad^2 AND resolve the per-step process
 * noise Q11 = sigma_b^2 * dt = 4.5e-8 rad^2/s^2. The required dynamic range is
 *
 *     log2( 1.0 / 4.5e-8 ) = 24.4 bits
 *
 * and if the process noise rounds to zero the filter stops believing it can be wrong:
 * P collapses, the gain goes to zero, and the estimate freezes -- a failure that looks
 * like excellent convergence right up to the moment it is catastrophically wrong. The
 * requirement is therefore
 *
 *     2^-n <= Q11 / 2      =>     n >= log2(2 / 4.5e-8) = 25.4 fractional bits
 *
 * Q1.30 clears it with 4.6 bits of margin. analysis/cov_bits_sweep.py measures where
 * it actually breaks; the prediction is stated in reference/error_budget.py first.
 *
 * 3. WHAT ONE GLOBAL FORMAT WOULD COST
 * ------------------------------------
 * A single format for the whole filter must hold the widest range, the gyro rate, so
 * it would be Q4.27. Every covariance entry then loses 3 fractional bits: ulp goes
 * from 9.31e-10 to 7.45e-9, the process-noise margin from 4.6 bits to 1.6, and the
 * gain from 30 fractional bits to 27. analysis/cov_bits_sweep.py and
 * analysis/gain_bits_sweep.py both run that configuration, so the cost of the
 * beginner's mistake is a measured number in results/, not a warning in a comment.
 *
 * 4. ARITHMETIC
 * -------------
 * All products go through int64 and come back saturated, never wrapped. Wrapping is
 * the wrong failure mode for an estimator: a covariance that wraps from +2 to -2 is
 * still a plausible-looking int32 and the filter carries on with a negative variance.
 * Saturation is also wrong, but it is wrong in a bounded, detectable direction, and
 * q_sat_events() counts it so a test can assert it never happened on a good run and
 * did happen when provoked.
 *
 * Rounding is round-half-up: add half a destination ulp before the arithmetic shift.
 * An arithmetic right shift alone truncates toward -infinity, which is a *biased*
 * error of -ulp/2 per operation; in a recursion that bias integrates into drift, which
 * is precisely the quantity this project is budgeting. Round-half-up leaves a residual
 * bias of at most 2^-(n+1) only on exact ties, which are measure-zero here.
 */

#ifndef QFORMAT_H
#define QFORMAT_H

#include <stdint.h>

/* ---- nominal formats: fractional bit counts ---------------------------------- */
#define Q_ANG_FRAC   28   /* Q3.28  angles and innovations, rad                   */
#define Q_RATE_FRAC  27   /* Q4.27  gyro rate, rad/s                              */
#define Q_BIAS_FRAC  30   /* Q1.30  gyro bias state, rad/s                        */
#define Q_DT_FRAC    31   /* Q0.31  sample interval, s                            */
#define Q_COV_FRAC   30   /* Q1.30  every entry of P, plus Q, R and S             */
#define Q_GAIN_FRAC  30   /* Q1.30  Kalman gain K0, dimensionless, in [0,1)       */
#define Q_GAIN1_FRAC 26   /* Q5.26  Kalman gain K1, 1/s, |K1| <= 25 -- see above  */

/* The "one global format" comparison of section 3: the width the gyro rate forces. */
#define Q_GLOBAL_FRAC 27

/* Half-ulp of a format, as a double. Host-side only -- this is the quantity the
 * error budget propagates, and it never appears in the filter's own arithmetic. */
#define Q_HALF_ULP(frac) (1.0 / (double)(1LL << ((frac) + 1)))

#define Q_MAX32  ((int32_t)0x7FFFFFFF)
#define Q_MIN32  ((int32_t)0x80000000)

/* Count of saturation events since the last q_sat_reset(). A saturating estimator is
 * not a working estimator; this counter is how a test finds out. */
extern unsigned long q_sat_events_count;
void q_sat_reset(void);
unsigned long q_sat_events(void);

/* Clamp a 64-bit intermediate into int32, counting the clamp. */
static inline int32_t q_sat(int64_t v)
{
    if (v > (int64_t)Q_MAX32) { q_sat_events_count++; return Q_MAX32; }
    if (v < (int64_t)Q_MIN32) { q_sat_events_count++; return Q_MIN32; }
    return (int32_t)v;
}

/* Same-format add and subtract. int64 intermediate so the overflow is detected rather
 * than being undefined behaviour on signed int32 overflow. */
static inline int32_t q_add(int32_t a, int32_t b) { return q_sat((int64_t)a + b); }
static inline int32_t q_sub(int32_t a, int32_t b) { return q_sat((int64_t)a - b); }
static inline int32_t q_neg(int32_t a)            { return q_sat(-(int64_t)a); }

/* Round-half-up arithmetic right shift of a 64-bit intermediate.
 * shift may be 0 (no rounding needed) or negative (a left shift, i.e. the destination
 * format is finer than the product -- exact, no rounding, but it can overflow, which
 * is why it still goes through q_sat). */
static inline int64_t q_round_shift(int64_t v, int shift)
{
    if (shift > 0) {
        if (shift >= 63) return 0;                      /* everything rounds away */
        return (v + ((int64_t)1 << (shift - 1))) >> shift;
    }
    if (shift < 0) {
        int s = -shift;
        /* A left shift past the int64 headroom would wrap silently, which is the one
         * thing this file exists to prevent. Clamp to something q_sat will saturate. */
        if (s >= 32 || v > (((int64_t)1 << 62) >> s) || v < -(((int64_t)1 << 62) >> s))
            return v >= 0 ? (int64_t)1 << 62 : -((int64_t)1 << 62);
        return v << s;
    }
    return v;
}

/* General multiply. `a` has fa fractional bits, `b` has fb, the result has fr:
 * the raw product has fa + fb, so the shift is (fa + fb - fr). */
static inline int32_t q_mul_f(int32_t a, int32_t b, int fa, int fb, int fr)
{
    return q_sat(q_round_shift((int64_t)a * (int64_t)b, fa + fb - fr));
}

/* Multiply when both operands and the result share one format. */
static inline int32_t q_mul(int32_t a, int32_t b, int frac)
{
    return q_mul_f(a, b, frac, frac, frac);
}

/* General divide. Numerator is promoted before the shift so no precision is lost
 * ahead of the division. Division by zero is impossible in this filter (S >= R > 0)
 * but an estimator that has already gone wrong can get there, so it is handled: the
 * result saturates in the sign of the numerator and the event is counted. */
static inline int32_t q_div_f(int32_t a, int32_t b, int fa, int fb, int fr)
{
    int64_t num;
    int shift = fr - fa + fb;      /* (a/b) has fa - fb frac bits; we want fr */
    if (b == 0) { q_sat_events_count++; return a >= 0 ? Q_MAX32 : Q_MIN32; }
    num = (int64_t)a;
    if (shift >= 0) {
        /* Keep the shift inside int64: |a| < 2^31, so shift <= 32 is safe. */
        if (shift > 31) { q_sat_events_count++; return a >= 0 ? Q_MAX32 : Q_MIN32; }
        num <<= shift;
    } else {
        num = q_round_shift(num, -shift);
    }
    /* C integer division truncates toward zero; that is a biased error of up to a
     * whole ulp and it would break the half-ulp contract the error budget is built
     * on. Round to nearest by adding half the divisor with the numerator's sign. */
    {
        int64_t den = (int64_t)b, half;
        if (den < 0) { den = -den; num = -num; }
        half = den >> 1;
        num = (num >= 0) ? (num + half) : (num - half);
        return q_sat(num / den);
    }
}

/* Move a value between formats. Widening (to > from) is exact; narrowing rounds. */
static inline int32_t q_rescale(int32_t v, int from_frac, int to_frac)
{
    return q_sat(q_round_shift((int64_t)v, from_frac - to_frac));
}

/* ---- host-side conversion ---------------------------------------------------------
 * Used to load compile-time constants and to log results. On the RP2040 these are
 * evaluated by the compiler, not at run time -- there is no double arithmetic in the
 * filter itself. Both are declared here rather than hidden so that the one place
 * doubles touch this code is obvious. */
int32_t q_from_double(double x, int frac);
double  q_to_double(int32_t v, int frac);

#endif /* QFORMAT_H */
