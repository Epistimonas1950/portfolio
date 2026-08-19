/* qformat_selftest.c -- the fixed-point arithmetic, checked against its contract.
 *
 * Printed as one `name=PASS|FAIL detail` line per check so tests/test_qformat.py can
 * shell out and assert on it without a C test framework. Exit status is the number of
 * failures, so `make host && ./build/qtest` is also usable on its own.
 *
 * The check that matters here is saturation. An int32 covariance entry that wraps from
 * +1.9 rad^2 to -1.9 rad^2 is still a perfectly ordinary int32; nothing downstream can
 * tell, the filter carries on with a negative variance, and the estimate is wrong in a
 * way that looks like a modelling problem for the rest of the project's life. Every
 * operation in qformat.h therefore goes through int64 and comes back clamped.
 */

#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>

#include "qformat.h"

static int failures = 0;

static void check(const char *name, int ok, const char *fmt, ...)
{
    va_list ap;
    printf("%s=%s ", name, ok ? "PASS" : "FAIL");
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    putchar('\n');
    if (!ok) failures++;
}

int main(void)
{
    const int F = 30;
    int i;

    /* 1. Round-trip. Q(x) must be within half an ulp of x for every format in use. */
    {
        const int fracs[] = { Q_ANG_FRAC, Q_RATE_FRAC, Q_BIAS_FRAC, Q_COV_FRAC,
                              Q_GAIN_FRAC };
        double worst = 0.0;
        int worst_frac = 0;
        for (i = 0; i < 5; i++) {
            int f = fracs[i];
            double half = Q_HALF_ULP(f);
            double x;
            for (x = -0.9; x < 0.9; x += 0.00037) {
                double back = q_to_double(q_from_double(x, f), f);
                double e = fabs(back - x) / half;
                if (e > worst) { worst = e; worst_frac = f; }
            }
        }
        check("roundtrip_within_half_ulp", worst <= 1.0 + 1e-9,
              "worst=%.6f half-ulps (frac=%d)", worst, worst_frac);
    }

    /* 2. Saturating add: +max plus +max must clamp, not wrap to a negative. */
    {
        int32_t r;
        q_sat_reset();
        r = q_add(Q_MAX32, Q_MAX32);
        check("add_saturates_high", r == Q_MAX32 && q_sat_events() == 1,
              "got %" PRId32 " events=%lu", r, q_sat_events());
        q_sat_reset();
        r = q_add(Q_MIN32, Q_MIN32);
        check("add_saturates_low", r == Q_MIN32 && q_sat_events() == 1,
              "got %" PRId32 " events=%lu", r, q_sat_events());
        q_sat_reset();
        r = q_sub(Q_MIN32, Q_MAX32);
        check("sub_saturates_low", r == Q_MIN32 && q_sat_events() == 1,
              "got %" PRId32 " events=%lu", r, q_sat_events());
    }

    /* 3. Saturating multiply. 1.9 * 1.9 = 3.61 does not fit Q1.30 (max 2). It must
     *    clamp to +max, and it must NOT come back as a plausible small number. */
    {
        int32_t a = q_from_double(1.9, F), r;
        q_sat_reset();
        r = q_mul(a, a, F);
        check("mul_saturates_positive", r == Q_MAX32 && q_sat_events() == 1,
              "1.9*1.9 in Q1.%d -> %.6f events=%lu", F, q_to_double(r, F),
              q_sat_events());
        q_sat_reset();
        r = q_mul(a, q_from_double(-1.9, F), F);
        check("mul_saturates_negative", r == Q_MIN32 && q_sat_events() == 1,
              "1.9*-1.9 -> %.6f events=%lu", q_to_double(r, F), q_sat_events());
    }

    /* 4. No spurious saturation on ordinary work: the whole covariance range. */
    {
        double x;
        int clean = 1;
        q_sat_reset();
        for (x = -1.4; x < 1.4; x += 0.0011) {
            int32_t p = q_from_double(x, F);
            (void)q_mul(p, q_from_double(0.5, F), F);
            (void)q_add(p, q_from_double(0.3, F));
        }
        clean = (q_sat_events() == 0);
        check("no_spurious_saturation", clean, "events=%lu", q_sat_events());
    }

    /* 5. Rounding is to nearest, not truncation toward -inf. Truncation is a biased
     *    -ulp/2 per operation, and a bias inside a recursion is drift -- the exact
     *    quantity this project budgets. Check the bias over many products is small. */
    {
        double bias = 0.0;
        int n = 0;
        double x;
        for (x = -0.9; x < 0.9; x += 0.0007) {
            double y = 0.3141592653589793;
            int32_t p = q_mul(q_from_double(x, F), q_from_double(y, F), F);
            bias += q_to_double(p, F) - x * y;
            n++;
        }
        bias /= n;
        check("rounding_is_unbiased", fabs(bias) < 0.05 * Q_HALF_ULP(F),
              "mean error = %.4g ulp/2 (%d products)", bias / Q_HALF_ULP(F), n);
    }

    /* 6. The Kalman gain divide. K0 = P00/(P00+R) must land in [0,1) for the whole
     *    covariance range, and must be accurate to half an ulp of the gain format. */
    {
        double worst = 0.0, r_val = 1e-4;
        double p;
        int in_range = 1;
        int32_t rq = q_from_double(r_val, F);
        for (p = 1e-7; p < 1.5; p *= 1.07) {
            int32_t pq = q_from_double(p, F);
            int32_t sq = q_add(pq, rq);
            int32_t k  = q_div_f(pq, sq, F, F, Q_GAIN_FRAC);
            double got = q_to_double(k, Q_GAIN_FRAC);
            double want = q_to_double(pq, F) / q_to_double(sq, F);
            double e = fabs(got - want) / Q_HALF_ULP(Q_GAIN_FRAC);
            if (e > worst) worst = e;
            if (!(got >= 0.0 && got < 1.0)) in_range = 0;
        }
        check("gain_divide_accurate", worst <= 1.0 + 1e-6, "worst=%.4f half-ulps", worst);
        check("gain_in_unit_interval", in_range, "K0 stayed in [0,1)");
    }

    /* 7. Division by zero is contained: saturate and count, never trap or wrap. */
    {
        int32_t r;
        q_sat_reset();
        r = q_div_f(q_from_double(0.5, F), 0, F, F, F);
        check("div_by_zero_contained", r == Q_MAX32 && q_sat_events() == 1,
              "got %" PRId32 " events=%lu", r, q_sat_events());
    }

    /* 8. Cross-format rescale: widening is exact, narrowing is within half an ulp of
     *    the destination. Q1.30 bias -> Q4.27 rate is the one the predict step does. */
    {
        double worst = 0.0, x;
        for (x = -0.49; x < 0.49; x += 0.00031) {
            int32_t b = q_from_double(x, Q_BIAS_FRAC);
            int32_t w = q_rescale(b, Q_BIAS_FRAC, Q_RATE_FRAC);
            double e = fabs(q_to_double(w, Q_RATE_FRAC) - q_to_double(b, Q_BIAS_FRAC));
            if (e / Q_HALF_ULP(Q_RATE_FRAC) > worst) worst = e / Q_HALF_ULP(Q_RATE_FRAC);
        }
        check("rescale_narrowing_half_ulp", worst <= 1.0 + 1e-9,
              "worst=%.6f half-ulps", worst);
    }

    printf("failures=%d\n", failures);
    return failures;
}
