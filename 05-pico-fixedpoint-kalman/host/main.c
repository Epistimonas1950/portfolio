/* main.c -- replay an IMU trace through the fixed-point filter on the HOST.
 *
 * This is the piece that makes the project reproducible without a board. The filter
 * in firmware/ is portable C11 with no SDK dependency and no floating point, so plain
 * gcc compiles the identical arithmetic that an arm-none-eabi build would put on an
 * RP2040: same int32 storage, same int64 intermediates, same shifts, same rounding.
 * What the host cannot tell you is cycles and RAM; what it can tell you is every
 * number in the error budget, which is the part this project is about.
 *
 * Doubles appear in exactly two places here and in neither of them does the filter
 * see them: converting the CSV columns to fixed point on the way in, and computing
 * covariance diagnostics (determinant, symmetry residual, minimum eigenvalue) on the
 * way out. Those diagnostics are instrumentation. On device you would not compute an
 * eigenvalue; you would log the four int32s and do this offline, which is exactly
 * what the per-step CSV lets you do.
 *
 * Usage:
 *   kfhost --trace data/imu_capture.csv --variant joseph [options]
 *     --out FILE        per-step CSV (default: none, summary only)
 *     --decimate N      write every Nth row of --out (default 1)
 *     --gain-bits N     fractional bits for the Kalman gain  (default 30, max 30)
 *     --cov-bits N      fractional bits for P, Q, R, S       (default 30)
 *     --ang-bits N      fractional bits for angles           (default 28)
 *     --rate-bits N     fractional bits for the gyro rate    (default 27)
 *     --bias-bits N     fractional bits for the bias state   (default 30)
 *     --dt-bits N       fractional bits for dt               (default 31)
 *     --global-format N set all six to N (the "one format for everything" build)
 *     --dt SECONDS      override the sample interval (default: inferred from trace)
 *     --params          print the compiled-in filter parameters and exit
 *
 * A summary block of key=value lines always goes to stdout; reference/compare.py and
 * the tests parse it.
 */

#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kalman_fixed.h"

#define MAX_SAMPLES 400000

typedef struct {
    double t, gyro, accel, true_angle, true_bias;
} sample_t;

static sample_t g_samples[MAX_SAMPLES];
static int g_n = 0;

static int read_trace(const char *path)
{
    char line[512];
    FILE *fh = fopen(path, "r");
    if (!fh) { fprintf(stderr, "cannot open trace '%s'\n", path); return -1; }
    if (!fgets(line, sizeof line, fh)) { fclose(fh); return -1; }   /* header */
    while (fgets(line, sizeof line, fh)) {
        sample_t s;
        if (line[0] == '\n' || line[0] == '#') continue;
        if (sscanf(line, "%lf,%lf,%lf,%lf,%lf",
                   &s.t, &s.gyro, &s.accel, &s.true_angle, &s.true_bias) != 5) {
            fprintf(stderr, "malformed trace row: %s", line);
            fclose(fh);
            return -1;
        }
        if (g_n >= MAX_SAMPLES) { fprintf(stderr, "trace too long\n"); fclose(fh); return -1; }
        g_samples[g_n++] = s;
    }
    fclose(fh);
    return g_n;
}

/* Diagnostics on the symmetric part of P. A non-symmetric P has no eigen-decomposition
 * that means anything for a covariance, so the honest reading is: the symmetric part
 * carries the quadratic form (v^T P v depends only on it), and the residual
 * ||P - P^T|| measures how far the object has stopped being a covariance at all.
 * Both are reported; neither substitutes for the other. */
typedef struct { double sym, det, lambda_min; } diag_t;

static diag_t diagnose(const kf_t *kf)
{
    const int fc = kf->frac.cov;
    double p00 = q_to_double(kf->p[0], fc);
    double p01 = q_to_double(kf->p[1], fc);
    double p10 = q_to_double(kf->p[2], fc);
    double p11 = q_to_double(kf->p[3], fc);
    double s01 = 0.5 * (p01 + p10);
    double tr  = p00 + p11;
    double disc = (p00 - p11) * (p00 - p11) + 4.0 * s01 * s01;
    diag_t d;
    d.sym = fabs(p01 - p10);
    d.det = p00 * p11 - s01 * s01;
    d.lambda_min = 0.5 * (tr - sqrt(disc < 0.0 ? 0.0 : disc));
    return d;
}

static void usage(void)
{
    fputs("usage: kfhost --trace FILE --variant naive|joseph [--out FILE] "
          "[--decimate N] [--gain-bits N] [--cov-bits N] [--ang-bits N] "
          "[--rate-bits N] [--bias-bits N] [--dt-bits N] [--global-format N] "
          "[--dt SECONDS] [--params]\n", stderr);
}

int main(int argc, char **argv)
{
    const char *trace = NULL, *outpath = NULL;
    kf_variant_t variant = KF_JOSEPH;
    kf_frac_t frac = kf_frac_default();
    kf_params_t par = kf_params_default();
    double dt_override = -1.0;
    int decimate = 1, i;
    kf_t kf;
    FILE *out = NULL;

    for (i = 1; i < argc; i++) {
        const char *a = argv[i];
        #define NEXT() (i + 1 < argc ? argv[++i] : (usage(), exit(2), ""))
        if      (!strcmp(a, "--trace"))        trace = NEXT();
        else if (!strcmp(a, "--out"))          outpath = NEXT();
        else if (!strcmp(a, "--decimate"))     decimate = atoi(NEXT());
        else if (!strcmp(a, "--gain-bits")) {
            /* K1 keeps its four extra integer bits: the sweep varies precision, not
             * the range analysis. Setting them equal would conflate a bit-depth
             * result with a saturation artefact. */
            frac.gain  = atoi(NEXT());
            frac.gain1 = frac.gain - (Q_GAIN_FRAC - Q_GAIN1_FRAC);
        }
        else if (!strcmp(a, "--cov-bits"))     frac.cov  = atoi(NEXT());
        else if (!strcmp(a, "--ang-bits"))     frac.ang  = atoi(NEXT());
        else if (!strcmp(a, "--rate-bits"))    frac.rate = atoi(NEXT());
        else if (!strcmp(a, "--bias-bits"))    frac.bias = atoi(NEXT());
        else if (!strcmp(a, "--dt-bits"))      frac.dt   = atoi(NEXT());
        else if (!strcmp(a, "--dt"))           dt_override = atof(NEXT());
        else if (!strcmp(a, "--global-format")) {
            int n = atoi(NEXT());
            frac.ang = frac.rate = frac.bias = frac.dt = frac.cov = frac.gain = n;
            frac.gain1 = n;
        }
        else if (!strcmp(a, "--variant")) {
            const char *v = NEXT();
            if      (!strcmp(v, "naive"))  variant = KF_NAIVE;
            else if (!strcmp(v, "joseph")) variant = KF_JOSEPH;
            else { fprintf(stderr, "unknown variant '%s'\n", v); return 2; }
        }
        else if (!strcmp(a, "--params")) {
            printf("dt=%.10g\nsigma_gyro=%.10g\nsigma_bias=%.10g\nsigma_acc=%.10g\n"
                   "p0_angle=%.10g\np0_bias=%.10g\n",
                   par.dt, par.sigma_gyro, par.sigma_bias, par.sigma_acc,
                   par.p0_angle, par.p0_bias);
            printf("q_ang_frac=%d\nq_rate_frac=%d\nq_bias_frac=%d\nq_dt_frac=%d\n"
                   "q_cov_frac=%d\nq_gain_frac=%d\nq_gain1_frac=%d\n"
                   "q_global_frac=%d\n",
                   Q_ANG_FRAC, Q_RATE_FRAC, Q_BIAS_FRAC, Q_DT_FRAC,
                   Q_COV_FRAC, Q_GAIN_FRAC, Q_GAIN1_FRAC, Q_GLOBAL_FRAC);
            return 0;
        }
        else { usage(); return 2; }
        #undef NEXT
    }

    if (!trace) { usage(); return 2; }
    if (frac.gain > 30) {
        fprintf(stderr, "--gain-bits must be <= 30: 1.0 must be representable in the "
                        "gain format because the Joseph form needs 1 - K0.\n");
        return 2;
    }
    if (decimate < 1) decimate = 1;

    if (read_trace(trace) < 2) return 1;
    par.dt = dt_override > 0.0 ? dt_override : g_samples[1].t - g_samples[0].t;

    q_sat_reset();
    kf_init(&kf, &par, frac);

    if (outpath) {
        out = fopen(outpath, "w");
        if (!out) { fprintf(stderr, "cannot write '%s'\n", outpath); return 1; }
        fputs("k,t,angle,bias,p00,p01,p10,p11,sym_resid,det,lambda_min,k0,k1,"
              "true_angle,true_bias\n", out);
    }

    {
        double min_lambda = 1e300, max_sym = 0.0, max_abs_err = 0.0, sse = 0.0;
        double final_err = 0.0, angle = 0.0, bias = 0.0;
        long first_neg = -1, first_asym = -1, n_neg = 0;
        int k;

        for (k = 0; k < g_n; k++) {
            diag_t d;
            kf_predict(&kf, q_from_double(g_samples[k].gyro, frac.rate));
            kf_update(&kf, q_from_double(g_samples[k].accel, frac.ang), variant);
            d = diagnose(&kf);

            angle = q_to_double(kf.angle, frac.ang);
            bias  = q_to_double(kf.bias,  frac.bias);
            final_err = angle - g_samples[k].true_angle;
            if (fabs(final_err) > max_abs_err) max_abs_err = fabs(final_err);
            sse += final_err * final_err;
            if (d.lambda_min < min_lambda) min_lambda = d.lambda_min;
            if (d.sym > max_sym) max_sym = d.sym;
            if (d.lambda_min < 0.0) { n_neg++; if (first_neg < 0) first_neg = k; }
            if (d.sym > 0.0 && first_asym < 0) first_asym = k;

            if (out && (k % decimate == 0)) {
                fprintf(out,
                        "%d,%.6f,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.6g,%.6g,%.6g,"
                        "%.10g,%.10g,%.10g,%.10g\n",
                        k, g_samples[k].t, angle, bias,
                        q_to_double(kf.p[0], frac.cov), q_to_double(kf.p[1], frac.cov),
                        q_to_double(kf.p[2], frac.cov), q_to_double(kf.p[3], frac.cov),
                        d.sym, d.det, d.lambda_min,
                        q_to_double(kf.k0, frac.gain), q_to_double(kf.k1, frac.gain1),
                        g_samples[k].true_angle, g_samples[k].true_bias);
            }
        }
        if (out) fclose(out);

        printf("variant=%s\n", variant == KF_JOSEPH ? "joseph" : "naive");
        printf("samples=%d\ndt=%.10g\n", g_n, par.dt);
        printf("ang_bits=%d\nrate_bits=%d\nbias_bits=%d\ndt_bits=%d\n"
               "cov_bits=%d\ngain_bits=%d\ngain1_bits=%d\n",
               frac.ang, frac.rate, frac.bias, frac.dt, frac.cov, frac.gain,
               frac.gain1);
        printf("min_lambda=%.12g\n", min_lambda);
        printf("max_sym_resid=%.12g\n", max_sym);
        printf("first_negative_step=%ld\n", first_neg);
        printf("first_asymmetric_step=%ld\n", first_asym);
        printf("n_negative_steps=%ld\n", n_neg);
        printf("final_angle=%.12g\nfinal_bias=%.12g\n", angle, bias);
        printf("final_angle_error=%.12g\n", final_err);
        printf("max_abs_angle_error=%.12g\n", max_abs_err);
        printf("rms_angle_error=%.12g\n", sqrt(sse / (double)g_n));
        printf("final_p00=%.12g\nfinal_p11=%.12g\n",
               q_to_double(kf.p[0], frac.cov), q_to_double(kf.p[3], frac.cov));
        printf("saturation_events=%lu\n", q_sat_events());
    }
    return 0;
}
