/* main.c -- the host driver: stream a CSV through the tracker, write scores and drift.
 *
 * This is the program that would run on the board. It is built here for x86 with plain
 * `gcc -O2 -Wall -Wextra -std=c11 -lm` and no library beyond libm, so everything it
 * prints was actually produced by the C in src/, not by the numpy oracle.
 *
 * Three modes:
 *
 *   track      read a channels-per-column CSV, run the incremental tracker over every
 *              sample, write index,score,drift,flag to an output CSV and a key=value
 *              summary to stdout.
 *   rangefind  run the randomized range finder on the first N columns and report
 *              || A - Q Q^T A ||_F, so the bound in src/rangefinder.h can be checked
 *              against what the C actually achieves.
 *   selftest   internal checks of the linear algebra in src/linalg.c against identities
 *              that must hold exactly (orthogonality, reconstruction, ordering). Exits
 *              non-zero on failure; the unittest suite shells out to it.
 *
 * MEMORY, honestly accounted. The streaming loop is O(m r): at m = 24, r = 4 the
 * tracker state is 800 bytes and the scratch another 900. Two things are not O(1) and
 * are stated rather than hidden: the warm-up block held for the initial randomized SVD
 * (m * warmup scalars, 57 kB at the defaults) and the calibration window of scores
 * (2.4 kB), both of which are freed / finished with before the steady-state loop. No
 * measurement here was taken on a Pi Zero; see STATUS.md.
 */

#include "../src/detect.h"
#include "../src/forget.h"
#include "../src/incsvd.h"
#include "../src/linalg.h"
#include "../src/rangefinder.h"
#include "../src/rank.h"
#include "../src/reorth.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* --- CSV reading ---------------------------------------------------------------- */

typedef struct {
    FILE *fh;
    int ncols;
    long data_start;
    char *line;
    size_t cap;
} csv_t;

static int read_line(csv_t *c)
{
    /* Own line reader rather than getline(), which is POSIX and not C11: the whole
     * point of this build is that it needs nothing but a conforming C compiler. */
    size_t len = 0;
    int ch;
    for (;;) {
        ch = fgetc(c->fh);
        if (ch == EOF) break;
        if (len + 2 > c->cap) {
            size_t ncap = c->cap ? c->cap * 2 : 256;
            char *tmp = realloc(c->line, ncap);
            if (!tmp) return -1;
            c->line = tmp;
            c->cap = ncap;
        }
        if (ch == '\n') break;
        if (ch != '\r') c->line[len++] = (char)ch;
    }
    if (len == 0 && ch == EOF) return 0;
    c->line[len] = '\0';
    return 1;
}

static int csv_open(csv_t *c, const char *path)
{
    memset(c, 0, sizeof(*c));
    c->fh = fopen(path, "rb");
    if (!c->fh) {
        fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    if (read_line(c) != 1) {
        fprintf(stderr, "%s is empty\n", path);
        return -1;
    }
    int cols = 1, has_alpha = 0;
    for (const char *p = c->line; *p; ++p) {
        if (*p == ',') ++cols;
        if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z')) {
            /* 'e' and 'E' appear in exponents, so a header is only assumed when a
             * letter shows up that a float literal cannot contain. */
            if (*p != 'e' && *p != 'E' && *p != 'n' && *p != 'a' && *p != 'i' &&
                *p != 'f' && *p != 'N' && *p != 'A' && *p != 'I' && *p != 'F')
                has_alpha = 1;
        }
    }
    c->ncols = cols;
    c->data_start = has_alpha ? ftell(c->fh) : 0;
    fseek(c->fh, c->data_start, SEEK_SET);
    return 0;
}

static void csv_rewind(csv_t *c) { fseek(c->fh, c->data_start, SEEK_SET); }

static void csv_close(csv_t *c)
{
    if (c->fh) fclose(c->fh);
    free(c->line);
    memset(c, 0, sizeof(*c));
}

/* 1 = a row was read, 0 = end of file, -1 = malformed. */
static int csv_read(csv_t *c, scalar_t *row)
{
    int rc = read_line(c);
    if (rc <= 0) return rc;
    if (c->line[0] == '\0') return 0;
    const char *p = c->line;
    for (int j = 0; j < c->ncols; ++j) {
        char *end = NULL;
        const double v = strtod(p, &end);
        if (end == p) {
            fprintf(stderr, "malformed CSV field %d in: %s\n", j, c->line);
            return -1;
        }
        row[j] = (scalar_t)v;
        p = end;
        while (*p == ',' || *p == ' ') ++p;
    }
    return 1;
}

/* --- options -------------------------------------------------------------------- */

typedef struct {
    const char *input;
    const char *output;
    const char *basis;
    double lambda;
    int reorth;
    double reorth_tol;
    int check_every;
    int rank_max;
    int oversampling;
    int power_iters;
    double energy;
    const char *rank_mode;
    double quantile;
    int warmup;
    int calibration;
    int repeat;
    unsigned long seed;
    int rank;
    int columns;
} opts_t;

static void defaults(opts_t *o)
{
    memset(o, 0, sizeof(*o));
    o->lambda = 1.0;
    o->reorth = 1;
    /* 100 * eps: about 20x the floor a freshly orthonormalized m x r basis sits at, so
     * the monitor does not chase rounding noise, and far below where the drift starts
     * to matter. Scaling with the working precision means the same default is sensible
     * in the float build, where the floor is 10^9 times higher. */
    o->reorth_tol = 100.0 * SCALAR_EPS;
    o->check_every = 20;
    o->rank_max = 8;
    o->oversampling = 6;
    o->power_iters = 1;
    o->energy = 0.95;
    o->rank_mode = "energy";
    o->quantile = 0.99;
    o->warmup = 300;
    o->calibration = 300;
    o->repeat = 1;
    o->seed = 0;
    o->rank = 4;
    o->columns = -1;
}

static void usage(const char *prog)
{
    fprintf(stderr,
        "usage: %s track     --input FILE --output FILE [options]\n"
        "       %s rangefind --input FILE --rank K [--oversampling P] [--power-iters Q]\n"
        "       %s selftest\n"
        "\n"
        "track options (defaults in brackets):\n"
        "  --lambda F         exponential forgetting factor      [1.0]\n"
        "  --reorth on|off    periodic re-orthonormalization     [on]\n"
        "  --reorth-tol F     ||U^T U - I||_F trigger            [100*eps]\n"
        "  --check-every N    monitor period, in samples         [20]\n"
        "  --rank-max R       upper bound on the tracked rank    [8]\n"
        "  --oversampling P   sketch oversampling                [6]\n"
        "  --power-iters Q    subspace iterations                [1]\n"
        "  --energy F         energy threshold for rank choice   [0.95]\n"
        "  --rank-mode M      energy | gap                       [energy]\n"
        "  --quantile F       calibration quantile               [0.99]\n"
        "  --warmup N         samples used to build the subspace [300]\n"
        "  --calibration N    samples used for the threshold     [300]\n"
        "  --repeat K         replay the stream K times          [1]\n"
        "  --seed S           sketch seed                        [0]\n"
        "  --basis FILE       write the final basis U            [none]\n"
        "  --columns N        rangefind: use first N samples     [all]\n",
        prog, prog, prog);
}

static int parse(int argc, char **argv, opts_t *o)
{
    for (int i = 2; i < argc; ++i) {
        const char *k = argv[i];
        const int has_value = (i + 1 < argc);
#define NEED(name) if (!has_value) { fprintf(stderr, "%s needs a value\n", name); return -1; }
        if (!strcmp(k, "--input")) { NEED(k); o->input = argv[++i]; }
        else if (!strcmp(k, "--output")) { NEED(k); o->output = argv[++i]; }
        else if (!strcmp(k, "--basis")) { NEED(k); o->basis = argv[++i]; }
        else if (!strcmp(k, "--lambda")) { NEED(k); o->lambda = atof(argv[++i]); }
        else if (!strcmp(k, "--reorth")) { NEED(k); o->reorth = !strcmp(argv[++i], "on"); }
        else if (!strcmp(k, "--reorth-tol")) { NEED(k); o->reorth_tol = atof(argv[++i]); }
        else if (!strcmp(k, "--check-every")) { NEED(k); o->check_every = atoi(argv[++i]); }
        else if (!strcmp(k, "--rank-max")) { NEED(k); o->rank_max = atoi(argv[++i]); }
        else if (!strcmp(k, "--oversampling")) { NEED(k); o->oversampling = atoi(argv[++i]); }
        else if (!strcmp(k, "--power-iters")) { NEED(k); o->power_iters = atoi(argv[++i]); }
        else if (!strcmp(k, "--energy")) { NEED(k); o->energy = atof(argv[++i]); }
        else if (!strcmp(k, "--rank-mode")) { NEED(k); o->rank_mode = argv[++i]; }
        else if (!strcmp(k, "--quantile")) { NEED(k); o->quantile = atof(argv[++i]); }
        else if (!strcmp(k, "--warmup")) { NEED(k); o->warmup = atoi(argv[++i]); }
        else if (!strcmp(k, "--calibration")) { NEED(k); o->calibration = atoi(argv[++i]); }
        else if (!strcmp(k, "--repeat")) { NEED(k); o->repeat = atoi(argv[++i]); }
        else if (!strcmp(k, "--seed")) { NEED(k); o->seed = strtoul(argv[++i], NULL, 10); }
        else if (!strcmp(k, "--rank")) { NEED(k); o->rank = atoi(argv[++i]); }
        else if (!strcmp(k, "--columns")) { NEED(k); o->columns = atoi(argv[++i]); }
        else { fprintf(stderr, "unknown option %s\n", k); return -1; }
#undef NEED
    }
    return 0;
}

/* --- shared: read a whole (or partial) stream into a channels-by-samples matrix --- */

static scalar_t *read_matrix(const char *path, int max_cols, int *m_out, int *n_out)
{
    csv_t csv;
    if (csv_open(&csv, path) != 0) return NULL;
    const int m = csv.ncols;
    int cap = 1024, n = 0;
    scalar_t *row = malloc((size_t)m * sizeof(scalar_t));
    scalar_t *buf = malloc((size_t)cap * m * sizeof(scalar_t));   /* samples-major */
    if (!row || !buf) { free(row); free(buf); csv_close(&csv); return NULL; }

    int rc;
    while ((rc = csv_read(&csv, row)) == 1) {
        if (max_cols > 0 && n >= max_cols) break;
        if (n == cap) {
            cap *= 2;
            scalar_t *tmp = realloc(buf, (size_t)cap * m * sizeof(scalar_t));
            if (!tmp) { free(row); free(buf); csv_close(&csv); return NULL; }
            buf = tmp;
        }
        la_copy(row, buf + (size_t)n * m, m);
        ++n;
    }
    free(row);
    csv_close(&csv);
    if (rc < 0 || n == 0) { free(buf); return NULL; }

    /* Transpose into channels-by-samples, the layout every routine in src/ expects. */
    scalar_t *a = malloc((size_t)m * n * sizeof(scalar_t));
    if (!a) { free(buf); return NULL; }
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) a[(size_t)i * n + j] = buf[(size_t)j * m + i];
    free(buf);
    *m_out = m;
    *n_out = n;
    return a;
}

/* --- track ---------------------------------------------------------------------- */

static int mode_track(opts_t *o)
{
    if (!o->input || !o->output) { fprintf(stderr, "track needs --input and --output\n"); return 2; }

    csv_t csv;
    if (csv_open(&csv, o->input) != 0) return 2;
    const int m = csv.ncols;
    if (o->warmup < 2 || o->calibration < 2) {
        fprintf(stderr, "warmup and calibration must be >= 2\n"); csv_close(&csv); return 2;
    }

    /* 1. Buffer the warm-up block, channels by samples. */
    scalar_t *row = malloc((size_t)m * sizeof(scalar_t));
    scalar_t *block = malloc((size_t)m * o->warmup * sizeof(scalar_t));
    if (!row || !block) { fprintf(stderr, "out of memory\n"); return 2; }
    int filled = 0;
    while (filled < o->warmup) {
        int rc = csv_read(&csv, row);
        if (rc != 1) { fprintf(stderr, "stream has fewer than %d samples\n", o->warmup); return 2; }
        for (int i = 0; i < m; ++i) block[(size_t)i * o->warmup + filled] = row[i];
        ++filled;
    }

    /* 2. Initial subspace from the randomized range finder. */
    int ell = o->rank_max + o->oversampling;
    if (ell > m) ell = m;
    if (ell > o->warmup) ell = o->warmup;
    scalar_t *u_full = malloc((size_t)m * ell * sizeof(scalar_t));
    scalar_t *s_full = malloc((size_t)ell * sizeof(scalar_t));
    if (!u_full || !s_full) { fprintf(stderr, "out of memory\n"); return 2; }
    la_rng rng;
    la_rng_seed(&rng, (uint64_t)o->seed);
    if (rf_randomized_svd(block, m, o->warmup, ell, o->power_iters, &rng, u_full, s_full) != 0) {
        fprintf(stderr, "randomized SVD failed\n"); return 2;
    }

    /* 3. Rank selection: both criteria computed, one used, both reported. */
    const int r_energy = rank_by_energy(s_full, ell, o->energy);
    const int r_gap = rank_by_gap(s_full, ell, o->rank_max);
    int r = !strcmp(o->rank_mode, "gap") ? r_gap : r_energy;
    if (strcmp(o->rank_mode, "gap") && strcmp(o->rank_mode, "energy")) {
        fprintf(stderr, "--rank-mode must be energy or gap\n"); return 2;
    }
    if (r > o->rank_max) r = o->rank_max;
    if (r > ell) r = ell;
    if (r < 1) r = 1;

    incsvd_t trk;
    if (incsvd_alloc(&trk, m, r) != 0) { fprintf(stderr, "out of memory\n"); return 2; }
    for (int i = 0; i < m; ++i)
        la_copy(u_full + (size_t)i * ell, trk.u + (size_t)i * r, r);
    la_copy(s_full, trk.sigma, r);
    incsvd_set(&trk, trk.u, trk.sigma);
    trk.lambda = (scalar_t)o->lambda;
    trk.reorth_enabled = o->reorth;
    trk.reorth_tol = (scalar_t)o->reorth_tol;
    trk.check_every = o->check_every;

    /* The sketch is finished with. The warm-up block is released a few lines below,
     * as soon as its samples have been scored -- freeing both here rather than at the
     * end of main is the difference between a steady-state footprint of a few kilobytes
     * and one of sixty, and the steady state is the claim. */
    free(u_full);
    u_full = NULL;

    /* 4. Stream. Exactly one pass over the file: the warm-up samples are scored against
     *    the basis they built (without updating it again -- see incsvd_score), and the
     *    file pointer then simply continues from where the warm-up left off. Nothing is
     *    re-read and nothing is counted twice. `--repeat` > 1 rewinds deliberately, and
     *    is a device for the orthogonality-drift study only; it is not how the tracker
     *    would be deployed. */
    FILE *out = fopen(o->output, "w");
    if (!out) { fprintf(stderr, "cannot write %s: %s\n", o->output, strerror(errno)); return 2; }
    fprintf(out, "index,score,drift,flag\n");

    scalar_t *calib = malloc((size_t)o->calibration * sizeof(scalar_t));
    scalar_t *scratch = malloc((size_t)(m + r) * sizeof(scalar_t));
    if (!calib || !scratch) { fprintf(stderr, "out of memory\n"); return 2; }
    scalar_t threshold = 0;
    int have_threshold = 0, n_calib = 0, n_flagged = 0;
    long index = 0, n_first_pass = 0;
    const clock_t t0 = clock();

    for (int i = 0; i < o->warmup; ++i) {
        for (int c = 0; c < m; ++c) row[c] = block[(size_t)c * o->warmup + i];
        const scalar_t score = incsvd_score(&trk, row, scratch);
        fprintf(out, "%ld,%.10g,%.10g,%d\n", index, (double)score, (double)trk.drift, -1);
        ++index; ++n_first_pass;
    }
    free(block);
    block = NULL;                    /* steady state begins here: O(mr) and nothing more */

    while (csv_read(&csv, row) == 1) {
        const scalar_t score = incsvd_update(&trk, row);
        if (n_calib < o->calibration) {
            /* detect_threshold sorts its input, which is why the quantile is taken
             * from a buffer the stream no longer needs. */
            calib[n_calib++] = score;
            if (n_calib == o->calibration) {
                threshold = detect_threshold(calib, n_calib, o->quantile);
                have_threshold = 1;
            }
        }
        const int flag = have_threshold ? (score > threshold) : -1;
        if (flag == 1) ++n_flagged;
        fprintf(out, "%ld,%.10g,%.10g,%d\n", index, (double)score,
                (double)trk.drift, flag);
        ++index; ++n_first_pass;
    }
    for (int pass = 1; pass < o->repeat; ++pass) {
        csv_rewind(&csv);
        while (csv_read(&csv, row) == 1) {
            const scalar_t score = incsvd_update(&trk, row);
            const int flag = have_threshold ? (score > threshold) : -1;
            if (flag == 1) ++n_flagged;
            fprintf(out, "%ld,%.10g,%.10g,%d\n", index, (double)score,
                    (double)trk.drift, flag);
            ++index;
        }
    }
    const double seconds = (double)(clock() - t0) / CLOCKS_PER_SEC;
    fclose(out);

    if (o->basis) {
        FILE *bf = fopen(o->basis, "w");
        if (!bf) { fprintf(stderr, "cannot write %s\n", o->basis); return 2; }
        for (int j = 0; j < r; ++j) fprintf(bf, "%su%d", j ? "," : "", j);
        fprintf(bf, "\n");
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < r; ++j)
                fprintf(bf, "%s%.17g", j ? "," : "", (double)trk.u[(size_t)i * r + j]);
            fprintf(bf, "\n");
        }
        fclose(bf);
    }

    printf("scalar=%s\n", SCALAR_NAME);
    printf("channels=%d\n", m);
    printf("tracker_bytes=%zu\n", incsvd_bytes(&trk));
    printf("calibration_bytes=%zu\n", (size_t)o->calibration * sizeof(scalar_t));
    printf("warmup_block_bytes=%zu\n",
           (size_t)m * o->warmup * sizeof(scalar_t));
    printf("samples=%ld\n", n_first_pass);
    /* Rank-one updates and output rows are NOT the same number: the warm-up samples are
     * scored and written but never folded in, so `index` overcounts the updates by
     * `warmup`. Reporting the wrong one would put a figure in results/ that contradicts
     * the numpy oracle's count of the same experiment. */
    printf("updates=%ld\n", trk.steps);
    printf("rows=%ld\n", index);
    printf("sketch_width=%d\n", ell);
    printf("rank=%d\n", r);
    printf("rank_energy=%d\n", r_energy);
    printf("rank_gap=%d\n", r_gap);
    printf("lambda=%.17g\n", o->lambda);
    printf("effective_window=%.6g\n", forget_window_from_lambda((scalar_t)o->lambda));
    printf("reorth=%d\n", o->reorth);
    printf("reorth_tol=%.17g\n", o->reorth_tol);
    printf("n_reorth=%d\n", trk.n_reorth);
    printf("threshold=%.17g\n", (double)threshold);
    printf("n_flagged=%d\n", n_flagged);
    printf("final_drift=%.17g\n", (double)trk.drift);
    printf("max_drift=%.17g\n", (double)trk.max_drift);
    printf("seconds=%.6g\n", seconds);
    printf("us_per_sample=%.6g\n", index ? 1e6 * seconds / (double)index : 0.0);
    for (int i = 0; i < r; ++i) printf("sigma%d=%.10g\n", i, (double)trk.sigma[i]);
    for (int i = 0; i < ell; ++i) printf("spectrum%d=%.10g\n", i, (double)s_full[i]);

    incsvd_free(&trk);
    free(calib); free(scratch); free(row); free(s_full);
    csv_close(&csv);
    return 0;
}

/* --- rangefind ------------------------------------------------------------------ */

static int mode_rangefind(opts_t *o)
{
    if (!o->input) { fprintf(stderr, "rangefind needs --input\n"); return 2; }
    int m = 0, n = 0;
    scalar_t *a = read_matrix(o->input, o->columns, &m, &n);
    if (!a) { fprintf(stderr, "cannot read %s\n", o->input); return 2; }

    int ell = o->rank + o->oversampling;
    if (ell > m) ell = m;
    if (ell > n) ell = n;
    scalar_t *q = malloc((size_t)m * ell * sizeof(scalar_t));
    if (!q) { free(a); return 2; }

    la_rng rng;
    la_rng_seed(&rng, (uint64_t)o->seed);
    const clock_t t0 = clock();
    if (rf_range_finder(a, m, n, ell, o->power_iters, &rng, q) != 0) {
        fprintf(stderr, "range finder failed\n"); free(a); free(q); return 2;
    }
    const double seconds = (double)(clock() - t0) / CLOCKS_PER_SEC;
    const scalar_t err = rf_projection_error(a, m, n, q, ell);
    scalar_t *gram = malloc((size_t)ell * ell * sizeof(scalar_t));
    const scalar_t orth = gram ? la_orth_error(q, m, ell, gram) : (scalar_t)-1;

    printf("scalar=%s\n", SCALAR_NAME);
    printf("rows=%d\n", m);
    printf("columns=%d\n", n);
    printf("rank=%d\n", o->rank);
    printf("oversampling=%d\n", o->oversampling);
    printf("power_iters=%d\n", o->power_iters);
    printf("sketch_width=%d\n", ell);
    printf("proj_error=%.17g\n", (double)err);
    printf("q_orth_error=%.17g\n", (double)orth);
    printf("seconds=%.6g\n", seconds);

    free(gram); free(q); free(a);
    return 0;
}

/* --- selftest ------------------------------------------------------------------- */

static int check(const char *name, int ok, double value, double limit)
{
    printf("%-38s %-4s value=%.3e limit=%.3e\n", name, ok ? "PASS" : "FAIL", value, limit);
    return ok ? 0 : 1;
}

static int mode_selftest(void)
{
    /* Tolerances are stated relative to the working precision so the float build is
     * held to a float standard rather than a double one. */
    const double eps = (double)SCALAR_EPS;
    int failures = 0;
    la_rng rng;
    la_rng_seed(&rng, 12345u);

    /* Householder QR: A = QR exactly, Q^T Q = I to working precision. */
    {
        const int m = 14, n = 5;
        scalar_t a[14 * 5], acopy[14 * 5], q[14 * 5], r[5 * 5], work[14 + 5], gram[5 * 5];
        for (int i = 0; i < m * n; ++i) a[i] = la_rng_normal(&rng);
        la_copy(a, acopy, m * n);
        int rc = la_qr(a, m, n, q, r, work);
        double recon = 0.0;
        scalar_t qr[14 * 5];
        la_gemm(q, m, n, r, n, qr);
        for (int i = 0; i < m * n; ++i) {
            const double d = (double)qr[i] - (double)acopy[i];
            recon += d * d;
        }
        failures += check("qr: A = QR", rc == 0 && sqrt(recon) < 200 * eps * 10,
                          sqrt(recon), 200 * eps * 10);
        const double oe = (double)la_orth_error(q, m, n, gram);
        failures += check("qr: ||Q^T Q - I||_F", oe < 100 * eps, oe, 100 * eps);
    }

    /* One-sided Jacobi: reconstruction, orthogonality, and descending order. */
    for (int trial = 0; trial < 2; ++trial) {
        const int m = trial ? 9 : 7, n = trial ? 4 : 7;
        scalar_t a[9 * 4 > 7 * 7 ? 9 * 4 : 7 * 7];
        scalar_t acopy[9 * 4 > 7 * 7 ? 9 * 4 : 7 * 7];
        scalar_t u[9 * 4 > 7 * 7 ? 9 * 4 : 7 * 7];
        scalar_t v[7 * 7], s[7], gram[7 * 7];
        for (int i = 0; i < m * n; ++i) a[i] = la_rng_normal(&rng);
        la_copy(a, acopy, m * n);
        const int sweeps = la_jacobi_svd(a, m, n, u, s, v, 60);

        /* || U diag(s) V^T - A ||_F */
        double recon = 0.0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                double acc = 0.0;
                for (int k = 0; k < n; ++k)
                    acc += (double)u[(size_t)i * n + k] * (double)s[k] *
                           (double)v[(size_t)j * n + k];
                const double d = acc - (double)acopy[(size_t)i * n + j];
                recon += d * d;
            }
        }
        failures += check("jacobi: || U S V^T - A ||_F", sweeps >= 0 && sqrt(recon) < 2000 * eps,
                          sqrt(recon), 2000 * eps);
        const double ue = (double)la_orth_error(u, m, n, gram);
        failures += check("jacobi: ||U^T U - I||_F", ue < 200 * eps, ue, 200 * eps);
        const double ve = (double)la_orth_error(v, n, n, gram);
        failures += check("jacobi: ||V^T V - I||_F", ve < 200 * eps, ve, 200 * eps);
        int sorted = 1;
        for (int i = 0; i + 1 < n; ++i) if (s[i] < s[i + 1]) sorted = 0;
        failures += check("jacobi: singular values descending", sorted, sorted ? 0 : 1, 0.5);
    }

    /* Rank selection on a spectrum whose answer is known by construction. */
    {
        const scalar_t s[6] = {10, 5, 3, (scalar_t)0.02, (scalar_t)0.01, (scalar_t)0.005};
        const int re = rank_by_energy(s, 6, 0.95);
        const int rg = rank_by_gap(s, 6, 5);
        failures += check("rank: energy(0.95) = 3", re == 3, re, 3);
        failures += check("rank: gap = 3", rg == 3, rg, 3);
    }

    /* Forgetting: the window round trip, which is the one place a factor of two hides.
     * The tolerance is N_eff^2 * eps, not a fixed relative figure, because recovering
     * N_eff from lambda is ill-conditioned near lambda = 1: dN/N = 2 N dlambda/lambda,
     * so a stored lambda amplifies its own rounding error by 2 N_eff. At N_eff = 400
     * that is 800x, which is invisible in double and visible in float. */
    {
        const double n_eff = 400.0;
        const scalar_t lam = forget_lambda_from_window(n_eff);
        const double back = forget_window_from_lambda(lam);
        const double limit = 4.0 * n_eff * n_eff * eps;
        failures += check("forget: window round trip", fabs(back - n_eff) < limit,
                          fabs(back - n_eff), limit);
    }

    /* Quantile: on 0..100 the 0.99 quantile is 99 exactly under linear interpolation. */
    {
        scalar_t v[101];
        for (int i = 0; i <= 100; ++i) v[i] = (scalar_t)(100 - i);
        const double got = (double)detect_threshold(v, 101, 0.99);
        failures += check("detect: quantile(0..100, 0.99) = 99", fabs(got - 99.0) < 1e-9,
                          got, 99.0);
    }

    printf("failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}

/* --- entry ---------------------------------------------------------------------- */

int main(int argc, char **argv)
{
    if (argc < 2) { usage(argv[0]); return 2; }
    opts_t o;
    defaults(&o);
    if (!strcmp(argv[1], "selftest")) return mode_selftest();
    if (parse(argc, argv, &o) != 0) { usage(argv[0]); return 2; }
    if (!strcmp(argv[1], "track")) return mode_track(&o);
    if (!strcmp(argv[1], "rangefind")) return mode_rangefind(&o);
    usage(argv[0]);
    return 2;
}
