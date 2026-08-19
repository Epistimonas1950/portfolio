/* linalg.c -- see linalg.h for the conventions and the reasoning. */

#include "linalg.h"

#include <math.h>
#include <string.h>

/* --- level 1 ------------------------------------------------------------------- */

scalar_t la_dot(const scalar_t *x, const scalar_t *y, int n)
{
    /* Accumulate in double even in the float build. The dot product is where a
     * single-precision implementation loses accuracy first, and promoting the
     * accumulator is free on any machine with a double-precision FPU (the ARM11 in a
     * Pi Zero has VFPv2, so it is free there too). The stored data stays float; only
     * the running sum is widened. */
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += (double)x[i] * (double)y[i];
    return (scalar_t)sum;
}

scalar_t la_norm2(const scalar_t *x, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += (double)x[i] * (double)x[i];
    return (scalar_t)sqrt(sum);
}

void la_scale(scalar_t *x, int n, scalar_t alpha)
{
    for (int i = 0; i < n; ++i) x[i] *= alpha;
}

void la_axpy(scalar_t alpha, const scalar_t *x, scalar_t *y, int n)
{
    for (int i = 0; i < n; ++i) y[i] += alpha * x[i];
}

void la_zero(scalar_t *x, int n)
{
    memset(x, 0, (size_t)n * sizeof(scalar_t));
}

void la_copy(const scalar_t *src, scalar_t *dst, int n)
{
    memcpy(dst, src, (size_t)n * sizeof(scalar_t));
}

/* --- level 2 / 3 --------------------------------------------------------------- */

void la_matvec(const scalar_t *a, int m, int n, const scalar_t *x, scalar_t *y)
{
    for (int i = 0; i < m; ++i) y[i] = la_dot(a + (size_t)i * n, x, n);
}

void la_matvec_t(const scalar_t *a, int m, int n, const scalar_t *x, scalar_t *y)
{
    /* Row-major A^T x is a sequence of axpy's over the rows of A, which keeps the
     * access pattern sequential. Writing it as n dot products of strided columns
     * instead costs a cache miss per element and is the classic way a hand-written
     * transpose product ends up four times slower than it needs to be. */
    la_zero(y, n);
    for (int i = 0; i < m; ++i) {
        const scalar_t xi = x[i];
        const scalar_t *row = a + (size_t)i * n;
        for (int j = 0; j < n; ++j) y[j] += xi * row[j];
    }
}

void la_gemm(const scalar_t *a, int m, int k, const scalar_t *b, int n, scalar_t *c)
{
    la_zero(c, m * n);
    for (int i = 0; i < m; ++i) {
        scalar_t *crow = c + (size_t)i * n;
        for (int p = 0; p < k; ++p) {
            const scalar_t aip = a[(size_t)i * k + p];
            if (aip == (scalar_t)0) continue;
            const scalar_t *brow = b + (size_t)p * n;
            for (int j = 0; j < n; ++j) crow[j] += aip * brow[j];
        }
    }
}

void la_gemm_tn(const scalar_t *a, int m, int k, const scalar_t *b, int n, scalar_t *c)
{
    la_zero(c, k * n);
    for (int p = 0; p < m; ++p) {
        const scalar_t *arow = a + (size_t)p * k;
        const scalar_t *brow = b + (size_t)p * n;
        for (int i = 0; i < k; ++i) {
            const scalar_t aip = arow[i];
            if (aip == (scalar_t)0) continue;
            scalar_t *crow = c + (size_t)i * n;
            for (int j = 0; j < n; ++j) crow[j] += aip * brow[j];
        }
    }
}

/* --- Householder QR ------------------------------------------------------------- */

int la_qr(scalar_t *a, int m, int n, scalar_t *q, scalar_t *r, scalar_t *work)
{
    if (m < n || n < 1) return -1;
    scalar_t *tau = work;          /* n */
    scalar_t *v = work + n;        /* m */

    for (int j = 0; j < n; ++j) {
        /* Householder vector for rows j..m-1 of column j. */
        double norm = 0.0;
        for (int i = j; i < m; ++i) {
            const double aij = (double)a[(size_t)i * n + j];
            norm += aij * aij;
        }
        norm = sqrt(norm);
        if (norm == 0.0) { tau[j] = 0; continue; }

        const scalar_t ajj = a[(size_t)j * n + j];
        /* Sign choice: alpha = -sign(a_jj) ||x||. Taking the other sign makes
         * v_j = a_jj - alpha a difference of nearly equal numbers when a_jj is already
         * close to ||x||, which is cancellation in the one place it cannot be
         * tolerated -- the reflector would then be computed with no correct digits. */
        const scalar_t alpha = (ajj >= 0) ? (scalar_t)(-norm) : (scalar_t)norm;

        for (int i = j; i < m; ++i) v[i] = a[(size_t)i * n + j];
        v[j] -= alpha;
        const scalar_t vnorm = la_norm2(v + j, m - j);
        if (vnorm == 0) { tau[j] = 0; continue; }
        for (int i = j; i < m; ++i) v[i] /= vnorm;

        /* A[j:, j:] -= 2 v (v^T A[j:, j:]) */
        for (int col = j; col < n; ++col) {
            double vt = 0.0;
            for (int i = j; i < m; ++i) vt += (double)v[i] * (double)a[(size_t)i * n + col];
            const scalar_t f = (scalar_t)(2.0 * vt);
            for (int i = j; i < m; ++i) a[(size_t)i * n + col] -= f * v[i];
        }
        /* Store the reflector below the diagonal; tau marks it as present. */
        for (int i = j + 1; i < m; ++i) a[(size_t)i * n + j] = v[i];
        a[(size_t)j * n + j] = alpha;
        tau[j] = v[j];
    }

    /* R = upper triangle of the transformed A. */
    la_zero(r, n * n);
    for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j) r[(size_t)i * n + j] = a[(size_t)i * n + j];

    /* Q = H_0 H_1 ... H_{n-1} applied to the first n columns of I_m, accumulated
     * backwards so each reflector touches only the trailing block it owns. */
    la_zero(q, m * n);
    for (int j = 0; j < n; ++j) q[(size_t)j * n + j] = 1;
    for (int j = n - 1; j >= 0; --j) {
        if (tau[j] == 0) continue;
        v[j] = tau[j];
        for (int i = j + 1; i < m; ++i) v[i] = a[(size_t)i * n + j];
        for (int col = 0; col < n; ++col) {
            double vt = 0.0;
            for (int i = j; i < m; ++i) vt += (double)v[i] * (double)q[(size_t)i * n + col];
            const scalar_t f = (scalar_t)(2.0 * vt);
            for (int i = j; i < m; ++i) q[(size_t)i * n + col] -= f * v[i];
        }
    }
    return 0;
}

scalar_t la_orth_error(const scalar_t *u, int m, int r, scalar_t *work)
{
    la_gemm_tn(u, m, r, u, r, work);         /* work = U^T U, r x r */
    double sum = 0.0;
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < r; ++j) {
            double d = (double)work[(size_t)i * r + j] - (i == j ? 1.0 : 0.0);
            sum += d * d;
        }
    }
    return (scalar_t)sqrt(sum);
}

/* --- one-sided Jacobi SVD -------------------------------------------------------- */

static void swap_columns(scalar_t *a, int rows, int cols, int i, int j)
{
    for (int k = 0; k < rows; ++k) {
        scalar_t t = a[(size_t)k * cols + i];
        a[(size_t)k * cols + i] = a[(size_t)k * cols + j];
        a[(size_t)k * cols + j] = t;
    }
}

int la_jacobi_svd(scalar_t *a, int m, int n, scalar_t *u, scalar_t *s, scalar_t *v,
                  int max_sweeps)
{
    if (m < n || n < 1) return -1;

    /* V starts at the identity and accumulates every rotation, so at convergence
     * A V = U diag(s) and therefore A = U diag(s) V^T. */
    la_zero(v, n * n);
    for (int i = 0; i < n; ++i) v[(size_t)i * n + i] = 1;

    const double tol = 10.0 * (double)SCALAR_EPS;
    int sweep = 0;
    for (; sweep < max_sweeps; ++sweep) {
        int rotations = 0;
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                /* The 2x2 Gram matrix of columns p and q of the *current* A. */
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (int k = 0; k < m; ++k) {
                    const double ap = (double)a[(size_t)k * n + p];
                    const double aq = (double)a[(size_t)k * n + q];
                    alpha += ap * ap;
                    beta += aq * aq;
                    gamma += ap * aq;
                }
                if (gamma == 0.0) continue;
                /* Converged for this pair when the columns are orthogonal to within
                 * a relative tolerance. Testing |gamma| against an absolute epsilon
                 * instead makes the routine loop forever on badly scaled input. */
                if (fabs(gamma) <= tol * sqrt(alpha * beta)) continue;

                const double zeta = (beta - alpha) / (2.0 * gamma);
                /* t is the root of t^2 + 2 zeta t - 1 = 0 of SMALLER magnitude, written
                 * in the form that avoids cancellation for large |zeta|. The small root
                 * is the rotation of angle < pi/4, which is what keeps the sweep
                 * convergent. */
                const double t = (zeta >= 0.0)
                                     ? 1.0 / (zeta + sqrt(1.0 + zeta * zeta))
                                     : -1.0 / (-zeta + sqrt(1.0 + zeta * zeta));
                const double c = 1.0 / sqrt(1.0 + t * t);
                const double sn = c * t;

                for (int k = 0; k < m; ++k) {
                    const double ap = (double)a[(size_t)k * n + p];
                    const double aq = (double)a[(size_t)k * n + q];
                    a[(size_t)k * n + p] = (scalar_t)(c * ap - sn * aq);
                    a[(size_t)k * n + q] = (scalar_t)(sn * ap + c * aq);
                }
                for (int k = 0; k < n; ++k) {
                    const double vp = (double)v[(size_t)k * n + p];
                    const double vq = (double)v[(size_t)k * n + q];
                    v[(size_t)k * n + p] = (scalar_t)(c * vp - sn * vq);
                    v[(size_t)k * n + q] = (scalar_t)(sn * vp + c * vq);
                }
                ++rotations;
            }
        }
        if (rotations == 0) break;
    }

    /* Columns of the rotated A are now orthogonal: their norms are the singular
     * values and their directions are the left singular vectors. */
    for (int j = 0; j < n; ++j) {
        double norm = 0.0;
        for (int k = 0; k < m; ++k) {
            const double x = (double)a[(size_t)k * n + j];
            norm += x * x;
        }
        s[j] = (scalar_t)sqrt(norm);
    }

    /* Sort descending. Selection sort: n <= 9 here, and the permutation has to be
     * applied to three arrays, so an in-place algorithm with an explicit swap is
     * easier to get right than anything cleverer. Truncation downstream keeps the
     * FIRST r columns, so this sort is what makes truncation mean "drop the weakest". */
    for (int i = 0; i < n - 1; ++i) {
        int best = i;
        for (int j = i + 1; j < n; ++j) if (s[j] > s[best]) best = j;
        if (best == i) continue;
        scalar_t t = s[i]; s[i] = s[best]; s[best] = t;
        swap_columns(a, m, n, i, best);
        swap_columns(v, n, n, i, best);
    }

    for (int j = 0; j < n; ++j) {
        if (s[j] > 0) {
            const scalar_t inv = (scalar_t)1 / s[j];
            for (int k = 0; k < m; ++k) u[(size_t)k * n + j] = a[(size_t)k * n + j] * inv;
        } else {
            /* A zero singular value leaves its left singular vector undefined. Return
             * zeros rather than the normalized rounding noise that dividing by ~0 would
             * produce: callers truncate before this column, and a zero column that
             * escapes is obviously wrong, whereas a unit-norm garbage column is not. */
            for (int k = 0; k < m; ++k) u[(size_t)k * n + j] = 0;
        }
    }
    return (sweep >= max_sweeps) ? -1 : sweep;
}

/* --- PCG32 ---------------------------------------------------------------------- */

void la_rng_seed(la_rng *rng, uint64_t seed)
{
    rng->state = 0u;
    rng->inc = (seed << 1u) | 1u;
    rng->spare = 0;
    rng->has_spare = 0;
    (void)la_rng_u32(rng);
    rng->state += 0x853c49e6748fea9bULL + seed;
    (void)la_rng_u32(rng);
}

uint32_t la_rng_u32(la_rng *rng)
{
    const uint64_t old = rng->state;
    rng->state = old * 6364136223846793005ULL + rng->inc;
    const uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    const uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
}

scalar_t la_rng_uniform(la_rng *rng)
{
    /* Open interval (0, 1): Box-Muller takes a logarithm of this value. */
    return (scalar_t)(((double)la_rng_u32(rng) + 0.5) / 4294967296.0);
}

scalar_t la_rng_normal(la_rng *rng)
{
    if (rng->has_spare) { rng->has_spare = 0; return rng->spare; }
    const double u1 = (double)la_rng_uniform(rng);
    const double u2 = (double)la_rng_uniform(rng);
    const double radius = sqrt(-2.0 * log(u1));
    const double angle = 6.283185307179586476925286766559 * u2;
    rng->spare = (scalar_t)(radius * sin(angle));
    rng->has_spare = 1;
    return (scalar_t)(radius * cos(angle));
}

/* --- misc ----------------------------------------------------------------------- */

void la_sort_ascending(scalar_t *x, int n)
{
    /* Insertion sort. n is the calibration window, a few hundred, and this runs once
     * per stream -- not per sample. */
    for (int i = 1; i < n; ++i) {
        const scalar_t key = x[i];
        int j = i - 1;
        while (j >= 0 && x[j] > key) { x[j + 1] = x[j]; --j; }
        x[j + 1] = key;
    }
}
