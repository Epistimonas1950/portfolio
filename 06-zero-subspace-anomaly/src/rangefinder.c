/* rangefinder.c -- see rangefinder.h for the mathematics and the error bound. */

#include "rangefinder.h"

#include <math.h>
#include <stdlib.h>

int rf_range_finder(const scalar_t *a, int m, int n, int ell, int power_iters,
                    la_rng *rng, scalar_t *q_out)
{
    if (m < 1 || n < 1 || ell < 1 || ell > m || ell > n) return -1;

    const size_t n_ell = (size_t)n * ell;
    const size_t m_ell = (size_t)m * ell;
    scalar_t *omega = malloc(n_ell * sizeof(scalar_t));
    scalar_t *y = malloc(m_ell * sizeof(scalar_t));
    scalar_t *z = malloc(n_ell * sizeof(scalar_t));
    scalar_t *zq = malloc(n_ell * sizeof(scalar_t));
    scalar_t *rr = malloc((size_t)ell * ell * sizeof(scalar_t));
    scalar_t *work = malloc(((size_t)(m > n ? m : n) + ell) * sizeof(scalar_t));
    int rc = -2;
    if (!omega || !y || !z || !zq || !rr || !work) goto done;

    for (size_t i = 0; i < n_ell; ++i) omega[i] = la_rng_normal(rng);
    la_gemm(a, m, n, omega, ell, y);              /* Y = A Omega  (m x ell) */
    if (la_qr(y, m, ell, q_out, rr, work) != 0) goto done;

    for (int it = 0; it < power_iters; ++it) {
        /* Z = A^T Q, re-orthonormalized; then Y = A Z, re-orthonormalized. Both QRs
         * are what makes this subspace iteration rather than power iteration; see the
         * header for why omitting them destroys the weak directions. */
        la_gemm_tn(a, m, n, q_out, ell, z);             /* Z = A^T Q  (n x ell) */
        if (la_qr(z, n, ell, zq, rr, work) != 0) goto done;
        la_gemm(a, m, n, zq, ell, y);                   /* Y = A Z    (m x ell) */
        if (la_qr(y, m, ell, q_out, rr, work) != 0) goto done;
    }
    rc = 0;

done:
    free(omega); free(y); free(z); free(zq); free(rr); free(work);
    return rc;
}

int rf_randomized_svd(const scalar_t *a, int m, int n, int ell, int power_iters,
                      la_rng *rng, scalar_t *u_out, scalar_t *s_out)
{
    if (m < 1 || n < 1 || ell < 1 || ell > m || ell > n) return -1;

    scalar_t *q = malloc((size_t)m * ell * sizeof(scalar_t));
    scalar_t *bt = malloc((size_t)n * ell * sizeof(scalar_t));
    scalar_t *p = malloc((size_t)n * ell * sizeof(scalar_t));
    scalar_t *w = malloc((size_t)ell * ell * sizeof(scalar_t));
    int rc = -2;
    if (!q || !bt || !p || !w) goto done;

    if (rf_range_finder(a, m, n, ell, power_iters, rng, q) != 0) goto done;

    /* B = Q^T A is ell x n. Its transpose B^T = A^T Q is n x ell, which is tall, and
     * one-sided Jacobi wants tall input -- so form the transpose directly and read the
     * factorization off it: B^T = P diag(s) W^T  =>  B = W diag(s) P^T, and the left
     * singular vectors of A are U = Q W. Forming B and transposing it afterwards would
     * be the same arithmetic with an extra copy. */
    la_gemm_tn(a, m, n, q, ell, bt);
    if (la_jacobi_svd(bt, n, ell, p, s_out, w, 60) < 0) { rc = -3; goto done; }
    la_gemm(q, m, ell, w, ell, u_out);
    rc = 0;

done:
    free(q); free(bt); free(p); free(w);
    return rc;
}

scalar_t rf_projection_error(const scalar_t *a, int m, int n, const scalar_t *q,
                             int ell)
{
    scalar_t *col = malloc((size_t)m * sizeof(scalar_t));
    scalar_t *coef = malloc((size_t)ell * sizeof(scalar_t));
    if (!col || !coef) { free(col); free(coef); return (scalar_t)-1; }

    double total = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) col[i] = a[(size_t)i * n + j];
        la_matvec_t(q, m, ell, col, coef);              /* coef = Q^T a_j */
        for (int i = 0; i < m; ++i) {
            double proj = 0.0;
            const scalar_t *qrow = q + (size_t)i * ell;
            for (int k = 0; k < ell; ++k) proj += (double)qrow[k] * (double)coef[k];
            const double d = (double)col[i] - proj;
            total += d * d;
        }
    }
    free(col); free(coef);
    return (scalar_t)sqrt(total);
}
