/* reorth.c -- see reorth.h. Monitor and factorization-preserving repair. */

#include "reorth.h"

#include <stdlib.h>

scalar_t reorth_monitor(incsvd_t *t)
{
    /* work holds the r x r Gram matrix; it is sized max(m + r + 1, (r+1)^2) at alloc,
     * which covers r*r for every r >= 0. */
    return la_orth_error(t->u, t->m, t->r, t->work);
}

int reorth_restore(incsvd_t *t)
{
    const int m = t->m, r = t->r;
    scalar_t *acopy = malloc((size_t)m * r * sizeof(scalar_t));
    scalar_t *q = malloc((size_t)m * r * sizeof(scalar_t));
    scalar_t *rmat = malloc((size_t)r * r * sizeof(scalar_t));
    scalar_t *usmall = malloc((size_t)r * r * sizeof(scalar_t));
    scalar_t *vsmall = malloc((size_t)r * r * sizeof(scalar_t));
    scalar_t *snew = malloc((size_t)r * sizeof(scalar_t));
    scalar_t *work = malloc((size_t)(m + r) * sizeof(scalar_t));
    int rc = -2;
    if (!acopy || !q || !rmat || !usmall || !vsmall || !snew || !work) goto done;

    la_copy(t->u, acopy, m * r);
    if (la_qr(acopy, m, r, q, rmat, work) != 0) goto done;

    /* rmat <- R * diag(sigma): scaling COLUMN j by sigma_j, because Sigma multiplies
     * from the right. Scaling rows instead is the classic transposition bug and would
     * silently permute energy between components. */
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < r; ++j) rmat[(size_t)i * r + j] *= t->sigma[j];

    if (la_jacobi_svd(rmat, r, r, usmall, snew, vsmall, 60) < 0) { rc = -3; goto done; }

    la_gemm(q, m, r, usmall, r, t->u);          /* U <- Q U~ */
    la_copy(snew, t->sigma, r);                 /* Sigma <- S~ */
    ++t->n_reorth;
    rc = 0;

done:
    free(acopy); free(q); free(rmat); free(usmall); free(vsmall); free(snew); free(work);
    return rc;
}
