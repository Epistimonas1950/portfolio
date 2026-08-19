/* incsvd.c -- Brand's rank-one update. See incsvd.h for the derivation. */

#include "incsvd.h"

#include "detect.h"
#include "forget.h"
#include "reorth.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int imax(int a, int b) { return a > b ? a : b; }

int incsvd_alloc(incsvd_t *t, int m, int r)
{
    if (m < 1 || r < 1 || r > m) return -1;
    memset(t, 0, sizeof(*t));
    t->m = m;
    t->r = r;
    t->lambda = 1;
    t->reorth_enabled = 1;
    t->check_every = 20;
    t->reorth_tol = (scalar_t)(100.0 * SCALAR_EPS);

    const int r1 = r + 1;
    const int worksz = imax(m + r1, r1 * r1);
    t->u = calloc((size_t)m * r, sizeof(scalar_t));
    t->sigma = calloc((size_t)r, sizeof(scalar_t));
    t->mvec = calloc((size_t)r, sizeof(scalar_t));
    t->resid = calloc((size_t)m, sizeof(scalar_t));
    t->kmat = calloc((size_t)r1 * r1, sizeof(scalar_t));
    t->ku = calloc((size_t)r1 * r1, sizeof(scalar_t));
    t->kv = calloc((size_t)r1 * r1, sizeof(scalar_t));
    t->ks = calloc((size_t)r1, sizeof(scalar_t));
    t->ubig = calloc((size_t)m * r1, sizeof(scalar_t));
    t->unew = calloc((size_t)m * r, sizeof(scalar_t));
    t->work = calloc((size_t)worksz, sizeof(scalar_t));
    if (!t->u || !t->sigma || !t->mvec || !t->resid || !t->kmat || !t->ku ||
        !t->kv || !t->ks || !t->ubig || !t->unew || !t->work) {
        incsvd_free(t);
        return -2;
    }
    return 0;
}

void incsvd_free(incsvd_t *t)
{
    free(t->u); free(t->sigma); free(t->mvec); free(t->resid);
    free(t->kmat); free(t->ku); free(t->kv); free(t->ks);
    free(t->ubig); free(t->unew); free(t->work);
    memset(t, 0, sizeof(*t));
}

size_t incsvd_bytes(const incsvd_t *t)
{
    const int m = t->m, r = t->r, r1 = r + 1;
    const int worksz = imax(m + r1, r1 * r1);
    const size_t elems = (size_t)m * r          /* u      */
                       + (size_t)r              /* sigma  */
                       + (size_t)r              /* mvec   */
                       + (size_t)m              /* resid  */
                       + 3u * (size_t)r1 * r1   /* kmat, ku, kv */
                       + (size_t)r1             /* ks     */
                       + (size_t)m * r1         /* ubig   */
                       + (size_t)m * r          /* unew   */
                       + (size_t)worksz;        /* work   */
    return elems * sizeof(scalar_t) + sizeof(incsvd_t);
}

void incsvd_set(incsvd_t *t, const scalar_t *u, const scalar_t *sigma)
{
    la_copy(u, t->u, t->m * t->r);
    la_copy(sigma, t->sigma, t->r);
    t->steps = 0;
    t->n_reorth = 0;
    t->drift = reorth_monitor(t);
    t->max_drift = t->drift;
}

scalar_t incsvd_score(const incsvd_t *t, const scalar_t *a, scalar_t *scratch)
{
    const int m = t->m, r = t->r;
    scalar_t *coef = scratch;                    /* r */
    scalar_t *resid = scratch + r;               /* m */
    la_matvec_t(t->u, m, r, a, coef);
    la_copy(a, resid, m);
    for (int i = 0; i < m; ++i) {
        const scalar_t *urow = t->u + (size_t)i * r;
        double proj = 0.0;
        for (int j = 0; j < r; ++j) proj += (double)urow[j] * (double)coef[j];
        resid[i] -= (scalar_t)proj;
    }
    return detect_score(la_norm2(resid, m), la_norm2(a, m));
}

scalar_t incsvd_update(incsvd_t *t, const scalar_t *a)
{
    const int m = t->m, r = t->r, r1 = r + 1;

    /* 1. Split the sample: coordinates inside the subspace, residual outside it. */
    la_matvec_t(t->u, m, r, a, t->mvec);            /* mvec = U^T a */
    la_copy(a, t->resid, m);
    for (int i = 0; i < m; ++i) {
        const scalar_t *urow = t->u + (size_t)i * r;
        double proj = 0.0;
        for (int j = 0; j < r; ++j) proj += (double)urow[j] * (double)t->mvec[j];
        t->resid[i] -= (scalar_t)proj;              /* p = a - U mvec */
    }
    scalar_t rho = la_norm2(t->resid, m);
    const scalar_t a_norm = la_norm2(a, m);

    /* 2. Score against the CURRENT subspace, before the update folds `a` in. */
    const scalar_t score = detect_score(rho, a_norm);

    /* 3. Forgetting: Sigma <- lambda Sigma. See forget.h for lambda <-> window. */
    forget_apply(t->sigma, r, t->lambda);

    /* 4. The rho guard (see incsvd.h). Setting rho and q to zero rather than branching
     *    makes the update degenerate exactly into the rank-preserving form. */
    const scalar_t rho_floor = (scalar_t)(sqrt((double)SCALAR_EPS) * (double)a_norm);
    if (rho <= rho_floor) {
        rho = 0;
        la_zero(t->resid, m);
    } else {
        la_scale(t->resid, m, (scalar_t)1 / rho);   /* q = p / rho */
    }

    /* 5. The small matrix K = [[Sigma, mvec], [0, rho]], (r+1) x (r+1). */
    la_zero(t->kmat, r1 * r1);
    for (int i = 0; i < r; ++i) {
        t->kmat[(size_t)i * r1 + i] = t->sigma[i];
        t->kmat[(size_t)i * r1 + r] = t->mvec[i];
    }
    t->kmat[(size_t)r * r1 + r] = rho;

    /* 6. Its SVD, O(r^3). la_jacobi_svd returns the singular values descending, which
     *    is what makes step 7's truncation drop the weakest direction. */
    if (la_jacobi_svd(t->kmat, r1, r1, t->ku, t->ks, t->kv, 60) < 0) {
        /* Refusing to update on a non-converged decomposition is the safe failure:
         * the previous basis is still orthonormal and still describes the data. */
        ++t->steps;
        return score;
    }

    /* 7. U <- [U q] U'[:, 0:r], Sigma <- S'[0:r]. */
    for (int i = 0; i < m; ++i) {
        scalar_t *dst = t->ubig + (size_t)i * r1;
        la_copy(t->u + (size_t)i * r, dst, r);
        dst[r] = t->resid[i];
    }
    /* ku is (r+1) x (r+1); the truncation to its first r columns is done by hand
     * because la_gemm has no submatrix view. */
    for (int i = 0; i < r1; ++i)
        la_copy(t->ku + (size_t)i * r1, t->work + (size_t)i * r, r);
    la_gemm(t->ubig, m, r1, t->work, r, t->unew);
    la_copy(t->unew, t->u, m * r);
    la_copy(t->ks, t->sigma, r);

    ++t->steps;

    /* 8. The orthogonality monitor, every check_every updates.
     *
     * `drift` and `max_drift` record the value AS MEASURED, before any repair. Storing
     * the post-repair value instead would make a run with reorthogonalization report a
     * drift of ~1e-16 forever and hide the fact that the monitor fired at all -- the
     * reported quantity would then be a property of the repair, not of the algorithm.
     * The saw-tooth this produces (rise to the threshold, drop, rise again) is the
     * honest picture and is what results/orthogonality_drift.csv contains. */
    if (t->check_every > 0 && (t->steps % t->check_every) == 0) {
        t->drift = reorth_monitor(t);
        if (t->drift > t->max_drift) t->max_drift = t->drift;
        if (t->reorth_enabled && t->drift > t->reorth_tol) reorth_restore(t);
    }
    return score;
}
