/* incsvd.h -- Brand's rank-one incremental SVD update. The algorithmic core.
 *
 * THE UPDATE
 * ----------
 * Hold a thin factorization A ~ U Sigma V^T with U (m x r) orthonormal, Sigma diagonal
 * and descending. A new sample `a` (m) arrives. Split it into the part the model
 * already explains and the part it does not:
 *
 *     mvec = U^T a          coordinates inside the subspace
 *     p    = a - U mvec     residual, orthogonal to the subspace
 *     rho  = || p ||        how much of `a` is new
 *
 * With q = p / rho the augmented matrix factors EXACTLY:
 *
 *     [ A  a ] = [ U  q ] [ Sigma  mvec ] [ V  0 ]^T
 *                         [   0     rho ] [ 0   1 ]
 *
 * The middle matrix K is (r+1) x (r+1) and small. Its SVD K = U' S' V'^T costs O(r^3),
 * and the updated factors are U_new = [U q] U', Sigma_new = S'. Truncating back to r
 * columns returns the state to (m x r) + r. That is the whole algorithm:
 *
 *     memory   O(mr + r^2)          -- independent of how many samples have been seen
 *     work     O(mr + r^3)          -- per sample, constant
 *     passes   one                   -- the stream is never re-read
 *
 * At m = 24, r = 4 the state is 24*4 + 4 = 100 doubles, 800 bytes. The batch
 * alternative in BRIEF.md is an SVD of the whole data matrix: O(mn) memory and O(mn^2)
 * work, which on an unbounded stream is not a slower option, it is not an option.
 *
 * V IS DELIBERATELY NOT TRACKED. It has one row per sample seen, so carrying it would
 * make memory grow with the length of the stream -- exactly what "constant memory"
 * forbids, and the reason the algorithm is shaped this way. Nothing the detector needs
 * lives in V.
 *
 * TWO THINGS THAT LOOK LIKE DETAILS AND ARE NOT
 * ---------------------------------------------
 * 1. TRUNCATION REQUIRES SORTED SINGULAR VALUES. One-sided Jacobi (src/linalg.c) emits
 *    them in arbitrary order. Slicing the first r columns of an unsorted U' discards
 *    the *dominant* direction roughly one time in r. Nothing crashes, no value becomes
 *    infinite, the scores stay plausible, and the tracked subspace is wrong. la_jacobi_svd
 *    sorts descending for this reason and this reason alone.
 *
 * 2. THE rho GUARD. When `a` lies in the current subspace to machine precision, rho is
 *    pure rounding noise and q = p / rho is a random unit vector; appending it injects
 *    a garbage direction into the basis. The guard is rho <= sqrt(eps) ||a||, the
 *    standard re-orthogonalization criterion. It is applied by setting rho = 0 and
 *    q = 0 rather than by branching into a separate rank-preserving code path: with
 *    rho = 0 the last row of K vanishes, so every left singular vector of K with a
 *    nonzero singular value has zero last entry, and the update degenerates exactly
 *    into U <- U U'[0:r, 0:r]. One code path, no second version to keep in sync.
 *
 * FORGETTING and REORTHOGONALIZATION are applied inside incsvd_update; see forget.h and
 * reorth.h for their derivations.
 */

#ifndef SUBSPACE_INCSVD_H
#define SUBSPACE_INCSVD_H

#include "linalg.h"

typedef struct {
    int m;                  /* channels */
    int r;                  /* tracked rank, fixed after initialization */
    scalar_t *u;            /* m x r, orthonormal columns */
    scalar_t *sigma;        /* r, descending */

    scalar_t lambda;        /* exponential forgetting factor, 1 = remember everything */
    int reorth_enabled;     /* run the monitor's repair step at all */
    int check_every;        /* how often the orthogonality monitor runs */
    scalar_t reorth_tol;    /* || U^T U - I ||_F above which U is repaired */

    long steps;             /* rank-one updates applied so far */
    scalar_t drift;         /* last measured || U^T U - I ||_F */
    scalar_t max_drift;     /* largest value it has ever taken */
    int n_reorth;           /* how many times the repair fired */

    /* scratch, allocated once at init so the streaming path never calls malloc */
    scalar_t *mvec;         /* r     */
    scalar_t *resid;        /* m     */
    scalar_t *kmat;         /* (r+1)^2 */
    scalar_t *ku;           /* (r+1)^2 */
    scalar_t *kv;           /* (r+1)^2 */
    scalar_t *ks;           /* r+1   */
    scalar_t *ubig;         /* m x (r+1) */
    scalar_t *unew;         /* m x r */
    scalar_t *work;         /* max(m + r + 1, (r+1)^2) */
} incsvd_t;

/* Allocate state for m channels and rank r. Returns 0 or negative. */
int incsvd_alloc(incsvd_t *t, int m, int r);
void incsvd_free(incsvd_t *t);

/* Install an initial orthonormal basis and spectrum (copied in). */
void incsvd_set(incsvd_t *t, const scalar_t *u, const scalar_t *sigma);

/* One rank-one update. Returns the residual-energy score of `a`, computed against the
 * subspace as it was BEFORE `a` was folded in -- scoring afterwards lets every sample
 * partially explain itself, which flatters isolated anomalies most, i.e. exactly the
 * samples the detector exists to catch.
 */
scalar_t incsvd_update(incsvd_t *t, const scalar_t *a);

/* The residual-energy score of `a` against the current subspace, with NO update. Used
 * for the warm-up block, whose samples have already contributed to the initial basis
 * through the randomized SVD: feeding them to incsvd_update as well would count them
 * twice, inflating Sigma by sqrt(1 + warmup/n) and shortening the tracker's effective
 * memory by the same factor. The subspace barely moves either way -- the measured
 * difference was 0.05 degrees -- but "each sample is seen once" is a claim this project
 * makes, so it should be true rather than nearly true. */
scalar_t incsvd_score(const incsvd_t *t, const scalar_t *a, scalar_t *scratch);

/* Total heap held by the tracker in the steady state, in bytes. Reported by the host
 * driver rather than worked out by hand for the README, because a memory claim nobody
 * measured is exactly the kind of number this portfolio is not allowed to contain. */
size_t incsvd_bytes(const incsvd_t *t);

#endif /* SUBSPACE_INCSVD_H */
