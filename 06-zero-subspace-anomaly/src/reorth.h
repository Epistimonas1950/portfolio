/* reorth.h -- the orthogonality monitor and the repair that preserves the factorization.
 *
 * BRIEF.md section 3: repeated rank-one updates lose orthogonality of U in floating
 * point, the error accumulates, and the subspace silently rotates. The failure is
 * silent in the strict sense -- no exception, no NaN, no value out of range. The only
 * way to know is to measure, so:
 *
 *     drift = || U^T U - I ||_F
 *
 * is computed every `check_every` samples at O(m r^2), and U is repaired when it
 * exceeds a threshold. Measured trajectories are in results/orthogonality_drift.csv;
 * over 20 000 updates at m = 24, r = 4 the drift grows by roughly two and a half orders
 * of magnitude above its floor without repair, and is pinned at the threshold with it.
 *
 * THE REPAIR IS NOT GRAM-SCHMIDT ON U. That is the tempting one-liner and it is wrong.
 * U Sigma V^T is a factorization; replacing U by the Q of U = QR changes what it
 * factors unless Sigma moves too. The step that leaves the product invariant is
 *
 *     U Sigma = Q R Sigma = Q (R Sigma) = Q (U~ S~ V~^T)
 *     =>  U <- Q U~ ,   Sigma <- S~        (V absorbs V~, and we do not track V)
 *
 * so the repair is a thin QR followed by an r x r SVD, O(m r^2 + r^3), amortized over
 * check_every samples. Because R is within `drift` of the identity, S~ is within
 * `drift` of Sigma: the correction is tiny. That is exactly why skipping it looks
 * harmless -- right up until the accumulated rotation is no longer tiny, at which point
 * the basis is no longer a basis and every projection computed from it is wrong by an
 * amount nothing in the pipeline reports.
 */

#ifndef SUBSPACE_REORTH_H
#define SUBSPACE_REORTH_H

#include "incsvd.h"

/* || U^T U - I ||_F. Uses the tracker's own scratch; does not modify the state. */
scalar_t reorth_monitor(incsvd_t *t);

/* Thin QR + small SVD repair described above. Returns 0, or negative if the small SVD
 * failed to converge -- in which case the state is left untouched rather than half
 * updated. */
int reorth_restore(incsvd_t *t);

#endif /* SUBSPACE_REORTH_H */
