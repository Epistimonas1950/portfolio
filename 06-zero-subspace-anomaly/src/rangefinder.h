/* rangefinder.h -- randomized range finding, and the guarantee that justifies it.
 *
 * THE PROBLEM
 * -----------
 * Given A (m x n, one sample per column) find an orthonormal Q with few columns such
 * that A is close to its projection onto range(Q). The optimum over all rank-k
 * projections is fixed by Eckart-Young:
 *
 *     min_{rank(B) <= k} || A - B ||_F  =  ( sum_{j>k} sigma_j^2 )^{1/2}
 *
 * THE ALGORITHM
 * -------------
 * Draw a Gaussian Omega (n x (k+p)), form Y = A Omega, take Q = orth(Y). Each column
 * of Y is a random linear combination of the columns of A, weighted by the spectrum,
 * so k+p random probes capture the top-k action of A with high probability. With q
 * power iterations Y = (A A^T)^q A Omega.
 *
 * THE BOUND  (Halko, Martinsson & Tropp 2011, arXiv:0909.4061)
 * -----------------------------------------------------------
 * For a Gaussian test matrix with oversampling p >= 2,
 *
 *     E || (I - Q Q^T) A ||_F  <=  ( 1 + k/(p-1) )^{1/2} ( sum_{j>k} sigma_j^2 )^{1/2}
 *
 *     E || (I - Q Q^T) A ||_2  <=  ( 1 + sqrt(k/(p-1)) ) sigma_{k+1}
 *                                  + ( e sqrt(k+p) / p ) ( sum_{j>k} sigma_j^2 )^{1/2}
 *
 * WHAT p BUYS: the constant, and nothing else. At k = 4, p = 6 the Frobenius factor is
 * (1 + 4/5)^{1/2} = 1.34 -- in expectation, within 34% of the best rank-4 projection
 * that exists. At p = 1 the factor is undefined; at p = 0 there is no guarantee at all,
 * because Y then has exactly k columns and a single draw nearly orthogonal to a
 * singular direction loses that direction outright. The deviation bounds in the same
 * paper fail with probability decaying like p^{-p}, so a handful of extra columns also
 * buys a failure probability small enough to stop thinking about. Cost: p extra columns
 * of memory and O(mnp) extra work.
 *
 * WHAT q BUYS: everything p cannot. p does not help a slowly decaying spectrum, because
 * the tail ( sum_{j>k} sigma_j^2 )^{1/2} is then genuinely large. Power iteration
 * replaces A by B = (A A^T)^q A, whose singular values are sigma_j^{2q+1}, so the ratio
 * of tail to signal is raised to the power 2q+1 before the bound is applied. Combined
 * with || (I - QQ^T) A || <= || (I - QQ^T) B ||^{1/(2q+1)} this gives
 *
 *     E || (I - QQ^T) A ||_2  <=  [ (1 + sqrt(k/(p-1))) sigma_{k+1}^{2q+1}
 *                                  + (e sqrt(k+p)/p) ( sum_{j>k} sigma_j^{2(2q+1)} )^{1/2}
 *                                ]^{1/(2q+1)}
 *
 * which tends to sigma_{k+1}, the optimum, as q grows. Cost: 2q extra passes over A.
 *
 * WHY THE ORTHONORMALIZATION INSIDE THE LOOP. Forming (A A^T)^q A Omega by repeated
 * multiplication is numerically hopeless: the singular values of the product span
 * kappa^{2q+1}, so at q = 2 and kappa = 10^3 the weak directions are already below
 * double precision and are rounded away entirely. Re-orthonormalizing between every
 * application costs O(m(k+p)^2) and restores them. This is the difference between the
 * power-iteration and subspace-iteration forms of the algorithm, and it is the one
 * place where a faithful transcription of the formula silently produces garbage.
 *
 * HONEST SCOPE. With m = 24 channels a randomized sketch is not *necessary* -- an exact
 * SVD of a 24 x 300 warm-up block costs microseconds. It is here because it is the
 * routine that keeps the warm-up affordable as the channel count grows (a 256-bin
 * spectrogram, say), and because the bound above is the part of the project worth
 * being able to state. The measured accuracy against the Eckart-Young floor is in
 * results/rangefinder.csv, and the README says the same thing in words.
 */

#ifndef SUBSPACE_RANGEFINDER_H
#define SUBSPACE_RANGEFINDER_H

#include "linalg.h"

/* Q (m x ell), orthonormal, whose range approximates the dominant range of A (m x n).
 * `ell` = k + p is the sketch width; `power_iters` = q. Allocates internally: this runs
 * once during warm-up, never in the per-sample path. Returns 0 or negative on failure.
 */
int rf_range_finder(const scalar_t *a, int m, int n, int ell, int power_iters,
                    la_rng *rng, scalar_t *q_out);

/* Approximate leading singular triplets of A via the sketch:
 *   Q = range finder,  B = Q^T A (ell x n),  B = W diag(s) P^T,  U = Q W.
 * Writes u_out (m x ell) and s_out (ell), sorted descending. The caller truncates,
 * because rank selection needs to see the tail of the spectrum.
 */
int rf_randomized_svd(const scalar_t *a, int m, int n, int ell, int power_iters,
                      la_rng *rng, scalar_t *u_out, scalar_t *s_out);

/* || A - Q Q^T A ||_F, computed as an explicit residual rather than by the Pythagorean
 * shortcut || A ||_F^2 - || Q^T A ||_F^2. The shortcut cancels catastrophically once
 * the projection is good, which is precisely the regime the error is being measured in.
 */
scalar_t rf_projection_error(const scalar_t *a, int m, int n, const scalar_t *q,
                             int ell);

#endif /* SUBSPACE_RANGEFINDER_H */
