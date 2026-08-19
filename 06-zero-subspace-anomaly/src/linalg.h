/* linalg.h -- the hand-written dense linear algebra this project runs on.
 *
 * There is no BLAS, no LAPACK, no Eigen and no external dependency of any kind here,
 * and that is the point of the project rather than an oversight. The target board in
 * BRIEF.md is an armv6 Raspberry Pi Zero, where the prebuilt numerical stack that the
 * rest of the world assumes does not exist. Everything below is C11 that compiles with
 * `gcc -std=c11 -lm` and nothing else, so it builds for the host and would cross-compile
 * for the board unchanged.
 *
 * Conventions, stated once:
 *
 *   - Matrices are ROW-MAJOR and dense. An m x n matrix `a` has element (i, j) at
 *     a[i * n + j]. There are no leading-dimension parameters: every matrix here is
 *     exactly as wide as it claims to be, and views are made by copying. At r <= 8 the
 *     copies are a few hundred bytes and the simplification is worth more than the
 *     bandwidth.
 *   - Vectors are contiguous arrays.
 *   - `scalar_t` is double by default and float under -DSUBSPACE_FLOAT32. The float
 *     build exists because single precision is a defensible choice on a 512 MB board
 *     and because the orthogonality drift it produces is two orders of magnitude larger
 *     -- the same experiment, run at two precisions, is worth more than either alone.
 *   - Functions return 0 on success and a negative value on failure. Nothing here
 *     allocates; the caller passes scratch. That keeps the streaming path free of
 *     malloc, which is what makes the per-sample memory genuinely constant.
 */

#ifndef SUBSPACE_LINALG_H
#define SUBSPACE_LINALG_H

#include <float.h>
#include <stddef.h>
#include <stdint.h>

#ifdef SUBSPACE_FLOAT32
typedef float scalar_t;
#define SCALAR_EPS FLT_EPSILON
#define SCALAR_NAME "float"
#else
typedef double scalar_t;
#define SCALAR_EPS DBL_EPSILON
#define SCALAR_NAME "double"
#endif

/* --- level 1 ------------------------------------------------------------------- */

scalar_t la_dot(const scalar_t *x, const scalar_t *y, int n);
scalar_t la_norm2(const scalar_t *x, int n);
void la_scale(scalar_t *x, int n, scalar_t alpha);
void la_axpy(scalar_t alpha, const scalar_t *x, scalar_t *y, int n);
void la_zero(scalar_t *x, int n);
void la_copy(const scalar_t *src, scalar_t *dst, int n);

/* --- level 2 / 3 --------------------------------------------------------------- */

/* y (m) = A (m x n) * x (n) */
void la_matvec(const scalar_t *a, int m, int n, const scalar_t *x, scalar_t *y);

/* y (n) = A^T (n x m) * x (m) */
void la_matvec_t(const scalar_t *a, int m, int n, const scalar_t *x, scalar_t *y);

/* C (m x n) = A (m x k) * B (k x n) */
void la_gemm(const scalar_t *a, int m, int k, const scalar_t *b, int n, scalar_t *c);

/* C (k x n) = A^T (k x m) * B (m x n), i.e. A is m x k */
void la_gemm_tn(const scalar_t *a, int m, int k, const scalar_t *b, int n, scalar_t *c);

/* --- orthogonalization ---------------------------------------------------------- */

/* Thin Householder QR of A (m x n), m >= n.
 *
 * Writes Q (m x n, orthonormal columns) and R (n x n, upper triangular) with A = QR.
 * `a` is overwritten. `work` needs m + n scalars.
 *
 * Householder rather than classical Gram-Schmidt because the reflectors are orthogonal
 * to working precision by construction: the computed Q satisfies ||Q^T Q - I|| = O(eps)
 * independently of the conditioning of A, whereas classical Gram-Schmidt loses
 * orthogonality proportionally to kappa(A) and modified Gram-Schmidt to kappa(A) * eps.
 * The whole project is about measuring loss of orthogonality, so the routine that
 * restores it must not be the thing that causes it.
 */
int la_qr(scalar_t *a, int m, int n, scalar_t *q, scalar_t *r, scalar_t *work);

/* || U^T U - I ||_F for U (m x r). Zero iff the columns are exactly orthonormal.
 * `work` needs r * r scalars. This is the monitored quantity of section 3 of BRIEF.md.
 */
scalar_t la_orth_error(const scalar_t *u, int m, int r, scalar_t *work);

/* --- SVD ------------------------------------------------------------------------ */

/* One-sided Jacobi SVD of A (m x n) with m >= n:  A = U diag(s) V^T.
 *
 * `a` is overwritten. U is m x n, s is n, V is n x n. Singular values and the matching
 * columns of U and V are returned SORTED DESCENDING -- Jacobi produces them in no
 * particular order, and every caller here truncates, so an unsorted result silently
 * discards the dominant direction rather than the weakest one. Returns the number of
 * sweeps used, or -1 if it hit `max_sweeps` without converging.
 *
 * One-sided Jacobi rather than Golub-Kahan bidiagonalization: it is about thirty lines,
 * it computes the small singular values to high *relative* accuracy, and the matrices
 * it is applied to here are at most 9 x 9. Its usual drawback, being slower than
 * bidiagonalization for large n, is irrelevant at this size.
 */
int la_jacobi_svd(scalar_t *a, int m, int n, scalar_t *u, scalar_t *s, scalar_t *v,
                  int max_sweeps);

/* --- deterministic pseudo-random numbers ---------------------------------------- */

/* PCG-XSH-RR 32-bit. Chosen over rand() because rand() is implementation-defined --
 * the same seed gives different streams on the host and on the board, and a
 * "reproducible" randomized algorithm that is not reproducible across machines is
 * worse than useless when the point is to compare the two.
 */
typedef struct {
    uint64_t state;
    uint64_t inc;
    scalar_t spare;      /* Box-Muller produces normals in pairs; keep the second */
    int has_spare;
} la_rng;

void la_rng_seed(la_rng *rng, uint64_t seed);
uint32_t la_rng_u32(la_rng *rng);
scalar_t la_rng_uniform(la_rng *rng);
scalar_t la_rng_normal(la_rng *rng);

/* --- misc ----------------------------------------------------------------------- */

/* Sorts `x` ascending in place. Used for the empirical quantile in detect.c. */
void la_sort_ascending(scalar_t *x, int n);

#endif /* SUBSPACE_LINALG_H */
