/* forget.h -- the exponential forgetting factor, and where its value comes from.
 *
 * BRIEF.md section 4: "apply an exponential forgetting factor lambda to Sigma so old
 * data decays. Choose lambda from the desired effective window length and say how."
 * Here is the how.
 *
 * The state factors A ~ U Sigma V^T, so the second moment the subspace is extracted
 * from is A A^T = U Sigma^2 U^T. Scaling Sigma by lambda before each update therefore
 * scales the *energy* contribution of everything already accumulated by lambda^2. A
 * sample that arrived k updates ago has since been scaled k times, so it enters the
 * current second moment with weight
 *
 *     w_k = lambda^{2k}
 *
 * The effective window is the total weight, i.e. the number of equally-weighted samples
 * that would carry the same total:
 *
 *     N_eff = sum_{k>=0} lambda^{2k} = 1 / (1 - lambda^2)
 *
 *     =>   lambda = sqrt( 1 - 1/N_eff )
 *
 * The convention matters and is easy to get wrong by a factor of two: lambda applied to
 * SIGMA gives 1/(1 - lambda^2); the same symbol applied to a covariance or to Sigma^2
 * gives 1/(1 - lambda). Both appear in the literature. This code multiplies Sigma, so
 * the first form is the right one, and the round trip
 * forget_window_from_lambda(forget_lambda_from_window(N)) == N is asserted in the
 * test suite rather than left as a comment.
 *
 * Worked value: N_eff = 400 samples gives lambda = sqrt(1 - 1/400) = 0.99875. The
 * half-life follows from lambda^{2k} = 1/2, k = ln(1/2) / (2 ln lambda) = 277 samples.
 *
 * lambda IS AN ILL-CONDITIONED PARAMETERIZATION near 1. Differentiating N_eff gives
 * dN_eff / N_eff = 2 N_eff (dlambda / lambda), so storing lambda amplifies its own
 * rounding error by 2 N_eff when the window is read back. At N_eff = 400 that is a
 * factor of 800: invisible in double, and in the -DSUBSPACE_FLOAT32 build it is the
 * difference between a window of 400 and a window of 400.007. Harmless at these sizes,
 * but it is the reason the round-trip test in `tracker selftest` uses a tolerance of
 * N_eff^2 * eps rather than a fixed relative one, and it would stop being harmless for
 * a window of 10^5 samples in single precision.
 *
 * THE TRADEOFF, stated plainly. Small N_eff tracks a moving subspace but shortens the
 * memory the detector's null distribution rests on, so the score variance rises and
 * small anomalies drown. Large N_eff is the reverse. lambda = 1 is the special case
 * of remembering everything; on the rotating stream in data/ it is left behind by the
 * data and flags the whole tail as anomalous, which is the measurement in
 * analysis/forgetting_study.py.
 */

#ifndef SUBSPACE_FORGET_H
#define SUBSPACE_FORGET_H

#include "linalg.h"

/* lambda = sqrt(1 - 1/n_eff). Returns 1 for n_eff <= 0, meaning "never forget". */
scalar_t forget_lambda_from_window(double n_eff);

/* N_eff = 1/(1 - lambda^2), the inverse of the above. Returns HUGE_VAL at lambda = 1. */
double forget_window_from_lambda(scalar_t lambda);

/* Sigma <- lambda * Sigma. One line, but it is the line that turns a growing batch SVD
 * into a sliding-window one. */
void forget_apply(scalar_t *sigma, int r, scalar_t lambda);

#endif /* SUBSPACE_FORGET_H */
