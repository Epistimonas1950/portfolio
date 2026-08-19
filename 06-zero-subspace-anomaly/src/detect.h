/* detect.h -- the residual-energy detector and its warm-up-calibrated threshold.
 *
 * THE SCORE.  Normal operation lives near the tracked subspace, so what makes a sample
 * anomalous is energy ORTHOGONAL to it:
 *
 *     s(a) = || a - U U^T a ||^2 / || a ||^2   in [0, 1]
 *
 * The normalization by ||a||^2 is what makes the score comparable between a quiet
 * stretch and a loud one. Without it the detector fires on every increase in overall
 * amplitude, which on a motor is a change in load, not a fault. The numerator is
 * already computed by the incremental update as rho^2 -- the detector is free.
 *
 * Under the generator's model (see oracle/generate_data.py) a normal sample is
 * U c + sigma e, so the numerator is sigma^2 chi^2_{m-r} and the score has mean
 * approximately sigma^2 (m - r) / E||a||^2 with relative standard deviation
 * sqrt(2/(m-r)). At m = 24, r = 4 that is 32%: the null distribution is wide, skewed,
 * and nothing like Gaussian.
 *
 * THE THRESHOLD.  Which is exactly why it is not "mean + 3 sd" and not a constant. It
 * is an empirical quantile of the scores observed over a known-normal calibration
 * window:
 *
 *     tau = Quantile_{1-alpha}( { s(a_t) : t in calibration window } )
 *
 * so the only number chosen by hand is alpha, a false-positive rate, which is a
 * quantity a person operating the machine can actually reason about. On the board this
 * is the "record it while it is healthy" step. It costs one array of `calibration`
 * scores -- 300 doubles, 2.4 kB -- held only until the quantile is taken, after which
 * the detector is again O(1) in memory.
 *
 * WHAT THIS CANNOT DO. The threshold is calibrated once. If the machine's healthy
 * behaviour changes in a way the subspace tracker follows but the score distribution
 * does not (a change in noise level, say), the false-positive rate moves and nothing
 * here notices. Recalibrating on a rolling window of samples the detector itself calls
 * normal is the obvious fix and is not implemented; see STATUS.md.
 */

#ifndef SUBSPACE_DETECT_H
#define SUBSPACE_DETECT_H

#include "linalg.h"

/* s = rho^2 / ||a||^2, given the residual norm the incremental update already found.
 * Returns 0 for a zero sample rather than dividing by zero: a silent sensor is not an
 * anomaly, it is a different problem. */
scalar_t detect_score(scalar_t rho, scalar_t a_norm);

/* Empirical (1 - alpha) quantile of `scores`, which is SORTED IN PLACE. Uses linear
 * interpolation between order statistics, matching numpy's default so the C and the
 * numpy oracle produce the same threshold from the same scores. */
scalar_t detect_threshold(scalar_t *scores, int n, double quantile);

#endif /* SUBSPACE_DETECT_H */
