/* rank.h -- choosing r from the singular-value spectrum instead of hardcoding it.
 *
 * BRIEF.md section 5 is explicit: do not hardcode r. Two criteria are implemented,
 * both are always computed, and both are reported in results/spectrum.csv. They answer
 * different questions and they disagree on the multimode stream in data/, which is the
 * most useful thing about having both.
 *
 * ENERGY THRESHOLD.  smallest r with  sum_{i<=r} sigma_i^2 / sum_i sigma_i^2 >= tau.
 * The natural criterion when the downstream quantity is a residual *energy*, which it
 * is here: keeping 95% of the energy bounds a normal sample's score by 5% almost by
 * definition. Its weakness is that it does not care where the spectrum actually breaks
 * -- on a slowly decaying spectrum it returns a large r without complaint, because
 * there is no break to find.
 *
 * GAP CRITERION.  r maximizing sigma_r / sigma_{r+1}. The right criterion under a
 * signal-plus-isotropic-noise model, where the spectrum genuinely has a cliff and the
 * cliff's location is the rank. Its weakness is the mirror image: with no cliff it
 * returns whichever adjacent pair happened to be furthest apart, which is noise.
 *
 * MEASURED on the streams in data/ (results/spectrum.csv):
 *   single mode     energy(0.95) = 4, gap = 4      -- agree, and the true rank is 4
 *   rotating        energy(0.95) = 4, gap = 4      -- agree
 *   multimode       energy(0.95) = 5, gap = 6      -- DISAGREE
 * The union of two overlapping rank-4 modes has rank 6, which the gap criterion finds;
 * the energy criterion stops at 5 because the sixth component carries under 5% of the
 * energy. Neither is wrong. The disagreement is the signal that a single subspace is
 * the wrong model for that stream, and it is available before a single anomaly has
 * been scored.
 */

#ifndef SUBSPACE_RANK_H
#define SUBSPACE_RANK_H

#include "linalg.h"

/* Smallest r in [1, n] with cumulative energy >= threshold. */
int rank_by_energy(const scalar_t *sigma, int n, double threshold);

/* r in [1, min(r_max, n-1)] maximizing sigma_r / sigma_{r+1}. */
int rank_by_gap(const scalar_t *sigma, int n, int r_max);

#endif /* SUBSPACE_RANK_H */
