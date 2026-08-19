/* detect.c -- see detect.h. */

#include "detect.h"

#include <math.h>

scalar_t detect_score(scalar_t rho, scalar_t a_norm)
{
    if (!(a_norm > 0)) return 0;
    const double ratio = (double)rho / (double)a_norm;
    return (scalar_t)(ratio * ratio);
}

scalar_t detect_threshold(scalar_t *scores, int n, double quantile)
{
    if (n < 1) return 0;
    if (n == 1) return scores[0];
    if (quantile <= 0.0) quantile = 0.0;
    if (quantile >= 1.0) quantile = 1.0;

    la_sort_ascending(scores, n);
    /* Linear interpolation on the position (n-1)*q. This is numpy's default 'linear'
     * method; matching it exactly is what lets oracle/compare.py attribute a threshold
     * difference between the C and the oracle to the scores rather than to two
     * defensible-but-different quantile conventions. */
    const double pos = quantile * (double)(n - 1);
    const int lo = (int)floor(pos);
    const int hi = (lo + 1 < n) ? lo + 1 : n - 1;
    const double frac = pos - (double)lo;
    return (scalar_t)((1.0 - frac) * (double)scores[lo] + frac * (double)scores[hi]);
}
