/* rank.c -- see rank.h for what each criterion is for and where they disagree. */

#include "rank.h"

int rank_by_energy(const scalar_t *sigma, int n, double threshold)
{
    if (n < 1) return 0;
    double total = 0.0;
    for (int i = 0; i < n; ++i) total += (double)sigma[i] * (double)sigma[i];
    if (total <= 0.0) return 1;
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        acc += (double)sigma[i] * (double)sigma[i];
        if (acc / total >= threshold) return i + 1;
    }
    return n;
}

int rank_by_gap(const scalar_t *sigma, int n, int r_max)
{
    if (n < 2) return n;
    int top = n - 1;
    if (r_max > 0 && r_max < top) top = r_max;
    int best = 1;
    double best_ratio = -1.0;
    for (int i = 0; i < top; ++i) {
        const double denom = (double)sigma[i + 1];
        /* An exactly zero sigma_{i+1} means the spectrum has run out; the gap there is
         * infinite and uninformative, so it is treated as the largest finite gap seen
         * rather than as +inf, which would always win. */
        const double ratio = (denom > 0.0) ? (double)sigma[i] / denom : best_ratio;
        if (ratio > best_ratio) { best_ratio = ratio; best = i + 1; }
    }
    return best;
}
