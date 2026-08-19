/* forget.c -- see forget.h for the derivation of lambda from the window length. */

#include "forget.h"

#include <math.h>

scalar_t forget_lambda_from_window(double n_eff)
{
    if (n_eff <= 0.0) return (scalar_t)1;
    if (n_eff <= 1.0) return (scalar_t)0;          /* window of one sample */
    return (scalar_t)sqrt(1.0 - 1.0 / n_eff);
}

double forget_window_from_lambda(scalar_t lambda)
{
    const double l = (double)lambda;
    const double denom = 1.0 - l * l;
    if (denom <= 0.0) return HUGE_VAL;             /* lambda = 1: infinite memory */
    return 1.0 / denom;
}

void forget_apply(scalar_t *sigma, int r, scalar_t lambda)
{
    if (lambda == (scalar_t)1) return;
    la_scale(sigma, r, lambda);
}
