/* qformat.c -- the two things in qformat.h that cannot be a static inline.
 *
 * The saturation counter has to have exactly one definition, and the double
 * conversions are deliberately out of line so that it is trivial to check, with nm,
 * that nothing else in the filter links against floating point:
 *
 *     nm build/kalman_joseph.o | grep -i -E 'aeabi_[df]|__mul[ds]f|__adddf'
 *
 * On the host that check is decorative; on an RP2040 build it is the difference
 * between a 4-cycle multiply and a 100-cycle library call.
 */

#include "qformat.h"

#include <math.h>

unsigned long q_sat_events_count = 0;

void q_sat_reset(void) { q_sat_events_count = 0; }

unsigned long q_sat_events(void) { return q_sat_events_count; }

/* Round to nearest, ties away from zero, then saturate -- the same rule the integer
 * path uses, so a constant loaded here and a value computed by q_mul agree. */
int32_t q_from_double(double x, int frac)
{
    double scaled = x * (double)((int64_t)1 << frac);
    if (!(scaled > -2147483649.0 && scaled < 2147483648.0)) {
        q_sat_events_count++;
        return scaled > 0 ? Q_MAX32 : Q_MIN32;
    }
    return (int32_t)llround(scaled);
}

double q_to_double(int32_t v, int frac)
{
    return (double)v / (double)((int64_t)1 << frac);
}
