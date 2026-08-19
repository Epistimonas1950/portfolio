/* imu.c -- host stub. See imu.h for what the real driver would be.
 *
 * imu_init and imu_read fail loudly and unconditionally: there is no board. They are
 * compiled into the host build anyway so that the interface stays type-checked and so
 * that nobody can mistake a link error for a missing feature.
 */

#include "imu.h"

#include "qformat.h"

imu_status_t imu_init(void)
{
    /* A real implementation reads WHO_AM_I (0x75) and checks it reads 0x68. */
    return IMU_ERR_NO_HARDWARE;
}

imu_status_t imu_read(imu_sample_t *out)
{
    if (out) { out->gyro_rate = 0; out->accel_angle = 0; }
    return IMU_ERR_NO_HARDWARE;
}

/* raw / lsb_per_dps gives deg/s; times pi/180 gives rad/s. Done as one 64-bit
 * multiply-then-divide so the intermediate keeps full precision and the result is
 * rounded once, matching the contract every other conversion in this project keeps.
 *
 * pi/180 in Q0.31 = 37487776 (0.0174532925...), exact to 4.7e-10.
 */
int32_t imu_gyro_to_q(int16_t raw_lsb, int lsb_per_dps, int frac)
{
    const int64_t deg_to_rad_q31 = 37487776;   /* pi/180 in Q0.31 */
    int64_t v;
    if (lsb_per_dps <= 0) { return 0; }
    v = (int64_t)raw_lsb * deg_to_rad_q31;                  /* Q0.31 * LSB */
    v = q_round_shift(v, 31 - frac);                        /* -> Q?.frac * LSB */
    /* Rounded divide, same rule as q_div_f. */
    v = (v >= 0) ? (v + lsb_per_dps / 2) : (v - lsb_per_dps / 2);
    return q_sat(v / lsb_per_dps);
}
