/* imu.h -- MPU-6050 / LSM6DS3 front end. NOT BUILT AGAINST HARDWARE. STUB.
 *
 * There is no board attached to the machine this repo was developed on and no Pico SDK
 * installed, so there is no way to test an I2C driver here. Rather than ship an
 * untested driver that looks finished, this header records the interface and the
 * register sequence, and imu.c returns IMU_ERR_NO_HARDWARE on the host. STATUS.md
 * lists it under "needs hardware".
 *
 * What the real driver would do, for an MPU-6050 at address 0x68 on i2c0:
 *
 *   PWR_MGMT_1  (0x6B) = 0x01   wake, clock from the X gyro PLL (more stable than the
 *                               internal 8 MHz oscillator, which drifts with temperature
 *                               and would corrupt the sample interval -- and dt is a
 *                               filter parameter, so a wrong dt is a wrong Qd)
 *   SMPLRT_DIV  (0x19) = 0x04   1 kHz / (1 + 4) = 200 Hz, matching KF_DT_DEFAULT
 *   CONFIG      (0x1A) = 0x03   DLPF 44 Hz: above the 1.7 Hz motion bandwidth, below
 *                               the 100 Hz Nyquist limit
 *   GYRO_CONFIG (0x1B) = 0x08   +-500 deg/s full scale. This is the number Q4.27 in
 *                               qformat.h is derived from; changing it changes the
 *                               format.
 *   ACCEL_CONFIG(0x1C) = 0x00   +-2 g, the most sensitive range, since the accelerometer
 *                               is used as an inclinometer at rest
 *   burst read 0x3B..0x48       14 bytes: ax ay az temp gx gy gz, big-endian int16
 *
 * The conversions into the formats of qformat.h are the interesting part and they are
 * pure integer arithmetic, so they are implemented and testable on the host:
 *
 *   rate  = raw_gyro * (2 pi / 360) / 65.5 LSB/(deg/s)   -> Q4.27
 *   angle = atan2(ay, az)                                -> Q3.28
 *
 * The atan2 is the one place a real port needs a decision this repo has not made: a
 * CORDIC or a polynomial in fixed point, both of which carry their own error budget on
 * top of the one measured here. It is listed in STATUS.md as not built rather than
 * hand-waved.
 */

#ifndef IMU_H
#define IMU_H

#include <stdint.h>

typedef enum {
    IMU_OK = 0,
    IMU_ERR_NO_HARDWARE = -1,   /* returned by the host stub, always */
    IMU_ERR_BUS = -2,
    IMU_ERR_WHOAMI = -3
} imu_status_t;

typedef struct {
    int32_t gyro_rate;    /* Q_RATE_FRAC, rad/s   */
    int32_t accel_angle;  /* Q_ANG_FRAC,  rad     */
} imu_sample_t;

imu_status_t imu_init(void);
imu_status_t imu_read(imu_sample_t *out);

/* Raw-to-format conversions. Pure integer, no bus, so these are host-testable and are
 * the only part of this file that is real code rather than a description. */
int32_t imu_gyro_to_q(int16_t raw_lsb, int lsb_per_dps, int frac);

#endif /* IMU_H */
