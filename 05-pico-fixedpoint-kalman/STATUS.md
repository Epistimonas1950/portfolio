# Status

Honest accounting of what runs, what is stubbed, and what needs hardware. The
portfolio's thesis is that the measurements are real, so this file is part of the
deliverable.

**Last verified on this machine:**
`make host` → gcc 13.3.0, exit 0, no warnings with `-Wall -Wextra -std=c11`.
`make test` → **29 tests, OK**, 9.4 s.
`make results` → 6 CSVs in `results/`, exit 0.
Python 3.12.3, numpy 1.26.4, gcc 13.3.0. No other dependency, no network, no board.

## Runnable today, on this machine, with gcc + numpy

| Component | File | State |
|---|---|---|
| Q-format table: range → format → step → half-ulp, per quantity | `firmware/qformat.h` | complete, documented, tested |
| Saturating fixed-point add/sub/mul/div/rescale, round-to-nearest | `firmware/qformat.h` | complete, 12 checks in `build/qtest` |
| Saturation event counter | `firmware/qformat.c` | complete; it found a real range error in `K1` |
| 2-state fixed-point Kalman filter, predict | `firmware/kalman_fixed.c` | complete, tested |
| Naive covariance update `P⁺ = (I−KH)P⁻` | `firmware/kalman_fixed.c` | complete — and diverges on purpose |
| Joseph covariance update | `firmware/kalman_joseph.c` | complete, tested |
| Host replay driver, per-step CSV + diagnostics | `host/main.c` | complete |
| Fixed-point arithmetic self-test | `host/qformat_selftest.c` | complete |
| Float64 reference filter | `reference/kalman_float.py` | complete, tested |
| Synthetic IMU trace with known truth | `reference/generate_trace.py` | complete, seeded, tested |
| Error budget, written and run before measurement | `reference/error_budget.py` | complete |
| Predicted-vs-measured comparison | `reference/compare.py` | complete |
| Gain bit-depth sweep | `analysis/gain_bits_sweep.py` | runs, CSV committed |
| Covariance bit-depth sweep + one-global-format build | `analysis/cov_bits_sweep.py` | runs, CSV committed |
| Sample-rate / step-size study | `analysis/dt_study.py` | runs, CSV committed |

## Present but not runnable here

| Component | Why |
|---|---|
| `analysis/plot_results.py` | needs matplotlib, which is not installed. It exits with a message naming the package. Every number it would draw is already in `results/*.csv`, so the project reproduces without it — you just do not get pictures. `results/divergence.png` and `results/predicted_vs_measured.png` are therefore **absent**, not empty. |
| `make pico` | needs `arm-none-eabi-gcc` and `PICO_SDK_PATH`; neither exists here. The target detects that and says so rather than failing obscurely. |

## Stubbed, and saying so

| Component | File | What it would need |
|---|---|---|
| MPU-6050 / LSM6DS3 I²C driver | `firmware/imu.h`, `firmware/imu.c` | A board. `imu_init` and `imu_read` return `IMU_ERR_NO_HARDWARE` unconditionally. The header records the register sequence and the reasoning behind each setting (the ±500 °/s full scale is where `Q4.27` comes from), because that part is a design decision I can defend; the bus transactions are not written, because untested driver code that looks finished is worse than an honest stub. The one piece that is pure integer arithmetic — raw LSB → Q4.27 conversion — *is* implemented and compiles into the host build. |

## Not built at all

These are parts of [BRIEF.md](BRIEF.md) that are absent rather than mocked.

- **Square-root / Cholesky-factor filter.** The brief marks it optional. It is not
  implemented. It would be the right answer if the specification tightened: §2 of
  `firmware/qformat.h` shows the covariance consuming 24.4 of the 31 available
  magnitude bits, and propagating a factor `S` with `P = S Sᵀ` halves that requirement.
  I would use Potter's scalar-measurement update, which needs a fixed-point square root
  — and that square root carries its own error budget, which is exactly why I did not
  want to half-build it.
- **Cycle counting and RAM high-water measurement.** The brief's results table has
  columns for these. **They are empty in `results/comparison.csv` and there is no
  estimate anywhere in this repo**, because there is no board and a cycles-per-update
  figure derived from an instruction-count guess is a fabricated measurement. The one
  thing I will say is a count, not a timing: the Joseph update is 9 multiplies and
  1 subtract against the naive form's 4 multiplies, which you can read off
  `firmware/kalman_joseph.c`.
- **TensorFlow Lite Micro gesture/keyword classifier** fed by the filtered stream.
  Needs the SDK, a trained model and a board.
- **Recorded IMU dataset.** There is no capture, so `data/imu_capture.csv` is
  synthetic and `reference/generate_trace.py` says so at the top. See the limitation
  section of [README.md](README.md) — this is the biggest gap in the repo and it is
  the first thing I would close.
- **On-device validation of any number in `results/`.** Everything here is the host
  build. The arithmetic is identical by construction (portable C11, `int32_t`/`int64_t`
  only, no floating point in the filter); the claim that it is identical is an argument
  from the source, not a measurement.

## What the numbers here do and do not support

**Supported.** The error budget predicts, from the format choices alone and before any
fixed-point run: the gain precision at which the textbook covariance update loses
positive definiteness (12.29 bits predicted, 12 measured); the naive form's symmetry
residual across 14 gain precisions (within 1.05–13× at every point, median 1.6×); the
covariance precision at which the process noise underflows (23.4 bits predicted;
survives at 24, collapses at 22); and the sample-rate ceiling that a covariance word
length imposes (302 Hz and 2416 Hz predicted, 400 Hz and 3200 Hz measured — both 1.32×).
The Joseph form's symmetry residual is exactly zero at every precision tested.

One qualification on the divergence, stated here because the README's headline table is
where a reader meets it: at Q1.12 the naive covariance is indefinite for **6 steps out of
12000**, not for the whole run. It recovers. The estimate does not — its error is still
7x Joseph's for the remaining 59 seconds, because the six bad steps set the state that
every later gain acts on.

**Supported with a stated bracket.** The magnitude of the fixed-point state error. The
budget gives a lower estimate (independent roundings, 6.8e-5°) and a strict upper bound
(ℓ1 gain, 0.125°); the measurement lands at 4.5e-4°, inside both, 6.7× above the lower.
The gap is diagnosed rather than excused — see the README.

**Not supported.** Anything about cycles, RAM, real IMU data, an RP2040, or a
classifier. No number in this repo speaks to any of those.
